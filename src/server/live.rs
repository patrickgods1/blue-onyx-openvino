//! Live config for the pages:
//! - `GET /v1/config`: the effective config, its revision and change log, per model what is
//!   configured vs what the running worker uses, what a restart would apply, and the canonical
//!   field values the Config page's forms are merged against (ETag / If-None-Match);
//! - the 3-way merge of the Config page's forms ([`server_form_edit`], [`models_card_edit`])
//!   and their conflict answers (409 JSON, or the Config page with a conflict card).

use super::{AppState, ConfigView, render};
use crate::config::{Config, parse_config_form, validate_config, validate_models};
use crate::config_merge::{
    self, Conflict, Fields, MergeError, Pending, Resolution, Scope, form_fields,
};
use crate::registry::normalize_name;
use crate::startup::ModelState;
use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Changes listed by `GET /v1/config`.
pub(super) const CHANGES_SHOWN: usize = 20;

// ---------------------------------------------------------------------------------------------
// Benchmark facts (cached by file stamp: the results file can be tens of MB)

/// What the last benchmark says about a model.
#[derive(Debug, Clone, Default)]
pub(super) struct BenchFacts {
    pub model: String,
    pub recommended: Option<String>,
    pub recommendation: String,
    /// Best confidence threshold that can be applied, and its objective.
    pub best_threshold: Option<f32>,
    pub objective: String,
}

#[derive(Debug, Clone, Default)]
pub(super) struct BenchSummary {
    /// Normalized model name -> facts.
    pub models: HashMap<String, BenchFacts>,
    pub count: usize,
    pub timestamp: String,
}

type BenchCache = Option<(PathBuf, Option<SystemTime>, u64, Arc<Option<BenchSummary>>)>;
static BENCH_CACHE: Mutex<BenchCache> = Mutex::new(None);

/// The saved benchmark results' per-model facts (None when there are none), re-read only when
/// the file changes.
pub(super) fn bench_summary(path: &Path) -> Arc<Option<BenchSummary>> {
    let md = std::fs::metadata(path).ok();
    let stamp = md.as_ref().map(|m| (m.modified().ok(), m.len()));
    let mut cache = BENCH_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let (Some(c), Some((mt, len))) = (cache.as_ref(), stamp)
        && c.0 == path
        && c.1 == mt
        && c.2 == len
    {
        return c.3.clone();
    }
    let Some((mt, len)) = stamp else {
        return Arc::new(None);
    };
    let summary = crate::benchmark::BenchmarkResults::load_or_warn(path).map(|r| BenchSummary {
        count: r.models.len(),
        timestamp: r.timestamp.clone(),
        models: r
            .models
            .iter()
            .map(|m| {
                (
                    normalize_name(&m.model),
                    BenchFacts {
                        model: m.model.clone(),
                        recommended: m.recommended.clone(),
                        recommendation: m.recommendation.clone(),
                        best_threshold: m.threshold.as_ref().and_then(|t| t.apply_value()),
                        objective: m
                            .threshold
                            .as_ref()
                            .map(|t| t.objective.clone())
                            .unwrap_or_else(|| m.threshold_objective.to_string()),
                    },
                )
            })
            .collect(),
    });
    let summary = Arc::new(summary);
    *cache = Some((path.to_path_buf(), mt, len, summary.clone()));
    summary
}

// ---------------------------------------------------------------------------------------------
// Running state

/// What the running worker of a model uses.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RunningView {
    /// A worker of this generation serves the model.
    pub present: bool,
    /// "Ready", "Loading", "Failed:", "Lazy", "Waiting for download", "not loaded".
    pub state: String,
    /// Badge class: ok, loading, wait, fail, lazy, off.
    pub state_class: &'static str,
    pub detail: Option<String>,
    /// Execution provider ("OpenVINO CPU", "ONNX Runtime CoreML (loading)").
    pub provider: Option<String>,
    /// Canonical spec of the loaded device.
    pub device: Option<String>,
    /// Effective confidence threshold of the worker.
    pub threshold: Option<f64>,
    /// One line for the Models card's Running column.
    pub text: String,
}

pub(super) fn running_view(state: &AppState, name: &str) -> RunningView {
    let Some(w) = state.registry.by_name(name) else {
        return RunningView {
            present: false,
            state: "not loaded".into(),
            state_class: "off",
            detail: None,
            provider: None,
            device: None,
            threshold: None,
            text: "not loaded".into(),
        };
    };
    let (st, class, detail) = super::state_view(w);
    let provider = w.execution_provider();
    let device = w
        .device
        .read()
        .ok()
        .and_then(|d| d.as_ref().map(|d| d.spec.clone()));
    let key = normalize_name(name);
    let threshold = state
        .running
        .models
        .iter()
        .find(|m| normalize_name(&m.effective_name()) == key)
        .map(|m| {
            crate::config::round_threshold_f64(
                m.confidence_threshold
                    .unwrap_or(state.running.confidence_threshold),
            )
        });
    let text = match w.state.get() {
        ModelState::Failed(m) => format!("Failed: {m}"),
        ModelState::Ready => match threshold {
            Some(t) => format!("{provider} \u{b7} threshold {t:.2}"),
            None => provider.clone(),
        },
        _ => match &detail {
            Some(d) => format!("{provider} \u{b7} {st}: {d}"),
            None => provider.clone(),
        },
    };
    RunningView {
        present: true,
        state: st,
        state_class: class,
        detail,
        provider: Some(provider),
        device,
        threshold,
        text,
    }
}

/// Device selection of the loaded runtimes (cheap; None while they are busy compiling).
fn quick_selection(state: &AppState) -> Option<crate::backend::select::Selection> {
    let rt = state.registry.runtimes()?;
    let rt = rt.try_lock().ok()?;
    Some(rt.selection(None))
}

/// The device `m` runs on after a restart, as a canonical spec (`auto` resolved).
fn configured_resolved(
    state: &AppState,
    cfg: &Config,
    m: &crate::config::ModelConfig,
    sel: Option<&crate::backend::select::Selection>,
) -> Option<String> {
    match sel {
        Some(sel) => crate::benchmark::configured_device(cfg, m, sel),
        None => match cfg.device_spec_for(m).ok()? {
            crate::backend::DeviceSpec::Auto => state
                .registry
                .by_name(&m.effective_name())
                .and_then(|w| w.planned.as_ref().map(|p| p.to_string())),
            d => Some(d.to_string()),
        },
    }
}

// ---------------------------------------------------------------------------------------------
// GET /v1/config

/// The payload of `GET /v1/config`.
pub(super) fn payload(state: &AppState) -> Value {
    state.config.check_disk();
    let (revision, cfg) = state.config.snapshot();
    let pending = config_merge::pending(&state.running, &cfg);
    let bench = bench_summary(&crate::benchmark::results_path(&state.config_path));
    let sel = quick_selection(state);
    let default = cfg.effective_default_index();
    let thr = crate::config::round_threshold_f64;
    let models: Vec<Value> = cfg
        .models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let name = m.effective_name();
            let key = normalize_name(&name);
            let reasons = pending.for_model(&name);
            let resolved = configured_resolved(state, &cfg, m, sel.as_ref());
            let facts = (*bench)
                .as_ref()
                .and_then(|b| b.models.get(&key).cloned());
            json!({
                "name": name,
                "key": key,
                "family": m.family.to_string(),
                "path": m.path.display().to_string(),
                "enabled": m.enabled,
                "default": default == Some(i),
                "device": m.device,
                "deviceFor": cfg.device_for(m),
                "resolvedDevice": resolved,
                "threshold": m.confidence_threshold.map(thr),
                "effectiveThreshold": thr(m.confidence_threshold.unwrap_or(cfg.confidence_threshold)),
                "running": running_view(state, &name),
                "restartNeeded": !reasons.is_empty(),
                "reasons": reasons,
                "bench": facts.map(|f| json!({
                    "recommended": f.recommended,
                    "recommendation": f.recommendation,
                    "inUse": f.recommended.is_some() && f.recommended == resolved,
                    "bestThreshold": f.best_threshold.map(thr),
                    "objective": f.objective,
                    "thresholdInUse": f.best_threshold.is_some_and(|t| {
                        thr(t) == thr(m.confidence_threshold.unwrap_or(cfg.confidence_threshold))
                    }),
                    "href": format!("/benchmark#m-{}", super::encode_uri_component(&f.model)),
                })),
            })
        })
        .collect();
    // Workers of this generation whose model left the config.
    let gone: Vec<String> = state
        .registry
        .workers()
        .iter()
        .filter(|w| crate::benchmark::find_model(&cfg, &w.name).is_none())
        .map(|w| w.name.clone())
        .collect();
    json!({
        "success": true,
        "revision": revision,
        "epoch": state.config.epoch(),
        "generation": state.config.generation(),
        "path": state.config_path.display().to_string(),
        "fileError": state.config.disk_error(),
        "config": cfg,
        "globalThreshold": thr(cfg.confidence_threshold),
        "restartNeeded": pending.restart_needed(),
        "pendingCount": pending.count(),
        "pending": pending_json(&pending, &cfg),
        "models": models,
        "loadedNotConfigured": gone,
        "changes": state.config.changes(CHANGES_SHOWN),
        "forms": {
            "server": {
                "view": ConfigView::from_config(&cfg),
                "base": form_fields(&cfg, Scope::ServerForm),
            },
            "models": {
                "base": form_fields(&cfg, Scope::ModelsCard),
            },
        },
    })
}

/// Pending reasons as a flat list ("IPcam-general: device: auto → openvino:cpu").
fn pending_json(p: &Pending, cfg: &Config) -> Value {
    let display = |key: &str| {
        cfg.models
            .iter()
            .map(|m| m.effective_name())
            .find(|n| normalize_name(n) == key)
            .unwrap_or_else(|| key.to_string())
    };
    let mut all: Vec<String> = p.global.clone();
    for (k, reasons) in &p.models {
        for r in reasons {
            all.push(format!("{}: {r}", display(k)));
        }
    }
    all.extend(p.process.iter().cloned());
    json!({
        "global": p.global,
        "process": p.process,
        "models": p.models,
        "all": all,
    })
}

/// `GET /v1/config` with a weak ETag over the payload (If-None-Match answers 304).
pub(super) async fn config_json(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    let body = payload(&state);
    let text = body.to_string();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    let etag = format!("W/\"{}-{:016x}\"", body["revision"], h.finish());
    let matches = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|t| t.trim() == etag));
    let etag_value = HeaderValue::from_str(&etag).unwrap_or(HeaderValue::from_static("W/\"0\""));
    let rev = HeaderValue::from_str(&body["revision"].to_string())
        .unwrap_or(HeaderValue::from_static("0"));
    if matches {
        return (
            StatusCode::NOT_MODIFIED,
            [
                (header::ETAG, etag_value),
                (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
            ],
        )
            .into_response();
    }
    let mut resp = (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
            (header::ETAG, etag_value),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
        ],
        text,
    )
        .into_response();
    resp.headers_mut().insert("x-config-revision", rev);
    resp
}

// ---------------------------------------------------------------------------------------------
// Form merge

/// Merge metadata a Config page form posts: `base` (canonical values it was rendered from)
/// and conflict resolutions (`resolve.<key>` = mine|current, `resolve_cur.<key>` = the
/// current value the user saw, `resolve_all`).
#[derive(Debug, Default)]
pub(super) struct FormMeta {
    pub base: Option<Fields>,
    pub resolutions: BTreeMap<String, Resolution>,
}

impl FormMeta {
    pub fn parse(pairs: &[(String, String)]) -> anyhow::Result<Self> {
        let get = |k: &str| pairs.iter().find(|(x, _)| x == k).map(|(_, v)| v.as_str());
        let base = match get("base").map(str::trim).filter(|b| !b.is_empty()) {
            Some(b) => Some(
                serde_json::from_str::<Fields>(b)
                    .map_err(|e| anyhow::anyhow!("invalid form base: {e}"))?,
            ),
            None => None,
        };
        let all = get("resolve_all").map(str::trim);
        let mut resolutions = BTreeMap::new();
        for (k, v) in pairs {
            let Some(key) = k.strip_prefix("resolve_cur.") else {
                continue;
            };
            let seen: Option<Value> = if v.trim().is_empty() {
                None
            } else {
                Some(
                    serde_json::from_str(v)
                        .map_err(|e| anyhow::anyhow!("invalid resolve value: {e}"))?,
                )
            };
            let choice = all
                .filter(|a| !a.is_empty())
                .or_else(|| get(&format!("resolve.{key}")).map(str::trim));
            match choice {
                Some("mine") => {
                    resolutions.insert(key.to_string(), Resolution::Mine { seen });
                }
                Some("current") => {
                    resolutions.insert(key.to_string(), Resolution::Current);
                }
                _ => {}
            }
        }
        Ok(Self { base, resolutions })
    }
}

fn merged(
    c: &mut Config,
    scope: Scope,
    base: &Fields,
    mine: &Fields,
    meta: &FormMeta,
) -> anyhow::Result<()> {
    match config_merge::merge(c, scope, base, mine, &meta.resolutions) {
        Ok(new) => {
            validate_config(&new)?;
            *c = new;
            Ok(())
        }
        Err(MergeError::Invalid(e)) => Err(e),
        Err(e @ MergeError::Conflicts(_)) => Err(anyhow::Error::new(e)),
    }
}

/// The main `/config` form over the current config `c`: merged against its `base` when the
/// form carries one, else applied as submitted (API clients).
pub(super) fn server_form_edit(
    c: &mut Config,
    form: &HashMap<String, String>,
    meta: &FormMeta,
) -> anyhow::Result<()> {
    let Some(base) = &meta.base else {
        return crate::config::apply_config_form(c, form);
    };
    let mut mine = c.clone();
    parse_config_form(&mut mine, form)?;
    if form.contains_key("models_json") {
        validate_models(&mine.models)?;
    }
    let mine = form_fields(&mine, Scope::ServerForm);
    merged(c, Scope::ServerForm, base, &mine, meta)
}

fn field<'a>(form: &'a [(String, String)], k: &'a str) -> impl Iterator<Item = &'a str> {
    form.iter()
        .filter(move |(key, _)| key == k)
        .map(|(_, v)| v.as_str())
}

/// The Models card's submission as canonical fields, for the models listed in `base`: `enabled`
/// (repeated), `default_model`, `device.<name>` ("" = global), `threshold.<name>` ("" =
/// global). A field the form does not carry keeps its base value.
pub(super) fn card_mine(base: &Fields, form: &[(String, String)]) -> anyhow::Result<Fields> {
    let names: Vec<String> = base
        .keys()
        .filter_map(|k| match config_merge::parse_key(k) {
            config_merge::Key::Model {
                field: "enabled",
                name,
            } => Some(name.to_string()),
            _ => None,
        })
        .collect();
    let display = |n: &str| {
        base.get(&format!("model:name:{n}"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| n.to_string())
    };
    let known = |name: &str| {
        let k = normalize_name(name);
        names.iter().find(|n| **n == k).cloned()
    };
    let mut mine = base.clone();
    let mut enabled: Vec<String> = Vec::new();
    for e in field(form, "enabled") {
        match known(e) {
            Some(n) => enabled.push(n),
            None => anyhow::bail!("'{e}' is not one of the configured models"),
        }
    }
    let default = field(form, "default_model")
        .next()
        .map(str::trim)
        .filter(|d| !d.is_empty());
    if let Some(d) = default {
        let Some(n) = known(d) else {
            anyhow::bail!("default model '{d}' is not one of the configured models");
        };
        // The card's radio values are the configured names; keep the base's spelling.
        let name = match base.get(config_merge::DEFAULT_KEY).and_then(Value::as_str) {
            Some(b) if normalize_name(b) == n => b.to_string(),
            _ => d.to_string(),
        };
        mine.insert(config_merge::DEFAULT_KEY.into(), Value::String(name));
        if !enabled.contains(&n) {
            enabled.push(n);
        }
    }
    for n in &names {
        mine.insert(
            format!("model:enabled:{n}"),
            Value::Bool(enabled.contains(n)),
        );
    }
    for (k, v) in form {
        if let Some(name) = k.strip_prefix("device.") {
            let Some(n) = known(name) else {
                anyhow::bail!("'{name}' is not one of the configured models");
            };
            let v = v.trim();
            let value = if v.is_empty() || v.eq_ignore_ascii_case("global") {
                Value::Null
            } else {
                crate::backend::spec::parse(v)
                    .map_err(|e| anyhow::anyhow!("model '{}' device: {e}", display(&n)))?;
                Value::String(v.to_string())
            };
            mine.insert(format!("model:device:{n}"), value);
        } else if let Some(name) = k.strip_prefix("threshold.") {
            let Some(n) = known(name) else {
                anyhow::bail!("'{name}' is not one of the configured models");
            };
            let v = v.trim();
            let value = if v.is_empty() {
                Value::Null
            } else {
                let t: f32 = v.parse().map_err(|_| {
                    anyhow::anyhow!("model '{}' threshold: '{v}' is not a number", display(&n))
                })?;
                if !(t.is_finite() && (0.0..=1.0).contains(&t)) {
                    anyhow::bail!(
                        "model '{}' threshold: must be between 0 and 1, got {v}",
                        display(&n)
                    );
                }
                json!(crate::config::round_threshold_f64(t))
            };
            mine.insert(format!("model:confidence_threshold:{n}"), value);
        }
    }
    Ok(mine)
}

/// The Models card over the current config `c`: merged against its `base` when present, else
/// applied as submitted (enabled set, default, per-model devices).
pub(super) fn models_card_edit(
    c: &mut Config,
    form: &[(String, String)],
    meta: &FormMeta,
) -> anyhow::Result<()> {
    let Some(base) = &meta.base else {
        let enabled: Vec<String> = field(form, "enabled").map(str::to_string).collect();
        let default = field(form, "default_model").next();
        crate::config::apply_models_selection(c, &enabled, default)?;
        let devices: Vec<(String, Option<String>)> = form
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("device.")
                    .map(|name| (name.to_string(), Some(v.clone())))
            })
            .collect();
        crate::config::apply_model_devices(c, &devices)?;
        let mut thresholds = Vec::new();
        for (k, v) in form {
            if let Some(name) = k.strip_prefix("threshold.") {
                let v = v.trim();
                let t = if v.is_empty() {
                    None
                } else {
                    Some(
                        v.parse::<f32>()
                            .map_err(|_| anyhow::anyhow!("threshold: '{v}' is not a number"))?,
                    )
                };
                thresholds.push((name.to_string(), t));
            }
        }
        for (name, t) in thresholds {
            let key = normalize_name(&name);
            let Some(m) = c
                .models
                .iter_mut()
                .find(|m| normalize_name(&m.effective_name()) == key)
            else {
                anyhow::bail!("'{name}' is not one of the configured models");
            };
            m.confidence_threshold = t;
        }
        return validate_config(c);
    };
    let mine = card_mine(base, form)?;
    merged(c, Scope::ModelsCard, base, &mine, meta)
}

/// The conflicts of a failed save, if that is why it failed.
pub(super) fn conflicts_of(e: &anyhow::Error) -> Option<&[Conflict]> {
    match e.downcast_ref::<MergeError>() {
        Some(MergeError::Conflicts(c)) => Some(c),
        _ => None,
    }
}

pub(super) fn wants_json(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("application/json"))
}

/// 409 JSON for a conflict: the fields with the user's and the current values.
pub(super) fn conflict_json(state: &AppState, conflicts: &[Conflict]) -> Response {
    (
        StatusCode::CONFLICT,
        Json(json!({
            "success": false,
            "conflict": true,
            "message": config_merge::conflict_message(conflicts),
            "conflicts": conflicts,
            "revision": state.config.revision(),
        })),
    )
        .into_response()
}

/// JSON answer of a Config page form.
pub(super) fn saved_json(state: &AppState, message: String, restarting: bool) -> Response {
    Json(json!({
        "success": true,
        "message": message,
        "revision": state.config.revision(),
        "restarting": restarting,
    }))
    .into_response()
}

pub(super) fn error_json(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(json!({ "success": false, "message": message })),
    )
        .into_response()
}

/// One row of the conflict card.
pub(super) struct ConflictRow {
    pub key: String,
    pub label: String,
    pub mine: String,
    pub current: String,
    /// JSON of the current value ("" = absent), posted back as `resolve_cur.<key>`.
    pub current_json: String,
}

/// The conflict card of the Config page (no-JS answer): the user's submission is posted again
/// with a choice per field.
pub(super) struct ConflictView {
    pub action: &'static str,
    pub message: String,
    pub rows: Vec<ConflictRow>,
    /// The user's submission (hidden fields).
    pub replay: Vec<(String, String)>,
}

impl ConflictView {
    pub fn new(action: &'static str, conflicts: &[Conflict], form: &[(String, String)]) -> Self {
        Self {
            action,
            message: config_merge::conflict_message(conflicts),
            rows: conflicts
                .iter()
                .map(|c| ConflictRow {
                    key: c.key.clone(),
                    label: c.label.clone(),
                    mine: c.mine_text.clone(),
                    current: c.current_text.clone(),
                    current_json: c.current.as_ref().map(Value::to_string).unwrap_or_default(),
                })
                .collect(),
            replay: form
                .iter()
                .filter(|(k, _)| {
                    !k.starts_with("resolve.")
                        && !k.starts_with("resolve_cur.")
                        && k != "resolve_all"
                })
                .cloned()
                .collect(),
        }
    }
}

/// The Config page with the conflict card (409).
pub(super) fn conflict_page(
    state: &AppState,
    action: &'static str,
    conflicts: &[Conflict],
    form: &[(String, String)],
) -> Response {
    let mut t = super::config_template(state, None, None, None);
    t.conflict = Some(ConflictView::new(action, conflicts, form));
    (StatusCode::CONFLICT, render(&t)).into_response()
}
