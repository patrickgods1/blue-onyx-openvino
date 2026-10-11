//! Field-level view of the config for the web UI: flatten a [`Config`] into canonical
//! `key -> JSON value` maps, describe what changed between two configs, compute what a restart
//! would apply, and 3-way merge a form submission into the current config.
//!
//! Keys:
//! - `port`, `device`, ... : top-level config fields (`benchmark.<field>` for the benchmark section);
//! - `model::<name>`: the model exists (value `true`);
//! - `model:<field>:<name>`: a field of a model entry (`device`, `confidence_threshold`, ...);
//! - `default`: the model that serves `/v1/vision/detection` (the Models card's radio);
//! - `models:order`: the models' order (normalized names).
//!
//! `<name>` is the model's normalized name ([`normalize_name`]), so entries are identified by
//! name, not by position.
//!
//! A form posts the canonical values it was rendered from (its *base*). For every key the merge
//! keeps the current server value when the user did not change it (submitted == base), takes
//! the user's value when only the user changed it (current == base), accepts it when both made
//! the same change, and reports a [`Conflict`] when both changed it differently: nothing is
//! overwritten silently.

use crate::config::{Config, ModelConfig};
use crate::registry::normalize_name;
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// Canonical field values (see the module docs for the keys).
pub type Fields = BTreeMap<String, Value>;

pub const ORDER_KEY: &str = "models:order";
pub const DEFAULT_KEY: &str = "default";

/// Which form a set of fields belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// The main `/config` form: [`crate::config::FORM_FIELDS`] and the whole models list (JSON).
    ServerForm,
    /// The Models card: per model `enabled`, `device`, `confidence_threshold`, and the default.
    ModelsCard,
}

/// Model fields the Models card edits.
pub const CARD_MODEL_FIELDS: &[&str] = &["enabled", "device", "confidence_threshold"];

pub fn model_key(field: &str, name: &str) -> String {
    format!("model:{field}:{}", normalize_name(name))
}

pub fn presence_key(name: &str) -> String {
    format!("model::{}", normalize_name(name))
}

/// A parsed key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key<'a> {
    Global(&'a str),
    Bench(&'a str),
    Order,
    Default,
    Presence(&'a str),
    Model { field: &'a str, name: &'a str },
}

pub fn parse_key(k: &str) -> Key<'_> {
    if k == ORDER_KEY {
        return Key::Order;
    }
    if k == DEFAULT_KEY {
        return Key::Default;
    }
    if let Some(rest) = k.strip_prefix("model:")
        && let Some((field, name)) = rest.split_once(':')
    {
        return if field.is_empty() {
            Key::Presence(name)
        } else {
            Key::Model { field, name }
        };
    }
    if let Some(f) = k.strip_prefix("benchmark.") {
        return Key::Bench(f);
    }
    Key::Global(k)
}

/// The config as a JSON object (thresholds rounded, see [`crate::config::thr`]).
fn config_object(c: &Config) -> Map<String, Value> {
    let mut n = c.clone();
    n.normalize();
    match serde_json::to_value(&n) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

fn model_object(m: &ModelConfig) -> Map<String, Value> {
    match serde_json::to_value(m) {
        Ok(Value::Object(o)) => o,
        _ => Map::new(),
    }
}

/// Normalized name and JSON object of every model, in config order (names filled in).
fn models_of(c: &Config) -> Vec<(String, Map<String, Value>)> {
    let mut n = c.clone();
    n.normalize();
    n.models
        .iter()
        .map(|m| (normalize_name(&m.effective_name()), model_object(m)))
        .collect()
}

/// Effective default model's name (what serves `/v1/vision/detection`), as JSON.
fn default_value(c: &Config) -> Value {
    c.effective_default_index()
        .map(|i| Value::String(c.models[i].effective_name()))
        .unwrap_or(Value::Null)
}

/// Every field of `c`: globals, `benchmark.*`, `default_model`, models (presence, fields, order).
pub fn flatten_all(c: &Config) -> Fields {
    let mut out = Fields::new();
    for (k, v) in config_object(c) {
        match (k.as_str(), v) {
            ("models", _) => {}
            ("benchmark", Value::Object(b)) => {
                for (bk, bv) in b {
                    out.insert(format!("benchmark.{bk}"), bv);
                }
            }
            (_, v) => {
                out.insert(k, v);
            }
        }
    }
    add_models(&mut out, c, None);
    out
}

fn add_models(out: &mut Fields, c: &Config, only: Option<&[&str]>) {
    let models = models_of(c);
    if only.is_none() {
        out.insert(
            ORDER_KEY.to_string(),
            Value::Array(
                models
                    .iter()
                    .map(|(n, _)| Value::String(n.clone()))
                    .collect(),
            ),
        );
    }
    for (name, obj) in models {
        if only.is_none() {
            out.insert(format!("model::{name}"), Value::Bool(true));
        }
        for (f, v) in obj {
            if only.is_none_or(|o| o.contains(&f.as_str())) {
                out.insert(format!("model:{f}:{name}"), v);
            }
        }
    }
}

/// The fields a form shows, as rendered from `c` (its base).
pub fn form_fields(c: &Config, scope: Scope) -> Fields {
    let mut out = Fields::new();
    match scope {
        Scope::ServerForm => {
            let obj = config_object(c);
            for &k in crate::config::FORM_FIELDS {
                if let Some(v) = obj.get(k) {
                    out.insert(k.to_string(), v.clone());
                }
            }
            add_models(&mut out, c, None);
        }
        Scope::ModelsCard => {
            out.insert(DEFAULT_KEY.to_string(), default_value(c));
            add_models(&mut out, c, Some(CARD_MODEL_FIELDS));
        }
    }
    out
}

/// One field both the user and someone else changed, to different values.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Conflict {
    pub key: String,
    /// "IPcam-general · device", "Port".
    pub label: String,
    /// Value when the form was rendered (None = absent, e.g. the model did not exist; omitted
    /// from JSON, unlike a JSON `null` value).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base: Option<Value>,
    /// The user's value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mine: Option<Value>,
    /// The server's current value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<Value>,
    /// Display text of `mine` / `current`.
    pub mine_text: String,
    pub current_text: String,
}

/// How the user resolved a conflict on the conflict page.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    /// Keep the user's value; `seen` is the server value the user saw when choosing (if the
    /// server changed the field again since, it is a conflict again).
    Mine { seen: Option<Value> },
    /// Take the server's current value.
    Current,
}

/// Why a merge produced no config.
#[derive(Debug)]
pub enum MergeError {
    Conflicts(Vec<Conflict>),
    Invalid(anyhow::Error),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::Conflicts(c) => write!(f, "{}", conflict_message(c)),
            MergeError::Invalid(e) => write!(f, "{e:#}"),
        }
    }
}

impl std::error::Error for MergeError {}

/// "Not saved: changed elsewhere since this page was loaded: IPcam-general · device (yours:
/// openvino:gpu, now: openvino:cpu)."
pub fn conflict_message(conflicts: &[Conflict]) -> String {
    let parts: Vec<String> = conflicts
        .iter()
        .map(|c| {
            format!(
                "{} (yours: {}, now: {})",
                c.label, c.mine_text, c.current_text
            )
        })
        .collect();
    format!(
        "Not saved: {} changed elsewhere since this page was loaded: {}. Choose which value \
         to keep.",
        if conflicts.len() == 1 {
            "a field was"
        } else {
            "fields were"
        },
        parts.join("; ")
    )
}

/// Display names of the models mentioned in `maps` (normalized name -> name as configured).
fn display_names(maps: &[&Fields]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for m in maps {
        for (k, v) in m.iter() {
            if let Key::Model {
                field: "name",
                name,
            } = parse_key(k)
                && let Some(s) = v.as_str()
            {
                out.entry(name.to_string()).or_insert_with(|| s.to_string());
            }
        }
    }
    out
}

fn field_label(field: &str) -> String {
    match field {
        "confidence_threshold" => "threshold".into(),
        "enabled" => "load".into(),
        "gpu_precision" => "GPU precision".into(),
        "object_filter" => "object filter".into(),
        other => other.replace('_', " "),
    }
}

fn global_label(k: &str) -> String {
    match k {
        "port" => "Port".into(),
        "request_timeout_secs" => "Request timeout".into(),
        "worker_queue_size" => "Worker queue size".into(),
        "device" => "Device".into(),
        "gpu_index" => "GPU index".into(),
        "force_cpu" => "Force CPU".into(),
        "cache_dir" => "Cache dir".into(),
        "confidence_threshold" => "Confidence threshold".into(),
        "nms_iou" => "NMS IoU".into(),
        "object_filter" => "Object filter".into(),
        "log_level" => "Log level".into(),
        "log_path" => "Log directory".into(),
        "save_image_path" => "Save annotated images to".into(),
        "save_ref_image" => "Save the unannotated image".into(),
        "intra_threads" => "CPU threads".into(),
        "auto_download" => "Auto download".into(),
        "allow_large_downloads" => "Allow large downloads".into(),
        "default_model" => "Default model".into(),
        other => other.replace('_', " "),
    }
}

/// Human label of a key, with model display names from `names`.
pub fn label(key: &str, names: &BTreeMap<String, String>) -> String {
    let show = |n: &str| names.get(n).cloned().unwrap_or_else(|| n.to_string());
    match parse_key(key) {
        Key::Global(k) => global_label(k),
        Key::Bench(k) => format!("Benchmark {}", k.replace('_', " ")),
        Key::Order => "Models order".into(),
        Key::Default => "Default model".into(),
        Key::Presence(n) => format!("Model {}", show(n)),
        Key::Model { field, name } => format!("{} \u{b7} {}", show(name), field_label(field)),
    }
}

/// Display text of a value of `key` (None = absent).
pub fn value_text(key: &str, v: Option<&Value>) -> String {
    let (field, presence) = match parse_key(key) {
        Key::Model { field, .. } => (field, false),
        Key::Presence(_) => ("", true),
        Key::Global(k) => (k, false),
        Key::Default => ("default", false),
        _ => ("", false),
    };
    match v {
        None if presence => "removed".into(),
        None => "(none)".into(),
        Some(Value::Bool(true)) if presence => "present".into(),
        Some(Value::Null) => match field {
            "device" | "confidence_threshold" | "object_filter" => "global".into(),
            "default" => "first enabled".into(),
            _ => "(none)".into(),
        },
        Some(Value::String(s)) if s.is_empty() => "(empty)".into(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => if *b { "on" } else { "off" }.into(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => {
            if a.is_empty() {
                "(none)".into()
            } else {
                a.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        }
        Some(v) => v.to_string(),
    }
}

/// One changed field, for the change log.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldChange {
    pub key: String,
    pub label: String,
    pub from: String,
    pub to: String,
}

/// Fields that differ between `old` and `new`, and a one-line summary grouped by model:
/// "IPcam-general device → openvino:cpu, threshold → 0.35; Port → 4000".
pub fn describe_changes(old: &Config, new: &Config) -> (Vec<FieldChange>, String) {
    let a = flatten_all(old);
    let b = flatten_all(new);
    let names = display_names(&[&b, &a]);
    let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    let mut changes = Vec::new();
    // Summary parts: per model (in first-seen order) and globals.
    let mut model_parts: Vec<(String, Vec<(usize, String)>)> = Vec::new();
    // Model fields in the order the UI shows them.
    let rank = |f: &str| {
        [
            "name",
            "path",
            "family",
            "enabled",
            "device",
            "confidence_threshold",
        ]
        .iter()
        .position(|x| *x == f)
        .unwrap_or(99)
    };
    let mut global_parts: Vec<String> = Vec::new();
    let mut model_part = |name: &str, field: &str, text: String| {
        let display = names.get(name).cloned().unwrap_or_else(|| name.to_string());
        let item = (rank(field), text);
        match model_parts.iter_mut().find(|(n, _)| *n == display) {
            Some((_, v)) => v.push(item),
            None => model_parts.push((display, vec![item])),
        }
    };
    for k in keys {
        let (x, y) = (a.get(k), b.get(k));
        if x == y {
            continue;
        }
        let from = value_text(k, x);
        let to = value_text(k, y);
        changes.push(FieldChange {
            key: k.clone(),
            label: label(k, &names),
            from: from.clone(),
            to: to.clone(),
        });
        match parse_key(k) {
            Key::Presence(n) => {
                model_part(n, "", if y.is_some() { "added" } else { "removed" }.into());
            }
            // Fields of an added/removed model are covered by "added"/"removed".
            Key::Model { name, .. } if x.is_none() || y.is_none() => {
                let _ = name;
            }
            Key::Model { field, name } => {
                model_part(name, field, format!("{} \u{2192} {to}", field_label(field)));
            }
            Key::Order => global_parts.push("models reordered".into()),
            _ => global_parts.push(format!("{} \u{2192} {to}", label(k, &names))),
        }
    }
    let mut parts: Vec<String> = model_parts
        .into_iter()
        .map(|(n, mut v)| {
            v.sort_by_key(|(r, _)| *r);
            let v: Vec<String> = v.into_iter().map(|(_, t)| t).collect();
            format!("{n} {}", v.join(", "))
        })
        .collect();
    parts.extend(global_parts);
    (changes, parts.join("; "))
}

// ---------------------------------------------------------------------------------------------
// Merge

/// 3-way decision for one key: Ok(resolved value) or Err(()) on a conflict.
fn merge_one<'a>(
    b: Option<&'a Value>,
    m: Option<&'a Value>,
    c: Option<&'a Value>,
) -> Result<Option<&'a Value>, ()> {
    if m == b {
        Ok(c)
    } else if c == b || m == c {
        Ok(m)
    } else {
        Err(())
    }
}

fn names_of(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// `list` restricted to the names in `keep`, in `list` order.
fn restricted(list: &[String], keep: &BTreeSet<&str>) -> Vec<String> {
    list.iter()
        .filter(|n| keep.contains(n.as_str()))
        .cloned()
        .collect()
}

/// Merge the user's `mine` (submitted) over `current`, given the `base` the form was rendered
/// from, applying `resolutions` chosen on a conflict page. Returns the merged config (not yet
/// validated as a whole) or the conflicts.
pub fn merge(
    current: &Config,
    scope: Scope,
    base: &Fields,
    mine: &Fields,
    resolutions: &BTreeMap<String, Resolution>,
) -> Result<Config, MergeError> {
    let cur = form_fields(current, scope);
    let mut base = base.clone();
    for (k, r) in resolutions {
        let v = match r {
            Resolution::Mine { seen } => seen.clone(),
            Resolution::Current => mine.get(k).cloned(),
        };
        match v {
            Some(v) => base.insert(k.clone(), v),
            None => base.remove(k),
        };
    }
    let mut names = display_names(&[mine, &cur, &base]);
    for m in &current.models {
        let n = m.effective_name();
        names.entry(normalize_name(&n)).or_insert(n);
    }
    let keys: BTreeSet<&String> = base
        .keys()
        .chain(mine.keys())
        .chain(cur.keys())
        .filter(|k| parse_key(k) != Key::Order)
        .collect();
    let mut resolved = Fields::new();
    let mut conflicts = Vec::new();
    for k in keys {
        let (b, m, c) = (base.get(k), mine.get(k), cur.get(k));
        match merge_one(b, m, c) {
            Ok(Some(v)) => {
                resolved.insert(k.clone(), v.clone());
            }
            Ok(None) => {}
            Err(()) => conflicts.push(Conflict {
                key: k.clone(),
                label: label(k, &names),
                base: b.cloned(),
                mine: m.cloned(),
                current: c.cloned(),
                mine_text: value_text(k, m),
                current_text: value_text(k, c),
            }),
        }
    }
    // Order (server form only): the user's order when only the user reordered the models
    // both saw, the current order otherwise; a conflict when both reordered differently.
    let order = if scope == Scope::ServerForm {
        let final_names: BTreeSet<&str> = resolved
            .keys()
            .filter_map(|k| match parse_key(k) {
                Key::Presence(n) => Some(n),
                _ => None,
            })
            .collect();
        let (b, m, c) = (
            names_of(base.get(ORDER_KEY)),
            names_of(mine.get(ORDER_KEY)),
            names_of(cur.get(ORDER_KEY)),
        );
        let bs: BTreeSet<&str> = b.iter().map(String::as_str).collect();
        let ms: BTreeSet<&str> = m.iter().map(String::as_str).collect();
        let cs: BTreeSet<&str> = c.iter().map(String::as_str).collect();
        let bm: BTreeSet<&str> = bs.intersection(&ms).copied().collect();
        let bc: BTreeSet<&str> = bs.intersection(&cs).copied().collect();
        let user_reordered = restricted(&m, &bm) != restricted(&b, &bm);
        let server_reordered = restricted(&c, &bc) != restricted(&b, &bc);
        let common: BTreeSet<&str> = bm.intersection(&bc).copied().collect();
        let primary = if user_reordered && server_reordered {
            if restricted(&m, &common) != restricted(&c, &common) {
                let key = ORDER_KEY.to_string();
                conflicts.push(Conflict {
                    label: label(&key, &names),
                    mine_text: m.join(", "),
                    current_text: c.join(", "),
                    base: base.get(ORDER_KEY).cloned(),
                    mine: mine.get(ORDER_KEY).cloned(),
                    current: cur.get(ORDER_KEY).cloned(),
                    key,
                });
            }
            &c
        } else if user_reordered {
            &m
        } else {
            &c
        };
        let mut order: Vec<String> = Vec::new();
        for n in primary.iter().chain(c.iter()).chain(m.iter()) {
            if final_names.contains(n.as_str()) && !order.contains(n) {
                order.push(n.clone());
            }
        }
        Some(order)
    } else {
        None
    };
    if !conflicts.is_empty() {
        return Err(MergeError::Conflicts(conflicts));
    }
    apply(current, scope, &resolved, order.as_deref()).map_err(MergeError::Invalid)
}

/// Write `resolved` (every key of the scope; absent = removed) over `current`.
fn apply(
    current: &Config,
    scope: Scope,
    resolved: &Fields,
    order: Option<&[String]>,
) -> anyhow::Result<Config> {
    let mut obj = config_object(current);
    let mut default: Option<String> = None;
    for (k, v) in resolved {
        match parse_key(k) {
            Key::Global(g) => {
                obj.insert(g.to_string(), v.clone());
            }
            Key::Bench(b) => {
                if let Some(Value::Object(bo)) = obj.get_mut("benchmark") {
                    bo.insert(b.to_string(), v.clone());
                }
            }
            Key::Default => default = v.as_str().map(str::to_string),
            _ => {}
        }
    }
    // Models: per-model field maps.
    let current_models = models_of(current);
    let mut fields_by_model: BTreeMap<&str, Map<String, Value>> = BTreeMap::new();
    for (k, v) in resolved {
        if let Key::Model { field, name } = parse_key(k) {
            fields_by_model
                .entry(name)
                .or_default()
                .insert(field.to_string(), v.clone());
        }
    }
    let models: Vec<Value> = match scope {
        Scope::ServerForm => {
            let order = order.unwrap_or_default();
            let mut out = Vec::new();
            for name in order {
                // Every field of the model is in scope: the resolved ones are the model.
                let Some(fields) = fields_by_model.get(name.as_str()) else {
                    continue;
                };
                out.push(Value::Object(fields.clone()));
            }
            out
        }
        Scope::ModelsCard => current_models
            .into_iter()
            .map(|(name, mut o)| {
                for &f in CARD_MODEL_FIELDS {
                    match resolved.get(&format!("model:{f}:{name}")) {
                        Some(v) => {
                            o.insert(f.to_string(), v.clone());
                        }
                        None => {
                            o.remove(f);
                        }
                    }
                }
                Value::Object(o)
            })
            .collect(),
    };
    obj.insert("models".into(), Value::Array(models));
    let mut c: Config =
        serde_json::from_value(Value::Object(obj)).map_err(|e| anyhow::anyhow!("models: {e}"))?;
    if scope == Scope::ModelsCard {
        // The card's radio: the chosen default is enabled too (like `apply_models_selection`).
        let now = default_value(current);
        let chosen = default.map(Value::String).unwrap_or(Value::Null);
        if chosen != now {
            match chosen.as_str() {
                Some(name) => {
                    let key = normalize_name(name);
                    let Some(m) = c
                        .models
                        .iter_mut()
                        .find(|m| normalize_name(&m.effective_name()) == key)
                    else {
                        anyhow::bail!("default model '{name}' is not one of the configured models");
                    };
                    m.enabled = true;
                    c.default_model = Some(m.effective_name());
                }
                None => c.default_model = None,
            }
        }
    }
    c.normalize();
    Ok(c)
}

// ---------------------------------------------------------------------------------------------
// Pending restart

/// What a restart would change: differences between the config this generation runs with and
/// the current config.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Pending {
    /// Server-wide reasons ("Port: 32168 → 4000", "default model: a → b").
    pub global: Vec<String>,
    /// Reasons per model (normalized name), e.g. "device: auto → openvino:cpu".
    pub models: BTreeMap<String, Vec<String>>,
    /// Changes only a process restart applies (`log_path`).
    pub process: Vec<String>,
}

impl Pending {
    pub fn count(&self) -> usize {
        self.global.len() + self.models.values().map(Vec::len).sum::<usize>() + self.process.len()
    }

    pub fn restart_needed(&self) -> bool {
        self.count() > 0
    }

    pub fn for_model(&self, name: &str) -> &[String] {
        self.models
            .get(&normalize_name(name))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

/// Globals a restart does not need: the log level applies immediately, the benchmark section
/// is read when a benchmark starts.
fn live_global(k: &str) -> bool {
    matches!(k, "log_level" | "default_model") || k.starts_with("benchmark.")
}

fn thr_text(t: f32) -> String {
    format!("{:.2}", crate::config::round_threshold_f64(t))
}

/// What a restart would apply, comparing the running config with the current one.
pub fn pending(running: &Config, current: &Config) -> Pending {
    let mut out = Pending::default();
    let (a, b) = (flatten_all(running), flatten_all(current));
    let names = display_names(&[&b, &a]);
    let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    for k in keys {
        let Key::Global(g) = parse_key(k) else {
            continue;
        };
        let (x, y) = (a.get(k), b.get(k));
        if x == y || live_global(g) {
            continue;
        }
        let text = format!(
            "{}: {} \u{2192} {}",
            label(k, &names),
            value_text(k, x),
            value_text(k, y)
        );
        if g == "log_path" {
            out.process
                .push(format!("{text} (needs a process restart)"));
        } else {
            out.global.push(text);
        }
    }
    let name_of = |c: &Config, i: usize| c.models[i].effective_name();
    let (da, db) = (
        running
            .effective_default_index()
            .map(|i| name_of(running, i)),
        current
            .effective_default_index()
            .map(|i| name_of(current, i)),
    );
    if da.as_deref().map(normalize_name) != db.as_deref().map(normalize_name) {
        out.global.push(format!(
            "default model: {} \u{2192} {}",
            da.as_deref().unwrap_or("none"),
            db.as_deref().unwrap_or("none")
        ));
    }
    let find = |c: &'_ Config, key: &str| -> Option<ModelConfig> {
        c.models
            .iter()
            .find(|m| normalize_name(&m.effective_name()) == key)
            .cloned()
    };
    let mut all: Vec<String> = Vec::new();
    for m in running.models.iter().chain(current.models.iter()) {
        let k = normalize_name(&m.effective_name());
        if !all.contains(&k) {
            all.push(k);
        }
    }
    for key in all {
        let (r, c) = (find(running, &key), find(current, &key));
        let mut reasons = Vec::new();
        match (r.filter(|m| m.enabled), c) {
            (None, Some(c)) if c.enabled => reasons.push("enabled; not loaded yet".to_string()),
            (None, _) => {}
            (Some(_), None) => reasons.push("removed from the config; still loaded".to_string()),
            (Some(_), Some(c)) if !c.enabled => reasons.push("disabled; still loaded".to_string()),
            (Some(r), Some(c)) => {
                let (dr, dc) = (running.device_for(&r), current.device_for(&c));
                if dr != dc {
                    reasons.push(format!("device: {dr} \u{2192} {dc}"));
                }
                let tr = r
                    .confidence_threshold
                    .unwrap_or(running.confidence_threshold);
                let tc = c
                    .confidence_threshold
                    .unwrap_or(current.confidence_threshold);
                if thr_text(tr) != thr_text(tc) {
                    reasons.push(format!(
                        "threshold: {} \u{2192} {}",
                        thr_text(tr),
                        thr_text(tc)
                    ));
                }
                let (or, oc) = (model_object(&r), model_object(&c));
                for (f, v) in &oc {
                    if matches!(
                        f.as_str(),
                        "name" | "device" | "confidence_threshold" | "enabled"
                    ) {
                        continue;
                    }
                    if or.get(f) != Some(v) {
                        reasons.push(format!("{} changed", field_label(f)));
                    }
                }
            }
        }
        if !reasons.is_empty() {
            out.models.insert(key, reasons);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> Config {
        let mut c = Config {
            models: ["IPcam-general", "yolo26s"]
                .iter()
                .map(|n| ModelConfig {
                    name: Some(n.to_string()),
                    path: format!("models/{n}.onnx").into(),
                    ..Default::default()
                })
                .collect(),
            default_model: Some("IPcam-general".into()),
            ..Default::default()
        };
        c.normalize();
        c
    }

    fn set(f: &mut Fields, k: &str, v: Value) {
        f.insert(k.to_string(), v);
    }

    #[test]
    fn keys_round_trip() {
        assert_eq!(parse_key("port"), Key::Global("port"));
        assert_eq!(parse_key("benchmark.warmup"), Key::Bench("warmup"));
        assert_eq!(parse_key("model::a.b:c"), Key::Presence("a.b:c"));
        assert_eq!(
            parse_key("model:device:a.b:c"),
            Key::Model {
                field: "device",
                name: "a.b:c"
            }
        );
        assert_eq!(
            model_key("device", "IPcam-general.onnx"),
            "model:device:ipcam-general"
        );
    }

    #[test]
    fn thresholds_flatten_as_two_decimals() {
        let mut c = cfg();
        c.models[0].confidence_threshold = Some(0.35);
        c.confidence_threshold = 0.4;
        let f = flatten_all(&c);
        assert_eq!(
            f["model:confidence_threshold:ipcam-general"].to_string(),
            "0.35"
        );
        assert_eq!(f["confidence_threshold"].to_string(), "0.4");
        assert_eq!(f["nms_iou"].to_string(), "0.5");
    }

    /// Scalar field: the five merge cases.
    #[test]
    fn scalar_merge_cases() {
        let base_cfg = cfg();
        let base = form_fields(&base_cfg, Scope::ServerForm);
        let none = BTreeMap::new();
        // No change anywhere.
        let out = merge(&base_cfg, Scope::ServerForm, &base, &base, &none).unwrap();
        assert_eq!(out, base_cfg);
        // User-only change.
        let mut mine = base.clone();
        set(&mut mine, "port", json!(4000));
        let out = merge(&base_cfg, Scope::ServerForm, &base, &mine, &none).unwrap();
        assert_eq!(out.port, 4000);
        // Server-only change: kept.
        let mut server = base_cfg.clone();
        server.port = 5000;
        let out = merge(&server, Scope::ServerForm, &base, &base, &none).unwrap();
        assert_eq!(out.port, 5000);
        // Both, same value.
        server.port = 4000;
        let out = merge(&server, Scope::ServerForm, &base, &mine, &none).unwrap();
        assert_eq!(out.port, 4000);
        // Both, different values: conflict naming both values.
        server.port = 5000;
        let Err(MergeError::Conflicts(c)) = merge(&server, Scope::ServerForm, &base, &mine, &none)
        else {
            panic!("expected a conflict");
        };
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].key, "port");
        assert_eq!(c[0].label, "Port");
        assert_eq!(
            (c[0].mine_text.as_str(), c[0].current_text.as_str()),
            ("4000", "5000")
        );
        // Resolutions.
        let mut r = BTreeMap::new();
        r.insert(
            "port".to_string(),
            Resolution::Mine {
                seen: Some(json!(5000)),
            },
        );
        assert_eq!(
            merge(&server, Scope::ServerForm, &base, &mine, &r)
                .unwrap()
                .port,
            4000
        );
        r.insert("port".to_string(), Resolution::Current);
        assert_eq!(
            merge(&server, Scope::ServerForm, &base, &mine, &r)
                .unwrap()
                .port,
            5000
        );
        // "Use mine" against a value the user did not see is a conflict again.
        r.insert(
            "port".to_string(),
            Resolution::Mine {
                seen: Some(json!(4500)),
            },
        );
        assert!(merge(&server, Scope::ServerForm, &base, &mine, &r).is_err());
    }

    /// The stale-page bug: the models JSON of a page loaded before a benchmark apply carries
    /// the old device/threshold; saving it must keep the applied values.
    #[test]
    fn models_json_merge_per_model_field() {
        let base_cfg = cfg();
        let base = form_fields(&base_cfg, Scope::ServerForm);
        let mut server = base_cfg.clone();
        server.models[0].device = Some("openvino:cpu".into());
        server.models[0].confidence_threshold = Some(0.35);
        let none = BTreeMap::new();
        // Unchanged submission (stale page): server values kept.
        let out = merge(&server, Scope::ServerForm, &base, &base, &none).unwrap();
        assert_eq!(out, server);
        // The user changed another model's field: both changes survive.
        let mut mine = base.clone();
        set(&mut mine, "model:lazy:yolo26s", json!(true));
        let out = merge(&server, Scope::ServerForm, &base, &mine, &none).unwrap();
        assert_eq!(out.models[0].device.as_deref(), Some("openvino:cpu"));
        assert_eq!(out.models[0].confidence_threshold, Some(0.35));
        assert!(out.models[1].lazy);
        // The user changed the same field differently: conflict.
        set(&mut mine, "model:device:ipcam-general", json!("ort:coreml"));
        let Err(MergeError::Conflicts(c)) = merge(&server, Scope::ServerForm, &base, &mine, &none)
        else {
            panic!("expected a conflict");
        };
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].label, "IPcam-general \u{b7} device");
        assert_eq!(c[0].mine_text, "ort:coreml");
        assert_eq!(c[0].current_text, "openvino:cpu");
        // Both set the same value: fine.
        set(
            &mut mine,
            "model:device:ipcam-general",
            json!("openvino:cpu"),
        );
        assert!(merge(&server, Scope::ServerForm, &base, &mine, &none).is_ok());
    }

    #[test]
    fn models_json_add_remove_and_order() {
        let base_cfg = cfg();
        let base = form_fields(&base_cfg, Scope::ServerForm);
        let none = BTreeMap::new();
        // Server added a model; the user removed yolo26s: both happen.
        let mut server = base_cfg.clone();
        server.models.push(ModelConfig {
            name: Some("extra".into()),
            path: "models/extra.onnx".into(),
            ..Default::default()
        });
        let mut user = base_cfg.clone();
        user.models.remove(1);
        let mine = form_fields(&user, Scope::ServerForm);
        let out = merge(&server, Scope::ServerForm, &base, &mine, &none).unwrap();
        let names: Vec<String> = out.models.iter().map(|m| m.effective_name()).collect();
        assert_eq!(names, ["IPcam-general", "extra"]);
        // The user removed a model the server changed: conflict.
        let mut server2 = base_cfg.clone();
        server2.models[1].device = Some("openvino:cpu".into());
        let Err(MergeError::Conflicts(c)) = merge(&server2, Scope::ServerForm, &base, &mine, &none)
        else {
            panic!("expected a conflict");
        };
        assert!(c.iter().any(|c| c.key == "model:device:yolo26s"), "{c:?}");
        // The user reordered; the server changed a field: order and field both applied.
        let mut user = base_cfg.clone();
        user.models.reverse();
        let mine = form_fields(&user, Scope::ServerForm);
        let out = merge(&server2, Scope::ServerForm, &base, &mine, &none).unwrap();
        assert_eq!(out.models[0].effective_name(), "yolo26s");
        assert_eq!(out.models[0].device.as_deref(), Some("openvino:cpu"));
        // The user added a model with a name the server also added differently: conflict.
        let mut user = base_cfg.clone();
        user.models.push(ModelConfig {
            name: Some("extra".into()),
            path: "models/other.onnx".into(),
            ..Default::default()
        });
        let mine = form_fields(&user, Scope::ServerForm);
        assert!(matches!(
            merge(&server, Scope::ServerForm, &base, &mine, &none),
            Err(MergeError::Conflicts(_))
        ));
    }

    #[test]
    fn models_card_merge() {
        let base_cfg = cfg();
        let base = form_fields(&base_cfg, Scope::ModelsCard);
        assert_eq!(base["default"], json!("IPcam-general"));
        assert_eq!(base["model:device:ipcam-general"], Value::Null);
        let none = BTreeMap::new();
        let mut server = base_cfg.clone();
        server.models[0].device = Some("openvino:cpu".into());
        server.models[0].confidence_threshold = Some(0.35);
        // Stale card: unchanged rows keep the server's device and threshold.
        let out = merge(&server, Scope::ModelsCard, &base, &base, &none).unwrap();
        assert_eq!(out, server);
        // The user unchecks yolo26s and sets a threshold on it.
        let mut mine = base.clone();
        set(&mut mine, "model:enabled:yolo26s", json!(false));
        set(&mut mine, "model:confidence_threshold:yolo26s", json!(0.6));
        let out = merge(&server, Scope::ModelsCard, &base, &mine, &none).unwrap();
        assert!(!out.models[1].enabled);
        assert_eq!(out.models[1].confidence_threshold, Some(0.6));
        assert_eq!(out.models[0].device.as_deref(), Some("openvino:cpu"));
        // Default radio: choosing a disabled model enables it.
        let mut server3 = base_cfg.clone();
        server3.models[1].enabled = false;
        let base3 = form_fields(&server3, Scope::ModelsCard);
        let mut mine = base3.clone();
        set(&mut mine, "default", json!("yolo26s"));
        let out = merge(&server3, Scope::ModelsCard, &base3, &mine, &none).unwrap();
        assert!(out.models[1].enabled);
        assert_eq!(out.default_model.as_deref(), Some("yolo26s"));
        // A model added on the server after the card was rendered is left alone.
        let mut server4 = base_cfg.clone();
        server4.models.push(ModelConfig {
            name: Some("extra".into()),
            path: "models/extra.onnx".into(),
            enabled: false,
            ..Default::default()
        });
        let out = merge(&server4, Scope::ModelsCard, &base, &base, &none).unwrap();
        assert_eq!(out, server4);
        // Both changed the device differently: conflict.
        let mut mine = base.clone();
        set(&mut mine, "model:device:ipcam-general", json!("ort:cpu"));
        let Err(MergeError::Conflicts(c)) = merge(&server, Scope::ModelsCard, &base, &mine, &none)
        else {
            panic!("expected a conflict");
        };
        assert_eq!(c[0].key, "model:device:ipcam-general");
        assert_eq!(c[0].label, "IPcam-general \u{b7} device");
        assert!(conflict_message(&c).contains("yours: ort:cpu, now: openvino:cpu"));
    }

    #[test]
    fn change_summaries() {
        let a = cfg();
        let mut b = a.clone();
        b.models[0].device = Some("openvino:cpu".into());
        b.models[0].confidence_threshold = Some(0.35);
        b.port = 4000;
        let (changes, summary) = describe_changes(&a, &b);
        assert_eq!(changes.len(), 3);
        assert_eq!(
            summary,
            "IPcam-general device \u{2192} openvino:cpu, threshold \u{2192} 0.35; Port \u{2192} 4000"
        );
        b.models.push(ModelConfig {
            name: Some("x".into()),
            path: "x.onnx".into(),
            ..Default::default()
        });
        let (_, summary) = describe_changes(&a, &b);
        assert!(summary.contains("x added"), "{summary}");
    }

    #[test]
    fn pending_restart_reasons() {
        let running = cfg();
        assert!(!pending(&running, &running).restart_needed());
        let mut c = running.clone();
        c.models[0].device = Some("openvino:cpu".into());
        c.models[0].confidence_threshold = Some(0.35);
        c.log_level = crate::config::LogLevel::Debug;
        c.benchmark.warmup = 9;
        let p = pending(&running, &c);
        assert_eq!(
            p.for_model("IPcam-general"),
            [
                "device: auto \u{2192} openvino:cpu",
                "threshold: 0.50 \u{2192} 0.35"
            ]
        );
        assert!(p.global.is_empty(), "{:?}", p.global);
        assert_eq!(p.count(), 2);
        // Global changes and enable/disable.
        c.port = 4000;
        c.models[1].enabled = false;
        c.log_path = Some("logs".into());
        let p = pending(&running, &c);
        assert_eq!(p.global, ["Port: 32168 \u{2192} 4000"]);
        assert_eq!(p.for_model("yolo26s"), ["disabled; still loaded"]);
        assert_eq!(p.process.len(), 1);
        // A threshold that only differs below 0.01 is not pending.
        let mut d = running.clone();
        d.confidence_threshold = 0.5000001;
        assert!(!pending(&running, &d).restart_needed());
    }
}
