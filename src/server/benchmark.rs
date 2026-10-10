//! Benchmark page (`/benchmark`) and API:
//! - `GET /v1/benchmark`: the current/last run (state, progress, ETA, partial results), the
//!   saved results (`benchmark.json`, without per-image details), the cross-model ranking, the
//!   datasets and the config's per-model devices;
//! - `POST /v1/benchmark`: start a run; `POST /v1/benchmark/cancel`;
//! - `POST /v1/benchmark/apply`: write per-model devices into the config;
//! - `POST /v1/benchmark/settings`: save the form as the `benchmark` config defaults;
//! - `GET /v1/benchmark/images?model=&device=`: per-image drill-down (ground truth and
//!   predictions); `GET /v1/benchmark/image?set=&file=`: the image file itself (only files
//!   listed in the saved results).
//!
//! The run itself is [`crate::benchmark::service`].

use super::{
    AppState, DevicesSnapshot, devices_snapshot, global_device_label, render, save_config,
    schedule_restart,
};
use crate::benchmark::images::{self, ImageSet};
use crate::benchmark::service::{RunContext, RunRequest, default_models};
use crate::benchmark::{BenchmarkResults, configured_device, results_path};
use crate::config::{Config, DatasetRef, DeviceChange, apply_model_devices};
use askama::Template;
use axum::Json;
use axum::extract::rejection::{FormRejection, QueryRejection};
use axum::extract::{Form, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

/// Shown on the page and in the API: benchmarking competes with live requests.
pub(super) const LIVE_WARNING: &str = "Benchmarking runs on the same CPU/GPU as live detections: \
     Blue Iris requests are slower while it runs, and its numbers are noisier when the server is \
     busy. Run it when the cameras are quiet.";

struct ModelOption {
    name: String,
    family: String,
    enabled: bool,
    exists: bool,
    checked: bool,
}

struct DeviceOptionRow {
    spec: String,
    label: String,
    runnable: bool,
    checked: bool,
    reason: Option<String>,
}

/// One dataset row of the form.
struct DatasetRow {
    /// Value posted as `dataset` (`coco-cctv`, `sample`, `dir:/path`).
    value: String,
    title: String,
    /// "120 images · 45 MB · CC BY 4.0".
    detail: String,
    /// `installed`, `partial`, `available`, `embedded`, `folder`, `missing`, `unavailable`.
    state: &'static str,
    state_text: String,
    checked: bool,
    /// Resource id to download (built-in sets not fully installed).
    download_id: Option<String>,
    /// A user folder (removable from the defaults).
    folder: bool,
    selectable: bool,
}

#[derive(Template)]
#[template(path = "benchmark.html")]
struct BenchmarkTemplate {
    nav: &'static str,
    version: &'static str,
    models: Vec<ModelOption>,
    devices: Vec<DeviceOptionRow>,
    datasets: Vec<DatasetRow>,
    reference_models: Vec<(String, bool)>,
    repeat: usize,
    warmup: usize,
    max_images: usize,
    accuracy_weight: String,
    warning: &'static str,
    force_cpu: bool,
    results_file: String,
    speed_thresholds: String,
    accuracy_thresholds: String,
}

fn model_options(c: &Config) -> Vec<ModelOption> {
    let defaults = if c.benchmark.models.is_empty() {
        default_models(c)
    } else {
        c.benchmark.models.clone()
    };
    let key = crate::registry::normalize_name;
    c.models
        .iter()
        .map(|m| {
            let exists = c.data_path(&m.path).is_file();
            let name = m.effective_name();
            ModelOption {
                checked: exists && defaults.iter().any(|d| key(d) == key(&name)),
                name,
                family: m.family.to_string(),
                enabled: m.enabled,
                exists,
            }
        })
        .collect()
}

/// Every device option, runnable first (downloadable ones count as not runnable here: the
/// benchmark only measures what is installed).
fn device_rows(snap: &DevicesSnapshot, c: &Config) -> Vec<DeviceOptionRow> {
    let chosen: Vec<String> = c
        .benchmark
        .devices
        .iter()
        .filter_map(|d| crate::backend::spec::parse(d).ok().map(|s| s.to_string()))
        .collect();
    let (ok, bad): (Vec<_>, Vec<_>) = snap.selection.options.iter().partition(|o| o.runnable);
    ok.into_iter()
        .chain(bad)
        .map(|o| {
            let spec = o.spec.to_string();
            DeviceOptionRow {
                checked: o.runnable && (chosen.is_empty() || chosen.contains(&spec)),
                spec,
                label: o.label.clone(),
                runnable: o.runnable,
                reason: (!o.runnable).then(|| match &o.download {
                    Some(d) => format!("not installed ({})", d.summary),
                    None => o.reason.clone().unwrap_or_else(|| "not runnable".into()),
                }),
            }
        })
        .collect()
}

/// Download state of a built-in set from the manager (busy) or the disk.
fn dataset_rows(state: &AppState, c: &Config) -> Vec<DatasetRow> {
    let selected: Vec<String> = c.benchmark.datasets.iter().map(|d| d.label()).collect();
    let is_sel = |id: &str| selected.iter().any(|s| s.eq_ignore_ascii_case(id));
    let root = c.data_root();
    let rows_by_id: HashMap<String, crate::resources::status::ResourceRow> =
        crate::resources::status::rows(state.resources.as_ref(), c)
            .into_iter()
            .filter(|r| r.kind == "bench-images")
            .map(|r| (r.id.to_string(), r))
            .collect();
    let mut out = Vec::new();
    for m in images::builtin_manifests() {
        let (set, missing) = ImageSet::from_manifest(m, &images::builtin_dir(&root, &m.id));
        let rid = format!("{}{}", crate::resources::catalog::BENCH_ID_PREFIX, m.id);
        let busy = rows_by_id.get(&rid).filter(|r| r.busy);
        let (st, text) = match busy {
            Some(r) => ("downloading", r.state_text.clone()),
            None if missing == 0 => ("installed", "installed".to_string()),
            None if !set.images.is_empty() => (
                "partial",
                format!("{} of {} images", set.images.len(), m.images.len()),
            ),
            None => ("available", "not downloaded".to_string()),
        };
        out.push(DatasetRow {
            value: m.id.clone(),
            title: m.title.clone(),
            detail: format!(
                "{} images \u{b7} {} \u{b7} {} \u{b7} {}",
                m.images.len(),
                crate::resources::catalog::format_size(m.size()),
                if m.has_ground_truth() {
                    "ground truth"
                } else {
                    "no ground truth"
                },
                m.license
            ),
            state: st,
            state_text: text,
            checked: is_sel(&m.id),
            download_id: (missing > 0 && busy.is_none()).then_some(rid),
            folder: false,
            selectable: true,
        });
    }
    for id in images::KNOWN_SET_IDS {
        if images::builtin_manifest(id).is_none() {
            out.push(DatasetRow {
                value: id.to_string(),
                title: id.to_string(),
                detail: "curated set; not included in this build yet".into(),
                state: "unavailable",
                state_text: "not in this build".into(),
                checked: is_sel(id),
                download_id: None,
                folder: false,
                selectable: true,
            });
        }
    }
    out.push(DatasetRow {
        value: images::SAMPLE_ID.into(),
        title: "Embedded sample".into(),
        detail: "1 image (dog, bicycle, car) \u{b7} works offline \u{b7} no ground truth".into(),
        state: "embedded",
        state_text: "always available".into(),
        checked: is_sel(images::SAMPLE_ID),
        download_id: None,
        folder: false,
        selectable: true,
    });
    for d in &c.benchmark.datasets {
        if let Some((dir, _, _)) = d.dir() {
            let abs = c.data_path(&dir);
            let ok = abs.is_dir();
            out.push(DatasetRow {
                value: d.label(),
                title: d.label(),
                detail: abs.display().to_string(),
                state: if ok { "folder" } else { "missing" },
                state_text: if ok {
                    "folder on the server".into()
                } else {
                    "directory not found".into()
                },
                checked: true,
                download_id: None,
                folder: true,
                selectable: ok,
            });
        }
    }
    out
}

/// `GET /benchmark`.
pub(super) async fn page(State(state): State<Arc<AppState>>) -> Response {
    let snap = devices_snapshot(&state);
    let cfg = state.config_read().clone();
    let b = &cfg.benchmark;
    use crate::benchmark::grade::{ACCURACY_THRESHOLDS as A, SPEED_THRESHOLDS_MS as S};
    render(&BenchmarkTemplate {
        nav: "benchmark",
        version: crate::VERSION,
        models: model_options(&cfg),
        devices: device_rows(&snap, &cfg),
        datasets: dataset_rows(&state, &cfg),
        reference_models: cfg
            .models
            .iter()
            .filter(|m| cfg.data_path(&m.path).is_file())
            .map(|m| {
                let n = m.effective_name();
                let sel = b.reference_model.as_deref() == Some(n.as_str());
                (n, sel)
            })
            .collect(),
        repeat: b.repeat_per_image.max(1),
        warmup: b.warmup,
        max_images: b.max_images_per_dataset,
        accuracy_weight: format!("{:.2}", b.weights.accuracy_share()),
        warning: LIVE_WARNING,
        force_cpu: cfg.force_cpu,
        results_file: results_path(&state.config_path).display().to_string(),
        speed_thresholds: format!("A < {} ms, B < {}, C < {}, D < {}", S[0], S[1], S[2], S[3]),
        accuracy_thresholds: format!(
            "A \u{2265} {:.2}, B \u{2265} {:.2}, C \u{2265} {:.2}, D \u{2265} {:.2}",
            A[0], A[1], A[2], A[3]
        ),
    })
}

/// Normalized model name -> spec of the device it is loaded on (this generation).
fn loaded_devices(state: &AppState) -> HashMap<String, String> {
    state
        .registry
        .workers()
        .iter()
        .filter_map(|w| {
            let d = w.device.read().ok()?.clone()?;
            Some((crate::registry::normalize_name(&w.name), d.spec))
        })
        .collect()
}

/// The config's view for the page: global device, and per model its `device`, the device it
/// runs on (loaded, else what the config resolves to) and its execution provider.
fn config_view(state: &AppState, snap: &DevicesSnapshot) -> serde_json::Value {
    let cfg = state.config_read().clone();
    let loaded = loaded_devices(state);
    let models: Vec<serde_json::Value> = cfg
        .models
        .iter()
        .map(|m| {
            let name = m.effective_name();
            let key = crate::registry::normalize_name(&name);
            let effective = loaded
                .get(&key)
                .cloned()
                .filter(|_| m.enabled)
                .or_else(|| configured_device(&cfg, m, &snap.selection));
            let provider = m
                .enabled
                .then(|| state.registry.by_name(&name))
                .flatten()
                .map(|w| w.execution_provider());
            serde_json::json!({
                "name": name,
                "enabled": m.enabled,
                "device": m.device,
                "effective": effective,
                "provider": provider,
            })
        })
        .collect();
    serde_json::json!({
        "forceCpu": cfg.force_cpu,
        "globalDevice": cfg.device,
        "globalLabel": global_device_label(&cfg, snap),
        "models": models,
        "benchmark": cfg.benchmark,
    })
}

/// `GET /v1/benchmark`: the current/last run (state, progress, partial results), the saved
/// results (without per-image details), the ranking and the config's per-model devices.
pub(super) async fn status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let snap = devices_snapshot(&state);
    let path = results_path(&state.config_path);
    let results = BenchmarkResults::load_or_warn(&path);
    let ranking = results.as_ref().map(BenchmarkResults::ranking);
    let run = state.benchmark.status();
    Json(serde_json::json!({
        "success": true,
        "state": run.state,
        "running": run.running,
        "run": run,
        "results": results.as_ref().map(BenchmarkResults::summary),
        "ranking": ranking,
        "resultsFile": path.display().to_string(),
        "config": config_view(&state, &snap),
        "warning": LIVE_WARNING,
    }))
}

fn json_error(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({ "success": false, "message": message })),
    )
        .into_response()
}

/// Parse a start/settings form over the configured defaults and check the datasets.
fn parse_request(state: &AppState, form: &[(String, String)]) -> anyhow::Result<RunRequest> {
    let cfg = state.config_read().clone();
    let req = RunRequest::from_form(form, &cfg.benchmark)?;
    images::validate(&req.datasets, &cfg.data_root())?;
    Ok(req)
}

/// `POST /v1/benchmark` (urlencoded; every field optional, defaults from the `benchmark`
/// config): `model`, `device`, `dataset` (repeated), `max_images`, `repeat`, `warmup`,
/// `reference_model`, `accuracy_weight`. Starts a run in the background: 202, 400 (bad input)
/// or 409 (a run is in progress).
pub(super) async fn start(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    let req = match parse_request(&state, &form) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    if state.benchmark.is_running() {
        return json_error(
            StatusCode::CONFLICT,
            "a benchmark is already running; cancel it or wait for it to finish".into(),
        );
    }
    let ctx = RunContext {
        config: state.config_read().clone(),
        results_path: results_path(&state.config_path),
        runtimes: state.registry.runtimes().cloned(),
        loaded: loaded_devices(&state),
        resources: state.resources.clone(),
    };
    info!(
        models = ?req.models,
        devices = ?req.devices,
        datasets = ?req.datasets.iter().map(DatasetRef::label).collect::<Vec<_>>(),
        repeat = req.repeat,
        warmup = req.warmup,
        "benchmark started from the web UI"
    );
    match state.benchmark.start(ctx, req) {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "success": true,
                "message": "benchmark started; poll GET /v1/benchmark for progress",
            })),
        )
            .into_response(),
        Err(e) => json_error(StatusCode::CONFLICT, e.to_string()),
    }
}

/// `POST /v1/benchmark/settings`: the same fields as a start, saved as the `benchmark` config
/// defaults (plus `add_folder` = a directory to add as `dir:<path>`).
pub(super) async fn settings(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let mut form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    let folder: Option<String> = form
        .iter()
        .find(|(k, _)| k == "add_folder")
        .map(|(_, v)| v.trim().to_string())
        .filter(|v| !v.is_empty());
    if let Some(f) = &folder {
        let v = if f.starts_with(images::DIR_PREFIX) {
            f.clone()
        } else {
            format!("{}{f}", images::DIR_PREFIX)
        };
        if !form.iter().any(|(k, x)| k == "dataset" && *x == v) {
            form.push(("dataset".into(), v));
        }
    }
    let req = match parse_request(&state, &form) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    match save_config(&state, |c| {
        req.to_config(&mut c.benchmark);
        Ok(())
    }) {
        Ok(_) => Json(serde_json::json!({
            "success": true,
            "message": format!(
                "Benchmark defaults saved to {} (datasets: {}).",
                state.config_path.display(),
                req.datasets.iter().map(DatasetRef::label).collect::<Vec<_>>().join(", ")
            ),
        }))
        .into_response(),
        Err(e) => json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

/// `POST /v1/benchmark/cancel`: 200 when a run was asked to stop, 409 when none runs.
pub(super) async fn cancel(State(state): State<Arc<AppState>>) -> Response {
    if state.benchmark.cancel() {
        info!("benchmark cancel requested from the web UI");
        Json(serde_json::json!({ "success": true, "message": "cancelling the benchmark" }))
            .into_response()
    } else {
        json_error(StatusCode::CONFLICT, "no benchmark is running".into())
    }
}

/// `GET /v1/benchmark/images?model=&device=`: the images of a model's run on a device with
/// their ground truth and predictions (TP/FP flags, matched/missed objects).
pub(super) async fn images_detail(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    let q = query.map(|Query(q)| q).unwrap_or_default();
    let (Some(model), Some(device)) = (q.get("model"), q.get("device")) else {
        return json_error(StatusCode::BAD_REQUEST, "give `model` and `device`".into());
    };
    let Some(results) = BenchmarkResults::load_or_warn(&results_path(&state.config_path)) else {
        return json_error(StatusCode::NOT_FOUND, "no benchmark results yet".into());
    };
    let Some(m) = results.find(model) else {
        return json_error(StatusCode::NOT_FOUND, format!("no results for '{model}'"));
    };
    let Some(run) = m
        .devices
        .iter()
        .find(|d| d.device == *device)
        .and_then(|d| d.run.as_ref())
    else {
        return json_error(
            StatusCode::NOT_FOUND,
            format!("'{model}' did not run on {device}"),
        );
    };
    let images: Vec<serde_json::Value> = run
        .per_image
        .iter()
        .map(|r| {
            let info = results
                .image_set(&r.set)
                .and_then(|s| s.images.iter().find(|i| i.file == r.file));
            serde_json::json!({
                "set": r.set,
                "file": r.file,
                "width": info.map(|i| i.width),
                "height": info.map(|i| i.height),
                "tags": info.map(|i| i.tags.clone()).unwrap_or_default(),
                "objects": info.and_then(|i| i.objects.clone()),
                "gtMatched": r.gt_matched,
                "preds": r.preds,
                "counts": r.counts,
                "totalMs": r.total_ms,
                "inferMs": r.infer_ms,
            })
        })
        .collect();
    Json(serde_json::json!({
        "success": true,
        "model": m.model,
        "device": device,
        "groundTruth": run.accuracy.as_ref().map(|a| a.ground_truth.clone()),
        "images": images,
    }))
    .into_response()
}

/// `GET /v1/benchmark/image?set=&file=`: an image of the saved results (the embedded sample, or
/// a file listed in the results under its dataset's directory; nothing else is served).
pub(super) async fn image_file(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    let q = query.map(|Query(q)| q).unwrap_or_default();
    let (Some(set), Some(file)) = (q.get("set"), q.get("file")) else {
        return json_error(StatusCode::BAD_REQUEST, "give `set` and `file`".into());
    };
    let ctype = |f: &str| {
        if f.to_ascii_lowercase().ends_with(".png") {
            "image/png"
        } else {
            "image/jpeg"
        }
    };
    if set == images::SAMPLE_ID {
        return (
            [(header::CONTENT_TYPE, "image/jpeg")],
            crate::benchmark::DEFAULT_IMAGE,
        )
            .into_response();
    }
    let path = BenchmarkResults::load_or_warn(&results_path(&state.config_path))
        .and_then(|r| r.image_set(set).and_then(|s| s.image_path(file)));
    let Some(path) = path else {
        return json_error(
            StatusCode::NOT_FOUND,
            "not an image of the saved results".into(),
        );
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => (
            [
                (header::CONTENT_TYPE, ctype(file)),
                (header::CACHE_CONTROL, "private, max-age=600"),
            ],
            bytes,
        )
            .into_response(),
        Err(e) => json_error(StatusCode::NOT_FOUND, format!("{}: {e}", path.display())),
    }
}

fn truthy(v: &str) -> bool {
    !matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "off" | "no"
    )
}

/// `POST /v1/benchmark/apply`: set per-model devices in the config file.
/// - `model=<name>&device=<spec>`: one model ("" or `global` = follow the global device);
/// - `all=1` (optionally limited by repeated `model`): every recommendation of the saved results.
///
/// `restart=1` restarts the server afterwards (like `POST /config/restart`). Answers JSON with
/// the changes; 400 on bad input, 404 when `all` finds no recommendation.
pub(super) async fn apply(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    let field = |k: &str| {
        form.iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v.as_str())
    };
    let models: Vec<String> = form
        .iter()
        .filter(|(k, _)| k == "model")
        .map(|(_, v)| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect();
    let restart = field("restart").is_some_and(truthy);
    let picks: Vec<(String, Option<String>)> = if field("all").is_some_and(truthy) {
        let path = results_path(&state.config_path);
        let results = match BenchmarkResults::load(&path) {
            Ok(Some(r)) => r,
            Ok(None) => {
                return json_error(
                    StatusCode::NOT_FOUND,
                    "no benchmark results yet; run a benchmark first".into(),
                );
            }
            Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        };
        let only = (!models.is_empty()).then_some(models.as_slice());
        let recs = results.recommendations(only);
        if recs.is_empty() {
            return json_error(
                StatusCode::NOT_FOUND,
                "the saved results recommend no device for these models".into(),
            );
        }
        // Results may name models that were removed from the config since: skip those.
        let cfg = state.config_read().clone();
        recs.into_iter()
            .filter(|(m, _)| crate::benchmark::find_model(&cfg, m).is_some())
            .map(|(m, d)| (m, Some(d)))
            .collect()
    } else {
        match (models.as_slice(), field("device")) {
            ([model], Some(device)) => vec![(model.clone(), Some(device.to_string()))],
            _ => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "give `model` and `device`, or `all=1` to apply every recommendation".into(),
                );
            }
        }
    };
    let mut changes: Vec<DeviceChange> = Vec::new();
    let saved = save_config(&state, |c| {
        changes = apply_model_devices(c, &picks)?;
        Ok(())
    });
    if let Err(e) = saved {
        warn!("benchmark apply refused: {e:#}");
        return json_error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    let force_cpu = state.config_read().force_cpu;
    let mut message = if changes.is_empty() {
        "No change: the models already use these devices.".to_string()
    } else {
        format!(
            "Saved {}: {}.",
            state.config_path.display(),
            changes
                .iter()
                .map(DeviceChange::describe)
                .collect::<Vec<_>>()
                .join("; ")
        )
    };
    if force_cpu {
        message.push_str(" Note: Force CPU is on and overrides per-model devices.");
    }
    let restarting = restart && !changes.is_empty();
    if restarting {
        message.push_str(" Restarting the server to load the models on their new devices.");
        schedule_restart(&state);
    } else if !changes.is_empty() {
        message.push_str(" Restart the server to apply.");
    }
    info!("benchmark apply: {message}");
    Json(serde_json::json!({
        "success": true,
        "message": message,
        "changes": changes,
        "restarting": restarting,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::super::router;
    use super::*;
    use crate::config::ModelConfig;
    use crate::metrics::{Metrics, ModelMetrics};
    use crate::registry::{ModelRegistry, RuntimeInfo};
    use crate::worker::WorkerHandle;
    use axum::body::Body;
    use axum::http::{Request, header};
    use std::time::Duration;
    use tower::ServiceExt;

    fn state() -> Arc<AppState> {
        let dir = std::env::temp_dir().join(format!("bo_bench_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let metrics = Metrics::new("0.0.0");
        let mm = Arc::new(ModelMetrics::new("ipcam-general", "CPU", ""));
        let workers = vec![WorkerHandle::failed(
            "ipcam-general",
            "model file not found",
            mm,
        )];
        let info = RuntimeInfo {
            openvino_version: "2026.0.0-test".into(),
            available_devices: vec!["CPU".into()],
            has_gpu: false,
        };
        let registry = ModelRegistry::from_handles(workers, Some(0), info);
        let config = Config {
            models: ["ipcam-general", "dfine-s"]
                .iter()
                .map(|n| ModelConfig {
                    name: Some(n.to_string()),
                    path: dir.join(format!("{n}.onnx")),
                    enabled: *n == "ipcam-general",
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        Arc::new(AppState::new(
            Arc::new(registry),
            Arc::new(metrics),
            config,
            dir.join("cfg.json"),
        ))
    }

    async fn call(state: &Arc<AppState>, req: Request<Body>) -> (StatusCode, serde_json::Value) {
        let resp = router(state.clone()).oneshot(req).await.unwrap();
        let status = resp.status();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v = serde_json::from_slice(&body)
            .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&body).into()));
        (status, v)
    }

    fn post(uri: &str, body: &str) -> Request<Body> {
        Request::post(uri)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn get(uri: &str) -> Request<Body> {
        Request::get(uri).body(Body::empty()).unwrap()
    }

    #[tokio::test]
    async fn page_lists_models_and_devices() {
        let st = state();
        let resp = router(st.clone()).oneshot(get("/benchmark")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(html.contains("<a href=\"/benchmark\" class=\"active\">Benchmark</a>"));
        assert!(
            html.contains("name=\"model\" value=\"ipcam-general\""),
            "{html}"
        );
        // Files are missing: listed but not checked.
        assert!(!html.contains("value=\"ipcam-general\" checked"));
        assert!(html.contains("name=\"device\" value=\"openvino:cpu\" checked"));
        // ONNX Runtime is not in this probe: listed, disabled.
        assert!(html.contains("name=\"device\" value=\"ort:cpu\" disabled"));
        // The configured datasets are checked; built-in sets offer a download.
        assert!(
            html.contains("name=\"dataset\" value=\"coco-cctv\" checked"),
            "{html}"
        );
        assert!(html.contains("data-download=\"bench:coco-cctv\""));
        assert!(html.contains("name=\"dataset\" value=\"sample\">"));
        assert!(html.contains("competes") || html.contains("same CPU/GPU"));
        assert!(html.contains("fetch(\"/v1/benchmark\""));
    }

    #[tokio::test]
    async fn start_status_busy_and_cancel() {
        let st = state();
        let (s, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["state"], "idle");
        assert_eq!(v["running"], false);
        assert!(v["results"].is_null());
        assert_eq!(v["config"]["models"][0]["name"], "ipcam-general");
        assert_eq!(v["config"]["models"][1]["enabled"], false);

        // Bad input.
        let (s, v) = call(&st, post("/v1/benchmark", "repeat=0")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(v["message"].as_str().unwrap().contains("repeat"));
        let (s, v) = call(&st, post("/v1/benchmark", "dataset=nope")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(v["message"].as_str().unwrap().contains("valid ids"), "{v}");
        let (s, _) = call(
            &st,
            post("/v1/benchmark", "model=ipcam-general&device=vulkan"),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = call(&st, post("/v1/benchmark/cancel", "")).await;
        assert_eq!(s, StatusCode::CONFLICT);

        // A held run: a second start is refused with 409, cancel ends it.
        let mut req = RunRequest::defaults(&crate::config::BenchmarkConfig::default());
        req.models = vec!["ipcam-general".into()];
        st.benchmark
            .start_job(req, |h| {
                while !h.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(crate::benchmark::Cancelled.into())
            })
            .unwrap();
        let (s, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(
            (s, v["running"].clone()),
            (StatusCode::OK, serde_json::json!(true))
        );
        let (s, v) = call(&st, post("/v1/benchmark", "model=ipcam-general")).await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        let (s, _) = call(&st, post("/v1/benchmark/cancel", "")).await;
        assert_eq!(s, StatusCode::OK);
        assert!(st.benchmark.wait_idle(Duration::from_secs(5)));
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(v["state"], "cancelled");

        // A real start: the model file is missing, so the run ends quickly without inference
        // and records why (or fails when no runtime can be initialized in this environment).
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark",
                "model=ipcam-general&device=openvino:cpu&repeat=1&warmup=0",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        assert!(st.benchmark.wait_idle(Duration::from_secs(60)));
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        match v["state"].as_str().unwrap() {
            "done" => {
                let err = v["run"]["models"][0]["error"].as_str().unwrap();
                assert!(err.contains("not found"), "{v}");
                // Saved, so the page shows it after a reload.
                assert_eq!(v["results"]["models"][0]["model"], "ipcam-general");
            }
            "failed" => assert!(v["run"]["error"].is_string(), "{v}"),
            other => panic!("unexpected state {other}: {v}"),
        }
    }

    #[tokio::test]
    async fn apply_one_and_all() {
        let st = state();
        let cfg_path = st.config_path.clone();

        // `all` without results: 404.
        let (s, _) = call(&st, post("/v1/benchmark/apply", "all=1")).await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        // One model.
        let (s, v) = call(
            &st,
            post("/v1/benchmark/apply", "model=DFINE-S&device=ort%3Acoreml"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"][0]["model"], "dfine-s");
        assert_eq!(v["changes"][0]["to"], "ort:coreml");
        assert_eq!(v["restarting"], false);
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("Restart the server")
        );
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[1].device.as_deref(), Some("ort:coreml"));
        assert_eq!(*st.config_read(), saved);

        // Invalid input changes nothing.
        for body in [
            "model=dfine-s&device=vulkan",
            "model=zzz&device=ort%3Acpu",
            "model=dfine-s",
            "",
        ] {
            let (s, _) = call(&st, post("/v1/benchmark/apply", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(Config::load(&cfg_path).unwrap(), saved, "{body}");
        }

        // Every recommendation of the saved results; models no longer configured are skipped.
        let mk = |name: &str, rec: Option<&str>| {
            let mut m = crate::benchmark::ModelResult::failed(name, "", String::new());
            m.error = None;
            m.recommended = rec.map(str::to_string);
            m
        };
        let results = BenchmarkResults::new(
            Default::default(),
            Default::default(),
            vec![
                mk("ipcam-general", Some("openvino:cpu")),
                mk("dfine-s", Some("ort:coreml")),
                mk("removed", Some("ort:cpu")),
                mk("none", None),
            ],
        );
        results.save(&results_path(&cfg_path)).unwrap();
        let (s, v) = call(&st, post("/v1/benchmark/apply", "all=1&restart=1")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        // dfine-s already had ort:coreml: one change.
        assert_eq!(v["changes"].as_array().unwrap().len(), 1, "{v}");
        assert_eq!(v["changes"][0]["model"], "ipcam-general");
        assert_eq!(v["restarting"], true);
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[0].device.as_deref(), Some("openvino:cpu"));
        tokio::time::timeout(Duration::from_secs(5), st.restart.cancelled())
            .await
            .expect("restart requested");

        // The config page shows the per-model Device select and the recommendation.
        let resp = router(st.clone()).oneshot(get("/config")).await.unwrap();
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(html.contains("name=\"device.ipcam-general\""), "{html}");
        assert!(html.contains("<option value=\"\">Global (auto"), "{html}");
        assert!(html.contains("benchmark: openvino:cpu (in use)"), "{html}");
        assert!(html.contains("benchmark: ort:coreml (in use)"), "{html}");
    }

    #[tokio::test]
    async fn models_card_saves_per_model_devices() {
        let st = state();
        let body = "enabled=ipcam-general&default_model=ipcam-general&device.ipcam-general=ort%3Acpu&device.dfine-s=&action=save";
        let resp = router(st.clone())
            .oneshot(post("/config/models", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let saved = Config::load(&st.config_path).unwrap();
        assert_eq!(saved.models[0].device.as_deref(), Some("ort:cpu"));
        assert_eq!(saved.models[1].device, None);
        // The select shows the saved value; "" = global.
        let resp = router(st.clone()).oneshot(get("/config")).await.unwrap();
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(
            html.contains("<option value=\"ort:cpu\" selected"),
            "{html}"
        );
        // A bad device is rejected and nothing changes.
        let body = "enabled=ipcam-general&device.ipcam-general=vulkan&action=save";
        let resp = router(st.clone())
            .oneshot(post("/config/models", body))
            .await
            .unwrap();
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(html.contains("Models not saved"), "{html}");
        assert_eq!(Config::load(&st.config_path).unwrap(), saved);
    }
}
