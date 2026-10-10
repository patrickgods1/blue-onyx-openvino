//! axum HTTP server: CodeProject.AI compatible detection endpoints plus the web UI
//! (welcome, stats, test, config pages rendered with askama templates from `templates/`).
//! Handler flow and page set modeled on blue-onyx `server.rs` and its templates (MIT).

use crate::api::{VisionCustomListResponse, VisionDetectionRequest, VisionDetectionResponse};
use crate::backend::detect::HardwareInfo;
use crate::backend::select::{OpenVinoProbe, OrtProbe, RuntimeProbe, Selection, select};
use crate::backend::spec::{self, DeviceSpec};
use crate::cli::LogReloadHandle;
use crate::config::{Config, FORM_FIELDS, LogLevel, apply_config_form, apply_models_selection};
use crate::metrics::{Metrics, Stat};
use crate::registry::ModelRegistry;
use crate::setup_onnxruntime::InstalledOrt;
use crate::startup::ModelState;
use crate::worker::WorkerHandle;
use askama::Template;
use axum::extract::multipart::MultipartRejection;
use axum::extract::rejection::{FormRejection, QueryRejection};
use axum::extract::{DefaultBodyLimit, Form, Multipart, Path, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use crossbeam_channel::TrySendError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

/// Maximum accepted request body (multipart image upload).
pub const BODY_LIMIT: usize = 32 * 1024 * 1024;
/// How long a handler waits for the worker's reply.
pub const RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
/// Delay between answering `POST /config/restart` and stopping the server, so the reply gets out.
const RESTART_DELAY: Duration = Duration::from_millis(250);

static STYLE_CSS: &str = include_str!("../assets/style.css");
static FAVICON_ICO: &[u8] = include_bytes!("../assets/favicon.ico");

pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    pub metrics: Arc<Metrics>,
    /// In-memory copy of the config file; `/config` edits it and writes it to `config_path`.
    pub config: Arc<RwLock<Config>>,
    /// Process start (survives in-process restarts).
    pub started: Instant,
    pub config_path: PathBuf,
    /// Runtime log level control; `None` when logging was not initialized by this process.
    pub log_reload: Option<LogReloadHandle>,
    /// Cancelled by `POST /config/restart`. [`serve`] stops when it fires; the main binary
    /// uses a child of the shutdown token so it also stops the workers of this generation.
    pub restart: CancellationToken,
}

impl AppState {
    /// State without a log reload handle and with a fresh restart token.
    pub fn new(
        registry: Arc<ModelRegistry>,
        metrics: Arc<Metrics>,
        config: Config,
        config_path: PathBuf,
    ) -> Self {
        Self {
            registry,
            metrics,
            config: Arc::new(RwLock::new(config)),
            started: Instant::now(),
            config_path,
            log_reload: None,
            restart: CancellationToken::new(),
        }
    }

    fn config_read(&self) -> RwLockReadGuard<'_, Config> {
        self.config.read().unwrap_or_else(|e| e.into_inner())
    }

    fn config_write(&self) -> RwLockWriteGuard<'_, Config> {
        self.config.write().unwrap_or_else(|e| e.into_inner())
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(welcome))
        .route("/v1/vision/detection", post(detection_default))
        .route("/v1/vision/custom/list", get(custom_list).post(custom_list))
        .route("/v1/vision/custom/{model}", post(detection_custom))
        .route("/v1/status/updateavailable", get(update_available))
        .route("/v1/devices", get(devices_json))
        .route("/stats", get(stats_page))
        .route("/stats.json", get(stats_json))
        .route("/prometheus", get(prometheus))
        .route("/test", get(test_page).post(test_submit))
        .route("/config", get(config_page).post(config_submit))
        .route("/config/models", post(config_models))
        .route("/config/restart", post(config_restart))
        .route("/config/loglevel", post(config_loglevel))
        .route("/static/style.css", get(style_css))
        .route("/favicon.ico", get(favicon))
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(state)
}

/// Bind `0.0.0.0:port` and serve until `shutdown` or `state.restart` is cancelled.
pub async fn serve(
    state: Arc<AppState>,
    port: u16,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    // A few retries cover a just-closed listener from the previous generation on restart.
    let mut attempt = 0;
    let listener = loop {
        match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => break l,
            Err(e) if attempt < 5 && e.kind() == std::io::ErrorKind::AddrInUse => {
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(e) => {
                anyhow::bail!("binding {addr}: {e} (is another instance running on port {port}?)")
            }
        }
    };
    info!("listening on http://{addr}");
    let restart = state.restart.clone();
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = shutdown.cancelled() => {}
                _ = restart.cancelled() => {}
            }
        })
        .await?;
    info!("HTTP server stopped");
    Ok(())
}

/// State text for the HTML pages and `/stats`: "Initializing (downloading ...)", "Ready", ...;
/// a lazy model that has not been requested yet says so.
fn state_label(w: &WorkerHandle) -> String {
    let s = w.state.get();
    match (&s, w.lazy) {
        (ModelState::Initializing, true) if w.accepts_while_initializing() => {
            "Lazy (loads on first request)".to_string()
        }
        _ => w.state.describe(),
    }
}

/// `1d 2h 3m 4s`, leading zero units omitted.
fn format_uptime(d: Duration) -> String {
    let s = d.as_secs();
    let (days, h, m, sec) = (s / 86_400, s / 3600 % 24, s / 60 % 60, s % 60);
    match (days, h, m) {
        (0, 0, 0) => format!("{sec}s"),
        (0, 0, _) => format!("{m}m {sec}s"),
        (0, _, _) => format!("{h}h {m}m {sec}s"),
        _ => format!("{days}d {h}h {m}m {sec}s"),
    }
}

fn wants_html(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("text/html"))
}

fn render<T: Template>(t: &T) -> Response {
    match t.render() {
        Ok(html) => Html(html).into_response(),
        Err(e) => {
            error!("template rendering failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("template error: {e}"),
            )
                .into_response()
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Welcome page

struct ModelRow {
    name: String,
    is_default: bool,
    state: String,
    provider: String,
    requests: u64,
    queue: String,
}

#[derive(Template)]
#[template(path = "welcome.html")]
struct WelcomeTemplate {
    nav: &'static str,
    version: &'static str,
    openvino_version: String,
    onnxruntime: String,
    devices: String,
    gpus: Vec<String>,
    auto_pick: String,
    uptime: String,
    port: u16,
    default_model: String,
    models: Vec<ModelRow>,
}

/// Hardware, runtimes and device options as seen by this process.
struct DevicesSnapshot {
    hardware: HardwareInfo,
    selection: Selection,
    openvino_version: String,
    openvino_devices: Vec<String>,
    onnxruntime: Option<InstalledOrt>,
}

impl DevicesSnapshot {
    /// Display text of the device `auto` resolves to.
    fn auto_pick_label(&self) -> String {
        match self.selection.auto_pick() {
            Some(o) => format!("{} ({})", o.spec, o.label),
            None => "nothing runnable".to_string(),
        }
    }

    fn onnxruntime_label(&self) -> String {
        match &self.onnxruntime {
            Some(o) => format!("{} ({})", o.version, o.flavor),
            None => "not installed (run setup-onnxruntime)".to_string(),
        }
    }
}

/// Snapshot of the process's devices. Uses the shared runtimes when they are free (workers hold
/// the lock while compiling a model; waiting would stall the page), else rebuilds the options from
/// the OpenVINO facts the registry captured at startup.
fn devices_snapshot(state: &AppState) -> DevicesSnapshot {
    let info = &state.registry.runtime_info;
    let live = state.registry.runtimes().and_then(|rt| {
        let rt = rt.try_lock().ok()?;
        Some((
            rt.hardware().clone(),
            rt.selection(None),
            rt.probe().clone(),
        ))
    });
    let probe = live.as_ref().map(|(_, _, p)| p.clone());
    let live = live.map(|(h, s, _)| (h, s));
    let (hardware, selection) = live.unwrap_or_else(|| {
        let hardware = crate::backend::detect::hardware().clone();
        let probe = if info.openvino_version.is_empty() && info.available_devices.is_empty() {
            RuntimeProbe {
                openvino: OpenVinoProbe::unavailable(crate::backend::OV_UNAVAILABLE),
                ort: OrtProbe::not_in_build(),
            }
        } else {
            RuntimeProbe::openvino_only(&info.available_devices)
        };
        let selection = select(&hardware, &probe, true);
        (hardware, selection)
    });
    // Options a download would make runnable ("downloadable").
    let mut selection = selection;
    {
        let cfg = state.config_read();
        let root = crate::resources::download_root(&cfg);
        let installed = crate::resources::detect_installed(&cfg, &root, probe);
        crate::resources::resolve::annotate_selection(
            &mut selection,
            &cfg,
            &hardware,
            &installed,
            true,
        );
    }
    // The install the ONNX Runtime loader picks (explicit dir, ORT_DYLIB_PATH, active flavor,
    // legacy flat dir, ...).
    let explicit = state.config_read().onnxruntime_dir.clone();
    let onnxruntime = crate::backend::libs::find_onnxruntime(explicit.as_deref())
        .library
        .and_then(|lib| lib.parent().map(std::path::Path::to_path_buf))
        .and_then(|dir| crate::setup_onnxruntime::installed(&dir));
    DevicesSnapshot {
        hardware,
        selection,
        openvino_version: info.openvino_version.clone(),
        openvino_devices: info.available_devices.clone(),
        onnxruntime,
    }
}

/// `GET /v1/devices`: detected hardware, runtimes and every device option with `auto`'s pick.
async fn devices_json(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let snap = devices_snapshot(&state);
    Json(serde_json::json!({
        "success": true,
        "hardware": snap.hardware,
        "runtimes": {
            "openvino": {
                "version": snap.openvino_version,
                "devices": snap.openvino_devices,
            },
            "onnxruntime": snap.onnxruntime.as_ref().map(|o| serde_json::json!({
                "version": o.version,
                "flavor": o.flavor,
            })),
        },
        "selection": snap.selection,
        "autoPick": snap.selection.auto_pick().map(|o| o.spec.to_string()),
    }))
}

fn default_name(reg: &ModelRegistry) -> String {
    reg.default_model()
        .map(|w| w.name.clone())
        .unwrap_or_default()
}

async fn welcome(State(state): State<Arc<AppState>>) -> Response {
    let reg = &state.registry;
    let snap = devices_snapshot(&state);
    let default = default_name(reg);
    let models = reg
        .workers()
        .iter()
        .map(|w| ModelRow {
            name: w.name.clone(),
            is_default: w.name == default,
            state: state_label(w),
            provider: w.execution_provider(),
            requests: w.metrics.requests.load(Ordering::Relaxed),
            queue: format!("{}/{}", w.sender.len(), w.queue_capacity()),
        })
        .collect();
    render(&WelcomeTemplate {
        nav: "home",
        version: crate::VERSION,
        openvino_version: if reg.runtime_info.openvino_version.is_empty() {
            "not available (run setup-openvino)".to_string()
        } else {
            reg.runtime_info.openvino_version.clone()
        },
        onnxruntime: snap.onnxruntime_label(),
        devices: if reg.runtime_info.available_devices.is_empty() {
            "none".to_string()
        } else {
            reg.runtime_info.available_devices.join(", ")
        },
        gpus: snap.hardware.gpus.iter().map(|g| g.to_string()).collect(),
        auto_pick: snap.auto_pick_label(),
        uptime: format_uptime(state.started.elapsed()),
        port: state.config_read().port,
        default_model: default,
        models,
    })
}

// ---------------------------------------------------------------------------------------------
// Detection endpoints

/// Parse the multipart body: `image` (required), `min_confidence` (parse errors become 0.0)
/// and any other text fields, returned in the map (e.g. `model` on the test page).
async fn read_multipart(
    mut multipart: Multipart,
) -> Result<(VisionDetectionRequest, HashMap<String, String>), String> {
    let mut req = VisionDetectionRequest::default();
    let mut fields = HashMap::new();
    let mut got_image = false;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return Err(format!("Invalid multipart body: {e}")),
        };
        match field.name().map(str::to_string) {
            Some(name) if name == "min_confidence" => {
                let text = field.text().await.unwrap_or_default();
                req.min_confidence = text.trim().parse::<f32>().unwrap_or(0.0);
                fields.insert(name, text.trim().to_string());
            }
            Some(name) if name == "image" => {
                if let Some(n) = field.file_name() {
                    req.image_name = n.to_string();
                }
                req.image_data = field
                    .bytes()
                    .await
                    .map_err(|e| format!("Reading image field: {e}"))?;
                got_image = true;
            }
            Some(name) => {
                if let Ok(text) = field.text().await {
                    fields.insert(name, text);
                }
            }
            None => {}
        }
    }
    if !got_image || req.image_data.is_empty() {
        return Err("No image provided (expected multipart field 'image')".into());
    }
    Ok((req, fields))
}

async fn parse_detection_body(
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<(VisionDetectionRequest, HashMap<String, String>), String> {
    match multipart {
        Ok(mp) => read_multipart(mp).await,
        Err(e) => Err(format!("Expected a multipart/form-data body: {e}")),
    }
}

async fn detection_default(
    State(state): State<Arc<AppState>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Json<VisionDetectionResponse> {
    let start = Instant::now();
    let Some(handle) = state.registry.default_model() else {
        return Json(VisionDetectionResponse::error(
            "No default model: enable a model on the Config page",
        ));
    };
    let req = parse_detection_body(multipart).await.map(|(r, _)| r);
    Json(run_detection(&state, handle, req, start, "detect").await)
}

async fn detection_custom(
    State(state): State<Arc<AppState>>,
    Path(model): Path<String>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Json<VisionDetectionResponse> {
    let start = Instant::now();
    let Some(handle) = state.registry.by_name(&model) else {
        warn!("request for unknown model '{model}'");
        let mut r = VisionDetectionResponse::error(format!("Unknown model '{model}'"));
        r.command = "custom".into();
        return Json(r);
    };
    let req = parse_detection_body(multipart).await.map(|(r, _)| r);
    Json(run_detection(&state, handle, req, start, "custom").await)
}

async fn run_detection(
    state: &AppState,
    handle: &WorkerHandle,
    req: Result<VisionDetectionRequest, String>,
    start: Instant,
    command: &str,
) -> VisionDetectionResponse {
    let mut resp = dispatch(handle, req, start).await;
    resp.command = command.to_string();
    resp.analysisRoundTripMs = start.elapsed().as_millis().min(i32::MAX as u128) as i32;
    if resp.success {
        handle
            .metrics
            .round_trip_ms
            .record(start.elapsed().as_millis() as u64);
    }
    resp.canUseGPU = state.registry.runtime_info.has_gpu;
    resp
}

async fn dispatch(
    handle: &WorkerHandle,
    req: Result<VisionDetectionRequest, String>,
    start: Instant,
) -> VisionDetectionResponse {
    let name = &handle.name;
    let req = match req {
        Ok(r) => r,
        Err(e) => return VisionDetectionResponse::error(e),
    };
    handle.metrics.requests.fetch_add(1, Ordering::Relaxed);

    match handle.state.get() {
        ModelState::Ready => {}
        ModelState::Initializing if handle.accepts_while_initializing() => {}
        ModelState::Initializing => {
            return VisionDetectionResponse::error(match handle.state.detail() {
                Some(d) => format!("Model '{name}' is initializing ({d})"),
                None => format!("Model '{name}' is initializing"),
            });
        }
        ModelState::Failed(m) => {
            return VisionDetectionResponse::error(format!("Model '{name}' failed to load: {m}"));
        }
    }

    if handle.is_full() {
        warn!(model = %name, "worker queue is full, server is overloaded; rejecting request");
        handle.metrics.dropped.fetch_add(1, Ordering::Relaxed);
        return VisionDetectionResponse::error("Worker queue is full");
    }

    let (tx, rx) = tokio::sync::oneshot::channel();
    match handle.sender.try_send((req, tx, start)) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            handle.metrics.dropped.fetch_add(1, Ordering::Relaxed);
            return VisionDetectionResponse::error("Worker queue is full");
        }
        Err(TrySendError::Disconnected(_)) => {
            handle.metrics.dropped.fetch_add(1, Ordering::Relaxed);
            return VisionDetectionResponse::error(format!("Model '{name}' worker is not running"));
        }
    }

    match tokio::time::timeout(RESPONSE_TIMEOUT, rx).await {
        Ok(Ok(r)) => r,
        Ok(Err(_)) => {
            warn!(model = %name, "worker dropped the request without replying");
            handle.metrics.dropped.fetch_add(1, Ordering::Relaxed);
            VisionDetectionResponse::error("Worker dropped the request")
        }
        Err(_) => {
            warn!(model = %name, "timed out waiting for the worker");
            handle.metrics.dropped.fetch_add(1, Ordering::Relaxed);
            VisionDetectionResponse::error("Operation timed out")
        }
    }
}

async fn custom_list(State(state): State<Arc<AppState>>) -> Json<VisionCustomListResponse> {
    let reg = &state.registry;
    Json(VisionCustomListResponse {
        success: true,
        models: reg.names(),
        moduleId: crate::MODULE_ID.to_string(),
        moduleName: crate::MODULE_NAME.to_string(),
        command: "list".to_string(),
        statusData: None,
        inferenceDevice: reg
            .default_model()
            .map(|w| w.execution_provider())
            .unwrap_or_default(),
        analysisRoundTripMs: 0,
        processedBy: crate::PROCESSED_BY.to_string(),
        timestampUTC: chrono::Utc::now().to_rfc3339(),
    })
}

async fn update_available() -> impl IntoResponse {
    Json(crate::update::check_update(crate::VERSION).await)
}

// ---------------------------------------------------------------------------------------------
// Stats

struct TimingCells {
    avg: String,
    min: String,
    max: String,
}

impl TimingCells {
    fn from_stat(s: &Stat) -> Self {
        if s.count() == 0 {
            let dash = || "-".to_string();
            return Self {
                avg: dash(),
                min: dash(),
                max: dash(),
            };
        }
        Self {
            avg: format!("{:.1}", s.avg()),
            min: s.min().to_string(),
            max: s.max().to_string(),
        }
    }
}

struct StatsRow {
    name: String,
    is_default: bool,
    state: String,
    device: String,
    provider: String,
    fell_back: &'static str,
    requests: u64,
    dropped: u64,
    queue: String,
    /// Inference, process, round trip.
    timings: [TimingCells; 3],
}

#[derive(Template)]
#[template(path = "stats.html")]
struct StatsTemplate {
    nav: &'static str,
    version: &'static str,
    openvino_version: String,
    uptime: String,
    models: Vec<StatsRow>,
}

async fn stats_page(State(state): State<Arc<AppState>>) -> Response {
    let reg = &state.registry;
    let default = default_name(reg);
    let models = reg
        .workers()
        .iter()
        .map(|w| {
            let m = &w.metrics;
            let dev = w.device.read().ok().and_then(|d| d.clone());
            StatsRow {
                name: w.name.clone(),
                is_default: w.name == default,
                state: state_label(w),
                device: dev
                    .as_ref()
                    .map(|d| d.actual.clone())
                    .unwrap_or_else(|| m.requested_device.clone()),
                provider: w.execution_provider(),
                fell_back: match dev.as_ref().map(|d| d.fell_back) {
                    Some(true) => "yes",
                    Some(false) => "no",
                    None => "-",
                },
                requests: m.requests.load(Ordering::Relaxed),
                dropped: m.dropped.load(Ordering::Relaxed),
                queue: format!("{}/{}", w.sender.len(), w.queue_capacity()),
                timings: [
                    TimingCells::from_stat(&m.inference_ms),
                    TimingCells::from_stat(&m.process_ms),
                    TimingCells::from_stat(&m.round_trip_ms),
                ],
            }
        })
        .collect();
    render(&StatsTemplate {
        nav: "stats",
        version: crate::VERSION,
        openvino_version: reg.runtime_info.openvino_version.clone(),
        uptime: format_uptime(state.started.elapsed()),
        models,
    })
}

async fn stats_json(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let reg = &state.registry;
    let default = reg.default_model().map(|w| w.name.clone());
    let models: Vec<serde_json::Value> = reg
        .workers()
        .iter()
        .map(|w| {
            let m = &w.metrics;
            let dev = w.device.read().ok().and_then(|d| d.clone());
            let stat = |s: &Stat| {
                serde_json::json!({ "count": s.count(), "avg": s.avg(), "min": s.min(), "max": s.max() })
            };
            serde_json::json!({
                "name": w.name,
                "default": default.as_deref() == Some(w.name.as_str()),
                "state": w.state.describe(),
                "stateDetail": w.state.detail(),
                "lazy": w.lazy,
                "requestedDevice": m.requested_device,
                "device": dev.as_ref().map(|d| d.actual.clone()),
                "deviceName": dev.as_ref().map(|d| d.full_name.clone()),
                "fellBack": dev.as_ref().map(|d| d.fell_back),
                "executionProvider": w.execution_provider(),
                "queueLength": w.sender.len(),
                "queueCapacity": w.queue_capacity(),
                "requests": m.requests.load(Ordering::Relaxed),
                "dropped": m.dropped.load(Ordering::Relaxed),
                "inferenceMs": stat(&m.inference_ms),
                "processMs": stat(&m.process_ms),
                "roundTripMs": stat(&m.round_trip_ms),
            })
        })
        .collect();
    Json(serde_json::json!({
        "version": crate::VERSION,
        "uptimeSecs": state.metrics.uptime().as_secs(),
        "openvino": reg.runtime_info,
        "models": models,
    }))
}

async fn prometheus(State(state): State<Arc<AppState>>) -> Response {
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        crate::metrics::render_prometheus(&state.metrics, &state.registry.gauges()),
    )
        .into_response()
}

// ---------------------------------------------------------------------------------------------
// Test page

struct ModelChoice {
    value: String,
    selected: bool,
    is_default: bool,
}

struct TestResult {
    summary: String,
    /// `data:image/jpeg;base64,...` of the annotated image; empty when it could not be drawn.
    image_data_uri: String,
    json: String,
}

#[derive(Template)]
#[template(path = "test.html")]
struct TestTemplate {
    nav: &'static str,
    version: &'static str,
    models: Vec<ModelChoice>,
    min_confidence: String,
    error: Option<String>,
    result: Option<TestResult>,
}

fn test_template(
    state: &AppState,
    selected: &str,
    min_confidence: String,
    error: Option<String>,
    result: Option<TestResult>,
) -> TestTemplate {
    let default = default_name(&state.registry);
    let selected = if selected.is_empty() {
        default.as_str()
    } else {
        selected
    };
    let models = state
        .registry
        .names()
        .into_iter()
        .map(|n| ModelChoice {
            selected: crate::registry::normalize_name(&n)
                == crate::registry::normalize_name(selected),
            is_default: n == default,
            value: n,
        })
        .collect();
    TestTemplate {
        nav: "test",
        version: crate::VERSION,
        models,
        min_confidence,
        error,
        result,
    }
}

async fn test_page(State(state): State<Arc<AppState>>) -> Response {
    render(&test_template(&state, "", String::new(), None, None))
}

/// Same dispatch path as the API (metrics and queues behave identically), then draws the
/// predictions on the uploaded image off the async thread.
async fn test_submit(
    State(state): State<Arc<AppState>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let start = Instant::now();
    let (req, fields) = match parse_detection_body(multipart).await {
        Ok(v) => v,
        Err(e) => return render(&test_template(&state, "", String::new(), Some(e), None)),
    };
    let model = fields.get("model").map(|s| s.trim()).unwrap_or_default();
    let min_conf = fields.get("min_confidence").cloned().unwrap_or_default();
    let (handle, command) = if model.is_empty() {
        (state.registry.default_model(), "detect")
    } else {
        (state.registry.by_name(model), "custom")
    };
    let Some(handle) = handle else {
        let e = if model.is_empty() {
            "No default model: enable a model on the Config page".to_string()
        } else {
            format!("Unknown model '{model}'")
        };
        return render(&test_template(&state, model, min_conf, Some(e), None));
    };

    let image = req.image_data.clone();
    let resp = run_detection(&state, handle, Ok(req), start, command).await;
    let json = serde_json::to_string_pretty(&resp).unwrap_or_default();
    let mut summary = if resp.success {
        format!(
            "{} object(s) from model '{}' on {}: inference {} ms, process {} ms, round trip {} ms.",
            resp.count,
            handle.name,
            resp.executionProvider,
            resp.inferenceMs,
            resp.processMs,
            resp.analysisRoundTripMs
        )
    } else {
        format!("Detection failed: {}", resp.message)
    };
    let mut image_data_uri = String::new();
    if resp.success {
        let preds = resp.predictions.clone();
        let drawn = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let img = crate::image::decode(&image)?;
            let jpeg =
                crate::image::encode_jpeg(&crate::image::draw_predictions(&img, &preds), 85)?;
            Ok(format!(
                "data:image/jpeg;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(jpeg)
            ))
        })
        .await;
        match drawn {
            Ok(Ok(uri)) => image_data_uri = uri,
            Ok(Err(e)) => summary.push_str(&format!(" (could not draw the image: {e:#})")),
            Err(e) => summary.push_str(&format!(" (could not draw the image: {e})")),
        }
    }
    let selected = handle.name.clone();
    render(&test_template(
        &state,
        &selected,
        min_conf,
        None,
        Some(TestResult {
            summary,
            image_data_uri,
            json,
        }),
    ))
}

// ---------------------------------------------------------------------------------------------
// Config page

/// Form values as strings, so a rejected submission can be shown again as typed.
struct ConfigView {
    port: String,
    request_timeout_secs: String,
    worker_queue_size: String,
    device: String,
    gpu_index: String,
    force_cpu: bool,
    cache_dir: String,
    confidence_threshold: String,
    nms_iou: String,
    object_filter: String,
    log_level: String,
    log_path: String,
    save_image_path: String,
    save_ref_image: bool,
    intra_threads: String,
    models_json: String,
}

impl ConfigView {
    fn from_config(c: &Config) -> Self {
        let path = |p: &Option<PathBuf>| {
            p.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        };
        Self {
            port: c.port.to_string(),
            request_timeout_secs: c.request_timeout_secs.to_string(),
            worker_queue_size: c.worker_queue_size.to_string(),
            device: c.device.clone(),
            gpu_index: c.gpu_index.to_string(),
            force_cpu: c.force_cpu,
            cache_dir: c.cache_dir.clone(),
            confidence_threshold: c.confidence_threshold.to_string(),
            nms_iou: c.nms_iou.to_string(),
            object_filter: c.object_filter.join(", "),
            log_level: c.log_level.as_str().to_string(),
            log_path: path(&c.log_path),
            save_image_path: path(&c.save_image_path),
            save_ref_image: c.save_ref_image,
            intra_threads: c.intra_threads.to_string(),
            models_json: serde_json::to_string_pretty(&c.models).unwrap_or_default(),
        }
    }

    /// Replace the values with what the user submitted.
    fn overlay(&mut self, form: &HashMap<String, String>) {
        for &key in FORM_FIELDS {
            let v = form.get(key).cloned();
            match key {
                "force_cpu" => self.force_cpu = v.is_some(),
                "save_ref_image" => self.save_ref_image = v.is_some(),
                _ => {
                    let Some(v) = v else { continue };
                    let slot = match key {
                        "port" => &mut self.port,
                        "request_timeout_secs" => &mut self.request_timeout_secs,
                        "worker_queue_size" => &mut self.worker_queue_size,
                        "device" => &mut self.device,
                        "gpu_index" => &mut self.gpu_index,
                        "cache_dir" => &mut self.cache_dir,
                        "confidence_threshold" => &mut self.confidence_threshold,
                        "nms_iou" => &mut self.nms_iou,
                        "object_filter" => &mut self.object_filter,
                        "log_level" => &mut self.log_level,
                        "log_path" => &mut self.log_path,
                        "save_image_path" => &mut self.save_image_path,
                        "intra_threads" => &mut self.intra_threads,
                        _ => continue,
                    };
                    *slot = v;
                }
            }
        }
        if let Some(v) = form.get("models_json") {
            self.models_json = v.clone();
        }
    }
}

/// One row of the Models card.
struct ModelSelectRow {
    name: String,
    family: String,
    path: String,
    /// The model file exists (resolved against the exe dir).
    exists: bool,
    enabled: bool,
    is_default: bool,
}

fn model_choices(c: &Config) -> Vec<ModelSelectRow> {
    let default = c.effective_default_index();
    c.models
        .iter()
        .enumerate()
        .map(|(i, m)| ModelSelectRow {
            name: m.effective_name(),
            family: m.family.to_string(),
            path: m.path.display().to_string(),
            exists: c.data_path(&m.path).is_file(),
            enabled: m.enabled,
            is_default: default == Some(i),
        })
        .collect()
}

struct LevelChoice {
    value: &'static str,
    selected: bool,
}

#[derive(Template)]
#[template(path = "config.html")]
struct ConfigTemplate {
    nav: &'static str,
    version: &'static str,
    config_path: String,
    c: ConfigView,
    models: Vec<ModelSelectRow>,
    log_levels: Vec<LevelChoice>,
    device_choices: Vec<DeviceChoice>,
    message: Option<String>,
    error: Option<String>,
}

/// One `<option>` of the Device select.
struct DeviceChoice {
    value: String,
    label: String,
    selected: bool,
    disabled: bool,
}

/// Options for the Device select: `auto` first, then every runnable option, then the ones that
/// cannot run (disabled, with the reason). The current value is always present; when it is not
/// runnable it stays enabled so saving the form does not silently change it.
fn device_choices(snap: &DevicesSnapshot, current: &str, gpu_index: &str) -> Vec<DeviceChoice> {
    let parsed = spec::parse(current).ok();
    let gpu_index = gpu_index.trim().parse::<u32>().unwrap_or(0);
    let is_current = |d: &crate::backend::spec::Device| match parsed {
        Some(DeviceSpec::Device(p)) => p == *d || p.with_default_index(gpu_index) == *d,
        _ => false,
    };
    let mut out = vec![DeviceChoice {
        value: "auto".to_string(),
        label: format!("Auto \u{2014} currently: {}", snap.auto_pick_label()),
        selected: matches!(parsed, Some(DeviceSpec::Auto)),
        disabled: false,
    }];
    let (ok, bad): (Vec<_>, Vec<_>) = snap.selection.options.iter().partition(|o| o.runnable);
    for o in ok {
        out.push(DeviceChoice {
            value: o.spec.to_string(),
            label: format!("{} \u{2014} {}", o.spec, o.label),
            selected: is_current(&o.spec),
            disabled: false,
        });
    }
    // Downloadable options are selectable (the download itself is started from the Resources
    // card / `fetch`; until it is installed the model falls back as usual).
    let (dl, bad): (Vec<_>, Vec<_>) = bad.into_iter().partition(|o| o.is_downloadable());
    for o in dl {
        let summary = o.download.as_ref().map_or("", |d| d.summary.as_str());
        out.push(DeviceChoice {
            value: o.spec.to_string(),
            label: format!("{} \u{2014} {summary}", o.spec),
            selected: is_current(&o.spec),
            disabled: false,
        });
    }
    for o in bad {
        let selected = is_current(&o.spec);
        out.push(DeviceChoice {
            value: o.spec.to_string(),
            label: format!(
                "{} \u{2014} not available: {}",
                o.spec,
                o.reason.as_deref().unwrap_or("not runnable")
            ),
            selected,
            disabled: !selected,
        });
    }
    if !out.iter().any(|c| c.selected) {
        out.push(DeviceChoice {
            value: current.trim().to_string(),
            label: format!("{} \u{2014} not a known option", current.trim()),
            selected: true,
            disabled: false,
        });
    }
    out
}

fn config_template(
    state: &AppState,
    message: Option<String>,
    error: Option<String>,
    submitted: Option<&HashMap<String, String>>,
) -> ConfigTemplate {
    let (mut c, models) = {
        let cfg = state.config_read();
        (ConfigView::from_config(&cfg), model_choices(&cfg))
    };
    if let Some(form) = submitted {
        c.overlay(form);
    }
    let log_levels = [
        LogLevel::Trace,
        LogLevel::Debug,
        LogLevel::Info,
        LogLevel::Warn,
        LogLevel::Error,
    ]
    .iter()
    .map(|l| LevelChoice {
        value: l.as_str(),
        selected: l.as_str().eq_ignore_ascii_case(c.log_level.trim()),
    })
    .collect();
    let device_choices = device_choices(&devices_snapshot(state), &c.device, &c.gpu_index);
    ConfigTemplate {
        nav: "config",
        version: crate::VERSION,
        config_path: state.config_path.display().to_string(),
        c,
        models,
        log_levels,
        device_choices,
        message,
        error,
    }
}

async fn config_page(State(state): State<Arc<AppState>>) -> Response {
    render(&config_template(&state, None, None, None))
}

/// Apply `edit` to a copy of the config, write it to the config file and, on success, make it
/// the in-memory config. Returns the success message for the config page ("restart to apply",
/// plus the outcome of applying a changed log level immediately).
fn save_config(
    state: &AppState,
    edit: impl FnOnce(&mut Config) -> anyhow::Result<()>,
) -> anyhow::Result<String> {
    let level = {
        let mut cfg = state.config_write();
        let mut new = cfg.clone();
        edit(&mut new)?;
        new.save(&state.config_path)?;
        let level_changed = new.log_level != cfg.log_level;
        *cfg = new;
        level_changed.then_some(cfg.log_level)
    };
    info!(path = %state.config_path.display(), "config saved from the web UI");
    let mut msg = format!(
        "Saved to {}. Restart the server to apply the changes.",
        state.config_path.display()
    );
    if let Some(level) = level {
        match apply_log_level(state, level) {
            Ok(()) => msg.push_str(&format!(
                " The log level ({}) was applied immediately.",
                level.as_str()
            )),
            Err(e) => msg.push_str(&format!(" Log level not applied: {e:#}.")),
        }
    }
    Ok(msg)
}

/// Validate and save the form. Changes apply on restart, except the log level (immediately).
/// `default_model` and the `enabled` flags are only changed when the form carries them (the
/// page's main form does not; the Models card posts to `/config/models`).
async fn config_submit(
    State(state): State<Arc<AppState>>,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => {
            return render(&config_template(
                &state,
                None,
                Some(format!("Invalid form submission: {e}")),
                None,
            ));
        }
    };
    match save_config(&state, |c| apply_config_form(c, &form)) {
        Ok(msg) => render(&config_template(&state, Some(msg), None, None)),
        Err(e) => {
            warn!("rejected config form: {e:#}");
            render(&config_template(
                &state,
                None,
                Some(format!("Not saved: {e:#}")),
                Some(&form),
            ))
        }
    }
}

/// The Models card: `enabled` (repeated, one per checked model), `default_model` (radio) and
/// `action` (`save` or `restart`). Saves like `config_submit`; `restart` then restarts like
/// `POST /config/restart`.
async fn config_models(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => {
            return render(&config_template(
                &state,
                None,
                Some(format!("Invalid form submission: {e}")),
                None,
            ));
        }
    };
    fn field<'a>(form: &'a [(String, String)], k: &'a str) -> impl Iterator<Item = &'a str> {
        form.iter()
            .filter(move |(key, _)| key == k)
            .map(|(_, v)| v.as_str())
    }
    let enabled: Vec<String> = field(&form, "enabled").map(str::to_string).collect();
    let default = field(&form, "default_model").next();
    let restart = field(&form, "action").any(|a| a == "restart");
    match save_config(&state, |c| apply_models_selection(c, &enabled, default)) {
        Ok(_) if restart => restart_response(&state),
        Ok(msg) => render(&config_template(&state, Some(msg), None, None)),
        Err(e) => {
            warn!("rejected models selection: {e:#}");
            render(&config_template(
                &state,
                None,
                Some(format!("Models not saved: {e:#}")),
                None,
            ))
        }
    }
}

fn apply_log_level(state: &AppState, level: LogLevel) -> anyhow::Result<()> {
    match &state.log_reload {
        Some(h) => h.set_level(level),
        None => anyhow::bail!("runtime log level control is not available"),
    }
}

#[derive(Template)]
#[template(path = "message.html")]
struct MessageTemplate {
    nav: &'static str,
    version: &'static str,
    heading: String,
    message: String,
    refresh_secs: u32,
    refresh_url: String,
}

/// Stops the HTTP server and the workers of this generation; the main binary then reloads the
/// config file and starts again.
async fn config_restart(State(state): State<Arc<AppState>>) -> Response {
    restart_response(&state)
}

/// Cancel this generation's restart token shortly (so the reply gets out) and render the
/// "Restarting" page.
fn restart_response(state: &AppState) -> Response {
    info!("restart requested from the web UI");
    let token = state.restart.clone();
    tokio::spawn(async move {
        tokio::time::sleep(RESTART_DELAY).await;
        token.cancel();
    });
    render(&MessageTemplate {
        nav: "config",
        version: crate::VERSION,
        heading: "Restarting".into(),
        message: "The server is reloading its config and recompiling the enabled models. This \
                  page returns to the home page in a few seconds."
            .into(),
        refresh_secs: 5,
        refresh_url: "/".into(),
    })
}

/// `level` from the urlencoded body or the query string. Applied immediately through the
/// tracing reload handle and persisted as `log_level` in the config file. Browsers (Accept:
/// text/html) get the config page back, other clients JSON.
async fn config_loglevel(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    let level = form
        .ok()
        .and_then(|Form(f)| f.get("level").cloned())
        .or_else(|| query.ok().and_then(|Query(q)| q.get("level").cloned()));
    let result = (|| -> anyhow::Result<LogLevel> {
        let raw = level.ok_or_else(|| anyhow::anyhow!("missing field 'level'"))?;
        let level: LogLevel = raw.trim().parse()?;
        apply_log_level(&state, level)?;
        let mut cfg = state.config_write();
        let mut new = cfg.clone();
        new.log_level = level;
        new.save(&state.config_path)?;
        *cfg = new;
        Ok(level)
    })();
    let html = wants_html(&headers);
    match result {
        Ok(level) => {
            info!(level = level.as_str(), "log level changed from the web UI");
            let msg = format!("Log level set to {} and saved.", level.as_str());
            if html {
                render(&config_template(&state, Some(msg), None, None))
            } else {
                Json(
                    serde_json::json!({ "success": true, "message": msg, "level": level.as_str() }),
                )
                .into_response()
            }
        }
        Err(e) => {
            let msg = format!("Log level not changed: {e:#}");
            if html {
                render(&config_template(&state, None, Some(msg), None))
            } else {
                (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "success": false, "message": msg })),
                )
                    .into_response()
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Static assets

async fn style_css() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/css; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        STYLE_CSS,
    )
        .into_response()
}

async fn favicon() -> Response {
    (
        [
            (header::CONTENT_TYPE, "image/x-icon"),
            (header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        FAVICON_ICO,
    )
        .into_response()
}

async fn fallback(uri: axum::http::Uri) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "success": false,
            "error": format!("Not found: {}", uri.path()),
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::metrics::ModelMetrics;
    use crate::registry::RuntimeInfo;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// State with two models that failed to load (no OpenVINO needed); the second name contains
    /// markup to check escaping.
    fn test_state() -> Arc<AppState> {
        let dir = std::env::temp_dir().join(format!("bo_srv_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let names = ["ipcam-general", "<b>evil</b>"];
        let metrics = Metrics::new("0.0.0");
        let workers = names
            .iter()
            .map(|n| {
                let mm = Arc::new(ModelMetrics::new(*n, "CPU", ""));
                metrics.models.write().unwrap().push(mm.clone());
                WorkerHandle::failed(*n, "model file not found: x.onnx", mm)
            })
            .collect();
        let info = RuntimeInfo {
            openvino_version: "2026.0.0-test".into(),
            available_devices: vec!["CPU".into()],
            has_gpu: false,
        };
        let registry = ModelRegistry::from_handles(workers, Some(0), info);
        let config = Config {
            models: names
                .iter()
                .map(|n| ModelConfig {
                    name: Some(n.to_string()),
                    path: format!("models/{n}.onnx").into(),
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

    async fn call(state: &Arc<AppState>, req: Request<Body>) -> (StatusCode, HeaderMap, Vec<u8>) {
        let resp = router(state.clone()).oneshot(req).await.unwrap();
        let status = resp.status();
        let headers = resp.headers().clone();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec();
        (status, headers, body)
    }

    async fn get_text(state: &Arc<AppState>, uri: &str) -> (StatusCode, String) {
        let (s, _, b) = call(state, Request::get(uri).body(Body::empty()).unwrap()).await;
        (s, String::from_utf8(b).unwrap())
    }

    fn post_form(uri: &str, body: &str, accept_html: bool) -> Request<Body> {
        let mut b =
            Request::post(uri).header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
        if accept_html {
            b = b.header(header::ACCEPT, "text/html");
        }
        b.body(Body::from(body.to_string())).unwrap()
    }

    fn urlencode(s: &str) -> String {
        s.bytes()
            .map(|b| match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' => {
                    (b as char).to_string()
                }
                _ => format!("%{b:02X}"),
            })
            .collect()
    }

    fn multipart(fields: &[(&str, &[u8])]) -> Request<Body> {
        let boundary = "XBOUNDARYX";
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            if *name == "image" {
                body.extend_from_slice(
                    b"Content-Disposition: form-data; name=\"image\"; filename=\"t.jpg\"\r\n\
                      Content-Type: image/jpeg\r\n\r\n",
                );
            } else {
                body.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
                );
            }
            body.extend_from_slice(value);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        Request::post("/test")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap()
    }

    #[test]
    fn uptime_format() {
        assert_eq!(format_uptime(Duration::from_secs(5)), "5s");
        assert_eq!(format_uptime(Duration::from_secs(65)), "1m 5s");
        assert_eq!(format_uptime(Duration::from_secs(3 * 3600 + 5)), "3h 0m 5s");
        assert_eq!(format_uptime(Duration::from_secs(90_061)), "1d 1h 1m 1s");
    }

    #[tokio::test]
    async fn pages_render_and_escape() {
        let state = test_state();
        let (s, home) = get_text(&state, "/").await;
        assert_eq!(s, StatusCode::OK);
        assert!(home.contains("2026.0.0-test"), "{home}");
        assert!(home.contains("<td>ipcam-general</td>"));
        // The HTTP integration tests poll for `<td>Ready</td>` / `Failed:` in this page.
        assert!(home.contains("<td>Failed: model file not found: x.onnx</td>"));
        assert!(!home.contains("<b>evil</b>") && home.contains("&#60;b&#62;evil"));
        assert!(home.contains("href=\"/static/style.css\""));

        let (s, stats) = get_text(&state, "/stats").await;
        assert_eq!(s, StatusCode::OK);
        assert!(stats.contains("http-equiv=\"refresh\" content=\"5\""));
        assert!(stats.contains("ipcam-general") && stats.contains("title=\"CPU\""));
        assert!(!stats.contains("<b>evil</b>"));

        let (s, json) = get_text(&state, "/stats.json").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["models"][0]["name"], "ipcam-general");
        assert_eq!(v["models"][0]["default"], true);

        let (s, test) = get_text(&state, "/test").await;
        assert_eq!(s, StatusCode::OK);
        assert!(test.contains("<option value=\"ipcam-general\" selected>ipcam-general (default)"));
        assert!(test.contains("enctype=\"multipart/form-data\""));

        let (s, cfg) = get_text(&state, "/config").await;
        assert_eq!(s, StatusCode::OK);
        assert!(cfg.contains("name=\"models_json\""));
        assert!(cfg.contains("&#34;ipcam-general&#34;"), "{cfg}");
        assert!(cfg.contains("<option value=\"info\" selected>"));
    }

    fn snapshot(devices: &[&str]) -> DevicesSnapshot {
        use crate::backend::detect::{GpuAdapter, GpuVendor};
        let hardware = HardwareInfo::new(
            "linux",
            "x86_64",
            vec![
                GpuAdapter {
                    vendor: GpuVendor::Intel,
                    name: "UHD 630".into(),
                    vram_mb: 0,
                    index: 0,
                    discrete: false,
                },
                GpuAdapter {
                    vendor: GpuVendor::Nvidia,
                    name: "RTX 3060".into(),
                    vram_mb: 12288,
                    index: 1,
                    discrete: true,
                },
            ],
        );
        let names: Vec<String> = devices.iter().map(|d| d.to_string()).collect();
        let selection = select(&hardware, &RuntimeProbe::openvino_only(&names), true);
        DevicesSnapshot {
            hardware,
            selection,
            openvino_version: "test".into(),
            openvino_devices: names,
            onnxruntime: None,
        }
    }

    #[test]
    fn device_choices_list_auto_first_and_grey_out_unrunnable() {
        let snap = snapshot(&["CPU", "GPU.0"]);
        let c = device_choices(&snap, "auto", "0");
        assert_eq!(c[0].value, "auto");
        assert!(
            c[0].label.starts_with("Auto \u{2014} currently: "),
            "{}",
            c[0].label
        );
        assert!(c[0].selected && !c[0].disabled);
        assert_eq!(c.iter().filter(|d| d.selected).count(), 1);
        let gpu = c
            .iter()
            .find(|d| d.value == "openvino:gpu.0")
            .expect("gpu option");
        assert!(!gpu.disabled);
        // ONNX Runtime is not in this probe: its options are listed, disabled, with a reason.
        let cuda = c
            .iter()
            .find(|d| d.value.starts_with("ort:cuda"))
            .expect("cuda option");
        assert!(cuda.disabled && !cuda.selected && cuda.label.contains("not available"));
        // Runnable options come before the disabled ones.
        let first_disabled = c.iter().position(|d| d.disabled).unwrap();
        assert!(c[..first_disabled].iter().all(|d| !d.disabled));
        assert!(c[first_disabled..].iter().all(|d| d.disabled));
    }

    #[test]
    fn device_choices_select_current_value() {
        let snap = snapshot(&["CPU", "GPU.0"]);
        // Legacy spellings map to their option.
        for cur in ["GPU", "openvino:gpu", " gpu "] {
            let c = device_choices(&snap, cur, "0");
            let sel: Vec<_> = c.iter().filter(|d| d.selected).collect();
            assert_eq!(sel.len(), 1, "{cur}");
            assert_eq!(sel[0].value, "openvino:gpu.0", "{cur}");
        }
        // A current value that cannot run stays selected and enabled (so saving keeps it).
        let c = device_choices(&snap, "ort:cuda", "0");
        let sel: Vec<_> = c.iter().filter(|d| d.selected).collect();
        assert_eq!(sel.len(), 1);
        assert!(!sel[0].disabled && sel[0].label.contains("not available"));
        // An unparseable value is kept as an extra option.
        let c = device_choices(&snap, "vulkan", "0");
        let last = c.last().unwrap();
        assert!(last.selected && last.value == "vulkan");
    }

    #[tokio::test]
    async fn devices_endpoint_and_pages() {
        let state = test_state();
        let (s, h, b) = call(
            &state,
            Request::get("/v1/devices").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(
            h[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("application/json")
        );
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["success"], true);
        assert!(v["hardware"]["gpus"].is_array());
        assert_eq!(v["runtimes"]["openvino"]["version"], "2026.0.0-test");
        assert_eq!(v["runtimes"]["openvino"]["devices"][0], "CPU");
        let opts = v["selection"]["options"].as_array().unwrap();
        let cpu = opts
            .iter()
            .find(|o| o["spec"] == "openvino:cpu")
            .expect("cpu option");
        assert_eq!(cpu["runnable"], true);
        assert!(opts.iter().any(|o| o["spec"] == "ort:cpu"));
        assert!(v["selection"]["auto"].is_array());
        assert_eq!(v["autoPick"], v["selection"]["auto"][0]);

        let (_, cfg) = get_text(&state, "/config").await;
        assert!(cfg.contains("<select name=\"device\">"), "{cfg}");
        assert!(cfg.contains("<option value=\"auto\""));
        assert!(cfg.contains("currently:"));
        let (_, home) = get_text(&state, "/").await;
        assert!(
            home.contains("ONNX Runtime") && home.contains("GPUs") && home.contains("Auto picks")
        );
    }

    #[tokio::test]
    async fn static_assets() {
        let state = test_state();
        let (s, h, b) = call(
            &state,
            Request::get("/favicon.ico").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(h[header::CONTENT_TYPE], "image/x-icon");
        assert_eq!(&b[..4], &[0, 0, 1, 0]);
        let (s, h, b) = call(
            &state,
            Request::get("/static/style.css")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(
            h[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/css")
        );
        assert!(
            String::from_utf8(b)
                .unwrap()
                .contains("prefers-color-scheme")
        );
        let (s, _) = get_text(&state, "/nope").await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_page_uses_dispatch_path() {
        let state = test_state();
        let img = crate::image::RgbImage {
            width: 8,
            height: 8,
            rgb: vec![100; 8 * 8 * 3],
        };
        let jpeg = crate::image::encode_jpeg(&img, 90).unwrap();
        let (s, _, b) = call(
            &state,
            multipart(&[
                ("image", &jpeg),
                ("model", b"IPCAM-General.onnx"),
                ("min_confidence", b"0.3"),
            ]),
        )
        .await;
        let page = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(page.contains("Detection failed: Model &#39;ipcam-general&#39; failed to load"));
        assert!(
            page.contains("&#34;command&#34;: &#34;custom&#34;"),
            "{page}"
        );
        assert!(page.contains("value=\"0.3\""));
        // Counted like an API request.
        let w = state.registry.by_name("ipcam-general").unwrap();
        assert_eq!(w.metrics.requests.load(Ordering::Relaxed), 1);

        let (_, _, b) = call(&state, multipart(&[("model", b"ipcam-general")])).await;
        assert!(String::from_utf8(b).unwrap().contains("No image provided"));
        let (_, _, b) = call(&state, multipart(&[("image", &jpeg), ("model", b"zzz")])).await;
        assert!(
            String::from_utf8(b)
                .unwrap()
                .contains("Unknown model &#39;zzz&#39;")
        );
    }

    #[tokio::test]
    async fn config_save_and_validation() {
        let state = test_state();
        let path = state.config_path.clone();

        let bad = format!("port=4000&models_json={}", urlencode("[{not json"));
        let (s, _, b) = call(&state, post_form("/config", &bad, true)).await;
        let page = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(
            page.contains("class=\"error\">Not saved: models:"),
            "{page}"
        );
        // The rejected submission is shown as typed; nothing changed or written.
        assert!(page.contains("[{not json</textarea>"));
        assert!(!path.exists());
        assert_eq!(state.config_read().port, crate::DEFAULT_PORT);

        let models =
            r#"[{"name":"ipcam-general","path":"models/ipcam-general.onnx","family":"yolo5"}]"#;
        let good = format!(
            "port=4000&request_timeout_secs=20&device=CPU&force_cpu=on&confidence_threshold=0.4\
             &object_filter={}&log_level=info&default_model=ipcam-general&models_json={}",
            urlencode("person, car"),
            urlencode(models)
        );
        let (_, _, b) = call(&state, post_form("/config", &good, true)).await;
        let page = String::from_utf8(b).unwrap();
        assert!(page.contains("Restart the server to apply"), "{page}");
        let saved = Config::load(&path).unwrap();
        assert_eq!(saved.port, 4000);
        assert!(saved.force_cpu);
        assert_eq!(saved.object_filter, vec!["person", "car"]);
        assert_eq!(saved.models.len(), 1);
        assert_eq!(*state.config_read(), saved);

        // A models list with every entry disabled is rejected; file and memory unchanged.
        let off =
            r#"[{"name":"ipcam-general","path":"models/ipcam-general.onnx","enabled":false}]"#;
        let body = format!("port=4002&models_json={}", urlencode(off));
        let (s, _, b) = call(&state, post_form("/config", &body, true)).await;
        let page = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(
            page.contains("Not saved: models: at least one model must be enabled"),
            "{page}"
        );
        assert_eq!(Config::load(&path).unwrap(), saved);
        assert_eq!(*state.config_read(), saved);
    }

    #[tokio::test]
    async fn config_models_selection() {
        let state = test_state();
        let path = state.config_path.clone();
        let evil = urlencode("<b>evil</b>");

        let (_, cfg) = get_text(&state, "/config").await;
        assert!(cfg.contains("action=\"/config/models\""), "{cfg}");
        assert!(cfg.contains(
            "name=\"enabled\" value=\"ipcam-general\" aria-label=\"Load ipcam-general\" checked"
        ));
        assert!(cfg.contains(
            "name=\"default_model\" value=\"ipcam-general\" aria-label=\"Default ipcam-general\" checked"
        ));
        assert!(cfg.contains("(missing)"));
        assert!(!cfg.contains("<b>evil</b>"));
        // The main form no longer carries default_model.
        assert!(!cfg.contains("name=\"default_model\" value=\"\""));

        // Enable a subset with an explicit default.
        let body = format!("enabled={evil}&default_model={evil}&action=save");
        let (s, _, b) = call(&state, post_form("/config/models", &body, true)).await;
        let page = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(page.contains("Restart the server to apply"), "{page}");
        let saved = Config::load(&path).unwrap();
        let on: Vec<bool> = saved.models.iter().map(|m| m.enabled).collect();
        assert_eq!(on, [false, true]);
        assert_eq!(saved.default_model.as_deref(), Some("<b>evil</b>"));
        assert_eq!(*state.config_read(), saved);
        assert!(page.contains(
            "name=\"enabled\" value=\"ipcam-general\" aria-label=\"Load ipcam-general\">"
        ));

        // A default that is not checked gets enabled.
        let body = format!("enabled={evil}&default_model=ipcam-general&action=save");
        call(&state, post_form("/config/models", &body, true)).await;
        let saved = Config::load(&path).unwrap();
        assert!(saved.models.iter().all(|m| m.enabled));
        assert_eq!(saved.default_model.as_deref(), Some("ipcam-general"));

        // The main form keeps default_model and the enabled flags.
        let body = "enabled=ipcam-general&default_model=ipcam-general&action=save";
        call(&state, post_form("/config/models", body, true)).await;
        let models_json = serde_json::to_string(&state.config_read().models).unwrap();
        let main = format!("port=4001&models_json={}", urlencode(&models_json));
        call(&state, post_form("/config", &main, true)).await;
        let saved = Config::load(&path).unwrap();
        assert_eq!(saved.port, 4001);
        assert_eq!(saved.default_model.as_deref(), Some("ipcam-general"));
        let on: Vec<bool> = saved.models.iter().map(|m| m.enabled).collect();
        assert_eq!(on, [true, false]);

        // Zero enabled models are rejected and nothing changes.
        let (s, _, b) = call(&state, post_form("/config/models", "action=save", true)).await;
        let page = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(
            page.contains("class=\"error\">Models not saved: select at least one model"),
            "{page}"
        );
        assert_eq!(Config::load(&path).unwrap(), saved);
        assert_eq!(*state.config_read(), saved);
        let (_, _, b) = call(
            &state,
            post_form("/config/models", "enabled=zzz&action=save", true),
        )
        .await;
        assert!(String::from_utf8(b).unwrap().contains("Models not saved"));
        assert!(!state.restart.is_cancelled());

        // Save and restart.
        let (s, _, b) = call(
            &state,
            post_form(
                "/config/models",
                &format!("enabled=ipcam-general&enabled={evil}&action=restart"),
                true,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8(b).unwrap().contains("Restarting"));
        let saved = Config::load(&path).unwrap();
        assert!(saved.models.iter().all(|m| m.enabled));
        assert_eq!(saved.default_model, None);
        tokio::time::timeout(Duration::from_secs(5), state.restart.cancelled())
            .await
            .expect("restart token cancelled");
    }

    #[tokio::test]
    async fn loglevel_and_restart() {
        let state = test_state();
        // No reload handle in tests: the change is refused and nothing is persisted.
        let (s, _, b) = call(&state, post_form("/config/loglevel", "level=debug", false)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(String::from_utf8(b).unwrap().contains("not available"));
        let (s, _, _) = call(&state, post_form("/config/loglevel", "level=loud", false)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _, b) = call(&state, post_form("/config/loglevel", "", true)).await;
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8(b).unwrap().contains("missing field"));
        assert!(!state.config_path.exists());

        assert!(!state.restart.is_cancelled());
        let (s, _, b) = call(
            &state,
            Request::post("/config/restart")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8(b).unwrap().contains("Restarting"));
        tokio::time::timeout(Duration::from_secs(5), state.restart.cancelled())
            .await
            .expect("restart token cancelled");
    }
}
