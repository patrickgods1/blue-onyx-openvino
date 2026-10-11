//! axum HTTP server: CodeProject.AI compatible detection endpoints plus the web UI
//! (welcome, stats, test, config and logs pages rendered with askama templates from `templates/`).
//! Handler flow and page set modeled on blue-onyx `server.rs` and its templates (MIT).

use crate::api::{VisionCustomListResponse, VisionDetectionRequest, VisionDetectionResponse};
use crate::backend::detect::HardwareInfo;
use crate::backend::select::{OpenVinoProbe, OrtProbe, RuntimeProbe, Selection, select};
use crate::backend::spec::{self, DeviceSpec};
use crate::cli::LogReloadHandle;
use crate::config::{Config, FORM_FIELDS, LogLevel};
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
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

/// Maximum accepted request body (multipart image upload).
pub const BODY_LIMIT: usize = 32 * 1024 * 1024;
/// How long a handler waits for the worker's reply.
pub const RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
/// Delay between answering `POST /config/restart` and stopping the server, so the reply gets out.
const RESTART_DELAY: Duration = Duration::from_millis(250);

mod benchmark;
mod live;

static STYLE_CSS: &str = include_str!("../assets/style.css");
static LIVE_JS: &str = include_str!("../assets/live.js");
static FAVICON_ICO: &[u8] = include_bytes!("../assets/favicon.ico");

pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    pub metrics: Arc<Metrics>,
    /// The live config (revision, change log, file sync). Every change goes through
    /// [`save_config`] / [`crate::config_store::ConfigStore::update`]. Shared by every
    /// generation of the process.
    pub config: Arc<crate::config_store::ConfigStore>,
    /// The config this generation's registry was started with (what is running; the pages
    /// compare it with `config` to show what a restart would apply).
    pub running: Arc<Config>,
    /// Process start (survives in-process restarts).
    pub started: Instant,
    pub config_path: PathBuf,
    /// Runtime log level control; `None` when logging was not initialized by this process.
    pub log_reload: Option<LogReloadHandle>,
    /// Cancelled by `POST /config/restart`. [`serve`] stops when it fires; the main binary
    /// uses a child of the shutdown token so it also stops the workers of this generation.
    pub restart: CancellationToken,
    /// Download manager and this generation's provisioning plan (None in tests that do not
    /// exercise resources).
    pub resources: Option<crate::resources::status::ResourcesCtx>,
    /// Benchmark runner (one per process, shared by every generation).
    pub benchmark: Arc<crate::benchmark::service::BenchmarkService>,
}

impl AppState {
    /// State without a log reload handle and with a fresh restart token.
    pub fn new(
        registry: Arc<ModelRegistry>,
        metrics: Arc<Metrics>,
        config: Config,
        config_path: PathBuf,
    ) -> Self {
        let mut running = config.clone();
        running.normalize();
        Self {
            registry,
            metrics,
            config: Arc::new(crate::config_store::ConfigStore::new(
                config,
                config_path.clone(),
            )),
            running: Arc::new(running),
            started: Instant::now(),
            config_path,
            log_reload: None,
            restart: CancellationToken::new(),
            resources: None,
            benchmark: crate::benchmark::service::BenchmarkService::new(),
        }
    }

    fn config_read(&self) -> crate::config_store::ConfigRef<'_> {
        self.config.read()
    }

    /// Recent log events for the Logs page (kept by the logging setup's reload handle).
    fn log_buffer(&self) -> Option<&crate::logbuf::LogBuffer> {
        self.log_reload.as_ref().map(LogReloadHandle::buffer)
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
        .route("/v1/config", get(live::config_json))
        .route("/v1/resources", get(resources_json))
        .route("/v1/resources/download", post(resources_download))
        .route("/v1/resources/remove", post(resources_remove))
        .route("/v1/resources/add-to-config", post(resources_add_to_config))
        .route("/v1/resources/export", post(resources_export))
        .route("/v1/resources/export/cancel", post(resources_export_cancel))
        .route("/stats", get(stats_page))
        .route("/stats.json", get(stats_json))
        .route("/prometheus", get(prometheus))
        .route("/test", get(test_page).post(test_submit))
        .route("/config", get(config_page).post(config_submit))
        .route("/config/models", post(config_models))
        .route("/config/restart", post(config_restart))
        .route("/config/loglevel", post(config_loglevel))
        .route("/logs", get(logs_page))
        .route("/logs.json", get(logs_json))
        .route("/benchmark", get(benchmark::page))
        .route(
            "/v1/benchmark",
            get(benchmark::status).post(benchmark::start),
        )
        .route("/v1/benchmark/cancel", post(benchmark::cancel))
        .route("/v1/benchmark/apply", post(benchmark::apply))
        .route(
            "/v1/benchmark/apply-threshold",
            post(benchmark::apply_threshold),
        )
        .route(
            "/v1/benchmark/threshold-search",
            get(benchmark::search_options).post(benchmark::threshold_search),
        )
        .route("/v1/benchmark/settings", post(benchmark::settings))
        .route("/v1/benchmark/images", get(benchmark::images_detail))
        .route("/v1/benchmark/image", get(benchmark::image_file))
        .route("/static/style.css", get(style_css))
        .route("/static/live.js", get(live_js))
        .route("/favicon.ico", get(favicon))
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(state)
}

/// Bind `0.0.0.0:port` (and `[::1]:port`, best effort) and serve until `shutdown` or
/// `state.restart` is cancelled.
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
    // `localhost` may resolve to ::1 first, and some clients do not fall back to 127.0.0.1.
    let loopback6 = std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, port));
    let listener6 = match tokio::net::TcpListener::bind(loopback6).await {
        Ok(l) => Some(l),
        Err(e) => {
            warn!("not listening on {loopback6}: {e}");
            None
        }
    };
    info!("listening on {addr}; open http://localhost:{port}/");
    let restart = state.restart.clone();
    let stopped = move || {
        let (shutdown, restart) = (shutdown.clone(), restart.clone());
        async move {
            tokio::select! {
                _ = shutdown.cancelled() => {}
                _ = restart.cancelled() => {}
            }
        }
    };
    let app = router(state);
    let v4 = axum::serve(listener, app.clone()).with_graceful_shutdown(stopped());
    match listener6 {
        Some(l6) => {
            let v6 = axum::serve(l6, app).with_graceful_shutdown(stopped());
            tokio::try_join!(v4.into_future(), v6.into_future())?;
        }
        None => v4.await?,
    }
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
    /// Short state text ("Ready", "Loading", "Waiting for download", "Failed:", "Lazy").
    state: String,
    /// CSS class of the state badge: `ok`, `loading`, `wait`, `fail`, `lazy`.
    state_class: &'static str,
    /// Download progress, failure reason, ...
    state_detail: Option<String>,
    provider: String,
    requests: u64,
    queue: String,
    /// What a restart would change for this model ("" = nothing).
    pending: String,
}

/// Short state, badge class and detail of a worker for the pages. Ready stays exactly
/// "Ready" and a failure starts with "Failed:" (the HTTP integration tests poll for both).
fn state_view(w: &WorkerHandle) -> (String, &'static str, Option<String>) {
    match w.state.get() {
        ModelState::Ready => ("Ready".into(), "ok", None),
        ModelState::Failed(m) => ("Failed:".into(), "fail", Some(m)),
        ModelState::Initializing if w.accepts_while_initializing() => {
            ("Lazy".into(), "lazy", Some("loads on first request".into()))
        }
        ModelState::Initializing => match w.state.detail() {
            Some(d) => ("Waiting for download".into(), "wait", Some(d)),
            None => ("Loading".into(), "loading", None),
        },
    }
}

/// Distinct execution providers of the loaded models, in config order.
fn providers_in_use(reg: &ModelRegistry) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for w in reg.workers() {
        let p = w
            .device
            .read()
            .ok()
            .and_then(|d| d.as_ref().map(|d| d.execution_provider()));
        if let Some(p) = p
            && !out.contains(&p)
        {
            out.push(p);
        }
    }
    out
}

/// "Force CPU is on — overrides device auto (would use ort:coreml)" when `force_cpu` is set.
fn force_cpu_note(cfg: &Config, snap: &DevicesSnapshot) -> Option<String> {
    if !cfg.force_cpu {
        return None;
    }
    let would = match spec::parse(&cfg.device) {
        Ok(DeviceSpec::Auto) => snap
            .selection
            .auto_pick()
            .map_or_else(|| "nothing runnable".to_string(), |o| o.spec.to_string()),
        Ok(d) => d.with_default_index(cfg.gpu_index).to_string(),
        Err(_) => cfg.device.trim().to_string(),
    };
    Some(format!(
        "Force CPU is on \u{2014} overrides device {} (would use {would}). Uncheck Force CPU on \
         the Config page to use it.",
        cfg.device.trim()
    ))
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
    /// Distinct execution providers of the loaded models.
    in_use: String,
    force_cpu_note: Option<String>,
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
    // legacy flat dir, ...), searched under `download_dir` like the loader does.
    let ort_opts = state.config_read().ort_options();
    let onnxruntime = ort_opts
        .lookup()
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
    let pending = crate::config_merge::pending(&state.running, &state.config_read());
    let models = reg
        .workers()
        .iter()
        .map(|w| {
            let (state, state_class, state_detail) = state_view(w);
            ModelRow {
                pending: pending.for_model(&w.name).join("; "),
                name: w.name.clone(),
                is_default: w.name == default,
                state,
                state_class,
                state_detail,
                provider: w.execution_provider(),
                requests: w.metrics.requests.load(Ordering::Relaxed),
                queue: format!("{}/{}", w.sender.len(), w.queue_capacity()),
            }
        })
        .collect();
    let in_use = providers_in_use(reg);
    let force_cpu_note = force_cpu_note(&state.config_read(), &snap);
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
        in_use: if in_use.is_empty() {
            "nothing loaded yet".to_string()
        } else {
            in_use.join("; ")
        },
        force_cpu_note,
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
    state_class: &'static str,
    device: String,
    provider: String,
    /// "OpenVINO 2026.0.0", "ONNX Runtime 1.24.4"; "-" before the model has loaded.
    runtime: String,
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
    uptime: String,
    models: Vec<StatsRow>,
}

/// Runtime name and version a loaded model runs on ("OpenVINO 2026.0.0").
fn runtime_text(
    reg: &ModelRegistry,
    ort_version: &Option<String>,
    dev: Option<&crate::backend::DeviceInfo>,
) -> String {
    use crate::backend::spec::Runtime;
    match dev.map(|d| d.runtime) {
        None => "-".to_string(),
        Some(Runtime::OpenVino) => {
            // "2026.4.0-22959-99c81491cc3-releases/2026/4 (OpenVINO Runtime)" -> "2026.4.0".
            let v = &reg.runtime_info.openvino_version;
            let short = v.split(['-', ' ']).next().unwrap_or(v);
            format!("OpenVINO {short}").trim().to_string()
        }
        Some(Runtime::Ort) => match ort_version {
            Some(v) => format!("ONNX Runtime {v}"),
            None => "ONNX Runtime".to_string(),
        },
    }
}

/// Version of the loaded ONNX Runtime, when the shared runtimes are free to ask.
fn ort_version(reg: &ModelRegistry) -> Option<String> {
    let rt = reg.runtimes()?.try_lock().ok()?;
    rt.onnxruntime_version()
}

async fn stats_page(State(state): State<Arc<AppState>>) -> Response {
    let reg = &state.registry;
    let default = default_name(reg);
    let ortv = ort_version(reg);
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
                state_class: state_view(w).1,
                runtime: runtime_text(reg, &ortv, dev.as_ref()),
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
        uptime: format_uptime(state.started.elapsed()),
        models,
    })
}

async fn stats_json(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let reg = &state.registry;
    let default = reg.default_model().map(|w| w.name.clone());
    let ortv = ort_version(reg);
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
                "stateText": state_label(w),
                "stateClass": state_view(w).1,
                "runtime": runtime_text(reg, &ortv, dev.as_ref()),
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
        "uptime": format_uptime(state.started.elapsed()),
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
// Resources (phase 7.5)

/// `GET /v1/resources`: every downloadable resource for this platform with its state, size,
/// what it provides and which models wait for it.
async fn resources_json(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let cfg = state.config_read().clone();
    let rows = crate::resources::status::rows(state.resources.as_ref(), &cfg);
    let hw = crate::backend::detect::hardware();
    let manual = state
        .resources
        .as_ref()
        .and_then(|r| r.provision.resolution.manual_message());
    Json(serde_json::json!({
        "success": true,
        "platform": format!("{}-{}", hw.os, hw.arch),
        "autoDownload": cfg.auto_download,
        "allowLargeDownloads": cfg.allow_large_downloads,
        "downloadRoot": cfg.data_root().display().to_string(),
        "manualMessage": manual,
        "resources": rows,
        "localModels": crate::resources::status::local_models(state.resources.as_ref(), &cfg),
    }))
}

/// Form (or urlencoded API) fields of the resource actions: `id`, optional `confirm_large`.
fn action_fields(form: &HashMap<String, String>) -> (String, bool) {
    let id = form
        .get("id")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let confirm = form.get("confirm_large").is_some_and(|v| {
        !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off"
        )
    });
    (id, confirm)
}

/// Answer a resource action: the config page (browsers) or `{success, message}` JSON with
/// 200 / 404 / 409 / 500 / 503.
fn action_response(
    state: &AppState,
    headers: &HeaderMap,
    result: Result<String, crate::resources::status::ActionError>,
) -> Response {
    if let Err(e) = &result {
        warn!("resource action refused: {e}");
    }
    if wants_html(headers) {
        let (msg, err) = match result {
            Ok(m) => (Some(m), None),
            Err(e) => (None, Some(e.to_string())),
        };
        return render(&config_template(state, msg, err, None));
    }
    match result {
        Ok(m) => Json(serde_json::json!({"success": true, "message": m})).into_response(),
        Err(e) => (
            StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(serde_json::json!({"success": false, "message": e.to_string()})),
        )
            .into_response(),
    }
}

fn no_manager() -> crate::resources::status::ActionError {
    crate::resources::status::ActionError::Failed(
        "this server has no download manager (run the service, or use `blue-onyx-prism fetch`)"
            .into(),
    )
}

/// `POST /v1/resources/download` (`id`, `confirm_large` for large resources).
async fn resources_download(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    let form = form.map(|Form(f)| f).unwrap_or_default();
    let (id, confirm) = action_fields(&form);
    let cfg = state.config_read().clone();
    let result = match &state.resources {
        Some(ctx) => crate::resources::status::download(ctx, &cfg, &id, confirm),
        None => Err(no_manager()),
    };
    action_response(&state, &headers, result)
}

/// `POST /v1/resources/remove` (`id`).
async fn resources_remove(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    let form = form.map(|Form(f)| f).unwrap_or_default();
    let (id, _) = action_fields(&form);
    let cfg = state.config_read().clone();
    let result = crate::resources::status::remove(state.resources.as_ref(), &cfg, &id);
    action_response(&state, &headers, result)
}

/// `POST /v1/resources/add-to-config` (`id` of an installed model): appends it to `models` and
/// saves the config file (restart to load it).
async fn resources_add_to_config(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    let form = form.map(|Form(f)| f).unwrap_or_default();
    let (id, _) = action_fields(&form);
    let mut outcome = String::new();
    let saved = save_config(&state, "resources add-to-config", |c| {
        outcome = crate::resources::status::add_to_config(state.resources.as_ref(), c, &id)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(())
    });
    let result = match saved {
        Ok(_) => Ok(format!(
            "{outcome}. Saved to {}.",
            state.config_path.display()
        )),
        Err(e) => match e.downcast::<crate::resources::status::ActionError>() {
            Ok(a) => Err(a),
            Err(e) => Err(crate::resources::status::ActionError::Failed(format!(
                "{e:#}"
            ))),
        },
    };
    action_response(&state, &headers, result)
}

/// `POST /v1/resources/export` (`id` of a YOLO26 model, `confirm_large` when the export
/// toolchain is not installed yet, `add_to_config` to append it to `models` when done): queues the
/// export (one at a time). It runs Ultralytics' exporter locally; see `resources::export`.
async fn resources_export(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    let form = form.map(|Form(f)| f).unwrap_or_default();
    let (id, confirm) = action_fields(&form);
    let add = form.get("add_to_config").is_some_and(|v| {
        !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off"
        )
    });
    let cfg = state.config_read().clone();
    let on_done: Option<crate::resources::export::OnDone> = add.then(|| {
        let config = state.config.clone();
        let id = id.clone();
        Box::new(move |_: &crate::resources::export::ExportOutput| {
            add_export_to_config(&config, &id)
        }) as crate::resources::export::OnDone
    });
    let result = match &state.resources {
        Some(ctx) => crate::resources::status::export(ctx, &cfg, &id, confirm, on_done).map(|m| {
            if add {
                format!("{m}; it is added to the config when done")
            } else {
                m
            }
        }),
        None => Err(no_manager()),
    };
    action_response(&state, &headers, result)
}

/// After an export: append the model to the config (through the config store, which first
/// picks up changes made to the file meanwhile, writes the file and logs the change).
fn add_export_to_config(config: &crate::config_store::ConfigStore, id: &str) {
    let mut msg = String::new();
    match config.update("resources export", |c| {
        msg =
            crate::resources::status::add_to_config(None, c, id).map_err(|e| anyhow::anyhow!(e))?;
        Ok(())
    }) {
        Ok(_) => info!("{msg} ({})", config.path().display()),
        Err(e) => warn!("not adding {id} to the config: {e:#}"),
    }
}

/// `POST /v1/resources/export/cancel` (`id`): cancels a queued or running export (the exporter
/// process is stopped).
async fn resources_export_cancel(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    let form = form.map(|Form(f)| f).unwrap_or_default();
    let (id, _) = action_fields(&form);
    let result = match &state.resources {
        Some(ctx) => crate::resources::status::cancel_export(ctx, &id),
        None => Err(no_manager()),
    };
    action_response(&state, &headers, result)
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

/// Form values as strings, so a rejected submission can be shown again as typed (also the
/// `forms.server.view` of `GET /v1/config`, which the page compares its inputs with).
#[derive(serde::Serialize)]
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
    auto_download: bool,
    allow_large_downloads: bool,
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
            confidence_threshold: crate::config::round_threshold_f64(c.confidence_threshold)
                .to_string(),
            nms_iou: crate::config::clean_f32(c.nms_iou).to_string(),
            object_filter: c.object_filter.join(", "),
            log_level: c.log_level.as_str().to_string(),
            log_path: path(&c.log_path),
            save_image_path: path(&c.save_image_path),
            save_ref_image: c.save_ref_image,
            intra_threads: c.intra_threads.to_string(),
            models_json: serde_json::to_string_pretty(&c.models).unwrap_or_default(),
            auto_download: c.auto_download,
            allow_large_downloads: c.allow_large_downloads,
        }
    }

    /// Replace the values with what the user submitted.
    fn overlay(&mut self, form: &HashMap<String, String>) {
        for &key in FORM_FIELDS {
            let v = form.get(key).cloned();
            match key {
                "force_cpu" => self.force_cpu = v.is_some(),
                "save_ref_image" => self.save_ref_image = v.is_some(),
                "auto_download" | "allow_large_downloads"
                    if !form.contains_key("download_settings") => {}
                "auto_download" => self.auto_download = v.is_some(),
                "allow_large_downloads" => self.allow_large_downloads = v.is_some(),
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
    /// Normalized name (the row's merge keys are `model:<field>:<key>`).
    key: String,
    family: String,
    path: String,
    /// The model file exists (resolved against the exe dir).
    exists: bool,
    enabled: bool,
    is_default: bool,
    /// Per-model Device select: "Global (...)" first, then every device option.
    devices: Vec<DeviceChoice>,
    /// Per-model threshold as typed in the input ("" = global).
    threshold: String,
    /// What the running worker uses ("OpenVINO CPU · threshold 0.50", "not loaded", ...).
    running: String,
    running_class: &'static str,
    /// What a restart would change for this model (empty = nothing).
    pending: Vec<String>,
    /// Latest benchmark recommendation for this model.
    bench: Option<BenchHint>,
    /// Best confidence threshold of the latest benchmark: (value "0.35", objective "f1",
    /// "in use" / "configured; restart to use" when it is the configured threshold).
    best_threshold: Option<(String, String, Option<&'static str>)>,
    /// The file is missing and made by exporting this resource (`model:yolo26s`).
    needs_export: Option<String>,
}

/// "benchmark: ort:coreml" link on the Models card.
struct BenchHint {
    text: String,
    title: String,
    /// `/benchmark#m-<name>` (the Benchmark page's id for the model's card).
    href: String,
}

/// JavaScript `encodeURIComponent` (the Benchmark page builds its card ids with it).
fn encode_uri_component(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'_'
            | b'.'
            | b'!'
            | b'~'
            | b'*'
            | b'\''
            | b'('
            | b')' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Display text of the global device: "auto (ort:coreml)", "openvino:gpu.1", "Force CPU".
fn global_device_label(c: &Config, snap: &DevicesSnapshot) -> String {
    if c.force_cpu {
        return "Force CPU".to_string();
    }
    match spec::parse(&c.device) {
        Ok(DeviceSpec::Auto) => match snap.selection.auto_pick() {
            Some(o) => format!("auto \u{2192} {}", o.spec),
            None => "auto".to_string(),
        },
        Ok(d) => d.with_default_index(c.gpu_index).to_string(),
        Err(_) => c.device.trim().to_string(),
    }
}

/// Options of a model's Device select. `current` None = follows the global device.
fn model_device_choices(
    snap: &DevicesSnapshot,
    current: Option<&str>,
    global: &str,
) -> Vec<DeviceChoice> {
    let mut v = device_choices(snap, current.unwrap_or("auto"), "0");
    if current.is_none() {
        for c in &mut v {
            c.selected = false;
        }
    }
    v.insert(
        0,
        DeviceChoice {
            value: String::new(),
            label: format!("Global ({global})"),
            selected: current.is_none(),
            disabled: false,
        },
    );
    v
}

fn model_choices(
    c: &Config,
    snap: &DevicesSnapshot,
    state: &AppState,
    bench: Option<&live::BenchSummary>,
) -> Vec<ModelSelectRow> {
    let default = c.effective_default_index();
    let global = global_device_label(c, snap);
    let pending = crate::config_merge::pending(&state.running, c);
    let thr = crate::config::round_threshold_f64;
    c.models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let name = m.effective_name();
            let key = crate::registry::normalize_name(&name);
            let facts = bench.and_then(|b| b.models.get(&key));
            let run = live::running_view(state, &name);
            let hint = facts.map(|r| {
                let configured = crate::benchmark::configured_device(c, m, &snap.selection);
                let text = match &r.recommended {
                    Some(d) if configured.as_deref() == Some(d.as_str()) => {
                        if run.device.as_deref() == Some(d.as_str()) {
                            format!("benchmark: {d} (in use)")
                        } else {
                            format!("benchmark: {d} (configured; restart to use)")
                        }
                    }
                    Some(d) => format!("benchmark recommends {d}"),
                    None => "benchmark: no recommendation".to_string(),
                };
                BenchHint {
                    text,
                    title: r.recommendation.clone(),
                    href: format!("/benchmark#m-{}", encode_uri_component(&r.model)),
                }
            });
            let effective = thr(m.confidence_threshold.unwrap_or(c.confidence_threshold));
            let best_threshold = facts.and_then(|f| {
                let t = thr(f.best_threshold?);
                let state = if t != effective {
                    None
                } else if run.threshold == Some(t) {
                    Some("in use")
                } else {
                    Some("configured; restart to use")
                };
                Some((format!("{t:.2}"), f.objective.clone(), state))
            });
            let exists = c.data_path(&m.path).is_file();
            let needs_export = (!exists
                && m.path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("onnx")))
            .then(|| crate::resources::catalog::export_model(&m.path.to_string_lossy()))
            .flatten()
            .map(|r| r.id.to_string());
            ModelSelectRow {
                devices: model_device_choices(snap, m.device.as_deref(), &global),
                key,
                family: m.family.to_string(),
                path: m.path.display().to_string(),
                exists,
                needs_export,
                enabled: m.enabled,
                is_default: default == Some(i),
                threshold: m
                    .confidence_threshold
                    .map(|t| thr(t).to_string())
                    .unwrap_or_default(),
                running: run.text,
                running_class: run.state_class,
                pending: pending.for_model(&name).to_vec(),
                bench: hint,
                best_threshold,
                name,
            }
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
    /// Resources card, grouped (Runtimes, GPU libraries, Models by family).
    resource_groups: Vec<crate::resources::status::ResourceGroup>,
    /// Model files in `models_dir` that no config entry uses.
    local_models: Vec<crate::resources::status::LocalModel>,
    resources_enabled: bool,
    /// Resolved `models_dir` (where YOLO26 exports go).
    models_dir: String,
    /// Where the YOLO26 export toolchain lives (`<data root>/tools`).
    tools_dir: String,
    /// AGPL-3.0 notice shown with the YOLO26 group.
    yolo26_notice: &'static str,
    force_cpu_note: Option<String>,
    /// One line about the `benchmark` config section and the last results.
    bench_summary: String,
    /// Config revision the page was rendered from.
    revision: u64,
    /// Canonical values the main form / the Models card were rendered from (JSON, posted back
    /// as `base` so a save merges instead of overwriting).
    server_base: String,
    card_base: String,
    /// Global confidence threshold (placeholder of the per-model inputs).
    global_threshold: String,
    /// What a restart would apply (all reasons).
    pending: Vec<String>,
    /// A save that conflicted with changes made elsewhere (no-JS answer).
    conflict: Option<live::ConflictView>,
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
    state.config.check_disk();
    let snap = devices_snapshot(state);
    let bench = live::bench_summary(&crate::benchmark::results_path(&state.config_path));
    let (revision, cfg) = state.config.snapshot();
    let mut c = ConfigView::from_config(&cfg);
    let models = model_choices(&cfg, &snap, state, (*bench).as_ref());
    let mut server_base = serde_json::to_string(&crate::config_merge::form_fields(
        &cfg,
        crate::config_merge::Scope::ServerForm,
    ))
    .unwrap_or_default();
    let card_base = serde_json::to_string(&crate::config_merge::form_fields(
        &cfg,
        crate::config_merge::Scope::ModelsCard,
    ))
    .unwrap_or_default();
    if let Some(form) = submitted {
        c.overlay(form);
        // A rejected submission shown as typed keeps the base it was made from.
        if let Some(b) = form.get("base").filter(|b| !b.trim().is_empty()) {
            server_base = b.clone();
        }
    }
    let log_levels = level_choices(&c.log_level);
    let device_choices = device_choices(&snap, &c.device, &c.gpu_index);
    let resource_groups = crate::resources::status::grouped(crate::resources::status::rows(
        state.resources.as_ref(),
        &cfg,
    ));
    let local_models = crate::resources::status::local_models(state.resources.as_ref(), &cfg);
    let models_dir = cfg.data_path(&cfg.models_dir).display().to_string();
    let force_cpu_note = force_cpu_note(&cfg, &snap);
    let tools_dir = cfg
        .data_root()
        .join(crate::resources::export::TOOLS_DIR)
        .display()
        .to_string();
    let bench_summary = {
        let b = &cfg.benchmark;
        let datasets: Vec<String> = b.datasets.iter().map(|d| d.label()).collect();
        let last = match (*bench).as_ref() {
            Some(r) => format!(
                " Last results: {} model(s), updated {}.",
                r.count, r.timestamp
            ),
            None => " No results yet.".to_string(),
        };
        format!(
            "Datasets: {}; {} image(s) per dataset; {} timed run(s) per image; grades weigh accuracy {:.0}%.{last}",
            if datasets.is_empty() {
                "none".to_string()
            } else {
                datasets.join(", ")
            },
            if b.max_images_per_dataset == 0 {
                "all".to_string()
            } else {
                b.max_images_per_dataset.to_string()
            },
            b.repeat_per_image,
            b.weights.accuracy_share() * 100.0
        )
    };
    let pending = {
        let p = crate::config_merge::pending(&state.running, &cfg);
        let mut all = p.global.clone();
        for m in &cfg.models {
            for r in p.for_model(&m.effective_name()) {
                all.push(format!("{}: {r}", m.effective_name()));
            }
        }
        all.extend(p.process.iter().cloned());
        all
    };
    ConfigTemplate {
        bench_summary,
        resource_groups,
        local_models,
        models_dir,
        tools_dir,
        yolo26_notice: crate::resources::catalog::YOLO26_NOTICE,
        force_cpu_note,
        resources_enabled: state.resources.is_some(),
        nav: "config",
        version: crate::VERSION,
        config_path: state.config_path.display().to_string(),
        c,
        models,
        log_levels,
        device_choices,
        message,
        error,
        revision,
        server_base,
        card_base,
        global_threshold: format!(
            "{:.2}",
            crate::config::round_threshold_f64(cfg.confidence_threshold)
        ),
        pending,
        conflict: None,
    }
}

async fn config_page(State(state): State<Arc<AppState>>) -> Response {
    render(&config_template(&state, None, None, None))
}

/// Apply `edit` to the current config through the config store (one writer for everything:
/// it bumps the revision and logs the change under `source`), write the file and, when the log
/// level changed, apply it now. Returns the message for the page ("restart to apply", ...).
fn save_config(
    state: &AppState,
    source: &str,
    edit: impl FnOnce(&mut Config) -> anyhow::Result<()>,
) -> anyhow::Result<String> {
    let up = state.config.update(source, edit)?;
    if !up.changed {
        return Ok(format!(
            "No changes: {} already has these values.",
            state.config_path.display()
        ));
    }
    let level = state.config_read().log_level;
    let level = (level != up.before.log_level).then_some(level);
    info!(path = %state.config_path.display(), revision = up.revision, source, "config saved from the web UI");
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
///
/// The page's form carries `base` (the values it was rendered from): the submission is merged
/// field by field into the current config ([`crate::config_merge`]), so a page opened before
/// another change (Benchmark apply, another tab, the CLI) does not revert it; a field changed
/// on both sides is a conflict (409: the page with a conflict card, or JSON). Clients that
/// send `Accept: application/json` get JSON.
async fn config_submit(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let json = live::wants_json(&headers);
    let pairs = match form {
        Ok(Form(f)) => f,
        Err(e) => {
            let msg = format!("Invalid form submission: {e}");
            if json {
                return live::error_json(StatusCode::BAD_REQUEST, msg);
            }
            return render(&config_template(&state, None, Some(msg), None));
        }
    };
    let form: HashMap<String, String> = pairs.iter().cloned().collect();
    let meta = match live::FormMeta::parse(&pairs) {
        Ok(m) => m,
        Err(e) => {
            let msg = format!("Not saved: {e:#}");
            if json {
                return live::error_json(StatusCode::BAD_REQUEST, msg);
            }
            return render(&config_template(&state, None, Some(msg), None));
        }
    };
    let devices = |c: &Config| {
        (
            c.device.clone(),
            c.force_cpu,
            c.gpu_index,
            c.models
                .iter()
                .map(|m| m.device.clone())
                .collect::<Vec<_>>(),
        )
    };
    let before = devices(&state.config_read());
    match save_config(&state, "config page", |c| {
        live::server_form_edit(c, &form, &meta)
    }) {
        Ok(msg) => {
            // A device change to an option that needs a download (the dropdown's
            // "will download ..." entries, or a per-model `device`): queue it and restart. The
            // new generation waits for the download (or runs on CPU meanwhile) and switches
            // once it is installed, through the usual provisioning restart.
            let cfg = state.config_read().clone();
            if devices(&cfg) != before
                && let Some(ctx) = &state.resources
            {
                let queued = crate::resources::status::queue_needs(ctx, &cfg);
                if !queued.is_empty() {
                    info!("device change needs downloads: {}", queued.join(", "));
                    let msg = format!(
                        "Saved. Downloading {} for the new device setting; the server                          restarts now, serves on what it can meanwhile, and switches when                          the download is installed (progress on the home and Config pages).",
                        queued.join(", ")
                    );
                    if json {
                        schedule_restart(&state);
                        return live::saved_json(&state, msg, true);
                    }
                    return restart_response_with(&state, msg);
                }
            }
            if json {
                return live::saved_json(&state, msg, false);
            }
            render(&config_template(&state, Some(msg), None, None))
        }
        Err(e) => {
            if let Some(conflicts) = live::conflicts_of(&e) {
                warn!("config form conflicts with newer changes: {e}");
                return if json {
                    live::conflict_json(&state, conflicts)
                } else {
                    live::conflict_page(&state, "/config", conflicts, &pairs)
                };
            }
            warn!("rejected config form: {e:#}");
            let msg = format!("Not saved: {e:#}");
            if json {
                return live::error_json(StatusCode::BAD_REQUEST, msg);
            }
            render(&config_template(&state, None, Some(msg), Some(&form)))
        }
    }
}

/// The Models card: `enabled` (repeated, one per checked model), `default_model` (radio),
/// `device.<name>` / `threshold.<name>` ("" = global), `base` (see [`config_submit`]) and
/// `action` (`save` or `restart`). Saves like `config_submit`; `restart` then restarts like
/// `POST /config/restart`.
async fn config_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let json = live::wants_json(&headers);
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => {
            let msg = format!("Invalid form submission: {e}");
            if json {
                return live::error_json(StatusCode::BAD_REQUEST, msg);
            }
            return render(&config_template(&state, None, Some(msg), None));
        }
    };
    let restart = form.iter().any(|(k, a)| k == "action" && a == "restart");
    let result = live::FormMeta::parse(&form).and_then(|meta| {
        save_config(&state, "config page (Models card)", |c| {
            live::models_card_edit(c, &form, &meta)
        })
    });
    match result {
        Ok(msg) if restart => {
            if json {
                schedule_restart(&state);
                return live::saved_json(&state, msg, true);
            }
            restart_response(&state)
        }
        Ok(msg) if json => live::saved_json(&state, msg, false),
        Ok(msg) => render(&config_template(&state, Some(msg), None, None)),
        Err(e) => {
            if let Some(conflicts) = live::conflicts_of(&e) {
                warn!("Models card conflicts with newer changes: {e}");
                return if json {
                    live::conflict_json(&state, conflicts)
                } else {
                    live::conflict_page(&state, "/config/models", conflicts, &form)
                };
            }
            warn!("rejected models selection: {e:#}");
            let msg = format!("Models not saved: {e:#}");
            if json {
                return live::error_json(StatusCode::BAD_REQUEST, msg);
            }
            render(&config_template(&state, None, Some(msg), None))
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
    /// The server is restarting: poll until it answers, then go to `refresh_url` (the meta
    /// refresh stays as the no-JS fallback).
    restarting: bool,
}

/// Stops the HTTP server and the workers of this generation; the main binary then reloads the
/// config file and starts again.
async fn config_restart(State(state): State<Arc<AppState>>) -> Response {
    restart_response(&state)
}

/// Cancel this generation's restart token shortly (so the reply gets out) and render the
/// "Restarting" page.
fn restart_response(state: &AppState) -> Response {
    restart_response_with(
        state,
        "The server is reloading its config and recompiling the enabled models. This page \
         reloads as soon as the server answers again."
            .into(),
    )
}

/// Cancel this generation's restart token after [`RESTART_DELAY`] (so the reply gets out).
fn schedule_restart(state: &AppState) {
    info!("restart requested from the web UI");
    let token = state.restart.clone();
    tokio::spawn(async move {
        tokio::time::sleep(RESTART_DELAY).await;
        token.cancel();
    });
}

/// Restart like `POST /config/restart`, showing `message`.
fn restart_response_with(state: &AppState, message: String) -> Response {
    schedule_restart(state);
    render(&MessageTemplate {
        nav: "config",
        version: crate::VERSION,
        heading: "Restarting\u{2026}".into(),
        message,
        refresh_secs: 5,
        refresh_url: "/config".into(),
        restarting: true,
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
        state.config.update("log level", |c| {
            c.log_level = level;
            Ok(())
        })?;
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
// Logs page

#[derive(Template)]
#[template(path = "logs.html")]
struct LogsTemplate {
    nav: &'static str,
    version: &'static str,
    log_levels: Vec<LevelChoice>,
    /// Log capture is available (this process initialized logging).
    available: bool,
    capacity: usize,
}

fn level_choices(current: &str) -> Vec<LevelChoice> {
    [
        LogLevel::Trace,
        LogLevel::Debug,
        LogLevel::Info,
        LogLevel::Warn,
        LogLevel::Error,
    ]
    .iter()
    .map(|l| LevelChoice {
        value: l.as_str(),
        selected: l.as_str().eq_ignore_ascii_case(current.trim()),
    })
    .collect()
}

async fn logs_page(State(state): State<Arc<AppState>>) -> Response {
    let level = state.config_read().log_level.as_str().to_string();
    render(&LogsTemplate {
        nav: "logs",
        version: crate::VERSION,
        log_levels: level_choices(&level),
        available: state.log_buffer().is_some(),
        capacity: crate::logbuf::DEFAULT_CAPACITY,
    })
}

/// `GET /logs.json?after=<seq>&level=<min>&limit=<n>`: captured log events newer than `after`
/// at `level` or more severe, oldest first. `last` is the cursor for the next poll; a `last`
/// below the client's cursor means the process restarted.
async fn logs_json(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    let q = query.map(|Query(q)| q).unwrap_or_default();
    let bad = |msg: String| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "success": false, "message": msg })),
        )
            .into_response()
    };
    let after = match q.get("after").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        None => 0,
        Some(v) => match v.parse::<u64>() {
            Ok(n) => n,
            Err(_) => return bad(format!("after: '{v}' is not a number")),
        },
    };
    let min = match q.get("level").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        None => None,
        Some(v) if v.eq_ignore_ascii_case("all") => None,
        Some(v) => match crate::logbuf::parse_level(v) {
            Some(l) => Some(l),
            None => return bad(format!("level: '{v}' is not trace|debug|info|warn|error")),
        },
    };
    let limit = q
        .get("limit")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(crate::logbuf::MAX_PAGE)
        .clamp(1, crate::logbuf::MAX_PAGE);
    let level = state.config_read().log_level.as_str();
    match state.log_buffer() {
        Some(buf) => {
            let page = buf.since(after, min, limit);
            Json(serde_json::json!({
                "success": true,
                "level": level,
                "entries": page.entries,
                "last": page.last,
                "first": page.first,
                "truncated": page.truncated,
            }))
            .into_response()
        }
        None => Json(serde_json::json!({
            "success": false,
            "message": "log capture is not available in this process",
            "level": level,
            "entries": [],
            "last": 0,
            "first": 1,
            "truncated": false,
        }))
        .into_response(),
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

/// Shared page script: polls `GET /v1/config` and keeps the header's pending-restart indicator
/// (and the pages that listen) up to date.
async fn live_js() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        LIVE_JS,
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
        assert!(home.contains(
            "<td>Failed:<br><span class=\"detail\">model file not found: x.onnx</span></td>"
        ));
        assert!(home.contains("<tr class=\"st-fail\">"));
        assert!(home.contains("In use") && home.contains("nothing loaded yet"));
        assert!(home.contains("OpenVINO can use"));
        assert!(!home.contains("Force CPU is on"));
        // Prometheus moved to the footer, Logs is in the nav.
        assert!(home.contains("<a href=\"/logs\">Logs</a>"));
        assert!(home.contains("<a href=\"/prometheus\">Prometheus metrics</a>"));
        assert!(!home.contains("<b>evil</b>") && home.contains("&#60;b&#62;evil"));
        assert!(home.contains("href=\"/static/style.css\""));

        let (s, stats) = get_text(&state, "/stats").await;
        assert_eq!(s, StatusCode::OK);
        assert!(!stats.contains("http-equiv=\"refresh\""));
        assert!(stats.contains("fetch(\"/stats.json\""));
        assert!(stats.contains("class=\"stats sticky-first\""));
        assert!(stats.contains("ipcam-general") && stats.contains("title=\"CPU\""));
        assert!(!stats.contains("<b>evil</b>"));

        let (s, json) = get_text(&state, "/stats.json").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["models"][0]["name"], "ipcam-general");
        assert_eq!(v["models"][0]["default"], true);
        assert_eq!(v["models"][0]["stateClass"], "fail");
        assert_eq!(v["models"][0]["runtime"], "-");
        assert_eq!(v["models"][0]["executionProvider"], "not loaded");

        let (s, test) = get_text(&state, "/test").await;
        assert_eq!(s, StatusCode::OK);
        assert!(test.contains("<option value=\"ipcam-general\" selected>ipcam-general (default)"));
        assert!(test.contains("enctype=\"multipart/form-data\""));

        let (s, cfg) = get_text(&state, "/config").await;
        assert_eq!(s, StatusCode::OK);
        assert!(cfg.contains("name=\"models_json\""));
        assert!(cfg.contains("&#34;ipcam-general&#34;"), "{cfg}");
        assert!(cfg.contains("<option value=\"info\" selected>"));
        // One log level control on this page (the Server form); the JSON is under Advanced.
        assert_eq!(cfg.matches("name=\"log_level\"").count(), 1);
        assert!(!cfg.contains("action=\"/config/loglevel\""));
        assert!(cfg.contains("<summary>Advanced: Models (JSON)</summary>"));
        assert!(cfg.contains("nav class=\"toc\""));
        assert!(cfg.contains("Local models (not in config)"));
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
                    cuda: None,
                },
                GpuAdapter {
                    vendor: GpuVendor::Nvidia,
                    name: "RTX 3060".into(),
                    vram_mb: 12288,
                    index: 1,
                    discrete: true,
                    cuda: None,
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
        assert!(
            cfg.contains("<select name=\"device\" id=\"device\">"),
            "{cfg}"
        );
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

    // ---- resources (phase 7.5) ----

    use crate::resources::catalog::{Layout, Part, Provides, Resource, ResourceKind};
    use crate::resources::provision::{Provision, ProvisionOptions, Provisioner};
    use crate::resources::status::ResourcesCtx;

    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    /// Serve `files` (name -> bytes) on 127.0.0.1; returns the base URL.
    fn serve_files(files: Vec<(&'static str, Vec<u8>)>) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut head = Vec::new();
                let mut b = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    if s.read(&mut b).map_or(true, |n| n == 0) {
                        break;
                    }
                    head.push(b[0]);
                }
                let head = String::from_utf8_lossy(&head).to_string();
                let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
                match files.iter().find(|(n, _)| path == format!("/{n}")) {
                    Some((_, body)) => {
                        let _ = write!(
                            s,
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = s.write_all(body);
                    }
                    None => {
                        let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                    }
                }
            }
        });
        base
    }

    fn model_res(
        id: &'static str,
        name: &'static str,
        base: &str,
        files: &[(&str, &[u8])],
        size: Option<u64>,
    ) -> &'static Resource {
        let parts: Vec<Part> = files
            .iter()
            .map(|(f, data)| Part {
                url: leak(format!("{base}/{f}")),
                sha256: leak(crate::resources::manager::tests_sha256(data)),
                size: size.unwrap_or(data.len() as u64),
                file_name: leak(f.to_string()),
                archive: None,
                layout: Layout::File,
            })
            .collect();
        Box::leak(Box::new(Resource {
            id,
            kind: ResourceKind::Model,
            version: "1",
            platform: None,
            parts: Box::leak(parts.into_boxed_slice()),
            dest: "models",
            provides: Provides::Model {
                name,
                family: crate::model::ModelFamilyKind::Yolo5,
            },
            title: leak(format!("Model {name}")),
            description: "fixture",
            license: "MIT",
        }))
    }

    /// State with a download manager over a temp data root, a fixture model served locally and
    /// a fake large model that is never fetched; OpenVINO counts as loaded.
    fn resources_state() -> (Arc<AppState>, PathBuf) {
        let root = std::env::temp_dir().join(format!("bo_res_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let base = serve_files(vec![
            ("fixture-model.onnx", b"onnx bytes".to_vec()),
            ("fixture-model.yaml", b"NAMES:\n- person\n".to_vec()),
        ]);
        let fixture = model_res(
            "model:fixture-model",
            "fixture-model",
            &base,
            &[
                ("fixture-model.onnx", b"onnx bytes"),
                ("fixture-model.yaml", b"NAMES:\n- person\n"),
            ],
            None,
        );
        let big = model_res(
            "model:big-model",
            "big-model",
            "http://127.0.0.1:9",
            &[("big-model.onnx", b"x")],
            Some(600 * 1024 * 1024),
        );
        let config = Config {
            download_dir: Some(root.clone()),
            models: vec![ModelConfig {
                path: "models/other.onnx".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let provisioner = Arc::new(Provisioner::with_options(ProvisionOptions {
            extra_models: vec![fixture, big],
            ..ProvisionOptions::default()
        }));
        let registry = ModelRegistry::from_handles(
            vec![],
            None,
            RuntimeInfo {
                openvino_version: "2026.0.0-test".into(),
                available_devices: vec!["CPU".into()],
                has_gpu: false,
            },
        );
        let mut state = AppState::new(
            Arc::new(registry),
            Arc::new(Metrics::new("0.0.0")),
            config.clone(),
            root.join("cfg.json"),
        );
        state.resources = Some(ResourcesCtx {
            provisioner,
            provision: Arc::new(Provision::none(&config)),
            openvino_loaded: true,
            ort_loaded: None,
        });
        (Arc::new(state), root)
    }

    async fn resources(state: &Arc<AppState>) -> serde_json::Value {
        let (s, body) = get_text(state, "/v1/resources").await;
        assert_eq!(s, StatusCode::OK);
        serde_json::from_str(&body).unwrap()
    }

    fn row<'a>(j: &'a serde_json::Value, id: &str) -> &'a serde_json::Value {
        j["resources"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == id)
            .unwrap_or_else(|| panic!("{id} not listed"))
    }

    async fn post_action(
        state: &Arc<AppState>,
        uri: &str,
        body: &str,
    ) -> (StatusCode, serde_json::Value) {
        let (s, _, b) = call(state, post_form(uri, body, false)).await;
        (s, serde_json::from_slice(&b).unwrap())
    }

    #[tokio::test]
    async fn resources_download_add_and_remove() {
        let (state, root) = resources_state();
        let j = resources(&state).await;
        assert_eq!(j["success"], true);
        assert_eq!(j["autoDownload"], true);
        let fx = row(&j, "model:fixture-model");
        assert_eq!(fx["state"], "available");
        assert_eq!(fx["kind"], "model");
        assert_eq!(fx["provides"][0], "model:fixture-model");
        assert_eq!(fx["large"], false);
        // The platform's catalog is listed too.
        assert_eq!(fx["group"], "models-yolov5");
        let ov = row(&j, "openvino-runtime");
        assert_eq!(ov["group"], "runtimes");
        assert!(ov["size"].as_u64().unwrap() > 0);
        assert!(ov["provides"][0].as_str().unwrap().starts_with("openvino:"));

        // Unknown ids and large downloads without confirmation are refused.
        let (s, b) = post_action(&state, "/v1/resources/download", "id=nope").await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{b}");
        let (s, b) = post_action(&state, "/v1/resources/download", "id=model%3Abig-model").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        assert!(b["message"].as_str().unwrap().contains("confirm"), "{b}");
        assert_eq!(
            row(&resources(&state).await, "model:big-model")["large"],
            true
        );

        // Download the fixture and wait until it is installed.
        let (s, b) =
            post_action(&state, "/v1/resources/download", "id=model%3Afixture-model").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        let started = Instant::now();
        loop {
            let j = resources(&state).await;
            let st = row(&j, "model:fixture-model")["state"]
                .as_str()
                .unwrap()
                .to_string();
            if st == "installed" {
                break;
            }
            assert!(st != "failed", "{j}");
            assert!(started.elapsed() < Duration::from_secs(20), "stuck in {st}");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(root.join("models/fixture-model.onnx").is_file());
        let fx = row(&resources(&state).await, "model:fixture-model").clone();
        assert_eq!(fx["removable"], true);
        assert_eq!(fx["can_add_to_config"], true);

        // Add to config: saved, and the row no longer offers it.
        let (s, b) = post_action(
            &state,
            "/v1/resources/add-to-config",
            "id=model%3Afixture-model",
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(
            b["message"]
                .as_str()
                .unwrap()
                .contains("added 'fixture-model'"),
            "{b}"
        );
        assert!(
            state
                .config_read()
                .models
                .iter()
                .any(|m| m.effective_name() == "fixture-model")
        );
        let saved = Config::load(&state.config_path).unwrap();
        assert_eq!(saved.models.len(), 2);
        assert_eq!(
            row(&resources(&state).await, "model:fixture-model")["can_add_to_config"],
            false
        );

        // The new entry is disabled (another model was enabled), so it can be removed.
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=model%3Afixture-model").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(!root.join("models/fixture-model.onnx").exists());
        assert_eq!(
            row(&resources(&state).await, "model:fixture-model")["state"],
            "available"
        );

        // A loaded runtime is not removed: restart required.
        std::fs::create_dir_all(root.join("openvino")).unwrap();
        std::fs::write(
            root.join("openvino/.installed.json"),
            r#"{"id":"openvino-runtime","version":"x","sha256":[],"files":[]}"#,
        )
        .unwrap();
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=openvino-runtime").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        assert!(
            b["message"].as_str().unwrap().contains("restart required"),
            "{b}"
        );
        assert!(root.join("openvino/.installed.json").exists());
        let ov = row(&resources(&state).await, "openvino-runtime").clone();
        assert_eq!(ov["state"], "installed");
        assert_eq!(ov["loaded"], true);
        assert_eq!(ov["removable"], false);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn config_page_renders_the_resources_card() {
        let (state, root) = resources_state();
        let (s, html) = get_text(&state, "/config").await;
        assert_eq!(s, StatusCode::OK);
        assert!(html.contains("<h2>Resources</h2>"), "{html}");
        assert!(html.contains("model:fixture-model"));
        assert!(html.contains("action=\"/v1/resources/download\""));
        assert!(
            html.contains("Download 629 MB (large)"),
            "large button shows the size"
        );
        assert!(html.contains("export_yolo26.py"));
        assert!(html.contains("name=\"auto_download\" checked"));
        assert!(html.contains("name=\"allow_large_downloads\">"));
        // A browser download goes back to the page with a notice.
        let (s, _, b) = call(
            &state,
            post_form("/v1/resources/download", "id=model%3Abig-model", true),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let b = String::from_utf8(b).unwrap();
        assert!(b.contains("confirm the large download"), "{b}");
        // Without a manager (plain test state) the card says so and downloads are refused.
        let plain = test_state();
        let (_, html) = get_text(&plain, "/config").await;
        assert!(html.contains("has no download manager"));
        let (s, _) = post_action(&plain, "/v1/resources/download", "id=openvino-runtime").await;
        assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR);
        let _ = std::fs::remove_dir_all(&root);
    }
    #[tokio::test]
    async fn force_cpu_note_on_home_and_config() {
        let state = test_state();
        state
            .config
            .update("test", |c| {
                c.force_cpu = true;
                c.device = "auto".into();
                Ok(())
            })
            .unwrap();
        let (_, home) = get_text(&state, "/").await;
        assert!(
            home.contains("Force CPU is on \u{2014} overrides device auto (would use "),
            "{home}"
        );
        let (_, cfg) = get_text(&state, "/config").await;
        assert!(cfg.contains("<select name=\"device\" id=\"device\" disabled>"));
        assert!(cfg.contains("Overridden by Force CPU: Force CPU is on"));
    }

    fn logs_state() -> Arc<AppState> {
        let base = test_state();
        let mut state = AppState::new(
            base.registry.clone(),
            base.metrics.clone(),
            base.config_read().clone(),
            base.config_path.clone(),
        );
        state.log_reload = Some(LogReloadHandle::detached(LogLevel::Info));
        Arc::new(state)
    }

    #[tokio::test]
    async fn logs_page_and_endpoint() {
        let state = logs_state();
        let buf = state.log_buffer().unwrap().clone();
        buf.push(
            tracing::Level::INFO,
            "blue_onyx_prism::registry",
            "model ready".into(),
        );
        buf.push(
            tracing::Level::WARN,
            "blue_onyx_prism::server",
            "queue <full>".into(),
        );
        buf.push(tracing::Level::DEBUG, "x", "detail".into());

        let (s, page) = get_text(&state, "/logs").await;
        assert_eq!(s, StatusCode::OK);
        assert!(page.contains("id=\"logview\""));
        assert!(page.contains("<option value=\"info\" selected>info</option>"));
        assert!(page.contains("class=\"active\">Logs</a>"));

        let (s, body) = get_text(&state, "/logs.json").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["success"], true);
        assert_eq!(v["last"], 3);
        assert_eq!(v["entries"].as_array().unwrap().len(), 3);
        assert_eq!(v["entries"][1]["message"], "queue <full>");
        assert_eq!(v["entries"][1]["level"], "WARN");
        assert_eq!(v["level"], "info");

        let (_, body) = get_text(&state, "/logs.json?after=1&level=warn").await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let e = v["entries"].as_array().unwrap();
        assert_eq!(e.len(), 1);
        assert_eq!(e[0]["seq"], 2);

        let (_, body) = get_text(&state, "/logs.json?after=3").await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(v["entries"].as_array().unwrap().is_empty());
        assert_eq!(v["last"], 3);

        let (s, _) = get_text(&state, "/logs.json?level=loud").await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = get_text(&state, "/logs.json?after=x").await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        // The level control applies through the handle and is saved.
        let (s, _, b) = call(&state, post_form("/config/loglevel", "level=debug", false)).await;
        assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
        assert_eq!(
            Config::load(&state.config_path).unwrap().log_level,
            LogLevel::Debug
        );

        // Without a handle the endpoint says so.
        let plain = test_state();
        let (s, body) = get_text(&plain, "/logs.json").await;
        assert_eq!(s, StatusCode::OK);
        assert!(body.contains("\"success\":false"));
    }

    #[tokio::test]
    async fn local_models_listed_and_added() {
        let (state, root) = resources_state();
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        for f in [
            "mynet.xml",
            "mynet.bin",
            "mynet.onnx",
            "custom.xml",
            "custom.bin",
            "other.onnx",
            "IPcam-general.onnx",
            "yolo26s.onnx",
            "notes.txt",
        ] {
            std::fs::write(models.join(f), b"x").unwrap();
        }
        let j = resources(&state).await;
        let local = j["localModels"].as_array().unwrap();
        let ids: Vec<&str> = local.iter().map(|l| l["id"].as_str().unwrap()).collect();
        // `other.onnx` is configured, IPcam-general is a catalog download, yolo26s.onnx is an
        // export (its own row); .onnx wins over .xml.
        assert_eq!(ids, ["local:custom.xml", "local:mynet.onnx"], "{j}");
        assert_eq!(local[1]["also"], "mynet.xml");
        assert_eq!(local[1]["family"], "auto");
        assert_eq!(local[1]["group"], "local-models");
        assert_eq!(local[0]["format"], "openvino-ir");

        let (_, html) = get_text(&state, "/config").await;
        assert!(html.contains("value=\"local:mynet.onnx\""), "{html}");
        assert!(html.contains("<h3 id=\"res-runtimes\">Runtimes</h3>"));

        let (s, b) = post_action(
            &state,
            "/v1/resources/add-to-config",
            "id=local%3Amynet.onnx",
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(b["message"].as_str().unwrap().contains("added 'mynet'"));
        let last = state.config.changes(1).pop().expect("logged");
        assert_eq!(last.source, "resources add-to-config");
        assert!(last.summary.contains("mynet added"), "{}", last.summary);
        let saved = Config::load(&state.config_path).unwrap();
        let m = saved
            .models
            .iter()
            .find(|m| m.effective_name() == "mynet")
            .unwrap();
        assert_eq!(m.path, PathBuf::from("models/mynet.onnx"));
        assert_eq!(m.family, crate::model::ModelFamilyKind::Auto);
        let j = resources(&state).await;
        let ids: Vec<&str> = j["localModels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["local:custom.xml"]);

        for bad in [
            "id=local%3A..%2Fx.onnx",
            "id=local%3Anotes.txt",
            "id=local%3Amissing.onnx",
        ] {
            let (s, b) = post_action(&state, "/v1/resources/add-to-config", bad).await;
            assert_eq!(s, StatusCode::NOT_FOUND, "{bad}: {b}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- YOLO26 export ----

    /// A server whose exporter uses the fake toolchain (no network, no Python). The config has an
    /// enabled `other.onnx` and a `yolo26s` entry whose file is not exported yet.
    fn export_state(fake: Arc<crate::resources::export::fake::Fake>) -> (Arc<AppState>, PathBuf) {
        let root = std::env::temp_dir().join(format!("bop-srv-export-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("models")).unwrap();
        let config = Config {
            download_dir: Some(root.clone()),
            models: vec![
                ModelConfig {
                    path: "models/other.onnx".into(),
                    ..Default::default()
                },
                ModelConfig {
                    path: "models/yolo26s.onnx".into(),
                    family: crate::model::ModelFamilyKind::Yolo26,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let provisioner = Arc::new(Provisioner::with_options(ProvisionOptions {
            export_hooks: Some((fake.clone(), fake)),
            ..ProvisionOptions::default()
        }));
        let registry = ModelRegistry::from_handles(
            vec![],
            None,
            RuntimeInfo {
                openvino_version: "2026.0.0-test".into(),
                available_devices: vec!["CPU".into()],
                has_gpu: false,
            },
        );
        let mut state = AppState::new(
            Arc::new(registry),
            Arc::new(Metrics::new("0.0.0")),
            config.clone(),
            root.join("cfg.json"),
        );
        // What the runner plans for this config (the resolver's "needs export").
        let hw = crate::backend::detect::hardware();
        let installed = crate::resources::detect_installed(&config, &root, None);
        let resolution = crate::resources::needed(&config, hw, &installed);
        let mut provision = Provision::none(&config);
        provision.resolution = resolution;
        state.resources = Some(ResourcesCtx {
            provisioner,
            provision: Arc::new(provision),
            openvino_loaded: true,
            ort_loaded: None,
        });
        (Arc::new(state), root)
    }

    fn export_supported() -> bool {
        let p = crate::resources::catalog::Platform::current();
        crate::resources::export::unsupported_reason(p.os, p.arch).is_none()
    }

    fn exporter(state: &AppState) -> Arc<crate::resources::export::Exporter> {
        state
            .resources
            .as_ref()
            .unwrap()
            .provisioner
            .exporter()
            .expect("exporter started")
    }

    #[tokio::test]
    async fn yolo26_export_endpoints() {
        if !export_supported() {
            return;
        }
        let fake = Arc::new(crate::resources::export::fake::Fake::new());
        let (state, root) = export_state(fake.clone());
        let j = resources(&state).await;
        let n = row(&j, "model:yolo26n");
        assert_eq!(n["group"], "models-yolo26");
        assert_eq!(n["action"], "export");
        assert_eq!(n["state"], "available", "{n}");
        assert_eq!(n["large"], true, "toolchain not installed yet");
        assert!(n["confirm"].as_str().unwrap().contains("AGPL-3.0"));
        assert!(
            n["confirm"]
                .as_str()
                .unwrap()
                .contains("runs Ultralytics' exporter locally")
        );
        // The configured, not yet exported yolo26s is "needs export", never auto-exported.
        let s_row = row(&j, "model:yolo26s");
        assert_eq!(s_row["state"], "needs-export", "{s_row}");
        assert_eq!(s_row["blocking"], serde_json::json!(["yolo26s"]));
        let uv = row(&j, "tool:uv");
        assert_eq!(uv["action"], "toolchain");
        assert_eq!(uv["state"], "available");
        assert_eq!(uv["removable"], false);

        // The Download endpoint refuses exports; Export without confirmation is refused too.
        let (s, b) = post_action(&state, "/v1/resources/download", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        let (s, b) = post_action(&state, "/v1/resources/export", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        assert!(b["message"].as_str().unwrap().contains("confirm"), "{b}");
        assert!(fake.cmds().is_empty());
        let (s, _) = post_action(&state, "/v1/resources/export", "id=model%3Anope").await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        let (s, _) = post_action(&state, "/v1/resources/export", "id=openvino-runtime").await;
        assert_eq!(s, StatusCode::CONFLICT);

        // Confirmed export, added to the config when done.
        let (s, b) = post_action(
            &state,
            "/v1/resources/export",
            "id=model%3Ayolo26n&confirm_large=1&add_to_config=1",
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{b}");
        let st = exporter(&state)
            .wait("model:yolo26n", Some(Duration::from_secs(20)))
            .unwrap();
        assert_eq!(
            st.state,
            crate::resources::export::ExportState::Installed,
            "{st:?}"
        );
        assert!(root.join("models/yolo26n.onnx").is_file());
        assert!(root.join("models/yolo26n.yaml").is_file());
        let last = state.config.changes(1).pop().expect("logged");
        assert_eq!(last.source, "resources export");
        // Memory and file agree (the in-memory config got the model too).
        assert_eq!(
            *state.config_read(),
            Config::load(&state.config_path).unwrap()
        );
        let saved = Config::load(&state.config_path).unwrap();
        let m = saved
            .models
            .iter()
            .find(|m| m.effective_name() == "yolo26n")
            .expect("added to the config file");
        assert_eq!(m.path, PathBuf::from("models/yolo26n.onnx"));
        assert_eq!(m.family, crate::model::ModelFamilyKind::Yolo26);
        assert_eq!(m.classes, Some(PathBuf::from("models/yolo26n.yaml")));
        assert!(!m.enabled, "another model is enabled");

        let j = resources(&state).await;
        let n = row(&j, "model:yolo26n");
        assert_eq!(n["state"], "installed", "{n}");
        assert_eq!(n["state_text"], "exported");
        assert_eq!(n["can_add_to_config"], false, "already in the config");
        assert_eq!(n["removable"], true);
        assert_eq!(n["large"], false, "toolchain installed now");
        let uv = row(&j, "tool:uv");
        assert_eq!(uv["state"], "installed", "{uv}");
        assert_eq!(uv["removable"], true);
        // Not listed again as a local model.
        assert!(j["localModels"].as_array().unwrap().is_empty(), "{j}");

        // Re-export of an exported model is a no-op; the second size needs no confirmation now.
        let (s, b) = post_action(&state, "/v1/resources/export", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::OK);
        assert!(b["message"].as_str().unwrap().contains("already exported"));
        let (s, b) = post_action(&state, "/v1/resources/export", "id=model%3Ayolo26s").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        exporter(&state).wait("model:yolo26s", Some(Duration::from_secs(20)));
        assert!(root.join("models/yolo26s.onnx").is_file());

        // yolo26s is used by an enabled model: Remove is refused. yolo26n can go.
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=model%3Ayolo26s").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(!root.join("models/yolo26n.onnx").exists());
        assert!(!root.join("models/.yolo26n.installed.json").exists());
        // Remove export toolchain.
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=tool%3Auv").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(!root.join("tools/yolo26-env").exists());
        assert!(!root.join("tools/uv").exists());
        let j = resources(&state).await;
        assert_eq!(row(&j, "tool:uv")["state"], "available");
        assert_eq!(row(&j, "model:yolo26n")["state"], "available");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn yolo26_export_cancel_and_page() {
        if !export_supported() {
            return;
        }
        let fake = Arc::new(crate::resources::export::fake::Fake {
            hang_on: Some("export_yolo26.py"),
            ..crate::resources::export::fake::Fake::new()
        });
        let (state, root) = export_state(fake);
        {
            // allow_large_downloads: no confirmation needed.
            state
                .config
                .update("test", |c| {
                    c.allow_large_downloads = true;
                    Ok(())
                })
                .unwrap();
        }
        let (s, _) = post_action(&state, "/v1/resources/cancel", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::NOT_FOUND, "no such endpoint");
        let (s, b) = post_action(&state, "/v1/resources/export/cancel", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::CONFLICT, "nothing to cancel: {b}");
        let (s, b) = post_action(&state, "/v1/resources/export", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        // Wait until it hangs in the exporter: the row is busy with stage text and progress.
        let deadline = Instant::now() + Duration::from_secs(20);
        let n = loop {
            let j = resources(&state).await;
            let n = row(&j, "model:yolo26n").clone();
            if n["state_text"]
                .as_str()
                .unwrap()
                .starts_with("exporting: exporting to ONNX (5/6)")
            {
                break n;
            }
            assert!(Instant::now() < deadline, "{n}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        assert_eq!(n["state"], "exporting");
        assert_eq!(n["busy"], true);
        assert_eq!(n["cancellable"], true);
        assert!(n["progress"].as_u64().unwrap() >= 78, "{n}");
        assert!(!n["log"].as_array().unwrap().is_empty());
        // Busy: the toolchain cannot be removed, the page shows Cancel and the group.
        let j = resources(&state).await;
        assert_eq!(row(&j, "tool:uv")["removable"], false);
        let (_, html) = get_text(&state, "/config").await;
        assert!(
            html.contains("YOLO26 (exported locally, AGPL-3.0)"),
            "{html}"
        );
        assert!(html.contains("action=\"/v1/resources/export/cancel\""));
        assert!(html.contains("needs export"), "Models card marks yolo26s");
        assert!(html.contains("href=\"#res-models-yolo26\""));
        assert!(html.contains("Remove export toolchain") || html.contains("an export is running"));
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=tool%3Auv").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");

        let (s, b) = post_action(&state, "/v1/resources/export/cancel", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        let st = exporter(&state)
            .wait("model:yolo26n", Some(Duration::from_secs(20)))
            .unwrap();
        assert_eq!(st.state, crate::resources::export::ExportState::Cancelled);
        let j = resources(&state).await;
        let n = row(&j, "model:yolo26n");
        assert_eq!(n["state"], "available");
        assert_eq!(n["state_text"], "export cancelled");
        assert!(!root.join("models/yolo26n.onnx").exists());
        // The page renders the Export button with the AGPL confirm.
        let (_, html) = get_text(&state, "/config").await;
        assert!(html.contains("action=\"/v1/resources/export\""));
        assert!(
            html.contains("data-confirm=\"Export YOLO26 nano?"),
            "{html}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- live config: revision, merge of stale pages, /v1/config ----

    fn html_unescape(s: &str) -> String {
        s.replace("&#34;", "\"")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&#x27;", "'")
            .replace("&lt;", "<")
            .replace("&#60;", "<")
            .replace("&gt;", ">")
            .replace("&#62;", ">")
            .replace("&amp;", "&")
            .replace("&#38;", "&")
    }

    /// The hidden `base` of a Config page form: 0 = Models card, 1 = the main form.
    fn page_base(html: &str, idx: usize) -> String {
        let marker = "name=\"base\" value=\"";
        let mut rest = html;
        for _ in 0..=idx {
            let i = rest.find(marker).expect("base field");
            rest = &rest[i + marker.len()..];
        }
        html_unescape(&rest[..rest.find('"').unwrap()])
    }

    /// The main form as a browser posts it from a page rendered from `c`, with `base` and the
    /// user's `edits`.
    fn server_form_body(c: &Config, base: &str, edits: &[(&str, &str)]) -> String {
        let view = serde_json::to_value(ConfigView::from_config(c)).unwrap();
        let mut pairs: Vec<(String, String)> = Vec::new();
        for (k, v) in view.as_object().unwrap() {
            match v {
                serde_json::Value::Bool(true) => pairs.push((k.clone(), "on".into())),
                serde_json::Value::String(s) => pairs.push((k.clone(), s.clone())),
                _ => {}
            }
        }
        for (k, v) in edits {
            pairs.retain(|(x, _)| x != k);
            pairs.push((k.to_string(), v.to_string()));
        }
        pairs.push(("download_settings".into(), "1".into()));
        pairs.push(("base".into(), base.into()));
        pairs
            .iter()
            .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
            .collect::<Vec<_>>()
            .join("&")
    }

    fn post_json(uri: &str, body: &str) -> Request<Body> {
        Request::post(uri)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .header(header::ACCEPT, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn first_model(state: &AppState) -> (Option<String>, Option<f32>) {
        let c = state.config_read();
        (c.models[0].device.clone(), c.models[0].confidence_threshold)
    }

    /// The reported bug: a Config page opened before a Benchmark apply, then saved, reverted
    /// the applied device and threshold (main form and Models card). Both now merge.
    #[tokio::test]
    async fn stale_config_page_keeps_benchmark_apply() {
        let state = test_state();
        let evil = urlencode("<b>evil</b>");
        state.config.update("test", |_| Ok(())).unwrap();
        // The page is opened...
        let (_, page) = get_text(&state, "/config").await;
        let (card_base, server_base) = (page_base(&page, 0), page_base(&page, 1));
        let stale = state.config_read().clone();
        assert!(page.contains("data-revision=\"1\""));
        // ...then the Benchmark page applies a device and a threshold.
        let (s, _, b) = call(
            &state,
            post_form(
                "/v1/benchmark/apply",
                "model=ipcam-general&device=openvino:cpu",
                false,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
        let (s, _, b) = call(
            &state,
            post_form(
                "/v1/benchmark/apply-threshold",
                "model=ipcam-general&threshold=0.35",
                false,
            ),
        )
        .await;
        let text = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK, "{text}");
        // Rounded in the API answer (was 0.3499999940395355).
        assert!(text.contains("\"to\":0.35"), "{text}");
        assert!(!text.contains("0.34999"), "{text}");
        assert_eq!(state.config.revision(), 3);
        let applied = (Some("openvino:cpu".to_string()), Some(0.35));
        assert_eq!(first_model(&state), applied);

        // 1. The stale main form saved as is: nothing reverted, nothing changed.
        let body = server_form_body(&stale, &server_base, &[]);
        let (s, _, b) = call(&state, post_form("/config", &body, true)).await;
        let html = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(html.contains("No changes"), "{html}");
        assert_eq!(first_model(&state), applied);
        assert_eq!(state.config.revision(), 3);
        // With a user edit: the edit is saved, the applied values stay (memory and file).
        let body = server_form_body(&stale, &server_base, &[("port", "4000")]);
        let (s, _, b) = call(&state, post_form("/config", &body, true)).await;
        assert_eq!(s, StatusCode::OK);
        assert!(
            String::from_utf8(b)
                .unwrap()
                .contains("Restart the server to apply")
        );
        assert_eq!(state.config_read().port, 4000);
        assert_eq!(first_model(&state), applied);
        let file = Config::load(&state.config_path).unwrap();
        assert_eq!(file, *state.config_read());
        let last = state.config.changes(1).pop().unwrap();
        assert_eq!((last.source.as_str(), last.revision), ("config page", 4));
        assert_eq!(last.summary, "Port \u{2192} 4000");

        // 2. The stale Models card with one user edit (a threshold on the other model).
        let card = format!(
            "enabled=ipcam-general&enabled={evil}&default_model=ipcam-general&device.ipcam-general=\
             &device.{evil}=&threshold.ipcam-general=&threshold.{evil}=0.6&base={}&action=save",
            urlencode(&card_base)
        );
        let (s, _, b) = call(&state, post_form("/config/models", &card, true)).await;
        assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
        assert_eq!(first_model(&state), applied);
        assert_eq!(
            state.config_read().models[1].confidence_threshold,
            Some(0.6)
        );
        let last = state.config.changes(1).pop().unwrap();
        assert_eq!(last.source, "config page (Models card)");
        assert_eq!(last.summary, "<b>evil</b> threshold \u{2192} 0.6");

        // 3. A conflicting edit: the stale card picks another device for the same model.
        let conflicting = card.replace("device.ipcam-general=&", "device.ipcam-general=ort:cpu&");
        let rev = state.config.revision();
        let (s, _, b) = call(&state, post_form("/config/models", &conflicting, true)).await;
        let html = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::CONFLICT);
        assert!(html.contains("Not saved: changed elsewhere"), "{html}");
        assert!(html.contains("ipcam-general \u{b7} device"), "{html}");
        assert!(
            html.contains("<code>ort:cpu</code>") && html.contains("<code>openvino:cpu</code>")
        );
        assert!(html.contains("value=\"mine\" checked> Use mine"));
        assert!(html.contains("name=\"resolve_cur.model:device:ipcam-general\""));
        assert_eq!(state.config.revision(), rev, "nothing saved");
        assert_eq!(first_model(&state), applied);
        // The same as JSON (the page's script).
        let (s, _, b) = call(&state, post_json("/config/models", &conflicting)).await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(s, StatusCode::CONFLICT);
        assert_eq!(j["conflict"], true);
        assert_eq!(j["conflicts"][0]["key"], "model:device:ipcam-general");
        assert_eq!(j["conflicts"][0]["mine_text"], "ort:cpu");
        assert_eq!(j["conflicts"][0]["current_text"], "openvino:cpu");
        assert_eq!(j["conflicts"][0]["current"], "openvino:cpu");
        assert!(
            j["conflicts"][0].get("base").is_some(),
            "base was null, not absent"
        );
        // "Use current" keeps the applied device (the rest of the card is still saved).
        let key = urlencode("model:device:ipcam-general");
        let cur = urlencode("\"openvino:cpu\"");
        let resolved = format!("{conflicting}&resolve.{key}=current&resolve_cur.{key}={cur}");
        let (s, _, b) = call(&state, post_json("/config/models", &resolved)).await;
        assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
        assert_eq!(first_model(&state), applied);
        // "Use mine" takes the user's device.
        let resolved = format!("{conflicting}&resolve.{key}=mine&resolve_cur.{key}={cur}");
        let (s, _, b) = call(&state, post_json("/config/models", &resolved)).await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(s, StatusCode::OK, "{j}");
        assert_eq!(j["revision"], state.config.revision());
        assert_eq!(first_model(&state).0.as_deref(), Some("ort:cpu"));
        // "Use mine" against a value the user has not seen is a conflict again.
        let stale_choice = format!("{conflicting}&resolve.{key}=mine&resolve_cur.{key}={cur}")
            .replace(
                "device.ipcam-general=ort:cpu",
                "device.ipcam-general=ort:coreml",
            );
        let (s, _, _) = call(&state, post_json("/config/models", &stale_choice)).await;
        assert_eq!(s, StatusCode::CONFLICT);

        // 4. The stale main form's models JSON changes the applied threshold differently.
        let mut edited = stale.clone();
        edited.models[0].confidence_threshold = Some(0.4);
        let body = server_form_body(&edited, &server_base, &[]);
        let (s, _, b) = call(&state, post_json("/config", &body)).await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(s, StatusCode::CONFLICT, "{j}");
        let keys: Vec<&str> = j["conflicts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["key"].as_str().unwrap())
            .collect();
        assert_eq!(keys, ["model:confidence_threshold:ipcam-general"]);
        assert!(
            j["message"]
                .as_str()
                .unwrap()
                .contains("yours: 0.4, now: 0.35"),
            "{j}"
        );
        // No-JS: the conflict page replays the submission for "Use mine for all".
        let (s, _, b) = call(&state, post_form("/config", &body, true)).await;
        let html = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::CONFLICT);
        assert!(
            html.contains("name=\"resolve_all\" value=\"mine\""),
            "{html}"
        );
        assert!(html.contains("<input type=\"hidden\" name=\"models_json\""));
        assert_eq!(first_model(&state).1, Some(0.35));
    }

    /// Legacy clients (no `base`) keep the old apply-as-submitted behavior.
    #[tokio::test]
    async fn forms_without_base_apply_as_submitted() {
        let state = test_state();
        let (s, _, _) = call(
            &state,
            post_form(
                "/config/models",
                "enabled=ipcam-general&device.ipcam-general=openvino:cpu&threshold.ipcam-general=0.456&action=save",
                true,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(
            first_model(&state),
            (Some("openvino:cpu".to_string()), Some(0.46))
        );
    }

    #[tokio::test]
    async fn config_endpoint_etag_pending_and_disk_changes() {
        let state = test_state();
        let get = |etag: Option<&str>| {
            let mut r = Request::get("/v1/config");
            if let Some(e) = etag {
                r = r.header(header::IF_NONE_MATCH, e);
            }
            r.body(Body::empty()).unwrap()
        };
        let (s, h, b) = call(&state, get(None)).await;
        assert_eq!(s, StatusCode::OK);
        let etag = h[header::ETAG].to_str().unwrap().to_string();
        assert!(etag.starts_with("W/\"1-"), "{etag}");
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(j["revision"], 1);
        assert_eq!(j["restartNeeded"], false);
        assert_eq!(j["models"][0]["name"], "ipcam-general");
        assert_eq!(j["models"][0]["running"]["present"], true);
        assert_eq!(j["models"][0]["running"]["threshold"], 0.5);
        assert_eq!(j["models"][0]["default"], true);
        assert!(j["forms"]["server"]["base"]["model:device:ipcam-general"].is_null());
        assert_eq!(j["forms"]["models"]["base"]["default"], "ipcam-general");
        assert_eq!(
            j["forms"]["server"]["view"]["port"],
            crate::DEFAULT_PORT.to_string()
        );
        // Unchanged: 304.
        let (s, h, b) = call(&state, get(Some(&etag))).await;
        assert_eq!(s, StatusCode::NOT_MODIFIED);
        assert!(b.is_empty());
        assert_eq!(h[header::ETAG].to_str().unwrap(), etag);

        // A change: new revision, pending restart for the model, the change log entry.
        let (s, _, _) = call(
            &state,
            post_form(
                "/v1/benchmark/apply-threshold",
                "model=ipcam-general&threshold=0.35",
                false,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let (s, h, b) = call(&state, get(Some(&etag))).await;
        assert_eq!(s, StatusCode::OK);
        assert_ne!(h[header::ETAG].to_str().unwrap(), etag);
        let text = String::from_utf8(b).unwrap();
        assert!(text.contains("\"threshold\":0.35"), "{text}");
        assert!(!text.contains("0.34999"), "{text}");
        let j: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(j["revision"], 2);
        assert_eq!(j["restartNeeded"], true);
        assert_eq!(j["pendingCount"], 1);
        assert_eq!(j["models"][0]["restartNeeded"], true);
        assert_eq!(
            j["models"][0]["reasons"][0],
            "threshold: 0.50 \u{2192} 0.35"
        );
        // Running: what the worker uses (still the old threshold until a restart).
        assert_eq!(j["models"][0]["running"]["threshold"], 0.5);
        assert_eq!(j["models"][1]["restartNeeded"], false);
        assert_eq!(j["changes"][0]["source"], "benchmark apply-threshold");
        assert_eq!(
            j["changes"][0]["summary"],
            "ipcam-general threshold \u{2192} 0.35"
        );
        // The config file says 0.35, too.
        let file = std::fs::read_to_string(&state.config_path).unwrap();
        assert!(file.contains("\"confidence_threshold\": 0.35"), "{file}");
        // The other pages carry the pending state.
        let (_, home) = get_text(&state, "/").await;
        assert!(
            home.contains("title=\"threshold: 0.50 \u{2192} 0.35\">restart needed"),
            "{home}"
        );
        let (_, cfg) = get_text(&state, "/config").await;
        assert!(
            cfg.contains("<li>ipcam-general: threshold: 0.50 \u{2192} 0.35</li>"),
            "{cfg}"
        );
        assert!(cfg.contains("value=\"0.35\" placeholder=\"0.50\""), "{cfg}");

        // Another process edits the file (the CLI, an editor): picked up on the next poll.
        std::thread::sleep(Duration::from_millis(20));
        let mut other = Config::load(&state.config_path).unwrap();
        other.port = 4321;
        other.save(&state.config_path).unwrap();
        let (_, _, b) = call(&state, get(None)).await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(j["revision"], 3);
        assert_eq!(j["changes"][1]["source"], "file changed on disk");
        assert_eq!(j["changes"][1]["summary"], "Port \u{2192} 4321");
        assert_eq!(j["pending"]["global"][0], "Port: 32168 \u{2192} 4321");
        assert_eq!(j["pendingCount"], 2);
        assert_eq!(state.config_read().port, 4321);
    }

    #[tokio::test]
    async fn log_level_writer_bumps_the_revision() {
        let mut st = Arc::try_unwrap(test_state()).ok().unwrap();
        st.log_reload = Some(LogReloadHandle::detached(LogLevel::Info));
        let state = Arc::new(st);
        let (s, _, _) = call(&state, post_form("/config/loglevel", "level=debug", false)).await;
        assert_eq!(s, StatusCode::OK);
        let last = state.config.changes(1).pop().unwrap();
        assert_eq!((last.source.as_str(), last.revision), ("log level", 2));
        assert_eq!(last.summary, "Log level \u{2192} debug");
        // The log level applies now: no restart needed for it.
        let (_, _, b) = call(
            &state,
            Request::get("/v1/config").body(Body::empty()).unwrap(),
        )
        .await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(j["restartNeeded"], false);
    }
}
