//! axum HTTP server: CodeProject.AI compatible detection endpoints plus the web UI
//! (welcome, stats, test, config pages rendered with askama templates from `templates/`).
//! Handler flow and page set modeled on blue-onyx `server.rs` and its templates (MIT).

use crate::api::{VisionCustomListResponse, VisionDetectionRequest, VisionDetectionResponse};
use crate::cli::LogReloadHandle;
use crate::config::{Config, FORM_FIELDS, LogLevel, apply_config_form};
use crate::metrics::{Metrics, Stat};
use crate::registry::ModelRegistry;
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
        .route("/stats", get(stats_page))
        .route("/stats.json", get(stats_json))
        .route("/prometheus", get(prometheus))
        .route("/test", get(test_page).post(test_submit))
        .route("/config", get(config_page).post(config_submit))
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

fn state_str(s: &ModelState) -> String {
    match s {
        ModelState::Initializing => "Initializing".into(),
        ModelState::Ready => "Ready".into(),
        ModelState::Failed(m) => format!("Failed: {m}"),
    }
}

/// State text for the HTML pages; a lazy model that has not been requested yet says so.
fn state_label(w: &WorkerHandle) -> String {
    let s = w.state.get();
    match (&s, w.lazy) {
        (ModelState::Initializing, true) if w.accepts_while_initializing() => {
            "Lazy (loads on first request)".to_string()
        }
        _ => state_str(&s),
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
    devices: String,
    uptime: String,
    port: u16,
    default_model: String,
    models: Vec<ModelRow>,
}

fn default_name(reg: &ModelRegistry) -> String {
    reg.default_model()
        .map(|w| w.name.clone())
        .unwrap_or_default()
}

async fn welcome(State(state): State<Arc<AppState>>) -> Response {
    let reg = &state.registry;
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
        openvino_version: reg.core_info.openvino_version.clone(),
        devices: reg.core_info.available_devices.join(", "),
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
            "No default model configured",
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
    resp.canUseGPU = state.registry.core_info.has_gpu;
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
            return VisionDetectionResponse::error(format!("Model '{name}' is initializing"));
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
        openvino_version: reg.core_info.openvino_version.clone(),
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
                "state": state_str(&w.state.get()),
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
        "openvino": reg.core_info,
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
            "No default model configured".to_string()
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
    default_model: String,
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
            default_model: c.default_model.clone().unwrap_or_default(),
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
                        "default_model" => &mut self.default_model,
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
    log_levels: Vec<LevelChoice>,
    message: Option<String>,
    error: Option<String>,
}

fn config_template(
    state: &AppState,
    message: Option<String>,
    error: Option<String>,
    submitted: Option<&HashMap<String, String>>,
) -> ConfigTemplate {
    let mut c = ConfigView::from_config(&state.config_read());
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
    ConfigTemplate {
        nav: "config",
        version: crate::VERSION,
        config_path: state.config_path.display().to_string(),
        c,
        log_levels,
        message,
        error,
    }
}

async fn config_page(State(state): State<Arc<AppState>>) -> Response {
    render(&config_template(&state, None, None, None))
}

/// Validate and save the form. Changes apply on restart, except the log level (immediately).
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
    let result = {
        let mut cfg = state.config_write();
        let mut new = cfg.clone();
        apply_config_form(&mut new, &form)
            .and_then(|()| new.save(&state.config_path))
            .map(|()| {
                let level_changed = new.log_level != cfg.log_level;
                *cfg = new;
                level_changed.then_some(cfg.log_level)
            })
    };
    match result {
        Ok(level) => {
            info!(path = %state.config_path.display(), "config saved from the web UI");
            let mut msg = format!(
                "Saved to {}. Restart the server to apply the changes.",
                state.config_path.display()
            );
            if let Some(level) = level {
                match apply_log_level(&state, level) {
                    Ok(()) => msg.push_str(&format!(
                        " The log level ({}) was applied immediately.",
                        level.as_str()
                    )),
                    Err(e) => msg.push_str(&format!(" Log level not applied: {e:#}.")),
                }
            }
            render(&config_template(&state, Some(msg), None, None))
        }
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
        message: "The server is reloading its config and recompiling the models. This page \
                  returns to the home page in a few seconds."
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
    use crate::registry::CoreInfo;
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
        let info = CoreInfo {
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
