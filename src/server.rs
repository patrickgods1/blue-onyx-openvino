//! axum HTTP server: CodeProject.AI compatible detection endpoints plus status pages.
//! Handler flow modeled on blue-onyx `server.rs` (MIT).

use crate::api::{VisionCustomListResponse, VisionDetectionRequest, VisionDetectionResponse};
use crate::config::Config;
use crate::metrics::Metrics;
use crate::registry::ModelRegistry;
use crate::startup::ModelState;
use crate::worker::WorkerHandle;
use axum::extract::multipart::MultipartRejection;
use axum::extract::{DefaultBodyLimit, Multipart, Path, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use crossbeam_channel::TrySendError;
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

/// Maximum accepted request body (multipart image upload).
pub const BODY_LIMIT: usize = 32 * 1024 * 1024;
/// How long a handler waits for the worker's reply.
pub const RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);

pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    pub metrics: Arc<Metrics>,
    pub config: Arc<RwLock<Config>>,
    pub started: Instant,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(welcome))
        .route("/v1/vision/detection", post(detection_default))
        .route("/v1/vision/custom/list", get(custom_list).post(custom_list))
        .route("/v1/vision/custom/{model}", post(detection_custom))
        .route("/v1/status/updateavailable", get(update_available))
        .route("/stats", get(stats))
        .route("/prometheus", get(prometheus))
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(state)
}

/// Bind `0.0.0.0:port` and serve until `shutdown` is cancelled.
pub async fn serve(
    state: Arc<AppState>,
    port: u16,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        anyhow::anyhow!("binding {addr}: {e} (is another instance running on port {port}?)")
    })?;
    info!("listening on http://{addr}");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await?;
    info!("HTTP server stopped");
    Ok(())
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn state_str(s: &ModelState) -> String {
    match s {
        ModelState::Initializing => "Initializing".into(),
        ModelState::Ready => "Ready".into(),
        ModelState::Failed(m) => format!("Failed: {m}"),
    }
}

async fn welcome(State(state): State<Arc<AppState>>) -> Html<String> {
    let reg = &state.registry;
    let default = reg
        .default_model()
        .map(|w| w.name.clone())
        .unwrap_or_default();
    let mut rows = String::new();
    for w in reg.workers() {
        let s = w.state.get();
        let label = match (&s, w.lazy) {
            (ModelState::Initializing, true) if w.accepts_while_initializing() => {
                "Lazy (loads on first request)".to_string()
            }
            _ => state_str(&s),
        };
        rows.push_str(&format!(
            "<tr><td>{}{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
            html_escape(&w.name),
            if w.name == default { " (default)" } else { "" },
            html_escape(&label),
            html_escape(&w.execution_provider()),
            w.metrics.requests.load(Ordering::Relaxed),
            w.queue_capacity(),
        ));
    }
    Html(format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Blue Onyx OpenVINO</title></head>\
         <body><h1>Blue Onyx OpenVINO {ver}</h1>\
         <p>OpenVINO {ov}; devices: {devs}; uptime {up} s</p>\
         <table border=\"1\" cellpadding=\"4\"><tr><th>Model</th><th>State</th><th>Device</th>\
         <th>Requests</th><th>Queue</th></tr>\n{rows}</table>\
         <p><a href=\"/stats\">/stats</a> | <a href=\"/prometheus\">/prometheus</a> | \
         <a href=\"/v1/vision/custom/list\">/v1/vision/custom/list</a> | \
         <a href=\"/v1/status/updateavailable\">/v1/status/updateavailable</a></p>\
         <p>POST multipart <code>image</code> to <code>/v1/vision/detection</code> or \
         <code>/v1/vision/custom/&lt;model&gt;</code>.</p></body></html>",
        ver = crate::VERSION,
        ov = html_escape(&reg.core_info.openvino_version),
        devs = html_escape(&reg.core_info.available_devices.join(", ")),
        up = state.started.elapsed().as_secs(),
    ))
}

/// Parse the multipart body into a request. `min_confidence` parse errors become 0.0.
async fn read_request(mut multipart: Multipart) -> Result<VisionDetectionRequest, String> {
    let mut req = VisionDetectionRequest::default();
    let mut got_image = false;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return Err(format!("Invalid multipart body: {e}")),
        };
        match field.name() {
            Some("min_confidence") => {
                let text = field.text().await.unwrap_or_default();
                req.min_confidence = text.trim().parse::<f32>().unwrap_or(0.0);
            }
            Some("image") => {
                if let Some(n) = field.file_name() {
                    req.image_name = n.to_string();
                }
                req.image_data = field
                    .bytes()
                    .await
                    .map_err(|e| format!("Reading image field: {e}"))?;
                got_image = true;
            }
            _ => {}
        }
    }
    if !got_image || req.image_data.is_empty() {
        return Err("No image provided (expected multipart field 'image')".into());
    }
    Ok(req)
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
    Json(run_detection(&state, handle, multipart, start, "detect").await)
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
    Json(run_detection(&state, handle, multipart, start, "custom").await)
}

async fn run_detection(
    state: &AppState,
    handle: &WorkerHandle,
    multipart: Result<Multipart, MultipartRejection>,
    start: Instant,
    command: &str,
) -> VisionDetectionResponse {
    let mut resp = dispatch(handle, multipart, start).await;
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
    multipart: Result<Multipart, MultipartRejection>,
    start: Instant,
) -> VisionDetectionResponse {
    let name = &handle.name;
    let req = match multipart {
        Ok(mp) => match read_request(mp).await {
            Ok(r) => r,
            Err(e) => return VisionDetectionResponse::error(e),
        },
        Err(e) => {
            return VisionDetectionResponse::error(format!(
                "Expected a multipart/form-data body: {e}"
            ));
        }
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

async fn stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let reg = &state.registry;
    let default = reg.default_model().map(|w| w.name.clone());
    let models: Vec<serde_json::Value> = reg
        .workers()
        .iter()
        .map(|w| {
            let m = &w.metrics;
            let dev = w.device.read().ok().and_then(|d| d.clone());
            let stat = |s: &crate::metrics::Stat| {
                serde_json::json!({ "count": s.count(), "avg": s.avg(), "min": s.min(), "max": s.max() })
            };
            serde_json::json!({
                "name": w.name,
                "default": default.as_deref() == Some(w.name.as_str()),
                "state": state_str(&w.state.get()),
                "lazy": w.lazy,
                "requestedDevice": m.device,
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
        crate::metrics::render_prometheus(&state.metrics),
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
