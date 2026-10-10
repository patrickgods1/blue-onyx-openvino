//! Per-model inference thread.
//!
//! Each configured model gets one OS thread that owns its [`Backend`] (the `InferRequest` is
//! created on that thread) and receives work over a bounded crossbeam channel. Loading walks the
//! device candidates of the model's [`DeviceSpec`] (e.g. GPU, then CPU): a candidate is accepted
//! only once it compiles, creates its request and passes the warm-up inference. The channel is
//! created before the model is compiled so the HTTP layer can route to it immediately; the
//! effective queue depth is a "soft capacity" computed after the warm-up inference
//! (`request_timeout / warmup_ms`, clamped to 1..=64) and checked by the server through
//! [`WorkerHandle::is_full`]. Flow modeled on blue-onyx `worker.rs` (MIT).

use crate::api::{VisionDetectionRequest, VisionDetectionResponse};
use crate::backend::{Backend, Candidate, DeviceInfo, DeviceSpec, LoadRequest, Runtimes};
use crate::config::ModelConfig;
use crate::metrics::{ModelGauges, ModelMetrics, ModelStateLabel};
use crate::model::preprocess::Preprocessor;
use crate::model::{Family, PostParams};
use crate::startup::{ModelState, StateHandle};
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

/// A queued detection request: the request, where to send the reply, and when it arrived.
pub type WorkItem = (
    VisionDetectionRequest,
    tokio::sync::oneshot::Sender<VisionDetectionResponse>,
    Instant,
);

/// Hard capacity of the channel when the queue is auto-sized.
pub const MAX_AUTO_QUEUE: usize = 64;

/// How often a blocked worker re-checks the shutdown token.
const POLL: Duration = Duration::from_millis(250);

/// Everything a worker needs to load and run one model. Paths are already resolved.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub name: String,
    pub model: ModelConfig,
    /// Where to run; expanded into candidates by `Runtimes::plan`.
    pub device: DeviceSpec,
    pub load: LoadRequest,
    pub classes: Vec<String>,
    pub object_filter: Vec<String>,
    pub confidence_threshold: f32,
    pub nms_iou: f32,
    pub request_timeout: Duration,
    /// 0 = auto-size from the warm-up inference time.
    pub queue_size: usize,
    pub save_image_path: Option<PathBuf>,
    pub save_ref_image: bool,
    /// Compile on the first request instead of at startup.
    pub lazy: bool,
    /// Reported as `canUseGPU` (some GPU-class device option of any runtime can run).
    pub can_use_gpu: bool,
}

/// One-shot "this worker is done with its startup load" signal, used by the registry to make
/// workers compile strictly in config order.
#[derive(Clone, Debug, Default)]
pub struct LoadGate(Arc<(Mutex<bool>, Condvar)>);

impl LoadGate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open(&self) {
        let (lock, cv) = &*self.0;
        let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
        *g = true;
        cv.notify_all();
    }

    pub fn is_open(&self) -> bool {
        *self.0.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Block until the gate opens or `shutdown` is cancelled. Returns true when opened.
    pub fn wait(&self, shutdown: &CancellationToken) -> bool {
        let (lock, cv) = &*self.0;
        let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
        while !*g {
            if shutdown.is_cancelled() {
                return false;
            }
            g = cv
                .wait_timeout(g, POLL)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }
}

/// HTTP-side handle to a worker thread.
pub struct WorkerHandle {
    pub name: String,
    pub sender: Sender<WorkItem>,
    pub state: StateHandle,
    pub metrics: Arc<ModelMetrics>,
    pub device: Arc<RwLock<Option<DeviceInfo>>>,
    pub join: Option<JoinHandle<()>>,
    /// Model compiles on first request (state stays `Initializing` until then).
    pub lazy: bool,
    soft_capacity: Arc<AtomicUsize>,
    gate: LoadGate,
    load_started: Arc<AtomicBool>,
}

impl WorkerHandle {
    /// A handle for a model that could not even be set up (missing file, bad classes file).
    /// It is `Failed` from the start and has no thread.
    pub fn failed(
        name: impl Into<String>,
        msg: impl Into<String>,
        metrics: Arc<ModelMetrics>,
    ) -> Self {
        let (sender, _rx) = crossbeam_channel::bounded(1);
        let state = StateHandle::new();
        state.set(ModelState::Failed(msg.into()));
        let gate = LoadGate::new();
        gate.open();
        Self {
            name: name.into(),
            sender,
            state,
            metrics,
            device: Arc::new(RwLock::new(None)),
            join: None,
            lazy: false,
            soft_capacity: Arc::new(AtomicUsize::new(0)),
            gate,
            load_started: Arc::new(AtomicBool::new(true)),
        }
    }

    /// True when the queue holds at least the (soft) capacity computed after warm-up.
    pub fn is_full(&self) -> bool {
        self.sender.is_full() || self.sender.len() >= self.soft_capacity.load(Ordering::Relaxed)
    }

    /// Effective queue depth limit.
    pub fn queue_capacity(&self) -> usize {
        self.soft_capacity.load(Ordering::Relaxed)
    }

    /// Opens once this worker has finished its startup work (loaded, failed, or deferred as lazy).
    pub fn load_gate(&self) -> LoadGate {
        self.gate.clone()
    }

    /// True for a lazy model that has not started compiling: the server may enqueue to trigger it.
    pub fn accepts_while_initializing(&self) -> bool {
        self.lazy && !self.load_started.load(Ordering::Relaxed)
    }

    /// Lifecycle state for metrics: a lazy model that has not started compiling is `Lazy`.
    pub fn state_label(&self) -> ModelStateLabel {
        match self.state.get() {
            ModelState::Ready => ModelStateLabel::Ready,
            ModelState::Failed(_) => ModelStateLabel::Failed,
            ModelState::Initializing if self.accepts_while_initializing() => ModelStateLabel::Lazy,
            ModelState::Initializing => ModelStateLabel::Initializing,
        }
    }

    /// Snapshot of the live per-model values exported to Prometheus.
    pub fn gauges(&self) -> ModelGauges {
        ModelGauges {
            name: self.name.clone(),
            device: self.metrics.device(),
            provider: self.metrics.execution_provider(),
            state: self.state_label(),
            queue_length: self.sender.len(),
            queue_capacity: self.queue_capacity(),
        }
    }

    /// Execution provider string of the loaded model, or a placeholder before load.
    pub fn execution_provider(&self) -> String {
        self.device
            .read()
            .ok()
            .and_then(|d| d.as_ref().map(|d| d.execution_provider()))
            .unwrap_or_else(|| "OpenVINO (not loaded)".to_string())
    }

    /// Wait for the thread to exit (drop the sender first so it can see the disconnect).
    pub fn join(mut self) {
        let join = self.join.take();
        drop(self);
        if let Some(j) = join
            && j.join().is_err()
        {
            error!("worker thread panicked");
        }
    }
}

/// `clamp(timeout_ms / max(warmup_ms, 1), 1, 64)`.
pub fn auto_queue_size(request_timeout: Duration, warmup_ms: u64) -> usize {
    let t = request_timeout.as_millis() as u64;
    (t / warmup_ms.max(1)).clamp(1, MAX_AUTO_QUEUE as u64) as usize
}

/// Request `min_confidence` overrides the configured threshold when > 0.
pub fn effective_threshold(min_confidence: f32, configured: f32) -> f32 {
    if min_confidence > 0.0 && min_confidence.is_finite() {
        min_confidence
    } else {
        configured
    }
}

/// Spawn a worker that starts loading immediately.
pub fn spawn_worker(
    runtimes: Arc<Mutex<Runtimes>>,
    cfg: WorkerConfig,
    metrics: Arc<ModelMetrics>,
    shutdown: CancellationToken,
) -> WorkerHandle {
    spawn_worker_after(runtimes, cfg, metrics, shutdown, None)
}

/// Spawn a worker that waits for `after` to open before touching the runtimes, so models
/// compile in a deterministic order.
///
/// A lazy worker does not take part in the startup order: it starts serving its channel at once
/// (so its first request compiles it as soon as the runtimes mutex is free, without waiting for
/// every earlier model) and its handle's [`WorkerHandle::load_gate`] simply forwards `after`, so
/// the next model in the config still waits for the previous non-lazy one.
pub fn spawn_worker_after(
    runtimes: Arc<Mutex<Runtimes>>,
    cfg: WorkerConfig,
    metrics: Arc<ModelMetrics>,
    shutdown: CancellationToken,
    after: Option<LoadGate>,
) -> WorkerHandle {
    let hard_cap = if cfg.queue_size > 0 {
        cfg.queue_size
    } else {
        MAX_AUTO_QUEUE
    };
    let (sender, receiver) = crossbeam_channel::bounded::<WorkItem>(hard_cap);
    let state = StateHandle::new();
    let device = Arc::new(RwLock::new(None));
    let soft_capacity = Arc::new(AtomicUsize::new(hard_cap));
    let gate = LoadGate::new();
    let handle_gate = if cfg.lazy {
        after.clone().unwrap_or_else(|| {
            let g = LoadGate::new();
            g.open();
            g
        })
    } else {
        gate.clone()
    };
    let after = if cfg.lazy { None } else { after };
    let load_started = Arc::new(AtomicBool::new(false));

    let ctx = WorkerCtx {
        runtimes,
        cfg: cfg.clone(),
        metrics: metrics.clone(),
        state: state.clone(),
        device: device.clone(),
        soft_capacity: soft_capacity.clone(),
        gate: gate.clone(),
        load_started: load_started.clone(),
        shutdown,
    };
    let name = cfg.name.clone();
    let join = std::thread::Builder::new()
        .name(format!("worker-{name}"))
        .spawn(move || ctx.run(receiver, after));
    let join = match join {
        Ok(j) => Some(j),
        Err(e) => {
            let msg = format!("could not spawn worker thread: {e}");
            error!(model = %name, "{msg}");
            state.set(ModelState::Failed(msg));
            gate.open();
            None
        }
    };
    WorkerHandle {
        name,
        sender,
        state,
        metrics,
        device,
        join,
        lazy: cfg.lazy,
        soft_capacity,
        gate: handle_gate,
        load_started,
    }
}

/// Model state owned by the worker thread after a successful load.
struct Engine {
    backend: Backend,
    family: Box<dyn Family>,
    pre: Preprocessor,
    provider: String,
}

struct WorkerCtx {
    runtimes: Arc<Mutex<Runtimes>>,
    cfg: WorkerConfig,
    metrics: Arc<ModelMetrics>,
    state: StateHandle,
    device: Arc<RwLock<Option<DeviceInfo>>>,
    soft_capacity: Arc<AtomicUsize>,
    gate: LoadGate,
    load_started: Arc<AtomicBool>,
    shutdown: CancellationToken,
}

/// Elapsed milliseconds as the API's i32.
fn ms(d: Duration) -> i32 {
    d.as_millis().min(i32::MAX as u128) as i32
}

impl WorkerCtx {
    fn run(self, rx: Receiver<WorkItem>, after: Option<LoadGate>) {
        let name = self.cfg.name.clone();
        if let Some(prev) = after
            && !prev.wait(&self.shutdown)
        {
            self.gate.open();
            return;
        }

        let mut engine: Option<Engine> = None;
        if self.cfg.lazy {
            info!(model = %name, "lazy model: compiling on first request");
            self.gate.open();
        } else {
            engine = self.try_load();
            self.gate.open();
        }

        loop {
            let item = match rx.recv_timeout(POLL) {
                Ok(item) => item,
                Err(RecvTimeoutError::Timeout) => {
                    if self.shutdown.is_cancelled() {
                        break;
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            };
            if self.shutdown.is_cancelled() {
                break;
            }
            let (req, reply, enqueued) = item;

            if engine.is_none() {
                if let ModelState::Failed(msg) = self.state.get() {
                    let _ = reply.send(self.error_response(
                        format!("Model '{name}' failed to load: {msg}"),
                        0,
                        0,
                    ));
                    continue;
                }
                // Lazy model: the first request triggers the compile and is then served.
                engine = self.try_load();
                if engine.is_none() {
                    let msg = match self.state.get() {
                        ModelState::Failed(m) => m,
                        _ => "load failed".to_string(),
                    };
                    let _ = reply.send(self.error_response(
                        format!("Model '{name}' failed to load: {msg}"),
                        0,
                        0,
                    ));
                    continue;
                }
            } else {
                let waited = enqueued.elapsed();
                if waited > self.cfg.request_timeout {
                    self.metrics.dropped.fetch_add(1, Ordering::Relaxed);
                    warn!(
                        model = %name,
                        waited_ms = waited.as_millis() as u64,
                        timeout_ms = self.cfg.request_timeout.as_millis() as u64,
                        "dropping request that waited longer than the request timeout (server overloaded)"
                    );
                    let _ = reply.send(self.error_response(
                        format!(
                            "Request dropped: waited {} ms in queue (timeout {} ms)",
                            waited.as_millis(),
                            self.cfg.request_timeout.as_millis()
                        ),
                        0,
                        0,
                    ));
                    continue;
                }
            }

            let Some(eng) = engine.as_mut() else { continue };
            let resp = self.process(eng, req, enqueued);
            if reply.send(resp).is_err() {
                debug!(model = %name, "client went away before the response was sent");
            }
        }
        info!(model = %name, "worker exiting");
    }

    /// Load + warm up, updating state/device/soft capacity. Returns None on failure.
    fn try_load(&self) -> Option<Engine> {
        self.load_started.store(true, Ordering::Relaxed);
        self.state.set(ModelState::Initializing);
        match self.load() {
            Ok(engine) => {
                self.state.set(ModelState::Ready);
                Some(engine)
            }
            Err(e) => {
                let msg = format!("{e:#}");
                error!(model = %self.cfg.name, "failed to load model: {msg}");
                self.state.set(ModelState::Failed(msg));
                None
            }
        }
    }

    fn load(&self) -> Result<Engine> {
        let name = &self.cfg.name;
        let candidates = {
            let rt = self.runtimes.lock().unwrap_or_else(|e| e.into_inner());
            rt.plan(&self.cfg.device, &self.cfg.load.path)
        };
        info!(
            model = %name,
            path = %self.cfg.load.path.display(),
            device = %self.cfg.load.requested,
            candidates = %candidates.iter().map(|c| c.device.to_string()).collect::<Vec<_>>().join(", "),
            "loading model"
        );
        let label = format!("model {name}");
        let (engine, compile_ms, warmup_ms) =
            crate::backend::try_candidates(&label, &candidates, |cand| self.load_on(cand))
                .with_context(|| format!("loading {}", self.cfg.load.path.display()))?;

        let dev = engine.backend.info().device.clone();
        let (in_w, in_h) = engine.backend.info().input_size;
        let cap = if self.cfg.queue_size > 0 {
            self.cfg.queue_size
        } else {
            auto_queue_size(self.cfg.request_timeout, warmup_ms)
        };
        self.soft_capacity.store(cap, Ordering::Relaxed);
        if let Ok(mut d) = self.device.write() {
            *d = Some(dev.clone());
        }
        self.metrics
            .set_loaded(dev.actual.clone(), engine.provider.clone());
        info!(
            model = %name,
            family = %engine.family.kind(),
            device = %dev.actual,
            spec = %dev.spec,
            provider = %engine.provider,
            fell_back = dev.fell_back,
            input = %format!("{in_w}x{in_h}"),
            classes = self.cfg.classes.len(),
            compile_ms,
            warmup_ms,
            queue = cap,
            "model ready"
        );
        Ok(engine)
    }

    /// Compile (under the runtimes lock), create the request and warm up on one candidate.
    /// Returns the engine with its compile and warm-up times.
    fn load_on(&self, cand: &Candidate) -> Result<(Engine, u64, u64)> {
        let compiled = {
            let mut rt = self.runtimes.lock().unwrap_or_else(|e| e.into_inner());
            rt.compile(cand, &self.cfg.load)?
        };
        let info = compiled.info();
        let provider = info.device.execution_provider();
        let (in_w, in_h) = info.input_size;
        let compile_ms = info.compile_ms;
        let family = crate::model::make_family(
            self.cfg.model.family,
            &info.inputs,
            &info.outputs,
            self.cfg.classes.len(),
        )?;
        let mut pre = Preprocessor::new(in_w, in_h, family.resize_mode());
        let mut backend = compiled.into_backend()?;

        // Warm-up on a mid-gray frame the size of the model input.
        let gray = vec![114u8; in_w as usize * in_h as usize * 3];
        let t = Instant::now();
        let (chw, ctx) = pre.run(&gray, in_w, in_h)?;
        let extra = family.extra_inputs(&ctx);
        backend.infer(chw, &extra).context("warm-up inference")?;
        let warmup_ms = t.elapsed().as_millis() as u64;
        Ok((
            Engine {
                backend,
                family,
                pre,
                provider,
            },
            compile_ms,
            warmup_ms,
        ))
    }

    fn error_response(
        &self,
        msg: String,
        inference_ms: i32,
        process_ms: i32,
    ) -> VisionDetectionResponse {
        let provider = self
            .device
            .read()
            .ok()
            .and_then(|d| d.as_ref().map(|d| d.execution_provider()))
            .unwrap_or_default();
        VisionDetectionResponse {
            command: "detect".into(),
            executionProvider: provider,
            canUseGPU: self.cfg.can_use_gpu,
            inferenceMs: inference_ms,
            processMs: process_ms,
            ..VisionDetectionResponse::error(msg)
        }
    }

    fn process(
        &self,
        eng: &mut Engine,
        req: VisionDetectionRequest,
        enqueued: Instant,
    ) -> VisionDetectionResponse {
        let name = &self.cfg.name;
        let queue_ms = enqueued.elapsed().as_millis() as u64;
        let start = Instant::now();
        let mut infer_d = Duration::ZERO;
        let result = (|| -> Result<(Vec<crate::api::Prediction>, crate::image::RgbImage, [Duration; 3])> {
            let t = Instant::now();
            let img = crate::image::decode(&req.image_data)?;
            let decode_d = t.elapsed();

            let t = Instant::now();
            let (chw, ctx) = eng.pre.run(&img.rgb, img.width, img.height)?;
            let extra = eng.family.extra_inputs(&ctx);
            let pre_d = t.elapsed();

            let t = Instant::now();
            let outputs = eng.backend.infer(chw, &extra)?;
            infer_d = t.elapsed();

            let t = Instant::now();
            let params = PostParams {
                confidence_threshold: effective_threshold(req.min_confidence, self.cfg.confidence_threshold),
                nms_iou: self.cfg.nms_iou,
            };
            let dets = eng.family.postprocess(&outputs, &ctx, &params)?;
            let filter = (!self.cfg.object_filter.is_empty()).then_some(self.cfg.object_filter.as_slice());
            let preds = crate::model::to_predictions(&dets, &ctx, &self.cfg.classes, filter);
            let post_d = t.elapsed();
            Ok((preds, img, [decode_d, pre_d, post_d]))
        })();
        let process_d = start.elapsed();

        match result {
            Ok((predictions, img, [decode_d, pre_d, post_d])) => {
                self.metrics.inference_ms.record(infer_d.as_millis() as u64);
                self.metrics.process_ms.record(process_d.as_millis() as u64);
                debug!(
                    model = %name,
                    image = %req.image_name,
                    size = %format!("{}x{}", img.width, img.height),
                    count = predictions.len(),
                    queue_ms,
                    decode_ms = decode_d.as_secs_f64() * 1e3,
                    preprocess_ms = pre_d.as_secs_f64() * 1e3,
                    inference_ms = infer_d.as_secs_f64() * 1e3,
                    postprocess_ms = post_d.as_secs_f64() * 1e3,
                    process_ms = process_d.as_secs_f64() * 1e3,
                    "detection done"
                );
                if let Some(dir) = &self.cfg.save_image_path {
                    let file_name = if req.image_name.is_empty() || req.image_name == "image.jpg" {
                        format!("{}.jpg", uuid::Uuid::new_v4())
                    } else {
                        req.image_name.clone()
                    };
                    if let Err(e) = crate::image::save_annotated(
                        dir,
                        &file_name,
                        &img,
                        &predictions,
                        self.cfg.save_ref_image,
                    ) {
                        warn!(model = %name, "could not save annotated image: {e:#}");
                    }
                }
                VisionDetectionResponse {
                    success: true,
                    message: format!("Found {} objects", predictions.len()),
                    error: None,
                    count: predictions.len() as i32,
                    predictions,
                    command: "detect".into(),
                    moduleId: crate::MODULE_ID.to_string(),
                    executionProvider: eng.provider.clone(),
                    canUseGPU: self.cfg.can_use_gpu,
                    inferenceMs: ms(infer_d),
                    processMs: ms(process_d),
                    analysisRoundTripMs: 0,
                }
            }
            Err(e) => {
                warn!(model = %name, image = %req.image_name, "detection failed: {e:#}");
                let mut r = self.error_response(format!("{e:#}"), ms(infer_d), ms(process_d));
                r.executionProvider = eng.provider.clone();
                r
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_sizing() {
        let t = Duration::from_secs(15);
        assert_eq!(auto_queue_size(t, 100), 64);
        assert_eq!(auto_queue_size(t, 500), 30);
        assert_eq!(auto_queue_size(t, 0), 64);
        assert_eq!(auto_queue_size(Duration::from_secs(1), 5000), 1);
        assert_eq!(auto_queue_size(Duration::from_millis(1000), 300), 3);
    }

    #[test]
    fn threshold_override() {
        assert_eq!(effective_threshold(0.0, 0.5), 0.5);
        assert_eq!(effective_threshold(0.3, 0.5), 0.3);
        assert_eq!(effective_threshold(-1.0, 0.5), 0.5);
        assert_eq!(effective_threshold(f32::NAN, 0.5), 0.5);
    }

    #[test]
    fn gate_and_cancel() {
        let g = LoadGate::new();
        assert!(!g.is_open());
        let token = CancellationToken::new();
        let g2 = g.clone();
        let t2 = token.clone();
        let h = std::thread::spawn(move || g2.wait(&t2));
        std::thread::sleep(Duration::from_millis(20));
        g.open();
        assert!(h.join().unwrap());

        let g = LoadGate::new();
        let t3 = token.clone();
        let g3 = g.clone();
        let h = std::thread::spawn(move || g3.wait(&t3));
        token.cancel();
        assert!(!h.join().unwrap());
    }

    #[test]
    fn failed_handle() {
        let m = Arc::new(ModelMetrics::new("x", "CPU", ""));
        let h = WorkerHandle::failed("x", "missing file", m);
        assert_eq!(h.state.get(), ModelState::Failed("missing file".into()));
        assert!(h.is_full());
        assert!(!h.accepts_while_initializing());
        assert!(h.load_gate().is_open());
        h.join();
    }
}
