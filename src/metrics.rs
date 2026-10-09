//! Request/latency counters and Prometheus text rendering.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Lock-free running statistic in milliseconds.
#[derive(Debug)]
pub struct Stat {
    count: AtomicU64,
    total: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
}

impl Default for Stat {
    fn default() -> Self {
        Self::new()
    }
}

impl Stat {
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }

    pub fn record(&self, ms: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total.fetch_add(ms, Ordering::Relaxed);
        self.min.fetch_min(ms, Ordering::Relaxed);
        self.max.fetch_max(ms, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Average in ms (0.0 when nothing was recorded).
    pub fn avg(&self) -> f64 {
        let c = self.count();
        if c == 0 {
            0.0
        } else {
            self.total.load(Ordering::Relaxed) as f64 / c as f64
        }
    }

    /// Minimum in ms (0 when nothing was recorded).
    pub fn min(&self) -> u64 {
        if self.count() == 0 {
            0
        } else {
            self.min.load(Ordering::Relaxed)
        }
    }

    pub fn max(&self) -> u64 {
        self.max.load(Ordering::Relaxed)
    }
}

/// Per-model counters plus the device/provider strings, which start as the configured device
/// and an empty provider and are updated by the worker once the model has been compiled.
#[derive(Debug)]
pub struct ModelMetrics {
    pub name: String,
    /// Device requested in the config (`GPU`, `GPU.1`, `CPU`).
    pub requested_device: String,
    device: RwLock<String>,
    execution_provider: RwLock<String>,
    pub requests: AtomicU64,
    pub dropped: AtomicU64,
    pub inference_ms: Stat,
    pub process_ms: Stat,
    pub round_trip_ms: Stat,
}

impl ModelMetrics {
    pub fn new(
        name: impl Into<String>,
        device: impl Into<String>,
        execution_provider: impl Into<String>,
    ) -> Self {
        let device = device.into();
        Self {
            name: name.into(),
            requested_device: device.clone(),
            device: RwLock::new(device),
            execution_provider: RwLock::new(execution_provider.into()),
            requests: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            inference_ms: Stat::new(),
            process_ms: Stat::new(),
            round_trip_ms: Stat::new(),
        }
    }

    /// Device the model actually runs on (the requested device until it is loaded).
    pub fn device(&self) -> String {
        read_string(&self.device)
    }

    /// `executionProvider` string of the loaded model (empty until it is loaded).
    pub fn execution_provider(&self) -> String {
        read_string(&self.execution_provider)
    }

    /// Record where the model ended up after compilation (called by the worker).
    pub fn set_loaded(&self, device: impl Into<String>, execution_provider: impl Into<String>) {
        write_string(&self.device, device.into());
        write_string(&self.execution_provider, execution_provider.into());
    }
}

fn read_string(l: &RwLock<String>) -> String {
    l.read()
        .map(|g| g.clone())
        .unwrap_or_else(|e| e.into_inner().clone())
}

fn write_string(l: &RwLock<String>, v: String) {
    match l.write() {
        Ok(mut g) => *g = v,
        Err(e) => *e.into_inner() = v,
    }
}

/// Lifecycle state of a model as exported to Prometheus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStateLabel {
    /// Compiling (or waiting for its turn to compile).
    Initializing,
    Ready,
    Failed,
    /// `lazy: true` and not requested yet.
    Lazy,
}

impl ModelStateLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelStateLabel::Initializing => "initializing",
            ModelStateLabel::Ready => "ready",
            ModelStateLabel::Failed => "failed",
            ModelStateLabel::Lazy => "lazy",
        }
    }
}

/// Point-in-time per-model values that live outside [`ModelMetrics`] (worker state, queue).
/// Built by the registry for each `/prometheus` scrape.
#[derive(Debug, Clone)]
pub struct ModelGauges {
    pub name: String,
    pub device: String,
    pub provider: String,
    pub state: ModelStateLabel,
    pub queue_length: usize,
    pub queue_capacity: usize,
}

#[derive(Debug)]
pub struct Metrics {
    pub started: Instant,
    pub version: String,
    pub models: RwLock<Vec<Arc<ModelMetrics>>>,
}

impl Metrics {
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            started: Instant::now(),
            version: version.into(),
            models: RwLock::new(Vec::new()),
        }
    }

    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }
}

fn escape_label(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

const PREFIX: &str = "blue_onyx_openvino_";

/// Render all metrics in the Prometheus text exposition format. `gauges` adds the per-model
/// `model_info`, `model_ready`, `queue_length` and `queue_capacity` series.
pub fn render_prometheus(m: &Metrics, gauges: &[ModelGauges]) -> String {
    let models: Vec<Arc<ModelMetrics>> = m
        .models
        .read()
        .map(|g| g.clone())
        .unwrap_or_else(|e| e.into_inner().clone());
    let mut out = String::new();

    let _ = writeln!(out, "# HELP {PREFIX}info Build information.");
    let _ = writeln!(out, "# TYPE {PREFIX}info gauge");
    let _ = writeln!(
        out,
        "{PREFIX}info{{version=\"{}\"}} 1",
        escape_label(&m.version)
    );
    let _ = writeln!(
        out,
        "# HELP {PREFIX}uptime_seconds Seconds since the service started."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}uptime_seconds gauge");
    let _ = writeln!(out, "{PREFIX}uptime_seconds {}", m.uptime().as_secs());

    let counter = |out: &mut String, name: &str, help: &str, f: &dyn Fn(&ModelMetrics) -> u64| {
        let _ = writeln!(out, "# HELP {PREFIX}{name} {help}");
        let _ = writeln!(out, "# TYPE {PREFIX}{name} counter");
        for mm in &models {
            let _ = writeln!(
                out,
                "{PREFIX}{name}{{model=\"{}\"}} {}",
                escape_label(&mm.name),
                f(mm)
            );
        }
    };
    counter(
        &mut out,
        "requests_total",
        "Detection requests received.",
        &|mm| mm.requests.load(Ordering::Relaxed),
    );
    counter(
        &mut out,
        "dropped_total",
        "Requests dropped (queue full or timeout).",
        &|mm| mm.dropped.load(Ordering::Relaxed),
    );

    type Getter = fn(&ModelMetrics) -> &Stat;
    let stats: [(&str, &str, Getter); 3] = [
        ("inference_ms", "Model inference time in ms", |mm| {
            &mm.inference_ms
        }),
        ("process_ms", "Total processing time in ms", |mm| {
            &mm.process_ms
        }),
        ("round_trip_ms", "Request round trip time in ms", |mm| {
            &mm.round_trip_ms
        }),
    ];
    for (base, help, get) in stats {
        for suffix in ["avg", "min", "max"] {
            let name = format!("{base}_{suffix}");
            let _ = writeln!(out, "# HELP {PREFIX}{name} {help} ({suffix}).");
            let _ = writeln!(out, "# TYPE {PREFIX}{name} gauge");
            for mm in &models {
                let s = get(mm);
                let label = escape_label(&mm.name);
                let value = match suffix {
                    "avg" => format!("{:.3}", s.avg()),
                    "min" => s.min().to_string(),
                    _ => s.max().to_string(),
                };
                let _ = writeln!(out, "{PREFIX}{name}{{model=\"{label}\"}} {value}");
            }
        }
    }

    let gauge_header = |out: &mut String, name: &str, help: &str| {
        let _ = writeln!(out, "# HELP {PREFIX}{name} {help}");
        let _ = writeln!(out, "# TYPE {PREFIX}{name} gauge");
    };
    gauge_header(
        &mut out,
        "model_info",
        "Model device, execution provider and state (initializing|ready|failed|lazy); value is always 1.",
    );
    for g in gauges {
        let _ = writeln!(
            out,
            "{PREFIX}model_info{{model=\"{}\",device=\"{}\",provider=\"{}\",state=\"{}\"}} 1",
            escape_label(&g.name),
            escape_label(&g.device),
            escape_label(&g.provider),
            g.state.as_str()
        );
    }
    gauge_header(
        &mut out,
        "model_ready",
        "1 when the model is compiled and serving requests, else 0.",
    );
    for g in gauges {
        let ready = u8::from(g.state == ModelStateLabel::Ready);
        let _ = writeln!(
            out,
            "{PREFIX}model_ready{{model=\"{}\"}} {ready}",
            escape_label(&g.name)
        );
    }
    gauge_header(
        &mut out,
        "queue_length",
        "Requests waiting in the worker queue.",
    );
    for g in gauges {
        let _ = writeln!(
            out,
            "{PREFIX}queue_length{{model=\"{}\"}} {}",
            escape_label(&g.name),
            g.queue_length
        );
    }
    gauge_header(
        &mut out,
        "queue_capacity",
        "Effective worker queue depth limit (sized after warm-up).",
    );
    for g in gauges {
        let _ = writeln!(
            out,
            "{PREFIX}queue_capacity{{model=\"{}\"}} {}",
            escape_label(&g.name),
            g.queue_capacity
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_basics() {
        let s = Stat::new();
        assert_eq!((s.count(), s.min(), s.max()), (0, 0, 0));
        assert_eq!(s.avg(), 0.0);
        s.record(10);
        s.record(30);
        s.record(20);
        assert_eq!(s.count(), 3);
        assert_eq!(s.min(), 10);
        assert_eq!(s.max(), 30);
        assert!((s.avg() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn prometheus_rendering() {
        let m = Metrics::new("1.2.3");
        let mm = Arc::new(ModelMetrics::new("ipcam\"x", "CPU", "OpenVINO CPU"));
        mm.requests.fetch_add(5, Ordering::Relaxed);
        mm.dropped.fetch_add(1, Ordering::Relaxed);
        mm.inference_ms.record(12);
        mm.inference_ms.record(8);
        m.models.write().unwrap().push(mm);
        let text = render_prometheus(&m, &[]);
        assert!(text.contains("blue_onyx_openvino_requests_total{model=\"ipcam\\\"x\"} 5"));
        assert!(text.contains("blue_onyx_openvino_dropped_total{model=\"ipcam\\\"x\"} 1"));
        assert!(text.contains("blue_onyx_openvino_inference_ms_avg{model=\"ipcam\\\"x\"} 10.000"));
        assert!(text.contains("blue_onyx_openvino_inference_ms_min{model=\"ipcam\\\"x\"} 8"));
        assert!(text.contains("blue_onyx_openvino_inference_ms_max{model=\"ipcam\\\"x\"} 12"));
        assert!(text.contains("blue_onyx_openvino_uptime_seconds "));
        assert!(text.contains("# TYPE blue_onyx_openvino_round_trip_ms_avg gauge"));
        assert!(text.contains("# TYPE blue_onyx_openvino_model_info gauge"));
        assert!(!text.contains("blue_onyx_openvino_model_info{"));
    }

    #[test]
    fn device_and_provider_update_after_load() {
        let mm = ModelMetrics::new("m", "GPU", "");
        assert_eq!(mm.requested_device, "GPU");
        assert_eq!(mm.device(), "GPU");
        assert_eq!(mm.execution_provider(), "");
        mm.set_loaded("CPU", "OpenVINO CPU (Apple M2, fallback)");
        assert_eq!(mm.requested_device, "GPU");
        assert_eq!(mm.device(), "CPU");
        assert_eq!(mm.execution_provider(), "OpenVINO CPU (Apple M2, fallback)");
    }

    #[test]
    fn prometheus_model_gauges() {
        let m = Metrics::new("0.1.0");
        let gauges = vec![
            ModelGauges {
                name: "yolo26s".into(),
                device: "GPU".into(),
                provider: "OpenVINO GPU (Intel(R) UHD Graphics 630 (iGPU))".into(),
                state: ModelStateLabel::Ready,
                queue_length: 2,
                queue_capacity: 30,
            },
            ModelGauges {
                name: "ipcam-general".into(),
                device: "CPU".into(),
                provider: String::new(),
                state: ModelStateLabel::Initializing,
                queue_length: 0,
                queue_capacity: 64,
            },
            ModelGauges {
                name: "rt\"detr".into(),
                device: "CPU".into(),
                provider: String::new(),
                state: ModelStateLabel::Lazy,
                queue_length: 0,
                queue_capacity: 64,
            },
            ModelGauges {
                name: "broken".into(),
                device: "CPU".into(),
                provider: String::new(),
                state: ModelStateLabel::Failed,
                queue_length: 0,
                queue_capacity: 0,
            },
        ];
        let text = render_prometheus(&m, &gauges);
        let has = |line: &str| {
            assert!(
                text.lines().any(|l| l == line),
                "missing line {line:?} in:\n{text}"
            )
        };
        has("# TYPE blue_onyx_openvino_model_info gauge");
        has(
            "blue_onyx_openvino_model_info{model=\"yolo26s\",device=\"GPU\",provider=\"OpenVINO GPU (Intel(R) UHD Graphics 630 (iGPU))\",state=\"ready\"} 1",
        );
        has(
            "blue_onyx_openvino_model_info{model=\"ipcam-general\",device=\"CPU\",provider=\"\",state=\"initializing\"} 1",
        );
        has(
            "blue_onyx_openvino_model_info{model=\"rt\\\"detr\",device=\"CPU\",provider=\"\",state=\"lazy\"} 1",
        );
        has(
            "blue_onyx_openvino_model_info{model=\"broken\",device=\"CPU\",provider=\"\",state=\"failed\"} 1",
        );
        has("blue_onyx_openvino_model_ready{model=\"yolo26s\"} 1");
        has("blue_onyx_openvino_model_ready{model=\"ipcam-general\"} 0");
        has("blue_onyx_openvino_model_ready{model=\"broken\"} 0");
        has("blue_onyx_openvino_queue_length{model=\"yolo26s\"} 2");
        has("blue_onyx_openvino_queue_capacity{model=\"yolo26s\"} 30");
        has("blue_onyx_openvino_queue_capacity{model=\"broken\"} 0");
        has("# TYPE blue_onyx_openvino_queue_length gauge");
        has("# TYPE blue_onyx_openvino_queue_capacity gauge");
        has("# TYPE blue_onyx_openvino_model_ready gauge");
        // Every sample line is `name{labels} value` with a numeric value.
        for l in text.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "bad sample line {l:?}");
        }
    }
}
