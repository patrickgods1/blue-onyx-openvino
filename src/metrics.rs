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

#[derive(Debug)]
pub struct ModelMetrics {
    pub name: String,
    pub device: String,
    pub execution_provider: String,
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
        Self {
            name: name.into(),
            device: device.into(),
            execution_provider: execution_provider.into(),
            requests: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            inference_ms: Stat::new(),
            process_ms: Stat::new(),
            round_trip_ms: Stat::new(),
        }
    }
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

/// Render all metrics in the Prometheus text exposition format.
pub fn render_prometheus(m: &Metrics) -> String {
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
        let text = render_prometheus(&m);
        assert!(text.contains("blue_onyx_openvino_requests_total{model=\"ipcam\\\"x\"} 5"));
        assert!(text.contains("blue_onyx_openvino_dropped_total{model=\"ipcam\\\"x\"} 1"));
        assert!(text.contains("blue_onyx_openvino_inference_ms_avg{model=\"ipcam\\\"x\"} 10.000"));
        assert!(text.contains("blue_onyx_openvino_inference_ms_min{model=\"ipcam\\\"x\"} 8"));
        assert!(text.contains("blue_onyx_openvino_inference_ms_max{model=\"ipcam\\\"x\"} 12"));
        assert!(text.contains("blue_onyx_openvino_uptime_seconds "));
        assert!(text.contains("# TYPE blue_onyx_openvino_round_trip_ms_avg gauge"));
    }
}
