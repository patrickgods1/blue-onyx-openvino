//! Loads every configured model on one shared OpenVINO `Core` and owns their worker threads.
//!
//! Workers are spawned in config order; each waits for its predecessor's [`LoadGate`] before
//! locking the core, so compile order is deterministic and only one model compiles at a time.

use crate::backend::{CoreOptions, LoadRequest, OvCore};
use crate::config::{Config, ModelConfig};
use crate::metrics::{Metrics, ModelMetrics};
use crate::worker::{LoadGate, WorkerConfig, WorkerHandle, spawn_worker_after};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

/// Facts about the OpenVINO runtime, captured once at startup.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CoreInfo {
    pub openvino_version: String,
    pub available_devices: Vec<String>,
    pub has_gpu: bool,
}

pub struct ModelRegistry {
    workers: Vec<WorkerHandle>,
    /// Lowercase name -> index into `workers`.
    by_name: HashMap<String, usize>,
    default_idx: Option<usize>,
    pub core_info: CoreInfo,
}

/// Lowercase and strip a trailing `.onnx` / `.xml` (Blue Iris sends file stems, users may not).
pub fn normalize_name(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    for ext in [".onnx", ".xml"] {
        if let Some(stripped) = lower.strip_suffix(ext)
            && !stripped.is_empty()
        {
            return stripped.to_string();
        }
    }
    lower
}

/// Class names: explicit `classes` path, else `<model stem>.yaml` beside the model, else COCO-80.
pub fn resolve_class_names(model_path: &Path, classes: Option<&Path>) -> Result<Vec<String>> {
    if let Some(p) = classes {
        return crate::model::classes::load_class_names(p)
            .with_context(|| format!("loading class names from {}", p.display()));
    }
    let sidecar = model_path.with_extension("yaml");
    if sidecar.is_file() {
        return crate::model::classes::load_class_names(&sidecar)
            .with_context(|| format!("loading class names from {}", sidecar.display()));
    }
    Ok(crate::model::classes::coco80())
}

/// Build the per-model `LoadRequest` from the global config.
pub fn load_request(config: &Config, m: &ModelConfig, path: PathBuf) -> LoadRequest {
    let device = config.device_for(m);
    let allow_cpu_fallback = !device.eq_ignore_ascii_case("CPU");
    LoadRequest {
        path,
        device,
        gpu_precision: m.gpu_precision.clone(),
        allow_cpu_fallback,
    }
}

impl ModelRegistry {
    /// Create the `Core` and spawn one worker per configured model. Returns as soon as the
    /// workers are spawned; models compile in the background (see `WorkerHandle::state`).
    pub fn start(config: &Config, metrics: &Metrics, shutdown: CancellationToken) -> Result<Self> {
        if config.models.is_empty() {
            bail!(
                "no models configured: pass `--model <path> [--family yolo5|yolo8|yolo26|rtdetr]`, \
                 add entries to `models` in the config file, or fetch one with \
                 `blue-onyx-openvino download-models --name IPcam-general`"
            );
        }

        // Validate names before touching OpenVINO.
        let mut by_name = HashMap::new();
        for (i, m) in config.models.iter().enumerate() {
            let key = normalize_name(&m.effective_name());
            if by_name.insert(key.clone(), i).is_some() {
                bail!(
                    "duplicate model name '{key}' in config; set a distinct `name` for each entry"
                );
            }
        }

        let opts = CoreOptions {
            cache_dir: config.cache_dir_path(),
            intra_threads: config.intra_threads,
            openvino_dir: config.openvino_dir.as_deref().map(crate::resolve_path),
        };
        let core = OvCore::new(&opts).context("initializing OpenVINO")?;
        let core_info = CoreInfo {
            openvino_version: core.openvino_version(),
            available_devices: core.available_devices().to_vec(),
            has_gpu: core.has_gpu(),
        };
        info!(
            version = %core_info.openvino_version,
            devices = ?core_info.available_devices,
            cache_dir = ?opts.cache_dir,
            "OpenVINO core ready"
        );
        let core = Arc::new(Mutex::new(core));

        let mut workers = Vec::with_capacity(config.models.len());
        let mut prev: Option<LoadGate> = None;
        for m in &config.models {
            let name = m.effective_name();
            let path = crate::resolve_path(&m.path);
            let load = load_request(config, m, path.clone());
            let mm = Arc::new(ModelMetrics::new(
                name.clone(),
                load.device.clone(),
                String::new(),
            ));
            if let Ok(mut g) = metrics.models.write() {
                g.push(mm.clone());
            }

            let setup = (|| -> Result<Vec<String>> {
                if !path.is_file() {
                    bail!(
                        "model file not found: {} (download it with `download-models` or fix `path`)",
                        path.display()
                    );
                }
                let classes_path = m.classes.as_deref().map(crate::resolve_path);
                resolve_class_names(&path, classes_path.as_deref())
            })();
            let classes = match setup {
                Ok(c) => c,
                Err(e) => {
                    let msg = format!("{e:#}");
                    warn!(model = %name, "{msg}");
                    workers.push(WorkerHandle::failed(name, msg, mm));
                    continue;
                }
            };

            let resolved = ModelConfig {
                path,
                classes: m.classes.as_deref().map(crate::resolve_path),
                ..m.clone()
            };
            let cfg = WorkerConfig {
                name: name.clone(),
                load,
                classes,
                object_filter: m
                    .object_filter
                    .clone()
                    .unwrap_or_else(|| config.object_filter.clone()),
                confidence_threshold: m
                    .confidence_threshold
                    .unwrap_or(config.confidence_threshold),
                nms_iou: config.nms_iou,
                request_timeout: config.request_timeout(),
                queue_size: config.worker_queue_size,
                save_image_path: config.save_image_path.as_deref().map(crate::resolve_path),
                save_ref_image: config.save_ref_image,
                lazy: m.lazy,
                can_use_gpu: core_info.has_gpu,
                model: resolved,
            };
            let handle = spawn_worker_after(core.clone(), cfg, mm, shutdown.clone(), prev.take());
            prev = Some(handle.load_gate());
            workers.push(handle);
        }

        let default_idx = match (&config.default_model, config.default_model_index()) {
            (_, Some(i)) => Some(i),
            (Some(name), None) => {
                warn!("default_model '{name}' is not configured; using the first model");
                Some(0)
            }
            (None, None) => None,
        };
        if let Some(i) = default_idx {
            info!(model = %workers[i].name, "default model for /v1/vision/detection");
        }

        Ok(Self {
            workers,
            by_name,
            default_idx,
            core_info,
        })
    }

    pub fn default_model(&self) -> Option<&WorkerHandle> {
        self.default_idx.and_then(|i| self.workers.get(i))
    }

    /// Case-insensitive lookup; a trailing `.onnx` / `.xml` is ignored.
    pub fn by_name(&self, name: &str) -> Option<&WorkerHandle> {
        self.by_name
            .get(&normalize_name(name))
            .and_then(|&i| self.workers.get(i))
    }

    /// Model names in config order.
    pub fn names(&self) -> Vec<String> {
        self.workers.iter().map(|w| w.name.clone()).collect()
    }

    pub fn workers(&self) -> &[WorkerHandle] {
        &self.workers
    }

    /// Drop all senders and join the worker threads. Cancel the shutdown token first for a
    /// prompt exit; otherwise each worker exits once its channel disconnects.
    pub fn shutdown(self) {
        for w in self.workers {
            let name = w.name.clone();
            w.join();
            info!(model = %name, "worker stopped");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_normalization() {
        assert_eq!(normalize_name("IPcam-General"), "ipcam-general");
        assert_eq!(normalize_name("IPcam-general.onnx"), "ipcam-general");
        assert_eq!(normalize_name("yolo26s.XML"), "yolo26s");
        assert_eq!(normalize_name(".onnx"), ".onnx");
        assert_eq!(normalize_name(" a.b "), "a.b");
    }

    #[test]
    fn load_request_fallback_rules() {
        let mut c = Config::default();
        let m = ModelConfig::default();
        let r = load_request(&c, &m, "m.xml".into());
        assert_eq!(r.device, "GPU");
        assert!(r.allow_cpu_fallback);
        c.force_cpu = true;
        let r = load_request(&c, &m, "m.xml".into());
        assert_eq!(r.device, "CPU");
        assert!(!r.allow_cpu_fallback);
    }

    #[test]
    fn empty_config_is_an_error() {
        let c = Config::default();
        let m = Metrics::new("0");
        let err = ModelRegistry::start(&c, &m, CancellationToken::new())
            .err()
            .expect("must fail");
        assert!(format!("{err:#}").contains("--model"));
    }

    #[test]
    fn duplicate_names_rejected() {
        let mut c = Config::default();
        for p in ["a/x.onnx", "b/X.xml"] {
            c.models.push(ModelConfig {
                path: p.into(),
                ..Default::default()
            });
        }
        let m = Metrics::new("0");
        let err = ModelRegistry::start(&c, &m, CancellationToken::new())
            .err()
            .expect("must fail");
        assert!(format!("{err:#}").contains("duplicate"));
    }

    #[test]
    fn sidecar_classes_absent_falls_back() {
        let dir = std::env::temp_dir().join(format!("bo_reg_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let names = resolve_class_names(&dir.join("m.onnx"), None).unwrap();
        assert_eq!(names, crate::model::classes::coco80());
    }
}
