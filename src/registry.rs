//! Loads every enabled model through the shared [`Runtimes`] (one OpenVINO `Core`, one ONNX
//! Runtime library) and owns their worker threads.
//! Disabled models (`enabled: false`) stay in the config but get no worker and are not served.
//!
//! Workers are spawned in config order; each waits for its predecessor's [`LoadGate`] before
//! locking the core, so compile order is deterministic and only one model compiles at a time.

use crate::backend::{CoreOptions, DeviceSpec, LoadRequest, OrtOptions, Runtimes, Selection};
use crate::config::{Config, ModelConfig};
use crate::metrics::{Metrics, ModelGauges, ModelMetrics};
use crate::worker::{LoadGate, WorkerConfig, WorkerHandle, spawn_worker_after};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

pub use crate::backend::RuntimeInfo;

pub struct ModelRegistry {
    workers: Vec<WorkerHandle>,
    /// Lowercase name -> index into `workers`.
    by_name: HashMap<String, usize>,
    default_idx: Option<usize>,
    pub runtime_info: RuntimeInfo,
    /// The runtimes the workers load through (None for registries built from handles).
    runtimes: Option<Arc<Mutex<Runtimes>>>,
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

/// Build the per-model `LoadRequest` (the device spec comes from `Config::device_spec_for`).
pub fn load_request(config: &Config, m: &ModelConfig, path: PathBuf) -> LoadRequest {
    LoadRequest {
        path,
        requested: config.device_for(m),
        gpu_precision: m.gpu_precision.clone(),
    }
}

/// Index into the enabled models of the model serving `/v1/vision/detection`: `default_model`
/// matched with the same normalization as `/v1/vision/custom/{model}` (case-insensitive,
/// `.onnx`/`.xml` ignored), else the first enabled model. None when no model is enabled.
fn resolve_default(config: &Config) -> Option<usize> {
    let idx = config.effective_default_index()?;
    if let Some(name) = config
        .default_model
        .as_deref()
        .filter(|n| !n.trim().is_empty())
    {
        match config.default_model_index() {
            None => {
                warn!("default_model '{name}' is not configured; using the first enabled model")
            }
            Some(i) if !config.models[i].enabled => {
                warn!("default_model '{name}' is disabled; using the first enabled model")
            }
            Some(_) => {}
        }
    }
    Some(config.models[..idx].iter().filter(|m| m.enabled).count())
}

/// Label of the first runnable load candidate of `device` for `model` (the option label from
/// `backend::select`, e.g. "ONNX Runtime CoreML (Apple M1)"); None when nothing can run.
/// `no_coreml` skips CoreML candidates (a model family the CoreML provider rejects).
fn planned_label(
    runtimes: &Runtimes,
    device: &DeviceSpec,
    model: &Path,
    no_coreml: bool,
) -> Option<String> {
    let sel = runtimes.selection(Some(model));
    runtimes
        .plan(device, model)
        .into_iter()
        .filter(|c| !(no_coreml && c.device.target == crate::backend::spec::Target::CoreMl))
        .find_map(|c| sel.option(&c.device).filter(|o| o.runnable))
        .map(|o| o.label.clone())
}

impl ModelRegistry {
    /// Create the `Core` and spawn one worker per enabled model. Returns as soon as the
    /// workers are spawned; models compile in the background (see `WorkerHandle::state`).
    /// An empty `models` list is an error; models that are all disabled give an empty registry
    /// (the web UI stays up so a model can be enabled).
    pub fn start(config: &Config, metrics: &Metrics, shutdown: CancellationToken) -> Result<Self> {
        Self::start_with(config, metrics, shutdown, None)
    }

    /// [`Self::start`] for a generation planned by the [`Provisioner`]
    /// (`crate::resources::provision`): a model blocked on a download either loads on what is
    /// runnable now (its file is there and only a runtime is missing) or waits with progress in
    /// its state; with `auto_download: false` it is `Failed` with the command to run. Missing
    /// runtimes are not fatal while something is being provisioned.
    ///
    /// [`Provisioner`]: crate::resources::provision::Provisioner
    pub fn start_with(
        config: &Config,
        metrics: &Metrics,
        shutdown: CancellationToken,
        provision: Option<&crate::resources::provision::Provision>,
    ) -> Result<Self> {
        if config.models.is_empty() {
            bail!(
                "no models configured: pass `--model <path> [--family yolo5|yolo8|yolo26|rtdetr]`, \
                 add entries to `models` in the config file, or fetch one with \
                 `blue-onyx-prism download-models --name IPcam-general`"
            );
        }

        // Validate names (of all entries, enabled or not) before touching OpenVINO.
        let mut seen = std::collections::HashSet::new();
        for m in &config.models {
            let key = normalize_name(&m.effective_name());
            if !seen.insert(key.clone()) {
                bail!(
                    "duplicate model name '{key}' in config; set a distinct `name` for each entry"
                );
            }
        }
        let enabled: Vec<&ModelConfig> = config.enabled_models().collect();
        let by_name: HashMap<String, usize> = enabled
            .iter()
            .enumerate()
            .map(|(i, m)| (normalize_name(&m.effective_name()), i))
            .collect();
        let disabled = config.models.len() - enabled.len();
        if enabled.is_empty() {
            warn!(
                "all {disabled} configured models are disabled; nothing will be served until one \
                 is enabled on the Config page (or with `\"enabled\": true` in the config file)"
            );
        } else if disabled > 0 {
            info!(disabled, "skipping disabled models");
        }

        let opts = CoreOptions {
            cache_dir: config.cache_dir_path(),
            intra_threads: config.intra_threads,
            openvino_dir: config.openvino_dir_effective(),
        };
        let ort_opts: OrtOptions = config.ort_options();
        let runtimes = Runtimes::new_with(&opts, &ort_opts);
        if let Err(e) = runtimes.require_any() {
            match provision.filter(|p| p.has_needs()) {
                // Fresh install: the runtimes are being downloaded (or must be fetched); serve
                // HTTP with the models waiting / failed instead of exiting.
                Some(_) => warn!("{e:#} (the missing runtime is being provisioned)"),
                None => return Err(e),
            }
        }
        let runtime_info = runtimes.info();
        if runtimes.openvino().is_some() {
            info!(
                version = %runtime_info.openvino_version,
                devices = ?runtime_info.available_devices,
                cache_dir = ?opts.cache_dir,
                "OpenVINO core ready"
            );
        } else {
            warn!(
                "OpenVINO unavailable: {}",
                runtimes.openvino_error().unwrap_or("not initialized")
            );
        }
        match runtimes.onnxruntime_version() {
            Some(v) => info!(version = %v, "ONNX Runtime ready"),
            None => info!(
                "ONNX Runtime unavailable: {}",
                runtimes.onnxruntime_error().unwrap_or("not initialized")
            ),
        }
        let selection = runtimes.selection(None);
        info!(
            auto = %crate::backend::select::format_auto(&selection),
            can_use_gpu = runtime_info.has_gpu,
            "device options probed"
        );
        let best_cpu = runtimes.best_cpu();
        // Where each enabled model is headed (label of its first runnable load candidate),
        // computed before the workers can hold the runtimes lock: shown as "<label> (loading)".
        let planned_labels: Vec<Option<String>> = enabled
            .iter()
            .map(|m| {
                let device = if config.force_cpu {
                    best_cpu
                } else {
                    config.device_spec_for(m).ok()?
                };
                // RT-DETR never loads on CoreML (the ORT backend refuses it at load time).
                let no_coreml = m.family == crate::model::ModelFamilyKind::RtDetr;
                planned_label(&runtimes, &device, &config.data_path(&m.path), no_coreml)
            })
            .collect();
        let runtimes = Arc::new(Mutex::new(runtimes));

        let mut workers = Vec::with_capacity(enabled.len());
        let mut prev: Option<LoadGate> = None;
        for (mi, &m) in enabled.iter().enumerate() {
            let name = m.effective_name();
            let path = config.data_path(&m.path);
            let load = load_request(config, m, path.clone());
            let mm = Arc::new(ModelMetrics::new(
                name.clone(),
                load.requested.clone(),
                String::new(),
            ));
            if let Ok(mut g) = metrics.models.write() {
                g.push(mm.clone());
            }

            // Downloads this model waits for (phase 7.3).
            if let Some(p) = provision
                && let Some(msg) = p.manual_failure(&name)
            {
                warn!(model = %name, "{msg}");
                workers.push(WorkerHandle::failed(name, msg, mm));
                continue;
            }
            let mut wait = provision.and_then(|p| p.wait_for(&name));
            if let Some(w) = &wait {
                let only_runtimes = w.blocks().iter().all(|b| b.is_runtime());
                if only_runtimes && path.is_file() {
                    let device = config.device_spec_for(m).unwrap_or(best_cpu);
                    let rt = runtimes.lock().unwrap_or_else(|e| e.into_inner());
                    let sel = rt.selection(Some(&path));
                    let interim = rt
                        .plan(&device, &path)
                        .into_iter()
                        .find(|c| sel.option(&c.device).is_some_and(|o| o.runnable));
                    drop(rt);
                    if let Some(c) = interim {
                        let ids: Vec<&str> = w.blocks().iter().map(|b| b.resource.id).collect();
                        info!(
                            model = %name,
                            "loading on {} while {} downloads; switching after it is installed",
                            c.device,
                            ids.join(", ")
                        );
                        wait = None;
                    }
                }
            }

            let setup = (|| -> Result<(DeviceSpec, Vec<String>)> {
                // `force_cpu`: the best CPU option (OpenVINO, else ONNX Runtime).
                let device = if config.force_cpu {
                    best_cpu
                } else {
                    config.device_spec_for(m)?
                };
                if wait.is_some() {
                    // Class names are read after the files arrive (in the worker).
                    return Ok((device, Vec::new()));
                }
                if !path.is_file() {
                    bail!(
                        "model file not found: {} (download it with `download-models` or fix `path`)",
                        path.display()
                    );
                }
                let classes_path = m.classes.as_deref().map(|c| config.data_path(c));
                Ok((device, resolve_class_names(&path, classes_path.as_deref())?))
            })();
            let (device, classes) = match setup {
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
                classes: m.classes.as_deref().map(|c| config.data_path(c)),
                ..m.clone()
            };
            let cfg = WorkerConfig {
                name: name.clone(),
                device,
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
                can_use_gpu: runtime_info.has_gpu,
                model: resolved,
                wait,
            };
            let mut handle =
                spawn_worker_after(runtimes.clone(), cfg, mm, shutdown.clone(), prev.take());
            handle.set_planned(device, planned_labels.get(mi).cloned().flatten());
            prev = Some(handle.load_gate());
            workers.push(handle);
        }

        let default_idx = resolve_default(config);
        if let Some(i) = default_idx {
            info!(model = %workers[i].name, "default model for /v1/vision/detection");
        }

        Ok(Self {
            workers,
            by_name,
            default_idx,
            runtime_info,
            runtimes: Some(runtimes),
        })
    }

    /// Registry over already constructed handles (no OpenVINO involved). Used by tests that
    /// exercise the HTTP layer with e.g. [`WorkerHandle::failed`] handles.
    pub fn from_handles(
        workers: Vec<WorkerHandle>,
        default_idx: Option<usize>,
        runtime_info: RuntimeInfo,
    ) -> Self {
        let by_name = workers
            .iter()
            .enumerate()
            .map(|(i, w)| (normalize_name(&w.name), i))
            .collect();
        let default_idx = default_idx.filter(|&i| i < workers.len());
        Self {
            workers,
            by_name,
            default_idx,
            runtime_info,
            runtimes: None,
        }
    }

    /// The shared runtimes (None for registries built with [`Self::from_handles`]). Lock briefly:
    /// workers hold the lock while compiling.
    pub fn runtimes(&self) -> Option<&Arc<Mutex<Runtimes>>> {
        self.runtimes.as_ref()
    }

    /// Device options and the `auto` ranking for `model` (None: any model), from the shared
    /// runtimes. None for registries built with [`Self::from_handles`].
    pub fn selection(&self, model: Option<&Path>) -> Option<Selection> {
        let rt = self.runtimes.as_ref()?;
        let rt = rt.lock().unwrap_or_else(|e| e.into_inner());
        Some(rt.selection(model))
    }

    /// Live per-model gauges (state, device, queue) for `/prometheus`, in config order.
    pub fn gauges(&self) -> Vec<ModelGauges> {
        self.workers.iter().map(WorkerHandle::gauges).collect()
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

    /// Names of the served (enabled) models in config order.
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
        use crate::backend::plan_candidates;
        let gpu = ["CPU".to_string(), "GPU".to_string()];
        let plan = |c: &Config, m: &ModelConfig| -> Vec<String> {
            plan_candidates(&c.device_spec_for(m).unwrap(), &gpu)
                .iter()
                .map(|c| c.device.to_string())
                .collect()
        };
        let mut c = Config {
            device: "GPU".into(),
            ..Config::default()
        };
        let m = ModelConfig::default();
        let r = load_request(&c, &m, "m.xml".into());
        assert_eq!(r.requested, "GPU");
        assert_eq!(plan(&c, &m), ["openvino:gpu", "openvino:cpu"]);
        c.force_cpu = true;
        let r = load_request(&c, &m, "m.xml".into());
        assert_eq!(r.requested, "CPU");
        assert_eq!(plan(&c, &m), ["openvino:cpu"]);
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
    fn default_model_resolution() {
        let mut c = Config::default();
        for p in ["models/yolo26s.xml", "models/IPcam-general.onnx"] {
            c.models.push(ModelConfig {
                path: p.into(),
                ..Default::default()
            });
        }
        assert_eq!(resolve_default(&c), Some(0));
        for name in ["IPcam-general", "ipcam-GENERAL.onnx", " ipcam-general "] {
            c.default_model = Some(name.into());
            assert_eq!(resolve_default(&c), Some(1), "{name}");
        }
        c.default_model = Some("yolo26s.XML".into());
        assert_eq!(resolve_default(&c), Some(0));
        c.default_model = Some("nope".into());
        assert_eq!(resolve_default(&c), Some(0));
        c.default_model = Some(String::new());
        assert_eq!(resolve_default(&c), Some(0));
        c.models.clear();
        assert_eq!(resolve_default(&c), None);
    }

    #[test]
    fn default_model_resolution_over_enabled_models() {
        let mut c = Config::default();
        for (p, enabled) in [("a.onnx", false), ("b.onnx", true), ("c.onnx", true)] {
            c.models.push(ModelConfig {
                path: p.into(),
                enabled,
                ..Default::default()
            });
        }
        // Indices are into the enabled models (= the workers): b -> 0, c -> 1.
        assert_eq!(resolve_default(&c), Some(0));
        c.default_model = Some("c".into());
        assert_eq!(resolve_default(&c), Some(1));
        // Default names a disabled model: first enabled one.
        c.default_model = Some("a".into());
        assert_eq!(resolve_default(&c), Some(0));
        for m in &mut c.models {
            m.enabled = false;
        }
        assert_eq!(resolve_default(&c), None);
    }

    #[test]
    fn all_disabled_still_validates_names() {
        let mut c = Config::default();
        for p in ["a/x.onnx", "b/X.xml"] {
            c.models.push(ModelConfig {
                path: p.into(),
                enabled: false,
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
