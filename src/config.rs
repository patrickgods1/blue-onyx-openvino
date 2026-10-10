//! Service configuration: a JSON file next to the executable, merged with CLI flags.
//! Merge rule (copied from blue-onyx): a CLI value overrides the file value only when it differs
//! from the built-in default, and the merged result is written back to the file.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::model::ModelFamilyKind;

pub const CONFIG_FILE: &str = "blue_onyx_openvino_config.json";
pub const SERVICE_CONFIG_FILE: &str = "blue_onyx_openvino_config_service.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "trace" => LogLevel::Trace,
            "debug" => LogLevel::Debug,
            "info" => LogLevel::Info,
            "warn" | "warning" => LogLevel::Warn,
            "error" => LogLevel::Error,
            other => anyhow::bail!("unknown log level '{other}'"),
        })
    }
}

/// One model entry. Relative paths resolve against the executable directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    /// Name used in `/v1/vision/custom/{name}`; defaults to the file stem.
    pub name: Option<String>,
    /// `.xml` (OpenVINO IR, with `.bin` beside it) or `.onnx`.
    pub path: PathBuf,
    pub family: ModelFamilyKind,
    /// YAML with a `NAMES:` list. Defaults to `<stem>.yaml` next to the model, then COCO-80.
    pub classes: Option<PathBuf>,
    /// Per-model device override ("GPU", "GPU.1", "CPU"). None = global setting.
    pub device: Option<String>,
    pub confidence_threshold: Option<f32>,
    pub object_filter: Option<Vec<String>>,
    /// Compile on first request instead of at startup.
    pub lazy: bool,
    /// Per-model inference precision hint on GPU ("f16" default, "f32" escape hatch).
    pub gpu_precision: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: None,
            path: PathBuf::new(),
            family: ModelFamilyKind::Auto,
            classes: None,
            device: None,
            confidence_threshold: None,
            object_filter: None,
            lazy: false,
            gpu_precision: None,
        }
    }
}

impl ModelConfig {
    pub fn effective_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| {
            self.path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".to_string())
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub port: u16,
    /// How long a request may wait in the queue + processing before it is dropped.
    pub request_timeout_secs: u64,
    /// 0 = auto-size from timeout and measured inference time.
    pub worker_queue_size: usize,
    /// Global inference device: "GPU", "GPU.N" or "CPU".
    pub device: String,
    pub gpu_index: u32,
    pub force_cpu: bool,
    /// Compiled-model cache directory (relative to exe dir). Empty disables caching.
    pub cache_dir: String,
    /// Directory holding the OpenVINO runtime (archive layout). None = `<exe_dir>/openvino` or system.
    pub openvino_dir: Option<PathBuf>,
    pub confidence_threshold: f32,
    pub nms_iou: f32,
    /// Only report these labels (case-insensitive). Empty = all.
    pub object_filter: Vec<String>,
    pub log_level: LogLevel,
    pub log_path: Option<PathBuf>,
    /// Save annotated images here (optional).
    pub save_image_path: Option<PathBuf>,
    pub save_ref_image: bool,
    /// CPU inference threads (0 = OpenVINO default).
    pub intra_threads: usize,
    pub models_dir: PathBuf,
    /// Name of the model that serves `/v1/vision/detection`. None = first entry.
    pub default_model: Option<String>,
    pub models: Vec<ModelConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: crate::DEFAULT_PORT,
            request_timeout_secs: 15,
            worker_queue_size: 0,
            device: "GPU".to_string(),
            gpu_index: 0,
            force_cpu: false,
            cache_dir: "cache".to_string(),
            openvino_dir: None,
            confidence_threshold: 0.5,
            nms_iou: 0.5,
            object_filter: Vec::new(),
            log_level: LogLevel::Info,
            log_path: None,
            save_image_path: None,
            save_ref_image: false,
            intra_threads: 0,
            models_dir: PathBuf::from("models"),
            default_model: None,
            models: Vec::new(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).with_context(|| format!("writing config {}", path.display()))
    }

    pub fn default_config_path() -> PathBuf {
        crate::exe_dir().join(CONFIG_FILE)
    }

    pub fn service_config_path() -> PathBuf {
        crate::exe_dir().join(SERVICE_CONFIG_FILE)
    }

    /// The model that serves `/v1/vision/detection`.
    pub fn default_model_index(&self) -> Option<usize> {
        match &self.default_model {
            Some(name) => self.models.iter().position(|m| {
                crate::registry::normalize_name(&m.effective_name())
                    == crate::registry::normalize_name(name)
            }),
            None => (!self.models.is_empty()).then_some(0),
        }
    }

    /// Append `model` unless an entry with the same normalized name or the same path already
    /// exists. Returns whether it was added.
    pub fn add_model_if_absent(&mut self, model: ModelConfig) -> bool {
        let norm = |m: &ModelConfig| crate::registry::normalize_name(&m.effective_name());
        let new_name = norm(&model);
        let new_path = crate::resolve_path(&model.path);
        let exists = self
            .models
            .iter()
            .any(|m| norm(m) == new_name || crate::resolve_path(&m.path) == new_path);
        if !exists {
            self.models.push(model);
        }
        !exists
    }

    pub fn request_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.request_timeout_secs.max(1))
    }

    /// Effective device string for a model, honoring `force_cpu`, per-model override, `gpu_index`.
    pub fn device_for(&self, m: &ModelConfig) -> String {
        if self.force_cpu {
            return "CPU".to_string();
        }
        if let Some(d) = &m.device {
            return d.clone();
        }
        if self.device.eq_ignore_ascii_case("gpu") && self.gpu_index > 0 {
            return format!("GPU.{}", self.gpu_index);
        }
        self.device.clone()
    }

    /// Cache directory resolved against the exe dir, or None when disabled.
    pub fn cache_dir_path(&self) -> Option<PathBuf> {
        if self.cache_dir.trim().is_empty() {
            None
        } else {
            Some(crate::resolve_path(Path::new(&self.cache_dir)))
        }
    }
}

/// Fields edited by the `/config` web form, besides `models_json` (the `models` array as JSON).
/// Checkboxes (`force_cpu`, `save_ref_image`) are true when present with any value but
/// `false`/`off`/`0`, and false when absent (browsers omit unchecked boxes).
pub const FORM_FIELDS: &[&str] = &[
    "port",
    "request_timeout_secs",
    "worker_queue_size",
    "device",
    "gpu_index",
    "force_cpu",
    "cache_dir",
    "confidence_threshold",
    "nms_iou",
    "object_filter",
    "log_level",
    "log_path",
    "save_image_path",
    "save_ref_image",
    "intra_threads",
    "default_model",
];

/// Apply a submitted `/config` form to `config`. All fields are validated first; on any error
/// `config` is left untouched. Text fields that are absent keep their current value; empty
/// optional paths / `default_model` become `None`.
pub fn apply_config_form(
    config: &mut Config,
    form: &std::collections::HashMap<String, String>,
) -> Result<()> {
    use anyhow::bail;
    let mut c = config.clone();
    let get = |k: &str| form.get(k).map(|v| v.trim());
    fn num<T: std::str::FromStr>(field: &str, v: &str) -> Result<T> {
        v.parse::<T>()
            .map_err(|_| anyhow::anyhow!("{field}: '{v}' is not a valid number"))
    }
    let checkbox = |k: &str| {
        get(k).is_some_and(|v| !matches!(v.to_ascii_lowercase().as_str(), "false" | "off" | "0"))
    };
    let opt_path = |v: &str| (!v.is_empty()).then(|| PathBuf::from(v));

    if let Some(v) = get("port") {
        c.port = num("port", v)?;
        if c.port == 0 {
            bail!("port: must be 1-65535");
        }
    }
    if let Some(v) = get("request_timeout_secs") {
        c.request_timeout_secs = num("request_timeout_secs", v)?;
        if c.request_timeout_secs == 0 {
            bail!("request_timeout_secs: must be at least 1");
        }
    }
    if let Some(v) = get("worker_queue_size") {
        c.worker_queue_size = num("worker_queue_size", v)?;
    }
    if let Some(v) = get("device") {
        if v.is_empty() || v.chars().any(|ch| ch.is_whitespace() || ch.is_control()) {
            bail!("device: expected GPU, GPU.N or CPU, got '{v}'");
        }
        c.device = v.to_string();
    }
    if let Some(v) = get("gpu_index") {
        c.gpu_index = num("gpu_index", v)?;
    }
    c.force_cpu = checkbox("force_cpu");
    if let Some(v) = get("cache_dir") {
        c.cache_dir = v.to_string();
    }
    for (field, slot) in [
        ("confidence_threshold", &mut c.confidence_threshold),
        ("nms_iou", &mut c.nms_iou),
    ] {
        if let Some(v) = get(field) {
            let x: f32 = num(field, v)?;
            if !(0.0..=1.0).contains(&x) {
                bail!("{field}: must be between 0 and 1, got {x}");
            }
            *slot = x;
        }
    }
    if let Some(v) = get("object_filter") {
        c.object_filter = v
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
    }
    if let Some(v) = get("log_level") {
        c.log_level = v.parse().map_err(|e| anyhow::anyhow!("log_level: {e}"))?;
    }
    if let Some(v) = get("log_path") {
        c.log_path = opt_path(v);
    }
    if let Some(v) = get("save_image_path") {
        c.save_image_path = opt_path(v);
    }
    c.save_ref_image = checkbox("save_ref_image");
    if let Some(v) = get("intra_threads") {
        c.intra_threads = num("intra_threads", v)?;
    }
    if let Some(v) = form.get("models_json") {
        c.models = serde_json::from_str(v).map_err(|e| anyhow::anyhow!("models: {e}"))?;
    }
    if let Some(v) = get("default_model") {
        c.default_model = (!v.is_empty()).then(|| v.to_string());
    }

    if c.models.is_empty() {
        bail!("models: at least one model is required");
    }
    let mut seen = std::collections::HashSet::new();
    for m in &c.models {
        if m.path.as_os_str().is_empty() {
            bail!("models: every entry needs a `path`");
        }
        let key = crate::registry::normalize_name(&m.effective_name());
        if !seen.insert(key.clone()) {
            bail!("models: duplicate model name '{key}'; set a distinct `name` for each entry");
        }
    }
    if let Some(name) = &c.default_model
        && c.default_model_index().is_none()
    {
        bail!("default_model: '{name}' is not one of the configured models");
    }
    *config = c;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn form(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn with_model() -> Config {
        Config {
            models: vec![ModelConfig {
                path: "models/IPcam-general.onnx".into(),
                family: ModelFamilyKind::Yolo5,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn config_form_applies_all_fields() {
        let mut c = with_model();
        let models = r#"[{"name":"a","path":"m/a.xml","family":"yolo26"},
                         {"path":"m/IPcam-general.onnx","family":"yolo5","lazy":true}]"#;
        let f = form(&[
            ("port", " 4000 "),
            ("request_timeout_secs", "20"),
            ("worker_queue_size", "8"),
            ("device", "GPU.1"),
            ("gpu_index", "1"),
            ("force_cpu", "on"),
            ("cache_dir", ""),
            ("confidence_threshold", "0.35"),
            ("nms_iou", "0.6"),
            ("object_filter", " person, car ,,dog "),
            ("log_level", "DEBUG"),
            ("log_path", "logs"),
            ("save_image_path", ""),
            ("save_ref_image", "true"),
            ("intra_threads", "4"),
            ("default_model", "ipcam-general.onnx"),
            ("models_json", models),
        ]);
        apply_config_form(&mut c, &f).unwrap();
        assert_eq!(c.port, 4000);
        assert_eq!(c.request_timeout_secs, 20);
        assert_eq!(c.worker_queue_size, 8);
        assert_eq!(c.device, "GPU.1");
        assert_eq!(c.gpu_index, 1);
        assert!(c.force_cpu);
        assert_eq!(c.cache_dir, "");
        assert_eq!(c.confidence_threshold, 0.35);
        assert_eq!(c.nms_iou, 0.6);
        assert_eq!(c.object_filter, vec!["person", "car", "dog"]);
        assert_eq!(c.log_level, LogLevel::Debug);
        assert_eq!(c.log_path, Some(PathBuf::from("logs")));
        assert_eq!(c.save_image_path, None);
        assert!(c.save_ref_image);
        assert_eq!(c.intra_threads, 4);
        assert_eq!(c.models.len(), 2);
        assert!(c.models[1].lazy);
        assert_eq!(c.default_model_index(), Some(1));

        // Unchecked boxes are absent from the form; absent text fields keep their values.
        apply_config_form(&mut c, &form(&[("default_model", "")])).unwrap();
        assert!(!c.force_cpu && !c.save_ref_image);
        assert_eq!(c.port, 4000);
        assert_eq!(c.default_model, None);
        assert_eq!(c.models.len(), 2);
    }

    #[test]
    fn config_form_rejects_invalid_input_atomically() {
        let original = with_model();
        let cases: &[(&str, &str, &str)] = &[
            ("port", "0", "port"),
            ("port", "70000", "port"),
            ("port", "abc", "port"),
            ("request_timeout_secs", "0", "request_timeout_secs"),
            ("worker_queue_size", "-1", "worker_queue_size"),
            ("confidence_threshold", "1.5", "confidence_threshold"),
            ("nms_iou", "x", "nms_iou"),
            ("log_level", "loud", "log_level"),
            ("device", "G P U", "device"),
            ("models_json", "[{not json", "models"),
            ("models_json", "[]", "at least one"),
            ("models_json", r#"{"path":"a.xml"}"#, "models"),
            (
                "models_json",
                r#"[{"path":"a/x.onnx"},{"path":"b/X.xml"}]"#,
                "duplicate",
            ),
            ("models_json", r#"[{"name":"x"}]"#, "path"),
            ("default_model", "nope", "default_model"),
        ];
        for (field, value, needle) in cases {
            let mut c = original.clone();
            let f = form(&[(field, value), ("force_cpu", "on"), ("port", "1234")]);
            let f = if *field == "port" {
                form(&[(field, value), ("force_cpu", "on")])
            } else {
                f
            };
            let err = apply_config_form(&mut c, &f)
                .err()
                .unwrap_or_else(|| panic!("{field}={value} must fail"));
            assert!(
                format!("{err:#}").contains(needle),
                "{field}={value}: {err:#}"
            );
            assert_eq!(c, original, "{field}={value} must not modify the config");
        }
    }

    #[test]
    fn roundtrip_and_defaults() {
        let mut c = Config::default();
        c.models.push(ModelConfig {
            path: "models/yolo26s.xml".into(),
            family: ModelFamilyKind::Yolo26,
            ..Default::default()
        });
        let s = serde_json::to_string(&c).unwrap();
        let back: Config = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
        assert_eq!(back.models[0].effective_name(), "yolo26s");
        assert_eq!(back.default_model_index(), Some(0));
        let named = Config {
            default_model: Some("YOLO26S.xml".into()),
            ..back.clone()
        };
        assert_eq!(named.default_model_index(), Some(0));
        let empty: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.port, crate::DEFAULT_PORT);
    }

    #[test]
    fn add_model_if_absent_skips_duplicates() {
        let mut c = Config::default();
        let m = |name: &str, path: &str| ModelConfig {
            name: Some(name.into()),
            path: path.into(),
            ..Default::default()
        };
        assert!(c.add_model_if_absent(m("IPcam-general", "models/IPcam-general.onnx")));
        assert!(!c.add_model_if_absent(m("ipcam-general.onnx", "other/x.onnx")));
        assert!(!c.add_model_if_absent(m("different", "models/IPcam-general.onnx")));
        assert!(c.add_model_if_absent(ModelConfig {
            path: "models/delivery.onnx".into(),
            ..Default::default()
        }));
        assert_eq!(c.models.len(), 2);
    }

    #[test]
    fn device_resolution() {
        let mut c = Config::default();
        let m = ModelConfig::default();
        assert_eq!(c.device_for(&m), "GPU");
        c.gpu_index = 1;
        assert_eq!(c.device_for(&m), "GPU.1");
        c.force_cpu = true;
        assert_eq!(c.device_for(&m), "CPU");
        c.force_cpu = false;
        let m2 = ModelConfig {
            device: Some("CPU".into()),
            ..Default::default()
        };
        assert_eq!(c.device_for(&m2), "CPU");
    }
}
