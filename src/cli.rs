//! Command line, config resolution and logging setup.

use crate::config::{Config, LogLevel, ModelConfig};
use crate::model::ModelFamilyKind;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing_subscriber::{
    EnvFilter, Layer, Registry,
    layer::{Layered, SubscriberExt},
    reload,
    util::SubscriberInitExt,
};

#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    /// Download the OpenVINO runtime libraries next to the executable.
    SetupOpenvino {
        /// Destination directory (default: <exe_dir>/openvino).
        #[arg(long)]
        dest: Option<PathBuf>,
        /// OpenVINO version to download (default: the pinned version).
        #[arg(long)]
        version: Option<String>,
        /// Use an already downloaded archive instead of downloading.
        #[arg(long)]
        archive: Option<PathBuf>,
    },
    /// Download the ONNX Runtime libraries next to the executable (for the `ort:*` devices).
    SetupOnnxruntime {
        /// Which package to install; auto picks from the detected hardware.
        #[arg(long, value_enum, default_value = "auto")]
        flavor: crate::setup_onnxruntime::FlavorChoice,
        /// Destination directory (default: <exe_dir>/onnxruntime).
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Download models from Hugging Face.
    DownloadModels {
        /// Download every model in the catalog.
        #[arg(long)]
        all: bool,
        /// Model name (repeatable). See `list-models`.
        #[arg(long)]
        name: Vec<String>,
        /// Destination directory (default: the configured models dir).
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Append the downloaded models to the config file's `models` list.
        #[arg(long)]
        add_to_config: bool,
    },
    /// List downloadable models and whether they are present locally.
    ListModels {
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Download what the config needs (default), named resources, or everything for this
    /// platform: runtimes, ONNX Runtime flavors, NVIDIA libraries and models (for Docker builds
    /// and offline preparation).
    Fetch {
        /// Everything the config needs on this machine (the default without other flags).
        #[arg(long)]
        for_config: bool,
        /// Resource id (repeatable), e.g. `openvino-runtime`, `onnxruntime-cuda`,
        /// `nvidia-cuda-libs`, `model:IPcam-general`. See `list-resources`.
        #[arg(long)]
        resource: Vec<String>,
        /// Every resource available for this platform (large ones need --allow-large).
        #[arg(long)]
        all_for_platform: bool,
        /// Allow downloads over 500 MB (also config `allow_large_downloads`).
        #[arg(long)]
        allow_large: bool,
    },
    /// Show the downloadable resources for this platform: installed, needed by the config, or
    /// available, with sizes.
    ListResources {
        /// Instead, check that every pinned URL of every platform answers with its catalogued
        /// size (no full download; used by the weekly CI job). Exits non-zero on failure.
        #[arg(long)]
        check_urls: bool,
    },
    /// Show detected GPUs, every device option (runnable or why not) and what `auto` picks.
    ListDevices {
        /// Also show the load plan for this model file (repeatable; default: the enabled models
        /// in the config file).
        #[arg(long)]
        model: Vec<PathBuf>,
    },
}

#[derive(Debug, Clone, Default, Parser)]
#[command(
    name = "blue-onyx-prism",
    version = env!("CARGO_PKG_VERSION"),
    about = "Blue Iris compatible object detection service on native OpenVINO"
)]
pub struct Cli {
    /// Path to the JSON config file (default: <exe_dir>/blue_onyx_prism_config.json).
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// HTTP listen port.
    #[arg(long)]
    pub port: Option<u16>,
    /// Model file (.xml OpenVINO IR or .onnx). Replaces the configured model list unless
    /// already present in it.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Model family for --model.
    #[arg(long, value_enum)]
    pub family: Option<ModelFamilyKind>,
    /// Class names YAML for --model.
    #[arg(long)]
    pub classes: Option<PathBuf>,
    /// Name of the model that serves `/v1/vision/detection`.
    #[arg(long)]
    pub default_model: Option<String>,
    /// Inference device spec. Syntax: see `list-devices` for what can run here.
    #[arg(long, value_parser = parse_device_arg, long_help = device_help())]
    pub device: Option<String>,
    /// Device index added to a GPU device spec that has none (GPU -> GPU.N, ort:cuda -> ort:cuda:N).
    #[arg(long)]
    pub gpu_index: Option<u32>,
    /// Force CPU inference.
    #[arg(long)]
    pub force_cpu: bool,
    #[arg(long)]
    pub confidence_threshold: Option<f32>,
    /// Only report these labels, comma separated.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    pub object_filter: Option<Vec<String>>,
    #[arg(long, value_enum)]
    pub log_level: Option<LogLevel>,
    #[arg(long)]
    pub log_path: Option<PathBuf>,
    #[arg(long)]
    pub save_image_path: Option<PathBuf>,
    #[arg(long)]
    pub save_ref_image: bool,
    #[arg(long)]
    pub request_timeout_secs: Option<u64>,
    #[arg(long)]
    pub worker_queue_size: Option<usize>,
    #[arg(long)]
    pub intra_threads: Option<usize>,
    #[arg(long)]
    pub cache_dir: Option<String>,
    #[arg(long)]
    pub openvino_dir: Option<PathBuf>,
    /// Directory with the ONNX Runtime libraries (default: <exe_dir>/onnxruntime, see
    /// `setup-onnxruntime`).
    #[arg(long)]
    pub onnxruntime_dir: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Load the config file (explicit `--config`, else the default path if it exists, else
/// defaults), apply every CLI option that was given, and write the result back.
pub fn resolve_config(cli: &Cli) -> Result<(Config, PathBuf)> {
    let path = cli
        .config
        .clone()
        .unwrap_or_else(Config::default_config_path);
    let mut config = if path.exists() {
        Config::load(&path)?
    } else {
        Config::default()
    };
    let base = std::env::current_dir().context("reading current directory")?;
    merge_cli(&mut config, &cli.absolutized(&base));
    // Same names in memory as in the written file (a `--model` entry gets its file stem).
    config.fill_model_names();
    if let Err(e) = config.save(&path) {
        tracing::warn!("could not save merged config: {e:#}");
    }
    Ok((config, path))
}

/// `p` joined onto `base` when relative (no filesystem access, no canonicalization).
fn absolutize(p: &Path, base: &Path) -> PathBuf {
    if p.is_absolute() || p.as_os_str().is_empty() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

impl Cli {
    /// Copy of the CLI with every user-typed path made absolute against `base` (the current
    /// working directory). Paths typed on the command line are relative to where the user ran
    /// the command, whereas relative paths in the JSON config file resolve against the exe dir
    /// (`crate::resolve_path`); absolutizing here keeps the two rules apart, and the merged
    /// config is written back with absolute paths. An empty `--cache-dir` (disable caching)
    /// stays empty.
    pub fn absolutized(&self, base: &Path) -> Cli {
        let abs = |p: &Option<PathBuf>| p.as_deref().map(|p| absolutize(p, base));
        let mut c = self.clone();
        c.config = abs(&self.config);
        c.model = abs(&self.model);
        c.classes = abs(&self.classes);
        c.log_path = abs(&self.log_path);
        c.save_image_path = abs(&self.save_image_path);
        c.openvino_dir = abs(&self.openvino_dir);
        c.onnxruntime_dir = abs(&self.onnxruntime_dir);
        c.cache_dir = self.cache_dir.as_ref().map(|d| {
            if d.trim().is_empty() {
                d.clone()
            } else {
                absolutize(Path::new(d), base)
                    .to_string_lossy()
                    .into_owned()
            }
        });
        c.command = self.command.clone().map(|cmd| match cmd {
            Command::SetupOpenvino {
                dest,
                version,
                archive,
            } => Command::SetupOpenvino {
                dest: abs(&dest),
                version,
                archive: abs(&archive),
            },
            Command::SetupOnnxruntime { flavor, dir } => Command::SetupOnnxruntime {
                flavor,
                dir: abs(&dir),
            },
            Command::DownloadModels {
                all,
                name,
                dir,
                add_to_config,
            } => Command::DownloadModels {
                all,
                name,
                dir: abs(&dir),
                add_to_config,
            },
            Command::ListModels { dir } => Command::ListModels { dir: abs(&dir) },
            Command::Fetch { .. } | Command::ListResources { .. } => cmd,
            Command::ListDevices { model } => Command::ListDevices {
                model: model.iter().map(|p| absolutize(p, base)).collect(),
            },
        });
        c
    }
}

/// `--device` help: the spec syntax from the parser (single source of truth).
fn device_help() -> String {
    format!(
        "Inference device spec, one of: {}. Case-insensitive. `list-devices` shows what can run \
         on this machine and what `auto` picks.",
        crate::backend::spec::SPEC_SYNTAX
    )
}

/// Validate `--device` with the spec parser but keep the string as typed (it is written back to
/// the config file).
fn parse_device_arg(s: &str) -> Result<String, String> {
    crate::backend::spec::parse(s)
        .map(|_| s.trim().to_string())
        .map_err(|e| e.to_string())
}

fn merge_cli(config: &mut Config, cli: &Cli) {
    if let Some(v) = cli.port {
        config.port = v;
    }
    if let Some(v) = &cli.default_model {
        config.default_model = Some(v.clone());
    }
    if let Some(v) = &cli.device {
        config.device = v.clone();
    }
    if let Some(v) = cli.gpu_index {
        config.gpu_index = v;
    }
    if cli.force_cpu {
        config.force_cpu = true;
    }
    if let Some(v) = cli.confidence_threshold {
        config.confidence_threshold = v;
    }
    if let Some(v) = &cli.object_filter {
        config.object_filter = v.clone();
    }
    if let Some(v) = cli.log_level {
        config.log_level = v;
    }
    if let Some(v) = &cli.log_path {
        config.log_path = Some(v.clone());
    }
    if let Some(v) = &cli.save_image_path {
        config.save_image_path = Some(v.clone());
    }
    if cli.save_ref_image {
        config.save_ref_image = true;
    }
    if let Some(v) = cli.request_timeout_secs {
        config.request_timeout_secs = v;
    }
    if let Some(v) = cli.worker_queue_size {
        config.worker_queue_size = v;
    }
    if let Some(v) = cli.intra_threads {
        config.intra_threads = v;
    }
    if let Some(v) = &cli.cache_dir {
        config.cache_dir = v.clone();
    }
    if let Some(v) = &cli.openvino_dir {
        config.openvino_dir = Some(v.clone());
    }
    if let Some(v) = &cli.onnxruntime_dir {
        config.onnxruntime_dir = Some(v.clone());
    }
    if let Some(model) = &cli.model {
        // `--model` always loads that model: enable an existing entry, else replace the list.
        let mut found = false;
        for m in config.models.iter_mut().filter(|m| &m.path == model) {
            m.enabled = true;
            found = true;
        }
        if !found {
            config.models = vec![ModelConfig {
                path: model.clone(),
                family: cli.family.unwrap_or(ModelFamilyKind::Auto),
                classes: cli.classes.clone(),
                enabled: true,
                ..Default::default()
            }];
        }
    }
}

/// Load `blue_onyx_prism_config_service.json`, creating a default one (log level debug)
/// when missing.
pub fn for_service() -> Result<(Config, PathBuf)> {
    let path = Config::service_config_path();
    if path.exists() {
        Ok((Config::load(&path)?, path))
    } else {
        let config = Config {
            log_level: LogLevel::Debug,
            ..Default::default()
        };
        config
            .save(&path)
            .with_context(|| format!("creating default service config {}", path.display()))?;
        Ok((config, path))
    }
}

type FilterHandle = reload::Handle<EnvFilter, Registry>;

/// Lets the log level be changed at runtime; also keeps the file-appender flush guard alive.
/// Cheap to clone (the HTTP layer keeps one); the file log is flushed when the last clone drops.
#[derive(Clone)]
pub struct LogReloadHandle {
    handle: FilterHandle,
    /// The in-memory ring behind the web UI's Logs page.
    buffer: crate::logbuf::LogBuffer,
    _guard: Option<Arc<tracing_appender::non_blocking::WorkerGuard>>,
    /// Keeps the filter layer of a [`Self::detached`] handle alive (reloads need it).
    _detached: Option<Arc<reload::Layer<EnvFilter, Registry>>>,
}

impl std::fmt::Debug for LogReloadHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogReloadHandle").finish_non_exhaustive()
    }
}

impl LogReloadHandle {
    /// Replace the global level filter; takes effect immediately for every thread.
    pub fn set_level(&self, level: LogLevel) -> Result<()> {
        self.handle
            .reload(filter_for(level))
            .map_err(|e| anyhow::anyhow!("reloading log filter: {e}"))
    }

    /// The recent log events (the Logs page).
    pub fn buffer(&self) -> &crate::logbuf::LogBuffer {
        &self.buffer
    }

    /// A handle that is not installed as the global subscriber: its level filter and buffer
    /// work but see no events unless `buffer` is fed directly (tests of the HTTP layer).
    pub fn detached(level: LogLevel) -> Self {
        let (layer, handle) = reload::Layer::<EnvFilter, Registry>::new(filter_for(level));
        Self {
            handle,
            buffer: crate::logbuf::LogBuffer::default(),
            _guard: None,
            _detached: Some(Arc::new(layer)),
        }
    }
}

fn filter_for(level: LogLevel) -> EnvFilter {
    EnvFilter::new(level.as_str())
}

/// Subscriber below the extra layer passed to [`init_logging_with`]: the registry plus the
/// reloadable level filter.
pub type LogBase = Layered<reload::Layer<EnvFilter, Registry>, Registry>;

/// An additional log sink (e.g. the Windows event log in the service binary).
pub type ExtraLogLayer = Box<dyn Layer<LogBase> + Send + Sync>;

/// Initialize global tracing. Logs to stdout, or to a daily rolling file in `log_path`
/// when given.
pub fn init_logging(level: LogLevel, log_path: Option<&Path>) -> Result<LogReloadHandle> {
    init_logging_with(level, log_path, None)
}

/// [`init_logging`] plus an optional extra layer. The level filter (and thus
/// [`LogReloadHandle::set_level`]) applies to every sink, including `extra` and the in-memory
/// [`crate::logbuf::LogBuffer`] (always installed; see [`LogReloadHandle::buffer`]); `extra` may
/// add its own per-layer filter on top.
pub fn init_logging_with(
    level: LogLevel,
    log_path: Option<&Path>,
    extra: Option<ExtraLogLayer>,
) -> Result<LogReloadHandle> {
    let (filter, handle) = reload::Layer::new(filter_for(level));
    let buffer = crate::logbuf::LogBuffer::default();
    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(extra)
        .with(buffer.clone());
    let guard = match log_path {
        Some(dir) => {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating log dir {}", dir.display()))?;
            let appender = tracing_appender::rolling::daily(dir, "blue_onyx_prism.log");
            let (writer, guard) = tracing_appender::non_blocking(appender);
            registry
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_writer(writer)
                        .with_ansi(false),
                )
                .try_init()
                .map_err(|e| anyhow::anyhow!("initializing logging: {e}"))?;
            Some(guard)
        }
        None => {
            registry
                .with(tracing_subscriber::fmt::layer())
                .try_init()
                .map_err(|e| anyhow::anyhow!("initializing logging: {e}"))?;
            None
        }
    };
    Ok(LogReloadHandle {
        handle,
        buffer,
        _guard: guard.map(Arc::new),
        _detached: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cfg() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bo_cli_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("cfg.json")
    }

    #[test]
    fn parses_args() {
        let cli = Cli::try_parse_from([
            "x",
            "--port",
            "1234",
            "--family",
            "yolo5",
            "--force-cpu",
            "--object-filter",
            "person,car",
        ])
        .unwrap();
        assert_eq!(cli.port, Some(1234));
        assert_eq!(cli.family, Some(ModelFamilyKind::Yolo5));
        assert!(cli.force_cpu);
        assert_eq!(cli.object_filter.unwrap(), vec!["person", "car"]);
        let cli =
            Cli::try_parse_from(["x", "download-models", "--name", "a", "--name", "b"]).unwrap();
        match cli.command {
            Some(Command::DownloadModels { name, all, .. }) => {
                assert_eq!(name, vec!["a", "b"]);
                assert!(!all);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn device_flag_is_validated() {
        for ok in [
            "GPU",
            "gpu.1",
            "auto",
            "openvino:npu",
            "ort:cuda:1",
            " ort:coreml ",
        ] {
            let cli = Cli::try_parse_from(["x", "--device", ok]).unwrap();
            assert_eq!(cli.device.as_deref(), Some(ok.trim()));
        }
        for bad in ["G P U", "ort:gpu", "xpu", ""] {
            assert!(
                Cli::try_parse_from(["x", "--device", bad]).is_err(),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn default_model_flag_is_merged() {
        let cli = Cli::try_parse_from(["x", "--default-model", "yolo26s"]).unwrap();
        let mut cfg = Config::default();
        merge_cli(&mut cfg, &cli);
        assert_eq!(cfg.default_model.as_deref(), Some("yolo26s"));
        let mut cfg = Config {
            default_model: Some("a".into()),
            ..Default::default()
        };
        merge_cli(&mut cfg, &Cli::default());
        assert_eq!(cfg.default_model.as_deref(), Some("a"));
        let cli =
            Cli::try_parse_from(["x", "download-models", "--all", "--add-to-config"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::DownloadModels {
                add_to_config: true,
                ..
            })
        ));
    }

    #[test]
    fn absent_option_keeps_file_value() {
        let path = temp_cfg();
        let file = Config {
            port: 4000,
            confidence_threshold: 0.7,
            force_cpu: true,
            ..Default::default()
        };
        file.save(&path).unwrap();
        let cli = Cli {
            config: Some(path.clone()),
            ..Default::default()
        };
        let (c, p) = resolve_config(&cli).unwrap();
        assert_eq!(p, path);
        assert_eq!(c, file);
    }

    #[test]
    fn given_option_overrides_and_is_written_back() {
        let path = temp_cfg();
        Config {
            port: 4000,
            confidence_threshold: 0.7,
            ..Default::default()
        }
        .save(&path)
        .unwrap();
        let cli = Cli {
            config: Some(path.clone()),
            port: Some(5000),
            ..Default::default()
        };
        let (c, _) = resolve_config(&cli).unwrap();
        assert_eq!(c.port, 5000);
        assert_eq!(c.confidence_threshold, 0.7);
        assert_eq!(Config::load(&path).unwrap().port, 5000);
    }

    #[test]
    fn model_creates_single_entry() {
        let path = temp_cfg();
        let existing = ModelConfig {
            path: "models/old.xml".into(),
            ..Default::default()
        };
        Config {
            models: vec![existing.clone(), existing.clone()],
            ..Default::default()
        }
        .save(&path)
        .unwrap();
        let cli = Cli {
            config: Some(path.clone()),
            model: Some("models/new.onnx".into()),
            family: Some(ModelFamilyKind::Yolo5),
            classes: Some("c.yaml".into()),
            ..Default::default()
        };
        let (c, _) = resolve_config(&cli).unwrap();
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(c.models.len(), 1);
        assert_eq!(c.models[0].path, cwd.join("models/new.onnx"));
        assert_eq!(c.models[0].family, ModelFamilyKind::Yolo5);
        assert_eq!(c.models[0].classes, Some(cwd.join("c.yaml")));

        assert!(c.models[0].enabled);

        // Already present: list is untouched.
        let cli = Cli {
            config: Some(path.clone()),
            model: Some("models/new.onnx".into()),
            ..Default::default()
        };
        let (c2, _) = resolve_config(&cli).unwrap();
        assert_eq!(c2.models, c.models);

        // Already present but disabled: it gets enabled, other entries are kept as they are.
        let new_path = c.models[0].path.clone();
        Config {
            models: vec![
                ModelConfig {
                    enabled: true,
                    ..existing.clone()
                },
                ModelConfig {
                    path: new_path.clone(),
                    enabled: false,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
        .save(&path)
        .unwrap();
        let cli = Cli {
            config: Some(path),
            model: Some("models/new.onnx".into()),
            ..Default::default()
        };
        let (c3, _) = resolve_config(&cli).unwrap();
        assert_eq!(c3.models.len(), 2);
        assert!(c3.models[0].enabled && c3.models[1].enabled);
        assert_eq!(c3.models[1].path, new_path);
    }

    #[test]
    fn cli_paths_are_absolutized_against_cwd() {
        let base = if cfg!(windows) {
            PathBuf::from(r"C:\work")
        } else {
            PathBuf::from("/work")
        };
        let other_abs = if cfg!(windows) {
            PathBuf::from(r"D:\abs\m.xml")
        } else {
            PathBuf::from("/abs/m.xml")
        };
        let cli = Cli::try_parse_from([
            "x",
            "--model",
            "target/release/models/ipcam-bird.onnx",
            "--classes",
            "c.yaml",
            "--cache-dir",
            "target/release/cache",
            "--log-path",
            "logs",
            "--save-image-path",
            "imgs",
            "--openvino-dir",
            "ov",
            "--onnxruntime-dir",
            "ort",
            "--config",
            "cfg.json",
        ])
        .unwrap()
        .absolutized(&base);
        assert_eq!(
            cli.model.as_deref(),
            Some(base.join("target/release/models/ipcam-bird.onnx").as_path())
        );
        assert_eq!(cli.classes, Some(base.join("c.yaml")));
        assert_eq!(
            cli.cache_dir.as_deref().map(PathBuf::from),
            Some(base.join("target/release/cache"))
        );
        assert_eq!(cli.log_path, Some(base.join("logs")));
        assert_eq!(cli.save_image_path, Some(base.join("imgs")));
        assert_eq!(cli.openvino_dir, Some(base.join("ov")));
        assert_eq!(cli.onnxruntime_dir, Some(base.join("ort")));
        assert_eq!(cli.config, Some(base.join("cfg.json")));

        // Absolute paths, empty cache dir (= disabled) and absent options are unchanged.
        let cli = Cli {
            model: Some(other_abs.clone()),
            cache_dir: Some(String::new()),
            ..Default::default()
        }
        .absolutized(&base);
        assert_eq!(cli.model, Some(other_abs));
        assert_eq!(cli.cache_dir.as_deref(), Some(""));
        assert_eq!(cli.log_path, None);

        // Merged into the config, the absolute path survives `resolve_path` (no exe-dir join).
        let mut cfg = Config::default();
        let cli = Cli::try_parse_from(["x", "--model", "m/a.onnx", "--cache-dir", "cache"])
            .unwrap()
            .absolutized(&base);
        merge_cli(&mut cfg, &cli);
        assert_eq!(
            crate::resolve_path(&cfg.models[0].path),
            base.join("m/a.onnx")
        );
        assert_eq!(cfg.cache_dir_path(), Some(base.join("cache")));

        // Subcommand paths too.
        let cli = Cli::try_parse_from(["x", "download-models", "--name", "a", "--dir", "d"])
            .unwrap()
            .absolutized(&base);
        match cli.command {
            Some(Command::DownloadModels { dir, .. }) => assert_eq!(dir, Some(base.join("d"))),
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn missing_config_uses_defaults_and_creates_file() {
        let path = temp_cfg();
        let cli = Cli {
            config: Some(path.clone()),
            ..Default::default()
        };
        let (c, _) = resolve_config(&cli).unwrap();
        assert_eq!(c, Config::default());
        assert!(path.exists());
    }
}
