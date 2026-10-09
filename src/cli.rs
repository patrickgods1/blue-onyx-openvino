//! Command line, config resolution and logging setup.

use crate::config::{Config, LogLevel, ModelConfig};
use crate::model::ModelFamilyKind;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, reload, util::SubscriberInitExt};

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
    },
    /// List downloadable models and whether they are present locally.
    ListModels {
        #[arg(long)]
        dir: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Default, Parser)]
#[command(
    name = "blue-onyx-openvino",
    version = env!("CARGO_PKG_VERSION"),
    about = "Blue Iris compatible object detection service on native OpenVINO"
)]
pub struct Cli {
    /// Path to the JSON config file (default: <exe_dir>/blue_onyx_openvino_config.json).
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
    /// Inference device: GPU, GPU.N or CPU.
    #[arg(long)]
    pub device: Option<String>,
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
    merge_cli(&mut config, cli);
    if let Err(e) = config.save(&path) {
        tracing::warn!("could not save merged config: {e:#}");
    }
    Ok((config, path))
}

fn merge_cli(config: &mut Config, cli: &Cli) {
    if let Some(v) = cli.port {
        config.port = v;
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
    if let Some(model) = &cli.model
        && !config.models.iter().any(|m| &m.path == model)
    {
        config.models = vec![ModelConfig {
            path: model.clone(),
            family: cli.family.unwrap_or(ModelFamilyKind::Auto),
            classes: cli.classes.clone(),
            ..Default::default()
        }];
    }
}

/// Load `blue_onyx_openvino_config_service.json`, creating a default one (log level debug)
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
pub struct LogReloadHandle {
    handle: FilterHandle,
    _guard: Option<tracing_appender::non_blocking::WorkerGuard>,
}

impl LogReloadHandle {
    pub fn set_level(&self, level: LogLevel) -> Result<()> {
        self.handle
            .reload(filter_for(level))
            .map_err(|e| anyhow::anyhow!("reloading log filter: {e}"))
    }
}

fn filter_for(level: LogLevel) -> EnvFilter {
    EnvFilter::new(level.as_str())
}

/// Initialize global tracing. Logs to stdout, or to a daily rolling file in `log_path`
/// when given.
pub fn init_logging(level: LogLevel, log_path: Option<&Path>) -> Result<LogReloadHandle> {
    let (filter, handle) = reload::Layer::new(filter_for(level));
    let registry = tracing_subscriber::registry().with(filter);
    let guard = match log_path {
        Some(dir) => {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating log dir {}", dir.display()))?;
            let appender = tracing_appender::rolling::daily(dir, "blue_onyx_openvino.log");
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
        _guard: guard,
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
        let cli = Cli::try_parse_from(["x", "download-models", "--name", "a", "--name", "b"]).unwrap();
        match cli.command {
            Some(Command::DownloadModels { name, all, .. }) => {
                assert_eq!(name, vec!["a", "b"]);
                assert!(!all);
            }
            _ => panic!("wrong command"),
        }
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
        assert_eq!(c.models.len(), 1);
        assert_eq!(c.models[0].path, PathBuf::from("models/new.onnx"));
        assert_eq!(c.models[0].family, ModelFamilyKind::Yolo5);
        assert_eq!(c.models[0].classes, Some(PathBuf::from("c.yaml")));

        // Already present: list is untouched.
        let cli = Cli {
            config: Some(path),
            model: Some("models/new.onnx".into()),
            ..Default::default()
        };
        let (c2, _) = resolve_config(&cli).unwrap();
        assert_eq!(c2.models, c.models);
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
