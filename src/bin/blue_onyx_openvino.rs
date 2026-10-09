//! Main binary: CLI subcommands (setup-openvino, download-models, list-models) or the HTTP service.

use anyhow::{Context, Result};
use blue_onyx_openvino::cli::{self, Cli, Command};
use blue_onyx_openvino::config::{Config, LogLevel};
use blue_onyx_openvino::metrics::Metrics;
use blue_onyx_openvino::registry::ModelRegistry;
use blue_onyx_openvino::server::{self, AppState};
use blue_onyx_openvino::{download, setup_openvino, system_info};
use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

fn main() -> ExitCode {
    // Paths typed on the command line are relative to the cwd (config-file paths are
    // relative to the exe dir), so make them absolute before anything uses them.
    let cli = Cli::parse();
    let cli = match std::env::current_dir() {
        Ok(cwd) => cli.absolutized(&cwd),
        Err(_) => cli,
    };
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e:#}");
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// `--dir` if given, else the configured `models_dir` (from the config file when it exists),
/// resolved against the exe dir.
fn models_dir(cli: &Cli, dir: Option<PathBuf>) -> PathBuf {
    if let Some(d) = dir {
        return d;
    }
    let path = cli
        .config
        .clone()
        .unwrap_or_else(Config::default_config_path);
    let cfg = if path.exists() {
        Config::load(&path).unwrap_or_default()
    } else {
        Config::default()
    };
    blue_onyx_openvino::resolve_path(&cfg.models_dir)
}

fn run(cli: Cli) -> Result<()> {
    match cli.command.clone() {
        Some(Command::SetupOpenvino {
            dest,
            version,
            archive,
        }) => {
            let _log = cli::init_logging(LogLevel::Info, None)?;
            let opts = setup_openvino::SetupOptions {
                dest,
                version,
                archive,
                keep_archive: false,
            };
            let dir = setup_openvino::run(&opts)?;
            println!("OpenVINO runtime installed in {}", dir.display());
            return Ok(());
        }
        Some(Command::DownloadModels { all, name, dir }) => {
            let _log = cli::init_logging(LogLevel::Info, None)?;
            if !all && name.is_empty() {
                anyhow::bail!("specify --name <model> (repeatable) or --all; see `list-models`");
            }
            let dir = models_dir(&cli, dir);
            let files = download::download(&name, all, &dir)?;
            for f in &files {
                println!("{}", f.display());
            }
            return Ok(());
        }
        Some(Command::ListModels { dir }) => {
            download::print_list(&models_dir(&cli, dir));
            return Ok(());
        }
        None => {}
    }

    let (config, config_path) = cli::resolve_config(&cli)?;
    let _log = cli::init_logging(config.log_level, config.log_path.as_deref())?;
    info!(
        version = blue_onyx_openvino::VERSION,
        os = %system_info::os_description(),
        cpu = %system_info::cpu_name(),
        memory_gb = ?system_info::total_memory_gb(),
        config = %config_path.display(),
        "starting Blue Onyx OpenVINO"
    );
    if config.models.is_empty() {
        anyhow::bail!(
            "no models configured in {}. Start with `--model <path/to/model.onnx|.xml> --family yolo5` \
             or fetch one first with `blue-onyx-openvino download-models --name IPcam-general`",
            config_path.display()
        );
    }

    let shutdown = CancellationToken::new();
    let metrics = Arc::new(Metrics::new(blue_onyx_openvino::VERSION));
    let registry = Arc::new(ModelRegistry::start(&config, &metrics, shutdown.clone())?);
    info!(models = ?registry.names(), "models registered (compiling in the background)");

    let port = config.port;
    let state = Arc::new(AppState {
        registry: registry.clone(),
        metrics,
        config: Arc::new(RwLock::new(config)),
        started: Instant::now(),
    });

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    let served = rt.block_on(async {
        let token = shutdown.clone();
        tokio::spawn(async move {
            wait_for_signal().await;
            info!("shutdown requested");
            token.cancel();
        });
        server::serve(state, port, shutdown.clone()).await
    });
    shutdown.cancel();
    drop(rt);

    match Arc::try_unwrap(registry) {
        Ok(reg) => reg.shutdown(),
        Err(_) => warn!("registry still referenced at exit; workers stop via the shutdown token"),
    }
    info!("bye");
    served
}

async fn wait_for_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
