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
        Some(Command::DownloadModels {
            all,
            name,
            dir,
            add_to_config,
        }) => {
            let _log = cli::init_logging(LogLevel::Info, None)?;
            if !all && name.is_empty() {
                anyhow::bail!("specify --name <model> (repeatable) or --all; see `list-models`");
            }
            let dir = models_dir(&cli, dir);
            let files = download::download(&name, all, &dir)?;
            for f in &files {
                println!("{}", f.display());
            }
            if add_to_config {
                let path = cli
                    .config
                    .clone()
                    .unwrap_or_else(Config::default_config_path);
                let mut cfg = if path.exists() {
                    Config::load(&path)?
                } else {
                    Config::default()
                };
                let mut changed = false;
                for m in download::model_configs(&files, &blue_onyx_openvino::exe_dir()) {
                    let name = m.effective_name();
                    if cfg.add_model_if_absent(m) {
                        println!("added '{name}' to {}", path.display());
                        changed = true;
                    } else {
                        println!("'{name}' already in {}, skipped", path.display());
                    }
                }
                if changed {
                    cfg.save(&path)?;
                }
            }
            return Ok(());
        }
        Some(Command::ListModels { dir }) => {
            download::print_list(&models_dir(&cli, dir));
            return Ok(());
        }
        None => {}
    }

    let (mut config, config_path) = cli::resolve_config(&cli)?;
    // Initialized once; later level changes go through the reload handle (web UI, restart).
    let log = cli::init_logging(config.log_level, config.log_path.as_deref())?;
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

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    let shutdown = CancellationToken::new();
    {
        let token = shutdown.clone();
        rt.spawn(async move {
            wait_for_signal().await;
            info!("shutdown requested");
            token.cancel();
        });
    }

    let started = Instant::now();
    // Config to fall back to when a restart with the edited config fails to start.
    let mut previous: Option<Config> = None;
    loop {
        // One generation = one registry + one HTTP server. `POST /config/restart` cancels the
        // generation token; Ctrl-C/SIGTERM cancel `shutdown`, which cancels it too.
        let generation = shutdown.child_token();
        let metrics = Arc::new(Metrics::new(blue_onyx_openvino::VERSION));
        let registry = match ModelRegistry::start(&config, &metrics, generation.clone()) {
            Ok(r) => Arc::new(r),
            Err(e) => match previous.take() {
                Some(prev) => {
                    error!(
                        "restart with the new config failed: {e:#}; restoring the previous config"
                    );
                    config = prev;
                    continue;
                }
                None => return Err(e),
            },
        };
        info!(models = ?registry.names(), "models registered (compiling in the background)");

        let port = config.port;
        let state = Arc::new(AppState {
            registry: registry.clone(),
            metrics,
            config: Arc::new(RwLock::new(config.clone())),
            started,
            config_path: config_path.clone(),
            log_reload: Some(log.clone()),
            restart: generation.clone(),
        });
        let served = rt.block_on(server::serve(state, port, shutdown.clone()));
        // Stop this generation's workers whatever ended the server.
        generation.cancel();
        match Arc::try_unwrap(registry) {
            Ok(reg) => reg.shutdown(),
            Err(_) => warn!("registry still referenced; workers stop via the shutdown token"),
        }

        if shutdown.is_cancelled() {
            info!("bye");
            return served;
        }
        if let Err(e) = served {
            match previous.take() {
                Some(prev) => {
                    error!("{e:#}; restoring the previous config");
                    config = prev;
                    continue;
                }
                None => return Err(e),
            }
        }

        info!(config = %config_path.display(), "restarting: reloading the config file");
        let next = match Config::load(&config_path) {
            Ok(c) if !c.models.is_empty() => c,
            Ok(_) => {
                warn!("reloaded config has no models; keeping the current config");
                config.clone()
            }
            Err(e) => {
                warn!("{e:#}; keeping the current config");
                config.clone()
            }
        };
        if next.log_level != config.log_level
            && let Err(e) = log.set_level(next.log_level)
        {
            warn!("{e:#}");
        }
        if next.log_path != config.log_path {
            warn!("log_path changes take effect after the process is restarted");
        }
        previous = Some(std::mem::replace(&mut config, next));
    }
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
