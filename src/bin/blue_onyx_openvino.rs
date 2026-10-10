//! Main binary: CLI subcommands (setup-openvino, download-models, list-models) or the HTTP service.

use anyhow::Result;
use blue_onyx_openvino::cli::{self, Cli, Command};
use blue_onyx_openvino::config::{Config, LogLevel};
use blue_onyx_openvino::{download, runner, setup_openvino, system_info};
use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

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
                let models = download::model_configs(&files, &blue_onyx_openvino::exe_dir());
                let outcomes = download::add_to_config(&mut cfg, models);
                for o in &outcomes {
                    println!("{}", o.describe(&path));
                }
                if outcomes
                    .iter()
                    .any(|o| matches!(o, download::AddOutcome::Added { .. }))
                {
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

    let (config, config_path) = cli::resolve_config(&cli)?;
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
    runner::ensure_models(&config, &config_path)?;

    let rt = runner::build_runtime()?;
    let shutdown = CancellationToken::new();
    {
        let token = shutdown.clone();
        rt.spawn(async move {
            wait_for_signal().await;
            info!("shutdown requested");
            token.cancel();
        });
    }
    runner::run_server_on(&rt, config, config_path, log, shutdown)
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
