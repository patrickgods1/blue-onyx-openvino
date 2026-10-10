//! Main binary: CLI subcommands (setup-openvino, download-models, list-models, list-devices) or the
//! HTTP service.

use anyhow::Result;
use blue_onyx_prism::backend::{CoreOptions, Runtimes, select};
use blue_onyx_prism::cli::{self, Cli, Command};
use blue_onyx_prism::config::{Config, LogLevel, ModelConfig};
use blue_onyx_prism::{download, runner, setup_openvino, system_info};
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

/// The config file (`--config` or the default path) if it exists, else defaults. Read-only.
fn read_config(cli: &Cli) -> Config {
    let path = cli
        .config
        .clone()
        .unwrap_or_else(Config::default_config_path);
    if path.exists() {
        Config::load(&path).unwrap_or_default()
    } else {
        Config::default()
    }
}

/// `--dir` if given, else the configured `models_dir` (from the config file when it exists),
/// resolved against the exe dir.
fn models_dir(cli: &Cli, dir: Option<PathBuf>) -> PathBuf {
    if let Some(d) = dir {
        return d;
    }
    blue_onyx_prism::resolve_path(&read_config(cli).models_dir)
}

/// `list-devices`: hardware, runtimes, the option table, the `auto` pick, and the load plan of each
/// model. Works without OpenVINO (its options are shown as not runnable).
fn list_devices(cli: &Cli, models: &[PathBuf]) {
    let mut cfg = read_config(cli);
    if let Some(d) = &cli.openvino_dir {
        cfg.openvino_dir = Some(d.clone());
    }
    if let Some(d) = &cli.device {
        cfg.device = d.clone();
    }
    if cli.force_cpu {
        cfg.force_cpu = true;
    }
    let rt = Runtimes::new(&CoreOptions {
        cache_dir: None,
        intra_threads: 0,
        openvino_dir: cfg
            .openvino_dir
            .as_deref()
            .map(blue_onyx_prism::resolve_path),
    });
    let hw = rt.hardware();
    println!("Platform: {} {}", hw.os, hw.arch);
    println!("GPUs:");
    if hw.gpus.is_empty() {
        println!("  (none detected)");
    }
    for g in &hw.gpus {
        println!("  {g}");
    }
    println!("Runtimes:");
    match rt.openvino() {
        Some(core) => println!(
            "  OpenVINO {}: {}",
            core.openvino_version(),
            core.available_devices().join(", ")
        ),
        None => println!(
            "  OpenVINO: not available ({})",
            rt.openvino_error().unwrap_or("not initialized")
        ),
    }
    match &rt.probe().ort.error {
        None => println!("  ONNX Runtime: available"),
        Some(e) => println!("  ONNX Runtime: not available ({e})"),
    }
    let sel = rt.selection(None);
    println!("Device options:");
    print!("{}", select::format_options(&sel));
    let pick = sel
        .auto_pick()
        .map(|o| format!("{} ({})", o.spec, o.label))
        .unwrap_or_else(|| "nothing runnable".to_string());
    println!("auto: {pick}");
    println!("auto order: {}", select::format_auto(&sel));

    let entries: Vec<ModelConfig> = if models.is_empty() {
        cfg.enabled_models().cloned().collect()
    } else {
        models
            .iter()
            .map(|p| ModelConfig {
                path: p.clone(),
                ..ModelConfig::default()
            })
            .collect()
    };
    if entries.is_empty() {
        return;
    }
    println!("Models:");
    for m in &entries {
        let path = blue_onyx_prism::resolve_path(&m.path);
        let name = m.effective_name();
        let spec = match cfg.device_spec_for(m) {
            Ok(s) => s,
            Err(e) => {
                println!("  {name}: {e:#}");
                continue;
            }
        };
        let plan: Vec<String> = rt
            .plan(&spec, &path)
            .iter()
            .map(|c| c.device.to_string())
            .collect();
        let onnx = match select::onnx_path_for(&path) {
            Some(_) => "",
            None => ", no .onnx for ONNX Runtime",
        };
        println!(
            "  {name} ({}{onnx}): device {spec} -> {}",
            path.display(),
            plan.join(" -> ")
        );
    }
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
                let models = download::model_configs(&files, &blue_onyx_prism::exe_dir());
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
        Some(Command::ListDevices { model }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            list_devices(&cli, &model);
            return Ok(());
        }
        None => {}
    }

    let (config, config_path) = cli::resolve_config(&cli)?;
    // Initialized once; later level changes go through the reload handle (web UI, restart).
    let log = cli::init_logging(config.log_level, config.log_path.as_deref())?;
    info!(
        version = blue_onyx_prism::VERSION,
        os = %system_info::os_description(),
        cpu = %system_info::cpu_name(),
        memory_gb = ?system_info::total_memory_gb(),
        config = %config_path.display(),
        "starting Blue Onyx Prism"
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
