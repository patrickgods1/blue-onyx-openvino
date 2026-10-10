//! Main binary: CLI subcommands (setup-openvino, setup-onnxruntime, download-models, list-models,
//! fetch, list-resources, list-devices, benchmark) or the HTTP service.

use anyhow::Result;
use blue_onyx_prism::backend::{CoreOptions, OrtOptions, Runtimes, select};
use blue_onyx_prism::cli::{self, Cli, Command};
use blue_onyx_prism::config::{Config, LogLevel, ModelConfig};
use blue_onyx_prism::{
    download, resources, runner, setup_onnxruntime, setup_openvino, system_info,
};
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
    let cfg = read_config(cli);
    cfg.data_path(&cfg.models_dir)
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
    let ort_opts: OrtOptions = cfg.ort_options();
    let rt = Runtimes::new_with(
        &CoreOptions {
            cache_dir: None,
            intra_threads: 0,
            openvino_dir: cfg.openvino_dir_effective(),
        },
        &ort_opts,
    );
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
    let mut sel = rt.selection(None);
    let installed = resources::detect_installed(&cfg, &cfg.data_root(), Some(rt.probe().clone()));
    resources::resolve::annotate_selection(&mut sel, &cfg, hw, &installed, true);
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
        let path = cfg.data_path(&m.path);
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
                root: Some(resources::download_root(&read_config(&cli))),
            };
            let dir = setup_openvino::run(&opts)?;
            println!("OpenVINO runtime installed in {}", dir.display());
            return Ok(());
        }
        Some(Command::SetupOnnxruntime { flavor, dir }) => {
            let _log = cli::init_logging(LogLevel::Info, None)?;
            let opts = setup_onnxruntime::SetupOptions {
                dest: dir,
                flavor,
                keep_archive: false,
                root: Some(resources::download_root(&read_config(&cli))),
            };
            let (dir, flavor) = setup_onnxruntime::run(&opts)?;
            println!(
                "ONNX Runtime ({}) installed in {}",
                flavor.as_str(),
                dir.display()
            );
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
            let root = resources::download_root(&read_config(&cli));
            let files = download::download_with_root(&name, all, &dir, &root)?;
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
        Some(Command::Fetch {
            for_config,
            resource,
            all_for_platform,
            allow_large,
        }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            let req = resources::commands::FetchRequest {
                for_config,
                resources: resource,
                all_for_platform,
                allow_large,
            };
            return resources::commands::fetch(&read_config(&cli), &req);
        }
        Some(Command::ListResources { check_urls: true }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            return resources::commands::check_urls();
        }
        Some(Command::ListResources { check_urls: false }) => {
            let _log = cli::init_logging(LogLevel::Warn, None)?;
            let cfg = read_config(&cli);
            let root = resources::download_root(&cfg);
            let installed = resources::detect_installed(&cfg, &root, None);
            print!(
                "{}",
                resources::commands::list_resources(
                    &cfg,
                    blue_onyx_prism::backend::detect::hardware(),
                    &installed
                )
            );
            return Ok(());
        }
        Some(Command::Benchmark(args)) => {
            // Sets up its own stderr logging (warn, or info with --verbose).
            let code = blue_onyx_prism::benchmark::cli::main(*args, cli.config.clone());
            if code != ExitCode::SUCCESS {
                std::process::exit(1);
            }
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
