//! The server run loop shared by the CLI binary and the Windows service.
//!
//! One *generation* is one [`ModelRegistry`] plus one HTTP server. `POST /config/restart`
//! cancels the generation token: the loop reloads the config file, applies a changed log level,
//! and starts the next generation. If the new config fails to start, the previous one is
//! restored. Cancelling the `shutdown` token (Ctrl-C/SIGTERM in the CLI, Stop in the service)
//! ends the loop.

use crate::cli::LogReloadHandle;
use crate::config::Config;
use crate::metrics::Metrics;
use crate::registry::ModelRegistry;
use crate::server::{self, AppState};
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

/// Error for a config without models (the server has nothing to serve).
pub fn ensure_models(config: &Config, config_path: &std::path::Path) -> Result<()> {
    if config.models.is_empty() {
        anyhow::bail!(
            "no models configured in {}. Start with `--model <path/to/model.onnx|.xml> --family yolo5` \
             or fetch one first with `blue-onyx-prism download-models --name IPcam-general`",
            config_path.display()
        );
    }
    Ok(())
}

/// Build the current-thread tokio runtime the server runs on.
pub fn build_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")
}

/// Run the server until `shutdown` is cancelled, on a runtime built for this call.
/// See [`run_server_on`].
pub fn run_server(
    config: Config,
    config_path: PathBuf,
    log: LogReloadHandle,
    shutdown: CancellationToken,
) -> Result<()> {
    let rt = build_runtime()?;
    run_server_on(&rt, config, config_path, log, shutdown)
}

/// Run registry + HTTP server generations on `rt` until `shutdown` is cancelled (returns the
/// last server result, normally `Ok`). Returns an error when the first generation cannot start
/// (no models, OpenVINO init failure, port in use) and there is no previous config to restore.
pub fn run_server_on(
    rt: &tokio::runtime::Runtime,
    mut config: Config,
    config_path: PathBuf,
    log: LogReloadHandle,
    shutdown: CancellationToken,
) -> Result<()> {
    ensure_models(&config, &config_path)?;
    let started = Instant::now();
    // Config to fall back to when a restart with the edited config fails to start.
    let mut previous: Option<Config> = None;
    loop {
        // One generation = one registry + one HTTP server. `POST /config/restart` cancels the
        // generation token; cancelling `shutdown` cancels it too.
        let generation = shutdown.child_token();
        let metrics = Arc::new(Metrics::new(crate::VERSION));
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
