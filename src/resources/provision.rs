//! Service startup integration (docs/PLAN.md, phase 7.3): before each registry generation the
//! runner asks the [`Provisioner`] what the config needs; missing resources are queued on the
//! download manager (which outlives generations) and models blocked on them wait with live
//! progress in their state.
//!
//! Rules:
//! - **Model files** (`model:*`): the blocked worker waits and loads as soon as they are
//!   installed, in the same generation.
//! - **Runtimes** (OpenVINO, an ONNX Runtime flavor, the CUDA libraries): runtimes are created
//!   once per generation, so when one is installed the runner starts a new generation
//!   ([`Provision::restart_for`]). While it downloads, a model whose file is present and which
//!   has some runnable option *now* (e.g. OpenVINO CPU while ONNX Runtime CUDA downloads) loads
//!   on that in the meantime; otherwise it waits.
//! - A second ONNX Runtime flavor while one is already loaded in this process cannot be used
//!   before the process restarts; it is downloaded and marked active for the next start.
//! - **`auto_download: false`**: nothing is downloaded; blocked models are `Failed` with
//!   [`Resolution::manual_message`] (the missing resources and the command to run).
//! - Downloads are queued model files first, then CPU-fallback runtimes (small, so something can
//!   run early), then the rest.

use super::catalog::{Flavor, Resource, ResourceKind};
use super::manager::{self, Manager, ManagerOptions, State};
use super::resolve::{self, Need, Resolution};
use crate::config::Config;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

/// How long the service waits for another process's download lock before retrying later.
pub const SERVICE_LOCK_WAIT: Duration = Duration::from_secs(5);

/// Test hooks.
#[derive(Debug, Clone, Default)]
pub struct ProvisionOptions {
    /// Extra downloadable models (see `resolve::Installed::extra_models`).
    pub extra_models: Vec<&'static Resource>,
    /// Retry schedule override for the manager.
    pub backoff: Option<fn(u32) -> Duration>,
}

/// Owns the download manager across registry generations.
pub struct Provisioner {
    manager: std::sync::Mutex<Option<Arc<Manager>>>,
    opts: ProvisionOptions,
}

impl Default for Provisioner {
    fn default() -> Self {
        Self::new()
    }
}

impl Provisioner {
    pub fn new() -> Self {
        Self::with_options(ProvisionOptions::default())
    }

    pub fn with_options(opts: ProvisionOptions) -> Self {
        Self {
            manager: std::sync::Mutex::new(None),
            opts,
        }
    }

    /// The manager, once something was queued.
    pub fn manager(&self) -> Option<Arc<Manager>> {
        self.manager
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Extra downloadable models (test hook, see [`ProvisionOptions`]).
    pub fn extra_models(&self) -> &[&'static Resource] {
        &self.opts.extra_models
    }

    /// The download manager, started on first use (also used by the web UI's Download
    /// buttons).
    pub fn manager_for(&self, config: &Config) -> Arc<Manager> {
        let mut slot = self.manager.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = slot.as_ref() {
            return m.clone();
        }
        let mut mo = ManagerOptions::new(config.data_root());
        mo.retry = true;
        mo.lock_wait = Some(SERVICE_LOCK_WAIT);
        if let Some(b) = self.opts.backoff {
            mo.backoff = b;
        }
        let m = Arc::new(Manager::start(mo));
        *slot = Some(m.clone());
        m
    }

    /// Plan one generation: what is installed, what the config needs, mark the wanted ONNX
    /// Runtime flavor active when it is installed and none is loaded yet, and (with
    /// `auto_download`) queue the downloads. `loaded_ort` is the flavor this process already
    /// loaded in an earlier generation (it cannot change before the process restarts).
    pub fn prepare(&self, config: &Config, loaded_ort: Option<Flavor>) -> Provision {
        let root = config.data_root();
        let hw = crate::backend::detect::hardware();
        let mut installed = resolve::detect_installed(config, &root, None);
        installed.extra_models = self.opts.extra_models.clone();
        if loaded_ort.is_some() {
            installed.active_ort = loaded_ort;
        }
        let resolution = resolve::needed(config, hw, &installed);
        if resolution.ort_activation().is_some() {
            match manager::apply_ort_activation(&resolution, &root) {
                Ok(Some(f)) if loaded_ort.is_none() => {
                    info!("ONNX Runtime flavor '{f}' marked active")
                }
                Ok(Some(f)) => info!(
                    "ONNX Runtime flavor '{f}' marked active; it is used after the process restarts"
                ),
                Ok(None) => {}
                Err(e) => warn!("{e:#}"),
            }
        }
        for m in &resolution.models {
            if let Some(e) = &m.error {
                warn!(model = %m.model, "{e}");
            }
        }
        for n in &resolution.optional {
            info!(
                "{} could use {} ({}); set allow_large_downloads to download it",
                n.blocking_models.join(", "),
                n.id(),
                super::catalog::format_size(n.size())
            );
        }

        let mut jobs = Vec::new();
        let mut in_flight: Vec<(&'static str, String)> = Vec::new();
        let manager = if resolution.needs.is_empty() {
            self.manager()
        } else if config.auto_download {
            let list: Vec<String> = resolution
                .needs
                .iter()
                .map(|n| format!("{} ({})", n.resource.describe(), n.reason))
                .collect();
            info!("downloading missing resources: {}", list.join("; "));
            let m = self.manager_for(config);
            let mut ordered: Vec<&Need> = resolution.needs.iter().collect();
            ordered.sort_by_key(|n| priority(n));
            let ordered: Vec<Need> = ordered.into_iter().cloned().collect();
            for job in manager::jobs_for(&resolution, &ordered, &root) {
                let key = m.enqueue(job.clone());
                jobs.push((key, job));
            }
            Some(m)
        } else {
            // Nothing is queued automatically, but a download started by hand (web UI, `fetch`
            // in this process) still counts.
            if let Some(m) = self.manager() {
                for st in m.statuses() {
                    let wanted = resolution.needs.iter().any(|n| n.id() == st.id);
                    if wanted && !matches!(st.state, State::Failed { retry_at: None, .. }) {
                        in_flight.push((st.id, st.key.clone()));
                    }
                }
            }
            if let Some(msg) = resolution.manual_message() {
                warn!("{msg}");
            }
            self.manager()
        };

        // Jobs per need, in resolution order (keys as the manager knows them).
        let keys: HashMap<&str, String> = jobs
            .iter()
            .map(|(k, j)| (j.resource.id, k.clone()))
            .chain(in_flight.iter().map(|(id, k)| (*id, k.clone())))
            .collect();
        let in_flight: std::collections::HashSet<String> =
            in_flight.into_iter().map(|(_, k)| k).collect();
        let mut blocks: HashMap<String, Vec<Block>> = HashMap::new();
        for n in &resolution.needs {
            let key = keys
                .get(n.id())
                .cloned()
                .unwrap_or_else(|| n.id().to_string());
            for model in &n.blocking_models {
                blocks
                    .entry(crate::registry::normalize_name(model))
                    .or_default()
                    .push(Block {
                        key: key.clone(),
                        resource: n.resource,
                    });
            }
        }
        Provision {
            resolution,
            manager,
            blocks,
            auto_download: config.auto_download,
            loaded_ort,
            in_flight,
        }
    }
}

/// Download order: model files, then CPU fallbacks (they block nothing and are small, so a
/// model can run early), then the runtimes models wait for.
fn priority(n: &Need) -> u8 {
    match (n.resource.kind, n.blocking_models.is_empty()) {
        (ResourceKind::Model | ResourceKind::BenchImages, _) => 0,
        (_, true) => 1,
        _ => 2,
    }
}

/// One resource a model waits for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Manager key of the job (the resource id when nothing was queued).
    pub key: String,
    pub resource: &'static Resource,
}

impl Block {
    /// A runtime (needs a new generation), not a model file.
    pub fn is_runtime(&self) -> bool {
        !self.resource.kind.is_plain_files()
    }
}

/// The plan of one generation.
pub struct Provision {
    pub resolution: Resolution,
    manager: Option<Arc<Manager>>,
    /// Normalized model name -> what it waits for.
    blocks: HashMap<String, Vec<Block>>,
    pub auto_download: bool,
    /// ONNX Runtime flavor loaded by an earlier generation of this process.
    pub loaded_ort: Option<Flavor>,
    /// With `auto_download: false`: keys of downloads started by hand that models may wait for.
    in_flight: std::collections::HashSet<String>,
}

impl std::fmt::Debug for Provision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provision")
            .field("needs", &self.resolution.needs.len())
            .field("blocks", &self.blocks)
            .field("auto_download", &self.auto_download)
            .finish_non_exhaustive()
    }
}

/// What a blocked model should do now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaitStatus {
    /// Still downloading (or retrying); the text goes into the model state.
    Pending(String),
    /// Every file is installed: load now.
    Ready,
    /// A runtime it needs is installed; a new generation will load it.
    AwaitingRestart(String),
    /// A download failed for good.
    Failed(String),
}

/// A blocked model's view of its downloads (polled by its worker).
#[derive(Clone)]
pub struct Wait {
    manager: Arc<Manager>,
    blocks: Vec<Block>,
}

impl std::fmt::Debug for Wait {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wait")
            .field("blocks", &self.blocks)
            .finish()
    }
}

impl Wait {
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn poll(&self) -> WaitStatus {
        let mut pending: Vec<String> = Vec::new();
        let mut runtime_installed: Option<&'static str> = None;
        for b in &self.blocks {
            let state = self
                .manager
                .status(&b.key)
                .map(|s| s.state)
                .unwrap_or(State::Queued);
            match &state {
                State::Installed => {
                    if b.is_runtime() {
                        runtime_installed = Some(b.resource.title);
                    }
                }
                State::Failed { retry_at: None, .. } => {
                    return WaitStatus::Failed(state.describe_for(b.resource.title));
                }
                _ => pending.push(state.describe_for(b.resource.title)),
            }
        }
        // Queued behind another download (e.g. a CPU-fallback runtime): say which.
        let all_queued = self.blocks.iter().all(|b| {
            matches!(
                self.manager.status(&b.key).map(|s| s.state),
                None | Some(State::Queued) | Some(State::Installed)
            )
        });
        let active = all_queued
            .then(|| {
                self.manager.statuses().into_iter().find(|s| {
                    matches!(
                        s.state,
                        State::Downloading { .. } | State::Verifying | State::Extracting
                    )
                })
            })
            .flatten();
        match (pending.first(), runtime_installed) {
            (Some(first), _) => {
                let mut text = match pending.len() {
                    1 => first.clone(),
                    n => format!("{first} (+{} more)", n - 1),
                };
                if let Some(a) = active {
                    text.push_str(&format!("; first {}", a.state.describe_for(a.title)));
                }
                WaitStatus::Pending(text)
            }
            (None, Some(title)) => {
                WaitStatus::AwaitingRestart(format!("{title} installed; restarting to load it"))
            }
            (None, None) => WaitStatus::Ready,
        }
    }
}

impl Provision {
    /// No downloads involved (e.g. for registries started without provisioning).
    pub fn none(config: &Config) -> Self {
        Self {
            resolution: Resolution {
                needs: Vec::new(),
                optional: Vec::new(),
                models: Vec::new(),
                ort_flavor: None,
                active_ort: None,
                ort_pinned: false,
                auto_download: config.auto_download,
            },
            manager: None,
            blocks: HashMap::new(),
            auto_download: config.auto_download,
            loaded_ort: None,
            in_flight: Default::default(),
        }
    }

    /// What `model` (effective name) waits for.
    pub fn blocks(&self, model: &str) -> &[Block] {
        self.blocks
            .get(&crate::registry::normalize_name(model))
            .map_or(&[], Vec::as_slice)
    }

    /// Something is missing (downloading, or to be fetched by hand).
    pub fn has_needs(&self) -> bool {
        !self.resolution.needs.is_empty()
    }

    /// The wait handle for a blocked model (None when it is not blocked or nothing downloads).
    pub fn wait_for(&self, model: &str) -> Option<Wait> {
        let blocks = self.blocks(model);
        if blocks.is_empty() || !self.downloading_all(model) {
            return None;
        }
        Some(Wait {
            manager: self.manager.clone()?,
            blocks: blocks.to_vec(),
        })
    }

    /// `Failed` message for a blocked model when `auto_download` is off.
    pub fn manual_failure(&self, model: &str) -> Option<String> {
        if self.blocks(model).is_empty() || self.downloading_all(model) {
            return None;
        }
        self.resolution.manual_message()
    }

    /// Every resource `model` waits for is being downloaded (always with `auto_download`).
    fn downloading_all(&self, model: &str) -> bool {
        self.auto_download
            || self
                .blocks(model)
                .iter()
                .all(|b| self.in_flight.contains(&b.key))
    }

    pub fn manager(&self) -> Option<&Arc<Manager>> {
        self.manager.as_ref()
    }

    /// Whether installing `resource` warrants a new generation: it is a runtime, this process
    /// can still load it (not a second ONNX Runtime flavor), and some model of this generation
    /// is not served on its planned device yet (`not_ready`: names of models that are not
    /// `Ready`, or loaded on an interim device).
    pub fn restart_for(&self, resource: &Resource, not_ready: &[String]) -> Result<(), String> {
        if resource.kind == ResourceKind::Model {
            return Err("model files are loaded without a restart".into());
        }
        if resource.kind == ResourceKind::BenchImages {
            return Err("benchmark images need no restart".into());
        }
        if let (Some(f), Some(loaded)) = (resource.flavor(), self.loaded_ort)
            && f != loaded
        {
            return Err(format!(
                "ONNX Runtime '{loaded}' is already loaded in this process; '{f}' is used after \
                 the process restarts"
            ));
        }
        let blocked = self
            .blocks
            .values()
            .flatten()
            .any(|b| b.resource.id == resource.id);
        if blocked || !not_ready.is_empty() {
            Ok(())
        } else {
            Err("every model already runs on its planned device".into())
        }
    }

    /// Models of this generation that should load on what is runnable now while their runtime
    /// downloads (decided by the registry; recorded for [`Self::restart_for`]).
    pub fn is_blocked_on_runtime(&self, model: &str) -> bool {
        self.blocks(model).iter().any(Block::is_runtime)
    }
}

/// Log line for a manager event.
pub fn describe_event(ev: &manager::Event) -> String {
    match ev {
        manager::Event::Installed { id, target, .. } => {
            format!("{id} installed in {}", target.display())
        }
        manager::Event::Failed {
            id,
            error,
            retry_at,
            ..
        } => match retry_at {
            Some(t) => format!(
                "download of {id} failed: {error}; retrying in {} s",
                t.duration_since(std::time::SystemTime::now())
                    .unwrap_or_default()
                    .as_secs()
            ),
            None => format!("download of {id} failed: {error}"),
        },
    }
}
