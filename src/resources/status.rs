//! What the web UI shows and does with resources (phase 7.5): the rows of `GET /v1/resources`
//! and the config page's Resources card, and the Download / Remove / Add-to-config actions.
//!
//! - **Download** queues the resource on the shared download manager. Resources over the
//!   large-download threshold need `confirm_large` (the button that shows the size supplies it)
//!   unless `allow_large_downloads` is set.
//! - **Remove** deletes an install we made (a `.installed.json` manifest names it). It refuses
//!   while the resource downloads, when this process has the runtime loaded (OpenVINO, the
//!   loaded ONNX Runtime flavor, preloaded CUDA libraries: "restart required"), and when an
//!   enabled model of the config uses the model files.
//! - **Add to config** appends an installed catalog model to `models` with the same rules as
//!   `download-models --add-to-config`, or a model file found in `models_dir` that no config
//!   entry references (id `local:<file>`, see [`local_models`]; family `auto`).
//!
//! Rows carry a `group` ([`group_of`]) so the UI shows Runtimes, GPU libraries and Models
//! (by family) as separate tables ([`grouped`]).

use super::catalog::{self, Flavor, Resource, ResourceKind};
use super::manager::{self, Job, State, Status};
use super::provision::{Provision, Provisioner};
use super::resolve::{self, Installed, Resolution};
use crate::config::Config;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What the HTTP layer of one generation knows about resources.
#[derive(Clone)]
pub struct ResourcesCtx {
    /// Owns the download manager for the whole run.
    pub provisioner: Arc<Provisioner>,
    /// This generation's plan (what models wait for).
    pub provision: Arc<Provision>,
    /// OpenVINO is loaded in this process (its files are in use).
    pub openvino_loaded: bool,
    /// ONNX Runtime flavor loaded in this process.
    pub ort_loaded: Option<Flavor>,
}

impl std::fmt::Debug for ResourcesCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourcesCtx")
            .field("openvino_loaded", &self.openvino_loaded)
            .field("ort_loaded", &self.ort_loaded)
            .finish_non_exhaustive()
    }
}

/// One resource as shown by `/v1/resources` and the Resources card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceRow {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    /// `openvino-runtime`, `onnx-runtime`, `cuda-libs`, `model`.
    pub kind: &'static str,
    /// UI section: `runtimes`, `gpu-libraries`, `models-yolov5`, `models-rt-detr`,
    /// `models-d-fine`, `models-rf-detr` (see [`group_of`]).
    pub group: &'static str,
    pub version: &'static str,
    pub license: &'static str,
    pub size: u64,
    /// "29 MB".
    pub size_text: String,
    /// Over the large-download threshold (needs confirmation / `allow_large_downloads`).
    pub large: bool,
    /// `installed`, `downloading`, `queued`, `verifying`, `extracting`, `failed`, `needed`,
    /// `optional` (needed by a better device, gated by `allow_large_downloads`), `available`.
    pub state: &'static str,
    /// Human-readable state, e.g. "downloading 42% (18/44 MB)".
    pub state_text: String,
    /// Queued, downloading, verifying or extracting.
    pub busy: bool,
    /// Download progress in percent while downloading.
    pub progress: Option<u8>,
    /// Last error of a failed download.
    pub error: Option<String>,
    /// Seconds until the next automatic retry of a failed download.
    pub retry_in: Option<u64>,
    /// Device options / model it provides (`ort:coreml`, `model:IPcam-general`).
    pub provides: Vec<String>,
    /// Enabled models that wait for it.
    pub blocking: Vec<String>,
    /// The ONNX Runtime flavor marked active (`active.txt`) or loaded.
    pub active: bool,
    /// Loaded by this process right now (cannot be removed before a restart).
    pub loaded: bool,
    /// Remove is possible now.
    pub removable: bool,
    /// Why Remove is not possible (when installed).
    pub remove_blocked: Option<String>,
    /// A model resource that can be appended to `models` (installed, not in the config yet).
    pub can_add_to_config: bool,
    /// Where it installs.
    pub target: String,
}

/// Why an action was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionError {
    NotFound(String),
    /// Refused in the current state (409): large without confirmation, in use, downloading.
    Conflict(String),
    Failed(String),
}

impl ActionError {
    pub fn message(&self) -> &str {
        match self {
            ActionError::NotFound(m) | ActionError::Conflict(m) | ActionError::Failed(m) => m,
        }
    }

    /// HTTP status code.
    pub fn status(&self) -> u16 {
        match self {
            ActionError::NotFound(_) => 404,
            ActionError::Conflict(_) => 409,
            ActionError::Failed(_) => 500,
        }
    }
}

impl std::error::Error for ActionError {}

impl std::fmt::Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

fn kind_str(k: ResourceKind) -> &'static str {
    match k {
        ResourceKind::OpenVinoRuntime => "openvino-runtime",
        ResourceKind::OnnxRuntime(_) => "onnx-runtime",
        ResourceKind::CudaLibs => "cuda-libs",
        ResourceKind::Model => "model",
    }
}

/// Group keys in display order with their titles.
pub const GROUPS: &[(&str, &str)] = &[
    ("runtimes", "Runtimes"),
    ("gpu-libraries", "GPU libraries"),
    ("models-yolov5", "Models \u{b7} YOLOv5 (IPcam and custom)"),
    ("models-rt-detr", "Models \u{b7} RT-DETRv2"),
    ("models-d-fine", "Models \u{b7} D-FINE"),
    ("models-rf-detr", "Models \u{b7} RF-DETR"),
];

/// Group key of the local (unconfigured) model files.
pub const LOCAL_GROUP: &str = "local-models";

/// UI section of a resource: runtimes (OpenVINO, ONNX Runtime flavors incl. DirectML), GPU
/// libraries (NVIDIA CUDA/cuDNN), and models by family, keyed on the id prefix.
pub fn group_of(r: &Resource) -> &'static str {
    match r.kind {
        ResourceKind::OpenVinoRuntime | ResourceKind::OnnxRuntime(_) => "runtimes",
        ResourceKind::CudaLibs => "gpu-libraries",
        ResourceKind::Model => model_group(r.id),
    }
}

fn model_group(id: &str) -> &'static str {
    let id = id.to_ascii_lowercase();
    let name = id.strip_prefix(catalog::MODEL_ID_PREFIX).unwrap_or(&id);
    if name.starts_with("rt-detr") {
        "models-rt-detr"
    } else if name.starts_with("dfine") {
        "models-d-fine"
    } else if name.starts_with("rfdetr") {
        "models-rf-detr"
    } else {
        "models-yolov5"
    }
}

/// Rows of one UI section.
#[derive(Debug, Clone)]
pub struct ResourceGroup {
    pub key: &'static str,
    pub title: &'static str,
    pub rows: Vec<ResourceRow>,
}

/// Split `rows` into the non-empty [`GROUPS`], in display order.
pub fn grouped(rows: Vec<ResourceRow>) -> Vec<ResourceGroup> {
    let mut out: Vec<ResourceGroup> = GROUPS
        .iter()
        .map(|&(key, title)| ResourceGroup {
            key,
            title,
            rows: Vec::new(),
        })
        .collect();
    for r in rows {
        if let Some(g) = out.iter_mut().find(|g| g.key == r.group) {
            g.rows.push(r);
        }
    }
    out.retain(|g| !g.rows.is_empty());
    out
}

/// Id prefix of local model files (`local:yolo26s.onnx`).
pub const LOCAL_ID_PREFIX: &str = "local:";

/// A model file in `models_dir` that no config entry references and that is not a catalog
/// download (those have their own rows).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LocalModel {
    /// `local:<file name>`.
    pub id: String,
    /// File name, e.g. `yolo26s.onnx` (the `.onnx` when both formats exist).
    pub file: String,
    /// Model name it gets when added (the file stem).
    pub name: String,
    /// `onnx` or `openvino-ir`.
    pub format: &'static str,
    /// The other format of the same stem when both exist (`yolo26s.xml`).
    pub also: Option<String>,
    pub size: u64,
    pub size_text: String,
    /// Always `auto` (picked from the model's outputs at load).
    pub family: &'static str,
    /// Full path.
    pub path: String,
    /// What it can run on.
    pub note: &'static str,
    pub group: &'static str,
}

fn model_ext(p: &Path) -> Option<&'static str> {
    match p
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("onnx") => Some("onnx"),
        Some("xml") => Some("xml"),
        _ => None,
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// Model files (`.onnx`, `.xml`) in `models_dir` not referenced by any config entry (by stem:
/// a configured `x.xml` hides `x.onnx` too) and not part of a catalog resource. When both
/// `x.xml` and `x.onnx` exist, the `.onnx` is offered (it runs on both runtimes and CoreML).
pub fn local_models(ctx: Option<&ResourcesCtx>, config: &Config) -> Vec<LocalModel> {
    let dir = config.data_path(&config.models_dir);
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let catalog_files: Vec<String> = platform_resources(ctx)
        .into_iter()
        .chain(catalog::MODELS.iter())
        .filter(|r| r.kind == ResourceKind::Model)
        .flat_map(|r| r.parts.iter().map(|p| p.file_name.to_ascii_lowercase()))
        .collect();
    let configured: Vec<PathBuf> = config
        .models
        .iter()
        .map(|m| config.data_path(&m.path))
        .collect();
    // stem -> (onnx, xml)
    let mut stems: std::collections::BTreeMap<String, (Option<PathBuf>, Option<PathBuf>)> =
        Default::default();
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        let Some(ext) = model_ext(&p) else { continue };
        let Some(stem) = p.file_stem().and_then(|s| s.to_str()).map(str::to_string) else {
            continue;
        };
        let slot = stems.entry(stem).or_default();
        if ext == "onnx" {
            slot.0 = Some(p);
        } else {
            slot.1 = Some(p);
        }
    }
    let mut out = Vec::new();
    for (stem, (onnx, xml)) in stems {
        let files: Vec<&PathBuf> = onnx.iter().chain(xml.iter()).collect();
        let is_catalog = files.iter().any(|f| {
            f.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| catalog_files.contains(&n.to_ascii_lowercase()))
        });
        let is_configured = files
            .iter()
            .any(|f| configured.iter().any(|c| same_file(c, f)));
        if is_catalog || is_configured {
            continue;
        }
        let (main, other) = match (&onnx, &xml) {
            (Some(o), x) => (o, x.as_ref()),
            (None, Some(x)) => (x, None),
            (None, None) => continue,
        };
        let file = main
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let is_onnx = model_ext(main) == Some("onnx");
        let mut size = std::fs::metadata(main).map_or(0, |m| m.len());
        if !is_onnx {
            size += std::fs::metadata(main.with_extension("bin")).map_or(0, |m| m.len());
        }
        out.push(LocalModel {
            id: format!("{LOCAL_ID_PREFIX}{file}"),
            name: stem,
            format: if is_onnx { "onnx" } else { "openvino-ir" },
            also: other.and_then(|o| o.file_name().map(|n| n.to_string_lossy().to_string())),
            size,
            size_text: catalog::format_size(size),
            family: "auto",
            path: main.display().to_string(),
            note: if is_onnx {
                "ONNX: runs on OpenVINO and ONNX Runtime (CoreML, CUDA, DirectML, CPU)"
            } else {
                "OpenVINO IR: OpenVINO devices only (export the .onnx for ONNX Runtime / CoreML)"
            },
            group: LOCAL_GROUP,
            file,
        });
    }
    out
}

/// Every resource of this platform, plus the provisioner's extra (test) models.
pub fn platform_resources(ctx: Option<&ResourcesCtx>) -> Vec<&'static Resource> {
    let hw = crate::backend::detect::hardware();
    let mut out: Vec<&'static Resource> = ctx
        .map(|c| c.provisioner.extra_models().to_vec())
        .unwrap_or_default();
    out.extend(catalog::for_platform(&hw.os, &hw.arch));
    out
}

/// Find a resource by id (case-insensitive) among [`platform_resources`].
pub fn find(ctx: Option<&ResourcesCtx>, id: &str) -> Option<&'static Resource> {
    let id = id.trim();
    platform_resources(ctx)
        .into_iter()
        .find(|r| r.id.eq_ignore_ascii_case(id))
}

/// Install directory of `res` for `config`: models in `models_dir`, runtimes under the data
/// root.
pub fn target_dir(config: &Config, res: &Resource) -> PathBuf {
    match res.kind {
        ResourceKind::Model => config.data_path(&config.models_dir),
        _ => config.data_root().join(res.dest),
    }
}

fn installed_snapshot(ctx: Option<&ResourcesCtx>, config: &Config) -> Installed {
    let mut installed = resolve::detect_installed(config, &config.data_root(), None);
    if let Some(c) = ctx {
        installed.extra_models = c.provisioner.extra_models().to_vec();
        if c.ort_loaded.is_some() {
            installed.active_ort = c.ort_loaded;
        }
    }
    installed
}

fn model_installed(res: &Resource, dir: &Path) -> bool {
    manager::is_installed(res, dir) || res.parts.iter().all(|p| dir.join(p.file_name).is_file())
}

fn is_installed(res: &Resource, installed: &Installed, config: &Config) -> bool {
    match res.kind {
        ResourceKind::Model => model_installed(res, &target_dir(config, res)),
        _ => installed.has(res.id),
    }
}

/// Our install of `res` (manifest names it), not a user-managed directory.
fn ours(res: &Resource, config: &Config) -> bool {
    let dir = target_dir(config, res);
    manager::read_manifest(&manager::manifest_path(res, &dir)).is_some_and(|m| m.id == res.id)
}

/// Enabled config models whose file is part of `res`.
fn models_using(res: &Resource, config: &Config) -> Vec<String> {
    let dir = target_dir(config, res);
    let files: Vec<PathBuf> = res.parts.iter().map(|p| dir.join(p.file_name)).collect();
    config
        .enabled_models()
        .filter(|m| files.contains(&config.data_path(&m.path)))
        .map(|m| m.effective_name())
        .collect()
}

/// Why `res` cannot be removed right now (None = it can).
fn remove_blocked(
    ctx: Option<&ResourcesCtx>,
    res: &Resource,
    config: &Config,
    status: Option<&Status>,
) -> Option<String> {
    if status.is_some_and(|s| !s.state.is_final()) {
        return Some("downloading; wait until it finishes".into());
    }
    let restart = "loaded by this process; restart required (stop the service, then remove it)";
    match res.kind {
        ResourceKind::Model => {
            let users = models_using(res, config);
            if !users.is_empty() {
                return Some(format!(
                    "used by enabled model(s) {}; disable them first",
                    users.join(", ")
                ));
            }
        }
        ResourceKind::OpenVinoRuntime if ctx.is_some_and(|c| c.openvino_loaded) => {
            return Some(restart.into());
        }
        ResourceKind::OnnxRuntime(f) if ctx.is_some_and(|c| c.ort_loaded == Some(f)) => {
            return Some(restart.into());
        }
        ResourceKind::CudaLibs
            if crate::backend::libs::cuda_libs_preloaded()
                .is_some_and(|d| d == target_dir(config, res)) =>
        {
            return Some(restart.into());
        }
        _ => {}
    }
    if res.kind != ResourceKind::Model && !ours(res, config) {
        return Some("not installed by Blue Onyx Prism (user-managed); remove it by hand".into());
    }
    None
}

/// The latest manager status for resource `id` (any target).
fn status_of<'a>(statuses: &'a [Status], id: &str) -> Option<&'a Status> {
    // Prefer an active job over a finished one.
    statuses
        .iter()
        .filter(|s| s.id == id)
        .min_by_key(|s| s.state.is_final())
}

/// Rows for `/v1/resources` and the Resources card.
pub fn rows(ctx: Option<&ResourcesCtx>, config: &Config) -> Vec<ResourceRow> {
    let hw = crate::backend::detect::hardware();
    let installed = installed_snapshot(ctx, config);
    let res: Resolution = resolve::needed(config, hw, &installed);
    let statuses: Vec<Status> = ctx
        .and_then(|c| c.provisioner.manager())
        .map(|m| m.statuses())
        .unwrap_or_default();
    let ort_root = config.data_root().join(crate::backend::libs::ORT_DIR_NAME);
    let active_flavor = crate::backend::libs::read_active_flavor(&ort_root);
    let configured: Vec<PathBuf> = config
        .models
        .iter()
        .map(|m| config.data_path(&m.path))
        .collect();

    platform_resources(ctx)
        .into_iter()
        .map(|r| {
            let status = status_of(&statuses, r.id);
            let on_disk = is_installed(r, &installed, config);
            let need = res.needs.iter().find(|n| n.id() == r.id);
            let optional = res.optional.iter().any(|n| n.id() == r.id);
            let (state, progress, error, retry_in) = match status.map(|s| &s.state) {
                Some(State::Downloading { bytes, total }) => (
                    "downloading",
                    Some(if *total > 0 {
                        (bytes * 100 / total) as u8
                    } else {
                        0
                    }),
                    None,
                    None,
                ),
                Some(State::Queued) => ("queued", None, None, None),
                Some(State::Verifying) => ("verifying", None, None, None),
                Some(State::Extracting) => ("extracting", None, None, None),
                Some(State::Failed {
                    error, retry_at, ..
                }) if !on_disk => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .map_or(0, |d| d.as_secs());
                    (
                        "failed",
                        None,
                        Some(error.clone()),
                        retry_at.map(|t| t.saturating_sub(now)),
                    )
                }
                _ if on_disk => ("installed", None, None, None),
                _ if need.is_some() => ("needed", None, None, None),
                _ if optional => ("optional", None, None, None),
                _ => ("available", None, None, None),
            };
            let state_text = match (status, state) {
                (Some(s), "downloading" | "queued" | "verifying" | "extracting" | "failed") => {
                    s.state.describe()
                }
                (_, "needed") => format!(
                    "needed: {}",
                    need.map_or(String::new(), |n| n.reason.clone())
                ),
                (_, "optional") => "would enable a better device (large download)".into(),
                (_, s) => s.to_string(),
            };
            let flavor = r.flavor();
            let loaded = match r.kind {
                ResourceKind::OpenVinoRuntime => on_disk && ctx.is_some_and(|c| c.openvino_loaded),
                ResourceKind::OnnxRuntime(f) => ctx.is_some_and(|c| c.ort_loaded == Some(f)),
                ResourceKind::CudaLibs => crate::backend::libs::cuda_libs_preloaded().is_some(),
                ResourceKind::Model => false,
            };
            let blocked = if on_disk {
                remove_blocked(ctx, r, config, status)
            } else {
                Some("not installed".into())
            };
            let can_add = r.kind == ResourceKind::Model
                && on_disk
                && !r
                    .parts
                    .iter()
                    .any(|p| configured.contains(&target_dir(config, r).join(p.file_name)));
            ResourceRow {
                id: r.id,
                title: r.title,
                description: r.description,
                kind: kind_str(r.kind),
                group: group_of(r),
                version: r.version,
                license: r.license,
                size: r.size(),
                size_text: catalog::format_size(r.size()),
                large: r.is_large(),
                state,
                state_text,
                busy: matches!(state, "downloading" | "queued" | "verifying" | "extracting"),
                progress,
                error,
                retry_in,
                provides: r.provides.labels(),
                blocking: need.map(|n| n.blocking_models.clone()).unwrap_or_default(),
                active: flavor.is_some()
                    && (flavor.map(|f| f.as_str()) == active_flavor.as_deref()
                        || flavor == ctx.and_then(|c| c.ort_loaded)),
                loaded,
                removable: on_disk && blocked.is_none(),
                remove_blocked: if on_disk { blocked } else { None },
                can_add_to_config: can_add,
                target: target_dir(config, r).display().to_string(),
            }
        })
        .collect()
}

/// Queue `id` for download. Large resources need `confirm_large` unless the config allows them.
/// Returns a message for the user.
pub fn download(
    ctx: &ResourcesCtx,
    config: &Config,
    id: &str,
    confirm_large: bool,
) -> Result<String, ActionError> {
    let res = find(Some(ctx), id)
        .ok_or_else(|| ActionError::NotFound(format!("unknown resource '{id}'")))?;
    let installed = installed_snapshot(Some(ctx), config);
    if is_installed(res, &installed, config) {
        return Ok(format!("{} is already installed", res.title));
    }
    if res.is_large() && !confirm_large && !config.allow_large_downloads {
        return Err(ActionError::Conflict(format!(
            "{} is {} ({}); confirm the large download",
            res.title,
            catalog::format_size(res.size()),
            res.license
        )));
    }
    let mut job = match res.kind {
        ResourceKind::Model => Job::into_dir(res, target_dir(config, res)),
        _ => Job::new(res, &config.data_root()),
    };
    job.activate_ort =
        res.flavor().is_some() && installed.active_ort.is_none() && !installed.ort_pinned;
    let manager = ctx.provisioner.manager_for(config);
    manager.forget(&job.key());
    manager.enqueue(job);
    Ok(format!(
        "downloading {} ({})",
        res.describe(),
        if res.kind == ResourceKind::Model {
            "add it to the config when it is installed"
        } else {
            "used after the next restart"
        }
    ))
}

/// Delete our install of `id` (see the module docs for when it is refused).
pub fn remove(
    ctx: Option<&ResourcesCtx>,
    config: &Config,
    id: &str,
) -> Result<String, ActionError> {
    let res =
        find(ctx, id).ok_or_else(|| ActionError::NotFound(format!("unknown resource '{id}'")))?;
    let installed = installed_snapshot(ctx, config);
    if !is_installed(res, &installed, config) {
        return Err(ActionError::Conflict(format!(
            "{} is not installed",
            res.title
        )));
    }
    let statuses: Vec<Status> = ctx
        .and_then(|c| c.provisioner.manager())
        .map(|m| m.statuses())
        .unwrap_or_default();
    if let Some(why) = remove_blocked(ctx, res, config, status_of(&statuses, res.id)) {
        return Err(ActionError::Conflict(format!(
            "cannot remove {}: {why}",
            res.title
        )));
    }
    let dir = target_dir(config, res);
    let fail =
        |e: std::io::Error, p: &Path| ActionError::Failed(format!("removing {}: {e}", p.display()));
    match res.kind {
        ResourceKind::Model => {
            for p in res.parts {
                let f = dir.join(p.file_name);
                if f.exists() {
                    std::fs::remove_file(&f).map_err(|e| fail(e, &f))?;
                }
            }
            let m = manager::manifest_path(res, &dir);
            let _ = std::fs::remove_file(m);
        }
        _ => {
            std::fs::remove_dir_all(&dir).map_err(|e| fail(e, &dir))?;
            if let (Some(f), Some(parent)) = (res.flavor(), dir.parent())
                && crate::backend::libs::read_active_flavor(parent).as_deref() == Some(f.as_str())
            {
                let _ = std::fs::remove_file(parent.join(crate::backend::libs::ORT_ACTIVE_FILE));
            }
        }
    }
    if let Some(m) = ctx.and_then(|c| c.provisioner.manager()) {
        for s in m.statuses().iter().filter(|s| s.id == res.id) {
            m.forget(&s.key);
        }
    }
    Ok(format!("removed {} from {}", res.title, dir.display()))
}

/// Append installed catalog model `id` to `config.models` (`download-models --add-to-config`
/// rules: disabled when another model is enabled). Returns the outcome text.
pub fn add_to_config(
    ctx: Option<&ResourcesCtx>,
    config: &mut Config,
    id: &str,
) -> Result<String, ActionError> {
    if let Some(file) = strip_prefix_ci(id.trim(), LOCAL_ID_PREFIX) {
        return add_local_to_config(config, file);
    }
    let res =
        find(ctx, id).ok_or_else(|| ActionError::NotFound(format!("unknown resource '{id}'")))?;
    if res.kind != ResourceKind::Model {
        return Err(ActionError::Conflict(format!(
            "{} is not a model",
            res.title
        )));
    }
    let dir = target_dir(config, res);
    if !model_installed(res, &dir) {
        return Err(ActionError::Conflict(format!(
            "{} is not installed; download it first",
            res.title
        )));
    }
    let files: Vec<PathBuf> = res.parts.iter().map(|p| dir.join(p.file_name)).collect();
    let mut models = crate::download::model_configs(&files, &config.data_root());
    if models.is_empty()
        && let catalog::Provides::Model { name, family } = res.provides
    {
        // Not in the static catalog (e.g. an extra test model): same shape by hand.
        let rel = |p: &Path| {
            p.strip_prefix(config.data_root())
                .map(Path::to_path_buf)
                .unwrap_or_else(|_| p.to_path_buf())
        };
        models.push(crate::config::ModelConfig {
            name: Some(name.to_string()),
            path: rel(&files[0]),
            family,
            classes: files.get(1).map(|p| rel(p)),
            ..Default::default()
        });
    }
    let out = crate::download::add_to_config(config, models);
    Ok(describe_outcomes(&out))
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    (s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix))
        .then(|| &s[prefix.len()..])
}

/// Append model file `file` (a plain file name in `models_dir`) with family `auto`.
fn add_local_to_config(config: &mut Config, file: &str) -> Result<String, ActionError> {
    let file = file.trim();
    let plain = !file.is_empty()
        && !file.contains(['/', '\\'])
        && file != "."
        && file != ".."
        && Path::new(file).file_name().and_then(|n| n.to_str()) == Some(file);
    if !plain || model_ext(Path::new(file)).is_none() {
        return Err(ActionError::NotFound(format!(
            "'{file}' is not a model file name (.onnx or .xml in the models directory)"
        )));
    }
    let models_dir = config.data_path(&config.models_dir);
    let full = models_dir.join(file);
    if !full.is_file() {
        return Err(ActionError::NotFound(format!(
            "{} does not exist",
            full.display()
        )));
    }
    // Relative to the data root like downloads (`models/x.onnx`), absolute otherwise.
    let path = full
        .strip_prefix(config.data_root())
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| full.clone());
    let name = Path::new(file)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| file.to_string());
    let out = crate::download::add_to_config(
        config,
        vec![crate::config::ModelConfig {
            name: Some(name),
            path,
            family: crate::model::ModelFamilyKind::Auto,
            ..Default::default()
        }],
    );
    Ok(describe_outcomes(&out))
}

fn describe_outcomes(out: &[crate::download::AddOutcome]) -> String {
    out.iter()
        .map(|o| match o {
            crate::download::AddOutcome::Added {
                name,
                enabled: true,
            } => {
                format!("added '{name}' to the config (enabled; restart to load it)")
            }
            crate::download::AddOutcome::Added { name, .. } => {
                format!("added '{name}' to the config (disabled; enable it in the Models card)")
            }
            crate::download::AddOutcome::Skipped { name } => {
                format!("'{name}' is already in the config")
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Download what `config` needs (respecting `allow_large_downloads`): used after the config
/// page changes a device to a downloadable option. Returns the queued resource titles.
pub fn queue_needs(ctx: &ResourcesCtx, config: &Config) -> Vec<String> {
    let hw = crate::backend::detect::hardware();
    let installed = installed_snapshot(Some(ctx), config);
    let r = resolve::needed(config, hw, &installed);
    if r.needs.is_empty() {
        return Vec::new();
    }
    let manager = ctx.provisioner.manager_for(config);
    manager::jobs_for(&r, &r.needs, &config.data_root())
        .into_iter()
        .map(|j| {
            let title = j.resource.describe();
            manager.forget(&j.key());
            manager.enqueue(j);
            title
        })
        .collect()
}
