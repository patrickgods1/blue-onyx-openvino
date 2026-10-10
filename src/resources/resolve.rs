//! Resolver (docs/PLAN.md, "Resolver"): what the config needs that is not installed yet.
//!
//! [`needed`] is pure over plain data: the [`Config`], the detected [`HardwareInfo`] and an
//! [`Installed`] snapshot. [`detect_installed`] is the small I/O step that builds the snapshot
//! from the download root (and, in a running service, the live [`RuntimeProbe`]).
//!
//! Rules:
//! - **Models**: an enabled model whose file (or default `<stem>.yaml`) is missing and whose file
//!   name matches a catalog model needs that model resource. A missing file that is not in the
//!   catalog is reported in [`ModelPlan::error`] (YOLO26 weights must be exported locally).
//! - **Runtimes**: each enabled model's device plan is computed with the real `select` /
//!   `plan` code on a *hypothetical* probe in which every runtime that is installed **or
//!   downloadable** for this platform is present. The candidates are walked in order and the
//!   first one whose runtime is installed or downloadable (and allowed: downloads over
//!   [`LARGE_DOWNLOAD_BYTES`](super::catalog::LARGE_DOWNLOAD_BYTES) need
//!   `allow_large_downloads`) wins; its missing resources become needs. Large downloads that
//!   were skipped are listed in [`Resolution::optional`] (the UI's "downloadable" options).
//! - **ONNX Runtime flavor**: one per process. The first model (the default model first) that
//!   settles on an ORT device fixes the flavor; later models only consider that flavor. A flavor
//!   other than the active one is downloaded next to it and is active after a restart
//!   ([`Need::after_restart`]). With `onnxruntime_dir` / `ORT_DYLIB_PATH` set the active flavor
//!   is pinned and never replaced.
//! - **`auto_download: false`**: the needs are still computed; [`Resolution::downloads`] is
//!   empty and [`Resolution::manual_message`] names the command to run instead.

use super::catalog::{self, Flavor, Resource};
use crate::backend::detect::{GpuVendor, HardwareInfo};
use crate::backend::plan;
use crate::backend::select::{
    self, EpStatus, OpenVinoProbe, OrtProbe, OvDevice, RuntimeProbe, Selection,
};
use crate::backend::spec::{Device, DeviceSpec, Runtime, Target};
use crate::config::{Config, ModelConfig};
use crate::model::ModelFamilyKind;
use serde::{Serialize, Serializer};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Command that downloads everything the config needs (for `auto_download: false`).
pub const FETCH_FOR_CONFIG: &str = "blue-onyx-prism fetch --for-config";

/// What is present, as plain data (built by [`detect_installed`], or by hand in tests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Installed {
    /// Ids of installed runtime/library resources (`openvino-runtime`, `onnxruntime-cuda`,
    /// `nvidia-cuda-libs`), whether downloaded by us or user-managed (`openvino_dir`).
    pub resources: BTreeSet<String>,
    /// ONNX Runtime flavor this process loads (or would load at startup); None = no ORT library.
    pub active_ort: Option<Flavor>,
    /// `onnxruntime_dir` / `ORT_DYLIB_PATH` is set: the active flavor is user-managed and never
    /// replaced.
    pub ort_pinned: bool,
    /// Model-related files that exist, spelled as in the config (model paths, `<stem>.yaml`
    /// class files, `<stem>.onnx` siblings of `.xml` models).
    pub files: BTreeSet<PathBuf>,
    /// Live probe of the loaded runtimes (running service). When set, an available OpenVINO
    /// counts as installed and the active ORT flavor's real EP status is used.
    pub probe: Option<RuntimeProbe>,
    /// This binary has ONNX Runtime support (the `onnxruntime` feature).
    pub ort_in_build: bool,
}

// Derivable only in builds without the `onnxruntime` feature.
#[allow(clippy::derivable_impls)]
impl Default for Installed {
    fn default() -> Self {
        Self {
            resources: BTreeSet::new(),
            active_ort: None,
            ort_pinned: false,
            files: BTreeSet::new(),
            probe: None,
            ort_in_build: cfg!(feature = "onnxruntime"),
        }
    }
}

impl Installed {
    pub fn has(&self, id: &str) -> bool {
        self.resources.contains(id)
    }

    pub fn has_file(&self, p: &Path) -> bool {
        self.files.contains(p)
    }

    /// Mark resource ids installed (builder for tests and callers).
    pub fn with_resources(mut self, ids: &[&str]) -> Self {
        self.resources.extend(ids.iter().map(|s| s.to_string()));
        self
    }

    /// Mark files present (builder).
    pub fn with_files<P: AsRef<Path>>(mut self, files: &[P]) -> Self {
        self.files
            .extend(files.iter().map(|p| p.as_ref().to_path_buf()));
        self
    }

    /// OpenVINO can be used without downloading anything.
    fn openvino_present(&self) -> bool {
        self.has(catalog::OPENVINO_RUNTIME_ID)
            || self
                .probe
                .as_ref()
                .is_some_and(|p| p.openvino.is_available())
    }

    /// The live probe's OpenVINO, when it is loaded.
    fn live_openvino(&self) -> Option<&OpenVinoProbe> {
        self.probe
            .as_ref()
            .map(|p| &p.openvino)
            .filter(|o| o.is_available())
    }

    /// Active flavor: the snapshot's, else the live probe's.
    fn active_flavor(&self) -> Option<Flavor> {
        self.active_ort.or_else(|| {
            self.live_ort()
                .and_then(|o| o.flavor.as_deref())
                .and_then(Flavor::parse)
        })
    }

    /// The live probe's ONNX Runtime, when a library is loaded.
    fn live_ort(&self) -> Option<&OrtProbe> {
        self.probe
            .as_ref()
            .map(|p| &p.ort)
            .filter(|o| o.is_available())
    }

    fn flavor_present(&self, f: Flavor) -> bool {
        self.has(f.resource_id()) || self.active_flavor() == Some(f)
    }
}

/// One resource to download.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Need {
    #[serde(serialize_with = "ser_resource_id")]
    pub resource: &'static Resource,
    /// Why, e.g. "IPcam-general on ort:cuda:0" or "model file models/IPcam-general.onnx is
    /// missing".
    pub reason: String,
    /// Enabled models that cannot load (on their chosen device) until this is installed.
    pub blocking_models: Vec<String>,
    /// Device option this resource enables for the first blocking model (runtimes only).
    #[serde(serialize_with = "ser_opt_device")]
    pub device: Option<Device>,
    /// Takes effect only after a process restart: a second ONNX Runtime flavor while another one
    /// is active.
    pub after_restart: bool,
    /// Models only: directory to put the files in, as written in the config (relative paths
    /// resolve against the exe dir). Runtimes go to `<download root>/<resource.dest>`.
    pub model_dir: Option<PathBuf>,
}

impl Need {
    pub fn id(&self) -> &'static str {
        self.resource.id
    }

    pub fn size(&self) -> u64 {
        self.resource.size()
    }

    /// Command that downloads just this resource.
    pub fn fetch_command(&self) -> String {
        format!("blue-onyx-prism fetch --resource {}", self.resource.id)
    }
}

fn ser_resource_id<S: Serializer>(r: &&'static Resource, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(r.id)
}

fn ser_opt_device<S: Serializer>(d: &Option<Device>, s: S) -> Result<S::Ok, S::Error> {
    match d {
        Some(d) => s.collect_str(d),
        None => s.serialize_none(),
    }
}

/// What one enabled model will run on once the needs are met.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelPlan {
    pub model: String,
    /// First candidate whose runtime is installed or will be downloaded. None when nothing can
    /// run the model (see `error`).
    #[serde(serialize_with = "ser_opt_device")]
    pub device: Option<Device>,
    /// Better candidates that were passed over, with why ("ort:cuda:0: needs NVIDIA CUDA 12 +
    /// cuDNN 9 libraries (1.9 GB); set allow_large_downloads").
    pub skipped: Vec<String>,
    /// Why the model cannot be provisioned (missing non-catalog file, no device).
    pub error: Option<String>,
}

/// Result of [`needed`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Resolution {
    /// Resources to download, deduplicated, in the order models need them (default model first).
    pub needs: Vec<Need>,
    /// Downloads that would enable a better device but are gated by `allow_large_downloads`.
    pub optional: Vec<Need>,
    /// Per enabled model, default model first.
    pub models: Vec<ModelPlan>,
    /// ONNX Runtime flavor the config wants active; None = no model settles on ONNX Runtime.
    pub ort_flavor: Option<Flavor>,
    /// Flavor active now (from [`Installed`]).
    pub active_ort: Option<Flavor>,
    /// The ONNX Runtime library is user-managed (`onnxruntime_dir` / `ORT_DYLIB_PATH`).
    pub ort_pinned: bool,
    /// Config `auto_download`.
    pub auto_download: bool,
}

impl Resolution {
    /// Needs the service should download now: all of them with `auto_download`, none without.
    pub fn downloads(&self) -> &[Need] {
        if self.auto_download { &self.needs } else { &[] }
    }

    /// Nothing is missing.
    pub fn is_satisfied(&self) -> bool {
        self.needs.is_empty()
    }

    /// The wanted ORT flavor differs from the active one: it is used after a restart.
    pub fn ort_restart_required(&self) -> bool {
        matches!((self.ort_flavor, self.active_ort), (Some(w), Some(a)) if w != a)
    }

    /// The flavor to mark active (`onnxruntime/active.txt`) so the next start loads it: the
    /// wanted flavor when it differs from the active one and ORT is not user-managed.
    pub fn ort_activation(&self) -> Option<Flavor> {
        let want = self.ort_flavor?;
        (!self.ort_pinned && self.active_ort != Some(want)).then_some(want)
    }

    /// Total size of [`Self::needs`] in bytes.
    pub fn total_size(&self) -> u64 {
        self.needs.iter().map(Need::size).sum()
    }

    /// Needs that block `model` (effective name).
    pub fn blocking<'a>(&'a self, model: &'a str) -> impl Iterator<Item = &'a Need> + 'a {
        self.needs
            .iter()
            .filter(move |n| n.blocking_models.iter().any(|m| m == model))
    }

    pub fn model(&self, name: &str) -> Option<&ModelPlan> {
        self.models.iter().find(|m| m.model == name)
    }

    /// With `auto_download: false` and something missing: what is missing and the command that
    /// fetches it.
    pub fn manual_message(&self) -> Option<String> {
        if self.auto_download || self.needs.is_empty() {
            return None;
        }
        let list: Vec<String> = self.needs.iter().map(|n| n.resource.describe()).collect();
        Some(format!(
            "auto_download is off; missing: {}. Run `{FETCH_FOR_CONFIG}` (on a machine with \
             network access, then copy the directory), or enable auto_download",
            list.join(", ")
        ))
    }
}

/// The resources `config` needs on `hw` given what is `installed`. Pure.
pub fn needed(config: &Config, hw: &HardwareInfo, installed: &Installed) -> Resolution {
    let mut r = Resolver {
        config,
        hw,
        installed,
        chosen: None,
        out: Resolution {
            needs: Vec::new(),
            optional: Vec::new(),
            models: Vec::new(),
            ort_flavor: None,
            active_ort: installed.active_flavor(),
            ort_pinned: installed.ort_pinned,
            auto_download: config.auto_download,
        },
    };
    if installed.ort_pinned {
        // A user-managed library is what loads; never plan another flavor.
        r.chosen = Some(installed.active_flavor().unwrap_or(Flavor::Cpu));
    }
    for m in ordered_models(config) {
        r.model(m);
    }
    r.out.ort_flavor = r.chosen.filter(|_| {
        r.out
            .models
            .iter()
            .any(|m| m.device.is_some_and(|d| d.runtime == Runtime::Ort))
    });
    let needs: BTreeSet<&str> = r.out.needs.iter().map(Need::id).collect();
    r.out.optional.retain(|n| !needs.contains(n.id()));
    r.out
}

/// Enabled models, the effective default model first.
fn ordered_models(config: &Config) -> Vec<&ModelConfig> {
    let first = config.effective_default_index();
    first
        .map(|i| &config.models[i])
        .into_iter()
        .chain(
            config
                .models
                .iter()
                .enumerate()
                .filter(|(i, m)| m.enabled && Some(*i) != first)
                .map(|(_, m)| m),
        )
        .collect()
}

/// Catalog model for a configured model path: an `.onnx` file named like a catalog model.
pub fn catalog_model_for(m: &ModelConfig) -> Option<&'static Resource> {
    let is_onnx = m
        .path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
    if !is_onnx {
        return None;
    }
    catalog::model(&m.path.file_name()?.to_string_lossy())
}

/// The class file a model uses by default: `classes`, else `<stem>.yaml` next to it.
fn classes_path(m: &ModelConfig) -> PathBuf {
    m.classes
        .clone()
        .unwrap_or_else(|| m.path.with_extension("yaml"))
}

/// Files [`detect_installed`] checks for one model.
fn model_files(m: &ModelConfig) -> [PathBuf; 3] {
    [
        m.path.clone(),
        classes_path(m),
        m.path.with_extension("onnx"),
    ]
}

struct Resolver<'a> {
    config: &'a Config,
    hw: &'a HardwareInfo,
    installed: &'a Installed,
    /// ORT flavor fixed by an earlier model (or pinned).
    chosen: Option<Flavor>,
    out: Resolution,
}

/// Outcome of checking one candidate.
enum Check {
    /// Usable; these resources are missing (possibly none).
    Ok(Vec<&'static Resource>, Option<Flavor>),
    /// Usable only with a large download that is not allowed.
    Large(Vec<&'static Resource>, String),
    /// Cannot run here.
    No(String),
}

impl Resolver<'_> {
    fn os(&self) -> &str {
        &self.hw.os
    }

    fn arch(&self) -> &str {
        &self.hw.arch
    }

    fn model(&mut self, m: &ModelConfig) {
        let name = m.effective_name();
        let entry = catalog_model_for(m);
        let family = match (m.family, entry.map(|e| e.provides)) {
            (ModelFamilyKind::Auto, Some(catalog::Provides::Model { family, .. })) => family,
            (f, _) => f,
        };
        let mut plan = ModelPlan {
            model: name.clone(),
            device: None,
            skipped: Vec::new(),
            error: None,
        };

        // Model files.
        let mut model_need: Option<String> = None;
        if !self.installed.has_file(&m.path) {
            match entry {
                Some(_) => model_need = Some(format!("model file {} is missing", m.path.display())),
                None => {
                    plan.error = Some(missing_model_message(&m.path));
                    self.out.models.push(plan);
                    return;
                }
            }
        } else if entry.is_some()
            && m.classes.is_none()
            && !self.installed.has_file(&classes_path(m))
        {
            model_need = Some(format!(
                "class file {} is missing",
                classes_path(m).display()
            ));
        }
        let has_onnx = m
            .path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
            || self.installed.has_file(&m.path.with_extension("onnx"));

        // Device plan.
        let spec = match self.config.device_spec_for(m) {
            Ok(s) => s,
            Err(e) => {
                plan.error = Some(format!("{e:#}"));
                self.out.models.push(plan);
                return;
            }
        };
        let (probe, ov_names) = self.hypothetical_probe();
        let sel = select::select(self.hw, &probe, has_onnx);
        let spec = if self.config.force_cpu {
            DeviceSpec::Device(plan::best_cpu(&sel))
        } else {
            spec
        };
        let candidates = plan::plan_for(&spec, &sel, &ov_names);

        let mut accepted: Option<(Device, Vec<&'static Resource>, Option<Flavor>)> = None;
        for cand in &candidates {
            match self.check(cand.device, family, &sel) {
                Check::Ok(res, flavor) => {
                    accepted = Some((cand.device, res, flavor));
                    break;
                }
                Check::Large(res, why) => {
                    plan.skipped.push(format!("{}: {why}", cand.device));
                    for r in res {
                        let reason = format!("{name} on {}", cand.device);
                        let need = self.need(r, reason, &name, Some(cand.device), None);
                        push_need(&mut self.out.optional, need);
                    }
                }
                Check::No(why) => plan.skipped.push(format!("{}: {why}", cand.device)),
            }
        }
        let Some((device, resources, flavor)) = accepted else {
            plan.error = Some(format!(
                "no device can run this model here ({})",
                if plan.skipped.is_empty() {
                    "no candidates".to_string()
                } else {
                    plan.skipped.join("; ")
                }
            ));
            self.out.models.push(plan);
            return;
        };
        plan.device = Some(device);
        if let Some(f) = flavor {
            self.chosen = Some(f);
        }

        if let (Some(reason), Some(entry)) = (model_need, entry) {
            let dir = m.path.parent().map(Path::to_path_buf).unwrap_or_default();
            let need = self.need(entry, reason, &name, None, Some(dir));
            push_need(&mut self.out.needs, need);
        }
        for r in resources {
            let need = self.need(r, format!("{name} on {device}"), &name, Some(device), None);
            push_need(&mut self.out.needs, need);
        }
        self.out.models.push(plan);
    }

    fn need(
        &self,
        resource: &'static Resource,
        reason: String,
        model: &str,
        device: Option<Device>,
        model_dir: Option<PathBuf>,
    ) -> Need {
        let active = self.installed.active_flavor();
        let after_restart = resource
            .flavor()
            .is_some_and(|f| active.is_some_and(|a| a != f));
        Need {
            resource,
            reason,
            blocking_models: vec![model.to_string()],
            device,
            after_restart,
            model_dir,
        }
    }

    /// ORT flavors this model may use: the chosen one, else every flavor installed or
    /// downloadable here.
    fn usable_flavors(&self) -> Vec<Flavor> {
        if let Some(c) = self.chosen {
            return vec![c];
        }
        Flavor::ALL
            .into_iter()
            .filter(|f| {
                self.installed.flavor_present(*f)
                    || catalog::onnxruntime(self.os(), self.arch(), *f).is_some()
            })
            .collect()
    }

    fn cuda_libs_present(&self) -> bool {
        self.installed.has(catalog::CUDA_LIBS_ID)
    }

    /// Real status of `target` from the live probe, when `flavor` is the loaded one.
    fn live_status(&self, flavor: Flavor, target: Target) -> Option<Option<String>> {
        let ort = self.installed.live_ort()?;
        (self.installed.active_flavor() == Some(flavor)).then(|| ort.ep_error(target))
    }

    /// A probe in which every runtime that is installed or downloadable is present, plus the
    /// OpenVINO device names for `plan_for`.
    fn hypothetical_probe(&self) -> (RuntimeProbe, Vec<String>) {
        let openvino = self.hypothetical_openvino();
        let names = openvino.devices.iter().map(|d| d.name.clone()).collect();
        (
            RuntimeProbe {
                openvino,
                ort: self.hypothetical_ort(),
            },
            names,
        )
    }

    fn hypothetical_openvino(&self) -> OpenVinoProbe {
        if let Some(live) = self.installed.live_openvino() {
            return live.clone();
        }
        let package = catalog::openvino_runtime(self.os(), self.arch());
        if let Some(p) = &self.installed.probe
            && self.installed.has(catalog::OPENVINO_RUNTIME_ID)
        {
            // Installed but failed to load: downloading again would not help.
            return p.openvino.clone();
        }
        if !self.installed.has(catalog::OPENVINO_RUNTIME_ID) && package.is_none() {
            return OpenVinoProbe::unavailable(format!(
                "no OpenVINO runtime package for {}/{}",
                self.os(),
                self.arch()
            ));
        }
        // OpenVINO lists the CPU, and the Intel GPUs where the GPU plugin ships.
        let gpu_plugin =
            package.is_none_or(|p| p.provides.provides_target(Runtime::OpenVino, Target::Gpu));
        let mut devices = vec![OvDevice::named("CPU")];
        if gpu_plugin {
            let intel: Vec<_> = self.hw.gpus_of(GpuVendor::Intel).collect();
            for (i, g) in intel.iter().enumerate() {
                devices.push(OvDevice {
                    name: if intel.len() == 1 {
                        "GPU".to_string()
                    } else {
                        format!("GPU.{i}")
                    },
                    full_name: g.name.clone(),
                    device_type: Some(if g.discrete { "discrete" } else { "integrated" }.into()),
                });
            }
        }
        OpenVinoProbe {
            error: None,
            devices,
        }
    }

    fn hypothetical_ort(&self) -> OrtProbe {
        if !self.installed.ort_in_build {
            return OrtProbe::not_in_build();
        }
        let flavors = self.usable_flavors();
        if flavors.is_empty() {
            return OrtProbe::unavailable(format!(
                "no ONNX Runtime package for {}/{}",
                self.os(),
                self.arch()
            ));
        }
        let mut providers: Vec<EpStatus> = Vec::new();
        for f in &flavors {
            for &t in f.targets() {
                let status = self.ep_status(*f, t);
                match providers.iter_mut().find(|p| p.target == t) {
                    Some(p) if p.error.is_some() => *p = status,
                    Some(_) => {}
                    None => providers.push(status),
                }
            }
        }
        OrtProbe {
            error: None,
            flavor: (flavors.len() == 1).then(|| flavors[0].as_str().to_string()),
            providers,
        }
    }

    /// Hypothetical EP status of `target` in `flavor`.
    fn ep_status(&self, flavor: Flavor, target: Target) -> EpStatus {
        let live = self.live_status(flavor, target);
        if let Some(None) = live {
            return EpStatus::usable(target);
        }
        match target {
            Target::TensorRt => EpStatus::broken(
                target,
                live.flatten().unwrap_or_else(|| {
                    "TensorRT libraries are not downloadable (install TensorRT 10 for CUDA 12)"
                        .to_string()
                }),
            ),
            Target::Cuda => match live.flatten() {
                // Broken although the libraries are installed: a download will not help.
                Some(e) if self.cuda_libs_present() => EpStatus::broken(target, e),
                _ if self.cuda_libs_present()
                    || catalog::cuda_libs_for(self.os(), self.arch()).is_some() =>
                {
                    EpStatus::usable(target)
                }
                e => EpStatus::broken(
                    target,
                    e.unwrap_or_else(|| {
                        format!(
                            "CUDA 12/cuDNN 9 libraries are not downloadable for {}/{}",
                            self.os(),
                            self.arch()
                        )
                    }),
                ),
            },
            _ => match live.flatten() {
                Some(e) => EpStatus::broken(target, e),
                None => EpStatus::usable(target),
            },
        }
    }

    /// Whether `device` can be provisioned, and with what.
    fn check(&self, device: Device, family: ModelFamilyKind, sel: &Selection) -> Check {
        if device.target == Target::CoreMl && family == ModelFamilyKind::RtDetr {
            return Check::No("RT-DETR models are not supported by CoreML".to_string());
        }
        match find_option(sel, &device) {
            None => return Check::No("not available on this machine".to_string()),
            Some(o) if !o.runnable => {
                return Check::No(o.reason.clone().unwrap_or_else(|| "cannot run".into()));
            }
            Some(_) => {}
        }
        let (missing, flavor) = match device.runtime {
            Runtime::OpenVino => {
                if self.installed.openvino_present() {
                    (Vec::new(), None)
                } else {
                    match catalog::openvino_runtime(self.os(), self.arch()) {
                        Some(r) => (vec![r], None),
                        None => return Check::No("no OpenVINO runtime package".to_string()),
                    }
                }
            }
            Runtime::Ort => match self.ort_resources(device.target) {
                Ok(v) => v,
                Err(e) => return Check::No(e),
            },
        };
        let large: Vec<&'static Resource> =
            missing.iter().copied().filter(|r| r.is_large()).collect();
        if !large.is_empty() && !self.config.allow_large_downloads {
            let what: Vec<String> = large.iter().map(|r| r.describe()).collect();
            return Check::Large(
                missing,
                format!(
                    "needs {}; set allow_large_downloads to download it",
                    what.join(", ")
                ),
            );
        }
        Check::Ok(missing, flavor)
    }

    /// Missing resources (and the flavor) for an ONNX Runtime target.
    fn ort_resources(
        &self,
        target: Target,
    ) -> Result<(Vec<&'static Resource>, Option<Flavor>), String> {
        let (os, arch) = (self.os(), self.arch());
        let flavor = match Flavor::providing(target) {
            Some(f) => f,
            None => self
                .chosen
                .or_else(|| self.installed.active_flavor())
                .or_else(|| {
                    // Every flavor has the CPU EP. On macOS the CoreML package is the CPU
                    // package, so take CoreML there (later models can then use it); elsewhere
                    // the plain CPU package.
                    let have = catalog::ort_flavors(os, arch);
                    [Flavor::CoreMl, Flavor::Cpu]
                        .into_iter()
                        .find(|f| {
                            have.contains(f)
                                && (*f == Flavor::Cpu
                                    || catalog::onnxruntime(os, arch, *f).map(|r| r.parts)
                                        == catalog::onnxruntime(os, arch, Flavor::Cpu)
                                            .map(|r| r.parts))
                        })
                        .or_else(|| have.first().copied())
                })
                .ok_or_else(|| format!("no ONNX Runtime package for {os}/{arch}"))?,
        };
        if let Some(c) = self.chosen
            && c != flavor
        {
            return Err(format!(
                "needs ONNX Runtime {flavor}, but {c} is the flavor in use (one per process)"
            ));
        }
        let mut missing = Vec::new();
        if !self.installed.flavor_present(flavor) {
            missing.push(
                catalog::onnxruntime(os, arch, flavor)
                    .ok_or_else(|| format!("no ONNX Runtime {flavor} package for {os}/{arch}"))?,
            );
        }
        if matches!(target, Target::Cuda | Target::TensorRt)
            && !self.cuda_libs_present()
            && self.live_status(Flavor::Cuda, target) != Some(None)
        {
            missing.push(
                catalog::cuda_libs_for(os, arch)
                    .ok_or_else(|| format!("no CUDA libraries package for {os}/{arch}"))?,
            );
        }
        Ok((missing, Some(flavor)))
    }
}

/// The option for `device`; a spec without an index (`ort:cuda`, `ort:directml`) matches the
/// first option of that runtime and target, as the backends default to the first device.
fn find_option<'a>(sel: &'a Selection, device: &Device) -> Option<&'a select::DeviceOption> {
    sel.option(device).or_else(|| {
        device.index.is_none().then(|| {
            sel.options
                .iter()
                .find(|o| o.spec.runtime == device.runtime && o.spec.target == device.target)
        })?
    })
}

/// Add `need`, merging with an existing need for the same resource.
fn push_need(list: &mut Vec<Need>, need: Need) {
    match list.iter_mut().find(|n| n.resource.id == need.resource.id) {
        Some(n) => {
            for m in need.blocking_models {
                if !n.blocking_models.contains(&m) {
                    n.blocking_models.push(m);
                }
            }
            n.after_restart |= need.after_restart;
            n.device = n.device.or(need.device);
        }
        None => list.push(need),
    }
}

fn missing_model_message(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if stem.starts_with("yolo26") {
        format!(
            "model file {} not found; YOLO26 weights are AGPL-3.0 and cannot be downloaded: \
             export them with scripts/export_yolo26.py",
            path.display()
        )
    } else {
        format!(
            "model file {} not found and not in the download catalog (`list-models` shows \
             what can be downloaded)",
            path.display()
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Installed-state detection (the only I/O in this module)
// ---------------------------------------------------------------------------------------------

/// Root that runtime resources install under: `download_dir`, else the exe dir.
pub fn download_root(config: &Config) -> PathBuf {
    config
        .download_dir
        .as_deref()
        .map(crate::resolve_path)
        .unwrap_or_else(crate::exe_dir)
}

/// `dir` holds a completed install of resource `id`: the manager's `.installed.json` names it,
/// or (installs from before the manager) a `VERSION` file is present.
fn installed_in(dir: &Path, id: &str) -> bool {
    match super::manager::read_manifest(&dir.join(super::manager::MANIFEST_FILE)) {
        Some(m) => m.id == id,
        None => dir.join("VERSION").is_file(),
    }
}

fn dir_non_empty(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|mut rd| rd.next().is_some())
}

/// Snapshot of what is installed under `root` (see [`download_root`]) for `config`.
/// `probe` is the live runtime probe of a running service, if any.
pub fn detect_installed(config: &Config, root: &Path, probe: Option<RuntimeProbe>) -> Installed {
    let mut out = Installed {
        probe,
        ..Installed::default()
    };

    // OpenVINO: our install, or a user-managed `openvino_dir`.
    let user_ov = config
        .openvino_dir
        .as_deref()
        .map(crate::resolve_path)
        .is_some_and(|d| d.exists());
    let ov_dir = root.join(crate::backend::libs::BUNDLED_DIR_NAME);
    if user_ov || installed_in(&ov_dir, catalog::OPENVINO_RUNTIME_ID) {
        out.resources
            .insert(catalog::OPENVINO_RUNTIME_ID.to_string());
    }

    // ONNX Runtime: the library the loader would pick, plus per-flavor installs.
    let ort_root = root.join(crate::backend::libs::ORT_DIR_NAME);
    let env = std::env::var_os(crate::backend::libs::ENV_ORT_DYLIB_PATH)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    out.ort_pinned = config.onnxruntime_dir.is_some() || env.is_some();
    let lookup = crate::backend::libs::find_onnxruntime_from(
        config.onnxruntime_dir.as_deref(),
        env.as_deref(),
        &ort_root,
    );
    if let Some(lib) = &lookup.library {
        let flavor = lib
            .parent()
            .and_then(crate::backend::libs::read_ort_flavor)
            .and_then(|f| Flavor::parse(&f))
            .unwrap_or(Flavor::Cpu);
        out.active_ort = Some(flavor);
        out.resources.insert(flavor.resource_id().to_string());
    }
    for f in Flavor::ALL {
        let dir = root.join(f.dest());
        let by_manifest = super::manager::read_manifest(&dir.join(super::manager::MANIFEST_FILE))
            .is_some_and(|m| m.id == f.resource_id());
        if by_manifest || crate::backend::libs::read_ort_flavor(&dir).as_deref() == Some(f.as_str())
        {
            out.resources.insert(f.resource_id().to_string());
        }
    }
    let cuda_dir = root.join(catalog::CUDA_LIBS_DEST);
    let cuda_manifest =
        super::manager::read_manifest(&cuda_dir.join(super::manager::MANIFEST_FILE));
    if cuda_manifest.map_or(dir_non_empty(&cuda_dir), |m| m.id == catalog::CUDA_LIBS_ID) {
        out.resources.insert(catalog::CUDA_LIBS_ID.to_string());
    }

    // Model files, spelled as in the config.
    for m in config.enabled_models() {
        for p in model_files(m) {
            if crate::resolve_path(&p).is_file() {
                out.files.insert(p);
            }
        }
    }
    out
}
