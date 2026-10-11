//! Device options and the `auto` ranking (docs/PLAN.md, "Auto ranking"). Pure: the inputs are
//! the detected [`HardwareInfo`], a [`RuntimeProbe`] of what each runtime offers, and whether the
//! model has an `.onnx` file; no runtime library is touched here.
//!
//! `auto` is the runnable options in this order:
//! 1. NVIDIA GPU with a usable CUDA EP -> `ort:cuda:<ordinal of the largest-VRAM NVIDIA GPU>`;
//!    a GPU whose compute capability the pinned ONNX Runtime CUDA build has no kernels for
//!    (e.g. Pascal 6.1 on Windows, see [`ort_cuda_unsupported`]) is not runnable on
//!    CUDA/TensorRT and falls to 3
//! 2. Intel GPUs listed by OpenVINO -> `openvino:gpu` (a single GPU), or the discrete ones
//!    (`DEVICE_TYPE=discrete`, Arc) first and then the integrated ones as `openvino:gpu.N`
//! 3. Windows only: DirectML on GPUs not already covered by 1 or 2 (AMD, NVIDIA without CUDA,
//!    others), discrete and larger VRAM first -> `ort:directml:<DXGI index>`
//! 4. macOS arm64 with the CoreML EP -> `ort:coreml`
//! 5. CPU: `openvino:cpu`, else `ort:cpu`
//!
//! TensorRT and NPU are listed but never picked automatically. ONNX Runtime options need an
//! `.onnx` file: a model given as `.xml` uses a sibling `<stem>.onnx`, otherwise its ORT options
//! are "needs ONNX export" and `auto` skips them.

use super::detect::{GpuAdapter, GpuVendor, HardwareInfo};
use super::spec::{Device, Runtime, Target};
use crate::resources::catalog;
use serde::{Serialize, Serializer};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// One device OpenVINO lists.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OvDevice {
    /// "CPU", "GPU", "GPU.0", "NPU".
    pub name: String,
    /// FULL_DEVICE_NAME, e.g. "Intel(R) UHD Graphics 630 (iGPU)"; empty when unknown.
    pub full_name: String,
    /// DEVICE_TYPE ("integrated" / "discrete") for GPUs, when queried.
    pub device_type: Option<String>,
}

impl OvDevice {
    /// A device known only by name.
    pub fn named(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Self::default()
        }
    }

    pub fn is_gpu(&self) -> bool {
        self.name.starts_with("GPU")
    }

    /// `DEVICE_TYPE=discrete`, or a full name marked "(dGPU)" when the type is unknown.
    pub fn is_discrete(&self) -> bool {
        match self.device_type.as_deref() {
            Some(t) => t.trim().eq_ignore_ascii_case("discrete"),
            None => self.full_name.contains("(dGPU)"),
        }
    }

    /// Intel GPU, or unknown (OpenVINO's GPU plugin targets Intel; non-Intel OpenCL devices it may
    /// list are not picked by `auto`).
    pub fn is_intel_or_unknown(&self) -> bool {
        self.full_name.is_empty() || self.full_name.contains("Intel")
    }

    /// Option spec: `GPU` -> `openvino:gpu`, `GPU.1` -> `openvino:gpu.1`, `CPU`, `NPU`.
    pub fn spec(&self) -> Option<Device> {
        match super::spec::parse(&self.name).ok()?.device() {
            Some(d) if d.runtime == Runtime::OpenVino => Some(*d),
            _ => None,
        }
    }
}

/// What OpenVINO offers on this machine.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpenVinoProbe {
    /// Why OpenVINO cannot be used (None = initialized).
    pub error: Option<String>,
    pub devices: Vec<OvDevice>,
}

impl OpenVinoProbe {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            error: Some(reason.into()),
            devices: Vec::new(),
        }
    }

    /// Initialized OpenVINO that lists these device names (no full names or types).
    pub fn with_device_names(names: &[String]) -> Self {
        Self {
            error: None,
            devices: names.iter().map(|n| OvDevice::named(n)).collect(),
        }
    }

    pub fn is_available(&self) -> bool {
        self.error.is_none()
    }

    fn gpus(&self) -> impl Iterator<Item = &OvDevice> {
        self.devices.iter().filter(|d| d.is_gpu())
    }

    fn find(&self, name: &str) -> Option<&OvDevice> {
        self.devices.iter().find(|d| d.name == name)
    }
}

/// Status of one ONNX Runtime execution provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpStatus {
    /// `Cuda`, `TensorRt`, `DirectMl`, `CoreMl` or `Cpu`.
    pub target: Target,
    /// Why the EP cannot be used (None = usable), e.g. "CUDA 12/cuDNN 9 not loadable".
    pub error: Option<String>,
}

impl EpStatus {
    pub fn usable(target: Target) -> Self {
        Self {
            target,
            error: None,
        }
    }

    pub fn broken(target: Target, reason: impl Into<String>) -> Self {
        Self {
            target,
            error: Some(reason.into()),
        }
    }
}

/// What ONNX Runtime offers. Phase 6.3 fills this from the loaded library; until then it is
/// [`OrtProbe::not_in_build`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrtProbe {
    /// Why ONNX Runtime cannot be used at all (None = library loaded), e.g. "run
    /// `setup-onnxruntime`".
    pub error: Option<String>,
    /// Installed package flavor ("cpu", "gpu", "directml", "coreml"), when known.
    pub flavor: Option<String>,
    /// EPs the library reports. An EP missing from the list is not in the installed flavor.
    pub providers: Vec<EpStatus>,
}

impl Default for OrtProbe {
    fn default() -> Self {
        Self::not_in_build()
    }
}

impl OrtProbe {
    /// This binary has no ONNX Runtime support.
    pub fn not_in_build() -> Self {
        Self::unavailable(super::ORT_UNAVAILABLE)
    }

    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            error: Some(reason.into()),
            flavor: None,
            providers: Vec::new(),
        }
    }

    pub fn installed(flavor: &str, providers: Vec<EpStatus>) -> Self {
        Self {
            error: None,
            flavor: Some(flavor.to_string()),
            providers,
        }
    }

    pub fn is_available(&self) -> bool {
        self.error.is_none()
    }

    /// Why `target` cannot run (None = usable).
    pub fn ep_error(&self, target: Target) -> Option<String> {
        if let Some(e) = &self.error {
            return Some(e.clone());
        }
        match self.providers.iter().find(|p| p.target == target) {
            Some(p) => p.error.clone(),
            None => Some(match &self.flavor {
                Some(f) => format!(
                    "the installed ONNX Runtime ({f}) has no {} execution provider",
                    target.display_name()
                ),
                None => format!(
                    "the installed ONNX Runtime has no {} execution provider",
                    target.display_name()
                ),
            }),
        }
    }
}

/// Everything the runtimes report, captured once at startup.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RuntimeProbe {
    pub openvino: OpenVinoProbe,
    pub ort: OrtProbe,
}

impl RuntimeProbe {
    /// OpenVINO listing `names`, no ONNX Runtime.
    pub fn openvino_only(names: &[String]) -> Self {
        Self {
            openvino: OpenVinoProbe::with_device_names(names),
            ort: OrtProbe::not_in_build(),
        }
    }
}

/// One selectable device. Three states: runnable; downloadable (not runnable, but `download`
/// says what to fetch to make it so); unavailable (not runnable, `reason` says why).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceOption {
    #[serde(serialize_with = "ser_display")]
    pub spec: Device,
    /// Human readable, e.g. "OpenVINO GPU (Intel(R) UHD Graphics 630 (iGPU))".
    pub label: String,
    pub runnable: bool,
    /// Why it cannot run (always set when `runnable` is false).
    pub reason: Option<String>,
    /// Not runnable now, but downloading these resources makes it runnable.
    pub download: Option<DownloadOffer>,
}

/// What a "downloadable" device option needs (from `resources::resolve::downloadable_options`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DownloadOffer {
    /// Resource ids, e.g. `["onnxruntime-cuda", "nvidia-cuda-libs"]`.
    pub resources: Vec<String>,
    /// Total download size in bytes.
    pub size: u64,
    /// Over the large-download threshold (needs `allow_large_downloads`).
    pub large: bool,
    /// "will download ONNX Runtime CUDA/TensorRT 1.24.4 (281 MB)".
    pub summary: String,
}

impl DeviceOption {
    /// Not runnable now, but downloadable.
    pub fn is_downloadable(&self) -> bool {
        !self.runnable && self.download.is_some()
    }

    /// "ok", "download" or "no".
    pub fn status(&self) -> &'static str {
        if self.runnable {
            "ok"
        } else if self.download.is_some() {
            "download"
        } else {
            "no"
        }
    }
}

fn ser_display<S: Serializer>(d: &Device, s: S) -> Result<S::Ok, S::Error> {
    s.collect_str(d)
}

/// All options for one model plus the `auto` ranking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Selection {
    /// Every option for this machine, roughly in ranking order.
    pub options: Vec<DeviceOption>,
    /// What `auto` tries, in order: runnable options only, CPU last. Empty only when no runtime
    /// can run anything.
    #[serde(serialize_with = "ser_devices")]
    pub auto: Vec<Device>,
}

fn ser_devices<S: Serializer>(v: &[Device], s: S) -> Result<S::Ok, S::Error> {
    s.collect_seq(v.iter().map(|d| d.to_string()))
}

impl Selection {
    pub fn option(&self, spec: &Device) -> Option<&DeviceOption> {
        self.options.iter().find(|o| &o.spec == spec)
    }

    /// The device `auto` tries first.
    pub fn auto_pick(&self) -> Option<&DeviceOption> {
        self.auto.first().and_then(|d| self.option(d))
    }

    /// Mark `spec` downloadable with `offer`. A runnable option is left alone (and `auto` is
    /// never changed: it only ever holds runnable options); an option this machine's selection
    /// does not list (e.g. OpenVINO GPU before OpenVINO is installed) is added, not runnable.
    pub fn offer_download(&mut self, spec: Device, label: &str, offer: DownloadOffer) {
        match self.options.iter_mut().find(|o| o.spec == spec) {
            Some(o) if o.runnable => {}
            Some(o) => o.download = Some(offer),
            None => self.options.push(DeviceOption {
                spec,
                label: label.to_string(),
                runnable: false,
                reason: Some("not installed".to_string()),
                download: Some(offer),
            }),
        }
    }

    /// Some GPU-class option can run (`canUseGPU`).
    pub fn any_gpu_runnable(&self) -> bool {
        self.options
            .iter()
            .any(|o| o.runnable && o.spec.is_gpu_like())
    }
}

/// The `.onnx` file ONNX Runtime would load for `model`: the model itself when it is `.onnx`,
/// else a sibling `<stem>.onnx` (for an `.xml` IR). None when there is none on disk.
pub fn onnx_path_for(model: &Path) -> Option<PathBuf> {
    let is_onnx = |p: &Path| {
        p.extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
    };
    if is_onnx(model) {
        return model.is_file().then(|| model.to_path_buf());
    }
    let sibling = model.with_extension("onnx");
    sibling.is_file().then_some(sibling)
}

/// Reason given to ORT options of a model without an `.onnx` file.
pub const NEEDS_ONNX: &str = "needs ONNX export (no <stem>.onnx next to the model)";

fn ov_label(kind: &str, dev: Option<&OvDevice>) -> String {
    match dev.map(|d| d.full_name.as_str()).filter(|n| !n.is_empty()) {
        Some(n) => format!("OpenVINO {kind} ({n})"),
        None => format!("OpenVINO {kind}"),
    }
}

fn gpu_label(target: Target, g: &GpuAdapter) -> String {
    if g.vram_mb > 0 {
        format!(
            "ONNX Runtime {} ({}, {} MB)",
            target.display_name(),
            g.name,
            g.vram_mb
        )
    } else {
        format!("ONNX Runtime {} ({})", target.display_name(), g.name)
    }
}

fn device(runtime: Runtime, target: Target, index: Option<u32>) -> Device {
    Device::new(runtime, target, index).expect("valid runtime/target combination")
}

/// Builds the option list; `push_ort` applies the ORT and ONNX checks.
struct Builder<'a> {
    probe: &'a RuntimeProbe,
    has_onnx: bool,
    options: Vec<DeviceOption>,
}

impl Builder<'_> {
    fn push(&mut self, spec: Device, label: String, reason: Option<String>) {
        self.options.push(DeviceOption {
            spec,
            label,
            runnable: reason.is_none(),
            reason,
            download: None,
        });
    }

    fn push_ort(&mut self, target: Target, index: Option<u32>, label: String) {
        self.push_ort_unless(target, index, label, None);
    }

    /// `push_ort`, with `hw_reason` (the hardware cannot run it) taking precedence.
    fn push_ort_unless(
        &mut self,
        target: Target,
        index: Option<u32>,
        label: String,
        hw_reason: Option<String>,
    ) {
        let reason = hw_reason
            .or_else(|| self.probe.ort.ep_error(target))
            .or_else(|| (!self.has_onnx).then(|| NEEDS_ONNX.to_string()));
        self.push(device(Runtime::Ort, target, index), label, reason);
    }

    fn ov_reason(&self) -> Option<String> {
        self.probe.openvino.error.clone()
    }

    fn runnable(&self, spec: &Device) -> bool {
        self.options.iter().any(|o| &o.spec == spec && o.runnable)
    }
}

/// Every option for this machine and model, and the `auto` ranking.
///
/// `has_onnx`: the model has an `.onnx` file ONNX Runtime can load (see [`onnx_path_for`]); pass
/// true when listing options without a particular model.
pub fn select(hw: &HardwareInfo, probe: &RuntimeProbe, has_onnx: bool) -> Selection {
    let mut b = Builder {
        probe,
        has_onnx,
        options: Vec::new(),
    };
    let ov = &probe.openvino;

    // NVIDIA: CUDA and TensorRT per GPU, as (CUDA ordinal, GPU) in ordinal order.
    let nvidia = cuda_ordinals(hw);
    for target in [Target::Cuda, Target::TensorRt] {
        for (ordinal, g) in &nvidia {
            b.push_ort_unless(
                target,
                Some(*ordinal),
                gpu_label(target, g),
                ort_cuda_unsupported(hw, g),
            );
        }
    }

    // OpenVINO GPUs as listed; if none is listed but an Intel GPU exists, say why it can't run.
    let ov_gpus: Vec<&OvDevice> = ov.gpus().collect();
    for d in &ov_gpus {
        if let Some(spec) = d.spec() {
            let reason = (!d.is_intel_or_unknown()).then(|| {
                "not an Intel GPU (OpenVINO's GPU plugin supports Intel GPUs only)".to_string()
            });
            b.push(spec, ov_label("GPU", Some(d)), reason);
        }
    }
    if ov_gpus.is_empty() && hw.has_vendor(GpuVendor::Intel) {
        let intel = hw.gpus_of(GpuVendor::Intel).next().expect("has Intel");
        let reason = b.ov_reason().unwrap_or_else(|| {
            "OpenVINO lists no GPU device (install the Intel graphics driver / compute runtime)"
                .to_string()
        });
        b.push(
            Device::OPENVINO_GPU,
            format!("OpenVINO GPU ({})", intel.name),
            Some(reason),
        );
    }

    // DirectML on every Windows GPU, discrete and larger first.
    let mut dml: Vec<&GpuAdapter> = Vec::new();
    if hw.is_windows() {
        dml = hw.gpus.iter().collect();
        dml.sort_by(|a, c| {
            c.discrete
                .cmp(&a.discrete)
                .then(c.vram_mb.cmp(&a.vram_mb))
                .then(a.index.cmp(&c.index))
        });
        for g in &dml {
            b.push_ort(
                Target::DirectMl,
                Some(g.index),
                gpu_label(Target::DirectMl, g),
            );
        }
    }

    if hw.is_macos() {
        b.push_ort(
            Target::CoreMl,
            None,
            "ONNX Runtime CoreML (Apple GPU and Neural Engine)".to_string(),
        );
    }

    if let Some(npu) = ov.find("NPU") {
        b.push(
            device(Runtime::OpenVino, Target::Npu, None),
            ov_label("NPU", Some(npu)),
            None,
        );
    }

    let ov_cpu_reason = b.ov_reason().or_else(|| {
        ov.find("CPU")
            .is_none()
            .then(|| "OpenVINO lists no CPU device".to_string())
    });
    b.push(
        Device::OPENVINO_CPU,
        ov_label("CPU", ov.find("CPU")),
        ov_cpu_reason,
    );
    b.push_ort(Target::Cpu, None, "ONNX Runtime CPU".to_string());

    // ---- auto ranking ----
    let mut auto: Vec<Device> = Vec::new();

    // 1. CUDA on the NVIDIA GPU with the most VRAM (first one on ties).
    let cuda = nvidia
        .iter()
        .map(|(ordinal, g)| (device(Runtime::Ort, Target::Cuda, Some(*ordinal)), *g))
        .filter(|(d, _)| b.runnable(d))
        .fold(None::<(Device, &GpuAdapter)>, |best, (d, g)| match best {
            Some((_, bg)) if bg.vram_mb >= g.vram_mb => best,
            _ => Some((d, g)),
        });
    if let Some((d, _)) = cuda {
        auto.push(d);
    }

    // 2. Intel GPUs via OpenVINO: one GPU as listed; several -> discrete first.
    let intel: Vec<&OvDevice> = ov_gpus
        .iter()
        .copied()
        .filter(|d| d.is_intel_or_unknown())
        .filter(|d| d.spec().is_some_and(|s| b.runnable(&s)))
        .collect();
    let mut ov_auto: Vec<Device> = intel
        .iter()
        .filter(|d| d.is_discrete())
        .chain(intel.iter().filter(|d| !d.is_discrete()))
        .filter_map(|d| d.spec())
        .collect();
    if ov_auto.len() > 1 && !intel.iter().any(|d| d.is_discrete()) {
        // No discrete GPU among several: keep the first, as a bare `GPU` request would.
        ov_auto.truncate(1);
    }
    let intel_covered = !ov_auto.is_empty();
    auto.extend(ov_auto);

    // 3. DirectML (Windows) on GPUs not covered by CUDA or OpenVINO.
    for g in &dml {
        let covered = match g.vendor {
            GpuVendor::Nvidia => cuda.is_some(),
            GpuVendor::Intel => intel_covered,
            _ => false,
        };
        let d = device(Runtime::Ort, Target::DirectMl, Some(g.index));
        if !covered && b.runnable(&d) {
            auto.push(d);
        }
    }

    // 4. CoreML on Apple silicon.
    let coreml = device(Runtime::Ort, Target::CoreMl, None);
    if hw.is_apple_silicon() && b.runnable(&coreml) {
        auto.push(coreml);
    }

    // 5. CPU.
    let ort_cpu = device(Runtime::Ort, Target::Cpu, None);
    if b.runnable(&Device::OPENVINO_CPU) {
        auto.push(Device::OPENVINO_CPU);
    } else if b.runnable(&ort_cpu) {
        auto.push(ort_cpu);
    }

    Selection {
        options: b.options,
        auto,
    }
}

/// The NVIDIA GPUs with their CUDA ordinals (what `ort:cuda:N` means), in ordinal order. The
/// driver's ordinals when it reported every NVIDIA GPU ([`GpuAdapter::cuda`]); otherwise the
/// position among the NVIDIA adapters in platform order (exact with one NVIDIA GPU).
pub fn cuda_ordinals(hw: &HardwareInfo) -> Vec<(u32, &GpuAdapter)> {
    let mut nvidia: Vec<&GpuAdapter> = hw.gpus_of(GpuVendor::Nvidia).collect();
    nvidia.sort_by_key(|g| g.index);
    if nvidia.iter().all(|g| g.cuda.is_some()) {
        let mut v: Vec<(u32, &GpuAdapter)> = nvidia
            .into_iter()
            .filter_map(|g| Some((g.cuda?.ordinal, g)))
            .collect();
        v.sort_by_key(|(o, _)| *o);
        v
    } else {
        (0u32..).zip(nvidia).collect()
    }
}

/// Why the pinned ONNX Runtime CUDA build (CUDA and TensorRT providers) cannot run on NVIDIA GPU
/// `g`: it has no kernels for the GPU's compute capability (see `catalog::ort_cuda_archs`).
/// None when it can, or when the compute capability is unknown (no CUDA driver).
pub fn ort_cuda_unsupported(hw: &HardwareInfo, g: &GpuAdapter) -> Option<String> {
    let cc = g.cuda?.compute_capability;
    let archs = catalog::ort_cuda_archs(&hw.os)?;
    if archs.runs_on(cc) {
        return None;
    }
    let needs = match archs.min() {
        Some(min) if cc < min => format!("needs {min}+"),
        _ => format!("has kernels for {} only", archs.describe()),
    };
    let instead = if hw.is_windows() {
        "use ort:directml or OpenVINO"
    } else {
        "use OpenVINO or ort:cpu"
    };
    Some(format!(
        "{} is compute capability {cc}; the ONNX Runtime {} CUDA build {needs}; {instead}",
        g.name,
        catalog::ONNXRUNTIME_VERSION
    ))
}

/// Plain-text table of `sel` (for `list-devices`): spec, status, label and reason.
pub fn format_options(sel: &Selection) -> String {
    let width = sel
        .options
        .iter()
        .map(|o| o.spec.to_string().len())
        .max()
        .unwrap_or(0)
        .max(4);
    let mut out = String::new();
    let _ = writeln!(out, "  {:<width$}  {:<8}  LABEL / REASON", "SPEC", "STATUS");
    for o in &sel.options {
        let status = o.status();
        let _ = writeln!(
            out,
            "  {:<width$}  {:<8}  {}",
            o.spec.to_string(),
            status,
            o.label
        );
        if let Some(d) = &o.download {
            let _ = writeln!(out, "  {:<width$}  {:<8}    -> {}", "", "", d.summary);
        } else if let Some(r) = &o.reason {
            let _ = writeln!(out, "  {:<width$}  {:<8}    -> {}", "", "", r);
        }
    }
    out
}

/// `auto` chain as text: "openvino:gpu -> openvino:cpu" (or "(nothing runnable)").
pub fn format_auto(sel: &Selection) -> String {
    if sel.auto.is_empty() {
        return "(nothing runnable)".to_string();
    }
    sel.auto
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(" -> ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ov_device_helpers() {
        let d = OvDevice {
            name: "GPU.1".into(),
            full_name: "Intel(R) Arc(TM) A770 Graphics (dGPU)".into(),
            device_type: None,
        };
        assert!(d.is_gpu() && d.is_discrete() && d.is_intel_or_unknown());
        assert_eq!(d.spec().unwrap().to_string(), "openvino:gpu.1");
        let d = OvDevice {
            device_type: Some("integrated".into()),
            ..d
        };
        assert!(!d.is_discrete(), "DEVICE_TYPE wins over the name");
        assert_eq!(OvDevice::named("CPU").spec().unwrap(), Device::OPENVINO_CPU);
        assert_eq!(OvDevice::named("HETERO").spec(), None);
    }

    #[test]
    fn ort_probe_reasons() {
        let p = OrtProbe::not_in_build();
        assert_eq!(
            p.ep_error(Target::Cpu).as_deref(),
            Some(super::super::ORT_UNAVAILABLE)
        );
        let p = OrtProbe::installed(
            "gpu",
            vec![
                EpStatus::usable(Target::Cpu),
                EpStatus::broken(Target::Cuda, "cudnn64_9.dll not found"),
            ],
        );
        assert_eq!(p.ep_error(Target::Cpu), None);
        assert_eq!(
            p.ep_error(Target::Cuda).as_deref(),
            Some("cudnn64_9.dll not found")
        );
        assert!(
            p.ep_error(Target::DirectMl)
                .unwrap()
                .contains("(gpu) has no DirectML")
        );
    }

    fn nvidia(name: &str, index: u32, cuda: Option<(u32, u32, u32)>) -> GpuAdapter {
        GpuAdapter {
            vendor: GpuVendor::Nvidia,
            name: name.into(),
            vram_mb: 8060,
            index,
            discrete: true,
            cuda: cuda.map(|(ordinal, major, minor)| super::super::detect::CudaInfo {
                ordinal,
                compute_capability: super::super::detect::ComputeCapability::new(major, minor),
            }),
        }
    }

    fn uhd630(index: u32) -> GpuAdapter {
        GpuAdapter {
            vendor: GpuVendor::Intel,
            name: "Intel(R) UHD Graphics 630".into(),
            vram_mb: 128,
            index,
            discrete: false,
            cuda: None,
        }
    }

    /// OpenVINO with CPU + the Intel GPU, ONNX Runtime with every EP usable.
    fn everything() -> RuntimeProbe {
        RuntimeProbe {
            openvino: OpenVinoProbe::with_device_names(&["CPU".into(), "GPU".into()]),
            ort: OrtProbe::installed(
                "cuda",
                [
                    Target::Cuda,
                    Target::TensorRt,
                    Target::DirectMl,
                    Target::Cpu,
                ]
                .into_iter()
                .map(EpStatus::usable)
                .collect(),
            ),
        }
    }

    fn auto_text(sel: &Selection) -> String {
        format_auto(sel)
    }

    #[test]
    fn cuda_needs_a_supported_compute_capability() {
        let probe = everything();
        let cuda0 = device(Runtime::Ort, Target::Cuda, Some(0));
        let trt0 = device(Runtime::Ort, Target::TensorRt, Some(0));

        // GTX 1070 Ti (6.1) on Windows: CUDA and TensorRT not runnable, DirectML takes the GPU.
        let gpu = nvidia("NVIDIA GeForce GTX 1070 Ti", 0, Some((0, 6, 1)));
        let hw = HardwareInfo::new("windows", "x86_64", vec![gpu, uhd630(1)]);
        let sel = select(&hw, &probe, true);
        for spec in [cuda0, trt0] {
            let o = sel.option(&spec).unwrap();
            assert!(!o.runnable, "{spec}");
            assert_eq!(
                o.reason.as_deref(),
                Some(
                    "NVIDIA GeForce GTX 1070 Ti is compute capability 6.1; the ONNX Runtime \
                     1.24.4 CUDA build needs 7.5+; use ort:directml or OpenVINO"
                )
            );
        }
        assert_eq!(
            auto_text(&sel),
            "openvino:gpu -> ort:directml:0 -> openvino:cpu"
        );

        // 8.6 and unknown: CUDA first, as before.
        for cuda in [Some((0, 8, 6)), Some((0, 7, 5)), Some((0, 12, 0)), None] {
            let gpu = nvidia("NVIDIA GeForce RTX 3060", 0, cuda);
            let hw = HardwareInfo::new("windows", "x86_64", vec![gpu, uhd630(1)]);
            let sel = select(&hw, &probe, true);
            assert!(sel.option(&cuda0).unwrap().runnable, "{cuda:?}");
            assert!(sel.option(&trt0).unwrap().runnable, "{cuda:?}");
            assert_eq!(
                auto_text(&sel),
                "ort:cuda:0 -> openvino:gpu -> openvino:cpu",
                "{cuda:?}"
            );
        }

        // 8.0 (A100) on Windows: above the minimum but in a gap of the Windows build.
        let hw = HardwareInfo::new(
            "windows",
            "x86_64",
            vec![nvidia("NVIDIA A100", 0, Some((0, 8, 0)))],
        );
        let reason = ort_cuda_unsupported(&hw, &hw.gpus[0]).unwrap();
        assert!(
            reason.contains("has kernels for 7.5, 8.6, 8.9 (PTX 9.0) only"),
            "{reason}"
        );

        // Linux: the CUDA build has sm_60 code, so Pascal runs; Maxwell (5.2) does not.
        let linux = |cc: (u32, u32, u32)| {
            HardwareInfo::new("linux", "x86_64", vec![nvidia("NVIDIA GPU", 0, Some(cc))])
        };
        let hw = linux((0, 6, 1));
        assert_eq!(ort_cuda_unsupported(&hw, &hw.gpus[0]), None);
        let hw = linux((0, 5, 2));
        let reason = ort_cuda_unsupported(&hw, &hw.gpus[0]).unwrap();
        assert!(
            reason.contains("needs 6.0+; use OpenVINO or ort:cpu"),
            "{reason}"
        );
        assert!(!select(&hw, &probe, true).option(&cuda0).unwrap().runnable);
    }

    #[test]
    fn cuda_ordinals_follow_the_driver() {
        // DXGI lists the 3060 first, the driver (fastest first) the 4090.
        let a = nvidia("NVIDIA GeForce RTX 3060", 0, Some((1, 8, 6)));
        let b = nvidia("NVIDIA GeForce RTX 4090", 2, Some((0, 8, 9)));
        let hw = HardwareInfo::new("windows", "x86_64", vec![a, uhd630(1), b]);
        let v: Vec<(u32, &str)> = cuda_ordinals(&hw)
            .into_iter()
            .map(|(o, g)| (o, g.name.as_str()))
            .collect();
        assert_eq!(
            v,
            [
                (0, "NVIDIA GeForce RTX 4090"),
                (1, "NVIDIA GeForce RTX 3060")
            ]
        );
        let sel = select(&hw, &everything(), true);
        let label = &sel
            .option(&device(Runtime::Ort, Target::Cuda, Some(1)))
            .unwrap()
            .label;
        assert!(label.contains("RTX 3060"), "{label}");

        // Any GPU unknown to the driver: positions in platform order, as before.
        let a = nvidia("NVIDIA GeForce RTX 3060", 0, Some((1, 8, 6)));
        let b = nvidia("NVIDIA GeForce RTX 4090", 2, None);
        let hw = HardwareInfo::new("windows", "x86_64", vec![a, b]);
        let v: Vec<(u32, u32)> = cuda_ordinals(&hw)
            .into_iter()
            .map(|(o, g)| (o, g.index))
            .collect();
        assert_eq!(v, [(0, 0), (1, 2)]);
    }

    #[test]
    fn onnx_sibling_lookup() {
        let dir = std::env::temp_dir().join(format!("bop-select-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let xml = dir.join("m.xml");
        let onnx = dir.join("m.onnx");
        let lone = dir.join("lone.xml");
        std::fs::write(&xml, "x").unwrap();
        std::fs::write(&lone, "x").unwrap();
        assert_eq!(onnx_path_for(&xml), None);
        std::fs::write(&onnx, "x").unwrap();
        assert_eq!(onnx_path_for(&xml), Some(onnx.clone()));
        assert_eq!(onnx_path_for(&onnx), Some(onnx.clone()));
        assert_eq!(onnx_path_for(&lone), None);
        assert_eq!(onnx_path_for(&dir.join("missing.onnx")), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
