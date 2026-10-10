//! Inference backends behind one runtime-neutral interface.
//!
//! Phase 6.1: only OpenVINO is implemented; [`Backend`], [`Compiled`] and [`Runtimes`] are shaped
//! so an ONNX Runtime variant drops in (docs/PLAN.md, "Multi-runtime backend").
//!
//! Usage (from `registry`/`worker`/benchmark):
//!
//! ```ignore
//! let mut rt = Runtimes::new(&CoreOptions { cache_dir, intra_threads, openvino_dir });
//! rt.require_any()?;                                   // fails like the old `OvCore::new`
//! let req = LoadRequest { path, requested: "GPU".into(), gpu_precision: None };
//! // Whole plan (GPU, then CPU) in one call, without warm-up:
//! let mut backend: Backend = rt.load(&spec::parse("GPU")?, &req)?;
//! // Or per candidate, compiling under the `Runtimes` lock and creating the request elsewhere:
//! let compiled: Compiled = rt.compile(&candidate, &req)?;   // `Compiled` is Send
//! let mut backend = compiled.into_backend()?;             // creates the InferRequest
//! let outs: Vec<NamedOutput> = backend.infer(&chw_f32, &extra_inputs)?;
//! ```

pub mod device;
pub mod libs;
pub mod plan;
pub mod spec;

pub use device::DeviceInfo;
pub use plan::{Candidate, plan_candidates, try_candidates};
pub use spec::{DeviceSpec, Runtime};

use crate::model::{ExtraInput, NamedOutput, PortSpec};
use anyhow::Result;
use std::path::PathBuf;

/// Options applied once per runtime.
#[derive(Debug, Clone, Default)]
pub struct CoreOptions {
    /// Compiled-model cache directory; None disables caching.
    pub cache_dir: Option<PathBuf>,
    /// CPU inference threads (0 = runtime default).
    pub intra_threads: usize,
    /// Explicit OpenVINO runtime directory (config `openvino_dir`).
    pub openvino_dir: Option<PathBuf>,
}

/// What to load. The device comes separately as a [`DeviceSpec`] / [`Candidate`].
#[derive(Debug, Clone)]
pub struct LoadRequest {
    pub path: PathBuf,
    /// Device as configured ("GPU", "GPU.1", "openvino:cpu", "auto", ...), reported as
    /// [`DeviceInfo::requested`].
    pub requested: String,
    /// "f16" (default on GPU) or "f32".
    pub gpu_precision: Option<String>,
}

/// Runtime-neutral facts about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model file this was loaded from (used for error context).
    pub path: PathBuf,
    pub inputs: Vec<PortSpec>,
    pub outputs: Vec<PortSpec>,
    pub device: DeviceInfo,
    /// Name of the image input tensor (e.g. "images" / "input").
    pub image_input: String,
    /// (width, height) of the image input.
    pub input_size: (u32, u32),
    /// Wall time spent in read+compile, for logging/stats.
    pub compile_ms: u64,
}

/// Facts about the available runtimes, captured once at startup.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RuntimeInfo {
    /// Empty when OpenVINO could not be initialized.
    pub openvino_version: String,
    /// OpenVINO device names ("CPU", "GPU.0", ...).
    pub available_devices: Vec<String>,
    pub has_gpu: bool,
}

/// Owns the OpenVINO `Core`. One per process; models are loaded sequentially through it.
pub struct OvCore {
    core: openvino::Core,
    opts: CoreOptions,
    available: Vec<String>,
}

impl OvCore {
    pub fn new(opts: &CoreOptions) -> Result<Self> {
        libs::prepare_environment(opts.openvino_dir.as_deref());
        let core = openvino::Core::new().map_err(|e| {
            anyhow::anyhow!("OpenVINO Core::new failed: {e:?}. {}", libs::diagnostics())
        })?;
        let available = core
            .available_devices()
            .map(|v| v.iter().map(|d| d.to_string()).collect())
            .unwrap_or_default();
        let mut me = Self {
            core,
            opts: opts.clone(),
            available,
        };
        me.apply_core_properties()?;
        Ok(me)
    }

    pub fn available_devices(&self) -> &[String] {
        &self.available
    }

    pub fn has_gpu(&self) -> bool {
        self.available.iter().any(|d| d.starts_with("GPU"))
    }

    pub fn openvino_version(&self) -> String {
        let v = openvino::version();
        format!("{} ({})", v.build_number, v.description)
    }

    fn apply_core_properties(&mut self) -> Result<()> {
        device::apply_core_properties(&mut self.core, &self.opts, &self.available)
    }

    /// Read, (reshape to static), compile on exactly `cand.device` and introspect ports.
    pub fn compile(&mut self, cand: &Candidate, req: &LoadRequest) -> Result<OvCompiled> {
        device::load_model(&mut self.core, &self.available, cand, req)
    }

    /// Human readable device description (FULL_DEVICE_NAME) for a device string.
    pub fn device_full_name(&self, device: &str) -> String {
        device::full_device_name(&self.core, device)
    }

    pub fn inner_mut(&mut self) -> &mut openvino::Core {
        &mut self.core
    }
}

/// An OpenVINO compiled model plus its metadata. `CompiledModel` is `Send`, so this can be moved
/// into a worker thread; the `InferRequest` is created there ([`OvBackend::new`]).
pub struct OvCompiled {
    pub(crate) compiled: openvino::CompiledModel,
    pub info: ModelInfo,
}

/// Per-worker OpenVINO inference state.
pub struct OvBackend {
    pub(crate) info: ModelInfo,
    pub(crate) request: openvino::InferRequest,
    pub(crate) input_tensor: openvino::Tensor,
    /// Kept alive for as long as its infer request.
    pub(crate) _compiled: openvino::CompiledModel,
}

impl OvBackend {
    pub fn new(model: OvCompiled) -> Result<Self> {
        device::create_backend(model)
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Run one inference. `chw` must be `3 * w * h` f32 values in CHW order.
    pub fn infer(&mut self, chw: &[f32], extra: &[ExtraInput]) -> Result<Vec<NamedOutput>> {
        device::run_inference(
            &self.info,
            &mut self.request,
            &mut self.input_tensor,
            chw,
            extra,
        )
    }
}

/// A compiled model of any runtime, before its inference request exists. `Send`.
pub enum Compiled {
    OpenVino(OvCompiled),
}

impl Compiled {
    pub fn info(&self) -> &ModelInfo {
        match self {
            Compiled::OpenVino(c) => &c.info,
        }
    }

    /// Create the per-thread inference state. Call on the thread that will run inference.
    pub fn into_backend(self) -> Result<Backend> {
        match self {
            Compiled::OpenVino(c) => Ok(Backend::OpenVino(OvBackend::new(c)?)),
        }
    }
}

/// Inference state of one model on one worker thread.
pub enum Backend {
    OpenVino(OvBackend),
}

impl Backend {
    pub fn info(&self) -> &ModelInfo {
        match self {
            Backend::OpenVino(b) => b.info(),
        }
    }

    /// Run one inference. `chw` must be `3 * w * h` f32 values in CHW order.
    pub fn infer(&mut self, chw: &[f32], extra: &[ExtraInput]) -> Result<Vec<NamedOutput>> {
        match self {
            Backend::OpenVino(b) => b.infer(chw, extra),
        }
    }
}

/// Error for ONNX Runtime specs until phase 6.3 adds the backend.
pub const ORT_UNAVAILABLE: &str = "ONNX Runtime support is not available in this build yet";

/// Every runtime this process can use. Shared behind a mutex; models compile one at a time.
pub struct Runtimes {
    ov: Option<OvCore>,
    /// Why OpenVINO could not be initialized (`{e:#}`).
    ov_error: Option<String>,
}

impl Runtimes {
    /// Initialize every runtime. Failures are kept (see [`Self::require_any`]), not returned.
    pub fn new(opts: &CoreOptions) -> Self {
        match OvCore::new(opts) {
            Ok(core) => Self {
                ov: Some(core),
                ov_error: None,
            },
            Err(e) => Self {
                ov: None,
                ov_error: Some(format!("{e:#}")),
            },
        }
    }

    /// Ok when at least one runtime is usable; otherwise the initialization error, worded as the
    /// old `OvCore::new(..).context("initializing OpenVINO")`.
    pub fn require_any(&self) -> Result<()> {
        if self.ov.is_some() {
            return Ok(());
        }
        let err = self
            .ov_error
            .as_deref()
            .unwrap_or("OpenVINO was not initialized");
        Err(anyhow::anyhow!("{err}").context("initializing OpenVINO"))
    }

    pub fn openvino(&self) -> Option<&OvCore> {
        self.ov.as_ref()
    }

    pub fn openvino_mut(&mut self) -> Option<&mut OvCore> {
        self.ov.as_mut()
    }

    pub fn openvino_error(&self) -> Option<&str> {
        self.ov_error.as_deref()
    }

    /// OpenVINO device names, empty without OpenVINO.
    pub fn available_ov_devices(&self) -> &[String] {
        self.ov
            .as_ref()
            .map(OvCore::available_devices)
            .unwrap_or(&[])
    }

    pub fn info(&self) -> RuntimeInfo {
        match &self.ov {
            Some(core) => RuntimeInfo {
                openvino_version: core.openvino_version(),
                available_devices: core.available_devices().to_vec(),
                has_gpu: core.has_gpu(),
            },
            None => RuntimeInfo::default(),
        }
    }

    /// Candidates for `spec` on this machine, in the order they should be tried.
    pub fn plan(&self, spec: &DeviceSpec) -> Vec<Candidate> {
        plan_candidates(spec, self.available_ov_devices())
    }

    /// Compile on exactly `cand.device` (no fallback).
    pub fn compile(&mut self, cand: &Candidate, req: &LoadRequest) -> Result<Compiled> {
        match cand.device.runtime {
            Runtime::OpenVino => {
                let Some(core) = self.ov.as_mut() else {
                    anyhow::bail!(
                        "OpenVINO is not available: {}",
                        self.ov_error.as_deref().unwrap_or("not initialized")
                    );
                };
                Ok(Compiled::OpenVino(core.compile(cand, req)?))
            }
            Runtime::Ort => anyhow::bail!("{ORT_UNAVAILABLE} ({})", cand.device),
        }
    }

    /// Resolve `spec` and load the first candidate that compiles and creates its request
    /// (no warm-up; the worker additionally warms up each candidate).
    pub fn load(&mut self, spec: &DeviceSpec, req: &LoadRequest) -> Result<Backend> {
        let candidates = self.plan(spec);
        let label = format!("model {}", req.path.display());
        try_candidates(&label, &candidates, |cand| {
            self.compile(cand, req)?.into_backend()
        })
    }
}
