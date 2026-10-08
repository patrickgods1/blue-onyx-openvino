//! OpenVINO backend: wraps `Core`, `CompiledModel` and `InferRequest`.
//!
//! Usage (from `registry`/`worker`/benchmark):
//!
//! ```ignore
//! let mut core = OvCore::new(&CoreOptions { cache_dir, intra_threads, openvino_dir })?;
//! let loaded = core.load(&LoadRequest { path, device: "GPU".into(), gpu_precision: None, allow_cpu_fallback: true })?;
//! // `loaded` is Send: move it into the worker thread, then:
//! let mut backend = OvBackend::new(loaded)?;        // creates the InferRequest (which is !Sync)
//! let outs: Vec<NamedOutput> = backend.infer(&chw_f32, &extra_inputs)?;
//! ```

pub mod device;
pub mod libs;

pub use device::{DeviceInfo, DeviceSelection};

use crate::model::{ExtraInput, NamedOutput, PortSpec};
use anyhow::Result;
use std::path::PathBuf;

/// Options applied once per `Core`.
#[derive(Debug, Clone, Default)]
pub struct CoreOptions {
    /// Compiled-model cache directory; None disables caching.
    pub cache_dir: Option<PathBuf>,
    /// CPU inference threads (0 = OpenVINO default).
    pub intra_threads: usize,
    /// Explicit OpenVINO runtime directory (config `openvino_dir`).
    pub openvino_dir: Option<PathBuf>,
}

/// What to load and where.
#[derive(Debug, Clone)]
pub struct LoadRequest {
    pub path: PathBuf,
    /// "GPU", "GPU.1", "CPU", ...
    pub device: String,
    /// "f16" (default on GPU) or "f32".
    pub gpu_precision: Option<String>,
    pub allow_cpu_fallback: bool,
}

/// A compiled model plus its port metadata. `CompiledModel` is `Send`, so this can be moved
/// into a worker thread; the `InferRequest` must be created there.
pub struct LoadedModel {
    pub compiled: openvino::CompiledModel,
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

/// Owns the OpenVINO `Core`. One per process; models are loaded sequentially through it.
pub struct OvCore {
    core: openvino::Core,
    opts: CoreOptions,
    available: Vec<String>,
}

impl OvCore {
    pub fn new(opts: &CoreOptions) -> Result<Self> {
        libs::prepare_environment(opts.openvino_dir.as_deref());
        let core = openvino::Core::new()
            .map_err(|e| anyhow::anyhow!("OpenVINO Core::new failed: {e:?}. {}", libs::diagnostics()))?;
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

    /// Read, (reshape to static), compile with fallback, and introspect ports.
    pub fn load(&mut self, req: &LoadRequest) -> Result<LoadedModel> {
        device::load_model(&mut self.core, &self.available, req)
    }

    /// Human readable device description (FULL_DEVICE_NAME) for a device string.
    pub fn device_full_name(&self, device: &str) -> String {
        device::full_device_name(&self.core, device)
    }

    pub fn inner_mut(&mut self) -> &mut openvino::Core {
        &mut self.core
    }
}

/// Per-worker inference state.
pub struct OvBackend {
    pub(crate) model: LoadedModel,
    pub(crate) request: openvino::InferRequest,
    pub(crate) input_tensor: openvino::Tensor,
}

impl OvBackend {
    pub fn new(model: LoadedModel) -> Result<Self> {
        device::create_backend(model)
    }

    pub fn model(&self) -> &LoadedModel {
        &self.model
    }

    /// Run one inference. `chw` must be `3 * w * h` f32 values in CHW order.
    pub fn infer(&mut self, chw: &[f32], extra: &[ExtraInput]) -> Result<Vec<NamedOutput>> {
        device::run_inference(
            &self.model,
            &mut self.request,
            &mut self.input_tensor,
            chw,
            extra,
        )
    }
}
