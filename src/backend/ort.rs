//! ONNX Runtime backend (Cargo feature `onnxruntime`), built on the `ort` 2.x crate with
//! `load-dynamic`: the ONNX Runtime shared library is opened at startup from config
//! `onnxruntime_dir`, `ORT_DYLIB_PATH` or `<exe_dir>/onnxruntime` (see
//! [`libs::find_onnxruntime`]), never linked. All `ort` crate code lives in this file.
//!
//! - [`OrtRuntime::new`] loads the library once per process (a process can hold only one ONNX
//!   Runtime) and probes its execution providers into an [`OrtProbe`]: `is_available()` per EP,
//!   plus for CUDA/TensorRT whether the provider library, CUDA runtime, cuDNN (and TensorRT) can
//!   be loaded, so the reason names what is missing.
//! - [`OrtRuntime::compile`] creates a session on exactly one EP (registration errors are
//!   errors, not a silent CPU fallback; the CPU fallback is the caller's candidate loop):
//!   CUDA `device_id`; TensorRT fp16 with its engine cache in `cache/tensorrt` (CUDA behind it
//!   for unsupported nodes); DirectML `device_id`, memory pattern off, sequential execution;
//!   CoreML MLProgram, compute units ALL, cache in `cache/coreml`; `intra_threads` everywhere.
//! - Named dynamic dimensions of the image input and RT-DETR's `orig_target_sizes` are pinned
//!   to `[1,3,640,640]` / `[1,2]` with free-dimension overrides (the same defaults as the
//!   OpenVINO reshape), so the ports reported in [`ModelInfo`] are static and `make_family`
//!   auto-detection is unchanged. Pinning is also what lets RT-DETR run on CoreML: an RT-DETR
//!   model that keeps an unpinnable dynamic input dimension is refused there (see
//!   [`unsupported_on`]).
//! - [`OrtBackend::infer`] feeds the CHW image without copying, converts extra inputs to the
//!   port's element type (i64/i32/f32) and copies every output into a [`NamedOutput`].

use super::detect::{GpuVendor, HardwareInfo};
#[cfg(test)]
use super::device::DEFAULT_IMAGE_SHAPE;
use super::device::{
    DEFAULT_TARGET_SIZES_SHAPE, dynamic_image_shape, extra_input_index, find_image_input,
};
use super::select::{EpStatus, NEEDS_ONNX, OrtProbe, onnx_path_for};
use super::spec::{Runtime, Target};
use super::{Candidate, DeviceInfo, LoadRequest, ModelInfo, libs};
use crate::model::{ExtraData, ExtraInput, NamedOutput, OutputBuf, PortElem, PortSpec};
use ::ort::ep::{self, ExecutionProvider};
use ::ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ::ort::session::{Session, SessionInputValue};
use ::ort::value::{DynValue, Tensor, TensorElementType, TensorRef, ValueType};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

/// The loaded ONNX Runtime library (process-wide).
#[derive(Debug, Clone)]
struct LoadedLib {
    path: PathBuf,
    version: String,
}

/// Set once the library is loaded and the `ort` environment committed. Failures are not cached,
/// so a later generation (e.g. after `setup-onnxruntime`) can retry.
static LOADED: Mutex<Option<LoadedLib>> = Mutex::new(None);

fn ort_err<R>(e: ::ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// `OrtGetApiBase()->GetVersionString()` of the library at `path`. The handle is leaked on
/// purpose: ONNX Runtime stays loaded for the process (`ort` keeps its own handle too) and
/// libraries with Objective-C code cannot be unloaded on macOS anyway.
fn library_version(path: &Path) -> Result<String> {
    type GetApiBase = unsafe extern "system" fn() -> *const ::ort::sys::OrtApiBase;
    // SAFETY: `path` is an ONNX Runtime shared library chosen by the user/installer; loading it
    // runs its initializers exactly as `ort::init_from` would right after. `OrtGetApiBase` has
    // the declared signature in every ONNX Runtime release (C API, `onnxruntime_c_api.h`), the
    // returned struct is static and `GetVersionString` returns a static NUL-terminated string.
    unsafe {
        let lib = libloading::Library::new(path)
            .with_context(|| format!("loading {}", path.display()))?;
        let get: libloading::Symbol<GetApiBase> = lib
            .get(b"OrtGetApiBase\0")
            .with_context(|| format!("{} does not export OrtGetApiBase", path.display()))?;
        let base = get();
        if base.is_null() {
            bail!("{}: OrtGetApiBase returned null", path.display());
        }
        let v = ((*base).GetVersionString)();
        let version = if v.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(v).to_string_lossy().into_owned()
        };
        std::mem::forget(lib);
        Ok(version)
    }
}

/// `(major, minor)` of a version string like "1.24.4".
pub(crate) fn parse_version(v: &str) -> Option<(u32, u32)> {
    let mut it = v.trim().split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    Some((major, minor))
}

/// Whether a library of `version` can serve the `ort` crate's API level (1.`min_minor`+).
pub(crate) fn version_supported(version: &str, min_minor: u32) -> bool {
    matches!(parse_version(version), Some((1, m)) if m >= min_minor)
}

fn load_library(path: &Path) -> Result<LoadedLib> {
    let mut slot = LOADED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(lib) = slot.as_ref() {
        if lib.path != path {
            tracing::warn!(
                "ONNX Runtime is already loaded from {}; {} is ignored until the process restarts",
                lib.path.display(),
                path.display()
            );
        }
        return Ok(lib.clone());
    }
    let version = library_version(path)?;
    if !version_supported(&version, ::ort::MINOR_VERSION) {
        bail!(
            "{} is ONNX Runtime {version}; 1.{}.0 or newer is required (run `setup-onnxruntime`)",
            path.display(),
            ::ort::MINOR_VERSION
        );
    }
    // Windows: DirectML.dll shipped next to onnxruntime.dll must win over the older copy in
    // System32, so load it by absolute path first.
    #[cfg(windows)]
    if let Some(dir) = path.parent() {
        let dml = dir.join("DirectML.dll");
        if dml.is_file() {
            match ::ort::util::preload_dylib(&dml) {
                Ok(()) => tracing::debug!("preloaded {}", dml.display()),
                Err(e) => tracing::warn!("could not preload {}: {e}", dml.display()),
            }
        }
    }
    let committed = ::ort::init_from(path)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_name("blue-onyx-prism")
        .with_telemetry(false)
        .commit();
    if !committed {
        tracing::debug!("ONNX Runtime environment was already configured");
    }
    // `ort` forwards every ONNX Runtime message to `tracing`, including the verbose/info ones it
    // prints per session; keep warnings and errors unless `ORT_LOG` asks for more.
    let level = match std::env::var("ORT_LOG").as_deref() {
        Ok("verbose") => ::ort::logging::LogLevel::Verbose,
        Ok("info") => ::ort::logging::LogLevel::Info,
        Ok("error") => ::ort::logging::LogLevel::Error,
        Ok("fatal") => ::ort::logging::LogLevel::Fatal,
        _ => ::ort::logging::LogLevel::Warning,
    };
    match ::ort::environment::Environment::current() {
        Ok(env) => env.set_log_level(level),
        Err(e) => bail!("creating the ONNX Runtime environment: {e}"),
    }
    let lib = LoadedLib {
        path: path.to_path_buf(),
        version,
    };
    *slot = Some(lib.clone());
    Ok(lib)
}

// ---- CUDA / TensorRT dependency checks ---------------------------------------------------------

/// Library names tried for each dependency, in order (any one loading is enough).
#[cfg(windows)]
mod dep_names {
    pub const CUDART: &[&str] = &["cudart64_12.dll", "cudart64_13.dll"];
    pub const CUDNN: &[&str] = &["cudnn64_9.dll"];
    pub const NVINFER: &[&str] = &["nvinfer_10.dll"];
    pub const CUDA_PROVIDER: &str = "onnxruntime_providers_cuda.dll";
    pub const TRT_PROVIDER: &str = "onnxruntime_providers_tensorrt.dll";
}
#[cfg(not(windows))]
mod dep_names {
    pub const CUDART: &[&str] = &["libcudart.so.12", "libcudart.so.13"];
    pub const CUDNN: &[&str] = &["libcudnn.so.9"];
    pub const NVINFER: &[&str] = &["libnvinfer.so.10"];
    pub const CUDA_PROVIDER: &str = "libonnxruntime_providers_cuda.so";
    pub const TRT_PROVIDER: &str = "libonnxruntime_providers_tensorrt.so";
}

/// One dependency of an NVIDIA execution provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dep {
    /// "CUDA 12 runtime", "cuDNN 9", ...
    pub label: &'static str,
    /// File names tried (the first is named in messages).
    pub names: &'static [&'static str],
    pub loadable: bool,
    /// Part of `nvidia-cuda-libs` (downloadable), not the provider or TensorRT.
    pub cuda_lib: bool,
}

/// Why an NVIDIA EP cannot run, naming every missing piece; None when all deps load.
#[cfg(test)]
pub(crate) fn missing_deps_reason(ep: &str, deps: &[Dep]) -> Option<String> {
    missing_deps_reason_with(ep, deps, None)
}

/// [`missing_deps_reason`]; when a missing CUDA library (runtime, cuDNN) could be downloaded,
/// `download_hint` says how (see [`cuda_download_hint`]).
pub(crate) fn missing_deps_reason_with(
    ep: &str,
    deps: &[Dep],
    download_hint: Option<&str>,
) -> Option<String> {
    let missing: Vec<&Dep> = deps.iter().filter(|d| !d.loadable).collect();
    if missing.is_empty() {
        return None;
    }
    let list: Vec<String> = missing
        .iter()
        .map(|d| format!("{} ({})", d.label, d.names.join(" / ")))
        .collect();
    let fix = match download_hint {
        Some(h) if missing.iter().any(|d| d.cuda_lib) => h.to_string(),
        _ => format!(
            "install it or add its directory to {}",
            if cfg!(windows) {
                "PATH"
            } else {
                "LD_LIBRARY_PATH"
            }
        ),
    };
    Some(format!(
        "{ep} execution provider needs {} which could not be loaded; {fix}",
        list.join(" and ")
    ))
}

/// "CUDA libraries missing — enable allow_large_downloads or run `blue-onyx-prism fetch
/// --resource nvidia-cuda-libs` (1.9 GB), or install CUDA 12 and cuDNN 9" when the catalog has
/// them for this platform; None otherwise.
pub(crate) fn cuda_download_hint(os: &str, arch: &str) -> Option<String> {
    let r = crate::resources::catalog::cuda_libs_for(os, arch)?;
    Some(format!(
        "CUDA libraries missing \u{2014} enable allow_large_downloads or run `blue-onyx-prism \
         fetch --resource {}` ({}), or install CUDA 12 and cuDNN 9",
        r.id,
        crate::resources::catalog::format_size(r.size())
    ))
}

/// Whether any of `names` loads (by name: system search path and libraries already preloaded
/// from `cuda-libs`; or by path from one of `dirs`).
fn any_loadable(names: &[&str], dirs: &[&Path]) -> bool {
    names.iter().any(|n| {
        let mut candidates: Vec<PathBuf> = dirs
            .iter()
            .map(|d| d.join(n))
            .filter(|p| p.is_file())
            .collect();
        candidates.push(PathBuf::from(n));
        candidates.iter().any(|p| {
            // SAFETY: probing NVIDIA runtime libraries (cudart, cuDNN, TensorRT) that the CUDA
            // execution provider would load itself when a session is created; the handle is
            // dropped right away and no symbol is used.
            unsafe { libloading::Library::new(p) }.is_ok()
        })
    })
}

fn nvidia_deps(target: Target, ort_dir: &Path, cuda_libs: Option<&Path>) -> Vec<Dep> {
    let dirs: Vec<&Path> = std::iter::once(ort_dir).chain(cuda_libs).collect();
    let mut deps = vec![
        Dep {
            label: "the ONNX Runtime CUDA provider",
            names: &[dep_names::CUDA_PROVIDER],
            loadable: ort_dir.join(dep_names::CUDA_PROVIDER).is_file(),
            cuda_lib: false,
        },
        Dep {
            label: "the CUDA runtime",
            names: dep_names::CUDART,
            loadable: any_loadable(dep_names::CUDART, &dirs),
            cuda_lib: true,
        },
        Dep {
            label: "cuDNN 9",
            names: dep_names::CUDNN,
            loadable: any_loadable(dep_names::CUDNN, &dirs),
            cuda_lib: true,
        },
    ];
    if target == Target::TensorRt {
        deps.push(Dep {
            label: "the ONNX Runtime TensorRT provider",
            names: &[dep_names::TRT_PROVIDER],
            loadable: ort_dir.join(dep_names::TRT_PROVIDER).is_file(),
            cuda_lib: false,
        });
        deps.push(Dep {
            label: "TensorRT 10",
            names: dep_names::NVINFER,
            loadable: any_loadable(dep_names::NVINFER, &dirs),
            cuda_lib: false,
        });
    }
    deps
}

// ---- runtime ---------------------------------------------------------------------------------

/// The ONNX Runtime library of this process plus the options sessions are created with.
pub struct OrtRuntime {
    lib: LoadedLib,
    cache_dir: Option<PathBuf>,
    intra_threads: usize,
    probe: OrtProbe,
    hw: HardwareInfo,
}

impl std::fmt::Debug for OrtRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtRuntime")
            .field("path", &self.lib.path)
            .field("version", &self.lib.version)
            .field("probe", &self.probe)
            .finish()
    }
}

impl OrtRuntime {
    /// Load the library at `path` (from [`libs::find_onnxruntime`]), then probe its execution
    /// providers.
    ///
    /// `cuda_libs`: the `onnxruntime/cuda-libs` directory of `nvidia-cuda-libs`. When it exists,
    /// an NVIDIA GPU is present and the CUDA provider is installed, its libraries are preloaded
    /// first (see [`libs::preload_cuda_libs`]) so the CUDA provider finds them.
    pub fn new(
        path: &Path,
        opts: &super::CoreOptions,
        hw: &HardwareInfo,
        cuda_libs: Option<&Path>,
    ) -> Result<Self> {
        let lib = load_library(path)
            .with_context(|| format!("loading ONNX Runtime from {}", path.display()))?;
        let ort_dir = lib.path.parent().unwrap_or(Path::new("."));
        let cuda_libs = cuda_libs.filter(|d| d.is_dir());
        if let Some(dir) = cuda_libs
            && hw.has_vendor(GpuVendor::Nvidia)
            && ort_dir.join(dep_names::CUDA_PROVIDER).is_file()
        {
            let r = libs::preload_cuda_libs(dir);
            tracing::info!(
                "preloaded {} CUDA libraries from {}",
                r.loaded.len(),
                dir.display()
            );
            for (p, e) in &r.failed {
                tracing::warn!("could not preload {}: {e}", p.display());
            }
        }
        let probe = probe_providers(&lib, hw, cuda_libs);
        tracing::info!(
            "ONNX Runtime {} loaded from {} (flavor {}); execution providers: {}",
            lib.version,
            lib.path.display(),
            probe.flavor.as_deref().unwrap_or("unknown"),
            probe
                .providers
                .iter()
                .map(|p| match &p.error {
                    None => p.target.display_name().to_string(),
                    Some(e) => format!("{} (unusable: {e})", p.target.display_name()),
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(Self {
            lib,
            cache_dir: opts.cache_dir.clone(),
            intra_threads: opts.intra_threads,
            probe,
            hw: hw.clone(),
        })
    }

    pub fn probe(&self) -> &OrtProbe {
        &self.probe
    }

    /// Library version, e.g. "1.24.4".
    pub fn version(&self) -> &str {
        &self.lib.version
    }

    pub fn library_path(&self) -> &Path {
        &self.lib.path
    }

    /// Create a session for `req` on exactly `cand.device` (an ONNX Runtime target).
    pub fn compile(&self, cand: &Candidate, req: &LoadRequest) -> Result<OrtCompiled> {
        let target = cand.device.target;
        if cand.device.runtime != Runtime::Ort {
            bail!("{} is not an ONNX Runtime device", cand.device);
        }
        if let Some(reason) = self.probe.ep_error(target) {
            bail!("{} cannot run: {reason}", cand.device);
        }
        let path = onnx_path_for(&req.path).with_context(|| {
            format!(
                "{} cannot load {}: {NEEDS_ONNX}",
                cand.device,
                req.path.display()
            )
        })?;
        let t0 = Instant::now();
        let index = cand.device.index.unwrap_or(0);
        // Ports first. On the CPU the real session doubles as the metadata source (rebuilt only
        // when dynamic dims need pinning); for accelerators an unoptimized CPU session reads
        // them, so the EP compiles once, already with the pinned shapes.
        let session = if target == Target::Cpu {
            Some(self.build_session(target, index, &path, &[])?)
        } else {
            None
        };
        let ((mut inputs, syms), meta_outputs) = {
            let meta;
            let s = match &session {
                Some(s) => s,
                None => {
                    meta = self.metadata_session(&path)?;
                    &meta
                }
            };
            (ports(s.inputs()), ports(s.outputs()).0)
        };
        let image_idx = find_image_input(&inputs).with_context(|| {
            format!(
                "model {}: no image input (named 'images'/'input' or 4-D) among {:?}",
                path.display(),
                inputs
                    .iter()
                    .map(|p| (&p.name, &p.shape))
                    .collect::<Vec<_>>()
            )
        })?;
        let extra_idx = extra_input_index(&inputs, image_idx, "orig_target_sizes");
        let image_default = dynamic_image_shape(&path, &meta_outputs);
        let overrides = dimension_overrides(&inputs, &syms, image_idx, extra_idx, &image_default);
        if let Some(reason) = unsupported_on(target, &inputs, &syms, &overrides) {
            bail!(
                "{} cannot run model {}: {reason}",
                cand.device,
                path.display()
            );
        }
        if !overrides.is_empty() {
            tracing::info!(
                "model {}: pinning dynamic dimensions {overrides:?}",
                path.display()
            );
        }
        let session = match session {
            Some(s) if overrides.is_empty() => s,
            _ => {
                let s = self.build_session(target, index, &path, &overrides)?;
                inputs = ports(s.inputs()).0;
                s
            }
        };
        // Report (and feed) static shapes for the image and `orig_target_sizes` inputs.
        let img = &mut inputs[image_idx];
        if img.shape.len() != 4 {
            bail!(
                "model {}: image input '{}' has rank {} (expected 4-D NCHW, shape {:?})",
                path.display(),
                img.name,
                img.shape.len(),
                img.shape
            );
        }
        img.shape = fixed_shape(&img.shape, &image_default);
        if img.elem != PortElem::F32 {
            bail!(
                "model {}: image input '{}' has element type {:?}; only f32 is supported",
                path.display(),
                img.name,
                img.elem
            );
        }
        let input_size = (img.shape[3] as u32, img.shape[2] as u32);
        let image_input = img.name.clone();
        if let Some(i) = extra_idx {
            let p = &mut inputs[i];
            if p.shape.is_empty() {
                p.shape = DEFAULT_TARGET_SIZES_SHAPE.to_vec();
            } else {
                p.shape = fixed_shape(&p.shape, &DEFAULT_TARGET_SIZES_SHAPE);
            }
        }
        let (outputs, _) = ports(session.outputs());
        let compile_ms = t0.elapsed().as_millis() as u64;
        let device = DeviceInfo {
            requested: req.requested.clone(),
            actual: device_actual(cand),
            full_name: self.full_name(cand),
            fell_back: cand.fell_back,
            runtime: Runtime::Ort,
            spec: cand.device.to_string(),
        };
        tracing::info!(
            "model {} loaded on {} in {compile_ms} ms; inputs {:?}, outputs {:?}",
            path.display(),
            device.execution_provider(),
            inputs
                .iter()
                .map(|p| (&p.name, &p.shape))
                .collect::<Vec<_>>(),
            outputs
                .iter()
                .map(|p| (&p.name, &p.shape))
                .collect::<Vec<_>>()
        );
        Ok(OrtCompiled {
            session,
            info: ModelInfo {
                path,
                inputs,
                outputs,
                device,
                image_input,
                input_size,
                compile_ms,
            },
        })
    }

    /// GPU name for CUDA/TensorRT (NVIDIA ordinal) and DirectML (DXGI index); empty otherwise.
    fn full_name(&self, cand: &Candidate) -> String {
        let index = cand.device.index.unwrap_or(0);
        match cand.device.target {
            Target::Cuda | Target::TensorRt => {
                let mut nvidia: Vec<_> = self.hw.gpus_of(GpuVendor::Nvidia).collect();
                nvidia.sort_by_key(|g| g.index);
                nvidia
                    .get(index as usize)
                    .map(|g| g.name.clone())
                    .unwrap_or_default()
            }
            Target::DirectMl => self
                .hw
                .gpus
                .iter()
                .find(|g| g.index == index)
                .map(|g| g.name.clone())
                .unwrap_or_default(),
            _ => String::new(),
        }
    }

    fn cache_subdir(&self, name: &str) -> Option<String> {
        let dir = self.cache_dir.as_ref()?.join(name);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!("could not create cache dir {}: {e}", dir.display());
            return None;
        }
        dir.to_str().map(str::to_string)
    }

    /// An unoptimized CPU session, only to read the model's ports.
    fn metadata_session(&self, path: &Path) -> Result<Session> {
        Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(ort_err)?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))
    }

    fn build_session(
        &self,
        target: Target,
        index: u32,
        path: &Path,
        overrides: &[(String, i64)],
    ) -> Result<Session> {
        let mut b: SessionBuilder = Session::builder().map_err(ort_err)?;
        if self.intra_threads > 0 {
            b = b.with_intra_threads(self.intra_threads).map_err(ort_err)?;
        }
        for (name, value) in overrides {
            b = b.with_dimension_override(name, *value).map_err(ort_err)?;
        }
        let device_id = index as i32;
        let eps = match target {
            Target::Cpu => Vec::new(),
            Target::Cuda => vec![
                ep::CUDA::default()
                    .with_device_id(device_id)
                    .build()
                    .error_on_failure(),
            ],
            Target::TensorRt => {
                let mut trt = ep::TensorRT::default()
                    .with_device_id(device_id)
                    .with_fp16(true);
                if let Some(dir) = self.cache_subdir("tensorrt") {
                    trt = trt
                        .with_engine_cache(true)
                        .with_engine_cache_path(&dir)
                        .with_timing_cache(true)
                        .with_timing_cache_path(&dir);
                }
                vec![
                    trt.build().error_on_failure(),
                    // Nodes TensorRT does not take run on CUDA rather than the CPU.
                    ep::CUDA::default()
                        .with_device_id(device_id)
                        .build()
                        .fail_silently(),
                ]
            }
            Target::DirectMl => {
                b = b
                    .with_memory_pattern(false)
                    .map_err(ort_err)?
                    .with_parallel_execution(false)
                    .map_err(ort_err)?;
                vec![
                    ep::DirectML::default()
                        .with_device_id(device_id)
                        .build()
                        .error_on_failure(),
                ]
            }
            Target::CoreMl => {
                let mut coreml = ep::CoreML::default()
                    .with_model_format(ep::coreml::ModelFormat::MLProgram)
                    .with_compute_units(ep::coreml::ComputeUnits::All);
                if let Some(dir) = self.cache_subdir("coreml") {
                    coreml = coreml.with_model_cache_dir(dir);
                }
                vec![coreml.build().error_on_failure()]
            }
            Target::Gpu | Target::Npu => bail!(
                "ONNX Runtime has no '{}' execution provider",
                target.as_str()
            ),
        };
        if !eps.is_empty() {
            b = b.with_execution_providers(eps).map_err(|e| {
                anyhow::anyhow!(
                    "registering the {} execution provider: {e}",
                    target.display_name()
                )
            })?;
        }
        b.commit_from_file(path).map_err(|e| {
            anyhow::anyhow!(
                "creating ONNX Runtime {} session for {}: {e}",
                target.display_name(),
                path.display()
            )
        })
    }
}

/// Input dimensions that stay dynamic after the free-dimension `overrides` (from
/// [`dimension_overrides`]), as "input[axis]". `symbols` is per input, as from [`ports`].
pub(crate) fn unpinned_dims(
    inputs: &[PortSpec],
    symbols: &[Vec<String>],
    overrides: &[(String, i64)],
) -> Vec<String> {
    let mut out = Vec::new();
    for (i, port) in inputs.iter().enumerate() {
        for (axis, &dim) in port.shape.iter().enumerate() {
            if dim >= 0 {
                continue;
            }
            let pinned = symbols
                .get(i)
                .and_then(|s| s.get(axis))
                .filter(|s| !s.is_empty())
                .is_some_and(|s| overrides.iter().any(|(name, _)| name == s));
            if !pinned {
                out.push(format!("{}[{axis}]", port.name));
            }
        }
    }
    out
}

/// Why a model with these inputs must not be run on `target`, if it must not. `symbols` and
/// `overrides` are the dimension names of `inputs` and the free-dimension overrides the session
/// will be created with.
///
/// RT-DETR (`orig_target_sizes` input) on CoreML (ONNX Runtime 1.24.4, the pinned version,
/// measured on an M1): with a *dynamic batch* the Core ML program has unbounded dimensions
/// (`_postprocessor_Tile_output_0`, the encoder reshapes), Core ML rejects it and MPSGraph
/// *aborts the process* while specializing the first partition (`'mps.concat' op invalid input
/// tensor shapes` -> `MPSGraphExecutable.mm: failed assertion 'original module failed
/// verification'`) instead of returning an error. With every input dimension pinned (the
/// xnorpx RT-DETRv2 exports name the batch `N`, which [`dimension_overrides`] pins to 1) the
/// graph is fully static, 673 of 691 nodes (rt-detrv2-s) go to Core ML in 7 partitions, and the
/// default options (MLProgram, compute units ALL: fp32 on the GPU) run it correctly (scores
/// within 2e-6 of the CPU provider) and ~1.6-1.9x faster than `openvino:cpu`. Measured
/// alternatives on rt-detrv2-s, infer p50: ALL / CPUAndGPU 51 ms; CPUAndNeuralEngine and
/// CPUOnly 105 ms; NeuralNetwork format 82 ms (scores off by 7e-3); low-precision GPU
/// accumulation and FastPrediction no change; a dynamic batch with RequireStaticInputShapes
/// avoids the abort but leaves only 163 of 925 nodes to Core ML (237 ms, slower than
/// `openvino:cpu`). So only an RT-DETR model whose inputs keep a dynamic dimension that cannot
/// be pinned (unnamed) is refused; the plan then falls back to the next candidate (CPU).
pub(crate) fn unsupported_on(
    target: Target,
    inputs: &[PortSpec],
    symbols: &[Vec<String>],
    overrides: &[(String, i64)],
) -> Option<String> {
    let rtdetr = inputs.iter().any(|p| p.name == "orig_target_sizes");
    if target != Target::CoreMl || !rtdetr {
        return None;
    }
    let dynamic = unpinned_dims(inputs, symbols, overrides);
    (!dynamic.is_empty()).then(|| {
        format!(
            "RT-DETR models with dynamic input dimensions ({}) are not supported by the CoreML \
             execution provider (MPSGraph aborts the process); export the model with a static \
             or named batch dimension",
            dynamic.join(", ")
        )
    })
}

/// `executionProvider`-independent device string: "cpu", "cuda:0", "coreml", ...
fn device_actual(cand: &Candidate) -> String {
    let t = cand.device.target.as_str();
    match cand.device.index {
        Some(i) => format!("{t}:{i}"),
        None => t.to_string(),
    }
}

fn probe_providers(lib: &LoadedLib, hw: &HardwareInfo, cuda_libs: Option<&Path>) -> OrtProbe {
    let dir = lib.path.parent().unwrap_or(Path::new("."));
    let flavor = libs::read_ort_flavor(dir);
    let mut providers = vec![EpStatus::usable(Target::Cpu)];
    let checks: [(Target, Box<dyn ExecutionProvider>); 4] = [
        (Target::Cuda, Box::new(ep::CUDA::default())),
        (Target::TensorRt, Box::new(ep::TensorRT::default())),
        (Target::DirectMl, Box::new(ep::DirectML::default())),
        (Target::CoreMl, Box::new(ep::CoreML::default())),
    ];
    for (target, provider) in checks {
        match provider.is_available() {
            Ok(false) => {}
            Err(e) => providers.push(EpStatus::broken(
                target,
                format!("querying {}: {e}", provider.name()),
            )),
            Ok(true) => {
                let reason = match target {
                    Target::Cuda | Target::TensorRt if !hw.has_vendor(GpuVendor::Nvidia) => {
                        Some("no NVIDIA GPU detected".to_string())
                    }
                    Target::Cuda | Target::TensorRt => missing_deps_reason_with(
                        target.display_name(),
                        &nvidia_deps(target, dir, cuda_libs),
                        cuda_download_hint(&hw.os, &hw.arch).as_deref(),
                    ),
                    _ => None,
                };
                providers.push(EpStatus {
                    target,
                    error: reason,
                });
            }
        }
    }
    OrtProbe {
        error: None,
        flavor,
        providers,
    }
}

// ---- ports and shapes ------------------------------------------------------------------------

fn port_elem(t: TensorElementType) -> PortElem {
    match t {
        TensorElementType::Float32 => PortElem::F32,
        TensorElementType::Float16 => PortElem::F16,
        TensorElementType::Int64 => PortElem::I64,
        TensorElementType::Int32 => PortElem::I32,
        TensorElementType::Uint8 => PortElem::U8,
        _ => PortElem::Other,
    }
}

/// Port specs plus, per port, the symbolic names of its dimensions ("" when unnamed).
fn ports(outlets: &[::ort::value::Outlet]) -> (Vec<PortSpec>, Vec<Vec<String>>) {
    outlets
        .iter()
        .map(|o| match o.dtype() {
            ValueType::Tensor {
                ty,
                shape,
                dimension_symbols,
            } => (
                PortSpec {
                    name: o.name().to_string(),
                    shape: shape.to_vec(),
                    elem: port_elem(*ty),
                },
                dimension_symbols.to_vec(),
            ),
            _ => (
                PortSpec {
                    name: o.name().to_string(),
                    shape: Vec::new(),
                    elem: PortElem::Other,
                },
                Vec::new(),
            ),
        })
        .unzip()
}

/// `shape` with every dynamic (negative) dim replaced by the default at that position.
pub(crate) fn fixed_shape(shape: &[i64], defaults: &[i64]) -> Vec<i64> {
    shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d >= 0 {
                d
            } else {
                defaults.get(i).copied().unwrap_or(1)
            }
        })
        .collect()
}

/// Free-dimension overrides that pin the named dynamic dims of the image input (to
/// `image_default`, usually `[1,3,640,640]`) and of `orig_target_sizes` (to `[1,2]`). A symbol
/// that would need two different values is left dynamic.
pub(crate) fn dimension_overrides(
    inputs: &[PortSpec],
    symbols: &[Vec<String>],
    image_idx: usize,
    extra_idx: Option<usize>,
    image_default: &[i64; 4],
) -> Vec<(String, i64)> {
    let mut out: Vec<(String, i64)> = Vec::new();
    let mut conflicts: Vec<String> = Vec::new();
    let targets = std::iter::once((image_idx, &image_default[..]))
        .chain(extra_idx.map(|i| (i, &DEFAULT_TARGET_SIZES_SHAPE[..])));
    for (idx, defaults) in targets {
        let (Some(port), Some(syms)) = (inputs.get(idx), symbols.get(idx)) else {
            continue;
        };
        for (pos, &dim) in port.shape.iter().enumerate() {
            let Some(sym) = syms.get(pos).filter(|s| !s.is_empty()) else {
                continue;
            };
            if dim >= 0 {
                continue;
            }
            let value = defaults.get(pos).copied().unwrap_or(1);
            match out.iter().find(|(s, _)| s == sym) {
                Some((_, v)) if *v != value => conflicts.push(sym.clone()),
                Some(_) => {}
                None => out.push((sym.clone(), value)),
            }
        }
    }
    out.retain(|(s, _)| !conflicts.contains(s));
    out
}

// ---- compiled model and inference ------------------------------------------------------------

/// An ONNX Runtime session plus its metadata. `Session` is `Send`, so this moves into the worker.
pub struct OrtCompiled {
    session: Session,
    pub info: ModelInfo,
}

impl OrtCompiled {
    pub fn into_backend(self) -> Result<OrtBackend> {
        let image_idx = self
            .info
            .inputs
            .iter()
            .position(|p| p.name == self.info.image_input)
            .with_context(|| {
                format!(
                    "model {}: image input '{}' not among inputs",
                    self.info.path.display(),
                    self.info.image_input
                )
            })?;
        let image_shape = self.info.inputs[image_idx].shape.clone();
        Ok(OrtBackend {
            session: self.session,
            info: self.info,
            image_idx,
            image_shape,
        })
    }
}

/// Per-worker ONNX Runtime inference state.
pub struct OrtBackend {
    session: Session,
    info: ModelInfo,
    image_idx: usize,
    image_shape: Vec<i64>,
}

/// An owned input tensor of `port`'s element type built from `extra` (converted as needed).
pub(crate) fn extra_value(port: &PortSpec, extra: &ExtraInput) -> Result<DynValue> {
    let n: usize = extra.shape.iter().product();
    let len = match &extra.data {
        ExtraData::I64(v) => v.len(),
        ExtraData::I32(v) => v.len(),
        ExtraData::F32(v) => v.len(),
    };
    if len != n {
        bail!(
            "extra input '{}': {len} values for shape {:?}",
            extra.name,
            extra.shape
        );
    }
    let shape: Vec<i64> = extra.shape.iter().map(|&d| d as i64).collect();
    let ctx = || format!("creating input tensor '{}'", extra.name);
    Ok(match port.elem {
        PortElem::I64 => {
            let v: Vec<i64> = match &extra.data {
                ExtraData::I64(v) => v.clone(),
                ExtraData::I32(v) => v.iter().map(|&x| x as i64).collect(),
                ExtraData::F32(v) => v.iter().map(|&x| x as i64).collect(),
            };
            Tensor::from_array((shape, v))
                .map_err(ort_err)
                .with_context(ctx)?
                .into_dyn()
        }
        PortElem::I32 => {
            let v: Vec<i32> = match &extra.data {
                ExtraData::I64(v) => v.iter().map(|&x| x as i32).collect(),
                ExtraData::I32(v) => v.clone(),
                ExtraData::F32(v) => v.iter().map(|&x| x as i32).collect(),
            };
            Tensor::from_array((shape, v))
                .map_err(ort_err)
                .with_context(ctx)?
                .into_dyn()
        }
        PortElem::F32 => {
            let v: Vec<f32> = match &extra.data {
                ExtraData::I64(v) => v.iter().map(|&x| x as f32).collect(),
                ExtraData::I32(v) => v.iter().map(|&x| x as f32).collect(),
                ExtraData::F32(v) => v.clone(),
            };
            Tensor::from_array((shape, v))
                .map_err(ort_err)
                .with_context(ctx)?
                .into_dyn()
        }
        other => bail!(
            "extra input '{}': unsupported port element type {other:?}",
            extra.name
        ),
    })
}

/// Copy one output value into an owned [`NamedOutput`].
fn read_output(name: &str, v: &DynValue) -> Result<NamedOutput> {
    let ty = v.dtype().tensor_type();
    let (shape, data) = match ty {
        Some(TensorElementType::Float32) => {
            let (s, d) = v.try_extract_tensor::<f32>().map_err(ort_err)?;
            (s.to_vec(), OutputBuf::F32(d.to_vec()))
        }
        Some(TensorElementType::Int64) => {
            let (s, d) = v.try_extract_tensor::<i64>().map_err(ort_err)?;
            (s.to_vec(), OutputBuf::I64(d.to_vec()))
        }
        Some(TensorElementType::Int32) => {
            let (s, d) = v.try_extract_tensor::<i32>().map_err(ort_err)?;
            (s.to_vec(), OutputBuf::I32(d.to_vec()))
        }
        other => bail!("output '{name}': unsupported element type {other:?}"),
    };
    Ok(NamedOutput {
        name: name.to_string(),
        shape: shape.iter().map(|&d| d.max(0) as usize).collect(),
        data,
    })
}

impl OrtBackend {
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Run one inference. `chw` must be `3 * w * h` f32 values in CHW order.
    pub fn infer(&mut self, chw: &[f32], extra: &[ExtraInput]) -> Result<Vec<NamedOutput>> {
        let where_ = || {
            format!(
                "model {} on {}",
                self.info.path.display(),
                self.info.device.spec
            )
        };
        let expected: i64 = self.image_shape.iter().product();
        if chw.len() as i64 != expected {
            bail!(
                "{}: input '{}' expects {expected} f32 values ({:?}), got {}",
                where_(),
                self.info.image_input,
                self.image_shape,
                chw.len()
            );
        }
        let image = TensorRef::from_array_view((self.image_shape.clone(), chw))
            .map_err(ort_err)
            .with_context(|| format!("{}: creating the image tensor", where_()))?;
        let mut inputs: Vec<(String, SessionInputValue<'_>)> = Vec::with_capacity(1 + extra.len());
        inputs.push((self.info.image_input.clone(), image.into()));
        for e in extra {
            let i = extra_input_index(&self.info.inputs, self.image_idx, &e.name).with_context(
                || {
                    format!(
                        "{}: extra input '{}' not among model inputs {:?}",
                        where_(),
                        e.name,
                        self.info.inputs.iter().map(|p| &p.name).collect::<Vec<_>>()
                    )
                },
            )?;
            let port = &self.info.inputs[i];
            let value = extra_value(port, e).with_context(where_)?;
            inputs.push((port.name.clone(), value.into()));
        }
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("{}: inference failed: {e}", where_()))?;
        let mut outs = Vec::with_capacity(self.info.outputs.len());
        for spec in &self.info.outputs {
            let v = outputs
                .get(&spec.name)
                .with_context(|| format!("{}: missing output '{}'", where_(), spec.name))?;
            outs.push(read_output(&spec.name, v).with_context(where_)?);
        }
        Ok(outs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn port(name: &str, shape: &[i64], elem: PortElem) -> PortSpec {
        PortSpec {
            name: name.into(),
            shape: shape.to_vec(),
            elem,
        }
    }

    fn syms(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn versions() {
        assert_eq!(parse_version("1.24.4"), Some((1, 24)));
        assert_eq!(parse_version(" 1.17.0\n"), Some((1, 17)));
        assert_eq!(parse_version("garbage"), None);
        assert!(version_supported("1.24.4", 17));
        assert!(version_supported("1.17.0", 17));
        assert!(!version_supported("1.16.3", 17));
        assert!(!version_supported("2.0.0", 17));
        assert!(!version_supported("", 17));
    }

    #[test]
    fn fixed_shapes() {
        assert_eq!(
            fixed_shape(&[-1, 3, -1, -1], &DEFAULT_IMAGE_SHAPE),
            vec![1, 3, 640, 640]
        );
        assert_eq!(
            fixed_shape(&[1, 3, 480, 640], &DEFAULT_IMAGE_SHAPE),
            vec![1, 3, 480, 640]
        );
        assert_eq!(
            fixed_shape(&[-1, 2], &DEFAULT_TARGET_SIZES_SHAPE),
            vec![1, 2]
        );
    }

    #[test]
    fn overrides_for_named_dynamic_dims() {
        // RT-DETR style: batch "N" shared by both inputs, image H/W named.
        let inputs = [
            port("images", &[-1, 3, -1, -1], PortElem::F32),
            port("orig_target_sizes", &[-1, 2], PortElem::I64),
        ];
        let s = [syms(&["N", "", "H", "W"]), syms(&["N", ""])];
        assert_eq!(
            dimension_overrides(&inputs, &s, 0, Some(1), &DEFAULT_IMAGE_SHAPE),
            vec![
                ("N".to_string(), 1),
                ("H".to_string(), 640),
                ("W".to_string(), 640)
            ]
        );
        // Static model: nothing to do.
        let st = [port("images", &[1, 3, 640, 640], PortElem::F32)];
        assert!(
            dimension_overrides(
                &st,
                &[syms(&["", "", "", ""])],
                0,
                None,
                &DEFAULT_IMAGE_SHAPE
            )
            .is_empty()
        );
        // Unnamed dynamic dims cannot be overridden.
        let un = [port("images", &[-1, 3, -1, -1], PortElem::F32)];
        assert!(
            dimension_overrides(
                &un,
                &[syms(&["", "", "", ""])],
                0,
                None,
                &DEFAULT_IMAGE_SHAPE
            )
            .is_empty()
        );
        // One symbol for H and the channel count would need 3 and 640: left dynamic.
        let bad = [port("images", &[1, -1, -1, 640], PortElem::F32)];
        assert!(
            dimension_overrides(
                &bad,
                &[syms(&["", "d", "d", ""])],
                0,
                None,
                &DEFAULT_IMAGE_SHAPE
            )
            .is_empty()
        );
    }

    #[test]
    fn rtdetr_on_coreml_needs_pinned_dims() {
        let inputs = [
            port("images", &[-1, 3, 640, 640], PortElem::F32),
            port("orig_target_sizes", &[-1, 2], PortElem::I64),
        ];
        let named = [syms(&["N", "", "", ""]), syms(&["N", ""])];
        let ov = dimension_overrides(&inputs, &named, 0, Some(1), &DEFAULT_IMAGE_SHAPE);
        assert_eq!(ov, vec![("N".to_string(), 1)]);
        assert!(unpinned_dims(&inputs, &named, &ov).is_empty());
        // Named batch, pinned: runs on CoreML.
        assert_eq!(unsupported_on(Target::CoreMl, &inputs, &named, &ov), None);
        // Without the override (or with unnamed dims) the batch stays dynamic: refused.
        assert_eq!(
            unpinned_dims(&inputs, &named, &[]),
            vec!["images[0]", "orig_target_sizes[0]"]
        );
        let r = unsupported_on(Target::CoreMl, &inputs, &named, &[]).unwrap();
        assert!(r.contains("images[0], orig_target_sizes[0]"), "{r}");
        let unnamed = [syms(&["", "", "", ""]), syms(&["", ""])];
        let ov = dimension_overrides(&inputs, &unnamed, 0, Some(1), &DEFAULT_IMAGE_SHAPE);
        assert!(unsupported_on(Target::CoreMl, &inputs, &unnamed, &ov).is_some());
        // Other targets and non-RT-DETR models are never refused.
        assert_eq!(unsupported_on(Target::Cpu, &inputs, &unnamed, &[]), None);
        let yolo = [port("images", &[-1, 3, -1, -1], PortElem::F32)];
        assert_eq!(
            unsupported_on(Target::CoreMl, &yolo, &[syms(&["", "", "", ""])], &[]),
            None
        );
        // Fully static RT-DETR: nothing to pin, runs.
        let st = [
            port("images", &[1, 3, 640, 640], PortElem::F32),
            port("orig_target_sizes", &[1, 2], PortElem::I64),
        ];
        assert_eq!(unsupported_on(Target::CoreMl, &st, &[], &[]), None);
    }

    #[test]
    fn nvidia_reasons_name_what_is_missing() {
        let dep = |label, names, loadable| Dep {
            label,
            names,
            loadable,
            cuda_lib: label != "TensorRT 10",
        };
        assert_eq!(
            missing_deps_reason(
                "CUDA",
                &[
                    dep("the CUDA runtime", &["cudart64_12.dll"], true),
                    dep("cuDNN 9", &["cudnn64_9.dll"], true)
                ]
            ),
            None
        );
        let r = missing_deps_reason(
            "CUDA",
            &[
                dep(
                    "the CUDA runtime",
                    &["libcudart.so.12", "libcudart.so.13"],
                    true,
                ),
                dep("cuDNN 9", &["libcudnn.so.9"], false),
            ],
        )
        .unwrap();
        assert!(
            r.starts_with("CUDA execution provider needs cuDNN 9 (libcudnn.so.9)"),
            "{r}"
        );
        assert!(!r.contains("cudart"), "{r}");
        let r = missing_deps_reason(
            "TensorRT",
            &[
                dep("the CUDA runtime", &["a", "b"], false),
                dep("TensorRT 10", &["c"], false),
            ],
        )
        .unwrap();
        assert!(
            r.contains("the CUDA runtime (a / b) and TensorRT 10 (c)"),
            "{r}"
        );

        // Downloadable CUDA libraries: the reason says how to get them.
        let hint = cuda_download_hint("linux", "x86_64").unwrap();
        assert!(hint.contains("fetch --resource nvidia-cuda-libs"), "{hint}");
        assert!(
            hint.contains("allow_large_downloads") && hint.contains("GB"),
            "{hint}"
        );
        assert!(cuda_download_hint("macos", "aarch64").is_none());
        let r = missing_deps_reason_with(
            "CUDA",
            &[dep("cuDNN 9", &["libcudnn.so.9"], false)],
            Some(&hint),
        )
        .unwrap();
        assert!(r.ends_with(&hint), "{r}");
        // Only TensorRT missing: not fixed by the download.
        let r = missing_deps_reason_with(
            "TensorRT",
            &[dep("TensorRT 10", &["c"], false)],
            Some(&hint),
        )
        .unwrap();
        assert!(r.contains("install it"), "{r}");
    }

    #[test]
    fn extra_values_are_validated() {
        let e = ExtraInput {
            name: "orig_target_sizes".into(),
            shape: vec![1, 2],
            data: ExtraData::I64(vec![640]),
        };
        let p = port("orig_target_sizes", &[1, 2], PortElem::I64);
        let err = extra_value(&p, &e).unwrap_err();
        assert!(format!("{err:#}").contains("1 values for shape [1, 2]"));
        let e = ExtraInput {
            data: ExtraData::I64(vec![640, 640]),
            ..e
        };
        let p = port("orig_target_sizes", &[1, 2], PortElem::U8);
        assert!(extra_value(&p, &e).is_err());
    }
}
