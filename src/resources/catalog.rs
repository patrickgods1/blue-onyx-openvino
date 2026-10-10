//! Resource catalog (docs/PLAN.md, "Resource catalog"): every file the service can download,
//! pinned by URL, SHA-256 and size. Pure data, no I/O.
//!
//! One table per resource kind:
//! - [`OPENVINO`]: `openvino-runtime` per `(os, arch)`, provides `openvino:*`.
//! - [`ONNXRUNTIME`]: `onnxruntime-{cpu,cuda,directml,coreml}` per `(os, arch)`. `cuda` includes
//!   TensorRT's provider library (TensorRT itself is not downloadable), `directml` is two NuGet
//!   packages (ONNX Runtime + `DirectML.dll`), `coreml` is the macOS package (which also serves
//!   `cpu` there).
//! - [`CUDA_LIBS`]: `nvidia-cuda-libs`, NVIDIA's CUDA 12.8 / cuDNN 9 redistributable wheels from
//!   PyPI (Windows x64, Linux x64). Large and opt-in.
//! - [`MODELS`]: the Hugging Face models of `download.rs` (`.onnx` + `.yaml`), pinned to a repo
//!   commit so the hashes cannot drift.
//!
//! This is the single source of truth for the pinned versions and URLs: `setup_openvino` and
//! `setup_onnxruntime` read their packages from here.
//!
//! Where the hashes come from: OpenVINO `<archive>.sha256` files on storage.openvinotoolkit.org;
//! the GitHub release API `digest` for ONNX Runtime archives; the PyPI JSON API for the NVIDIA
//! wheels; Hugging Face LFS `oid`s for `.onnx` files; NuGet packages and `.yaml` files were
//! downloaded and hashed (the yaml git blob ids match the Hugging Face tree).

use crate::backend::spec::{Runtime, Target};
use crate::model::ModelFamilyKind;
use serde::{Serialize, Serializer};

/// Pinned OpenVINO runtime version.
pub const OPENVINO_VERSION: &str = "2026.4.0";
/// Pinned ONNX Runtime version. The `ort` crate is built without `api-NN` features, so any ONNX
/// Runtime 1.17+ library loads; 1.24.4 is the newest release that ships every package we use
/// (the DirectML NuGet package stops at 1.24.x, and 1.25+ renamed the GPU archives).
pub const ONNXRUNTIME_VERSION: &str = "1.24.4";
/// Pinned DirectML redistributable (NuGet `Microsoft.AI.DirectML`).
pub const DIRECTML_VERSION: &str = "1.15.4";
/// Version label of `nvidia-cuda-libs`: CUDA 12.8 Update 1 components (ONNX Runtime 1.24's CUDA
/// 12 build needs CUDA >= 12.8) and cuDNN 9.8.
pub const CUDA_LIBS_VERSION: &str = "cuda-12.8.1+cudnn-9.8.0.87";

/// Downloads larger than this need `allow_large_downloads` (or a click in the UI).
pub const LARGE_DOWNLOAD_BYTES: u64 = 500 * 1024 * 1024;

/// Resource ids that are not per-flavor or per-model.
pub const OPENVINO_RUNTIME_ID: &str = "openvino-runtime";
pub const CUDA_LIBS_ID: &str = "nvidia-cuda-libs";
/// Install directory of `nvidia-cuda-libs` under the download root.
pub const CUDA_LIBS_DEST: &str = "onnxruntime/cuda-libs";
/// Model resource ids are `model:<name>` (catalog spelling, matched case-insensitively).
pub const MODEL_ID_PREFIX: &str = "model:";

/// `(os, arch)` as in `std::env::consts::{OS, ARCH}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Platform {
    pub os: &'static str,
    pub arch: &'static str,
}

impl Platform {
    pub const WINDOWS_X64: Platform = Platform::new("windows", "x86_64");
    pub const LINUX_X64: Platform = Platform::new("linux", "x86_64");
    pub const LINUX_ARM64: Platform = Platform::new("linux", "aarch64");
    pub const MACOS_ARM64: Platform = Platform::new("macos", "aarch64");

    pub const fn new(os: &'static str, arch: &'static str) -> Self {
        Self { os, arch }
    }

    /// The platform this binary was built for.
    pub fn current() -> Self {
        Self::new(std::env::consts::OS, std::env::consts::ARCH)
    }

    pub fn matches(&self, os: &str, arch: &str) -> bool {
        self.os == os && self.arch == arch
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.os, self.arch)
    }
}

/// Platforms release binaries are built for. Each has an OpenVINO runtime and at least the CPU
/// flavor of ONNX Runtime in the catalog (checked by a test).
pub const SHIPPED_PLATFORMS: &[Platform] = &[
    Platform::WINDOWS_X64,
    Platform::LINUX_X64,
    Platform::LINUX_ARM64,
    Platform::MACOS_ARM64,
];

/// Installed ONNX Runtime flavor (one per process; written to `flavor.txt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Flavor {
    Cpu,
    Cuda,
    DirectMl,
    CoreMl,
}

impl Flavor {
    pub const ALL: [Flavor; 4] = [Flavor::Cpu, Flavor::Cuda, Flavor::DirectMl, Flavor::CoreMl];

    pub const fn as_str(self) -> &'static str {
        match self {
            Flavor::Cpu => "cpu",
            Flavor::Cuda => "cuda",
            Flavor::DirectMl => "directml",
            Flavor::CoreMl => "coreml",
        }
    }

    /// Parse a flavor name (`flavor.txt`, `--flavor`). `gpu` (the GPU package's name) means
    /// `cuda`.
    pub fn parse(s: &str) -> Option<Flavor> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Flavor::Cpu,
            "cuda" | "gpu" => Flavor::Cuda,
            "directml" => Flavor::DirectMl,
            "coreml" => Flavor::CoreMl,
            _ => return None,
        })
    }

    /// Resource id: `onnxruntime-<flavor>`.
    pub const fn resource_id(self) -> &'static str {
        match self {
            Flavor::Cpu => "onnxruntime-cpu",
            Flavor::Cuda => "onnxruntime-cuda",
            Flavor::DirectMl => "onnxruntime-directml",
            Flavor::CoreMl => "onnxruntime-coreml",
        }
    }

    /// Install directory under the download root: `onnxruntime/<flavor>`.
    pub const fn dest(self) -> &'static str {
        match self {
            Flavor::Cpu => "onnxruntime/cpu",
            Flavor::Cuda => "onnxruntime/cuda",
            Flavor::DirectMl => "onnxruntime/directml",
            Flavor::CoreMl => "onnxruntime/coreml",
        }
    }

    /// Execution providers the flavor's package contains (CPU is in every flavor).
    pub const fn targets(self) -> &'static [Target] {
        match self {
            Flavor::Cpu => &[Target::Cpu],
            Flavor::Cuda => &[Target::Cuda, Target::TensorRt, Target::Cpu],
            Flavor::DirectMl => &[Target::DirectMl, Target::Cpu],
            Flavor::CoreMl => &[Target::CoreMl, Target::Cpu],
        }
    }

    /// The flavor whose package has `target`'s execution provider; None for CPU (every flavor
    /// has it) and for non-ORT targets.
    pub fn providing(target: Target) -> Option<Flavor> {
        match target {
            Target::Cuda | Target::TensorRt => Some(Flavor::Cuda),
            Target::DirectMl => Some(Flavor::DirectMl),
            Target::CoreMl => Some(Flavor::CoreMl),
            _ => None,
        }
    }
}

impl std::fmt::Display for Flavor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a resource is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case", tag = "type", content = "flavor")]
pub enum ResourceKind {
    OpenVinoRuntime,
    OnnxRuntime(Flavor),
    /// NVIDIA CUDA/cuDNN shared libraries for the CUDA execution provider.
    CudaLibs,
    Model,
}

/// Container format of a downloaded part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ArchiveKind {
    /// `.zip`, `.nupkg` and `.whl` files.
    Zip,
    /// `.tgz` / `.tar.gz`.
    TarGz,
}

/// Which entries of a part are kept, and where they go (whitelists live in the extractors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Layout {
    /// OpenVINO toolkit archive: `<top>/runtime/...` libraries, layout preserved.
    OpenVino,
    /// ONNX Runtime GitHub release archive: `<top>/lib/<libs>`, copied flat.
    OrtRelease,
    /// `Microsoft.ML.OnnxRuntime.DirectML` NuGet package: `runtimes/win-x64/native/<dlls>`.
    NugetOrt,
    /// `Microsoft.AI.DirectML` NuGet package: `bin/x64-win/DirectML.dll`.
    NugetDirectMl,
    /// NVIDIA PyPI wheel: `nvidia/<component>/{bin,lib}/<libs>`, copied flat.
    NvidiaWheel,
    /// A plain file saved as [`Part::file_name`].
    File,
}

/// One file to download.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Part {
    /// HTTPS URL, pinned to a version (and, for Hugging Face, a commit).
    pub url: &'static str,
    /// Lowercase hex SHA-256 of the whole file.
    pub sha256: &'static str,
    /// Exact size in bytes.
    pub size: u64,
    /// File name to download to (the final name for [`Layout::File`] parts).
    pub file_name: &'static str,
    /// None for plain files.
    pub archive: Option<ArchiveKind>,
    pub layout: Layout,
}

/// What a resource makes possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provides {
    /// Device options of a runtime (`openvino:gpu`, `ort:cuda`, ...).
    Runtime {
        runtime: Runtime,
        targets: &'static [Target],
    },
    /// Libraries that make `ort:cuda` usable (with the `cuda` flavor).
    CudaLibs,
    /// A model servable under `name`.
    Model {
        name: &'static str,
        family: ModelFamilyKind,
    },
}

impl Provides {
    /// Human-readable list: `["openvino:gpu", "openvino:cpu"]`, `["model:IPcam-general"]`.
    pub fn labels(&self) -> Vec<String> {
        match self {
            Provides::Runtime { runtime, targets } => targets
                .iter()
                .map(|t| format!("{}:{}", runtime.as_str(), t.as_str()))
                .collect(),
            Provides::CudaLibs => vec!["ort:cuda".to_string(), "ort:tensorrt".to_string()],
            Provides::Model { name, .. } => vec![format!("{MODEL_ID_PREFIX}{name}")],
        }
    }

    /// Whether this resource provides `runtime:target`.
    pub fn provides_target(&self, runtime: Runtime, target: Target) -> bool {
        matches!(self, Provides::Runtime { runtime: r, targets } if *r == runtime && targets.contains(&target))
    }
}

impl Serialize for Provides {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(self.labels())
    }
}

/// One downloadable resource (possibly several files).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Resource {
    /// `openvino-runtime`, `onnxruntime-cuda`, `nvidia-cuda-libs`, `model:IPcam-general`. Unique
    /// per platform (models are platform-independent).
    pub id: &'static str,
    pub kind: ResourceKind,
    pub version: &'static str,
    /// None = any platform (models).
    pub platform: Option<Platform>,
    pub parts: &'static [Part],
    /// Install directory relative to the download root (`download_dir`, default the exe dir):
    /// `openvino`, `onnxruntime/<flavor>`, `onnxruntime/cuda-libs`. Models use `models`, but the
    /// resolver places them where the config's model path points.
    pub dest: &'static str,
    pub provides: Provides,
    /// Human-readable name for logs and the UI.
    pub title: &'static str,
    /// License of the downloaded content (shown before opt-in downloads).
    pub license: &'static str,
}

impl Resource {
    /// Total download size in bytes.
    pub fn size(&self) -> u64 {
        self.parts.iter().map(|p| p.size).sum()
    }

    /// Larger than [`LARGE_DOWNLOAD_BYTES`]: needs `allow_large_downloads`.
    pub fn is_large(&self) -> bool {
        is_large(self.size())
    }

    /// Available on `(os, arch)` (always true for platform-independent resources).
    pub fn available_on(&self, os: &str, arch: &str) -> bool {
        self.platform.is_none_or(|p| p.matches(os, arch))
    }

    /// The ONNX Runtime flavor of an `onnxruntime-*` resource.
    pub fn flavor(&self) -> Option<Flavor> {
        match self.kind {
            ResourceKind::OnnxRuntime(f) => Some(f),
            _ => None,
        }
    }

    /// Model name of a model resource.
    pub fn model_name(&self) -> Option<&'static str> {
        match self.provides {
            Provides::Model { name, .. } => Some(name),
            _ => None,
        }
    }

    /// "ONNX Runtime CUDA 1.24.4 (281 MB)".
    pub fn describe(&self) -> String {
        format!(
            "{} {} ({})",
            self.title,
            self.version,
            format_size(self.size())
        )
    }
}

/// Larger than [`LARGE_DOWNLOAD_BYTES`].
pub fn is_large(bytes: u64) -> bool {
    bytes > LARGE_DOWNLOAD_BYTES
}

/// "83 MB", "1.7 GB" (decimal units, as download sizes are usually shown).
pub fn format_size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} kB", b / 1e3)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------------------------
// OpenVINO runtime
// ---------------------------------------------------------------------------------------------

macro_rules! ov_url {
    ($dir:literal, $file:literal) => {
        concat!(
            "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.4/",
            $dir,
            "/",
            $file
        )
    };
}

macro_rules! ov_part {
    ($dir:literal, $file:literal, $kind:expr, $sha:literal, $size:literal) => {
        Part {
            url: ov_url!($dir, $file),
            sha256: $sha,
            size: $size,
            file_name: $file,
            archive: Some($kind),
            layout: Layout::OpenVino,
        }
    };
}

const OV_GPU_CPU: &[Target] = &[Target::Gpu, Target::Cpu];
const OV_CPU: &[Target] = &[Target::Cpu];

const fn openvino(
    platform: Platform,
    part: &'static [Part],
    targets: &'static [Target],
) -> Resource {
    Resource {
        id: OPENVINO_RUNTIME_ID,
        kind: ResourceKind::OpenVinoRuntime,
        version: OPENVINO_VERSION,
        platform: Some(platform),
        parts: part,
        dest: "openvino",
        provides: Provides::Runtime {
            runtime: Runtime::OpenVino,
            targets,
        },
        title: "OpenVINO runtime",
        license: "Apache-2.0",
    }
}

/// `openvino-runtime` per platform. The Intel GPU plugin ships only for x86_64.
pub static OPENVINO: &[Resource] = &[
    openvino(
        Platform::WINDOWS_X64,
        &[ov_part!(
            "windows_vc_mt",
            "openvino_toolkit_windows_vc_mt_2026.4.0.22959.99c81491cc3_x86_64.zip",
            ArchiveKind::Zip,
            "82946205ac9f9f8113acf5f7fb5c705ef06a666aa15a8acf52cf2b4e1be90063",
            235127912
        )],
        OV_GPU_CPU,
    ),
    openvino(
        Platform::LINUX_X64,
        &[ov_part!(
            "linux",
            "openvino_toolkit_ubuntu24_2026.4.0.22959.99c81491cc3_x86_64.tgz",
            ArchiveKind::TarGz,
            "0bd86d578beb1e8805655f593315c69bf909008e212f801ed76e3a350927736f",
            120557925
        )],
        OV_GPU_CPU,
    ),
    openvino(
        Platform::LINUX_ARM64,
        &[ov_part!(
            "linux",
            "openvino_toolkit_ubuntu22_2026.4.0.22959.99c81491cc3_arm64.tgz",
            ArchiveKind::TarGz,
            "ed3f103ae8dd7a509c23645f0d34983378d3e609711a82d65301ac77d4a34800",
            43359520
        )],
        OV_CPU,
    ),
    openvino(
        Platform::MACOS_ARM64,
        &[ov_part!(
            "macos",
            "openvino_toolkit_macos_12_6_2026.4.0.22959.99c81491cc3_arm64.tgz",
            ArchiveKind::TarGz,
            "1e67db1fa25e8d736f35f5b3d663d2fd669d51c0d2892f682d0facc9c22b019c",
            44413951
        )],
        OV_CPU,
    ),
];

// ---------------------------------------------------------------------------------------------
// ONNX Runtime
// ---------------------------------------------------------------------------------------------

macro_rules! ort_release {
    ($file:literal, $kind:expr, $sha:literal, $size:literal) => {
        Part {
            url: concat!(
                "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/",
                $file
            ),
            sha256: $sha,
            size: $size,
            file_name: $file,
            archive: Some($kind),
            layout: Layout::OrtRelease,
        }
    };
}

macro_rules! nuget {
    ($id:literal, $ver:literal, $layout:expr, $sha:literal, $size:literal) => {
        Part {
            url: concat!(
                "https://api.nuget.org/v3-flatcontainer/",
                $id,
                "/",
                $ver,
                "/",
                $id,
                ".",
                $ver,
                ".nupkg"
            ),
            sha256: $sha,
            size: $size,
            file_name: concat!($id, ".", $ver, ".nupkg"),
            archive: Some(ArchiveKind::Zip),
            layout: $layout,
        }
    };
}

const fn ort(
    platform: Platform,
    flavor: Flavor,
    parts: &'static [Part],
    title: &'static str,
) -> Resource {
    Resource {
        id: flavor.resource_id(),
        kind: ResourceKind::OnnxRuntime(flavor),
        version: ONNXRUNTIME_VERSION,
        platform: Some(platform),
        parts,
        dest: flavor.dest(),
        provides: Provides::Runtime {
            runtime: Runtime::Ort,
            targets: flavor.targets(),
        },
        title,
        license: "MIT",
    }
}

const ORT_OSX_ARM64: Part = ort_release!(
    "onnxruntime-osx-arm64-1.24.4.tgz",
    ArchiveKind::TarGz,
    "93787795f47e1eee369182e43ed51b9e5da0878ab0346aecf4258979b8bba989",
    30937282
);

/// `onnxruntime-<flavor>` per platform (the combinations `setup-onnxruntime` supports).
pub static ONNXRUNTIME: &[Resource] = &[
    ort(
        Platform::WINDOWS_X64,
        Flavor::Cpu,
        &[ort_release!(
            "onnxruntime-win-x64-1.24.4.zip",
            ArchiveKind::Zip,
            "d2319fddfb6ea4db99ccc4b60c85c517bcd855721f5daa6a06d40d7cb2ee2357",
            74442783
        )],
        "ONNX Runtime CPU",
    ),
    ort(
        Platform::WINDOWS_X64,
        Flavor::Cuda,
        &[ort_release!(
            "onnxruntime-win-x64-gpu-1.24.4.zip",
            ArchiveKind::Zip,
            "ef3337a0b8184eb8beec310f7c83bd50376b3eefc43aab84ac8e452f6987df0a",
            280958859
        )],
        "ONNX Runtime CUDA/TensorRT",
    ),
    ort(
        Platform::WINDOWS_X64,
        Flavor::DirectMl,
        &[
            nuget!(
                "microsoft.ml.onnxruntime.directml",
                "1.24.4",
                Layout::NugetOrt,
                "57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681",
                12458649
            ),
            nuget!(
                "microsoft.ai.directml",
                "1.15.4",
                Layout::NugetDirectMl,
                "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9",
                202292617
            ),
        ],
        "ONNX Runtime DirectML",
    ),
    ort(
        Platform::LINUX_X64,
        Flavor::Cpu,
        &[ort_release!(
            "onnxruntime-linux-x64-1.24.4.tgz",
            ArchiveKind::TarGz,
            "3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747",
            8155822
        )],
        "ONNX Runtime CPU",
    ),
    ort(
        Platform::LINUX_X64,
        Flavor::Cuda,
        &[ort_release!(
            "onnxruntime-linux-x64-gpu-1.24.4.tgz",
            ArchiveKind::TarGz,
            "c5f804ff5d239b436fa59e9f2fb288a39f7eb9552f6a636c8b71e792e91a8808",
            205429115
        )],
        "ONNX Runtime CUDA/TensorRT",
    ),
    ort(
        Platform::LINUX_ARM64,
        Flavor::Cpu,
        &[ort_release!(
            "onnxruntime-linux-aarch64-1.24.4.tgz",
            ArchiveKind::TarGz,
            "866109a9248d057671a039b9d725be4bd86888e3754140e6701ec621be9d4d7e",
            7181958
        )],
        "ONNX Runtime CPU",
    ),
    ort(
        Platform::MACOS_ARM64,
        Flavor::CoreMl,
        &[ORT_OSX_ARM64],
        "ONNX Runtime CoreML",
    ),
    // The macOS package has both EPs; `cpu` is the same archive.
    ort(
        Platform::MACOS_ARM64,
        Flavor::Cpu,
        &[ORT_OSX_ARM64],
        "ONNX Runtime CPU",
    ),
];

// ---------------------------------------------------------------------------------------------
// NVIDIA CUDA / cuDNN libraries (PyPI wheels)
// ---------------------------------------------------------------------------------------------

macro_rules! wheel {
    ($url:literal, $sha:literal, $size:literal) => {
        Part {
            url: $url,
            sha256: $sha,
            size: $size,
            file_name: wheel_file_name($url),
            archive: Some(ArchiveKind::Zip),
            layout: Layout::NvidiaWheel,
        }
    };
}

/// Last path segment of a URL (const, for the wheel table).
const fn wheel_file_name(url: &'static str) -> &'static str {
    let bytes = url.as_bytes();
    let mut i = bytes.len();
    while i > 0 && bytes[i - 1] != b'/' {
        i -= 1;
    }
    let (_, tail) = bytes.split_at(i);
    match std::str::from_utf8(tail) {
        Ok(s) => s,
        Err(_) => panic!("URL is not UTF-8"),
    }
}

const fn cuda_libs(platform: Platform, parts: &'static [Part]) -> Resource {
    Resource {
        id: CUDA_LIBS_ID,
        kind: ResourceKind::CudaLibs,
        version: CUDA_LIBS_VERSION,
        platform: Some(platform),
        parts,
        dest: CUDA_LIBS_DEST,
        provides: Provides::CudaLibs,
        title: "NVIDIA CUDA 12 + cuDNN 9 libraries",
        license: "NVIDIA Software License Agreement (CUDA, cuDNN redistributables)",
    }
}

/// `nvidia-cuda-libs`: what ONNX Runtime's CUDA provider links (cudart, cuBLAS/cuBLASLt, cuDNN 9,
/// cuFFT, cuRAND; checked against `libonnxruntime_providers_cuda.so` 1.24.4), plus nvJitLink
/// (cuFFT's dependency) and NVRTC (cuDNN's runtime-compiled engines). The driver's `libcuda`
/// / `nvcuda.dll` is not included.
pub static CUDA_LIBS: &[Resource] = &[
    cuda_libs(
        Platform::WINDOWS_X64,
        &[
            wheel!(
                "https://files.pythonhosted.org/packages/30/a5/a515b7600ad361ea14bfa13fb4d6687abf500adc270f19e89849c0590492/nvidia_cuda_runtime_cu12-12.8.90-py3-none-win_amd64.whl",
                "c0c6027f01505bfed6c3b21ec546f69c687689aad5f1a377554bc6ca4aa993a8",
                944318
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/70/61/7d7b3c70186fb651d0fbd35b01dbfc8e755f69fd58f817f3d0f642df20c3/nvidia_cublas_cu12-12.8.4.1-py3-none-win_amd64.whl",
                "47e9b82132fa8d2b4944e708049229601448aaad7e6f296f630f2d1a32de35af",
                567544208
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/39/6a/5e9910b2b2c9dcddee9aaef372b3db0f08b9f7eaf1d462f859461a79caf9/nvidia_cudnn_cu12-9.8.0.87-py3-none-win_amd64.whl",
                "b4b5cfddc32aa4180f9d390ee99e9a9f55a89e7087329b41aba4319327e22466",
                684630375
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/7d/ec/ce1629f1e478bb5ccd208986b5f9e0316a78538dd6ab1d0484f012f8e2a1/nvidia_cufft_cu12-11.3.3.83-py3-none-win_amd64.whl",
                "7a64a98ef2a7c47f905aaf8931b69a3a43f27c55530c698bb2ed7c75c0b42cb7",
                192216559
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/b9/75/70c05b2f3ed5be3bb30b7102b6eb78e100da4bbf6944fd6725c012831cab/nvidia_curand_cu12-10.3.9.90-py3-none-win_amd64.whl",
                "f149a8ca457277da854f89cf282d6ef43176861926c7ac85b2a0fbd237c587ec",
                62765309
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/ed/d7/34f02dad2e30c31b10a51f6b04e025e5dd60e5f936af9045a9b858a05383/nvidia_nvjitlink_cu12-12.8.93-py3-none-win_amd64.whl",
                "bd93fbeeee850917903583587f4fc3a4eafa022e34572251368238ab5e6bd67f",
                268553710
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/45/51/52a3d84baa2136cc8df15500ad731d74d3a1114d4c123e043cb608d4a32b/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-win_amd64.whl",
                "7a4b6b2904850fe78e0bd179c4b655c404d4bb799ef03ddc60804247099ae909",
                73586838
            ),
        ],
    ),
    cuda_libs(
        Platform::LINUX_X64,
        &[
            wheel!(
                "https://files.pythonhosted.org/packages/0d/9b/a997b638fcd068ad6e4d53b8551a7d30fe8b404d6f1804abf1df69838932/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
                "adade8dcbd0edf427b7204d480d6066d33902cab2a4707dcfc48a2d0fd44ab90",
                954765
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/dc/61/e24b560ab2e2eaeb3c839129175fb330dfcfc29e5203196e5541a4c44682/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl",
                "8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142",
                594346921
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/77/f0/8236c886a061d203e51247aec2b8e3a8f5350178251ab57237daf2140680/nvidia_cudnn_cu12-9.8.0.87-py3-none-manylinux_2_27_x86_64.whl",
                "d6b02cd0e3e24aa31d0193a8c39fec239354360d7d81055edddb69f35d53a4c8",
                697999707
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/1f/13/ee4e00f30e676b66ae65b4f08cb5bcbb8392c03f54f2d5413ea99a5d1c80/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
                "4d2dd21ec0b88cf61b62e6b43564355e5222e4a3fb394cac0db101f2dd0d4f74",
                193118695
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/fb/aa/6584b56dc84ebe9cf93226a5cde4d99080c8e90ab40f0c27bda7a0f29aa1/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl",
                "b32331d4f4df5d6eefa0554c565b626c7216f87a06a4f56fab27c3b68a830ec9",
                63619976
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/f6/74/86a07f1d0f42998ca31312f998bd3b9a7eff7f52378f4f270c8679c77fb9/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
                "81ff63371a7ebd6e6451970684f916be2eab07321b73c9d244dc2b4da7f73b88",
                39254836
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/05/6b/32f747947df2da6994e999492ab306a903659555dddc0fbdeb9d71f75e52/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
                "a7756528852ef889772a84c6cd89d41dfa74667e24cca16bb31f8f061e3e9994",
                88040029
            ),
        ],
    ),
];

// ---------------------------------------------------------------------------------------------
// Models (Hugging Face, pinned commits)
// ---------------------------------------------------------------------------------------------

/// `xnorpx/rt-detr2-onnx` commit the RT-DETRv2 hashes belong to.
pub const RTDETR_REVISION: &str = "c7d9ed2f629cc8279ea0f6b84e1a86f7b82f0795";
/// `xnorpx/blue-onyx-yolo5` commit the YOLOv5 hashes belong to.
pub const YOLO5_REVISION: &str = "c0c3ff5178f5692236e4c1552d6a040135204f98";

macro_rules! hf_file {
    ($repo_rev:literal, $name:literal, $ext:literal, $sha:expr, $size:expr) => {
        Part {
            url: concat!("https://huggingface.co/", $repo_rev, "/", $name, $ext),
            sha256: $sha,
            size: $size,
            file_name: concat!($name, $ext),
            archive: None,
            layout: Layout::File,
        }
    };
}

/// Every RT-DETRv2 model uses the same COCO class list.
const RTDETR_YAML_SHA: &str = "ed029ee87019fa769269fcff15eda24a15c17dcea42b70f4dd0e3ab744503e7d";
const RTDETR_YAML_SIZE: u64 = 1193;

macro_rules! rtdetr {
    ($name:literal, $onnx_sha:literal, $onnx_size:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::Model,
            version: RTDETR_REVISION,
            platform: None,
            parts: &[
                hf_file!(
                    "xnorpx/rt-detr2-onnx/resolve/c7d9ed2f629cc8279ea0f6b84e1a86f7b82f0795",
                    $name,
                    ".onnx",
                    $onnx_sha,
                    $onnx_size
                ),
                hf_file!(
                    "xnorpx/rt-detr2-onnx/resolve/c7d9ed2f629cc8279ea0f6b84e1a86f7b82f0795",
                    $name,
                    ".yaml",
                    RTDETR_YAML_SHA,
                    RTDETR_YAML_SIZE
                ),
            ],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: ModelFamilyKind::RtDetr,
            },
            title: concat!("Model ", $name),
            license: "Apache-2.0",
        }
    };
}

macro_rules! yolo5 {
    ($name:literal, $onnx_sha:literal, $onnx_size:literal, $yaml_sha:literal, $yaml_size:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::Model,
            version: YOLO5_REVISION,
            platform: None,
            parts: &[
                hf_file!(
                    "xnorpx/blue-onyx-yolo5/resolve/c0c3ff5178f5692236e4c1552d6a040135204f98",
                    $name,
                    ".onnx",
                    $onnx_sha,
                    $onnx_size
                ),
                hf_file!(
                    "xnorpx/blue-onyx-yolo5/resolve/c0c3ff5178f5692236e4c1552d6a040135204f98",
                    $name,
                    ".yaml",
                    $yaml_sha,
                    $yaml_size
                ),
            ],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: ModelFamilyKind::Yolo5,
            },
            title: concat!("Model ", $name),
            license: "AGPL-3.0",
        }
    };
}

/// Downloadable models (the `download.rs` catalog). YOLO26 is not here: its weights are
/// AGPL-3.0 and exported locally with `scripts/export_yolo26.py`.
pub static MODELS: &[Resource] = &[
    rtdetr!(
        "rt-detrv2-s",
        "c849b46a43925d0c24d88c7cab096fc8158ade265b7be1ae433e3effe5758a6a",
        80530290
    ),
    rtdetr!(
        "rt-detrv2-ms",
        "c3315cec9f1ce83956ae1600b6224ead49bf1e5f347fb6a9a2da9740beb50a64",
        125541681
    ),
    rtdetr!(
        "rt-detrv2-m",
        "a3a7bc94da99678f788f35aeb6078e1045f4f038b266316e769c09fb51abc62d",
        132661010
    ),
    rtdetr!(
        "rt-detrv2-l",
        "4c580c4f579a0372c6bea171b0ff3ccea707146a34dc5cea1b817e3c0f4fc0ba",
        169218591
    ),
    rtdetr!(
        "rt-detrv2-x",
        "810a18839401187bd94004bd722d602158302e3f35400840e7cbcfacb71d61c2",
        300394655
    ),
    yolo5!(
        "delivery",
        "1e66d71f3d5bbd6140e15716a43ca0e25fc5132af0a959c2696c11c1bf643e56",
        29571692,
        "17e0d52b7df884f834b9894395847319c0e872725327076df8684ae2045e962b",
        155
    ),
    yolo5!(
        "IPcam-animal",
        "e62a9f8df5299476cf77e2351c711f7554e64b1200cca7756d7b58bbb9f32085",
        29616938,
        "9ca7e805c2c70013311688ebe1b339dfcb38f17216d5d499cc8195cc6b83ca61",
        227
    ),
    yolo5!(
        "ipcam-bird",
        "265363bff2eda770463be1a1054eaaf722dec3703ec5b672a155fd1315a486be",
        84407925,
        "d79b8168621ca94221010b958bdebc08ce899ff68d4d03dc60937685f110b847",
        851
    ),
    yolo5!(
        "IPcam-combined",
        "8f9f39828f979a4440151be2d32660e97f7a41e981af26002c5b638c4db9236a",
        29681724,
        "5224a2b0d32a0ea6ef9bb9a03ed54087280bffd0ee8228af3c46eb800b4a8315",
        308
    ),
    yolo5!(
        "IPcam-dark",
        "ca460c36ff279f0344756b3cadf936323fe01b28ef67b226cf00ca86b05808b7",
        29519768,
        "1e148765187f81e4fd931975810602eb5b7f976fb44ef4e679c578036e4b6558",
        114
    ),
    yolo5!(
        "IPcam-general",
        "3b338be21f271a2924051fc237086a03c49daecd7e0fb2a8e97411ea5a33c6dc",
        29465778,
        "4070e087499ab1bf111efc58d98e6703b9457ce06ff0c22fb79d548074afb9c4",
        52
    ),
    yolo5!(
        "package",
        "b233f5a622a59f5a7ac7b48fbed79b5989ad6225a9569b3a8866a0bd75d9a9d7",
        29452878,
        "6187ff891bb16e442c0ce36593c04be118ab613299ea6060c4b1fd8c2855fe81",
        23
    ),
];

// ---------------------------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------------------------

/// Every resource of every platform.
pub fn all() -> impl Iterator<Item = &'static Resource> {
    OPENVINO
        .iter()
        .chain(ONNXRUNTIME)
        .chain(CUDA_LIBS)
        .chain(MODELS)
}

/// Resources usable on `(os, arch)` (`fetch --all-for-platform`).
pub fn for_platform<'a>(
    os: &'a str,
    arch: &'a str,
) -> impl Iterator<Item = &'static Resource> + 'a {
    all().filter(move |r| r.available_on(os, arch))
}

/// Resource `id` for `(os, arch)`; model ids match case-insensitively.
pub fn find(id: &str, os: &str, arch: &str) -> Option<&'static Resource> {
    let id = id.trim();
    if let Some(name) = strip_prefix_ci(id, MODEL_ID_PREFIX) {
        return model(name);
    }
    for_platform(os, arch).find(|r| r.id.eq_ignore_ascii_case(id))
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    (s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix))
        .then(|| &s[prefix.len()..])
}

/// `openvino-runtime` for `(os, arch)`.
pub fn openvino_runtime(os: &str, arch: &str) -> Option<&'static Resource> {
    OPENVINO.iter().find(|r| r.available_on(os, arch))
}

/// `onnxruntime-<flavor>` for `(os, arch)`.
pub fn onnxruntime(os: &str, arch: &str, flavor: Flavor) -> Option<&'static Resource> {
    ONNXRUNTIME
        .iter()
        .find(|r| r.available_on(os, arch) && r.flavor() == Some(flavor))
}

/// ONNX Runtime flavors downloadable for `(os, arch)`.
pub fn ort_flavors(os: &str, arch: &str) -> Vec<Flavor> {
    Flavor::ALL
        .into_iter()
        .filter(|f| onnxruntime(os, arch, *f).is_some())
        .collect()
}

/// `nvidia-cuda-libs` for `(os, arch)`.
pub fn cuda_libs_for(os: &str, arch: &str) -> Option<&'static Resource> {
    CUDA_LIBS.iter().find(|r| r.available_on(os, arch))
}

/// Model resource by name (case-insensitive; a trailing `.onnx` is accepted), like
/// `download::find`.
pub fn model(name: &str) -> Option<&'static Resource> {
    let n = name.trim();
    let n = n
        .len()
        .checked_sub(5)
        .filter(|&i| n.is_char_boundary(i) && n[i..].eq_ignore_ascii_case(".onnx"))
        .map_or(n, |i| &n[..i]);
    MODELS
        .iter()
        .find(|r| r.model_name().is_some_and(|m| m.eq_ignore_ascii_case(n)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wheel_names_and_model_lookup() {
        assert_eq!(
            CUDA_LIBS[0].parts[0].file_name,
            "nvidia_cuda_runtime_cu12-12.8.90-py3-none-win_amd64.whl"
        );
        assert_eq!(
            model("ipcam-general.ONNX").unwrap().id,
            "model:IPcam-general"
        );
        assert_eq!(model("RT-DETRV2-S").unwrap().id, "model:rt-detrv2-s");
        assert!(model("yolo26s").is_none());
        assert_eq!(
            find("MODEL:ipcam-dark", "linux", "x86_64").unwrap().id,
            "model:IPcam-dark"
        );
        assert_eq!(
            find("onnxruntime-cuda", "windows", "x86_64").unwrap().parts[0].file_name,
            "onnxruntime-win-x64-gpu-1.24.4.zip"
        );
        assert!(find("onnxruntime-cuda", "macos", "aarch64").is_none());
    }

    #[test]
    fn sizes_and_large_flag() {
        assert_eq!(format_size(29465778), "29 MB");
        assert_eq!(format_size(1_800_000_000), "1.8 GB");
        assert_eq!(format_size(52), "52 B");
        assert!(!is_large(LARGE_DOWNLOAD_BYTES));
        assert!(is_large(LARGE_DOWNLOAD_BYTES + 1));
        for r in CUDA_LIBS {
            assert!(r.is_large(), "{} {}", r.id, r.size());
        }
        for r in OPENVINO.iter().chain(ONNXRUNTIME).chain(MODELS) {
            assert!(!r.is_large(), "{} {:?}", r.id, r.platform);
        }
    }

    #[test]
    fn provides_labels() {
        let ov = openvino_runtime("windows", "x86_64").unwrap();
        assert_eq!(ov.provides.labels(), ["openvino:gpu", "openvino:cpu"]);
        assert!(ov.provides.provides_target(Runtime::OpenVino, Target::Gpu));
        let mac = openvino_runtime("macos", "aarch64").unwrap();
        assert!(!mac.provides.provides_target(Runtime::OpenVino, Target::Gpu));
        let json =
            serde_json::to_value(onnxruntime("macos", "aarch64", Flavor::CoreMl).unwrap()).unwrap();
        assert_eq!(
            json["provides"],
            serde_json::json!(["ort:coreml", "ort:cpu"])
        );
        assert_eq!(
            json["kind"],
            serde_json::json!({"type": "onnx-runtime", "flavor": "coreml"})
        );
        for f in Flavor::ALL {
            assert_eq!(Flavor::parse(f.as_str()), Some(f));
        }
        assert_eq!(Flavor::parse(" GPU "), Some(Flavor::Cuda));
    }
}
