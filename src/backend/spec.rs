//! Device spec strings: `auto` or `runtime:target[:index]` (see docs/PLAN.md, "Device spec").
//!
//! | Spec                                   | Meaning                                  |
//! |----------------------------------------|------------------------------------------|
//! | `auto`                                 | ranked choice (6.1: same as `openvino:gpu`) |
//! | `openvino:gpu`, `openvino:gpu.N`       | OpenVINO GPU plugin (`GPU`, `GPU.N`)     |
//! | `openvino:cpu`, `openvino:npu`         | OpenVINO CPU / NPU plugin                |
//! | `ort:cuda[:N]`, `ort:tensorrt[:N]`     | ONNX Runtime NVIDIA execution providers  |
//! | `ort:directml[:N]`, `ort:coreml`       | ONNX Runtime DirectML / CoreML           |
//! | `ort:cpu`                              | ONNX Runtime CPU                         |
//!
//! Legacy bare `GPU`, `GPU.N`, `CPU` and `NPU` mean OpenVINO. Parsing is case-insensitive and
//! trims surrounding whitespace; [`std::fmt::Display`] gives the canonical lowercase form, which
//! parses back to the same value. Pure: no runtime libraries needed.

use anyhow::{Result, bail};
use std::fmt;

/// Inference runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Runtime {
    #[default]
    OpenVino,
    Ort,
}

impl Runtime {
    /// Spec prefix (`openvino`, `ort`).
    pub fn as_str(self) -> &'static str {
        match self {
            Runtime::OpenVino => "openvino",
            Runtime::Ort => "ort",
        }
    }

    /// Human readable name, as used in `executionProvider`.
    pub fn display_name(self) -> &'static str {
        match self {
            Runtime::OpenVino => "OpenVINO",
            Runtime::Ort => "ONNX Runtime",
        }
    }
}

/// Device or execution provider within a runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    Gpu,
    Cpu,
    Npu,
    Cuda,
    TensorRt,
    DirectMl,
    CoreMl,
}

impl Target {
    /// Spec name (`gpu`, `tensorrt`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Target::Gpu => "gpu",
            Target::Cpu => "cpu",
            Target::Npu => "npu",
            Target::Cuda => "cuda",
            Target::TensorRt => "tensorrt",
            Target::DirectMl => "directml",
            Target::CoreMl => "coreml",
        }
    }

    /// Human readable name, as used in `executionProvider`.
    pub fn display_name(self) -> &'static str {
        match self {
            Target::Gpu => "GPU",
            Target::Cpu => "CPU",
            Target::Npu => "NPU",
            Target::Cuda => "CUDA",
            Target::TensorRt => "TensorRT",
            Target::DirectMl => "DirectML",
            Target::CoreMl => "CoreML",
        }
    }

    fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "gpu" => Target::Gpu,
            "cpu" => Target::Cpu,
            "npu" => Target::Npu,
            "cuda" => Target::Cuda,
            "tensorrt" => Target::TensorRt,
            "directml" => Target::DirectMl,
            "coreml" => Target::CoreMl,
            _ => return None,
        })
    }

    /// Whether `runtime` offers this target.
    pub fn supported_by(self, runtime: Runtime) -> bool {
        match runtime {
            Runtime::OpenVino => matches!(self, Target::Gpu | Target::Cpu | Target::Npu),
            Runtime::Ort => matches!(
                self,
                Target::Cuda | Target::TensorRt | Target::DirectMl | Target::CoreMl | Target::Cpu
            ),
        }
    }

    /// Whether a device index (`gpu.N` / `cuda:N`) is accepted.
    pub fn takes_index(self) -> bool {
        matches!(
            self,
            Target::Gpu | Target::Cuda | Target::TensorRt | Target::DirectMl
        )
    }
}

/// A concrete runtime + target (+ device index). Construct through [`Device::new`] or
/// [`parse`] so the combination is always valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    pub runtime: Runtime,
    pub target: Target,
    pub index: Option<u32>,
}

impl Device {
    pub fn new(runtime: Runtime, target: Target, index: Option<u32>) -> Result<Self> {
        if !target.supported_by(runtime) {
            bail!(
                "{} has no '{}' device",
                runtime.display_name(),
                target.as_str()
            );
        }
        if index.is_some() && !target.takes_index() {
            bail!(
                "'{}:{}' takes no device index",
                runtime.as_str(),
                target.as_str()
            );
        }
        Ok(Self {
            runtime,
            target,
            index,
        })
    }

    pub const OPENVINO_CPU: Device = Device {
        runtime: Runtime::OpenVino,
        target: Target::Cpu,
        index: None,
    };

    pub const OPENVINO_GPU: Device = Device {
        runtime: Runtime::OpenVino,
        target: Target::Gpu,
        index: None,
    };

    /// OpenVINO device string ("GPU", "GPU.1", "CPU", "NPU"); None for ONNX Runtime.
    pub fn openvino_device(&self) -> Option<String> {
        if self.runtime != Runtime::OpenVino {
            return None;
        }
        let name = self.target.display_name();
        Some(match self.index {
            Some(i) => format!("{name}.{i}"),
            None => name.to_string(),
        })
    }

    pub fn is_cpu(&self) -> bool {
        self.target == Target::Cpu
    }

    /// A GPU-class accelerator: OpenVINO GPU, CUDA, TensorRT, DirectML or CoreML.
    pub fn is_gpu_like(&self) -> bool {
        matches!(
            self.target,
            Target::Gpu | Target::Cuda | Target::TensorRt | Target::DirectMl | Target::CoreMl
        )
    }

    /// This device with `index` filled in when it takes one and has none.
    pub fn with_default_index(self, index: u32) -> Self {
        if self.index.is_none() && self.target.takes_index() {
            Self {
                index: Some(index),
                ..self
            }
        } else {
            self
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.runtime.as_str(), self.target.as_str())?;
        match (self.runtime, self.index) {
            (_, None) => Ok(()),
            (Runtime::OpenVino, Some(i)) => write!(f, ".{i}"),
            (Runtime::Ort, Some(i)) => write!(f, ":{i}"),
        }
    }
}

/// Parsed `device` setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceSpec {
    /// Pick the best runtime and device for the hardware (6.1: OpenVINO GPU, then CPU).
    Auto,
    Device(Device),
}

impl DeviceSpec {
    pub const OPENVINO_CPU: DeviceSpec = DeviceSpec::Device(Device::OPENVINO_CPU);

    pub fn device(&self) -> Option<&Device> {
        match self {
            DeviceSpec::Auto => None,
            DeviceSpec::Device(d) => Some(d),
        }
    }

    pub fn runtime(&self) -> Option<Runtime> {
        self.device().map(|d| d.runtime)
    }

    /// OpenVINO device string ("GPU", "GPU.1", "CPU", "NPU"); None for `auto` and ONNX Runtime.
    pub fn openvino_device(&self) -> Option<String> {
        self.device().and_then(Device::openvino_device)
    }

    pub fn is_cpu(&self) -> bool {
        self.device().is_some_and(Device::is_cpu)
    }

    pub fn is_gpu_like(&self) -> bool {
        self.device().is_some_and(Device::is_gpu_like)
    }

    /// See [`Device::with_default_index`]; `auto` is unchanged.
    pub fn with_default_index(self, index: u32) -> Self {
        match self {
            DeviceSpec::Auto => DeviceSpec::Auto,
            DeviceSpec::Device(d) => DeviceSpec::Device(d.with_default_index(index)),
        }
    }
}

impl fmt::Display for DeviceSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceSpec::Auto => f.write_str("auto"),
            DeviceSpec::Device(d) => d.fmt(f),
        }
    }
}

impl std::str::FromStr for DeviceSpec {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        parse(s)
    }
}

/// Accepted forms, for error messages and help text.
pub const SPEC_SYNTAX: &str = "auto, openvino:gpu[.N], openvino:cpu, openvino:npu, \
    ort:cuda[:N], ort:tensorrt[:N], ort:directml[:N], ort:coreml, ort:cpu \
    (legacy GPU, GPU.N, CPU, NPU mean OpenVINO)";

/// Decimal device index: ASCII digits only (no sign, no whitespace).
fn parse_index(s: &str) -> Option<u32> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// OpenVINO `name` or `name.N` (`gpu.1`); only GPU takes an index.
fn parse_openvino(s: &str) -> Option<Device> {
    let (name, index) = match s.split_once('.') {
        Some((n, i)) => (n, Some(parse_index(i)?)),
        None => (s, None),
    };
    Device::new(Runtime::OpenVino, Target::from_name(name)?, index).ok()
}

/// ONNX Runtime `ep` or `ep:N`.
fn parse_ort(s: &str) -> Option<Device> {
    let (name, index) = match s.split_once(':') {
        Some((n, i)) => (n, Some(parse_index(i)?)),
        None => (s, None),
    };
    Device::new(Runtime::Ort, Target::from_name(name)?, index).ok()
}

/// Parse a device setting (case-insensitive, surrounding whitespace ignored).
pub fn parse(s: &str) -> Result<DeviceSpec> {
    let lower = s.trim().to_ascii_lowercase();
    let parsed = match lower.split_once(':') {
        _ if lower == "auto" => Some(DeviceSpec::Auto),
        Some(("openvino", rest)) => parse_openvino(rest).map(DeviceSpec::Device),
        Some(("ort", rest)) => parse_ort(rest).map(DeviceSpec::Device),
        Some(_) => None,
        // Legacy bare OpenVINO names.
        None => parse_openvino(&lower).map(DeviceSpec::Device),
    };
    match parsed {
        Some(spec) => Ok(spec),
        None => bail!("unknown device '{}': expected {SPEC_SYNTAX}", s.trim()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ov(target: Target, index: Option<u32>) -> DeviceSpec {
        DeviceSpec::Device(Device::new(Runtime::OpenVino, target, index).unwrap())
    }

    fn ort(target: Target, index: Option<u32>) -> DeviceSpec {
        DeviceSpec::Device(Device::new(Runtime::Ort, target, index).unwrap())
    }

    #[test]
    fn parses_canonical_and_legacy_forms() {
        let cases = [
            ("auto", DeviceSpec::Auto, "auto"),
            (" AUTO ", DeviceSpec::Auto, "auto"),
            ("openvino:gpu", ov(Target::Gpu, None), "openvino:gpu"),
            ("OpenVINO:GPU.1", ov(Target::Gpu, Some(1)), "openvino:gpu.1"),
            ("openvino:cpu", ov(Target::Cpu, None), "openvino:cpu"),
            ("openvino:npu", ov(Target::Npu, None), "openvino:npu"),
            ("GPU", ov(Target::Gpu, None), "openvino:gpu"),
            ("gpu.0", ov(Target::Gpu, Some(0)), "openvino:gpu.0"),
            ("GPU.12", ov(Target::Gpu, Some(12)), "openvino:gpu.12"),
            ("\tCPU\n", ov(Target::Cpu, None), "openvino:cpu"),
            ("NPU", ov(Target::Npu, None), "openvino:npu"),
            ("ort:cuda", ort(Target::Cuda, None), "ort:cuda"),
            ("ORT:CUDA:1", ort(Target::Cuda, Some(1)), "ort:cuda:1"),
            ("ort:tensorrt", ort(Target::TensorRt, None), "ort:tensorrt"),
            (
                "ort:tensorrt:2",
                ort(Target::TensorRt, Some(2)),
                "ort:tensorrt:2",
            ),
            ("ort:directml", ort(Target::DirectMl, None), "ort:directml"),
            (
                "ort:DirectML:0",
                ort(Target::DirectMl, Some(0)),
                "ort:directml:0",
            ),
            ("ort:coreml", ort(Target::CoreMl, None), "ort:coreml"),
            ("ort:cpu", ort(Target::Cpu, None), "ort:cpu"),
        ];
        for (input, want, canonical) in cases {
            let got = parse(input).unwrap_or_else(|e| panic!("{input:?}: {e:#}"));
            assert_eq!(got, want, "{input:?}");
            assert_eq!(got.to_string(), canonical, "{input:?}");
            assert_eq!(
                parse(&got.to_string()).unwrap(),
                got,
                "round trip {input:?}"
            );
            assert_eq!(input.parse::<DeviceSpec>().unwrap(), got);
        }
    }

    #[test]
    fn rejects_garbage() {
        for bad in [
            "",
            "   ",
            "G P U",
            "GPU 1",
            "gpu.",
            "gpu.x",
            "gpu.-1",
            "gpu.+1",
            "gpu.1.2",
            "gpu:1",
            "cpu.0",
            "npu.1",
            "xpu",
            "HETERO:GPU,CPU",
            "AUTO:GPU",
            "openvino",
            "openvino:",
            "openvino:cuda",
            "openvino:gpu:1",
            "openvino:gpu.99999999999",
            "ort",
            "ort:",
            "ort:gpu",
            "ort:npu",
            "ort:cuda.1",
            "ort:cuda:",
            "ort:cuda:1:2",
            "ort:coreml:0",
            "ort:cpu:0",
            "onnx:cuda",
            "auto:gpu",
        ] {
            let err = parse(bad)
                .err()
                .unwrap_or_else(|| panic!("{bad:?} must fail"));
            assert!(format!("{err}").contains("openvino:gpu"), "{bad:?}: {err}");
        }
    }

    #[test]
    fn openvino_device_strings() {
        let dev = |s: &str| parse(s).unwrap().openvino_device();
        assert_eq!(dev("GPU").as_deref(), Some("GPU"));
        assert_eq!(dev("openvino:gpu.1").as_deref(), Some("GPU.1"));
        assert_eq!(dev("cpu").as_deref(), Some("CPU"));
        assert_eq!(dev("openvino:npu").as_deref(), Some("NPU"));
        assert_eq!(dev("auto"), None);
        assert_eq!(dev("ort:cpu"), None);
        assert_eq!(dev("ort:cuda:1"), None);
    }

    #[test]
    fn helpers() {
        let p = |s: &str| parse(s).unwrap();
        assert!(p("CPU").is_cpu() && p("ort:cpu").is_cpu());
        assert!(!p("auto").is_cpu() && !p("GPU").is_cpu());
        for s in [
            "GPU",
            "gpu.1",
            "ort:cuda",
            "ort:tensorrt",
            "ort:directml",
            "ort:coreml",
        ] {
            assert!(p(s).is_gpu_like(), "{s}");
        }
        for s in ["auto", "CPU", "NPU", "ort:cpu"] {
            assert!(!p(s).is_gpu_like(), "{s}");
        }
        assert_eq!(p("GPU").runtime(), Some(Runtime::OpenVino));
        assert_eq!(p("ort:cpu").runtime(), Some(Runtime::Ort));
        assert_eq!(p("auto").runtime(), None);
    }

    #[test]
    fn default_index() {
        let with = |s: &str| parse(s).unwrap().with_default_index(2).to_string();
        assert_eq!(with("GPU"), "openvino:gpu.2");
        assert_eq!(with("GPU.1"), "openvino:gpu.1");
        assert_eq!(with("ort:cuda"), "ort:cuda:2");
        assert_eq!(with("ort:tensorrt"), "ort:tensorrt:2");
        assert_eq!(with("ort:directml:0"), "ort:directml:0");
        assert_eq!(with("ort:coreml"), "ort:coreml");
        assert_eq!(with("CPU"), "openvino:cpu");
        assert_eq!(with("NPU"), "openvino:npu");
        assert_eq!(with("auto"), "auto");
    }

    #[test]
    fn device_validation() {
        assert!(Device::new(Runtime::OpenVino, Target::Cuda, None).is_err());
        assert!(Device::new(Runtime::Ort, Target::Gpu, None).is_err());
        assert!(Device::new(Runtime::Ort, Target::Npu, None).is_err());
        assert!(Device::new(Runtime::Ort, Target::CoreMl, Some(0)).is_err());
        assert!(Device::new(Runtime::OpenVino, Target::Cpu, Some(0)).is_err());
        assert_eq!(
            Device::new(Runtime::OpenVino, Target::Cpu, None).unwrap(),
            Device::OPENVINO_CPU
        );
    }
}
