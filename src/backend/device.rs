//! Device selection, core properties, model loading with GPU->CPU fallback, tensor IO.
//! TODO(backend agent): implement every `bail!("not implemented")` function against the
//! `openvino` 0.11 crate. Keep the public signatures.

use super::{CoreOptions, LoadRequest, LoadedModel, OvBackend};
use crate::model::{ExtraInput, NamedOutput};
use anyhow::Result;

/// Which device a model was requested on and ended up on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceInfo {
    pub requested: String,
    /// "GPU", "GPU.1", "CPU"
    pub actual: String,
    /// FULL_DEVICE_NAME, e.g. "Intel(R) UHD Graphics 630 (iGPU)"
    pub full_name: String,
    pub fell_back: bool,
}

impl DeviceInfo {
    pub fn is_gpu(&self) -> bool {
        self.actual.starts_with("GPU")
    }
    /// String reported as `executionProvider` in API responses.
    pub fn execution_provider(&self) -> String {
        let kind = if self.is_gpu() { "GPU" } else { "CPU" };
        let fb = if self.fell_back { ", fallback" } else { "" };
        if self.full_name.is_empty() {
            format!("OpenVINO {kind}{fb}")
        } else {
            format!("OpenVINO {kind} ({}{fb})", self.full_name)
        }
    }
}

/// Parsed device preference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceSelection {
    pub primary: String,
    pub fallback: Option<String>,
}

impl DeviceSelection {
    pub fn from_request(req: &LoadRequest, available: &[String]) -> Self {
        let wants_gpu = req.device.to_ascii_uppercase().starts_with("GPU");
        let gpu_present = available.iter().any(|d| d.starts_with("GPU"));
        if wants_gpu && !gpu_present {
            return Self {
                primary: "CPU".into(),
                fallback: None,
            };
        }
        Self {
            primary: req.device.clone(),
            fallback: if wants_gpu && req.allow_cpu_fallback {
                Some("CPU".into())
            } else {
                None
            },
        }
    }
}

pub fn apply_core_properties(
    _core: &mut openvino::Core,
    _opts: &CoreOptions,
    _available: &[String],
) -> Result<()> {
    anyhow::bail!("backend::device::apply_core_properties not implemented")
}

pub fn full_device_name(_core: &openvino::Core, device: &str) -> String {
    device.to_string()
}

pub fn load_model(
    _core: &mut openvino::Core,
    _available: &[String],
    _req: &LoadRequest,
) -> Result<LoadedModel> {
    anyhow::bail!("backend::device::load_model not implemented")
}

pub fn create_backend(_model: LoadedModel) -> Result<OvBackend> {
    anyhow::bail!("backend::device::create_backend not implemented")
}

pub fn run_inference(
    _model: &LoadedModel,
    _request: &mut openvino::InferRequest,
    _input_tensor: &mut openvino::Tensor,
    _chw: &[f32],
    _extra: &[ExtraInput],
) -> Result<Vec<NamedOutput>> {
    anyhow::bail!("backend::device::run_inference not implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_rules() {
        let req = LoadRequest {
            path: "m.xml".into(),
            device: "GPU".into(),
            gpu_precision: None,
            allow_cpu_fallback: true,
        };
        let s = DeviceSelection::from_request(&req, &["CPU".into(), "GPU".into()]);
        assert_eq!(s.primary, "GPU");
        assert_eq!(s.fallback.as_deref(), Some("CPU"));
        let s = DeviceSelection::from_request(&req, &["CPU".into()]);
        assert_eq!(s.primary, "CPU");
        assert!(s.fallback.is_none());
        let info = DeviceInfo {
            requested: "GPU".into(),
            actual: "GPU".into(),
            full_name: "Intel(R) UHD Graphics 630".into(),
            fell_back: false,
        };
        assert_eq!(info.execution_provider(), "OpenVINO GPU (Intel(R) UHD Graphics 630)");
    }
}
