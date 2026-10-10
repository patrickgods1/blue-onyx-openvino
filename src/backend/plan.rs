//! Load plan: the ordered devices a model is tried on, and the loop that tries them.
//!
//! Pure (no runtime libraries): [`plan_candidates`] turns a [`DeviceSpec`] plus the devices
//! OpenVINO lists into candidates, and [`try_candidates`] runs a caller-supplied attempt
//! (compile + create request + warm-up in the worker) on each until one succeeds.
//!
//! Phase 6.1 semantics (unchanged from the old `DeviceSelection`):
//! - `openvino:gpu[.N]` -> GPU, then CPU as a fallback. When OpenVINO lists no GPU at all, only
//!   CPU is tried and it counts as a fallback.
//! - `openvino:cpu` / `openvino:npu` -> just that device.
//! - `auto` -> same as `openvino:gpu` (phase 6.2 replaces this with the hardware ranking).
//! - `ort:*` -> that device, then OpenVINO CPU as a fallback.

use super::spec::{Device, DeviceSpec, Runtime, Target};
use anyhow::Result;
use tracing::warn;

/// One device to try.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub device: Device,
    /// Reported as `DeviceInfo::fell_back` (", fallback" in `executionProvider`) when this
    /// candidate is the one that loads.
    pub fell_back: bool,
    /// Why the plan already deviates from the request before anything was tried (logged).
    pub note: Option<String>,
}

impl Candidate {
    fn new(device: Device, fell_back: bool) -> Self {
        Self {
            device,
            fell_back,
            note: None,
        }
    }
}

/// Ordered candidates for `spec`, given the device names OpenVINO lists (`["CPU", "GPU.0", ..]`).
pub fn plan_candidates(spec: &DeviceSpec, available_ov_devices: &[String]) -> Vec<Candidate> {
    let device = match spec {
        DeviceSpec::Auto => Device::OPENVINO_GPU,
        DeviceSpec::Device(d) => *d,
    };
    let cpu_fallback = Candidate::new(Device::OPENVINO_CPU, true);
    match (device.runtime, device.target) {
        (Runtime::OpenVino, Target::Gpu) => {
            if available_ov_devices.iter().any(|d| d.starts_with("GPU")) {
                vec![Candidate::new(device, false), cpu_fallback]
            } else {
                let requested = device.openvino_device().unwrap_or_default();
                vec![Candidate {
                    note: Some(format!(
                        "requested device {requested} not available (devices: {available_ov_devices:?}); using CPU"
                    )),
                    ..cpu_fallback
                }]
            }
        }
        (Runtime::OpenVino, _) => vec![Candidate::new(device, false)],
        (Runtime::Ort, _) => vec![Candidate::new(device, false), cpu_fallback],
    }
}

/// Run `attempt` on each candidate in order and return the first success. Failures are logged
/// (`warn!`, prefixed with `label`) and, when every candidate fails, listed in the error.
pub fn try_candidates<T>(
    label: &str,
    candidates: &[Candidate],
    mut attempt: impl FnMut(&Candidate) -> Result<T>,
) -> Result<T> {
    let mut failures: Vec<(String, anyhow::Error)> = Vec::new();
    for (i, cand) in candidates.iter().enumerate() {
        if let Some(note) = &cand.note {
            warn!("{label}: {note}");
        }
        match attempt(cand) {
            Ok(v) => return Ok(v),
            Err(e) => {
                match candidates.get(i + 1) {
                    Some(next) => warn!(
                        "{label}: {} failed: {e:#}; trying {}",
                        cand.device, next.device
                    ),
                    None if candidates.len() > 1 => warn!("{label}: {} failed: {e:#}", cand.device),
                    None => {}
                }
                failures.push((cand.device.to_string(), e));
            }
        }
    }
    match failures.len() {
        0 => anyhow::bail!("{label}: no device to try"),
        // A single candidate keeps its own error chain.
        1 => Err(failures.pop().map(|(_, e)| e).expect("one failure")),
        _ => {
            let list: Vec<String> = failures
                .iter()
                .map(|(dev, e)| format!("{dev}: {e:#}"))
                .collect();
            anyhow::bail!("every device failed ({})", list.join("; "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::spec::parse;

    fn devs(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// (spec, fell_back) pairs of a plan.
    fn plan(spec: &str, available: &[&str]) -> Vec<(String, bool)> {
        plan_candidates(&parse(spec).unwrap(), &devs(available))
            .into_iter()
            .map(|c| (c.device.to_string(), c.fell_back))
            .collect()
    }

    fn s(v: &[(&str, bool)]) -> Vec<(String, bool)> {
        v.iter().map(|(d, f)| (d.to_string(), *f)).collect()
    }

    #[test]
    fn gpu_with_and_without_gpu_present() {
        let with_gpu = ["CPU", "GPU"];
        assert_eq!(
            plan("GPU", &with_gpu),
            s(&[("openvino:gpu", false), ("openvino:cpu", true)])
        );
        assert_eq!(
            plan("gpu.1", &["CPU", "GPU.0", "GPU.1"]),
            s(&[("openvino:gpu.1", false), ("openvino:cpu", true)])
        );
        // No GPU listed: only CPU, and it counts as a fallback (old DeviceSelection behavior).
        assert_eq!(plan("GPU", &["CPU"]), s(&[("openvino:cpu", true)]));
        let c = plan_candidates(&parse("GPU").unwrap(), &devs(&["CPU"]));
        assert!(c[0].note.as_deref().unwrap().contains("GPU not available"));
        assert_eq!(plan("GPU.1", &[]), s(&[("openvino:cpu", true)]));
    }

    #[test]
    fn cpu_npu_auto_and_ort() {
        assert_eq!(plan("CPU", &["CPU", "GPU"]), s(&[("openvino:cpu", false)]));
        assert_eq!(
            plan("openvino:npu", &["CPU", "NPU"]),
            s(&[("openvino:npu", false)])
        );
        assert_eq!(
            plan("auto", &["CPU", "GPU"]),
            plan("openvino:gpu", &["CPU", "GPU"])
        );
        assert_eq!(plan("auto", &["CPU"]), s(&[("openvino:cpu", true)]));
        assert_eq!(
            plan("ort:cuda:1", &["CPU"]),
            s(&[("ort:cuda:1", false), ("openvino:cpu", true)])
        );
        assert_eq!(
            plan("ort:cpu", &["CPU"]),
            s(&[("ort:cpu", false), ("openvino:cpu", true)])
        );
        for c in plan_candidates(&parse("ort:coreml").unwrap(), &devs(&["CPU"])) {
            assert!(c.note.is_none());
        }
    }

    #[test]
    fn tries_in_order_until_success() {
        let cands = plan_candidates(&parse("GPU").unwrap(), &devs(&["CPU", "GPU"]));
        let mut tried = Vec::new();
        let got = try_candidates("m", &cands, |c| {
            tried.push(c.device.to_string());
            if c.device.is_cpu() {
                Ok(c.fell_back)
            } else {
                anyhow::bail!("compile failed")
            }
        })
        .unwrap();
        assert!(got, "CPU after a GPU failure is a fallback");
        assert_eq!(tried, ["openvino:gpu", "openvino:cpu"]);

        let mut n = 0;
        let first = try_candidates("m", &cands, |c| {
            n += 1;
            Ok(c.device)
        })
        .unwrap();
        assert_eq!((first, n), (Device::OPENVINO_GPU, 1));
    }

    #[test]
    fn error_lists_every_failure() {
        let cands = plan_candidates(&parse("ort:cuda").unwrap(), &devs(&["CPU"]));
        let err = try_candidates::<()>("m", &cands, |c| anyhow::bail!("boom on {}", c.device))
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("ort:cuda: boom on ort:cuda"), "{msg}");
        assert!(msg.contains("openvino:cpu: boom on openvino:cpu"), "{msg}");

        // One candidate: its error is returned as is.
        let cands = plan_candidates(&parse("CPU").unwrap(), &devs(&["CPU"]));
        let err = try_candidates::<()>("m", &cands, |_| {
            Err(anyhow::anyhow!("inner").context("outer"))
        })
        .unwrap_err();
        assert_eq!(format!("{err:#}"), "outer: inner");

        assert!(try_candidates::<()>("m", &[], |_| Ok(())).is_err());
    }
}
