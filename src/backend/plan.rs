//! Load plan: the ordered devices a model is tried on, and the loop that tries them.
//!
//! Pure (no runtime libraries): [`plan_for`] turns a [`DeviceSpec`] plus the machine's
//! [`Selection`] into candidates, and [`try_candidates`] runs a caller-supplied attempt
//! (compile + create request + warm-up in the worker) on each until one succeeds.
//!
//! Semantics:
//! - `auto` -> the `select` ranking ([`auto_candidates`]): the first candidate is not a
//!   fallback, every later one (down to CPU) is. A machine whose ranking is CPU only loads CPU as
//!   a regular pick, with a note when a GPU was found but nothing GPU-class can run.
//! - `openvino:gpu[.N]` -> GPU, then CPU as a fallback. When OpenVINO lists no GPU at all, only
//!   CPU is tried and it counts as a fallback (unchanged from the old `DeviceSelection`).
//! - `openvino:cpu` / `openvino:npu` -> just that device (plus ONNX Runtime CPU as a fallback
//!   when OpenVINO cannot run it, e.g. the OpenVINO runtime is missing).
//! - `ort:*` -> that device, then the best other CPU option ([`best_cpu`]: OpenVINO CPU, else
//!   ONNX Runtime CPU) as a fallback.
//!
//! "No GPU" fallbacks also use [`best_cpu`], so a legacy `GPU` config on a machine with only
//! ONNX Runtime lands on `ort:cpu`.

use super::detect::HardwareInfo;
use super::select::{RuntimeProbe, Selection, select};
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

/// Ordered candidates for `spec` on a machine described by `sel` (from [`select`]).
pub fn plan_for(
    spec: &DeviceSpec,
    sel: &Selection,
    available_ov_devices: &[String],
) -> Vec<Candidate> {
    match spec {
        DeviceSpec::Auto => auto_candidates(sel),
        DeviceSpec::Device(d) => explicit_candidates(*d, sel, available_ov_devices),
    }
}

/// Ordered candidates for `spec` knowing only the device names OpenVINO lists
/// (`["CPU", "GPU.0", ..]`): `auto` is ranked as if no other runtime or GPU were present, which
/// for OpenVINO with an Intel GPU is `openvino:gpu`, then `openvino:cpu`.
pub fn plan_candidates(spec: &DeviceSpec, available_ov_devices: &[String]) -> Vec<Candidate> {
    let sel = select(
        &HardwareInfo::default(),
        &RuntimeProbe::openvino_only(available_ov_devices),
        true,
    );
    plan_for(spec, &sel, available_ov_devices)
}

fn ort_cpu() -> Device {
    Device {
        runtime: Runtime::Ort,
        target: Target::Cpu,
        index: None,
    }
}

fn runnable(sel: &Selection, d: &Device) -> bool {
    sel.option(d).is_some_and(|o| o.runnable)
}

/// The best CPU option (`force_cpu`, "no GPU" fallbacks): OpenVINO CPU when it can run, else
/// ONNX Runtime CPU when it can, else OpenVINO CPU (so the error names the missing runtime).
pub fn best_cpu(sel: &Selection) -> Device {
    [Device::OPENVINO_CPU, ort_cpu()]
        .into_iter()
        .find(|d| runnable(sel, d))
        .unwrap_or(Device::OPENVINO_CPU)
}

/// CPU fallback for an explicit `device`: the first runnable CPU option other than `device`,
/// else OpenVINO CPU unless that is `device` itself.
fn fallback_cpu(sel: &Selection, device: Device) -> Option<Device> {
    [Device::OPENVINO_CPU, ort_cpu()]
        .into_iter()
        .filter(|d| *d != device)
        .find(|d| runnable(sel, d))
        .or_else(|| (device != Device::OPENVINO_CPU).then_some(Device::OPENVINO_CPU))
}

/// The `auto` ranking as candidates. Without anything runnable, OpenVINO CPU is still tried so
/// the error names the missing runtime.
pub fn auto_candidates(sel: &Selection) -> Vec<Candidate> {
    if sel.auto.is_empty() {
        return vec![Candidate {
            note: Some("no runnable device found; trying OpenVINO CPU".to_string()),
            ..Candidate::new(Device::OPENVINO_CPU, false)
        }];
    }
    let mut out: Vec<Candidate> = sel
        .auto
        .iter()
        .enumerate()
        .map(|(i, d)| Candidate::new(*d, i > 0))
        .collect();
    if out.len() == 1 && out[0].device.is_cpu() {
        let blocked: Vec<String> = sel
            .options
            .iter()
            .filter(|o| !o.runnable && o.spec.is_gpu_like())
            .map(|o| {
                format!(
                    "{}: {}",
                    o.spec,
                    o.reason.as_deref().unwrap_or("not runnable")
                )
            })
            .collect();
        if !blocked.is_empty() {
            out[0].note = Some(format!(
                "no GPU option can run ({}); using {}",
                blocked.join("; "),
                out[0].device
            ));
        }
    }
    out
}

/// An explicit device, then its fallback.
fn explicit_candidates(
    device: Device,
    sel: &Selection,
    available_ov_devices: &[String],
) -> Vec<Candidate> {
    let fallback = |d: Device| Candidate::new(d, true);
    match (device.runtime, device.target) {
        (Runtime::OpenVino, Target::Gpu) => {
            let cpu = best_cpu(sel);
            if available_ov_devices.iter().any(|d| d.starts_with("GPU")) {
                vec![Candidate::new(device, false), fallback(cpu)]
            } else {
                let requested = device.openvino_device().unwrap_or_default();
                let using = if cpu == Device::OPENVINO_CPU {
                    "CPU".to_string()
                } else {
                    cpu.to_string()
                };
                vec![Candidate {
                    note: Some(format!(
                        "requested device {requested} not available (devices: {available_ov_devices:?}); using {using}"
                    )),
                    ..fallback(cpu)
                }]
            }
        }
        (Runtime::OpenVino, _) => {
            let mut out = vec![Candidate::new(device, false)];
            if !runnable(sel, &device)
                && let Some(cpu) = fallback_cpu(sel, device).filter(|d| runnable(sel, d))
            {
                out.push(fallback(cpu));
            }
            out
        }
        (Runtime::Ort, _) => {
            let mut out = vec![Candidate::new(device, false)];
            out.extend(fallback_cpu(sel, device).map(fallback));
            out
        }
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
        // CPU-only machine: CPU is auto's regular pick, not a fallback, and there is no GPU to
        // complain about.
        assert_eq!(plan("auto", &["CPU"]), s(&[("openvino:cpu", false)]));
        let c = plan_candidates(&parse("auto").unwrap(), &devs(&["CPU"]));
        assert!(c[0].note.is_none());
        // Without OpenVINO devices nothing is runnable; CPU is still tried, with a note.
        let c = plan_candidates(&parse("auto").unwrap(), &[]);
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].device, c[0].fell_back), (Device::OPENVINO_CPU, false));
        assert!(c[0].note.is_some());
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
    fn auto_follows_the_selection() {
        use crate::backend::detect::{GpuAdapter, GpuVendor};
        use crate::backend::select::{EpStatus, OrtProbe};
        let nvidia = GpuAdapter {
            vendor: GpuVendor::Nvidia,
            name: "NVIDIA GeForce RTX 3060".into(),
            vram_mb: 12288,
            index: 0,
            discrete: true,
            cuda: None,
        };
        let hw = HardwareInfo::new("linux", "x86_64", vec![nvidia]);
        let ov = devs(&["CPU", "GPU"]);
        let mut probe = RuntimeProbe::openvino_only(&ov);
        probe.ort = OrtProbe::installed(
            "gpu",
            vec![
                EpStatus::usable(Target::Cuda),
                EpStatus::usable(Target::Cpu),
            ],
        );
        let sel = select(&hw, &probe, true);
        let got: Vec<(String, bool)> = plan_for(&DeviceSpec::Auto, &sel, &ov)
            .into_iter()
            .map(|c| (c.device.to_string(), c.fell_back))
            .collect();
        assert_eq!(
            got,
            s(&[
                ("ort:cuda:0", false),
                ("openvino:gpu", true),
                ("openvino:cpu", true)
            ])
        );
        // Explicit specs ignore the ranking.
        assert_eq!(
            plan_for(&parse("GPU").unwrap(), &sel, &ov),
            plan_candidates(&parse("GPU").unwrap(), &ov)
        );

        // NVIDIA found but CUDA unusable and no OpenVINO GPU: CPU with a note naming CUDA.
        probe.ort =
            OrtProbe::installed("gpu", vec![EpStatus::broken(Target::Cuda, "cudnn missing")]);
        let cpu_only = devs(&["CPU"]);
        probe.openvino = crate::backend::select::OpenVinoProbe::with_device_names(&cpu_only);
        let c = auto_candidates(&select(&hw, &probe, true));
        assert_eq!(c.len(), 1);
        assert!(!c[0].fell_back);
        let note = c[0].note.as_deref().unwrap();
        assert!(note.contains("ort:cuda:0: cudnn missing"), "{note}");
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

    #[test]
    fn ort_aware_cpu_fallbacks() {
        use crate::backend::detect::{GpuAdapter, GpuVendor};
        use crate::backend::select::{EpStatus, OpenVinoProbe, OrtProbe};
        let apple = GpuAdapter {
            vendor: GpuVendor::Apple,
            name: "Apple M1 GPU".into(),
            vram_mb: 0,
            index: 0,
            discrete: false,
            cuda: None,
        };
        let hw = HardwareInfo::new("macos", "aarch64", vec![apple]);
        let ort = OrtProbe::installed(
            "coreml",
            vec![
                EpStatus::usable(Target::Cpu),
                EpStatus::usable(Target::CoreMl),
            ],
        );
        let names = |sel: &Selection, spec: &str, ov: &[&str]| -> Vec<(String, bool)> {
            plan_for(&parse(spec).unwrap(), sel, &devs(ov))
                .into_iter()
                .map(|c| (c.device.to_string(), c.fell_back))
                .collect()
        };

        // OpenVINO and ONNX Runtime both present.
        let both = select(
            &hw,
            &RuntimeProbe {
                openvino: OpenVinoProbe::with_device_names(&devs(&["CPU"])),
                ort: ort.clone(),
            },
            true,
        );
        assert_eq!(best_cpu(&both), Device::OPENVINO_CPU);
        assert_eq!(
            names(&both, "auto", &["CPU"]),
            s(&[("ort:coreml", false), ("openvino:cpu", true)])
        );
        assert_eq!(
            names(&both, "ort:coreml", &["CPU"]),
            s(&[("ort:coreml", false), ("openvino:cpu", true)])
        );
        assert_eq!(
            names(&both, "ort:cpu", &["CPU"]),
            s(&[("ort:cpu", false), ("openvino:cpu", true)])
        );
        assert_eq!(names(&both, "CPU", &["CPU"]), s(&[("openvino:cpu", false)]));

        // ONNX Runtime only: CPU options and "no GPU" fallbacks land on ort:cpu.
        let ort_only = select(
            &hw,
            &RuntimeProbe {
                openvino: OpenVinoProbe::unavailable("missing"),
                ort,
            },
            true,
        );
        assert_eq!(
            best_cpu(&ort_only),
            parse("ort:cpu").unwrap().device().copied().unwrap()
        );
        assert_eq!(
            names(&ort_only, "auto", &[]),
            s(&[("ort:coreml", false), ("ort:cpu", true)])
        );
        let legacy_gpu = plan_for(&parse("GPU").unwrap(), &ort_only, &[]);
        assert_eq!(legacy_gpu.len(), 1);
        assert_eq!(legacy_gpu[0].device.to_string(), "ort:cpu");
        assert!(legacy_gpu[0].fell_back);
        assert!(
            legacy_gpu[0]
                .note
                .as_deref()
                .unwrap()
                .contains("using ort:cpu")
        );
        assert_eq!(
            names(&ort_only, "CPU", &[]),
            s(&[("openvino:cpu", false), ("ort:cpu", true)])
        );
        assert_eq!(
            names(&ort_only, "ort:cuda", &[]),
            s(&[("ort:cuda", false), ("ort:cpu", true)])
        );
        // ort:cpu itself: OpenVINO CPU is still tried so the error names it.
        assert_eq!(
            names(&ort_only, "ort:cpu", &[]),
            s(&[("ort:cpu", false), ("openvino:cpu", true)])
        );
        assert!(ort_only.any_gpu_runnable());
    }
}
