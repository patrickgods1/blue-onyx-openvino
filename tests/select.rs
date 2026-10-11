//! Device options and the `auto` ranking on made-up hardware (no runtime libraries needed).

use blue_onyx_prism::backend::detect::{GpuAdapter, GpuVendor, HardwareInfo};
use blue_onyx_prism::backend::select::{
    EpStatus, NEEDS_ONNX, OpenVinoProbe, OrtProbe, OvDevice, RuntimeProbe, Selection, select,
};
use blue_onyx_prism::backend::spec::Target;
use blue_onyx_prism::backend::{ORT_UNAVAILABLE, auto_candidates};

fn gpu(vendor: GpuVendor, name: &str, vram_mb: u64, index: u32, discrete: bool) -> GpuAdapter {
    GpuAdapter {
        vendor,
        name: name.into(),
        vram_mb,
        index,
        discrete,
        cuda: None,
    }
}

fn uhd630(index: u32) -> GpuAdapter {
    gpu(
        GpuVendor::Intel,
        "Intel(R) UHD Graphics 630",
        128,
        index,
        false,
    )
}

fn rtx3060(index: u32) -> GpuAdapter {
    gpu(
        GpuVendor::Nvidia,
        "NVIDIA GeForce RTX 3060",
        12288,
        index,
        true,
    )
}

fn ov_dev(name: &str, full_name: &str, device_type: Option<&str>) -> OvDevice {
    OvDevice {
        name: name.into(),
        full_name: full_name.into(),
        device_type: device_type.map(str::to_string),
    }
}

fn ov_cpu() -> OvDevice {
    ov_dev("CPU", "Intel(R) Core(TM) i5-8500 CPU @ 3.00GHz", None)
}

fn openvino(devices: Vec<OvDevice>) -> OpenVinoProbe {
    OpenVinoProbe {
        error: None,
        devices,
    }
}

/// OpenVINO with CPU and the UHD 630 as the only GPU.
fn openvino_igpu() -> OpenVinoProbe {
    openvino(vec![
        ov_cpu(),
        ov_dev(
            "GPU",
            "Intel(R) UHD Graphics 630 (iGPU)",
            Some("integrated"),
        ),
    ])
}

fn ort_gpu_flavor(cuda: Option<&str>) -> OrtProbe {
    let cuda = match cuda {
        None => EpStatus::usable(Target::Cuda),
        Some(reason) => EpStatus::broken(Target::Cuda, reason),
    };
    OrtProbe::installed(
        "gpu",
        vec![
            cuda,
            EpStatus::usable(Target::TensorRt),
            EpStatus::usable(Target::Cpu),
        ],
    )
}

fn auto(sel: &Selection) -> Vec<String> {
    sel.auto.iter().map(|d| d.to_string()).collect()
}

fn specs(sel: &Selection) -> Vec<String> {
    sel.options.iter().map(|o| o.spec.to_string()).collect()
}

fn reason<'a>(sel: &'a Selection, spec: &str) -> Option<&'a str> {
    let o = sel
        .options
        .iter()
        .find(|o| o.spec.to_string() == spec)
        .unwrap_or_else(|| panic!("no option {spec} in {:?}", specs(sel)));
    assert_eq!(o.runnable, o.reason.is_none(), "{spec}");
    o.reason.as_deref()
}

fn runnable(sel: &Selection, spec: &str) -> bool {
    reason(sel, spec).is_none()
}

#[test]
fn intel_igpu_only_is_backward_compatible() {
    // The primary target: Windows, i5-8500 + UHD 630, OpenVINO only.
    let hw = HardwareInfo::new("windows", "x86_64", vec![uhd630(0)]);
    let probe = RuntimeProbe {
        openvino: openvino_igpu(),
        ort: OrtProbe::not_in_build(),
    };
    let sel = select(&hw, &probe, true);
    assert_eq!(auto(&sel), ["openvino:gpu", "openvino:cpu"]);
    assert_eq!(
        specs(&sel),
        ["openvino:gpu", "ort:directml:0", "openvino:cpu", "ort:cpu"]
    );
    assert_eq!(reason(&sel, "ort:directml:0"), Some(ORT_UNAVAILABLE));
    assert_eq!(reason(&sel, "ort:cpu"), Some(ORT_UNAVAILABLE));
    let pick = sel.auto_pick().unwrap();
    assert_eq!(
        pick.label,
        "OpenVINO GPU (Intel(R) UHD Graphics 630 (iGPU))"
    );
    assert!(sel.any_gpu_runnable());

    // Same candidates and fallback flags as the old GPU -> CPU plan.
    let cands = auto_candidates(&sel);
    let got: Vec<(String, bool)> = cands
        .iter()
        .map(|c| (c.device.to_string(), c.fell_back))
        .collect();
    assert_eq!(
        got,
        [
            ("openvino:gpu".to_string(), false),
            ("openvino:cpu".to_string(), true)
        ]
    );
    assert!(cands.iter().all(|c| c.note.is_none()));

    // Linux, same box: no DirectML.
    let hw = HardwareInfo::new("linux", "x86_64", vec![uhd630(0)]);
    let sel = select(&hw, &probe, true);
    assert_eq!(specs(&sel), ["openvino:gpu", "openvino:cpu", "ort:cpu"]);
    assert_eq!(auto(&sel), ["openvino:gpu", "openvino:cpu"]);
}

#[test]
fn intel_gpu_not_listed_by_openvino() {
    let hw = HardwareInfo::new("linux", "x86_64", vec![uhd630(0)]);
    let probe = RuntimeProbe {
        openvino: openvino(vec![ov_cpu()]),
        ort: OrtProbe::not_in_build(),
    };
    let sel = select(&hw, &probe, true);
    assert!(
        reason(&sel, "openvino:gpu")
            .unwrap()
            .contains("lists no GPU")
    );
    assert_eq!(auto(&sel), ["openvino:cpu"]);
    let c = auto_candidates(&sel);
    assert!(!c[0].fell_back);
    assert!(c[0].note.as_deref().unwrap().contains("openvino:gpu"));
}

#[test]
fn nvidia_and_intel_igpu_with_cuda() {
    // DXGI order: the iGPU drives the display (0), the RTX is adapter 1 but CUDA device 0.
    let hw = HardwareInfo::new("windows", "x86_64", vec![uhd630(0), rtx3060(1)]);
    let probe = RuntimeProbe {
        openvino: openvino_igpu(),
        ort: ort_gpu_flavor(None),
    };
    let sel = select(&hw, &probe, true);
    assert_eq!(auto(&sel), ["ort:cuda:0", "openvino:gpu", "openvino:cpu"]);
    assert_eq!(
        sel.auto_pick().unwrap().label,
        "ONNX Runtime CUDA (NVIDIA GeForce RTX 3060, 12288 MB)"
    );
    // TensorRT runs but is never picked by auto.
    assert!(runnable(&sel, "ort:tensorrt:0"));
    // The gpu flavor has no DirectML.
    assert!(
        reason(&sel, "ort:directml:1")
            .unwrap()
            .contains("has no DirectML")
    );
    assert!(runnable(&sel, "ort:cpu"));
    assert_eq!(
        specs(&sel),
        [
            "ort:cuda:0",
            "ort:tensorrt:0",
            "openvino:gpu",
            "ort:directml:1",
            "ort:directml:0",
            "openvino:cpu",
            "ort:cpu"
        ]
    );

    let fell_back: Vec<bool> = auto_candidates(&sel).iter().map(|c| c.fell_back).collect();
    assert_eq!(fell_back, [false, true, true]);
}

#[test]
fn nvidia_and_intel_igpu_without_usable_cuda() {
    let hw = HardwareInfo::new("windows", "x86_64", vec![uhd630(0), rtx3060(1)]);
    let probe = RuntimeProbe {
        openvino: openvino_igpu(),
        ort: ort_gpu_flavor(Some("NVIDIA GPU found but CUDA 12/cuDNN 9 not loadable")),
    };
    let sel = select(&hw, &probe, true);
    assert_eq!(auto(&sel), ["openvino:gpu", "openvino:cpu"]);
    assert_eq!(
        reason(&sel, "ort:cuda:0"),
        Some("NVIDIA GPU found but CUDA 12/cuDNN 9 not loadable")
    );

    // ONNX Runtime not installed at all: same pick, ORT options carry that reason.
    let probe = RuntimeProbe {
        openvino: openvino_igpu(),
        ort: OrtProbe::unavailable("run `setup-onnxruntime`"),
    };
    let sel = select(&hw, &probe, true);
    assert_eq!(auto(&sel), ["openvino:gpu", "openvino:cpu"]);
    assert_eq!(reason(&sel, "ort:cuda:0"), Some("run `setup-onnxruntime`"));
    assert_eq!(
        reason(&sel, "ort:tensorrt:0"),
        Some("run `setup-onnxruntime`")
    );

    // DirectML flavor on the same box: the iGPU stays on OpenVINO, the RTX (not covered by CUDA)
    // is next through DirectML.
    let probe = RuntimeProbe {
        openvino: openvino_igpu(),
        ort: OrtProbe::installed(
            "directml",
            vec![
                EpStatus::usable(Target::DirectMl),
                EpStatus::usable(Target::Cpu),
            ],
        ),
    };
    let sel = select(&hw, &probe, true);
    assert_eq!(
        auto(&sel),
        ["openvino:gpu", "ort:directml:1", "openvino:cpu"]
    );
}

#[test]
fn largest_nvidia_gpu_wins() {
    let hw = HardwareInfo::new(
        "linux",
        "x86_64",
        vec![
            gpu(GpuVendor::Nvidia, "NVIDIA T400", 2048, 0, true),
            gpu(GpuVendor::Nvidia, "NVIDIA RTX A4000", 16376, 1, true),
            gpu(GpuVendor::Nvidia, "NVIDIA RTX A4000", 16376, 2, true),
        ],
    );
    let probe = RuntimeProbe {
        openvino: openvino(vec![ov_cpu()]),
        ort: ort_gpu_flavor(None),
    };
    let sel = select(&hw, &probe, true);
    // Ties keep the first.
    assert_eq!(auto(&sel), ["ort:cuda:1", "openvino:cpu"]);
}

#[test]
fn amd_on_windows_uses_directml() {
    let rx6600 = gpu(GpuVendor::Amd, "AMD Radeon RX 6600", 8176, 0, true);
    let hw = HardwareInfo::new("windows", "x86_64", vec![rx6600.clone()]);
    let probe = RuntimeProbe {
        openvino: openvino(vec![ov_cpu()]),
        ort: OrtProbe::installed(
            "directml",
            vec![
                EpStatus::usable(Target::DirectMl),
                EpStatus::usable(Target::Cpu),
            ],
        ),
    };
    let sel = select(&hw, &probe, true);
    assert_eq!(auto(&sel), ["ort:directml:0", "openvino:cpu"]);
    assert_eq!(
        sel.auto_pick().unwrap().label,
        "ONNX Runtime DirectML (AMD Radeon RX 6600, 8176 MB)"
    );
    assert!(!specs(&sel).iter().any(|s| s.starts_with("openvino:gpu")));

    // AMD dGPU + APU: discrete first.
    let apu = gpu(GpuVendor::Amd, "AMD Radeon(TM) Graphics", 512, 0, false);
    let dgpu = GpuAdapter { index: 1, ..rx6600 };
    let hw = HardwareInfo::new("windows", "x86_64", vec![apu, dgpu.clone()]);
    let sel = select(&hw, &probe, true);
    assert_eq!(
        auto(&sel),
        ["ort:directml:1", "ort:directml:0", "openvino:cpu"]
    );

    // Without ONNX Runtime: CPU, with a note about DirectML.
    let probe = RuntimeProbe {
        openvino: openvino(vec![ov_cpu()]),
        ort: OrtProbe::not_in_build(),
    };
    let hw = HardwareInfo::new("windows", "x86_64", vec![dgpu.clone()]);
    let sel = select(&hw, &probe, true);
    assert_eq!(auto(&sel), ["openvino:cpu"]);
    let note = auto_candidates(&sel)[0].note.clone().unwrap();
    assert!(note.contains("ort:directml:1"), "{note}");

    // Linux AMD: no DirectML, falls to CPU.
    let hw = HardwareInfo::new("linux", "x86_64", vec![dgpu]);
    let sel = select(&hw, &probe, true);
    assert_eq!(specs(&sel), ["openvino:cpu", "ort:cpu"]);
    assert_eq!(auto(&sel), ["openvino:cpu"]);
}

#[test]
fn mac_arm64() {
    let m2 = gpu(GpuVendor::Apple, "Apple M2 GPU", 0, 0, false);
    let hw = HardwareInfo::new("macos", "aarch64", vec![m2]);
    let ov = openvino(vec![ov_dev_cpu_mac()]);
    let coreml = OrtProbe::installed(
        "coreml",
        vec![
            EpStatus::usable(Target::CoreMl),
            EpStatus::usable(Target::Cpu),
        ],
    );

    let sel = select(
        &hw,
        &RuntimeProbe {
            openvino: ov.clone(),
            ort: coreml.clone(),
        },
        true,
    );
    assert_eq!(auto(&sel), ["ort:coreml", "openvino:cpu"]);
    assert_eq!(specs(&sel), ["ort:coreml", "openvino:cpu", "ort:cpu"]);

    // OpenVINO only (this build): CPU.
    let sel = select(
        &hw,
        &RuntimeProbe {
            openvino: ov.clone(),
            ort: OrtProbe::not_in_build(),
        },
        true,
    );
    assert_eq!(auto(&sel), ["openvino:cpu"]);
    assert_eq!(sel.auto_pick().unwrap().label, "OpenVINO CPU (Apple M2)");

    // ONNX Runtime only: CoreML then ORT CPU.
    let sel = select(
        &hw,
        &RuntimeProbe {
            openvino: OpenVinoProbe::unavailable("run `setup-openvino`"),
            ort: coreml.clone(),
        },
        true,
    );
    assert_eq!(auto(&sel), ["ort:coreml", "ort:cpu"]);
    assert_eq!(reason(&sel, "openvino:cpu"), Some("run `setup-openvino`"));

    // Intel Mac: CoreML is listed but not picked automatically.
    let hw = HardwareInfo::new("macos", "x86_64", vec![]);
    let sel = select(
        &hw,
        &RuntimeProbe {
            openvino: ov,
            ort: coreml,
        },
        true,
    );
    assert!(runnable(&sel, "ort:coreml"));
    assert_eq!(auto(&sel), ["openvino:cpu"]);
}

fn ov_dev_cpu_mac() -> OvDevice {
    ov_dev("CPU", "Apple M2", None)
}

#[test]
fn cpu_only() {
    let hw = HardwareInfo::new("linux", "x86_64", vec![]);
    let sel = select(
        &hw,
        &RuntimeProbe {
            openvino: openvino(vec![ov_cpu()]),
            ort: OrtProbe::not_in_build(),
        },
        true,
    );
    assert_eq!(specs(&sel), ["openvino:cpu", "ort:cpu"]);
    assert_eq!(auto(&sel), ["openvino:cpu"]);
    assert!(!sel.any_gpu_runnable());
    let c = auto_candidates(&sel);
    assert_eq!(c.len(), 1);
    assert!(!c[0].fell_back && c[0].note.is_none());

    // ONNX Runtime CPU when OpenVINO is missing.
    let sel = select(
        &hw,
        &RuntimeProbe {
            openvino: OpenVinoProbe::unavailable("no OpenVINO"),
            ort: OrtProbe::installed("cpu", vec![EpStatus::usable(Target::Cpu)]),
        },
        true,
    );
    assert_eq!(auto(&sel), ["ort:cpu"]);

    // Nothing at all: auto is empty, the plan still tries OpenVINO CPU (and reports why).
    let sel = select(
        &hw,
        &RuntimeProbe {
            openvino: OpenVinoProbe::unavailable("no OpenVINO"),
            ort: OrtProbe::not_in_build(),
        },
        true,
    );
    assert!(sel.auto.is_empty() && sel.auto_pick().is_none());
    let c = auto_candidates(&sel);
    assert_eq!(c[0].device.to_string(), "openvino:cpu");
    assert!(c[0].note.is_some());
}

#[test]
fn intel_arc_discrete_plus_igpu() {
    let arc = gpu(
        GpuVendor::Intel,
        "Intel(R) Arc(TM) A770 Graphics",
        16256,
        1,
        true,
    );
    let hw = HardwareInfo::new("windows", "x86_64", vec![uhd630(0), arc]);
    let probe = |arc_type: Option<&str>| RuntimeProbe {
        openvino: openvino(vec![
            ov_cpu(),
            ov_dev(
                "GPU.0",
                "Intel(R) UHD Graphics 630 (iGPU)",
                Some("integrated"),
            ),
            ov_dev("GPU.1", "Intel(R) Arc(TM) A770 Graphics (dGPU)", arc_type),
            ov_dev("NPU", "Intel(R) AI Boost", None),
        ]),
        ort: OrtProbe::not_in_build(),
    };
    let sel = select(&hw, &probe(Some("discrete")), true);
    assert_eq!(
        auto(&sel),
        ["openvino:gpu.1", "openvino:gpu.0", "openvino:cpu"]
    );
    // NPU is runnable but never auto.
    assert!(runnable(&sel, "openvino:npu"));
    // Without DEVICE_TYPE, the "(dGPU)" marker in the full name decides.
    let sel = select(&hw, &probe(None), true);
    assert_eq!(
        auto(&sel)[0],
        "openvino:gpu.1",
        "discrete from FULL_DEVICE_NAME"
    );
    let fell_back: Vec<bool> = auto_candidates(&sel).iter().map(|c| c.fell_back).collect();
    assert_eq!(fell_back, [false, true, true]);

    // Two GPUs, neither discrete: only the first, like a bare GPU request.
    let sel = select(
        &hw,
        &RuntimeProbe {
            openvino: openvino(vec![
                ov_cpu(),
                ov_dev(
                    "GPU.0",
                    "Intel(R) UHD Graphics 630 (iGPU)",
                    Some("integrated"),
                ),
                ov_dev(
                    "GPU.1",
                    "Intel(R) UHD Graphics 770 (iGPU)",
                    Some("integrated"),
                ),
            ]),
            ort: OrtProbe::not_in_build(),
        },
        true,
    );
    assert_eq!(auto(&sel), ["openvino:gpu.0", "openvino:cpu"]);

    // A non-Intel device in OpenVINO's GPU list is shown but not runnable.
    let sel = select(
        &hw,
        &RuntimeProbe {
            openvino: openvino(vec![
                ov_cpu(),
                ov_dev(
                    "GPU.0",
                    "Intel(R) UHD Graphics 630 (iGPU)",
                    Some("integrated"),
                ),
                ov_dev("GPU.1", "NVIDIA GeForce RTX 3060 (dGPU)", Some("discrete")),
            ]),
            ort: OrtProbe::not_in_build(),
        },
        true,
    );
    assert!(
        reason(&sel, "openvino:gpu.1")
            .unwrap()
            .contains("not an Intel GPU")
    );
    assert_eq!(auto(&sel), ["openvino:gpu.0", "openvino:cpu"]);
}

#[test]
fn xml_only_model_skips_onnx_runtime() {
    let hw = HardwareInfo::new("windows", "x86_64", vec![uhd630(0), rtx3060(1)]);
    let probe = RuntimeProbe {
        openvino: openvino_igpu(),
        ort: ort_gpu_flavor(None),
    };
    let sel = select(&hw, &probe, false);
    assert_eq!(auto(&sel), ["openvino:gpu", "openvino:cpu"]);
    assert_eq!(reason(&sel, "ort:cuda:0"), Some(NEEDS_ONNX));
    assert_eq!(reason(&sel, "ort:tensorrt:0"), Some(NEEDS_ONNX));
    assert_eq!(reason(&sel, "ort:cpu"), Some(NEEDS_ONNX));
    // An EP problem is reported ahead of the missing export.
    assert!(
        reason(&sel, "ort:directml:1")
            .unwrap()
            .contains("has no DirectML")
    );
    // OpenVINO options are unaffected.
    assert!(runnable(&sel, "openvino:gpu") && runnable(&sel, "openvino:cpu"));

    // ORT-only machine with an .xml model: nothing can run it.
    let probe = RuntimeProbe {
        openvino: OpenVinoProbe::unavailable("no OpenVINO"),
        ort: ort_gpu_flavor(None),
    };
    let sel = select(&hw, &probe, false);
    assert!(sel.auto.is_empty());
    // With an .onnx it would be CUDA, then ORT CPU.
    let sel = select(&hw, &probe, true);
    assert_eq!(auto(&sel), ["ort:cuda:0", "ort:cpu"]);
}

#[test]
fn options_serialize_and_format() {
    let hw = HardwareInfo::new("windows", "x86_64", vec![uhd630(0)]);
    let probe = RuntimeProbe {
        openvino: openvino_igpu(),
        ort: OrtProbe::not_in_build(),
    };
    let sel = select(&hw, &probe, true);
    let json = serde_json::to_value(&sel).unwrap();
    assert_eq!(json["auto"][0], "openvino:gpu");
    assert_eq!(json["options"][0]["spec"], "openvino:gpu");
    assert_eq!(json["options"][0]["runnable"], true);
    assert_eq!(json["options"][1]["reason"], ORT_UNAVAILABLE);

    let table = blue_onyx_prism::backend::select::format_options(&sel);
    assert!(table.lines().next().unwrap().contains("SPEC"));
    assert!(table.contains("openvino:gpu"));
    assert!(table.contains(ORT_UNAVAILABLE));
    assert_eq!(
        blue_onyx_prism::backend::select::format_auto(&sel),
        "openvino:gpu -> openvino:cpu"
    );
}
