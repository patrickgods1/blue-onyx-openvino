//! Resource catalog and resolver (docs/PLAN.md, "On-demand resources"). Synthetic hardware and
//! install snapshots; no network, no runtime libraries.

use blue_onyx_prism::backend::detect::{GpuAdapter, GpuVendor, HardwareInfo};
use blue_onyx_prism::backend::select::{EpStatus, OpenVinoProbe, OrtProbe, RuntimeProbe};
use blue_onyx_prism::backend::spec::Target;
use blue_onyx_prism::config::{Config, ModelConfig};
use blue_onyx_prism::model::ModelFamilyKind;
use blue_onyx_prism::resources::catalog::{
    self, CUDA_LIBS_ID, Flavor, Layout, OPENVINO_RUNTIME_ID, Platform, SHIPPED_PLATFORMS,
};
use blue_onyx_prism::resources::resolve::{FETCH_FOR_CONFIG, Installed, Resolution, needed};
use std::collections::BTreeSet;
use std::path::PathBuf;

// ---------------------------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------------------------

/// The ORT flavors `setup-onnxruntime` shipped per platform before the catalog existed.
fn expected_flavors(p: Platform) -> &'static [Flavor] {
    match (p.os, p.arch) {
        ("windows", "x86_64") => &[Flavor::Cpu, Flavor::Cuda, Flavor::DirectMl],
        ("linux", "x86_64") => &[Flavor::Cpu, Flavor::Cuda],
        ("linux", "aarch64") => &[Flavor::Cpu],
        ("macos", "aarch64") => &[Flavor::Cpu, Flavor::CoreMl],
        _ => &[],
    }
}

#[test]
fn every_shipped_platform_has_runtimes() {
    for &p in SHIPPED_PLATFORMS {
        let ov = catalog::openvino_runtime(p.os, p.arch)
            .unwrap_or_else(|| panic!("no openvino-runtime for {p}"));
        assert_eq!(ov.id, OPENVINO_RUNTIME_ID);
        assert!(ov.parts[0].url.contains(catalog::OPENVINO_VERSION), "{p}");
        let mut flavors = catalog::ort_flavors(p.os, p.arch);
        flavors.sort();
        let mut want = expected_flavors(p).to_vec();
        want.sort();
        assert_eq!(flavors, want, "ORT flavors for {p}");
        for f in flavors {
            let r = catalog::onnxruntime(p.os, p.arch, f).unwrap();
            assert_eq!(r.id, f.resource_id());
            assert_eq!(r.version, catalog::ONNXRUNTIME_VERSION);
            // setup-onnxruntime reads the same table.
            let parts = blue_onyx_prism::setup_onnxruntime::packages_for(p.os, p.arch, f).unwrap();
            assert_eq!(parts.len(), r.parts.len());
            assert_eq!(parts[0].sha256, r.parts[0].sha256);
        }
        let pkg = blue_onyx_prism::setup_openvino::package_for(p.os, p.arch).unwrap();
        assert_eq!(pkg.sha256, ov.parts[0].sha256);
        let cuda = catalog::cuda_libs_for(p.os, p.arch);
        assert_eq!(
            cuda.is_some(),
            flavors_has_cuda(p),
            "nvidia-cuda-libs exactly where the CUDA flavor is: {p}"
        );
    }
    assert!(catalog::openvino_runtime("macos", "x86_64").is_none());
}

fn flavors_has_cuda(p: Platform) -> bool {
    expected_flavors(p).contains(&Flavor::Cuda)
}

#[test]
fn hashes_urls_and_sizes_are_well_formed() {
    let mut keys = BTreeSet::new();
    for r in catalog::all() {
        assert!(
            keys.insert((r.id, r.platform)),
            "duplicate {} {:?}",
            r.id,
            r.platform
        );
        assert!(!r.parts.is_empty(), "{}", r.id);
        assert!(!r.dest.is_empty() && !r.dest.starts_with('/') && !r.dest.contains(".."));
        for p in r.parts {
            assert_eq!(p.sha256.len(), 64, "{} {}", r.id, p.url);
            assert!(
                p.sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "{} sha256 must be lowercase hex: {}",
                r.id,
                p.sha256
            );
            assert!(p.url.starts_with("https://"), "{}", p.url);
            assert!(p.size > 0, "{}", p.url);
            assert!(p.url.ends_with(p.file_name), "{} vs {}", p.url, p.file_name);
            assert!(!p.file_name.contains('/') && !p.file_name.contains('\\'));
            assert_eq!(p.archive.is_none(), p.layout == Layout::File, "{}", p.url);
        }
    }
    // Hugging Face downloads are pinned to a commit, not `main`.
    for r in catalog::MODELS {
        for p in r.parts {
            assert!(!p.url.contains("/resolve/main/"), "{}", p.url);
            assert!(p.url.contains(r.version), "{}", p.url);
        }
    }
}

#[test]
fn large_downloads_are_only_the_cuda_libs() {
    for r in catalog::all() {
        assert_eq!(
            r.is_large(),
            r.id == CUDA_LIBS_ID,
            "{} ({} bytes)",
            r.id,
            r.size()
        );
    }
    assert_eq!(catalog::LARGE_DOWNLOAD_BYTES, 500 * 1024 * 1024);
}

#[test]
fn download_models_reads_the_catalog() {
    // `download-models` / `list-models` use the catalog's model table directly.
    let dl = blue_onyx_prism::download::catalog();
    assert_eq!(dl.len(), catalog::MODELS.len());
    for r in dl {
        let name = r.model_name().unwrap();
        assert_eq!(blue_onyx_prism::download::find(name).unwrap().id, r.id);
        assert!(!r.description.is_empty(), "{name}");
        let files: Vec<&str> = r.parts.iter().map(|p| p.file_name).collect();
        assert_eq!(files, [format!("{name}.onnx"), format!("{name}.yaml")]);
    }
}

// ---------------------------------------------------------------------------------------------
// Resolver
// ---------------------------------------------------------------------------------------------

fn gpu(vendor: GpuVendor, name: &str, vram_mb: u64, index: u32, discrete: bool) -> GpuAdapter {
    GpuAdapter {
        vendor,
        name: name.into(),
        vram_mb,
        index,
        discrete,
    }
}

fn intel_igpu(index: u32) -> GpuAdapter {
    gpu(
        GpuVendor::Intel,
        "Intel(R) UHD Graphics 630",
        0,
        index,
        false,
    )
}

fn rtx(index: u32) -> GpuAdapter {
    gpu(
        GpuVendor::Nvidia,
        "NVIDIA GeForce RTX 3060",
        12288,
        index,
        true,
    )
}

fn win(gpus: Vec<GpuAdapter>) -> HardwareInfo {
    HardwareInfo::new("windows", "x86_64", gpus)
}

fn mac() -> HardwareInfo {
    HardwareInfo::new(
        "macos",
        "aarch64",
        vec![gpu(GpuVendor::Apple, "Apple M2 GPU", 0, 0, false)],
    )
}

fn model(path: &str) -> ModelConfig {
    ModelConfig {
        path: PathBuf::from(path),
        ..ModelConfig::default()
    }
}

fn config(models: Vec<ModelConfig>) -> Config {
    Config {
        models,
        ..Config::default()
    }
}

fn ipcam() -> Config {
    config(vec![model("models/IPcam-general.onnx")])
}

/// Model files present (as written in the config), no runtimes.
fn files_only(cfg: &Config) -> Installed {
    let mut files = Vec::new();
    for m in &cfg.models {
        files.push(m.path.clone());
        files.push(m.path.with_extension("yaml"));
    }
    Installed {
        ort_in_build: true,
        ..Installed::default()
    }
    .with_files(&files)
}

fn ids(r: &Resolution) -> Vec<&str> {
    r.needs.iter().map(|n| n.id()).collect()
}

fn pick(r: &Resolution, model: &str) -> String {
    r.model(model)
        .and_then(|m| m.device)
        .map(|d| d.to_string())
        .unwrap_or_else(|| format!("none: {:?}", r.model(model)))
}

#[test]
fn fresh_windows_intel_igpu_needs_openvino() {
    let cfg = ipcam();
    let r = needed(&cfg, &win(vec![intel_igpu(0)]), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:gpu");
    let n = &r.needs[0];
    assert_eq!(n.blocking_models, ["IPcam-general"]);
    assert!(!n.after_restart);
    assert_eq!(n.resource.platform, Some(Platform::WINDOWS_X64));
    assert_eq!(r.ort_flavor, None);
    assert!(r.optional.is_empty());
    assert_eq!(r.downloads().len(), 1);
}

#[test]
fn fresh_windows_nvidia_with_large_downloads_needs_cuda() {
    let mut cfg = ipcam();
    cfg.allow_large_downloads = true;
    let r = needed(&cfg, &win(vec![intel_igpu(0), rtx(1)]), &files_only(&cfg));
    // Plus OpenVINO for the plan's CPU fallback (blocks nothing).
    assert_eq!(
        ids(&r),
        ["onnxruntime-cuda", CUDA_LIBS_ID, OPENVINO_RUNTIME_ID]
    );
    assert!(r.needs[2].blocking_models.is_empty());
    assert!(
        r.needs[2].reason.contains("CPU fallback"),
        "{}",
        r.needs[2].reason
    );
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
    assert_eq!(r.ort_flavor, Some(Flavor::Cuda));
    assert!(!r.ort_restart_required());
    assert!(r.needs.iter().all(|n| !n.after_restart));
    assert_eq!(r.needs[0].device.unwrap().to_string(), "ort:cuda:0");
}

#[test]
fn fresh_windows_nvidia_without_large_downloads_falls_to_openvino() {
    let cfg = ipcam();
    // NVIDIA + Intel iGPU: the Intel GPU via OpenVINO.
    let r = needed(&cfg, &win(vec![intel_igpu(0), rtx(1)]), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:gpu");
    // The skipped CUDA option is offered as an optional (large) download.
    let opt: Vec<&str> = r.optional.iter().map(|n| n.id()).collect();
    assert_eq!(opt, ["onnxruntime-cuda", CUDA_LIBS_ID]);
    let skipped = &r.model("IPcam-general").unwrap().skipped;
    assert!(
        skipped[0].starts_with("ort:cuda:0: needs NVIDIA CUDA 12 + cuDNN 9 libraries"),
        "{skipped:?}"
    );
    assert!(skipped[0].contains("allow_large_downloads"));
    assert_eq!(r.ort_flavor, None);

    // NVIDIA only: OpenVINO CPU.
    let r = needed(&cfg, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");
}

#[test]
fn installed_system_cuda_needs_no_libs() {
    // The CUDA flavor is loaded and its EP works (CUDA installed system-wide).
    let cfg = ipcam();
    let mut inst = files_only(&cfg).with_resources(&["onnxruntime-cuda", OPENVINO_RUNTIME_ID]);
    inst.active_ort = Some(Flavor::Cuda);
    inst.probe = Some(RuntimeProbe {
        openvino: OpenVinoProbe::with_device_names(&["CPU".to_string()]),
        ort: OrtProbe::installed(
            "cuda",
            vec![
                EpStatus::usable(Target::Cuda),
                EpStatus::broken(Target::TensorRt, "libnvinfer missing"),
                EpStatus::usable(Target::Cpu),
            ],
        ),
    });
    let r = needed(&cfg, &win(vec![rtx(0)]), &inst);
    assert!(r.is_satisfied(), "{:?}", ids(&r));
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
    assert_eq!(r.ort_flavor, Some(Flavor::Cuda));
}

#[test]
fn fresh_macos_needs_coreml_flavor() {
    let cfg = ipcam();
    let r = needed(&cfg, &mac(), &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-coreml", OPENVINO_RUNTIME_ID]);
    assert_eq!(r.needs[0].blocking_models, ["IPcam-general"]);
    assert!(
        r.needs[1].blocking_models.is_empty(),
        "the CPU fallback blocks nothing"
    );
    assert_eq!(pick(&r, "IPcam-general"), "ort:coreml");
    assert_eq!(r.ort_flavor, Some(Flavor::CoreMl));
    assert_eq!(r.needs[0].resource.dest, "onnxruntime/coreml");
}

#[test]
fn rtdetr_on_macos_skips_coreml() {
    let cfg = config(vec![model("models/rt-detrv2-s.onnx")]);
    let r = needed(&cfg, &mac(), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "rt-detrv2-s"), "openvino:cpu");
    let skipped = &r.model("rt-detrv2-s").unwrap().skipped;
    assert!(skipped[0].contains("RT-DETR"), "{skipped:?}");
}

#[test]
fn fresh_linux_cpu_and_arm() {
    let cfg = ipcam();
    for hw in [
        HardwareInfo::new("linux", "x86_64", vec![]),
        HardwareInfo::new("linux", "aarch64", vec![]),
    ] {
        let r = needed(&cfg, &hw, &files_only(&cfg));
        assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID], "{}", hw.arch);
        assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");
        assert_eq!(r.needs[0].resource.platform.unwrap().arch, hw.arch);
    }
}

#[test]
fn everything_installed_needs_nothing() {
    let cfg = ipcam();
    let inst = files_only(&cfg).with_resources(&[OPENVINO_RUNTIME_ID]);
    let r = needed(&cfg, &win(vec![intel_igpu(0)]), &inst);
    assert!(r.is_satisfied());
    assert_eq!(r.manual_message(), None);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:gpu");
}

#[test]
fn missing_model_files() {
    let cfg = ipcam();
    let inst = Installed {
        ort_in_build: true,
        ..Installed::default()
    }
    .with_resources(&[OPENVINO_RUNTIME_ID]);
    let r = needed(&cfg, &HardwareInfo::new("linux", "x86_64", vec![]), &inst);
    assert_eq!(ids(&r), ["model:IPcam-general"]);
    let n = &r.needs[0];
    assert_eq!(n.model_dir.as_deref(), Some(std::path::Path::new("models")));
    assert_eq!(n.blocking_models, ["IPcam-general"]);
    assert_eq!(n.resource.parts.len(), 2);
    assert!(
        n.reason.contains("models/IPcam-general.onnx"),
        "{}",
        n.reason
    );

    // Only the class file missing: the model resource is still needed.
    let inst = inst.with_files(&["models/IPcam-general.onnx"]);
    let r = needed(&cfg, &HardwareInfo::new("linux", "x86_64", vec![]), &inst);
    assert_eq!(ids(&r), ["model:IPcam-general"]);
    assert!(r.needs[0].reason.contains("IPcam-general.yaml"));

    // A file that is not in the catalog cannot be fetched; YOLO26 says how to export it.
    let cfg = config(vec![
        model("models/yolo26s.xml"),
        model("models/custom.onnx"),
    ]);
    let r = needed(&cfg, &HardwareInfo::new("linux", "x86_64", vec![]), &inst);
    assert!(r.is_satisfied());
    assert!(
        r.model("yolo26s")
            .unwrap()
            .error
            .as_deref()
            .unwrap()
            .contains("export_yolo26.py")
    );
    assert!(
        r.model("custom")
            .unwrap()
            .error
            .as_deref()
            .unwrap()
            .contains("not in the download catalog")
    );
}

#[test]
fn ort_flavor_switch_is_active_after_restart() {
    // DirectML installed and active on an NVIDIA box; CUDA is better and allowed.
    let mut cfg = ipcam();
    cfg.allow_large_downloads = true;
    let mut inst = files_only(&cfg).with_resources(&[OPENVINO_RUNTIME_ID, "onnxruntime-directml"]);
    inst.active_ort = Some(Flavor::DirectMl);
    let r = needed(&cfg, &win(vec![rtx(0)]), &inst);
    assert_eq!(ids(&r), ["onnxruntime-cuda", CUDA_LIBS_ID]);
    assert!(
        r.needs[0].after_restart,
        "a second flavor loads after a restart"
    );
    assert!(!r.needs[1].after_restart);
    assert_eq!(r.ort_flavor, Some(Flavor::Cuda));
    assert_eq!(r.active_ort, Some(Flavor::DirectMl));
    assert!(r.ort_restart_required());

    // A user-managed ORT (`onnxruntime_dir`) is never replaced: DirectML on the NVIDIA GPU.
    inst.ort_pinned = true;
    let r = needed(&cfg, &win(vec![rtx(0)]), &inst);
    assert!(r.is_satisfied(), "{:?}", ids(&r));
    assert_eq!(pick(&r, "IPcam-general"), "ort:directml:0");
    assert!(!r.ort_restart_required());
}

#[test]
fn one_flavor_per_process_across_models() {
    // The default model pins DirectML; the second (auto) model then uses DirectML too instead of
    // asking for the CUDA flavor.
    let mut cfg = config(vec![
        ModelConfig {
            device: Some("ort:directml".into()),
            ..model("models/IPcam-general.onnx")
        },
        model("models/IPcam-dark.onnx"),
    ]);
    cfg.allow_large_downloads = true;
    let r = needed(&cfg, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-directml", OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "ort:directml");
    assert_eq!(pick(&r, "IPcam-dark"), "ort:directml:0");
    assert_eq!(r.needs[0].blocking_models, ["IPcam-general", "IPcam-dark"]);
    assert_eq!(r.ort_flavor, Some(Flavor::DirectMl));

    // With `default_model` naming the second model, it goes first and picks CUDA; the explicit
    // DirectML model then falls back to OpenVINO CPU.
    cfg.default_model = Some("IPcam-dark".into());
    let r = needed(&cfg, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(
        ids(&r),
        ["onnxruntime-cuda", CUDA_LIBS_ID, OPENVINO_RUNTIME_ID]
    );
    assert_eq!(pick(&r, "IPcam-dark"), "ort:cuda:0");
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");
    assert_eq!(r.models[0].model, "IPcam-dark");
}

#[test]
fn large_downloads_disallowed_for_explicit_cuda() {
    let cfg = config(vec![ModelConfig {
        device: Some("ort:cuda".into()),
        ..model("models/IPcam-general.onnx")
    }]);
    let r = needed(&cfg, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");
    let opt: Vec<&str> = r.optional.iter().map(|n| n.id()).collect();
    assert_eq!(opt, ["onnxruntime-cuda", CUDA_LIBS_ID]);
    // TensorRT is never downloadable.
    let cfg = config(vec![ModelConfig {
        device: Some("ort:tensorrt".into()),
        ..model("models/IPcam-general.onnx")
    }]);
    let mut big = cfg.clone();
    big.allow_large_downloads = true;
    let r = needed(&big, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");
    assert!(r.model("IPcam-general").unwrap().skipped[0].contains("TensorRT"));
}

#[test]
fn auto_download_off_reports_with_command() {
    let mut cfg = ipcam();
    cfg.auto_download = false;
    let inst = Installed {
        ort_in_build: true,
        ..Installed::default()
    };
    let r = needed(&cfg, &win(vec![intel_igpu(0)]), &inst);
    assert_eq!(ids(&r), ["model:IPcam-general", OPENVINO_RUNTIME_ID]);
    assert!(r.downloads().is_empty(), "nothing is downloaded");
    let msg = r.manual_message().unwrap();
    assert!(msg.contains(FETCH_FOR_CONFIG), "{msg}");
    assert!(msg.contains("OpenVINO runtime 2026.4.0 (235 MB)"), "{msg}");
    assert!(msg.contains("Model IPcam-general"), "{msg}");
    assert_eq!(
        r.needs[1].fetch_command(),
        "blue-onyx-prism fetch --resource openvino-runtime"
    );
    let json = serde_json::to_value(&r).unwrap();
    assert_eq!(json["needs"][1]["resource"], "openvino-runtime");
    assert_eq!(json["auto_download"], false);
}

#[test]
fn explicit_device_specs() {
    let linux = HardwareInfo::new("linux", "x86_64", vec![]);
    // ort:cpu on Linux: the plain CPU package.
    let cfg = config(vec![ModelConfig {
        device: Some("ort:cpu".into()),
        ..model("models/IPcam-general.onnx")
    }]);
    let r = needed(&cfg, &linux, &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-cpu", OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "ort:cpu");

    // ort:cpu on macOS takes the CoreML package (the same archive), so CoreML stays possible.
    let r = needed(&cfg, &mac(), &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-coreml", OPENVINO_RUNTIME_ID]);

    // openvino:gpu without an Intel GPU: OpenVINO CPU as the fallback.
    let cfg = config(vec![ModelConfig {
        device: Some("openvino:gpu".into()),
        ..model("models/IPcam-general.onnx")
    }]);
    let r = needed(&cfg, &mac(), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");

    // Global legacy "CPU" with force_cpu on Windows NVIDIA: OpenVINO CPU.
    let mut cfg = ipcam();
    cfg.force_cpu = true;
    cfg.allow_large_downloads = true;
    let r = needed(&cfg, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");

    // An `.xml` model has no ONNX for ORT: CoreML is skipped on macOS.
    let cfg = config(vec![ModelConfig {
        family: ModelFamilyKind::Yolo26,
        ..model("models/yolo26s.xml")
    }]);
    let r = needed(&cfg, &mac(), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    // ...unless a sibling `.onnx` exists.
    let inst = files_only(&cfg).with_files(&["models/yolo26s.onnx"]);
    let r = needed(&cfg, &mac(), &inst);
    assert_eq!(ids(&r), ["onnxruntime-coreml", OPENVINO_RUNTIME_ID]);
}

#[test]
fn disabled_models_and_builds_without_ort() {
    let mut cfg = config(vec![
        model("models/IPcam-general.onnx"),
        ModelConfig {
            enabled: false,
            ..model("models/rt-detrv2-s.onnx")
        },
    ]);
    let mut inst = files_only(&cfg);
    inst.files.remove(&PathBuf::from("models/rt-detrv2-s.onnx"));
    inst.ort_in_build = false;
    // macOS without ONNX Runtime support in the binary: OpenVINO CPU, and the disabled model's
    // missing file is ignored.
    let r = needed(&cfg, &mac(), &inst);
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(r.models.len(), 1);
    // A live probe with OpenVINO loaded counts as installed (e.g. a system install).
    inst.probe = Some(RuntimeProbe::openvino_only(&["CPU".to_string()]));
    let r = needed(&cfg, &mac(), &inst);
    assert!(r.is_satisfied());
    cfg.models[0].device = Some("bogus:device".into());
    let r = needed(&cfg, &mac(), &inst);
    assert!(r.model("IPcam-general").unwrap().error.is_some());
}

#[test]
fn detect_installed_reads_the_layout() {
    use blue_onyx_prism::resources::resolve::detect_installed;
    let root = std::env::temp_dir().join(format!("bop-resources-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("openvino")).unwrap();
    std::fs::write(root.join("openvino/VERSION"), "2026.4.0\n").unwrap();
    std::fs::create_dir_all(root.join("onnxruntime/cuda")).unwrap();
    std::fs::write(root.join("onnxruntime/cuda/flavor.txt"), "cuda\n").unwrap();
    std::fs::create_dir_all(root.join("models")).unwrap();
    let model_path = root.join("models/IPcam-general.onnx");
    std::fs::write(&model_path, "x").unwrap();
    let cfg = config(vec![ModelConfig {
        path: model_path.clone(),
        ..ModelConfig::default()
    }]);
    let inst = detect_installed(&cfg, &root, None);
    assert!(inst.has(OPENVINO_RUNTIME_ID));
    assert!(inst.has("onnxruntime-cuda"));
    assert!(!inst.has(CUDA_LIBS_ID));
    assert!(inst.has_file(&model_path));
    assert!(!inst.has_file(&model_path.with_extension("yaml")));
    // The per-flavor dir is not what the loader picks today (it reads `onnxruntime/` flat),
    // unless ORT_DYLIB_PATH points elsewhere.
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        assert_eq!(inst.active_ort, None);
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn fetch_jobs_follow_the_resolver() {
    use blue_onyx_prism::resources::commands::{FetchRequest, fetch_jobs};
    let root = PathBuf::from("/srv/bop");
    let cfg = ipcam();
    let hw = win(vec![rtx(0)]);
    let inst = files_only(&cfg);
    let ids = |jobs: &[blue_onyx_prism::resources::Job]| -> Vec<&str> {
        jobs.iter().map(|j| j.resource.id).collect()
    };
    // Default = --for-config: OpenVINO, and a note about the skipped CUDA libraries.
    let (jobs, notes) = fetch_jobs(&cfg, &hw, &inst, &root, &FetchRequest::default()).unwrap();
    assert_eq!(ids(&jobs), [OPENVINO_RUNTIME_ID]);
    assert_eq!(jobs[0].target, root.join("openvino"));
    assert!(
        notes
            .iter()
            .any(|n| n.contains(CUDA_LIBS_ID) && n.contains("--allow-large"))
    );
    // --allow-large: the CUDA flavor (marked active, nothing active yet) and the libraries.
    let req = FetchRequest {
        allow_large: true,
        ..FetchRequest::default()
    };
    let (jobs, _) = fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap();
    assert_eq!(
        ids(&jobs),
        ["onnxruntime-cuda", CUDA_LIBS_ID, OPENVINO_RUNTIME_ID]
    );
    assert!(jobs[0].activate_ort);
    assert_eq!(jobs[0].target, root.join("onnxruntime/cuda"));
    assert_eq!(jobs[1].target, root.join("onnxruntime/cuda-libs"));
    // Named resources; large ones need --allow-large; unknown ids list what exists.
    let req = FetchRequest {
        resources: vec!["model:ipcam-dark".into(), "onnxruntime-directml".into()],
        ..FetchRequest::default()
    };
    let (jobs, _) = fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap();
    assert_eq!(ids(&jobs), ["model:IPcam-dark", "onnxruntime-directml"]);
    assert!(jobs[0].target.ends_with("models"));
    let req = FetchRequest {
        resources: vec![CUDA_LIBS_ID.into()],
        ..FetchRequest::default()
    };
    // Naming the large resource is the opt-in.
    let (jobs, notes) = fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap();
    assert_eq!(ids(&jobs), [CUDA_LIBS_ID]);
    assert!(notes.iter().any(|n| n.contains("NVIDIA")), "{notes:?}");
    let req = FetchRequest {
        resources: vec!["onnxruntime-coreml".into()],
        ..FetchRequest::default()
    };
    assert!(
        format!(
            "{:#}",
            fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap_err()
        )
        .contains("available: openvino-runtime")
    );
    // Everything for the platform, without the large libraries.
    let req = FetchRequest {
        all_for_platform: true,
        ..FetchRequest::default()
    };
    let (jobs, notes) = fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap();
    assert_eq!(jobs.len(), 1 + 3 + catalog::MODELS.len());
    assert!(!ids(&jobs).contains(&CUDA_LIBS_ID));
    assert_eq!(notes.len(), 1);
}

#[test]
fn detect_installed_reads_manifests() {
    use blue_onyx_prism::resources::manager::{MANIFEST_FILE, Manifest};
    use blue_onyx_prism::resources::resolve::detect_installed;
    let root = std::env::temp_dir().join(format!("bop-manifest-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let write = |dir: &str, id: &str| {
        std::fs::create_dir_all(root.join(dir)).unwrap();
        let m = Manifest {
            id: id.into(),
            version: "x".into(),
            sha256: vec![],
            files: vec![],
        };
        std::fs::write(
            root.join(dir).join(MANIFEST_FILE),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
    };
    write("openvino", OPENVINO_RUNTIME_ID);
    write("onnxruntime/directml", "onnxruntime-directml");
    write("onnxruntime/cuda-libs", CUDA_LIBS_ID);
    // A manifest for something else does not count.
    write("onnxruntime/cpu", "onnxruntime-cuda");
    let inst = detect_installed(&Config::default(), &root, None);
    assert!(inst.has(OPENVINO_RUNTIME_ID));
    assert!(inst.has("onnxruntime-directml"));
    assert!(inst.has(CUDA_LIBS_ID));
    assert!(!inst.has("onnxruntime-cpu"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn cuda_libraries_system_install_vs_download() {
    let hw = win(vec![rtx(0)]);
    for allow_large in [false, true] {
        let mut cfg = ipcam();
        cfg.allow_large_downloads = allow_large;
        // A CUDA toolkit on the library path: no 1.8 GB download, large downloads or not.
        let mut inst = files_only(&cfg);
        inst.system_cuda = true;
        let r = needed(&cfg, &hw, &inst);
        assert_eq!(
            pick(&r, "IPcam-general"),
            "ort:cuda:0",
            "allow_large={allow_large}"
        );
        assert!(!ids(&r).contains(&CUDA_LIBS_ID), "{:?}", ids(&r));
        assert_eq!(ids(&r)[0], "onnxruntime-cuda");
        assert!(r.optional.is_empty());

        // No system CUDA: the libraries are needed (allowed) or the plan moves on (not allowed).
        let inst = files_only(&cfg);
        let r = needed(&cfg, &hw, &inst);
        if allow_large {
            assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
            assert!(ids(&r).contains(&CUDA_LIBS_ID));
        } else {
            assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");
            assert!(!ids(&r).contains(&CUDA_LIBS_ID));
            assert!(r.optional.iter().any(|n| n.id() == CUDA_LIBS_ID));
        }
    }
    // Installed nvidia-cuda-libs count like a system install.
    let cfg = ipcam();
    let inst = files_only(&cfg).with_resources(&[CUDA_LIBS_ID]);
    let r = needed(&cfg, &hw, &inst);
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
    assert_eq!(ids(&r), ["onnxruntime-cuda", OPENVINO_RUNTIME_ID]);
    // Linux too; never on macOS / Linux arm64 (no CUDA flavor there).
    let linux = HardwareInfo::new("linux", "x86_64", vec![rtx(0)]);
    let mut inst = files_only(&cfg);
    inst.system_cuda = true;
    assert_eq!(
        pick(&needed(&cfg, &linux, &inst), "IPcam-general"),
        "ort:cuda:0"
    );
    let arm = HardwareInfo::new("linux", "aarch64", vec![rtx(0)]);
    assert_eq!(
        pick(&needed(&cfg, &arm, &inst), "IPcam-general"),
        "openvino:cpu"
    );
}
