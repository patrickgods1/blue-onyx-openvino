//! Real ONNX Runtime inference on `ort:cpu` (and CoreML on macOS). Skipped when the ONNX Runtime
//! library or the model files are absent. The library is looked up in `ORT_DYLIB_PATH`,
//! `<target>/debug/onnxruntime` (where `cargo run -- setup-onnxruntime` installs it) and
//! `<repo>/onnxruntime`. Run with `cargo test --test backend_ort -- --nocapture` for timings.
#![cfg(feature = "onnxruntime")]

use blue_onyx_prism::backend::{
    CoreOptions, LoadRequest, OrtOptions, Runtimes, libs, spec, spec::Runtime,
};
use blue_onyx_prism::model::{ExtraData, ExtraInput, OutputBuf, PortElem};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The ONNX Runtime library to use, if one is installed somewhere we know.
fn ort_dir() -> Option<PathBuf> {
    let env = std::env::var_os(libs::ENV_ORT_DYLIB_PATH).map(PathBuf::from);
    let target_dir = blue_onyx_prism::exe_dir()
        .parent()
        .map(|d| d.join(libs::ORT_DIR_NAME))
        .unwrap_or_default();
    for cand in env
        .into_iter()
        .chain([target_dir, repo().join(libs::ORT_DIR_NAME)])
    {
        if let Some(lib) = libs::find_onnxruntime_from(Some(&cand), None, &cand).library {
            return Some(lib);
        }
    }
    None
}

/// One `Runtimes` per test process is plenty (ONNX Runtime loads once anyway); tests share it.
static RT: Mutex<Option<Runtimes>> = Mutex::new(None);

fn with_runtimes<T>(f: impl FnOnce(&mut Runtimes) -> T) -> Option<T> {
    let lib = ort_dir().or_else(|| {
        eprintln!("skipping: no ONNX Runtime library (run `cargo run -- setup-onnxruntime`)");
        None
    })?;
    let mut guard = RT.lock().unwrap_or_else(|e| e.into_inner());
    let rt = guard.get_or_insert_with(|| {
        Runtimes::new_with(
            &CoreOptions {
                cache_dir: Some(std::env::temp_dir().join("bop-ort-test-cache")),
                intra_threads: 0,
                openvino_dir: Some(repo().join("openvino")),
            },
            &OrtOptions {
                onnxruntime_dir: Some(lib),
                ..OrtOptions::default()
            },
        )
    });
    assert!(
        rt.has_ort(),
        "ONNX Runtime failed to load: {:?}",
        rt.onnxruntime_error()
    );
    Some(f(rt))
}

fn model(name: &str) -> Option<PathBuf> {
    let p = repo().join("models").join(name);
    if p.is_file() {
        Some(p)
    } else {
        eprintln!("skipping: {} not present", p.display());
        None
    }
}

fn req(path: &std::path::Path, requested: &str) -> LoadRequest {
    LoadRequest {
        path: path.to_path_buf(),
        requested: requested.into(),
        gpu_precision: None,
    }
}

#[test]
fn yolo5_on_ort_cpu() {
    let Some(path) = model("IPcam-general.onnx") else {
        return;
    };
    with_runtimes(|rt| {
        println!("ONNX Runtime {:?}", rt.onnxruntime_version());
        let probe = rt.probe().ort.clone();
        assert!(probe.is_available(), "{probe:?}");
        assert_eq!(probe.ep_error(spec::Target::Cpu), None);

        let mut backend = rt
            .load(&spec::parse("ort:cpu").unwrap(), &req(&path, "ort:cpu"))
            .expect("load on ort:cpu");
        let info = backend.info().clone();
        assert_eq!(info.device.runtime, Runtime::Ort);
        assert_eq!(info.device.spec, "ort:cpu");
        assert!(!info.device.fell_back);
        assert_eq!(info.device.execution_provider(), "ONNX Runtime CPU");
        assert_eq!(info.input_size, (640, 640));
        assert_eq!(info.inputs[0].shape, vec![1, 3, 640, 640]);
        assert_eq!(info.inputs[0].elem, PortElem::F32);
        println!(
            "inputs {:?} outputs {:?} compile {} ms",
            info.inputs, info.outputs, info.compile_ms
        );

        let chw = vec![114.0f32 / 255.0; 3 * 640 * 640];
        backend.infer(&chw, &[]).expect("warm-up");
        let t = Instant::now();
        let outs = backend.infer(&chw, &[]).expect("inference");
        println!("ort:cpu inference {:?}", t.elapsed());
        assert_eq!(outs.len(), 1);
        let o = &outs[0];
        assert_eq!(o.shape.len(), 3);
        assert_eq!(o.shape[0], 1);
        assert_eq!(o.shape[2], 8, "IPcam-general: 5 + 3 classes");
        match &o.data {
            OutputBuf::F32(v) => assert_eq!(v.len(), o.shape.iter().product::<usize>()),
            other => panic!("expected f32 output, got {other:?}"),
        }
        // Wrong input size is an error, not a crash.
        assert!(backend.infer(&chw[..10], &[]).is_err());
    });
}

#[test]
fn rtdetr_on_ort_cpu_with_i64_input() {
    let Some(path) = model("rt-detrv2-s.onnx") else {
        return;
    };
    with_runtimes(|rt| {
        let mut backend = rt
            .load(&spec::parse("ort:cpu").unwrap(), &req(&path, "ort:cpu"))
            .expect("load rt-detr on ort:cpu");
        let info = backend.info().clone();
        let sizes = info
            .inputs
            .iter()
            .find(|p| p.name == "orig_target_sizes")
            .expect("orig_target_sizes input");
        assert_eq!(sizes.shape, vec![1, 2]);
        assert_eq!(sizes.elem, PortElem::I64);
        let chw = vec![0.5f32; 3 * 640 * 640];
        let extra = [ExtraInput {
            name: "orig_target_sizes".into(),
            shape: vec![1, 2],
            data: ExtraData::I64(vec![640, 640]),
        }];
        let outs = backend.infer(&chw, &extra).expect("inference");
        let names: Vec<&str> = outs.iter().map(|o| o.name.as_str()).collect();
        for n in ["labels", "boxes", "scores"] {
            assert!(names.contains(&n), "{names:?}");
        }
        let labels = outs.iter().find(|o| o.name == "labels").unwrap();
        assert!(matches!(labels.data, OutputBuf::I64(_)));
        let boxes = outs.iter().find(|o| o.name == "boxes").unwrap();
        assert_eq!(boxes.shape.last(), Some(&4));

        // RT-DETR is never attempted on CoreML (it aborts the process); it falls back to CPU.
        if cfg!(target_os = "macos") {
            let b = rt
                .load(
                    &spec::parse("ort:coreml").unwrap(),
                    &req(&path, "ort:coreml"),
                )
                .expect("falls back to a CPU option");
            assert!(b.info().device.fell_back);
            assert!(b.info().device.spec.ends_with(":cpu"));
        }
    });
}

#[cfg(target_os = "macos")]
#[test]
fn yolo5_on_coreml() {
    let Some(path) = model("IPcam-general.onnx") else {
        return;
    };
    with_runtimes(|rt| {
        if let Some(reason) = rt.probe().ort.ep_error(spec::Target::CoreMl) {
            eprintln!("skipping: CoreML not usable: {reason}");
            return;
        }
        let mut backend = rt
            .load(
                &spec::parse("ort:coreml").unwrap(),
                &req(&path, "ort:coreml"),
            )
            .expect("load on ort:coreml");
        assert_eq!(
            backend.info().device.execution_provider(),
            "ONNX Runtime CoreML"
        );
        let chw = vec![114.0f32 / 255.0; 3 * 640 * 640];
        backend.infer(&chw, &[]).expect("warm-up");
        let t = Instant::now();
        let outs = backend.infer(&chw, &[]).expect("inference");
        println!("ort:coreml inference {:?}", t.elapsed());
        assert_eq!(outs[0].shape[2], 8);
    });
}

#[test]
fn xml_without_onnx_needs_export() {
    let Some(path) = model("yolo26n.xml") else {
        return;
    };
    if path.with_extension("onnx").is_file() {
        return;
    }
    with_runtimes(|rt| {
        let cand = blue_onyx_prism::backend::Candidate {
            device: *spec::parse("ort:cpu").unwrap().device().unwrap(),
            fell_back: false,
            note: None,
        };
        let err = rt
            .compile(&cand, &req(&path, "ort:cpu"))
            .err()
            .expect("needs .onnx");
        assert!(format!("{err:#}").contains("needs ONNX export"), "{err:#}");
    });
}
