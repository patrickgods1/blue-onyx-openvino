//! Real OpenVINO inference: sets up the runtime (from a local archive when available, else
//! download) and the IPcam-general model, then runs one inference on CPU and on GPU.
//! Set `BLUE_ONYX_SKIP_OPENVINO_TESTS=1` to skip (e.g. offline CI).
//! Run with `cargo test --test backend_cpu -- --nocapture` to see timings.

use blue_onyx_prism::backend::{CoreOptions, LoadRequest, Runtimes, spec};
use blue_onyx_prism::setup_openvino;
use std::path::{Path, PathBuf};

const MODEL_URL: &str = "https://huggingface.co/xnorpx/blue-onyx-yolo5/resolve/main/IPcam-general";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ensure_runtime(dir: &Path) {
    if dir.is_dir() {
        return;
    }
    // Set BLUE_ONYX_PRISM_ARCHIVE to a downloaded archive to skip the download.
    let archive = std::env::var_os("BLUE_ONYX_PRISM_ARCHIVE").map(PathBuf::from);
    setup_openvino::run(&setup_openvino::SetupOptions {
        dest: Some(dir.to_path_buf()),
        version: None,
        archive,
        keep_archive: false,
    })
    .expect("OpenVINO runtime setup");
}

fn ensure_model(models: &Path) -> PathBuf {
    let onnx = models.join("IPcam-general.onnx");
    for ext in ["onnx", "yaml"] {
        let p = models.join(format!("IPcam-general.{ext}"));
        if !p.is_file() {
            setup_openvino::download_to_file(&format!("{MODEL_URL}.{ext}"), &p)
                .unwrap_or_else(|e| panic!("downloading {}: {e:#}", p.display()));
        }
    }
    onnx
}

#[test]
fn cpu_and_gpu_inference() {
    if std::env::var_os("BLUE_ONYX_SKIP_OPENVINO_TESTS").is_some() {
        eprintln!("skipping: BLUE_ONYX_SKIP_OPENVINO_TESTS set");
        return;
    }
    let root = repo();
    let ov_dir = root.join("openvino");
    ensure_runtime(&ov_dir);
    let model_path = ensure_model(&root.join("models"));

    let mut rt = Runtimes::new(&CoreOptions {
        cache_dir: Some(root.join("cache")),
        intra_threads: 0,
        openvino_dir: Some(ov_dir),
    });
    rt.require_any().expect("Runtimes::new");
    let core = rt.openvino().expect("OpenVINO core");
    println!("OpenVINO {}", core.openvino_version());
    println!("available devices: {:?}", core.available_devices());
    for d in core.available_devices().to_vec() {
        println!("  {d}: {}", core.device_full_name(&d));
    }
    let has_gpu = core.has_gpu();
    let req = |requested: &str| LoadRequest {
        path: model_path.clone(),
        requested: requested.into(),
        gpu_precision: None,
    };

    // CPU, no fallback.
    let mut backend = rt
        .load(&spec::parse("CPU").unwrap(), &req("CPU"))
        .expect("load on CPU");
    let info = backend.info().clone();
    println!(
        "CPU: {:?} compile {} ms, inputs {:?}, outputs {:?}",
        info.device, info.compile_ms, info.inputs, info.outputs
    );
    assert_eq!(info.image_input, "images");
    assert_eq!(info.input_size, (640, 640));
    assert_eq!(info.inputs.len(), 1);
    assert_eq!(info.inputs[0].shape, vec![1, 3, 640, 640]);
    assert_eq!(info.outputs.len(), 1);
    assert_eq!(info.outputs[0].shape, vec![1, 25200, 8]);
    assert_eq!(info.device.actual, "CPU");
    assert_eq!(info.device.spec, "openvino:cpu");
    assert!(!info.device.fell_back);

    let chw = vec![0f32; 3 * 640 * 640];
    let t = std::time::Instant::now();
    let outs = backend.infer(&chw, &[]).expect("infer CPU");
    println!("CPU first inference {} ms", t.elapsed().as_millis());
    let t = std::time::Instant::now();
    backend.infer(&chw, &[]).expect("infer CPU");
    println!("CPU second inference {} ms", t.elapsed().as_millis());
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].shape, vec![1, 25200, 8]);
    assert_eq!(outs[0].as_f32().expect("f32 output").len(), 25200 * 8);
    assert!(backend.infer(&chw[1..], &[]).is_err(), "length check");

    // GPU with CPU fallback, loaded twice (second load should hit the compile cache).
    for run in 1..=2 {
        let mut backend = rt
            .load(&spec::parse("GPU").unwrap(), &req("GPU"))
            .expect("load on GPU (with fallback)");
        let dev = &backend.info().device;
        println!(
            "GPU run {run}: {:?} compile {} ms, provider '{}'",
            dev,
            backend.info().compile_ms,
            dev.execution_provider()
        );
        assert_eq!(dev.requested, "GPU");
        if !has_gpu {
            // No GPU listed: CPU, reported as a fallback (unchanged from before 6.1).
            assert_eq!(dev.actual, "CPU");
            assert!(dev.fell_back);
            assert!(dev.execution_provider().ends_with(", fallback)"));
        }
        let t = std::time::Instant::now();
        let outs = backend.infer(&chw, &[]).expect("infer GPU");
        let first = t.elapsed().as_millis();
        let t = std::time::Instant::now();
        backend.infer(&chw, &[]).expect("infer GPU");
        println!(
            "GPU run {run}: first inference {first} ms, second {} ms",
            t.elapsed().as_millis()
        );
        assert_eq!(outs[0].as_f32().expect("f32 output").len(), 25200 * 8);
    }

    // ONNX Runtime is not in this build yet: the spec falls back to OpenVINO CPU.
    let backend = rt
        .load(&spec::parse("ort:cuda").unwrap(), &req("ort:cuda"))
        .expect("ort:cuda falls back to CPU");
    let dev = &backend.info().device;
    assert_eq!((dev.actual.as_str(), dev.fell_back), ("CPU", true));
    assert_eq!(dev.requested, "ort:cuda");
}
