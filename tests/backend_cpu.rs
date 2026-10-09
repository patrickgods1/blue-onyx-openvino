//! Real OpenVINO inference: sets up the runtime (from a local archive when available, else
//! download) and the IPcam-general model, then runs one inference on CPU and on GPU.
//! Set `BLUE_ONYX_SKIP_OPENVINO_TESTS=1` to skip (e.g. offline CI).
//! Run with `cargo test --test backend_cpu -- --nocapture` to see timings.

use blue_onyx_openvino::backend::{CoreOptions, LoadRequest, OvBackend, OvCore};
use blue_onyx_openvino::setup_openvino;
use std::path::{Path, PathBuf};

const LOCAL_ZIP: &str = r"C:\Users\Pat\AppData\Local\Temp\claude\c--Users-Pat-Desktop-Projects-BlueOnyx\95393863-66db-44ad-ac05-c0501ff15827\scratchpad\ovzip\ov_win.zip";
const MODEL_URL: &str = "https://huggingface.co/xnorpx/blue-onyx-yolo5/resolve/main/IPcam-general";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ensure_runtime(dir: &Path) {
    if dir.is_dir() {
        return;
    }
    let local = PathBuf::from(LOCAL_ZIP);
    let archive = (cfg!(windows) && local.is_file()).then_some(local);
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

    let mut core = OvCore::new(&CoreOptions {
        cache_dir: Some(root.join("cache")),
        intra_threads: 0,
        openvino_dir: Some(ov_dir),
    })
    .expect("OvCore::new");
    println!("OpenVINO {}", core.openvino_version());
    println!("available devices: {:?}", core.available_devices());
    for d in core.available_devices().to_vec() {
        println!("  {d}: {}", core.device_full_name(&d));
    }

    // CPU, no fallback.
    let loaded = core
        .load(&LoadRequest {
            path: model_path.clone(),
            device: "CPU".into(),
            gpu_precision: None,
            allow_cpu_fallback: false,
        })
        .expect("load on CPU");
    println!(
        "CPU: {:?} compile {} ms, inputs {:?}, outputs {:?}",
        loaded.device, loaded.compile_ms, loaded.inputs, loaded.outputs
    );
    assert_eq!(loaded.image_input, "images");
    assert_eq!(loaded.input_size, (640, 640));
    assert_eq!(loaded.inputs.len(), 1);
    assert_eq!(loaded.inputs[0].shape, vec![1, 3, 640, 640]);
    assert_eq!(loaded.outputs.len(), 1);
    assert_eq!(loaded.outputs[0].shape, vec![1, 25200, 8]);
    assert_eq!(loaded.device.actual, "CPU");

    let mut backend = OvBackend::new(loaded).expect("backend");
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
        let loaded = core
            .load(&LoadRequest {
                path: model_path.clone(),
                device: "GPU".into(),
                gpu_precision: None,
                allow_cpu_fallback: true,
            })
            .expect("load on GPU (with fallback)");
        println!(
            "GPU run {run}: {:?} compile {} ms, provider '{}'",
            loaded.device,
            loaded.compile_ms,
            loaded.device.execution_provider()
        );
        let mut backend = OvBackend::new(loaded).expect("backend GPU");
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
}
