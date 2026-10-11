//! Startup provisioning (phase 7.3) with a local HTTP fixture instead of the real catalog:
//! a model fetched from an injected catalog entry goes Initializing (with download progress) to
//! Ready in the same registry generation; with `auto_download: false` it is Failed with the
//! command to run. Plus the "downloadable" device state.
//!
//! The load test needs `models/IPcam-general.{onnx,yaml}` and `openvino/` in the repo (served
//! by the fixture / used as the runtime) and skips with a message otherwise.

use blue_onyx_prism::backend::detect::{GpuAdapter, GpuVendor, HardwareInfo};
use blue_onyx_prism::backend::select::{RuntimeProbe, select};
use blue_onyx_prism::config::{Config, ModelConfig};
use blue_onyx_prism::metrics::Metrics;
use blue_onyx_prism::model::ModelFamilyKind;
use blue_onyx_prism::registry::ModelRegistry;
use blue_onyx_prism::resources::catalog::{
    Layout, OPENVINO_RUNTIME_ID, Part, Provides, Resource, ResourceKind,
};
use blue_onyx_prism::resources::provision::{ProvisionOptions, Provisioner};
use blue_onyx_prism::resources::resolve::{FETCH_FOR_CONFIG, Installed, annotate_selection};
use blue_onyx_prism::startup::ModelState;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn sha256(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Serve `files` (name -> bytes) over HTTP on 127.0.0.1, waiting `delay` before each body so
/// the download is observable.
fn serve(files: Vec<(String, Vec<u8>)>, delay: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let files = files.clone();
            std::thread::spawn(move || {
                let mut head = Vec::new();
                let mut b = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    if s.read(&mut b).map_or(true, |n| n == 0) {
                        return;
                    }
                    head.push(b[0]);
                }
                let head = String::from_utf8_lossy(&head).to_string();
                let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
                let Some((_, body)) = files.iter().find(|(n, _)| path == format!("/{n}")) else {
                    let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                    return;
                };
                std::thread::sleep(delay);
                let _ = write!(
                    s,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(body);
            });
        }
    });
    base
}

/// A model resource `fixture-model` whose two files are served by the fixture at `base`.
fn fixture_resource(base: &str, onnx: &[u8], yaml: &[u8]) -> &'static Resource {
    let part = |name: &str, data: &[u8]| Part {
        url: leak(format!("{base}/{name}")),
        sha256: leak(sha256(data)),
        size: data.len() as u64,
        file_name: leak(name.to_string()),
        archive: None,
        layout: Layout::File,
    };
    Box::leak(Box::new(Resource {
        id: "model:fixture-model",
        kind: ResourceKind::Model,
        version: "1",
        platform: None,
        parts: Box::leak(
            vec![
                part("fixture-model.onnx", onnx),
                part("fixture-model.yaml", yaml),
            ]
            .into_boxed_slice(),
        ),
        dest: "models",
        provides: Provides::Model {
            name: "fixture-model",
            family: ModelFamilyKind::Yolo5,
        },
        title: "Model fixture-model",
        description: "test fixture",
        license: "AGPL-3.0",
    }))
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bop-prov-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn config(root: &Path, openvino_dir: Option<PathBuf>) -> Config {
    Config {
        download_dir: Some(root.to_path_buf()),
        openvino_dir,
        device: "openvino:cpu".into(),
        cache_dir: String::new(),
        models: vec![ModelConfig {
            path: "models/fixture-model.onnx".into(),
            family: ModelFamilyKind::Yolo5,
            ..ModelConfig::default()
        }],
        ..Config::default()
    }
}

#[test]
fn model_download_goes_initializing_to_ready_without_restart() {
    let models = repo().join("models");
    let ov = repo().join("openvino");
    let (onnx, yaml) = (
        models.join("IPcam-general.onnx"),
        models.join("IPcam-general.yaml"),
    );
    if !onnx.is_file() || !yaml.is_file() || !ov.is_dir() {
        eprintln!(
            "skipping: need {}, {} and {}",
            onnx.display(),
            yaml.display(),
            ov.display()
        );
        return;
    }
    let (onnx, yaml) = (std::fs::read(onnx).unwrap(), std::fs::read(yaml).unwrap());
    let base = serve(
        vec![
            ("fixture-model.onnx".into(), onnx.clone()),
            ("fixture-model.yaml".into(), yaml.clone()),
        ],
        Duration::from_millis(1500),
    );
    let res = fixture_resource(&base, &onnx, &yaml);
    let root = tmp("ready");
    let cfg = config(&root, Some(ov));

    let prov = Provisioner::with_options(ProvisionOptions {
        extra_models: vec![res],
        ..ProvisionOptions::default()
    });
    let plan = prov.prepare(&cfg, None);
    let ids: Vec<&str> = plan.resolution.needs.iter().map(|n| n.id()).collect();
    assert_eq!(
        ids,
        ["model:fixture-model"],
        "OpenVINO comes from openvino_dir"
    );
    assert!(plan.wait_for("fixture-model").is_some());

    let metrics = Metrics::new("test");
    let token = CancellationToken::new();
    let reg = ModelRegistry::start_with(&cfg, &metrics, token.clone(), Some(&plan)).unwrap();
    let w = reg.by_name("fixture-model").unwrap();

    // Initializing with download progress first...
    let started = Instant::now();
    let mut detail = None;
    while started.elapsed() < Duration::from_secs(10) {
        if let Some(d) = w.state.detail() {
            detail = Some(d);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let detail = detail.expect("a download progress message");
    assert!(detail.contains("Model fixture-model"), "{detail}");
    assert_eq!(w.state.get(), ModelState::Initializing);

    // ...then Ready in the same registry (no new generation).
    while !w.state.is_ready() && started.elapsed() < Duration::from_secs(120) {
        if let ModelState::Failed(m) = w.state.get() {
            panic!("model failed: {m}");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(w.state.is_ready(), "state: {}", w.state.describe());
    assert_eq!(w.state.detail(), None);
    assert!(root.join("models/fixture-model.onnx").is_file());
    assert!(root.join("models/.fixture-model.installed.json").is_file());

    // The next generation needs nothing.
    let again = prov.prepare(&cfg, None);
    assert!(again.resolution.needs.is_empty());

    token.cancel();
    reg.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn auto_download_off_fails_with_the_command() {
    let res = fixture_resource("http://127.0.0.1:9", b"onnx", b"yaml");
    let root = tmp("manual");
    let mut cfg = config(&root, None);
    cfg.auto_download = false;
    let prov = Provisioner::with_options(ProvisionOptions {
        extra_models: vec![res],
        ..ProvisionOptions::default()
    });
    let plan = prov.prepare(&cfg, None);
    assert!(plan.has_needs());
    assert!(prov.manager().is_none(), "nothing is downloaded");
    let metrics = Metrics::new("test");
    let token = CancellationToken::new();
    // Starts even when no runtime is installed here: the models are reported, not fatal.
    let reg = ModelRegistry::start_with(&cfg, &metrics, token.clone(), Some(&plan)).unwrap();
    let w = reg.by_name("fixture-model").unwrap();
    match w.state.get() {
        ModelState::Failed(msg) => {
            assert!(msg.contains(FETCH_FOR_CONFIG), "{msg}");
            assert!(msg.contains("Model fixture-model"), "{msg}");
        }
        other => panic!("{other:?}"),
    }
    token.cancel();
    reg.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn downloadable_device_state() {
    // Fresh macOS: nothing runnable; CoreML and OpenVINO CPU are downloadable; auto stays empty.
    let mac = HardwareInfo::new(
        "macos",
        "aarch64",
        vec![GpuAdapter {
            vendor: GpuVendor::Apple,
            name: "Apple M2 GPU".into(),
            vram_mb: 0,
            index: 0,
            discrete: false,
            cuda: None,
        }],
    );
    let cfg = Config::default();
    let none = RuntimeProbe {
        openvino: blue_onyx_prism::backend::select::OpenVinoProbe::unavailable("missing"),
        ort: blue_onyx_prism::backend::select::OrtProbe::unavailable("missing"),
    };
    let inst = Installed {
        ort_in_build: true,
        ..Installed::default()
    };
    let mut sel = select(&mac, &none, true);
    annotate_selection(&mut sel, &cfg, &mac, &inst, true);
    assert!(sel.auto.is_empty(), "auto only holds runnable options");
    let coreml = sel
        .options
        .iter()
        .find(|o| o.spec.to_string() == "ort:coreml")
        .unwrap();
    assert!(coreml.is_downloadable() && !coreml.runnable);
    assert_eq!(coreml.status(), "download");
    let offer = coreml.download.as_ref().unwrap();
    assert_eq!(offer.resources, ["onnxruntime-coreml"]);
    assert!(
        offer
            .summary
            .starts_with("will download ONNX Runtime CoreML"),
        "{}",
        offer.summary
    );
    assert!(!offer.large);
    let ov = sel
        .options
        .iter()
        .find(|o| o.spec.to_string() == "openvino:cpu")
        .unwrap();
    assert_eq!(
        ov.download.as_ref().unwrap().resources,
        [OPENVINO_RUNTIME_ID]
    );
    let text = blue_onyx_prism::backend::select::format_options(&sel);
    assert!(text.contains("download"), "{text}");
    let json = serde_json::to_value(&sel).unwrap();
    assert_eq!(
        json["options"][0]["download"]["resources"][0],
        "onnxruntime-coreml"
    );

    // Windows NVIDIA with OpenVINO installed: OpenVINO CPU stays runnable (untouched); CUDA is
    // downloadable and flagged large (CUDA libraries).
    let win = HardwareInfo::new(
        "windows",
        "x86_64",
        vec![GpuAdapter {
            vendor: GpuVendor::Nvidia,
            name: "NVIDIA GeForce RTX 3060".into(),
            vram_mb: 12288,
            index: 0,
            discrete: true,
            cuda: None,
        }],
    );
    let probe = RuntimeProbe::openvino_only(&["CPU".to_string()]);
    let inst = Installed {
        ort_in_build: true,
        probe: Some(probe.clone()),
        ..Installed::default()
    }
    .with_resources(&[OPENVINO_RUNTIME_ID]);
    let mut sel = select(&win, &probe, true);
    let auto_before = sel.auto.clone();
    annotate_selection(&mut sel, &cfg, &win, &inst, true);
    assert_eq!(sel.auto, auto_before);
    let cpu = sel
        .options
        .iter()
        .find(|o| o.spec.to_string() == "openvino:cpu")
        .unwrap();
    assert!(cpu.runnable && cpu.download.is_none());
    let cuda = sel
        .options
        .iter()
        .find(|o| o.spec.to_string() == "ort:cuda:0")
        .unwrap();
    let offer = cuda.download.as_ref().unwrap();
    assert_eq!(offer.resources, ["onnxruntime-cuda", "nvidia-cuda-libs"]);
    assert!(offer.large);
    assert!(
        offer.summary.contains("needs allow_large_downloads"),
        "{}",
        offer.summary
    );
}
