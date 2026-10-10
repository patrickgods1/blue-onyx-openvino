//! End-to-end HTTP test for the phase-2 model families: YOLO26 (IR from
//! `scripts/export_yolo26.py`, both explicit and `auto` family) and RT-DETRv2 (ONNX), served
//! together on CPU and queried through `/v1/vision/custom/{model}`.
//!
//! Each model is used only if present under `models/` (`yolo26n.xml`, `yolo26s.xml`,
//! `rt-detrv2-s.onnx`); the test skips (passes with a message) when none are, or when `openvino/`
//! is missing. Set `BLUE_ONYX_TEST_ROOT` to point at another checkout that has them.

use blue_onyx_openvino::api::VisionDetectionResponse;
use blue_onyx_openvino::config::{Config, ModelConfig};
use blue_onyx_openvino::metrics::Metrics;
use blue_onyx_openvino::model::ModelFamilyKind;
use blue_onyx_openvino::registry::ModelRegistry;
use blue_onyx_openvino::server::{self, AppState};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

const TEST_IMAGE: &str = "tests/data/dog_bike_car.jpg";

fn repo_root() -> PathBuf {
    std::env::var_os("BLUE_ONYX_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .expect("free port")
}

async fn post_image(client: &reqwest::Client, url: &str, bytes: &[u8]) -> VisionDetectionResponse {
    let part = reqwest::multipart::Part::bytes(bytes.to_vec())
        .file_name("dog_bike_car.jpg")
        .mime_str("image/jpeg")
        .unwrap();
    let form = reqwest::multipart::Form::new()
        .part("image", part)
        .text("min_confidence", "0.4");
    let resp = client.post(url).multipart(form).send().await.expect("POST");
    assert!(resp.status().is_success(), "HTTP {}", resp.status());
    resp.json().await.expect("detection JSON")
}

#[tokio::test]
async fn yolo26_and_rtdetr_over_http() {
    let root = repo_root();
    let ov_dir = root.join("openvino");
    let models_dir = root.join("models");
    let candidates = [
        ("yolo26s", "yolo26s.xml", ModelFamilyKind::Yolo26),
        ("yolo26n-auto", "yolo26n.xml", ModelFamilyKind::Auto),
        ("rt-detrv2-s", "rt-detrv2-s.onnx", ModelFamilyKind::RtDetr),
    ];
    let models: Vec<ModelConfig> = candidates
        .iter()
        .filter(|(_, file, _)| models_dir.join(file).is_file())
        .map(|(name, file, family)| ModelConfig {
            name: Some(name.to_string()),
            path: models_dir.join(file),
            family: *family,
            ..Default::default()
        })
        .collect();
    if models.is_empty() || !ov_dir.is_dir() {
        eprintln!(
            "skipping yolo26_and_rtdetr_over_http: need {} and one of {:?} in {}",
            ov_dir.display(),
            candidates.iter().map(|c| c.1).collect::<Vec<_>>(),
            models_dir.display()
        );
        return;
    }
    let names: Vec<String> = models.iter().map(|m| m.name.clone().unwrap()).collect();
    let image = std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_IMAGE))
        .expect("test image");

    let cache = std::env::temp_dir().join(format!("bo_fam_cache_{}", std::process::id()));
    let config = Config {
        force_cpu: true,
        cache_dir: cache.to_string_lossy().to_string(),
        openvino_dir: Some(ov_dir),
        default_model: names.first().cloned(),
        models,
        ..Default::default()
    };
    let token = CancellationToken::new();
    let metrics = Arc::new(Metrics::new(blue_onyx_openvino::VERSION));
    let registry =
        Arc::new(ModelRegistry::start(&config, &metrics, token.clone()).expect("registry"));
    let config_path = std::env::temp_dir().join(format!(
        "bo_it_config_{}_{}.json",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let state = Arc::new(AppState::new(
        registry.clone(),
        metrics,
        config,
        config_path,
    ));
    let port = free_port();
    let server = tokio::spawn(server::serve(state, port, token.clone()));
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    // Wait until every model is compiled (GET / shows one state cell per model).
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if let Ok(r) = client.get(format!("{base}/")).send().await
            && let Ok(body) = r.text().await
        {
            assert!(!body.contains("Failed:"), "a model failed to load:\n{body}");
            if body.matches("<td>Ready</td>").count() == names.len() {
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "models did not become ready in time"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    for name in &names {
        let r = post_image(&client, &format!("{base}/v1/vision/custom/{name}"), &image).await;
        eprintln!("{name}: {r:?}");
        assert!(r.success, "{name}: detection failed: {r:?}");
        for want in ["dog", "bicycle"] {
            assert!(
                r.predictions.iter().any(|p| p.label == want),
                "{name}: no '{want}' in {:?}",
                r.predictions
            );
        }
        for p in &r.predictions {
            assert!(
                p.x_min < p.x_max && p.y_min < p.y_max,
                "{name}: bad box {p:?}"
            );
            assert!(p.confidence >= 0.4, "{name}: below min_confidence {p:?}");
        }
    }

    token.cancel();
    server.await.unwrap().unwrap();
    if let Ok(reg) = Arc::try_unwrap(registry) {
        reg.shutdown();
    }
    std::fs::remove_dir_all(&cache).ok();
}
