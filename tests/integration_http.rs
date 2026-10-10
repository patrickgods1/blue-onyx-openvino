//! End-to-end HTTP test: real OpenVINO, real model, real multipart upload.
//!
//! Skips (passes with a message) unless `models/IPcam-general.onnx` and `openvino/` exist
//! under the repo root. Set `BLUE_ONYX_TEST_ROOT` to point at another checkout that has them
//! (e.g. when running from a git worktree).

use blue_onyx_prism::api::VisionDetectionResponse;
use blue_onyx_prism::config::{Config, ModelConfig};
use blue_onyx_prism::metrics::Metrics;
use blue_onyx_prism::model::ModelFamilyKind;
use blue_onyx_prism::registry::ModelRegistry;
use blue_onyx_prism::server::{self, AppState};
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
async fn detection_over_http() {
    let root = repo_root();
    let model = root.join("models").join("IPcam-general.onnx");
    let ov_dir = root.join("openvino");
    if !model.is_file() || !ov_dir.is_dir() {
        eprintln!(
            "skipping detection_over_http: need {} and {}",
            model.display(),
            ov_dir.display()
        );
        return;
    }
    let image = std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_IMAGE))
        .expect("test image");

    let cache = std::env::temp_dir().join(format!("bo_it_cache_{}", std::process::id()));
    let config = Config {
        force_cpu: true,
        cache_dir: cache.to_string_lossy().to_string(),
        openvino_dir: Some(ov_dir),
        models: vec![ModelConfig {
            path: model,
            family: ModelFamilyKind::Yolo5,
            ..Default::default()
        }],
        ..Default::default()
    };
    let token = CancellationToken::new();
    let metrics = Arc::new(Metrics::new(blue_onyx_prism::VERSION));
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

    // Wait for the model to compile (GET / shows the state).
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        if let Ok(r) = client.get(format!("{base}/")).send().await
            && let Ok(body) = r.text().await
        {
            if body.contains("<td>Ready</td>") {
                break;
            }
            assert!(!body.contains("Failed:"), "model failed to load:\n{body}");
        }
        assert!(
            Instant::now() < deadline,
            "model did not become ready in time"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    eprintln!("default: {r:?}");
    assert!(r.success, "detection failed: {r:?}");
    assert!(!r.predictions.is_empty(), "no predictions: {r:?}");
    assert_eq!(r.count as usize, r.predictions.len());
    assert_eq!(r.command, "detect");

    let r = post_image(
        &client,
        &format!("{base}/v1/vision/custom/IPcam-general"),
        &image,
    )
    .await;
    assert!(r.success, "custom detection failed: {r:?}");
    assert_eq!(r.command, "custom");

    let r = post_image(&client, &format!("{base}/v1/vision/custom/nope"), &image).await;
    assert!(!r.success);
    assert!(r.error.unwrap_or_default().contains("Unknown model"));

    let list: serde_json::Value = client
        .post(format!("{base}/v1/vision/custom/list"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        list["models"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m == "IPcam-general")
    );

    token.cancel();
    server.await.unwrap().unwrap();
    if let Ok(reg) = Arc::try_unwrap(registry) {
        reg.shutdown();
    }
}
