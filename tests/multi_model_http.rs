//! Phase-3 milestone as a test: several models served concurrently from one process.
//!
//! `yolo26s.xml` (family yolo26, device "GPU", which falls back to CPU where there is no GPU
//! plugin, e.g. macOS) + `IPcam-general.onnx` (family yolo5, device "CPU"), plus `yolo26n.xml`
//! as a `lazy` third model when present. `default_model` = "yolo26s".
//!
//! Skips (passes with a message) when `openvino/` or the two required models are missing under
//! the repo root. Set `BLUE_ONYX_TEST_ROOT` to point at another checkout that has them.

use blue_onyx_prism::api::VisionDetectionResponse;
use blue_onyx_prism::config::{Config, ModelConfig};
use blue_onyx_prism::metrics::Metrics;
use blue_onyx_prism::model::ModelFamilyKind;
use blue_onyx_prism::registry::ModelRegistry;
use blue_onyx_prism::server::{self, AppState};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

const TEST_IMAGE: &str = "tests/data/dog_bike_car.jpg";
const PER_MODEL: usize = 8;
const IPCAM: &str = "IPcam-general";
const IPCAM_CLASSES: [&str; 3] = ["person", "vehicle", "unknown"];

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
    assert!(resp.status().is_success(), "{url}: HTTP {}", resp.status());
    resp.json().await.expect("detection JSON")
}

async fn get_stats(client: &reqwest::Client, base: &str) -> serde_json::Value {
    client
        .get(format!("{base}/stats.json"))
        .send()
        .await
        .expect("GET /stats.json")
        .json()
        .await
        .expect("stats JSON")
}

async fn get_prometheus(client: &reqwest::Client, base: &str) -> String {
    client
        .get(format!("{base}/prometheus"))
        .send()
        .await
        .expect("GET /prometheus")
        .text()
        .await
        .expect("prometheus text")
}

/// `/stats.json` row for `name`.
fn stats_model<'a>(stats: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    stats["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|m| m["name"] == name)
        .unwrap_or_else(|| panic!("no /stats.json row for {name}: {stats}"))
}

/// Value of the sample `blue_onyx_prism_<metric>{model="<model>"}`.
fn prom_value(text: &str, metric: &str, model: &str) -> f64 {
    let prefix = format!("blue_onyx_prism_{metric}{{model=\"{model}\"}} ");
    text.lines()
        .find_map(|l| l.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no {prefix:?} in:\n{text}"))
        .parse()
        .expect("numeric sample")
}

/// The `model_info` line for `model`.
fn prom_info<'a>(text: &'a str, model: &str) -> &'a str {
    let prefix = format!("blue_onyx_prism_model_info{{model=\"{model}\",");
    text.lines()
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no model_info for {model} in:\n{text}"))
}

#[tokio::test]
async fn multiple_models_served_concurrently() {
    let root = repo_root();
    let ov_dir = root.join("openvino");
    let models_dir = root.join("models");
    let yolo26s = models_dir.join("yolo26s.xml");
    let ipcam = models_dir.join("IPcam-general.onnx");
    let yolo26n = models_dir.join("yolo26n.xml");
    if !ov_dir.is_dir() || !yolo26s.is_file() || !ipcam.is_file() {
        eprintln!(
            "skipping multiple_models_served_concurrently: need {}, {} and {}",
            ov_dir.display(),
            yolo26s.display(),
            ipcam.display()
        );
        return;
    }
    let image = std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_IMAGE))
        .expect("test image");

    let mut models = vec![
        ModelConfig {
            name: Some("yolo26s".into()),
            path: yolo26s,
            family: ModelFamilyKind::Yolo26,
            device: Some("GPU".into()),
            ..Default::default()
        },
        // No explicit name: served as the file stem "IPcam-general".
        ModelConfig {
            path: ipcam,
            family: ModelFamilyKind::Yolo5,
            device: Some("CPU".into()),
            ..Default::default()
        },
    ];
    let lazy_name = yolo26n.is_file().then(|| "yolo26n-lazy".to_string());
    if let Some(name) = &lazy_name {
        models.push(ModelConfig {
            name: Some(name.clone()),
            path: yolo26n,
            family: ModelFamilyKind::Yolo26,
            device: Some("CPU".into()),
            lazy: true,
            ..Default::default()
        });
    } else {
        eprintln!("yolo26n.xml absent: skipping the lazy-model part");
    }
    let names: Vec<String> = models.iter().map(|m| m.effective_name()).collect();
    let eager = ["yolo26s", IPCAM];

    let cache = std::env::temp_dir().join(format!("bo_multi_cache_{}", std::process::id()));
    let config = Config {
        cache_dir: cache.to_string_lossy().to_string(),
        openvino_dir: Some(ov_dir),
        default_model: Some("yolo26s".into()),
        models,
        ..Default::default()
    };
    let token = CancellationToken::new();
    let metrics = Arc::new(Metrics::new(blue_onyx_prism::VERSION));
    let t_start = Instant::now();
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

    // Wait for the eager models; the lazy one must stay unloaded meanwhile.
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if let Ok(r) = client.get(format!("{base}/stats.json")).send().await
            && let Ok(stats) = r.json::<serde_json::Value>().await
        {
            for m in stats["models"].as_array().unwrap() {
                let s = m["state"].as_str().unwrap_or_default();
                assert!(!s.starts_with("Failed"), "a model failed to load: {m}");
            }
            if eager
                .iter()
                .all(|n| stats_model(&stats, n)["state"] == "Ready")
            {
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "models did not become ready in time"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    eprintln!("eager models ready after {:?}", t_start.elapsed());

    let stats = get_stats(&client, &base).await;
    let prom = get_prometheus(&client, &base).await;
    for n in eager {
        let row = stats_model(&stats, n);
        let device = row["device"].as_str().expect("device after load");
        let provider = row["executionProvider"].as_str().unwrap();
        assert!(provider.starts_with("OpenVINO "), "{n}: {row}");
        // Prometheus reflects the device/provider the worker actually got.
        let expect = format!(
            "blue_onyx_prism_model_info{{model=\"{n}\",device=\"{device}\",provider=\"{provider}\",state=\"ready\"}} 1"
        );
        assert_eq!(prom_info(&prom, n), expect);
        assert_eq!(prom_value(&prom, "model_ready", n), 1.0);
        assert_eq!(prom_value(&prom, "queue_length", n), 0.0);
        let cap = prom_value(&prom, "queue_capacity", n);
        assert!(cap >= 1.0, "{n}: queue_capacity {cap}");
        assert_eq!(cap, row["queueCapacity"].as_f64().unwrap());
        eprintln!("{n}: device={device} provider={provider} queue_capacity={cap}");
    }
    assert_eq!(stats_model(&stats, IPCAM)["device"], "CPU");
    if let Some(lazy) = &lazy_name {
        assert_eq!(stats_model(&stats, lazy)["state"], "Initializing");
        assert!(
            prom_info(&prom, lazy).ends_with("state=\"lazy\"} 1"),
            "{prom}"
        );
        assert_eq!(prom_value(&prom, "model_ready", lazy), 0.0);
    }

    // custom/list: every configured name, in config order.
    let list: serde_json::Value = client
        .post(format!("{base}/v1/vision/custom/list"))
        .send()
        .await
        .expect("custom/list")
        .json()
        .await
        .expect("list JSON");
    assert_eq!(list["success"], true, "{list}");
    assert_eq!(list["moduleId"], "ObjectDetectionPrism");
    assert_eq!(list["models"], serde_json::json!(names), "{list}");

    let mut sent: HashMap<&str, u64> = HashMap::new();

    // Default route goes to default_model (yolo26s).
    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    assert!(r.success, "default route: {r:?}");
    assert!(r.predictions.iter().any(|p| p.label == "dog"), "{r:?}");
    *sent.entry("yolo26s").or_default() += 1;
    let stats = get_stats(&client, &base).await;
    assert_eq!(stats_model(&stats, "yolo26s")["requests"], 1);
    assert_eq!(stats_model(&stats, IPCAM)["requests"], 0);

    // Unknown model: HTTP 200, success:false, counted nowhere.
    let r = post_image(&client, &format!("{base}/v1/vision/custom/nope"), &image).await;
    assert!(!r.success);
    assert!(
        r.error
            .as_deref()
            .unwrap_or_default()
            .contains("Unknown model"),
        "{r:?}"
    );

    // Concurrent, interleaved requests to both eager models. The IPcam requests alternate between
    // the exact stem and a differently cased name with extension (Blue Iris style).
    let t = Instant::now();
    let mut futs = Vec::new();
    for i in 0..PER_MODEL {
        let ipcam_path = if i % 2 == 0 {
            "IPcam-General.onnx"
        } else {
            "ipcam-general"
        };
        for (model, path) in [("yolo26s", "yolo26s"), (IPCAM, ipcam_path)] {
            let url = format!("{base}/v1/vision/custom/{path}");
            let (client, image) = (client.clone(), image.clone());
            futs.push(async move { (model, post_image(&client, &url, &image).await) });
            *sent.entry(model).or_default() += 1;
        }
    }
    let results = futures_util::future::join_all(futs).await;
    eprintln!(
        "{} concurrent requests over 2 models took {:?}",
        results.len(),
        t.elapsed()
    );
    for (model, r) in &results {
        assert!(r.success, "{model}: {r:?}");
        assert_eq!(r.count as usize, r.predictions.len());
        assert!(r.count > 0, "{model}: nothing detected: {r:?}");
        match *model {
            "yolo26s" => assert!(r.predictions.iter().any(|p| p.label == "dog"), "{r:?}"),
            _ => {
                assert!(
                    r.predictions
                        .iter()
                        .all(|p| IPCAM_CLASSES.contains(&p.label.as_str())),
                    "{r:?}"
                );
                // The car (and the bicycle) are "vehicle" for IPcam-general.
                assert!(r.predictions.iter().any(|p| p.label == "vehicle"), "{r:?}");
            }
        }
        for p in &r.predictions {
            assert!(
                p.x_min < p.x_max && p.y_min < p.y_max,
                "{model}: bad box {p:?}"
            );
        }
    }
    let (y, i): (Vec<_>, Vec<_>) = results.iter().partition(|(m, _)| *m == "yolo26s");
    eprintln!("yolo26s sample: {:?}", y[0].1.predictions);
    eprintln!("{IPCAM} sample: {:?}", i[0].1.predictions);
    let avg = |v: &[&(&str, VisionDetectionResponse)]| {
        v.iter().map(|(_, r)| r.inferenceMs as f64).sum::<f64>() / v.len() as f64
    };
    eprintln!(
        "avg inferenceMs under load: yolo26s {:.1}, {IPCAM} {:.1}",
        avg(&y),
        avg(&i)
    );

    // Lazy model: Initializing -> Ready after its first request.
    if let Some(lazy) = &lazy_name {
        let t = Instant::now();
        let r = post_image(&client, &format!("{base}/v1/vision/custom/{lazy}"), &image).await;
        eprintln!(
            "{lazy}: first request (compile + infer) took {:?}",
            t.elapsed()
        );
        assert!(r.success, "{lazy}: {r:?}");
        assert!(r.predictions.iter().any(|p| p.label == "dog"), "{r:?}");
        *sent.entry(lazy.as_str()).or_default() += 1;
        let stats = get_stats(&client, &base).await;
        assert_eq!(stats_model(&stats, lazy)["state"], "Ready");
        let prom = get_prometheus(&client, &base).await;
        assert!(
            prom_info(&prom, lazy).ends_with("state=\"ready\"} 1"),
            "{prom}"
        );
        assert_eq!(prom_value(&prom, "model_ready", lazy), 1.0);
    }

    // Per-model counters match exactly what was sent; nothing was dropped.
    let stats = get_stats(&client, &base).await;
    let prom = get_prometheus(&client, &base).await;
    for n in &names {
        let want = sent.get(n.as_str()).copied().unwrap_or(0);
        let row = stats_model(&stats, n);
        assert_eq!(row["requests"], want, "{n}: {row}");
        assert_eq!(row["dropped"], 0, "{n}: {row}");
        assert_eq!(prom_value(&prom, "requests_total", n), want as f64);
        assert_eq!(prom_value(&prom, "dropped_total", n), 0.0);
        assert_eq!(prom_value(&prom, "queue_length", n), 0.0);
        assert_eq!(row["inferenceMs"]["count"], want, "{n}: {row}");
    }

    token.cancel();
    server.await.unwrap().unwrap();
    if let Ok(reg) = Arc::try_unwrap(registry) {
        reg.shutdown();
    }
    std::fs::remove_dir_all(&cache).ok();
}
