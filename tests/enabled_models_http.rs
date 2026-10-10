//! `enabled: false` models stay in the config but are not loaded or served; with every model
//! disabled the HTTP server (and its web UI) still comes up.
//!
//! Switching models through the Models card (`POST /config/models`) and the in-process restart
//! (`runner::run_server_on`, as used by the binary and the Windows service) is covered too.
//!
//! Real OpenVINO: skips (passes with a message) unless `models/IPcam-general.onnx` (and, for the
//! switch test, `models/rt-detrv2-s.onnx`) and `openvino/` exist under the repo root (or
//! `BLUE_ONYX_TEST_ROOT`).

use blue_onyx_prism::api::VisionDetectionResponse;
use blue_onyx_prism::cli::{self, LogReloadHandle};
use blue_onyx_prism::config::{Config, LogLevel, ModelConfig};
use blue_onyx_prism::metrics::Metrics;
use blue_onyx_prism::model::ModelFamilyKind;
use blue_onyx_prism::registry::ModelRegistry;
use blue_onyx_prism::runner;
use blue_onyx_prism::server::{self, AppState};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
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
    let form = reqwest::multipart::Form::new().part("image", part);
    let resp = client.post(url).multipart(form).send().await.expect("POST");
    assert!(resp.status().is_success(), "HTTP {}", resp.status());
    resp.json().await.expect("detection JSON")
}

struct Running {
    base: String,
    token: CancellationToken,
    server: tokio::task::JoinHandle<anyhow::Result<()>>,
    registry: Arc<ModelRegistry>,
}

impl Running {
    async fn start(config: Config) -> Self {
        let token = CancellationToken::new();
        let metrics = Arc::new(Metrics::new(blue_onyx_prism::VERSION));
        let registry =
            Arc::new(ModelRegistry::start(&config, &metrics, token.clone()).expect("registry"));
        let config_path = std::env::temp_dir().join(format!(
            "bo_en_config_{}_{}.json",
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
        Self {
            base: format!("http://127.0.0.1:{port}"),
            token,
            server,
            registry,
        }
    }

    async fn stop(self) {
        self.token.cancel();
        self.server.await.unwrap().unwrap();
        if let Ok(reg) = Arc::try_unwrap(self.registry) {
            reg.shutdown();
        }
    }
}

/// GET `path`, retrying until the listener is up.
async fn get_text(client: &reqwest::Client, url: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(r) = client.get(url).send().await
            && let Ok(body) = r.text().await
        {
            return body;
        }
        assert!(Instant::now() < deadline, "server did not come up");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn disabled_models_are_not_served() {
    let root = repo_root();
    let model = root.join("models").join("IPcam-general.onnx");
    let ov_dir = root.join("openvino");
    if !model.is_file() || !ov_dir.is_dir() {
        eprintln!(
            "skipping disabled_models_are_not_served: need {} and {}",
            model.display(),
            ov_dir.display()
        );
        return;
    }
    let image = std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_IMAGE))
        .expect("test image");
    let cache = std::env::temp_dir().join(format!("bo_en_cache_{}", std::process::id()));
    let base_config = Config {
        force_cpu: true,
        cache_dir: cache.to_string_lossy().to_string(),
        openvino_dir: Some(ov_dir),
        ..Default::default()
    };
    let client = reqwest::Client::new();

    // One enabled model plus a disabled one that `default_model` points at (its file does not
    // even exist: disabled entries are never touched).
    let config = Config {
        default_model: Some("off-model".into()),
        models: vec![
            ModelConfig {
                name: Some("off-model".into()),
                path: root.join("models").join("does-not-exist.onnx"),
                enabled: false,
                ..Default::default()
            },
            ModelConfig {
                path: model.clone(),
                family: ModelFamilyKind::Yolo5,
                ..Default::default()
            },
        ],
        ..base_config.clone()
    };
    let t0 = Instant::now();
    let run = Running::start(config).await;
    assert_eq!(run.registry.names(), ["IPcam-general"]);
    assert_eq!(
        run.registry.default_model().map(|w| w.name.as_str()),
        Some("IPcam-general")
    );
    assert!(run.registry.by_name("off-model").is_none());

    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        let body = get_text(&client, &format!("{}/", run.base)).await;
        if body.contains("<td>Ready</td>") {
            assert!(!body.contains("off-model"), "{body}");
            break;
        }
        assert!(!body.contains("Failed:"), "model failed to load:\n{body}");
        assert!(
            Instant::now() < deadline,
            "model did not become ready in time"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    eprintln!("IPcam-general ready after {:?}", t0.elapsed());

    let r = post_image(
        &client,
        &format!("{}/v1/vision/detection", run.base),
        &image,
    )
    .await;
    assert!(r.success, "detection failed: {r:?}");
    assert!(!r.predictions.is_empty(), "no predictions: {r:?}");
    let r = post_image(
        &client,
        &format!("{}/v1/vision/custom/off-model", run.base),
        &image,
    )
    .await;
    assert!(!r.success);
    assert!(r.error.unwrap_or_default().contains("Unknown model"));
    let list: serde_json::Value = client
        .post(format!("{}/v1/vision/custom/list", run.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list["models"], serde_json::json!(["IPcam-general"]));

    // The config page still lists the disabled model, unchecked.
    let cfg = get_text(&client, &format!("{}/config", run.base)).await;
    assert!(
        cfg.contains("name=\"enabled\" value=\"off-model\" aria-label=\"Load off-model\">"),
        "{cfg}"
    );
    assert!(cfg.contains(
        "name=\"default_model\" value=\"IPcam-general\" aria-label=\"Default IPcam-general\" checked"
    ));
    run.stop().await;

    // Every model disabled: no workers, but the HTTP server and UI are up.
    let config = Config {
        models: vec![ModelConfig {
            path: model,
            family: ModelFamilyKind::Yolo5,
            enabled: false,
            ..Default::default()
        }],
        ..base_config
    };
    let run = Running::start(config).await;
    assert!(run.registry.names().is_empty());
    assert!(run.registry.default_model().is_none());
    let home = get_text(&client, &format!("{}/", run.base)).await;
    assert!(home.contains("No models are enabled"), "{home}");
    let cfg = get_text(&client, &format!("{}/config", run.base)).await;
    assert!(cfg.contains("action=\"/config/models\""));
    let r = post_image(
        &client,
        &format!("{}/v1/vision/detection", run.base),
        &image,
    )
    .await;
    assert!(!r.success);
    assert!(
        r.error
            .as_deref()
            .unwrap_or_default()
            .contains("enable a model"),
        "{r:?}"
    );
    run.stop().await;
}

// ---------------------------------------------------------------------------------------------
// In-process restart through the production run loop

/// The global subscriber can be installed once per test binary; every server shares the handle.
fn log_handle() -> LogReloadHandle {
    static LOG: OnceLock<LogReloadHandle> = OnceLock::new();
    LOG.get_or_init(|| cli::init_logging(LogLevel::Warn, None).expect("init logging"))
        .clone()
}

/// `runner::run_server_on` (registry + HTTP generations, reload on `POST /config/restart`) on
/// its own thread and runtime, exactly like the binary.
struct Served {
    base: String,
    shutdown: CancellationToken,
    thread: std::thread::JoinHandle<anyhow::Result<()>>,
}

impl Served {
    fn start(config: Config, config_path: PathBuf) -> Self {
        config.save(&config_path).expect("write config");
        let base = format!("http://127.0.0.1:{}", config.port);
        let shutdown = CancellationToken::new();
        let token = shutdown.clone();
        let log = log_handle();
        let thread = std::thread::spawn(move || {
            let rt = runner::build_runtime()?;
            runner::run_server_on(&rt, config, config_path, log, token)
        });
        Self {
            base,
            shutdown,
            thread,
        }
    }

    fn stop(self) {
        self.shutdown.cancel();
        self.thread
            .join()
            .expect("server thread panicked")
            .expect("server result");
    }
}

/// Poll `check` every 100 ms until it returns true; panic after `timeout`.
async fn wait_until<F, Fut>(what: &str, timeout: Duration, mut check: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let deadline = Instant::now() + timeout;
    while !check().await {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// `/v1/vision/custom/list` model names, None while the server is down (between generations).
async fn list_models(client: &reqwest::Client, base: &str) -> Option<Vec<String>> {
    let v: serde_json::Value = client
        .post(format!("{base}/v1/vision/custom/list"))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    serde_json::from_value(v["models"].clone()).ok()
}

/// Home page body, None while the server is down.
async fn home(client: &reqwest::Client, base: &str) -> Option<String> {
    client
        .get(format!("{base}/"))
        .send()
        .await
        .ok()?
        .text()
        .await
        .ok()
}

/// Wait until the running generation serves exactly `names` and all of them are Ready.
async fn wait_serving(client: &reqwest::Client, base: &str, names: &[&str]) {
    let expected: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    wait_until(
        &format!("{names:?} to be served"),
        Duration::from_secs(60),
        || async { list_models(client, base).await.as_ref() == Some(&expected) },
    )
    .await;
    wait_until(
        &format!("{names:?} to be ready"),
        Duration::from_secs(300),
        || async {
            let body = home(client, base).await.unwrap_or_default();
            assert!(!body.contains("Failed:"), "model failed to load:\n{body}");
            body.matches("<td>Ready</td>").count() == names.len()
        },
    )
    .await;
}

async fn post_form(client: &reqwest::Client, url: &str, body: &str) -> String {
    let resp = client
        .post(url)
        .header("content-type", "application/x-www-form-urlencoded")
        .body(body.to_string())
        .send()
        .await
        .expect("POST form");
    assert!(resp.status().is_success(), "{url}: HTTP {}", resp.status());
    resp.text().await.expect("form response")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bo_en_{tag}_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn base_config(ov_dir: &Path) -> Config {
    Config {
        port: free_port(),
        force_cpu: true,
        cache_dir: std::env::temp_dir()
            .join(format!("bo_en_cache_{}", std::process::id()))
            .to_string_lossy()
            .to_string(),
        openvino_dir: Some(ov_dir.to_path_buf()),
        log_level: LogLevel::Warn,
        ..Default::default()
    }
}

fn labels(r: &VisionDetectionResponse) -> Vec<String> {
    r.predictions.iter().map(|p| p.label.clone()).collect()
}

#[tokio::test]
async fn switch_models_with_restart() {
    let root = repo_root();
    let ipcam = root.join("models").join("IPcam-general.onnx");
    let rtdetr = root.join("models").join("rt-detrv2-s.onnx");
    let ov_dir = root.join("openvino");
    if !ipcam.is_file() || !rtdetr.is_file() || !ov_dir.is_dir() {
        eprintln!(
            "skipping switch_models_with_restart: need {}, {} and {}",
            ipcam.display(),
            rtdetr.display(),
            ov_dir.display()
        );
        return;
    }
    let image = std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_IMAGE))
        .expect("test image");
    let config_path = temp_dir("switch").join("cfg.json");
    let config = Config {
        models: vec![
            ModelConfig {
                name: Some("IPcam-general".into()),
                path: ipcam,
                family: ModelFamilyKind::Yolo5,
                ..Default::default()
            },
            ModelConfig {
                name: Some("rt-detrv2-s".into()),
                path: rtdetr,
                family: ModelFamilyKind::RtDetr,
                enabled: false,
                ..Default::default()
            },
        ],
        ..base_config(&ov_dir)
    };
    let client = reqwest::Client::new();
    let t0 = Instant::now();
    let served = Served::start(config, config_path.clone());
    let base = served.base.clone();
    wait_serving(&client, &base, &["IPcam-general"]).await;
    eprintln!(
        "generation 1 (IPcam-general) ready after {:?}",
        t0.elapsed()
    );

    // Switch to rt-detrv2-s and restart in-process.
    let t1 = Instant::now();
    let page = post_form(
        &client,
        &format!("{base}/config/models"),
        "enabled=rt-detrv2-s&default_model=rt-detrv2-s&action=restart",
    )
    .await;
    assert!(page.contains("Restarting"), "{page}");
    wait_serving(&client, &base, &["rt-detrv2-s"]).await;
    eprintln!("generation 2 (rt-detrv2-s) ready after {:?}", t1.elapsed());

    let saved = Config::load(&config_path).unwrap();
    let flags: Vec<(String, bool)> = saved
        .models
        .iter()
        .map(|m| (m.effective_name(), m.enabled))
        .collect();
    assert_eq!(
        flags,
        [
            ("IPcam-general".to_string(), false),
            ("rt-detrv2-s".to_string(), true)
        ]
    );
    assert_eq!(saved.default_model.as_deref(), Some("rt-detrv2-s"));

    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    assert!(r.success, "detection failed: {r:?}");
    assert!(labels(&r).iter().any(|l| l == "dog"), "no dog: {r:?}");
    let r = post_image(
        &client,
        &format!("{base}/v1/vision/custom/IPcam-general"),
        &image,
    )
    .await;
    assert!(!r.success);
    assert!(
        r.error
            .as_deref()
            .unwrap_or_default()
            .contains("Unknown model"),
        "{r:?}"
    );

    // Switch back with "Save" only: the file changes, the running generation does not.
    let page = post_form(
        &client,
        &format!("{base}/config/models"),
        "enabled=IPcam-general&default_model=IPcam-general&action=save",
    )
    .await;
    assert!(page.contains("Restart the server to apply"), "{page}");
    let saved = Config::load(&config_path).unwrap();
    assert!(saved.models[0].enabled && !saved.models[1].enabled);
    assert_eq!(saved.default_model.as_deref(), Some("IPcam-general"));
    assert_eq!(
        list_models(&client, &base).await.as_deref(),
        Some(&["rt-detrv2-s".to_string()][..])
    );
    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    assert!(r.success && labels(&r).iter().any(|l| l == "dog"), "{r:?}");
    let r = post_image(
        &client,
        &format!("{base}/v1/vision/custom/IPcam-general"),
        &image,
    )
    .await;
    assert!(
        !r.success,
        "IPcam-general must not be served before a restart"
    );

    // ... until the restart.
    let t2 = Instant::now();
    post_form(&client, &format!("{base}/config/restart"), "").await;
    wait_serving(&client, &base, &["IPcam-general"]).await;
    eprintln!(
        "generation 3 (IPcam-general) ready after {:?}",
        t2.elapsed()
    );
    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    assert!(r.success && !r.predictions.is_empty(), "{r:?}");

    served.stop();
}

#[tokio::test]
async fn restart_into_zero_enabled_models_keeps_http_up() {
    let root = repo_root();
    let model = root.join("models").join("IPcam-general.onnx");
    let ov_dir = root.join("openvino");
    if !model.is_file() || !ov_dir.is_dir() {
        eprintln!(
            "skipping restart_into_zero_enabled_models_keeps_http_up: need {} and {}",
            model.display(),
            ov_dir.display()
        );
        return;
    }
    let image = std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_IMAGE))
        .expect("test image");
    let config_path = temp_dir("zero").join("cfg.json");
    let config = Config {
        models: vec![ModelConfig {
            path: model,
            family: ModelFamilyKind::Yolo5,
            ..Default::default()
        }],
        ..base_config(&ov_dir)
    };
    let client = reqwest::Client::new();
    let served = Served::start(config.clone(), config_path.clone());
    let base = served.base.clone();
    wait_serving(&client, &base, &["IPcam-general"]).await;

    // Edit the file on disk (every model disabled), then restart from the web UI.
    let mut off = config;
    off.models[0].enabled = false;
    off.save(&config_path).unwrap();
    let t0 = Instant::now();
    let page = post_form(&client, &format!("{base}/config/restart"), "").await;
    assert!(page.contains("Restarting"), "{page}");
    wait_until(
        "the zero-model generation",
        Duration::from_secs(60),
        || async {
            home(&client, &base)
                .await
                .is_some_and(|b| b.contains("No models are enabled"))
        },
    )
    .await;
    eprintln!("zero-model generation up after {:?}", t0.elapsed());

    assert_eq!(list_models(&client, &base).await, Some(Vec::new()));
    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    assert!(!r.success, "{r:?}");
    let cfg = get_text(&client, &format!("{base}/config")).await;
    assert!(cfg.contains("action=\"/config/models\""), "{cfg}");
    assert!(cfg.contains("aria-label=\"Load IPcam-general\">"), "{cfg}");

    served.stop();
}
