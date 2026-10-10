//! Test client for a running Blue Onyx Prism server (CodeProject.AI compatible API).
//!
//! Sends an image to `/v1/vision/detection` or `/v1/vision/custom/{model}` (optionally many
//! times, optionally in parallel), prints the response or a latency summary and exits non-zero
//! if any request failed.

use anyhow::{Context, Result, bail};
use blue_onyx_prism::api::{Prediction, VisionDetectionResponse};
use clap::Parser;
use futures_util::StreamExt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

static DEFAULT_IMAGE: &[u8] = include_bytes!("../../tests/data/dog_bike_car.jpg");

#[derive(Debug, Parser)]
#[command(
    name = "test-blue-onyx-prism",
    version,
    about = "Send test images to a running Blue Onyx Prism server"
)]
struct Args {
    /// Server base URL.
    #[arg(long, default_value = "http://127.0.0.1:32168")]
    url: String,
    /// Named model: POST /v1/vision/custom/{model}. Default: /v1/vision/detection.
    #[arg(long)]
    model: Option<String>,
    /// Image to send (default: embedded dog_bike_car.jpg).
    #[arg(long)]
    image: Option<PathBuf>,
    /// min_confidence form field (0 = server default).
    #[arg(long, default_value_t = 0.0)]
    min_confidence: f32,
    /// Number of requests to send.
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    /// Delay between requests in milliseconds.
    #[arg(long, default_value_t = 0)]
    interval_ms: u64,
    /// Number of concurrent requests.
    #[arg(long, default_value_t = 1)]
    parallel: usize,
    /// List the configured models (POST /v1/vision/custom/list) and exit.
    #[arg(long)]
    list: bool,
    /// Save an annotated copy of the image (from the last response) to this path.
    #[arg(long)]
    save: Option<PathBuf>,
}

struct Outcome {
    ok: bool,
    round_trip_ms: f64,
    inference_ms: i32,
    process_ms: i32,
    response: Option<VisionDetectionResponse>,
    note: String,
}

async fn send_one(
    client: &reqwest::Client,
    url: &str,
    image: &[u8],
    min_confidence: f32,
    delay: Duration,
) -> Outcome {
    tokio::time::sleep(delay).await;
    let start = Instant::now();
    let part = reqwest::multipart::Part::bytes(image.to_vec()).file_name("image.jpg");
    let mut form = reqwest::multipart::Form::new().part("image", part);
    if min_confidence > 0.0 {
        form = form.text("min_confidence", min_confidence.to_string());
    }
    let result = async {
        let resp = client.post(url).multipart(form).send().await?;
        let status = resp.status();
        let body = resp.text().await?;
        Ok::<_, reqwest::Error>((status, body))
    }
    .await;
    let round_trip_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut out = Outcome {
        ok: false,
        round_trip_ms,
        inference_ms: 0,
        process_ms: 0,
        response: None,
        note: String::new(),
    };
    match result {
        Err(e) => out.note = format!("request failed: {e}"),
        Ok((status, body)) => {
            if !status.is_success() {
                out.note = format!("HTTP {status}: {body}");
            } else {
                match serde_json::from_str::<VisionDetectionResponse>(&body) {
                    Ok(r) => {
                        out.ok = r.success;
                        out.inference_ms = r.inferenceMs;
                        out.process_ms = r.processMs;
                        if !r.success {
                            out.note = format!(
                                "success=false: {}",
                                r.error.clone().unwrap_or(r.message.clone())
                            );
                        }
                        out.response = Some(r);
                    }
                    Err(e) => out.note = format!("invalid JSON ({e}): {body}"),
                }
            }
        }
    }
    out
}

fn stats(label: &str, mut v: Vec<f64>) {
    if v.is_empty() {
        return;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let p95 = v[(((v.len() as f64) * 0.95).ceil() as usize).clamp(1, v.len()) - 1];
    println!(
        "  {label:<22} min {:>8.1}  mean {:>8.1}  p95 {:>8.1}  max {:>8.1}",
        v[0],
        mean,
        p95,
        v[v.len() - 1]
    );
}

async fn run(args: Args) -> Result<bool> {
    let base = args.url.trim_end_matches('/').to_string();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    if args.list {
        let resp = client
            .post(format!("{base}/v1/vision/custom/list"))
            .send()
            .await
            .context("POST /v1/vision/custom/list")?;
        let status = resp.status();
        let body = resp.text().await?;
        match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(v) => println!("{}", serde_json::to_string_pretty(&v)?),
            Err(_) => println!("{body}"),
        }
        return Ok(status.is_success());
    }

    let image: Vec<u8> = match &args.image {
        Some(p) => std::fs::read(p).with_context(|| format!("reading {}", p.display()))?,
        None => DEFAULT_IMAGE.to_vec(),
    };
    let url = match &args.model {
        Some(m) => format!("{base}/v1/vision/custom/{m}"),
        None => format!("{base}/v1/vision/detection"),
    };
    let repeat = args.repeat.max(1);
    let parallel = args.parallel.max(1);
    let interval = Duration::from_millis(args.interval_ms);

    let wall = Instant::now();
    let outcomes: Vec<Outcome> = futures_util::stream::iter(0..repeat)
        .map(|i| {
            let delay = interval * (i / parallel) as u32;
            let (client, url, image) = (&client, &url, &image);
            async move { send_one(client, url, image, args.min_confidence, delay).await }
        })
        .buffered(parallel)
        .collect()
        .await;
    let wall_s = wall.elapsed().as_secs_f64();

    let failures = outcomes.iter().filter(|o| !o.ok).count();
    if repeat == 1 {
        let o = &outcomes[0];
        match &o.response {
            Some(r) => println!("{}", serde_json::to_string_pretty(r)?),
            None => println!("{}", o.note),
        }
        println!("round trip: {:.1} ms", o.round_trip_ms);
    } else {
        println!("POST {url}");
        println!(
            "{repeat} requests, parallel {parallel}: {} ok, {failures} failed, {wall_s:.2} s total",
            repeat - failures
        );
        println!("latency (ms):");
        stats(
            "client round trip",
            outcomes.iter().map(|o| o.round_trip_ms).collect(),
        );
        let ok: Vec<&Outcome> = outcomes.iter().filter(|o| o.response.is_some()).collect();
        stats(
            "server inferenceMs",
            ok.iter().map(|o| o.inference_ms as f64).collect(),
        );
        stats(
            "server processMs",
            ok.iter().map(|o| o.process_ms as f64).collect(),
        );
        for o in outcomes.iter().filter(|o| !o.ok).take(5) {
            eprintln!("failure: {}", o.note);
        }
    }

    if let Some(path) = &args.save {
        let preds: Vec<Prediction> = outcomes
            .iter()
            .rev()
            .find_map(|o| o.response.as_ref())
            .map(|r| r.predictions.clone())
            .unwrap_or_default();
        let img = blue_onyx_prism::image::decode(&image)?;
        let annotated = blue_onyx_prism::image::draw_predictions(&img, &preds);
        std::fs::write(path, blue_onyx_prism::image::encode_jpeg(&annotated, 95)?)
            .with_context(|| format!("writing {}", path.display()))?;
        println!("saved annotated image to {}", path.display());
    }
    Ok(failures == 0)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    if !rt.block_on(run(args))? {
        bail!("one or more requests failed");
    }
    Ok(())
}
