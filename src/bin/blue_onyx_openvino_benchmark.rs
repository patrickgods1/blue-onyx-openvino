//! Benchmark: compile each model, then time the production per-request pipeline
//! (decode -> preprocess -> infer -> postprocess, as in `worker.rs`) over N repeats.
//!
//! ```text
//! blue-onyx-openvino-benchmark --model models/IPcam-general.onnx --family yolo5 --device CPU --repeat 20
//! blue-onyx-openvino-benchmark --model models/yolo26s.xml --compare-cpu      # GPU vs CPU + confidence diff
//! blue-onyx-openvino-benchmark --json                                         # every model in the config
//! ```

use anyhow::{Context, Result, bail};
use blue_onyx_openvino::api::Prediction;
use blue_onyx_openvino::backend::{CoreOptions, LoadRequest, OvBackend, OvCore, libs};
use blue_onyx_openvino::config::{Config, ModelConfig};
use blue_onyx_openvino::model::preprocess::Preprocessor;
use blue_onyx_openvino::model::{ModelFamilyKind, PostParams};
use blue_onyx_openvino::registry::resolve_class_names;
use clap::Parser;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

/// Default benchmark image, embedded so the binary works standalone.
const DEFAULT_IMAGE: &[u8] = include_bytes!("../../tests/data/dog_bike_car.jpg");
const DEFAULT_IMAGE_NAME: &str = "dog_bike_car.jpg (embedded)";
/// Minimum IoU for a GPU detection and a CPU detection of the same label to count as the same
/// object in `--compare-cpu`.
const MATCH_IOU: f32 = 0.5;

#[derive(Debug, Parser)]
#[command(
    name = "blue-onyx-openvino-benchmark",
    version = env!("CARGO_PKG_VERSION"),
    about = "Benchmark Blue Onyx OpenVINO models with the production pre/post-processing pipeline"
)]
struct Args {
    /// Model file (.xml or .onnx), repeatable. Default: every model in the config file.
    #[arg(long)]
    model: Vec<PathBuf>,
    /// Model family for --model (default: auto-detect).
    #[arg(long, value_enum)]
    family: Option<ModelFamilyKind>,
    /// Class names YAML for --model (default: <model>.yaml beside the model, else COCO-80).
    #[arg(long)]
    classes: Option<PathBuf>,
    /// Device: GPU, GPU.N or CPU. GPU falls back to CPU like the server does.
    /// Default: GPU for --model, the configured device for config models.
    #[arg(long)]
    device: Option<String>,
    /// Force CPU inference.
    #[arg(long)]
    force_cpu: bool,
    /// JPEG/PNG to run on (default: the embedded tests/data/dog_bike_car.jpg).
    #[arg(long)]
    image: Option<PathBuf>,
    /// Timed iterations.
    #[arg(long, default_value_t = 100)]
    repeat: usize,
    /// Untimed warm-up iterations before the timed ones.
    #[arg(long, default_value_t = 5)]
    warmup: usize,
    /// Also run on CPU and compare timings and detections.
    #[arg(long)]
    compare_cpu: bool,
    /// Compiled-model cache directory ("" disables). Default: the configured cache dir.
    #[arg(long)]
    cache_dir: Option<String>,
    /// CPU inference threads (0 = OpenVINO default). Default: the configured intra_threads.
    #[arg(long)]
    threads: Option<usize>,
    /// Confidence threshold. Default: the configured confidence_threshold.
    #[arg(long)]
    min_confidence: Option<f32>,
    /// Config file for defaults (default: <exe_dir>/blue_onyx_openvino_config.json if present).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Print machine-readable JSON instead of tables.
    #[arg(long)]
    json: bool,
    /// Show info-level logs (on stderr).
    #[arg(long, short)]
    verbose: bool,
}

/// min / mean / p50 / p95 / max in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct Stats {
    min: f64,
    mean: f64,
    p50: f64,
    p95: f64,
    max: f64,
}

impl Stats {
    /// Nearest-rank percentiles over `samples` (ms). All zeros for an empty slice.
    fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                min: 0.0,
                mean: 0.0,
                p50: 0.0,
                p95: 0.0,
                max: 0.0,
            };
        }
        let mut s = samples.to_vec();
        s.sort_by(f64::total_cmp);
        Self {
            min: s[0],
            mean: s.iter().sum::<f64>() / s.len() as f64,
            p50: percentile(&s, 50.0),
            p95: percentile(&s, 95.0),
            max: s[s.len() - 1],
        }
    }
}

/// Nearest-rank percentile of an ascending slice: the smallest value with at least `p`% of the
/// samples at or below it.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let rank = ((p / 100.0) * n as f64).ceil() as usize;
    sorted[rank.clamp(1, n) - 1]
}

#[derive(Debug, Clone, Serialize)]
struct StageStats {
    decode: Stats,
    preprocess: Stats,
    infer: Stats,
    postprocess: Stats,
    total: Stats,
}

#[derive(Debug, Clone, Serialize)]
struct Detection {
    label: String,
    confidence: f32,
    x_min: usize,
    y_min: usize,
    x_max: usize,
    y_max: usize,
}

impl From<&Prediction> for Detection {
    fn from(p: &Prediction) -> Self {
        Self {
            label: p.label.clone(),
            confidence: p.confidence,
            x_min: p.x_min,
            y_min: p.y_min,
            x_max: p.x_max,
            y_max: p.y_max,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct RunResult {
    model: String,
    path: String,
    family: String,
    requested_device: String,
    device: String,
    device_name: String,
    fell_back: bool,
    input: String,
    image: String,
    image_size: String,
    /// read + reshape + compile wall time.
    compile_ms: f64,
    /// "disabled", "hit" (no new cache file written), "miss" (a blob was written) or "unknown".
    cache: String,
    warmup_iterations: usize,
    warmup_ms: f64,
    repeat: usize,
    stages_ms: StageStats,
    /// Sequential images per second over the timed loop.
    throughput_fps: f64,
    detections: Vec<Detection>,
}

#[derive(Debug, Clone, Serialize)]
struct Comparison {
    matched: usize,
    only_primary: usize,
    only_cpu: usize,
    max_confidence_diff: f32,
    mean_confidence_diff: f32,
}

#[derive(Debug, Clone, Serialize)]
struct ModelReport {
    primary: RunResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu: Option<RunResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<Comparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    version: &'static str,
    openvino_version: String,
    available_devices: Vec<String>,
    models: Vec<ModelReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    errors: Vec<String>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let level = if args.verbose { "info" } else { "warn" };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(level))
        .with_writer(std::io::stderr)
        .try_init();
    match run(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// One model to benchmark with its resolved settings.
struct Job {
    name: String,
    path: PathBuf,
    family: ModelFamilyKind,
    classes: Vec<String>,
    device: String,
    gpu_precision: Option<String>,
}

fn load_config(path: Option<&Path>) -> Result<Config> {
    let path = path
        .map(Path::to_path_buf)
        .unwrap_or_else(Config::default_config_path);
    if path.exists() {
        Config::load(&path)
    } else {
        Ok(Config::default())
    }
}

fn jobs(args: &Args, config: &Config) -> Result<Vec<Job>> {
    let device_override = if args.force_cpu {
        Some("CPU".to_string())
    } else {
        args.device.clone()
    };
    if !args.model.is_empty() {
        return args
            .model
            .iter()
            .map(|p| {
                let classes = resolve_class_names(p, args.classes.as_deref())?;
                Ok(Job {
                    name: ModelConfig {
                        path: p.clone(),
                        ..Default::default()
                    }
                    .effective_name(),
                    path: p.clone(),
                    family: args.family.unwrap_or(ModelFamilyKind::Auto),
                    classes,
                    device: device_override.clone().unwrap_or_else(|| "GPU".into()),
                    gpu_precision: None,
                })
            })
            .collect();
    }
    if config.models.is_empty() {
        bail!("no --model given and no models in the config file");
    }
    config
        .models
        .iter()
        .map(|m| {
            let path = blue_onyx_openvino::resolve_path(&m.path);
            let classes_path = m.classes.as_deref().map(blue_onyx_openvino::resolve_path);
            let classes = resolve_class_names(&path, classes_path.as_deref())?;
            Ok(Job {
                name: m.effective_name(),
                family: args.family.unwrap_or(m.family),
                classes,
                device: device_override
                    .clone()
                    .unwrap_or_else(|| config.device_for(m)),
                gpu_precision: m.gpu_precision.clone(),
                path,
            })
        })
        .collect()
}

fn run(args: Args) -> Result<bool> {
    let config = load_config(args.config.as_deref())?;
    let openvino_dir = config
        .openvino_dir
        .as_deref()
        .map(blue_onyx_openvino::resolve_path);
    libs::prepare_environment(openvino_dir.as_deref());

    if args.repeat == 0 {
        bail!("--repeat must be at least 1");
    }
    let jobs = jobs(&args, &config)?;
    let (image_bytes, image_name) = match &args.image {
        Some(p) => (
            std::fs::read(p).with_context(|| format!("reading image {}", p.display()))?,
            p.display().to_string(),
        ),
        None => (DEFAULT_IMAGE.to_vec(), DEFAULT_IMAGE_NAME.to_string()),
    };

    let cache_dir = match &args.cache_dir {
        Some(d) if d.trim().is_empty() => None,
        Some(d) => Some(PathBuf::from(d)),
        None => config.cache_dir_path(),
    };
    let opts = CoreOptions {
        cache_dir: cache_dir.clone(),
        intra_threads: args.threads.unwrap_or(config.intra_threads),
        openvino_dir,
    };
    let mut core = OvCore::new(&opts).context("initializing OpenVINO")?;
    let mut report = Report {
        version: blue_onyx_openvino::VERSION,
        openvino_version: core.openvino_version(),
        available_devices: core.available_devices().to_vec(),
        models: Vec::new(),
        errors: Vec::new(),
    };
    if !args.json {
        println!(
            "Blue Onyx OpenVINO {} benchmark | OpenVINO {} | devices {:?} | cache {}",
            report.version,
            report.openvino_version,
            report.available_devices,
            cache_dir
                .as_deref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "disabled".into())
        );
        println!(
            "image: {image_name} | warmup {} | repeat {}",
            args.warmup, args.repeat
        );
    }

    let params = PostParams {
        confidence_threshold: args.min_confidence.unwrap_or(config.confidence_threshold),
        nms_iou: config.nms_iou,
    };
    let bench = Bench {
        image: &image_bytes,
        image_name: &image_name,
        warmup: args.warmup,
        repeat: args.repeat,
        params,
        object_filter: &config.object_filter,
        cache_dir: cache_dir.as_deref(),
    };

    for job in &jobs {
        let primary = match bench.run(&mut core, job, &job.device, true) {
            Ok(r) => r,
            Err(e) => {
                let msg = format!("{}: {e:#}", job.name);
                if !args.json {
                    eprintln!("error: {msg}");
                }
                report.errors.push(msg);
                continue;
            }
        };
        let mut model_report = ModelReport {
            primary,
            cpu: None,
            comparison: None,
            note: None,
        };
        if args.compare_cpu {
            if model_report.primary.device == "CPU" {
                model_report.note =
                    Some("--compare-cpu skipped: the model already runs on CPU".into());
            } else {
                match bench.run(&mut core, job, "CPU", false) {
                    Ok(cpu) => {
                        model_report.comparison = Some(compare(
                            &model_report.primary.detections,
                            &cpu.detections,
                            MATCH_IOU,
                        ));
                        model_report.cpu = Some(cpu);
                    }
                    Err(e) => {
                        let msg = format!("{} on CPU: {e:#}", job.name);
                        if !args.json {
                            eprintln!("error: {msg}");
                        }
                        report.errors.push(msg);
                    }
                }
            }
        }
        if !args.json {
            print_model(&model_report);
        }
        report.models.push(model_report);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(report.errors.is_empty())
}

struct Bench<'a> {
    image: &'a [u8],
    image_name: &'a str,
    warmup: usize,
    repeat: usize,
    params: PostParams,
    object_filter: &'a [String],
    cache_dir: Option<&'a Path>,
}

/// Timings of one request (decode, preprocess, infer, postprocess; ms) and its result.
struct Sample {
    stages: [f64; 4],
    predictions: Vec<Prediction>,
    image_size: (u32, u32),
}

/// Files directly inside `dir` (the compiled-blob cache is flat).
fn cache_files(dir: &Path) -> std::collections::HashSet<PathBuf> {
    std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
        .unwrap_or_default()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

impl Bench<'_> {
    /// Compile `job` on `device` and time the per-request pipeline. `allow_fallback` mirrors
    /// the server: a GPU request may fall back to CPU.
    fn run(
        &self,
        core: &mut OvCore,
        job: &Job,
        device: &str,
        allow_fallback: bool,
    ) -> Result<RunResult> {
        let req = LoadRequest {
            path: job.path.clone(),
            device: device.to_string(),
            gpu_precision: job.gpu_precision.clone(),
            allow_cpu_fallback: allow_fallback && !device.eq_ignore_ascii_case("CPU"),
        };
        let before = self.cache_dir.map(cache_files);
        let t = Instant::now();
        let loaded = core
            .load(&req)
            .with_context(|| format!("loading {}", job.path.display()))?;
        let compile_ms = ms(t.elapsed());
        let cache = match (self.cache_dir, before) {
            (None, _) | (_, None) => "disabled".to_string(),
            (Some(dir), Some(before)) => {
                let after = cache_files(dir);
                if after.difference(&before).next().is_some() {
                    "miss".into()
                } else if after.is_empty() {
                    "unknown".into()
                } else {
                    "hit".into()
                }
            }
        };

        let dev = loaded.device.clone();
        let (in_w, in_h) = loaded.input_size;
        let family = blue_onyx_openvino::model::make_family(
            job.family,
            &loaded.inputs,
            &loaded.outputs,
            job.classes.len(),
        )?;
        let mut pre = Preprocessor::new(in_w, in_h, family.resize_mode());
        let mut backend = OvBackend::new(loaded)?;
        let filter = (!self.object_filter.is_empty()).then_some(self.object_filter);

        // One request exactly as `WorkerCtx::process` runs it, with per-stage timings.
        let mut iteration = || -> Result<Sample> {
            let t = Instant::now();
            let img = blue_onyx_openvino::image::decode(self.image)?;
            let decode = ms(t.elapsed());

            let t = Instant::now();
            let (chw, ctx) = pre.run(&img.rgb, img.width, img.height)?;
            let extra = family.extra_inputs(&ctx);
            let preprocess = ms(t.elapsed());

            let t = Instant::now();
            let outputs = backend.infer(chw, &extra)?;
            let infer = ms(t.elapsed());

            let t = Instant::now();
            let dets = family.postprocess(&outputs, &ctx, &self.params)?;
            let preds =
                blue_onyx_openvino::model::to_predictions(&dets, &ctx, &job.classes, filter);
            let postprocess = ms(t.elapsed());
            Ok(Sample {
                stages: [decode, preprocess, infer, postprocess],
                predictions: preds,
                image_size: (img.width, img.height),
            })
        };

        let t = Instant::now();
        for _ in 0..self.warmup {
            iteration().context("warm-up inference")?;
        }
        let warmup_ms = ms(t.elapsed());

        let mut samples: [Vec<f64>; 5] = Default::default();
        let mut last = Vec::new();
        let mut size = (0, 0);
        let t = Instant::now();
        for _ in 0..self.repeat {
            let sample = iteration()?;
            for (i, v) in sample.stages.iter().enumerate() {
                samples[i].push(*v);
            }
            samples[4].push(sample.stages.iter().sum());
            last = sample.predictions;
            size = sample.image_size;
        }
        let loop_s = t.elapsed().as_secs_f64();

        Ok(RunResult {
            model: job.name.clone(),
            path: job.path.display().to_string(),
            family: family.kind().to_string(),
            requested_device: device.to_string(),
            device: dev.actual.clone(),
            device_name: dev.full_name.clone(),
            fell_back: dev.fell_back,
            input: format!("{in_w}x{in_h}"),
            image: self.image_name.to_string(),
            image_size: format!("{}x{}", size.0, size.1),
            compile_ms,
            cache,
            warmup_iterations: self.warmup,
            warmup_ms,
            repeat: self.repeat,
            stages_ms: StageStats {
                decode: Stats::from_samples(&samples[0]),
                preprocess: Stats::from_samples(&samples[1]),
                infer: Stats::from_samples(&samples[2]),
                postprocess: Stats::from_samples(&samples[3]),
                total: Stats::from_samples(&samples[4]),
            },
            throughput_fps: if loop_s > 0.0 {
                self.repeat as f64 / loop_s
            } else {
                0.0
            },
            detections: last.iter().map(Detection::from).collect(),
        })
    }
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix = (a.x_max.min(b.x_max) as f32 - a.x_min.max(b.x_min) as f32).max(0.0);
    let iy = (a.y_max.min(b.y_max) as f32 - a.y_min.max(b.y_min) as f32).max(0.0);
    let inter = ix * iy;
    let area =
        |d: &Detection| (d.x_max.saturating_sub(d.x_min) * d.y_max.saturating_sub(d.y_min)) as f32;
    let union = area(a) + area(b) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Greedy one-to-one matching of same-label detections by descending IoU (>= `min_iou`);
/// reports confidence differences of matched pairs.
fn compare(primary: &[Detection], cpu: &[Detection], min_iou: f32) -> Comparison {
    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for (i, a) in primary.iter().enumerate() {
        for (j, b) in cpu.iter().enumerate() {
            if a.label == b.label {
                let v = iou(a, b);
                if v >= min_iou {
                    pairs.push((v, i, j));
                }
            }
        }
    }
    pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
    let mut used_a = vec![false; primary.len()];
    let mut used_b = vec![false; cpu.len()];
    let mut diffs = Vec::new();
    for (_, i, j) in pairs {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            diffs.push((primary[i].confidence - cpu[j].confidence).abs());
        }
    }
    let matched = diffs.len();
    Comparison {
        matched,
        only_primary: primary.len() - matched,
        only_cpu: cpu.len() - matched,
        max_confidence_diff: diffs.iter().copied().fold(0.0, f32::max),
        mean_confidence_diff: if matched == 0 {
            0.0
        } else {
            diffs.iter().sum::<f32>() / matched as f32
        },
    }
}

fn stats_row(name: &str, s: &Stats) -> String {
    format!(
        "  {name:<12} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
        s.min, s.mean, s.p50, s.p95, s.max
    )
}

fn print_run(r: &RunResult) {
    println!();
    println!(
        "== {} [{}] on {} ({}){}",
        r.model,
        r.family,
        r.device,
        if r.device_name.is_empty() {
            "?"
        } else {
            &r.device_name
        },
        if r.fell_back {
            format!(" -- FELL BACK from {}", r.requested_device)
        } else {
            String::new()
        }
    );
    println!(
        "  path {} | input {} | image {}",
        r.path, r.input, r.image_size
    );
    println!(
        "  compile {:.0} ms (cache: {}) | warmup {} x in {:.1} ms",
        r.compile_ms, r.cache, r.warmup_iterations, r.warmup_ms
    );
    println!(
        "  {:<12} {:>9} {:>9} {:>9} {:>9} {:>9}   (ms, {} runs)",
        "stage", "min", "mean", "p50", "p95", "max", r.repeat
    );
    let s = &r.stages_ms;
    println!("{}", stats_row("decode", &s.decode));
    println!("{}", stats_row("preprocess", &s.preprocess));
    println!("{}", stats_row("infer", &s.infer));
    println!("{}", stats_row("postprocess", &s.postprocess));
    println!("{}", stats_row("total", &s.total));
    println!(
        "  throughput {:.1} img/s | {} detections",
        r.throughput_fps,
        r.detections.len()
    );
    for d in &r.detections {
        println!(
            "    {:<14} {:.3}  [{}, {}, {}, {}]",
            d.label, d.confidence, d.x_min, d.y_min, d.x_max, d.y_max
        );
    }
}

fn print_model(m: &ModelReport) {
    print_run(&m.primary);
    if let Some(note) = &m.note {
        println!("  note: {note}");
    }
    let (Some(cpu), Some(c)) = (&m.cpu, &m.comparison) else {
        return;
    };
    print_run(cpu);
    let p = &m.primary;
    println!();
    println!("== {}: {} vs CPU (ms)", p.model, p.device);
    println!("  {:<12} {:>12} {:>12} {:>9}", "", p.device, "CPU", "ratio");
    let row = |name: &str, a: f64, b: f64| {
        let ratio = if a > 0.0 { b / a } else { 0.0 };
        println!("  {name:<12} {a:>12.2} {b:>12.2} {ratio:>8.2}x");
    };
    row("compile", p.compile_ms, cpu.compile_ms);
    row("infer p50", p.stages_ms.infer.p50, cpu.stages_ms.infer.p50);
    row(
        "infer mean",
        p.stages_ms.infer.mean,
        cpu.stages_ms.infer.mean,
    );
    row("total p50", p.stages_ms.total.p50, cpu.stages_ms.total.p50);
    row("total p95", p.stages_ms.total.p95, cpu.stages_ms.total.p95);
    println!(
        "  {:<12} {:>12.1} {:>12.1}",
        "img/s", p.throughput_fps, cpu.throughput_fps
    );
    println!(
        "  {:<12} {:>12} {:>12}",
        "detections",
        p.detections.len(),
        cpu.detections.len()
    );
    println!(
        "  matched {} (IoU >= {MATCH_IOU}, same label), only {} {}, only CPU {}; confidence diff max {:.4}, mean {:.4}",
        c.matched,
        p.device,
        c.only_primary,
        c.only_cpu,
        c.max_confidence_diff,
        c.mean_confidence_diff
    );
    if p.stages_ms.infer.p50 > cpu.stages_ms.infer.p50 && p.device != "CPU" {
        println!("  WARNING: {} inference is slower than CPU", p.device);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(label: &str, conf: f32, b: [usize; 4]) -> Detection {
        Detection {
            label: label.into(),
            confidence: conf,
            x_min: b[0],
            y_min: b[1],
            x_max: b[2],
            y_max: b[3],
        }
    }

    #[test]
    fn stats_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        let s = Stats::from_samples(&v);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 100.0);
        assert_eq!(s.p50, 50.0);
        assert_eq!(s.p95, 95.0);
        assert!((s.mean - 50.5).abs() < 1e-12);

        // Unsorted input, small n: p95 of 3 samples is the max; p50 the middle.
        let s = Stats::from_samples(&[3.0, 1.0, 2.0]);
        assert_eq!((s.p50, s.p95), (2.0, 3.0));
        let s = Stats::from_samples(&[7.0]);
        assert_eq!((s.min, s.p50, s.p95, s.max), (7.0, 7.0, 7.0, 7.0));
        assert_eq!(Stats::from_samples(&[]).max, 0.0);
    }

    #[test]
    fn iou_math() {
        let a = det("a", 1.0, [0, 0, 10, 10]);
        assert_eq!(iou(&a, &a), 1.0);
        // Half-overlap: inter 50, union 150.
        let b = det("a", 1.0, [5, 0, 15, 10]);
        assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
        let c = det("a", 1.0, [20, 20, 30, 30]);
        assert_eq!(iou(&a, &c), 0.0);
    }

    #[test]
    fn compare_matches_same_label_one_to_one() {
        let gpu = vec![
            det("dog", 0.90, [0, 0, 100, 100]),
            det("car", 0.80, [200, 200, 300, 300]),
            det("person", 0.6, [400, 0, 450, 100]),
        ];
        let cpu = vec![
            det("car", 0.78, [201, 200, 300, 301]),
            det("dog", 0.93, [1, 1, 100, 100]),
            // Same box as the GPU dog but another label: not a match.
            det("cat", 0.5, [0, 0, 100, 100]),
        ];
        let c = compare(&gpu, &cpu, 0.5);
        assert_eq!(c.matched, 2);
        assert_eq!(c.only_primary, 1);
        assert_eq!(c.only_cpu, 1);
        assert!((c.max_confidence_diff - 0.03).abs() < 1e-5);
        assert!((c.mean_confidence_diff - 0.025).abs() < 1e-5);

        // Two CPU boxes compete for one GPU box: only the higher-IoU one is matched.
        let gpu = vec![det("dog", 0.9, [0, 0, 100, 100])];
        let cpu = vec![
            det("dog", 0.5, [30, 0, 130, 100]),
            det("dog", 0.8, [0, 0, 100, 100]),
        ];
        let c = compare(&gpu, &cpu, 0.5);
        assert_eq!((c.matched, c.only_primary, c.only_cpu), (1, 0, 1));
        assert!((c.max_confidence_diff - 0.1).abs() < 1e-6);

        let c = compare(&[], &[], 0.5);
        assert_eq!(c.matched, 0);
        assert_eq!(c.max_confidence_diff, 0.0);
    }

    #[test]
    fn embedded_image_decodes() {
        let img = blue_onyx_openvino::image::decode(DEFAULT_IMAGE).unwrap();
        assert!(img.width > 0 && img.height > 0);
        assert_eq!(img.rgb.len(), (img.width * img.height * 3) as usize);
    }

    #[test]
    fn args_parse() {
        let a = Args::try_parse_from([
            "x",
            "--model",
            "a.onnx",
            "--model",
            "b.xml",
            "--family",
            "yolo5",
            "--repeat",
            "20",
            "--compare-cpu",
            "--json",
        ])
        .unwrap();
        assert_eq!(a.model.len(), 2);
        assert_eq!(a.repeat, 20);
        assert_eq!(a.warmup, 5);
        assert!(a.compare_cpu && a.json);
        let a = Args::try_parse_from(["x"]).unwrap();
        assert_eq!(a.repeat, 100);
    }
}
