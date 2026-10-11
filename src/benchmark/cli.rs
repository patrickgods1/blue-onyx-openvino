//! Command line front end of the benchmark: the `blue-onyx-prism-benchmark` binary and the
//! `blue-onyx-prism benchmark` subcommand both parse [`BenchArgs`] and call [`main`].
//!
//! ```text
//! blue-onyx-prism-benchmark --model models/IPcam-general.onnx --family yolo5 --device CPU --repeat 20
//! blue-onyx-prism-benchmark --model models/yolo26s.xml --compare-cpu      # GPU vs CPU + confidence diff
//! blue-onyx-prism-benchmark --json                                         # every enabled model in the config
//! blue-onyx-prism-benchmark --all-devices --apply                          # sweep, save, set per-model devices
//! blue-onyx-prism-benchmark --apply-threshold --threshold-objective f2      # also set per-model thresholds
//! blue-onyx-prism-benchmark --threshold-search --tag night --threshold-objective recall:0.9
//!                                                       # search the stored predictions, no inference
//! blue-onyx-prism-benchmark --all-devices --accuracy-metric roc_auc --threshold-objective youden
//!                                                       # grade by frame ROC AUC, pick by Youden's J
//! ```

use super::grade::AccuracyMetric;
use super::images::{self, ImageSet};
use super::report::{HardwareSummary, RuntimeVersions, Verdict};
use super::threshold::{Objective, ThresholdAdvice};
use super::{
    Bench, BenchmarkResults, Comparison, Job, MATCH_IOU, ModelResult, Phase, RunResult, Stats,
    SweepOptions, compare, config_models, configured_device, export, job_for_config, job_for_file,
    pick_reference, pseudo_ground_truth, reference_device, results_path, sweep,
};
use crate::backend::{CoreOptions, OrtOptions, Runtimes, libs, spec};
use crate::config::{
    Config, DatasetRef, DeviceChange, ModelConfig, ThresholdChange, apply_model_devices,
    apply_model_thresholds,
};
use crate::model::{ModelFamilyKind, PostParams};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;

#[derive(Debug, Clone, clap::Args)]
pub struct BenchArgs {
    /// Model file (.xml or .onnx), repeatable. Default: every enabled model in the config file.
    #[arg(long)]
    pub model: Vec<PathBuf>,
    /// Model family for --model (default: auto-detect).
    #[arg(long, value_enum)]
    pub family: Option<ModelFamilyKind>,
    /// Class names YAML for --model (default: <model>.yaml beside the model, else COCO-80).
    #[arg(long)]
    pub classes: Option<PathBuf>,
    /// Device spec, one of: auto, openvino:gpu[.N], openvino:cpu, openvino:npu, ort:cuda[:N],
    /// ort:tensorrt[:N], ort:directml[:N], ort:coreml, ort:cpu (legacy GPU, GPU.N, CPU, NPU mean
    /// OpenVINO). A GPU request falls back to CPU like the server does (without the warm-up check).
    /// Default: auto for --model, the configured device for config models.
    #[arg(long, value_parser = parse_device_arg)]
    pub device: Option<String>,
    /// Run every runnable device option (see `list-devices`; `benchmark.devices` of the config
    /// limits them) for each model, grade accuracy and
    /// speed, compare detections with the CPU reference, print the recommended device, and save
    /// the results to `benchmark.json` next to the config file (web UI Benchmark page). Cannot
    /// be combined with --device, --force-cpu or --compare-cpu.
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu"])]
    pub all_devices: bool,
    /// Implies --all-devices: write each benchmarked config model's recommended device into its
    /// `device` in the config file and print what changed (restart the server to apply).
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu", "model"])]
    pub apply: bool,
    /// Implies --all-devices: write each benchmarked config model's best confidence threshold
    /// (see --threshold-objective) into its `confidence_threshold` in the config file and print
    /// what changed. Independent of --apply (devices); both may be given.
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu", "model", "min_confidence"])]
    pub apply_threshold: bool,
    /// What the best confidence threshold optimizes: f1 (balance), f2 (favor recall: fewer
    /// missed objects) or precision:<p> (highest recall with precision >= p, e.g.
    /// precision:0.9 for fewer false alerts) or recall:<r> (highest precision with recall >= r),
    /// youden (frame-level Youden's J: alert rate on frames with objects minus false-alert rate
    /// on frames without) or fpr:<x> (highest frame alert rate with at most x false alerts per
    /// frame without objects, e.g. fpr:0.05). Default: benchmark.threshold_objective (f1).
    #[arg(long, value_parser = parse_objective_arg)]
    pub threshold_objective: Option<Objective>,
    /// What the accuracy grade, the ranking and the device recommendation use: ap50 (box-level
    /// AP@0.5 with small-object recall) or roc_auc (macro frame-level ROC AUC: how well a
    /// frame's top confidence separates frames with objects from frames without). Default:
    /// benchmark.accuracy_metric (ap50).
    #[arg(long, value_parser = parse_metric_arg)]
    pub accuracy_metric: Option<AccuracyMetric>,
    /// Search the best confidence threshold over the predictions stored by the last benchmark
    /// (`benchmark-preds.json` beside the config; no inference), filtered by --dataset, --tag,
    /// --class, --search-model, --search-device and --iou, for --threshold-objective (also
    /// `recall:<r>`). Runs the benchmark first when nothing is stored.
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu"])]
    pub threshold_search: bool,
    /// --threshold-search: only images with this tag (repeatable: all must match), e.g. night.
    #[arg(long, requires = "threshold_search")]
    pub tag: Vec<String>,
    /// --threshold-search: only these scored classes (repeatable), e.g. person.
    #[arg(long = "class", requires = "threshold_search")]
    pub class: Vec<String>,
    /// --threshold-search: only these models (repeatable; default every stored model).
    #[arg(long, requires = "threshold_search")]
    pub search_model: Vec<String>,
    /// --threshold-search: the device whose predictions to use (default each model's
    /// recommended device).
    #[arg(long, requires = "threshold_search", value_parser = parse_device_arg)]
    pub search_device: Option<String>,
    /// --threshold-search: matching IoU (default 0.5).
    #[arg(long, requires = "threshold_search")]
    pub iou: Option<f32>,
    /// Implies --all-devices: also write a standalone report of the saved results (`.html` or
    /// `.md`).
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu"])]
    pub report: Option<PathBuf>,
    /// Do not write `benchmark.json` (--all-devices).
    #[arg(long)]
    pub no_save: bool,
    /// Force CPU inference.
    #[arg(long)]
    pub force_cpu: bool,
    /// One JPEG/PNG to run on (overrides the datasets).
    #[arg(long, conflicts_with_all = ["dataset", "images"])]
    pub image: Option<PathBuf>,
    /// Dataset, repeatable: a built-in set id (see --list-datasets), `sample` (the embedded
    /// image) or `dir:<path>`. Default: `benchmark.datasets` of the config.
    #[arg(long)]
    pub dataset: Vec<String>,
    /// A directory of images (same as `--dataset dir:<path>`), with optional COCO JSON / YOLO
    /// ground truth (see --gt) and manifest.json tags.
    #[arg(long)]
    pub images: Option<PathBuf>,
    /// Ground truth for --images: a COCO instances JSON file or a YOLO labels directory
    /// (default: found in the directory).
    #[arg(long, requires = "images")]
    pub gt: Option<PathBuf>,
    /// Images per dataset, evenly spread (0 = all). Default: benchmark.max_images_per_dataset.
    #[arg(long)]
    pub max_images: Option<usize>,
    /// List the datasets (built-in sets, their download state, the configured selection) and exit.
    #[arg(long)]
    pub list_datasets: bool,
    /// Model whose detections are the pseudo ground truth of datasets without ground truth
    /// (default: benchmark.reference_model, else the most accurate installed model).
    #[arg(long)]
    pub reference_model: Option<String>,
    /// Share of accuracy in the overall grade, 0..1 (default: benchmark.weights, 0.6).
    #[arg(long)]
    pub accuracy_weight: Option<f64>,
    /// Timed runs per image. Default: 100 for a single image, else benchmark.repeat_per_image.
    #[arg(long)]
    pub repeat: Option<usize>,
    /// Untimed warm-up iterations. Default: benchmark.warmup.
    #[arg(long)]
    pub warmup: Option<usize>,
    /// Also run on CPU and compare timings and detections.
    #[arg(long)]
    pub compare_cpu: bool,
    /// Compiled-model cache directory ("" disables). Default: the configured cache dir.
    #[arg(long)]
    pub cache_dir: Option<String>,
    /// CPU inference threads (0 = OpenVINO default). Default: the configured intra_threads.
    #[arg(long)]
    pub threads: Option<usize>,
    /// Confidence threshold. Default: the configured confidence_threshold.
    #[arg(long)]
    pub min_confidence: Option<f32>,
    /// Config file for defaults (default: <exe_dir>/blue_onyx_prism_config.json if present).
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Print machine-readable JSON instead of tables.
    #[arg(long)]
    pub json: bool,
    /// Show info-level logs (on stderr).
    #[arg(long, short)]
    pub verbose: bool,
}

fn parse_objective_arg(s: &str) -> Result<Objective, String> {
    s.parse()
}

fn parse_metric_arg(s: &str) -> Result<AccuracyMetric, String> {
    s.parse()
}

/// Default timed runs for a single image (the pre-dataset default).
pub const SINGLE_IMAGE_REPEAT: usize = 100;

/// Validate `--device` with the spec parser, keeping the string as typed.
fn parse_device_arg(s: &str) -> Result<String, String> {
    spec::parse(s)
        .map(|_| s.trim().to_string())
        .map_err(|e| e.to_string())
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
    /// `--all-devices`: one run per runnable device option (`primary` is the recommended one, or
    /// the fastest by total p50 when none is recommended).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    all_devices: Vec<RunResult>,
    /// `--all-devices`: options that cannot run or failed for this model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    skipped_devices: Vec<String>,
    /// `--all-devices`: the recommended device and why.
    #[serde(skip_serializing_if = "Option::is_none")]
    recommended: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recommendation: Option<String>,
    /// `--all-devices`: device whose detections the others were compared with.
    #[serde(skip_serializing_if = "Option::is_none")]
    reference: Option<String>,
    /// `--all-devices`: the best confidence threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    threshold: Option<ThresholdAdvice>,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    version: &'static str,
    openvino_version: String,
    available_devices: Vec<String>,
    models: Vec<ModelReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    errors: Vec<String>,
    /// `--all-devices`: where the results were saved.
    #[serde(skip_serializing_if = "Option::is_none")]
    results_file: Option<String>,
    /// `--apply`: per-model device changes written to the config file.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    applied: Vec<DeviceChange>,
    /// `--apply-threshold`: per-model threshold changes written to the config file.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    applied_thresholds: Vec<ThresholdChange>,
    /// Datasets used (ids) and warnings about them.
    datasets: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
    /// `--all-devices`: models ranked on their recommended devices.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    ranking: Vec<super::report::RankRow>,
}

/// Parse-independent entry point: set up logging, run, map the outcome to an exit code.
/// `config_fallback` is the main binary's `--config` (used when the subcommand has none).
pub fn main(mut args: BenchArgs, config_fallback: Option<PathBuf>) -> ExitCode {
    if args.config.is_none() {
        args.config = config_fallback;
    }
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

fn load_config(path: &Path) -> Result<Config> {
    if path.exists() {
        Config::load(path)
    } else {
        Ok(Config::default())
    }
}

/// One job, with its config entry when it came from the config file.
struct Planned {
    job: Job,
    entry: Option<ModelConfig>,
}

fn jobs(args: &BenchArgs, config: &Config, errors: &mut Vec<String>) -> Result<Vec<Planned>> {
    let device_override = if args.force_cpu {
        Some("CPU".to_string())
    } else {
        args.device.clone()
    };
    let mut out = Vec::new();
    if !args.model.is_empty() {
        for p in &args.model {
            out.push(Planned {
                job: job_for_file(
                    p,
                    args.family,
                    args.classes.as_deref(),
                    device_override.as_deref(),
                )?,
                entry: None,
            });
        }
    } else {
        for m in config_models(config)? {
            match job_for_config(config, m, device_override.as_deref()) {
                Ok(job) => out.push(Planned {
                    job,
                    entry: Some(m.clone()),
                }),
                Err(e) => errors.push(format!("{}: {e:#}", m.effective_name())),
            }
        }
    }
    for p in &mut out {
        if let Some(f) = args.family {
            p.job.family = f;
        }
        if let Some(c) = args.min_confidence {
            p.job.confidence_threshold = Some(c);
        }
    }
    Ok(out)
}

/// The datasets of a run: `--image`, else `--dataset`/`--images`, else the config's.
fn dataset_refs(args: &BenchArgs, config: &Config) -> Vec<DatasetRef> {
    let mut refs: Vec<DatasetRef> = args.dataset.iter().cloned().map(DatasetRef::Id).collect();
    if let Some(dir) = &args.images {
        refs.push(DatasetRef::Dir {
            dir: dir.clone(),
            gt: args.gt.clone(),
            name: None,
        });
    }
    if refs.is_empty() {
        refs = config.benchmark.datasets.clone();
    }
    refs
}

/// Resolve the datasets, downloading missing built-in sets first when the config allows it.
fn load_sets(args: &BenchArgs, config: &Config) -> Result<(Vec<ImageSet>, Vec<String>)> {
    if let Some(p) = &args.image {
        return Ok((vec![ImageSet::single_file(p)?], Vec::new()));
    }
    let refs = dataset_refs(args, config);
    let max = args
        .max_images
        .unwrap_or(config.benchmark.max_images_per_dataset);
    let root = config.data_root();
    let mut r = images::resolve(&refs, &root, max)?;
    if !r.not_downloaded.is_empty() && config.benchmark.auto_download_datasets {
        let req = crate::resources::commands::FetchRequest {
            for_config: false,
            resources: r
                .not_downloaded
                .iter()
                .map(|id| format!("{}{id}", crate::resources::catalog::BENCH_ID_PREFIX))
                .collect(),
            all_for_platform: false,
            allow_large: false,
        };
        match crate::resources::commands::fetch(config, &req) {
            Ok(()) => r = images::resolve(&refs, &root, max)?,
            Err(e) => r
                .warnings
                .push(format!("downloading datasets failed: {e:#}")),
        }
    }
    Ok((r.sets, r.warnings))
}

/// `--list-datasets`.
fn list_datasets(config: &Config) {
    let root = config.data_root();
    let selected: Vec<String> = config
        .benchmark
        .datasets
        .iter()
        .map(|d| d.label())
        .collect();
    println!(
        "  {:<16} {:>7} {:>9} {:<12} {:<10} TITLE / LICENSE",
        "ID", "IMAGES", "SIZE", "STATE", "SELECTED"
    );
    let mark = |id: &str| {
        if selected.iter().any(|s| s.eq_ignore_ascii_case(id)) {
            "yes"
        } else {
            ""
        }
    };
    let sample = ImageSet::sample();
    println!(
        "  {:<16} {:>7} {:>9} {:<12} {:<10} {}",
        images::SAMPLE_ID,
        1,
        "-",
        "embedded",
        mark(images::SAMPLE_ID),
        sample.title
    );
    for id in images::KNOWN_SET_IDS {
        if images::builtin_manifest(id).is_none() {
            println!(
                "  {:<16} {:>7} {:>9} {:<12} {:<10} (not in this build)",
                id,
                "-",
                "-",
                "-",
                mark(id)
            );
        }
    }
    for m in images::builtin_manifests() {
        let (set, missing) = ImageSet::from_manifest(m, &images::builtin_dir(&root, &m.id));
        let state = if missing == 0 {
            "installed".to_string()
        } else if set.images.is_empty() {
            "available".to_string()
        } else {
            format!("{}/{}", set.images.len(), m.images.len())
        };
        println!(
            "  {:<16} {:>7} {:>9} {:<12} {:<10} {} ({})",
            m.id,
            m.images.len(),
            crate::resources::catalog::format_size(m.size()),
            state,
            mark(&m.id),
            m.title,
            m.license
        );
    }
    for d in &config.benchmark.datasets {
        if let Some((dir, _, _)) = d.dir() {
            println!(
                "  {:<16} {:>7} {:>9} {:<12} {:<10} {}",
                d.label(),
                "-",
                "-",
                if config.data_path(&dir).is_dir() {
                    "folder"
                } else {
                    "missing"
                },
                "yes",
                config.data_path(&dir).display()
            );
        }
    }
    println!(
        "Download a set with `blue-onyx-prism fetch --resource bench:<id>`; select datasets with \
         --dataset <id|dir:path> or `benchmark.datasets` in the config."
    );
}

/// Run the benchmark described by `args`. Ok(false) when some model failed.
pub fn run(args: BenchArgs) -> Result<bool> {
    let config_path = args
        .config
        .clone()
        .unwrap_or_else(Config::default_config_path);
    let config = load_config(&config_path)?;
    if args.list_datasets {
        list_datasets(&config);
        return Ok(true);
    }
    if args.threshold_search {
        let results = BenchmarkResults::load(&results_path(&config_path))?;
        let preds = super::search::StoredPreds::load(&super::search::preds_path(&config_path))?;
        match (results, preds) {
            (Some(r), Some(p)) if !p.runs.is_empty() => {
                return threshold_search_cli(&args, &config, &r, &p);
            }
            _ => {
                if !args.json {
                    eprintln!(
                        "--threshold-search: no stored predictions beside {}; running the benchmark first",
                        config_path.display()
                    );
                }
            }
        }
    }
    if (args.apply || args.apply_threshold) && !config_path.exists() {
        bail!(
            "{} needs a config file; {} does not exist",
            if args.apply {
                "--apply"
            } else {
                "--apply-threshold"
            },
            config_path.display()
        );
    }
    let objective = args
        .threshold_objective
        .unwrap_or(config.benchmark.threshold_objective);
    let metric = args
        .accuracy_metric
        .unwrap_or(config.benchmark.accuracy_metric);
    if let Some(w) = args.accuracy_weight
        && !(0.0..=1.0).contains(&w)
    {
        bail!("--accuracy-weight must be between 0 and 1");
    }
    let all_devices = args.all_devices
        || args.apply
        || args.apply_threshold
        || args.threshold_search
        || args.report.is_some();
    let openvino_dir = config.openvino_dir_effective();
    libs::prepare_environment(openvino_dir.as_deref());

    if args.repeat == Some(0) {
        bail!("--repeat must be at least 1");
    }
    let mut errors = Vec::new();
    let jobs = jobs(&args, &config, &mut errors)?;
    let (mut sets, warnings) = load_sets(&args, &config)?;
    let image_total: usize = sets.iter().map(|s| s.images.len()).sum();
    let repeat = args.repeat.unwrap_or(if image_total == 1 {
        SINGLE_IMAGE_REPEAT
    } else {
        config.benchmark.repeat_per_image.max(1)
    });
    let warmup = args.warmup.unwrap_or(config.benchmark.warmup);
    let weights = match args.accuracy_weight {
        Some(a) => super::grade::Weights {
            accuracy: a,
            speed: 1.0 - a,
        },
        None => config.benchmark.weights,
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
    let ort_opts: OrtOptions = config.ort_options();
    let runtimes = Runtimes::new_with(&opts, &ort_opts);
    runtimes.require_any()?;
    let info = runtimes.info();
    let versions = RuntimeVersions::of(&runtimes);
    let runtimes = Mutex::new(runtimes);
    let mut report = Report {
        version: crate::VERSION,
        openvino_version: info.openvino_version,
        available_devices: info.available_devices,
        models: Vec::new(),
        errors,
        results_file: None,
        applied: Vec::new(),
        applied_thresholds: Vec::new(),
        datasets: sets.iter().map(|s| s.id.clone()).collect(),
        warnings,
        ranking: Vec::new(),
    };
    if !args.json {
        println!(
            "Blue Onyx Prism {} benchmark | OpenVINO {} | ONNX Runtime {} | devices {:?} | cache {}",
            report.version,
            report.openvino_version,
            versions
                .onnxruntime
                .as_deref()
                .map(|v| match &versions.onnxruntime_flavor {
                    Some(f) => format!("{v} ({f})"),
                    None => v.to_string(),
                })
                .unwrap_or_else(|| "not loaded".into()),
            report.available_devices,
            cache_dir
                .as_deref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "disabled".into())
        );
        let names: Vec<String> = sets
            .iter()
            .map(|s| {
                format!(
                    "{} ({} images, {})",
                    s.id,
                    s.images.len(),
                    s.ground_truth.describe()
                )
            })
            .collect();
        println!("datasets: {}", names.join("; "));
        println!("warmup {warmup} | {repeat} timed run(s) per image");
        for w in &report.warnings {
            eprintln!("warning: {w}");
        }
        for e in &report.errors {
            eprintln!("error: {e}");
        }
    }

    let params = PostParams {
        confidence_threshold: args.min_confidence.unwrap_or(config.confidence_threshold),
        nms_iou: config.nms_iou,
    };
    let quiet = args.json;
    let progress = move |p: &super::Progress| {
        if quiet {
            return;
        }
        if p.phase == Phase::Loading {
            eprintln!("  {} on {} ...", p.model, p.device);
        } else if p.phase == Phase::Timing
            && p.total >= 50
            && p.done > 0
            && p.done.is_multiple_of(50)
        {
            eprintln!("    {}/{} requests", p.done, p.total);
        }
    };

    // Pseudo ground truth for datasets without it (accuracy relative to a reference model).
    if sets.iter().any(|s| !s.annotated()) {
        let reference = pick_reference(
            &config,
            args.reference_model
                .as_deref()
                .or(config.benchmark.reference_model.as_deref()),
        )
        .and_then(|m| job_for_config(&config, m, None));
        let mut pb = Bench::new(&[], 0, 1, params);
        pb.cache_dir = cache_dir.as_deref();
        pb.progress = Some(&progress);
        let outcome = reference.and_then(|job| {
            let sel = runtimes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .selection(Some(&job.path));
            let device = reference_device(&sel).context("no CPU device for pseudo ground truth")?;
            if !args.json {
                eprintln!("pseudo ground truth: {} on {device} ...", job.name);
            }
            if jobs
                .iter()
                .any(|p| p.job.name.eq_ignore_ascii_case(&job.name))
            {
                let w = format!(
                    "{} is also the pseudo-ground-truth reference: its accuracy on the datasets \
                     without ground truth is 100% by construction",
                    job.name
                );
                if !args.json {
                    eprintln!("warning: {w}");
                }
                report.warnings.push(w);
            }
            pseudo_ground_truth(&pb, &runtimes, &job, device, &mut sets)
        });
        if let Err(e) = outcome {
            let w = format!("accuracy is not scored for datasets without ground truth: {e:#}");
            if !args.json {
                eprintln!("warning: {w}");
            }
            report.warnings.push(w);
        }
    }

    let mut bench = Bench::new(&sets, warmup, repeat, params);
    bench.object_filter = &config.object_filter;
    bench.cache_dir = cache_dir.as_deref();
    bench.weights = weights;
    bench.threshold_objective = objective;
    bench.accuracy_metric = metric;

    // `benchmark.devices` of the config limits the sweep, as on the web UI's Benchmark page.
    let sweep_devices = super::service::parse_devices(&config.benchmark.devices)
        .context("benchmark.devices in the config")?;
    let mut swept: Vec<ModelResult> = Vec::new();
    for Planned { job, entry } in &jobs {
        if all_devices {
            let configured = entry.as_ref().and_then(|m| {
                let sel = runtimes
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .selection(Some(&job.path));
                configured_device(&config, m, &sel)
            });
            let opts = SweepOptions {
                devices: sweep_devices.clone(),
                configured,
                threshold_objective: objective,
            };
            bench.progress = Some(&progress);
            let result = sweep(&bench, &runtimes, job, &opts, &mut |_| {})?;
            bench.progress = None;
            match model_report(&result) {
                Ok(m) => {
                    if !args.json {
                        print_all_devices(&result, &m);
                    }
                    report.models.push(m);
                }
                Err(e) => {
                    let msg = format!("{}: {e:#}", job.name);
                    if !args.json {
                        eprintln!("error: {msg}");
                    }
                    report.errors.push(msg);
                }
            }
            swept.push(result);
            continue;
        }
        let primary = match bench.run(&runtimes, job, &job.device) {
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
            all_devices: Vec::new(),
            skipped_devices: Vec::new(),
            recommended: None,
            recommendation: None,
            reference: None,
            threshold: None,
        };
        if args.compare_cpu {
            if model_report.primary.device == "CPU" {
                model_report.note =
                    Some("--compare-cpu skipped: the model already runs on CPU".into());
            } else {
                match bench.run(&runtimes, job, "CPU") {
                    Ok(cpu) => {
                        let mut c = Comparison::default();
                        for (a, b) in model_report.primary.per_image.iter().zip(&cpu.per_image) {
                            let a: Vec<_> = a.preds.iter().map(super::Detection::from).collect();
                            let b: Vec<_> = b.preds.iter().map(super::Detection::from).collect();
                            c.merge(&compare(&a, &b, MATCH_IOU));
                        }
                        model_report.comparison = Some(c);
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

    if all_devices && !swept.is_empty() {
        let results = BenchmarkResults::new(HardwareSummary::current(), versions, swept.clone())
            .with_sets(&sets);
        report.ranking = results.ranking();
        if !args.json && report.ranking.len() > 1 {
            print_ranking(&report.ranking);
        }
        if !args.json {
            print_thresholds(&swept, objective);
            print_frame_roc(&swept);
        }
        let path = results_path(&config_path);
        let merged = if args.no_save {
            results
        } else {
            let mut merged =
                BenchmarkResults::merge(BenchmarkResults::load_or_warn(&path), results);
            // Models kept from earlier runs are graded like this run's.
            merged.regrade(metric, weights);
            merged.save(&path)?;
            super::search::save_beside(&path, &merged, &swept);
            if !args.json {
                println!("\nresults saved to {}", path.display());
            }
            report.results_file = Some(path.display().to_string());
            merged
        };
        if let Some(out) = &args.report {
            export::write(&merged, out)?;
            if !args.json {
                println!("report written to {}", out.display());
            }
        }
        if args.apply {
            let picks: Vec<(String, Option<String>)> = swept
                .iter()
                .filter_map(|m| m.recommended.clone().map(|d| (m.model.clone(), Some(d))))
                .collect();
            let mut cfg = Config::load(&config_path)?;
            let changes = apply_model_devices(&mut cfg, &picks)?;
            if !changes.is_empty() {
                cfg.save(&config_path)?;
            }
            if !args.json {
                println!();
                if changes.is_empty() {
                    println!(
                        "--apply: {} already uses the recommended devices; nothing changed",
                        config_path.display()
                    );
                } else {
                    println!("--apply: updated {}:", config_path.display());
                    for c in &changes {
                        println!("  {}", c.describe());
                    }
                    println!("Restart the server to load the models on their new devices.");
                }
                for m in swept.iter().filter(|m| m.recommended.is_none()) {
                    println!("  {}: unchanged ({})", m.model, m.recommendation);
                }
                if cfg.force_cpu {
                    println!(
                        "note: force_cpu is on in the config and overrides every per-model device"
                    );
                }
            }
            report.applied = changes;
        }
        if args.apply_threshold {
            report.applied_thresholds = apply_thresholds(&swept, &config_path, args.json)?;
        }
        if args.threshold_search {
            let preds = super::search::StoredPreds::merge(None, &merged, &swept);
            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            return Ok(
                threshold_search_cli(&args, &config, &merged, &preds)? && report.errors.is_empty()
            );
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(report.errors.is_empty())
}

/// `--threshold-search`: search the stored predictions and print the results.
fn threshold_search_cli(
    args: &BenchArgs,
    config: &Config,
    results: &BenchmarkResults,
    preds: &super::search::StoredPreds,
) -> Result<bool> {
    use super::search::{SearchRequest, search};
    let req = SearchRequest {
        models: args.search_model.clone(),
        device: args.search_device.clone(),
        datasets: args.dataset.clone(),
        tags: args.tag.clone(),
        classes: args.class.clone(),
        objective: args
            .threshold_objective
            .unwrap_or(config.benchmark.threshold_objective),
        iou: args.iou.unwrap_or(0.5),
    };
    let current = |m: &str| {
        super::find_model(config, m).map(|c| {
            c.confidence_threshold
                .unwrap_or(config.confidence_threshold)
        })
    };
    let out = search(results, preds, &req, &current)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(out.errors.is_empty());
    }
    let mut filters = Vec::new();
    if !req.datasets.is_empty() {
        filters.push(format!("datasets {}", req.datasets.join(", ")));
    }
    if !req.tags.is_empty() {
        filters.push(format!("tags {}", req.tags.join(" + ")));
    }
    if !req.classes.is_empty() {
        filters.push(format!("classes {}", req.classes.join(", ")));
    }
    println!(
        "== Confidence threshold search ({}; IoU {}; {}; {:.1} ms, no inference)",
        req.objective.describe(),
        req.iou,
        if filters.is_empty() {
            "all images".to_string()
        } else {
            filters.join("; ")
        },
        out.elapsed_ms
    );
    println!(
        "  {:<20} {:<14} {:>5} {:>5} {:>5} {:>5}   {:>5} {:>5} {:>5} {:>5} {:>6}  {:>4} {:>4}  {:>5} {:>11} {:>11}",
        "model",
        "device",
        "conf",
        "P",
        "R",
        "F1",
        "best",
        "P",
        "R",
        "F1",
        "dF1",
        "imgs",
        "objs",
        "AUC",
        "TPR/FPR cf",
        "TPR/FPR bs"
    );
    for r in &out.results {
        let c = &r.configured;
        let b = r.best.as_ref();
        println!(
            "  {:<20} {:<14} {:>5.2} {:>5} {:>5} {:>5}   {:>5} {:>5} {:>5} {:>5} {:>6}  {:>4} {:>4}  {:>5} {:>11} {:>11}{}",
            r.model,
            r.device,
            c.threshold,
            pct1(c.precision),
            pct1(c.recall),
            pct1(c.f1),
            b.map_or("-".into(), |b| format!(
                "{:.2}{}",
                b.threshold,
                if b.met { "" } else { "!" }
            )),
            pct1(b.and_then(|b| b.point.precision)),
            pct1(b.and_then(|b| b.point.recall)),
            pct1(b.and_then(|b| b.point.f1)),
            match (b.and_then(|b| b.point.f1), c.f1) {
                (Some(x), Some(y)) => format!("{:+.1}", (x - y) * 100.0),
                _ => "-".into(),
            },
            r.images,
            r.gt,
            pct1(r.roc.as_ref().and_then(|x| x.auc)),
            tpr_fpr(c),
            b.map_or("-".into(), |b| tpr_fpr(&b.point)),
            if r.relative { "  (relative)" } else { "" }
        );
    }
    for r in &out.results {
        if let Some(roc) = &r.roc {
            println!(
                "  {} frame ROC AUC: macro {} micro {} ({} positive / {} negative frame samples); {}",
                r.model,
                pct1(roc.auc),
                pct1(roc.micro_auc),
                roc.positives,
                roc.negatives,
                class_aucs(roc)
            );
        }
    }
    for r in &out.results {
        let Some(b) = &r.best else { continue };
        println!();
        println!(
            "  {} on {}: best {:.2} (exact optimum {:.3}, near-optimal range {}){}",
            r.model,
            r.device,
            b.threshold,
            b.exact_threshold.unwrap_or(b.threshold),
            b.plateau
                .map_or("-".into(), |[lo, hi]| format!("{lo:.3}..{hi:.3}")),
            b.note.as_ref().map_or(String::new(), |n| format!("; {n}"))
        );
        println!(
            "    {:>5} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6}",
            "conf", "P", "R", "F1", "F2", "FP/img", "TPR", "FPR"
        );
        let mut extra = vec![r.configured, b.point];
        if let Some(c) = r.current {
            extra.push(c);
        }
        for p in super::threshold::table_points(&r.grid, &extra) {
            let mut marks = Vec::new();
            if (b.threshold - p.threshold).abs() < 1e-6 {
                marks.push("best");
            }
            if (r.configured.threshold - p.threshold).abs() < 1e-6 {
                marks.push("configured");
            }
            if r.current
                .is_some_and(|c| (c.threshold - p.threshold).abs() < 1e-6)
            {
                marks.push("in config now");
            }
            println!(
                "    {:>5.2} {:>6} {:>6} {:>6} {:>6} {:>7.2} {:>6} {:>6}{}",
                p.threshold,
                pct1(p.precision),
                pct1(p.recall),
                pct1(p.f1),
                pct1(p.f2),
                p.fp_per_image,
                pct1(p.frame.and_then(|f| f.tpr)),
                pct1(p.frame.and_then(|f| f.fpr)),
                if marks.is_empty() {
                    String::new()
                } else {
                    format!("  <- {}", marks.join(", "))
                }
            );
        }
        let groups: Vec<String> = r
            .by_dataset
            .iter()
            .chain(&r.per_class)
            .filter_map(|g| Some(format!("{} {:.2}", g.key, g.best.as_ref()?.threshold)))
            .collect();
        if !groups.is_empty() {
            println!("    best by dataset/class: {}", groups.join(", "));
        }
        let aucs: Vec<String> = r
            .by_dataset
            .iter()
            .filter_map(|g| Some(format!("{} {}", g.key, pct1(Some(g.roc_auc?)))))
            .collect();
        if !aucs.is_empty() {
            println!("    frame ROC AUC by dataset: {}", aucs.join(", "));
        }
    }
    for e in &out.errors {
        eprintln!("error: {e}");
    }
    println!("{BLUE_IRIS_NOTE}");
    Ok(out.errors.is_empty())
}

/// `--apply-threshold`: write the best thresholds of `swept` into the config file.
fn apply_thresholds(
    swept: &[ModelResult],
    config_path: &Path,
    quiet: bool,
) -> Result<Vec<ThresholdChange>> {
    let picks: Vec<(String, f32)> = swept
        .iter()
        .filter_map(|m| {
            let t = m.threshold.as_ref()?.apply_value()?;
            Some((m.model.clone(), t))
        })
        .collect();
    let mut cfg = Config::load(config_path)?;
    // Results may name models that are not configured (--model files): skip those.
    let picks: Vec<(String, f32)> = picks
        .into_iter()
        .filter(|(m, _)| super::find_model(&cfg, m).is_some())
        .collect();
    let changes = apply_model_thresholds(&mut cfg, &picks)?;
    if !changes.is_empty() {
        cfg.save(config_path)?;
    }
    if !quiet {
        println!();
        if changes.is_empty() {
            println!(
                "--apply-threshold: {} already uses the best thresholds; nothing changed",
                config_path.display()
            );
        } else {
            println!("--apply-threshold: updated {}:", config_path.display());
            for c in &changes {
                println!("  {}", c.describe());
            }
            println!("Restart the server to use the new thresholds.");
        }
        for m in swept {
            match &m.threshold {
                None => println!("  {}: unchanged (no threshold curve)", m.model),
                Some(a) if a.apply_value().is_none() => println!(
                    "  {}: unchanged ({})",
                    m.model,
                    a.best
                        .as_ref()
                        .and_then(|b| b.note.clone())
                        .unwrap_or_else(|| "no usable threshold".into())
                ),
                _ => {}
            }
        }
        println!("{BLUE_IRIS_NOTE}");
    }
    Ok(changes)
}

/// How the server threshold relates to Blue Iris's own confidence settings.
pub const BLUE_IRIS_NOTE: &str = "note: Blue Iris can send its own min_confidence with each request (it then replaces the server threshold for that request), and each camera's minimum confidence filters the returned objects again on top.";

fn pct1(v: Option<f64>) -> String {
    v.map_or("-".to_string(), |v| format!("{:.1}", v * 100.0))
}

/// "91.0/6.2" (frame TPR / FPR in %), "-" without frame rates.
fn tpr_fpr(p: &super::metrics::ThresholdPoint) -> String {
    p.frame
        .as_ref()
        .map_or("-".into(), |f| format!("{}/{}", pct1(f.tpr), pct1(f.fpr)))
}

/// "person 97.1 (120+/80-), vehicle 93.0 (..)".
fn class_aucs(roc: &super::roc::FrameRoc) -> String {
    roc.per_class
        .iter()
        .map(|c| {
            format!(
                "{} {} ({}+/{}-)",
                c.class,
                pct1(c.auc),
                c.positives,
                c.negatives
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Frame-level ROC per model on the advice device: AUC, and frame TPR / FPR at the configured
/// and the best threshold.
fn print_frame_roc(swept: &[ModelResult]) {
    let rows: Vec<(&ModelResult, &ThresholdAdvice, &super::roc::FrameRoc)> = swept
        .iter()
        .filter_map(|m| {
            let a = m.threshold.as_ref()?;
            Some((m, a, a.roc.as_ref()?))
        })
        .collect();
    if rows.is_empty() {
        return;
    }
    println!();
    println!(
        "== Frame-level ROC per model (a frame alerts when its top confidence >= the threshold; %)"
    );
    println!(
        "  {:<20} {:<14} {:>5} {:>5}  {:>5} {:>11}  {:>5} {:>11}  per class",
        "model", "device", "AUC", "micro", "conf", "TPR/FPR", "best", "TPR/FPR"
    );
    for (m, a, roc) in rows {
        println!(
            "  {:<20} {:<14} {:>5} {:>5}  {:>5.2} {:>11}  {:>5} {:>11}  {}",
            m.model,
            a.device,
            pct1(roc.auc),
            pct1(roc.micro_auc),
            a.configured.threshold,
            tpr_fpr(&a.configured),
            a.best
                .as_ref()
                .map_or("-".into(), |b| format!("{:.2}", b.threshold)),
            a.best.as_ref().map_or("-".into(), |b| tpr_fpr(&b.point)),
            class_aucs(roc)
        );
    }
    println!(
        "  (AUC from the 0.05+ predictions: frames below 0.05 score 0, which slightly understates it)"
    );
}

/// The best-threshold summary across models.
fn print_thresholds(swept: &[ModelResult], objective: Objective) {
    let rows: Vec<(&ModelResult, &ThresholdAdvice)> = swept
        .iter()
        .filter_map(|m| m.threshold.as_ref().map(|a| (m, a)))
        .collect();
    if rows.is_empty() {
        return;
    }
    println!();
    println!(
        "== Best confidence threshold per model ({}; P/R/F1 in %)",
        objective.describe()
    );
    println!(
        "  {:<20} {:<14} {:>5} {:>5} {:>5} {:>5}   {:>5} {:>5} {:>5} {:>5}  {:>4} {:>4} {:>5}  night",
        "model", "device", "conf", "P", "R", "F1", "best", "P", "R", "F1", "F2", "P90", "dF1"
    );
    for (m, a) in rows {
        let c = &a.configured;
        let alt = |o: Objective| {
            a.alternatives
                .iter()
                .find(|p| p.objective == o.to_string())
                .map_or("-".to_string(), |p| {
                    format!("{:.2}{}", p.threshold, if p.met { "" } else { "!" })
                })
        };
        let night = a
            .by_dataset
            .iter()
            .chain(&a.by_tag)
            .find(|g| g.key.contains("night"))
            .and_then(|g| {
                g.best
                    .as_ref()
                    .map(|b| format!("{:.2} ({})", b.threshold, g.key))
            })
            .unwrap_or_else(|| "-".into());
        match &a.best {
            Some(b) => println!(
                "  {:<20} {:<14} {:>5.2} {:>5} {:>5} {:>5}   {:>5.2} {:>5} {:>5} {:>5}  {:>4} {:>4} {:>5}  {}{}",
                m.model,
                a.device,
                c.threshold,
                pct1(c.precision),
                pct1(c.recall),
                pct1(c.f1),
                b.threshold,
                pct1(b.point.precision),
                pct1(b.point.recall),
                pct1(b.point.f1),
                alt(Objective::F2),
                alt(Objective::Precision(0.9)),
                match (b.point.f1, c.f1) {
                    (Some(x), Some(y)) => format!("{:+.1}", (x - y) * 100.0),
                    _ => "-".into(),
                },
                night,
                if a.relative { "  (relative)" } else { "" }
            ),
            None => println!(
                "  {:<20} {:<14} no ground truth for a curve",
                m.model, a.device
            ),
        }
    }
    println!("  (! = precision 90% not reached; the most precise threshold is shown)");
}

/// The compact threshold table of one model's advice.
fn print_threshold_table(a: &ThresholdAdvice) {
    let best = a.best.as_ref().map(|b| b.threshold);
    let mut extra = vec![a.configured];
    if let Some(b) = &a.best {
        extra.push(b.point);
    }
    println!(
        "  confidence threshold on {} ({}{}):",
        a.device,
        a.objective().describe(),
        if a.relative { ", relative" } else { "" }
    );
    println!(
        "    {:>5} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6}",
        "conf", "P", "R", "F1", "F2", "FP/img", "TPR", "FPR"
    );
    for p in super::threshold::table_points(&a.curve, &extra) {
        let mut marks = Vec::new();
        if best.is_some_and(|b| (b - p.threshold).abs() < 1e-6) {
            marks.push("best");
        }
        if (a.configured.threshold - p.threshold).abs() < 1e-6 {
            marks.push("configured");
        }
        println!(
            "    {:>5.2} {:>6} {:>6} {:>6} {:>6} {:>7.2} {:>6} {:>6}{}",
            p.threshold,
            pct1(p.precision),
            pct1(p.recall),
            pct1(p.f1),
            pct1(p.f2),
            p.fp_per_image,
            pct1(p.frame.and_then(|f| f.tpr)),
            pct1(p.frame.and_then(|f| f.fpr)),
            if marks.is_empty() {
                String::new()
            } else {
                format!("  <- {}", marks.join(", "))
            }
        );
    }
    let groups: Vec<String> = a
        .by_dataset
        .iter()
        .chain(&a.by_tag)
        .filter_map(|g| {
            let b = g.best.as_ref()?;
            Some(format!("{} {:.2}", g.key, b.threshold))
        })
        .collect();
    if !groups.is_empty() {
        println!("    best by dataset/tag: {}", groups.join(", "));
    }
    let classes: Vec<String> = a
        .per_class
        .iter()
        .filter_map(|g| Some(format!("{} {:.2}", g.key, g.best.as_ref()?.threshold)))
        .collect();
    if !classes.is_empty() {
        println!("    best by class: {}", classes.join(", "));
    }
}

fn print_ranking(rows: &[super::report::RankRow]) {
    println!();
    let metric = rows.first().map_or(AccuracyMetric::Ap50, |r| r.metric);
    println!(
        "== Best model for this machine (each on its recommended device; accuracy graded by {})",
        metric.label()
    );
    println!(
        "  {:>2} {:<20} {:<16} {:>7} {:>8} {:>5} {:>9} {:>6} {:>6}",
        "#", "model", "device", "overall", "accuracy", "speed", "p50 ms", "AP50", "AUC"
    );
    for r in rows {
        println!(
            "  {:>2} {:<20} {:<16} {:>7} {:>8} {:>5} {:>9.1} {:>6} {:>6}",
            r.rank,
            r.model,
            r.device,
            r.overall.to_string(),
            r.accuracy.map_or("-".to_string(), |a| {
                if r.relative {
                    format!("{a}*")
                } else {
                    a.to_string()
                }
            }),
            r.speed.to_string(),
            r.p50_ms,
            r.ap50
                .map_or("-".to_string(), |v| format!("{:.1}", v * 100.0)),
            pct1(r.roc_auc)
        );
    }
}

/// The `--all-devices` JSON/table view of a sweep.
fn model_report(m: &ModelResult) -> Result<ModelReport> {
    let runs: Vec<RunResult> = m
        .devices
        .iter()
        .filter(|d| d.ok())
        .filter_map(|d| d.run.clone())
        .collect();
    let mut skipped = m.skipped.clone();
    skipped.extend(
        m.devices
            .iter()
            .filter(|d| !d.ok())
            .map(|d| match (&d.error, &d.run) {
                (Some(e), _) => format!("{}: {e}", d.device),
                (None, Some(r)) => {
                    format!("{}: fell back to {}, result dropped", d.device, r.device)
                }
                (None, None) => format!("{}: did not run", d.device),
            }),
    );
    let primary = m
        .recommended
        .as_ref()
        .and_then(|rec| m.devices.iter().find(|d| &d.device == rec))
        .or_else(|| m.fastest())
        .and_then(|d| d.run.clone());
    let Some(primary) = primary else {
        bail!(
            "no device option could run this model ({})",
            skipped.join("; ")
        );
    };
    Ok(ModelReport {
        primary,
        cpu: None,
        comparison: None,
        note: None,
        all_devices: runs,
        skipped_devices: skipped,
        recommended: m.recommended.clone(),
        recommendation: Some(m.recommendation.clone()),
        reference: m.reference.clone(),
        threshold: m.threshold.clone(),
    })
}

/// Short agreement text for tables: "ref", "ok", "minor (2/3)", "BAD (0/3)", "-".
pub fn agreement_text(r: &RunResult) -> String {
    match &r.agreement {
        None => "-".to_string(),
        Some(a) => {
            let total = a.matched + a.only_device.max(a.only_reference);
            match a.verdict {
                Verdict::Reference => "ref".to_string(),
                Verdict::Agrees => "ok".to_string(),
                Verdict::Minor => format!("minor ({}/{total})", a.matched),
                Verdict::Disagrees => format!("BAD ({}/{total})", a.matched),
            }
        }
    }
}

/// Comparison table of an `--all-devices` model report, fastest first, with grades, then the
/// recommended device's accuracy breakdown.
fn print_all_devices(result: &ModelResult, m: &ModelReport) {
    println!();
    println!(
        "== {}: all runnable devices ({} images x {} runs, ms)",
        m.primary.model, m.primary.images, m.primary.repeat
    );
    println!(
        "  {:<16} {:<26} {:>7} {:>9} {:>9} {:>9} {:>7} {:>6} {:>6} {:>6} {:>5} {:>5} {:<7} {:<12}",
        "requested",
        "device",
        "compile",
        "infer p50",
        "total p50",
        "total p95",
        "req/s",
        "AP50",
        "AP-95",
        "AUC",
        "P",
        "R",
        "O/A/S",
        "agreement"
    );
    let mut runs: Vec<&RunResult> = m.all_devices.iter().collect();
    runs.sort_by(|a, b| a.stages_ms.total.p50.total_cmp(&b.stages_ms.total.p50));
    let fastest = runs
        .first()
        .map(|r| r.stages_ms.total.p50)
        .unwrap_or_default();
    let pct = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{:.1}", v * 100.0));
    for r in &runs {
        let name = if r.device_name.is_empty() {
            r.device.clone()
        } else {
            format!("{} ({})", r.device, r.device_name)
        };
        let name: String = name.chars().take(26).collect();
        let mut marks = Vec::new();
        if r.stages_ms.total.p50 == fastest && runs.len() > 1 {
            marks.push("fastest");
        }
        if m.recommended.as_deref() == Some(r.requested_device.as_str()) {
            marks.push("recommended");
        }
        if result.configured.as_deref() == Some(r.requested_device.as_str()) {
            marks.push("configured");
        }
        let acc = r.accuracy.as_ref().map(|a| &a.overall);
        let grades = r.grades.as_ref().map_or("-".to_string(), |g| {
            format!(
                "{}/{}{}/{}",
                g.overall,
                g.accuracy.map_or("-".to_string(), |a| a.to_string()),
                if g.relative { "*" } else { "" },
                g.speed
            )
        });
        println!(
            "  {:<16} {:<26} {:>7.0} {:>9.2} {:>9.2} {:>9.2} {:>7.1} {:>6} {:>6} {:>6} {:>5} {:>5} {:<7} {:<12}{}",
            r.requested_device,
            name,
            r.compile_ms,
            r.stages_ms.infer.p50,
            r.stages_ms.total.p50,
            r.stages_ms.total.p95,
            r.throughput_fps,
            pct(acc.and_then(|a| a.ap50)),
            pct(acc.and_then(|a| a.ap50_95)),
            pct(r.accuracy.as_ref().and_then(|a| a.roc_auc())),
            pct(acc.and_then(|a| a.precision)),
            pct(acc.and_then(|a| a.recall)),
            grades,
            agreement_text(r),
            if marks.is_empty() {
                String::new()
            } else {
                format!("  <- {}", marks.join(", "))
            }
        );
    }
    for s in &m.skipped_devices {
        println!("  skipped {s}");
    }
    match &m.recommended {
        Some(d) => println!("  recommended: {d} ({})", result.recommendation),
        None => println!("  recommended: none ({})", result.recommendation),
    }
    let p = &m.primary;
    match &p.accuracy {
        Some(a) => {
            println!(
                "  accuracy on {} ({}; P/R at {:.2}): {} | small/medium/large recall {}",
                p.requested_device,
                a.ground_truth,
                a.threshold,
                a.classes.join(", "),
                a.overall
                    .by_size
                    .iter()
                    .map(|b| format!("{} {}", b.bucket, pct(b.recall)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            for b in a.by_dataset.iter().chain(&a.by_tag) {
                println!(
                    "    {:<24} {:>4} img {:>5} obj  AP50 {:>5}  AUC {:>5}  P {:>5}  R {:>5}{}",
                    b.key,
                    b.images,
                    b.gt,
                    pct(b.ap50),
                    pct(b.roc_auc),
                    pct(b.precision),
                    pct(b.recall),
                    if b.relative { "  (relative)" } else { "" }
                );
            }
        }
        None => {
            if let Some(n) = &p.accuracy_note {
                println!("  accuracy not scored: {n}");
            }
        }
    }
    if let Some(a) = &result.threshold {
        print_threshold_table(a);
    }
    if p.images > 1 {
        // Every bucket; one without images is n/a (it does not enter any grade).
        for bucket in super::images::RESOLUTION_BUCKETS {
            match p.by_resolution.iter().find(|b| b.bucket == bucket) {
                Some(b) => println!(
                    "    res:{:<6} {:>4} img  pre {:>7.2}  infer {:>7.2}  post {:>6.2}  total p50 {:>7.2} p95 {:>7.2}",
                    b.bucket,
                    b.images,
                    b.pre_p50,
                    b.infer_p50,
                    b.post_p50,
                    b.total_p50,
                    b.total_p95
                ),
                None => println!("    res:{bucket:<6}    0 img  n/a"),
            }
        }
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
        "  throughput {:.1} img/s | {} detections (first image) | {} image(s): {}",
        r.throughput_fps,
        r.detections.len(),
        r.images,
        r.image
    );
    if let Some(a) = &r.accuracy {
        let pct = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{:.1}", v * 100.0));
        println!(
            "  accuracy ({}): AP50 {} AP50-95 {} frame ROC AUC {} P {} R {} F1 {}",
            a.ground_truth,
            pct(a.overall.ap50),
            pct(a.overall.ap50_95),
            pct(a.roc_auc()),
            pct(a.overall.precision),
            pct(a.overall.recall),
            pct(a.overall.f1)
        );
    }
    if let Some(g) = &r.grades {
        println!(
            "  grades: overall {} (accuracy {} by {}, speed {})",
            g.overall,
            g.accuracy.map_or("-".to_string(), |a| a.to_string()),
            g.metric.label(),
            g.speed
        );
    }
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
    use clap::Parser;

    #[derive(Debug, Parser)]
    struct Cli {
        #[command(flatten)]
        args: BenchArgs,
    }

    fn parse(argv: &[&str]) -> Result<BenchArgs, clap::Error> {
        Cli::try_parse_from(argv).map(|c| c.args)
    }

    #[test]
    fn args_parse() {
        let a = parse(&[
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
        assert_eq!(a.repeat, Some(20));
        assert_eq!(a.warmup, None);
        assert!(a.compare_cpu && a.json);
        let a = parse(&["x"]).unwrap();
        assert_eq!(a.repeat, None);
        assert!(!a.all_devices && !a.apply && !a.no_save);
        let a = parse(&[
            "x",
            "--dataset",
            "coco-cctv",
            "--dataset",
            "dir:/srv/a",
            "--max-images",
            "10",
            "--report",
            "r.html",
            "--accuracy-weight",
            "0.5",
        ])
        .unwrap();
        assert_eq!(a.dataset, ["coco-cctv", "dir:/srv/a"]);
        assert_eq!(a.max_images, Some(10));
        assert!(parse(&["x", "--image", "a.jpg", "--dataset", "sample"]).is_err());
        assert!(
            parse(&["x", "--gt", "g.json"]).is_err(),
            "--gt needs --images"
        );
        assert!(parse(&["x", "--report", "r.md", "--device", "cpu"]).is_err());
    }

    #[test]
    fn device_all_devices_and_apply_flags() {
        let a = parse(&["x", "--device", "ORT:CUDA:1"]).unwrap();
        assert_eq!(a.device.as_deref(), Some("ORT:CUDA:1"));
        assert!(parse(&["x", "--device", "vulkan"]).is_err());
        let a = parse(&["x", "--all-devices"]).unwrap();
        assert!(a.all_devices);
        assert!(parse(&["x", "--all-devices", "--device", "cpu"]).is_err());
        assert!(parse(&["x", "--all-devices", "--force-cpu"]).is_err());
        let a = parse(&["x", "--all-devices", "--apply", "--config", "c.json"]).unwrap();
        assert!(a.apply);
        assert!(parse(&["x", "--apply", "--model", "a.onnx"]).is_err());
        assert!(parse(&["x", "--apply", "--device", "cpu"]).is_err());
        // Thresholds: separate from --apply, both allowed together.
        let a = parse(&[
            "x",
            "--apply",
            "--apply-threshold",
            "--threshold-objective",
            "precision:0.9",
        ])
        .unwrap();
        assert!(a.apply && a.apply_threshold);
        assert_eq!(a.threshold_objective, Some(Objective::Precision(0.9)));
        let a = parse(&["x", "--apply-threshold"]).unwrap();
        assert!(a.apply_threshold && !a.apply && a.threshold_objective.is_none());
        assert!(parse(&["x", "--threshold-objective", "f3"]).is_err());
        assert!(parse(&["x", "--apply-threshold", "--model", "a.onnx"]).is_err());
        assert!(parse(&["x", "--apply-threshold", "--min-confidence", "0.3"]).is_err());
    }

    #[test]
    fn apply_needs_an_existing_config() {
        let dir = std::env::temp_dir().join(format!("bop-bcli-{}", uuid::Uuid::new_v4()));
        let a = parse(&[
            "x",
            "--apply",
            "--config",
            dir.join("missing.json").to_str().unwrap(),
        ])
        .unwrap();
        let err = run(a).unwrap_err();
        assert!(
            format!("{err:#}").contains("needs a config file"),
            "{err:#}"
        );
        let a = parse(&[
            "x",
            "--apply-threshold",
            "--config",
            dir.join("missing.json").to_str().unwrap(),
        ])
        .unwrap();
        let err = run(a).unwrap_err();
        assert!(
            format!("{err:#}").contains("--apply-threshold needs a config file"),
            "{err:#}"
        );
    }

    #[test]
    fn threshold_search_uses_stored_predictions() {
        assert!(
            parse(&["x", "--tag", "night"]).is_err(),
            "--tag needs --threshold-search"
        );
        assert!(parse(&["x", "--threshold-search", "--device", "cpu"]).is_err());
        let a = parse(&[
            "x",
            "--threshold-search",
            "--tag",
            "night",
            "--class",
            "person",
            "--iou",
            "0.6",
            "--threshold-objective",
            "recall:0.9",
        ])
        .unwrap();
        assert_eq!((a.tag.len(), a.class.len(), a.iou), (1, 1, Some(0.6)));
        assert_eq!(a.threshold_objective, Some(Objective::Recall(0.9)));

        let dir = std::env::temp_dir().join(format!("bop-bsearch-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("cfg.json");
        Config::default().save(&cfg).unwrap();
        let (results, preds) = super::super::search::tests::fixture();
        results.save(&results_path(&cfg)).unwrap();
        preds.save(&super::super::search::preds_path(&cfg)).unwrap();
        let c = cfg.to_str().unwrap();
        assert!(
            run(parse(&["x", "--threshold-search", "--json", "--config", c]).unwrap()).unwrap()
        );
        assert!(
            run(parse(&[
                "x",
                "--threshold-search",
                "--tag",
                "night",
                "--search-device",
                "openvino:cpu",
                "--config",
                c
            ])
            .unwrap())
            .unwrap()
        );
        // A model without stored predictions: reported, not fatal; exit status false.
        assert!(
            !run(parse(&[
                "x",
                "--threshold-search",
                "--search-model",
                "m",
                "--search-model",
                "zzz",
                "--json",
                "--config",
                c
            ])
            .unwrap())
            .unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn apply_thresholds_writes_the_config() {
        let dir = std::env::temp_dir().join(format!("bop-bthr-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cfg.json");
        let cfg = Config {
            models: ["a", "b"]
                .iter()
                .map(|n| ModelConfig {
                    name: Some(n.to_string()),
                    path: format!("models/{n}.onnx").into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        cfg.save(&path).unwrap();
        let mk = |name: &str, t: Option<f32>| {
            let mut m = ModelResult::failed(name, "", String::new());
            m.threshold = t.map(super::super::threshold::test_advice);
            m
        };
        let swept = [mk("a", Some(0.31)), mk("b", None), mk("c.onnx", Some(0.2))];
        let changes = apply_thresholds(&swept, &path, true).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].model, "a");
        let saved = Config::load(&path).unwrap();
        assert_eq!(saved.models[0].confidence_threshold, Some(0.31));
        assert_eq!(saved.models[1].confidence_threshold, None);
        // Again: nothing to change.
        assert!(apply_thresholds(&swept, &path, true).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dataset_precedence() {
        let mut config = Config::default();
        config.benchmark.datasets = vec![DatasetRef::Id("sample".into())];
        let a = parse(&["x"]).unwrap();
        assert_eq!(dataset_refs(&a, &config), config.benchmark.datasets);
        let a = parse(&[
            "x",
            "--dataset",
            "coco-cctv",
            "--images",
            "/srv/a",
            "--gt",
            "/srv/g.json",
        ])
        .unwrap();
        let refs = dataset_refs(&a, &config);
        assert_eq!(refs[0], DatasetRef::Id("coco-cctv".into()));
        assert_eq!(refs[1].dir().unwrap().1, Some(PathBuf::from("/srv/g.json")));
        // An unknown dataset is a clear error before any runtime is loaded.
        let a = parse(&["x", "--dataset", "nope", "--model", "m.onnx"]).unwrap();
        let err = load_sets(&a, &config).unwrap_err();
        assert!(format!("{err:#}").contains("valid ids"), "{err:#}");
    }

    #[test]
    fn agreement_texts() {
        use super::super::report::Agreement;
        let mut r = RunResult {
            model: "m".into(),
            path: String::new(),
            family: String::new(),
            requested_device: String::new(),
            device: String::new(),
            device_name: String::new(),
            spec: String::new(),
            fell_back: false,
            input: String::new(),
            image: String::new(),
            image_size: String::new(),
            images: 1,
            compile_ms: 0.0,
            cache: String::new(),
            warmup_iterations: 0,
            warmup_ms: 0.0,
            repeat: 1,
            stages_ms: Default::default(),
            throughput_fps: 0.0,
            detections: Vec::new(),
            by_resolution: Vec::new(),
            accuracy: None,
            accuracy_note: None,
            grades: None,
            agreement: None,
            per_image: Vec::new(),
            eval_preds: Vec::new(),
        };
        assert_eq!(agreement_text(&r), "-");
        let a = |verdict, matched, only_device, only_reference| Agreement {
            reference: "openvino:cpu".into(),
            matched,
            only_device,
            only_reference,
            max_confidence_diff: 0.0,
            mean_confidence_diff: 0.0,
            score: 0.0,
            verdict,
        };
        r.agreement = Some(a(Verdict::Minor, 2, 0, 1));
        assert_eq!(agreement_text(&r), "minor (2/3)");
        r.agreement = Some(a(Verdict::Disagrees, 0, 0, 3));
        assert_eq!(agreement_text(&r), "BAD (0/3)");
        r.agreement = Some(a(Verdict::Reference, 3, 0, 0));
        assert_eq!(agreement_text(&r), "ref");
    }
}
