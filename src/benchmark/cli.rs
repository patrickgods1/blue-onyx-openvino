//! Command line front end of the benchmark: the `blue-onyx-prism-benchmark` binary and the
//! `blue-onyx-prism benchmark` subcommand both parse [`BenchArgs`] and call [`main`].
//!
//! ```text
//! blue-onyx-prism-benchmark --model models/IPcam-general.onnx --family yolo5 --device CPU --repeat 20
//! blue-onyx-prism-benchmark --model models/yolo26s.xml --compare-cpu      # GPU vs CPU + confidence diff
//! blue-onyx-prism-benchmark --json                                         # every enabled model in the config
//! blue-onyx-prism-benchmark --all-devices --apply                          # sweep, save, set per-model devices
//! ```

use super::images::{self, ImageSet};
use super::report::{HardwareSummary, RuntimeVersions, Verdict};
use super::{
    Bench, BenchmarkResults, Comparison, Job, MATCH_IOU, ModelResult, Phase, RunResult, Stats,
    SweepOptions, compare, config_models, configured_device, export, job_for_config, job_for_file,
    pick_reference, pseudo_ground_truth, reference_device, results_path, sweep,
};
use crate::backend::{CoreOptions, OrtOptions, Runtimes, libs, spec};
use crate::config::{Config, DatasetRef, DeviceChange, ModelConfig, apply_model_devices};
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
    /// Run every runnable device option (see `list-devices`) for each model, grade accuracy and
    /// speed, compare detections with the CPU reference, print the recommended device, and save
    /// the results to `benchmark.json` next to the config file (web UI Benchmark page). Cannot
    /// be combined with --device, --force-cpu or --compare-cpu.
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu"])]
    pub all_devices: bool,
    /// Implies --all-devices: write each benchmarked config model's recommended device into its
    /// `device` in the config file and print what changed (restart the server to apply).
    #[arg(long, conflicts_with_all = ["device", "force_cpu", "compare_cpu", "model"])]
    pub apply: bool,
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
    if args.apply && !config_path.exists() {
        bail!(
            "--apply needs a config file; {} does not exist",
            config_path.display()
        );
    }
    if let Some(w) = args.accuracy_weight
        && !(0.0..=1.0).contains(&w)
    {
        bail!("--accuracy-weight must be between 0 and 1");
    }
    let all_devices = args.all_devices || args.apply || args.report.is_some();
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
                devices: None,
                configured,
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
        let path = results_path(&config_path);
        let merged = if args.no_save {
            results
        } else {
            let merged = BenchmarkResults::merge(BenchmarkResults::load_or_warn(&path), results);
            merged.save(&path)?;
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
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(report.errors.is_empty())
}

fn print_ranking(rows: &[super::report::RankRow]) {
    println!();
    println!("== Best model for this machine (each on its recommended device)");
    println!(
        "  {:>2} {:<20} {:<16} {:>7} {:>8} {:>5} {:>9} {:>6}",
        "#", "model", "device", "overall", "accuracy", "speed", "p50 ms", "AP50"
    );
    for r in rows {
        println!(
            "  {:>2} {:<20} {:<16} {:>7} {:>8} {:>5} {:>9.1} {:>6}",
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
                .map_or("-".to_string(), |v| format!("{:.1}", v * 100.0))
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
        "  {:<16} {:<26} {:>7} {:>9} {:>9} {:>9} {:>7} {:>6} {:>6} {:>5} {:>5} {:<7} {:<12}",
        "requested",
        "device",
        "compile",
        "infer p50",
        "total p50",
        "total p95",
        "req/s",
        "AP50",
        "AP-95",
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
            "  {:<16} {:<26} {:>7.0} {:>9.2} {:>9.2} {:>9.2} {:>7.1} {:>6} {:>6} {:>5} {:>5} {:<7} {:<12}{}",
            r.requested_device,
            name,
            r.compile_ms,
            r.stages_ms.infer.p50,
            r.stages_ms.total.p50,
            r.stages_ms.total.p95,
            r.throughput_fps,
            pct(acc.and_then(|a| a.ap50)),
            pct(acc.and_then(|a| a.ap50_95)),
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
                    "    {:<24} {:>4} img {:>5} obj  AP50 {:>5}  P {:>5}  R {:>5}{}",
                    b.key,
                    b.images,
                    b.gt,
                    pct(b.ap50),
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
            "  accuracy ({}): AP50 {} AP50-95 {} P {} R {} F1 {}",
            a.ground_truth,
            pct(a.overall.ap50),
            pct(a.overall.ap50_95),
            pct(a.overall.precision),
            pct(a.overall.recall),
            pct(a.overall.f1)
        );
    }
    if let Some(g) = &r.grades {
        println!(
            "  grades: overall {} (accuracy {}, speed {})",
            g.overall,
            g.accuracy.map_or("-".to_string(), |a| a.to_string()),
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
