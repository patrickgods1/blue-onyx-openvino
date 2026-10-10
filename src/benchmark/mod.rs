//! Benchmark core shared by the `blue-onyx-prism-benchmark` binary, the `benchmark` subcommand
//! and the web UI's Benchmark page.
//!
//! A run compiles a model on a device, warms it up, then sends every image of the chosen
//! datasets ([`images`]) through the production per-request pipeline (decode -> preprocess ->
//! infer -> postprocess, exactly as `worker.rs` runs it), `repeat` timed times per image. Timing
//! uses the model's configured confidence threshold; the same inference outputs are also
//! post-processed (untimed) at [`EVAL_CONFIDENCE`] for AP. Accuracy ([`metrics`]) is measured
//! against the dataset's ground truth, or against pseudo ground truth from a reference model
//! ([`pseudo_ground_truth`]) for datasets without it; [`grade`] turns accuracy and speed into
//! letter grades. [`sweep`] runs one model on every chosen device, checks each device's
//! detections against a CPU reference device and recommends a device ([`report::recommend`]).
//! [`report`] holds the persisted results (`benchmark.json` next to the config file),
//! [`service`] the background run behind `/v1/benchmark`, [`cli`] the command line, and
//! [`export`] the standalone HTML / Markdown report.
//!
//! Runtimes are shared through a `Mutex<Runtimes>` that is held only while compiling, so the
//! server can benchmark through the registry's own runtimes (one OpenVINO `Core`, the ONNX
//! Runtime flavor the process loaded) while the workers keep serving.

pub mod cli;
pub mod export;
pub mod grade;
pub mod images;
pub mod metrics;
pub mod report;
pub mod service;

pub use report::{
    Agreement, BenchmarkResults, DeviceResult, ModelResult, NOISE_MARGIN, RESULTS_FILE,
    Recommendation, Verdict, recommend, results_path,
};

use crate::api::Prediction;
use crate::backend::spec::{self, Device, Runtime, Target};
use crate::backend::{Backend, Candidate, LoadRequest, Runtimes, Selection};
use crate::config::{Config, ModelConfig};
use crate::model::preprocess::Preprocessor;
use crate::model::{ModelFamilyKind, PostParams};
use crate::registry::resolve_class_names;
use anyhow::{Context, Result, bail};
use grade::{Grades, Weights};
use images::{GroundTruth, ImageSet};
use metrics::{ClassMap, Counts, EvalImage, GtBox, PredBox, Summary};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Default benchmark image, embedded so the benchmark works without any file.
pub const DEFAULT_IMAGE: &[u8] = include_bytes!("../../tests/data/dog_bike_car.jpg");
pub const DEFAULT_IMAGE_NAME: &str = "dog_bike_car.jpg (embedded)";
/// Minimum IoU for a detection and a reference detection of the same label to count as the same
/// object (device agreement).
pub const MATCH_IOU: f32 = 0.5;
/// Confidence threshold of the (untimed) post-processing used for AP.
pub const EVAL_CONFIDENCE: f32 = 0.05;
/// Pseudo ground truth: reference detections at or above this confidence are objects...
pub const PSEUDO_GT_CONFIDENCE: f32 = 0.5;
/// ...and those between this and [`PSEUDO_GT_CONFIDENCE`] are `ignore` regions (borderline
/// objects that neither count as misses nor as false positives).
pub const PSEUDO_IGNORE_CONFIDENCE: f32 = 0.3;
/// Predictions kept per image in the results (drill-down), most confident first.
pub const MAX_STORED_PREDS: usize = 100;

/// min / mean / p50 / p95 / max in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Stats {
    pub min: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

impl Stats {
    /// Nearest-rank percentiles over `samples` (ms). All zeros for an empty slice.
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
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
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let rank = ((p / 100.0) * n as f64).ceil() as usize;
    sorted[rank.clamp(1, n) - 1]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StageStats {
    pub decode: Stats,
    pub preprocess: Stats,
    pub infer: Stats,
    pub postprocess: Stats,
    /// The whole request (sum of the four stages).
    pub total: Stats,
}

/// One detection (pixel box), as reported to Blue Iris.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub label: String,
    pub confidence: f32,
    pub x_min: usize,
    pub y_min: usize,
    pub x_max: usize,
    pub y_max: usize,
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

impl From<&ImagePred> for Detection {
    fn from(p: &ImagePred) -> Self {
        let c = |v: f32| v.max(0.0).round() as usize;
        Self {
            label: p.label.clone(),
            confidence: p.confidence,
            x_min: c(p.bbox[0]),
            y_min: c(p.bbox[1]),
            x_max: c(p.bbox[2]),
            y_max: c(p.bbox[3]),
        }
    }
}

/// Speed of the images of one resolution bucket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucketSpeed {
    /// `sd`, `hd`, `fhd`, `4mp+`.
    pub bucket: String,
    pub images: usize,
    /// Decode + preprocess p50.
    pub pre_p50: f64,
    pub infer_p50: f64,
    pub post_p50: f64,
    pub total_p50: f64,
    pub total_p95: f64,
}

/// A stored prediction of one image (at the model's confidence threshold).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImagePred {
    pub label: String,
    pub confidence: f32,
    pub bbox: [f32; 4],
    /// True / false positive against the (pseudo) ground truth; None = not scored (class not
    /// scored, ignored region, or no ground truth).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp: Option<bool>,
}

/// One image of a device run (drill-down).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageRun {
    /// Dataset id.
    pub set: String,
    pub file: String,
    /// p50 over the image's timed runs.
    pub total_ms: f64,
    pub infer_ms: f64,
    pub preds: Vec<ImagePred>,
    /// Per ground-truth object of the image (dataset order): matched; None = not scored.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gt_matched: Vec<Option<bool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counts: Option<Counts>,
}

/// Metrics of a subset of images (one dataset or one tag).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Breakdown {
    /// Dataset id or tag.
    pub key: String,
    pub images: usize,
    /// Scored ground-truth objects.
    pub gt: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    /// Against pseudo ground truth.
    pub relative: bool,
}

impl Breakdown {
    fn of(key: &str, s: &Summary, relative: bool) -> Self {
        Self {
            key: key.to_string(),
            images: s.images,
            gt: s.counts.tp + s.counts.fn_,
            precision: s.precision,
            recall: s.recall,
            f1: s.f1,
            ap50: s.ap50,
            ap50_95: s.ap50_95,
            relative,
        }
    }
}

/// Accuracy of one model on one device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccuracyReport {
    /// "ground truth" or "relative to <reference>, not ground truth".
    pub ground_truth: String,
    pub relative: bool,
    /// Scored classes in the model's label space.
    pub classes: Vec<String>,
    /// Confidence threshold of precision / recall / F1.
    pub threshold: f32,
    pub overall: Summary,
    pub by_dataset: Vec<Breakdown>,
    pub by_tag: Vec<Breakdown>,
    /// AP over every class of the model (not only the CCTV ones), same images; None when the
    /// model has no other classes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all_classes: Option<AllClasses>,
}

/// AP over all of a model's classes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllClasses {
    /// Classes with ground truth in the images.
    pub classes: usize,
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
}

/// One image in a class map's label space: ground truth per original object (None = not
/// scored) and the evaluation input. Applies the set's `scored_labels`.
fn eval_input(
    map: &ClassMap,
    set: &ImageSet,
    gt: &[GtBox],
    eval_preds: &[PredBox],
) -> (Vec<Option<GtBox>>, EvalImage, Option<Vec<String>>) {
    let allowed = set.scored_labels.as_ref().map(|l| map.restrict(l));
    let ok = |c: &str| allowed.as_ref().is_none_or(|a| a.iter().any(|x| x == c));
    let mapped: Vec<Option<GtBox>> = gt
        .iter()
        .map(|g| {
            map.gt_class(&g.label).filter(|c| ok(c)).map(|c| GtBox {
                label: c.to_string(),
                ..g.clone()
            })
        })
        .collect();
    let eval = EvalImage {
        gt: mapped.iter().flatten().cloned().collect(),
        preds: map
            .map_preds(eval_preds)
            .into_iter()
            .filter(|p| ok(&p.label))
            .collect(),
    };
    (mapped, eval, allowed)
}

/// Result of running one model on one device over the datasets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    pub model: String,
    pub path: String,
    pub family: String,
    pub requested_device: String,
    /// Runtime device ("CPU", "GPU.0" for OpenVINO; "CoreML", ... for ONNX Runtime).
    pub device: String,
    pub device_name: String,
    /// Canonical spec of the device that ran ("openvino:cpu", "ort:coreml").
    #[serde(default)]
    pub spec: String,
    pub fell_back: bool,
    pub input: String,
    /// Dataset ids (comma separated) or the image file.
    pub image: String,
    /// "WxH" of a single image, else "N images".
    pub image_size: String,
    #[serde(default)]
    pub images: usize,
    /// read + reshape + compile + request creation wall time.
    pub compile_ms: f64,
    /// "disabled", "hit" (no new cache file written), "miss" (a blob was written) or "unknown".
    pub cache: String,
    pub warmup_iterations: usize,
    pub warmup_ms: f64,
    /// Timed runs per image.
    pub repeat: usize,
    pub stages_ms: StageStats,
    /// Sequential requests per second over the timed loop (1 concurrent client).
    pub throughput_fps: f64,
    /// Detections of the first image.
    pub detections: Vec<Detection>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub by_resolution: Vec<BucketSpeed>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<AccuracyReport>,
    /// Why there is no accuracy (no ground truth, no scored classes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accuracy_note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grades: Option<Grades>,
    /// Detections compared with the CPU reference device (device sweeps only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agreement: Option<Agreement>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_image: Vec<ImageRun>,
}

/// Detections of one run matched against another's (`--compare-cpu`, agreement).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Comparison {
    pub matched: usize,
    pub only_primary: usize,
    pub only_cpu: usize,
    pub max_confidence_diff: f32,
    pub mean_confidence_diff: f32,
}

impl Comparison {
    /// Sum of per-image comparisons (mean confidence difference weighted by matches).
    pub fn merge(&mut self, o: &Comparison) {
        let total = self.matched + o.matched;
        if total > 0 {
            self.mean_confidence_diff = (self.mean_confidence_diff * self.matched as f32
                + o.mean_confidence_diff * o.matched as f32)
                / total as f32;
        }
        self.matched = total;
        self.only_primary += o.only_primary;
        self.only_cpu += o.only_cpu;
        self.max_confidence_diff = self.max_confidence_diff.max(o.max_confidence_diff);
    }
}

/// Intersection over union of two pixel boxes.
pub fn iou(a: &Detection, b: &Detection) -> f32 {
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
pub fn compare(primary: &[Detection], cpu: &[Detection], min_iou: f32) -> Comparison {
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

/// One model to benchmark with its resolved settings.
#[derive(Debug, Clone)]
pub struct Job {
    pub name: String,
    pub path: PathBuf,
    pub family: ModelFamilyKind,
    pub classes: Vec<String>,
    /// Device spec for single-device runs ([`Bench::run`]); sweeps ignore it.
    pub device: String,
    pub gpu_precision: Option<String>,
    /// Per-model confidence threshold (else the bench's).
    pub confidence_threshold: Option<f32>,
    /// Per-model label filter (else the bench's).
    pub object_filter: Option<Vec<String>>,
}

/// Models to benchmark from a config: every enabled entry, in order.
pub fn config_models(config: &Config) -> Result<Vec<&ModelConfig>> {
    if config.models.is_empty() {
        bail!("no --model given and no models in the config file");
    }
    let models: Vec<&ModelConfig> = config.enabled_models().collect();
    if models.is_empty() {
        bail!("no --model given and every model in the config file is disabled");
    }
    Ok(models)
}

/// The config entry named `name` (matched like `/v1/vision/custom/{model}`).
pub fn find_model<'a>(config: &'a Config, name: &str) -> Option<&'a ModelConfig> {
    let key = crate::registry::normalize_name(name);
    config
        .models
        .iter()
        .find(|m| crate::registry::normalize_name(&m.effective_name()) == key)
}

/// A job for a model file given on the command line.
pub fn job_for_file(
    path: &Path,
    family: Option<ModelFamilyKind>,
    classes: Option<&Path>,
    device: Option<&str>,
) -> Result<Job> {
    Ok(Job {
        name: ModelConfig {
            path: path.to_path_buf(),
            ..Default::default()
        }
        .effective_name(),
        path: path.to_path_buf(),
        family: family.unwrap_or(ModelFamilyKind::Auto),
        classes: resolve_class_names(path, classes)?,
        device: device.unwrap_or("auto").to_string(),
        gpu_precision: None,
        confidence_threshold: None,
        object_filter: None,
    })
}

/// A job for a config entry: paths resolved like the registry does, device from the config
/// unless `device_override` is given.
pub fn job_for_config(
    config: &Config,
    m: &ModelConfig,
    device_override: Option<&str>,
) -> Result<Job> {
    let path = config.data_path(&m.path);
    if !path.is_file() {
        bail!("model file not found: {}", path.display());
    }
    let classes_path = m.classes.as_deref().map(|c| config.data_path(c));
    let classes = resolve_class_names(&path, classes_path.as_deref())?;
    let device = match device_override {
        Some(d) => d.to_string(),
        None => config.device_spec_for(m)?.to_string(),
    };
    Ok(Job {
        name: m.effective_name(),
        family: m.family,
        classes,
        device,
        gpu_precision: m.gpu_precision.clone(),
        confidence_threshold: m.confidence_threshold,
        object_filter: m.object_filter.clone(),
        path,
    })
}

/// The device a config model runs on (or would, after a restart), as a canonical spec:
/// `force_cpu` -> the best CPU option, `auto` -> the first option of the ranking, else the
/// configured device. None when the device does not parse or nothing is runnable for `auto`.
pub fn configured_device(config: &Config, m: &ModelConfig, sel: &Selection) -> Option<String> {
    if config.force_cpu {
        return Some(crate::backend::best_cpu(sel).to_string());
    }
    match config.device_spec_for(m).ok()? {
        spec::DeviceSpec::Auto => sel.auto.first().map(|d| d.to_string()),
        spec::DeviceSpec::Device(d) => Some(d.to_string()),
    }
}

/// Reference models for pseudo ground truth, most accurate first (matched as name prefixes,
/// case-insensitive).
pub const REFERENCE_PREFERENCE: [&str; 16] = [
    "rt-detrv2-x",
    "dfine-x",
    "rfdetr-large",
    "rt-detrv2-l",
    "dfine-l",
    "rfdetr-base",
    "rt-detrv2-m",
    "dfine-m",
    "yolo26x",
    "yolo26l",
    "rt-detrv2-ms",
    "rt-detrv2-s",
    "dfine-s",
    "yolo26m",
    "yolo26s",
    "yolo26n",
];

/// The pseudo-ground-truth reference: `explicit` when given (must be a configured model whose
/// file exists), else the most accurate present config model with COCO-style labels
/// ([`REFERENCE_PREFERENCE`]), else the largest present one with CCTV classes.
pub fn pick_reference<'a>(config: &'a Config, explicit: Option<&str>) -> Result<&'a ModelConfig> {
    let present = |m: &&ModelConfig| config.data_path(&m.path).is_file();
    if let Some(name) = explicit.map(str::trim).filter(|n| !n.is_empty()) {
        let m = find_model(config, name)
            .with_context(|| format!("reference model '{name}' is not a configured model"))?;
        if !present(&m) {
            bail!(
                "reference model '{name}': file {} not found",
                config.data_path(&m.path).display()
            );
        }
        return Ok(m);
    }
    let candidates: Vec<&ModelConfig> = config.models.iter().filter(present).collect();
    let rank = |m: &ModelConfig| {
        let n = m.effective_name().to_ascii_lowercase();
        REFERENCE_PREFERENCE
            .iter()
            .position(|p| n.starts_with(p))
            .unwrap_or(usize::MAX)
    };
    if let Some(m) = candidates
        .iter()
        .filter(|m| rank(m) != usize::MAX)
        .min_by_key(|m| rank(m))
    {
        return Ok(m);
    }
    // Unknown names: the largest file among models with many CCTV classes.
    candidates
        .into_iter()
        .filter(|m| {
            let path = config.data_path(&m.path);
            let classes = m.classes.as_deref().map(|c| config.data_path(c));
            resolve_class_names(&path, classes.as_deref())
                .map(|c| ClassMap::for_model(&c).classes.len() >= 6)
                .unwrap_or(false)
        })
        .max_by_key(|m| {
            std::fs::metadata(config.data_path(&m.path))
                .map(|md| md.len())
                .unwrap_or(0)
        })
        .context(
            "no reference model for pseudo ground truth: install a COCO model such as \
             rt-detrv2-x or dfine-s, or set benchmark.reference_model",
        )
}

/// What a benchmark is doing right now (for progress reporting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Compiling the model and creating the inference request.
    Loading,
    /// Untimed warm-up iterations.
    Warmup,
    /// Timed iterations over the images.
    Timing,
}

/// Progress of one device run, reported through [`Bench::progress`].
#[derive(Debug, Clone, Copy)]
pub struct Progress<'a> {
    pub model: &'a str,
    pub device: &'a str,
    pub phase: Phase,
    /// Iterations finished in this phase.
    pub done: usize,
    /// Iterations of this phase.
    pub total: usize,
    /// Image being processed (timing phase).
    pub image: &'a str,
}

/// Error returned when the cancel flag is set during a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("benchmark cancelled")]
pub struct Cancelled;

/// Whether `e` (or its cause chain) is a [`Cancelled`].
pub fn is_cancelled(e: &anyhow::Error) -> bool {
    e.chain().any(|c| c.downcast_ref::<Cancelled>().is_some())
}

/// Timing settings shared by every run of a benchmark.
pub struct Bench<'a> {
    /// Images to run (at least one image in total).
    pub sets: &'a [ImageSet],
    pub warmup: usize,
    /// Timed runs per image.
    pub repeat: usize,
    pub params: PostParams,
    pub object_filter: &'a [String],
    pub cache_dir: Option<&'a Path>,
    /// Checked before every iteration; a set flag ends the run with [`Cancelled`].
    pub cancel: Option<&'a AtomicBool>,
    /// Called at every phase change and after every iteration.
    pub progress: Option<&'a (dyn Fn(&Progress) + Sync)>,
    pub weights: Weights,
}

/// One request's timings (decode, preprocess, infer, postprocess; ms).
struct Sample {
    stages: [f64; 4],
    predictions: Vec<Prediction>,
    eval: Vec<Prediction>,
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

/// Lock the shared runtimes, ignoring poisoning (a panicked compile leaves them usable).
fn lock(rt: &Mutex<Runtimes>) -> std::sync::MutexGuard<'_, Runtimes> {
    rt.lock().unwrap_or_else(|e| e.into_inner())
}

fn to_pred(p: &Prediction) -> PredBox {
    PredBox {
        label: p.label.clone(),
        confidence: p.confidence,
        bbox: [
            p.x_min as f32,
            p.y_min as f32,
            p.x_max as f32,
            p.y_max as f32,
        ],
    }
}

impl<'a> Bench<'a> {
    /// Settings over `sets` with no cache dir, filter, cancel flag or progress callback.
    pub fn new(sets: &'a [ImageSet], warmup: usize, repeat: usize, params: PostParams) -> Self {
        Self {
            sets,
            warmup,
            repeat,
            params,
            object_filter: &[],
            cache_dir: None,
            cancel: None,
            progress: None,
            weights: Weights::default(),
        }
    }

    pub fn image_count(&self) -> usize {
        self.sets.iter().map(|s| s.images.len()).sum()
    }

    /// Dataset ids, comma separated.
    pub fn describe_sets(&self) -> String {
        self.sets
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(Cancelled.into());
        }
        Ok(())
    }

    fn report(
        &self,
        job: &Job,
        device: &str,
        phase: Phase,
        done: usize,
        total: usize,
        image: &str,
    ) {
        if let Some(cb) = self.progress {
            cb(&Progress {
                model: &job.name,
                device,
                phase,
                done,
                total,
                image,
            });
        }
    }

    /// Compile `job` on the device spec `device` and time it. The spec's candidates mirror the
    /// server: a GPU request may fall back to CPU (see [`RunResult::fell_back`]).
    pub fn run(&self, runtimes: &Mutex<Runtimes>, job: &Job, device: &str) -> Result<RunResult> {
        let spec = spec::parse(device)?;
        let candidates = lock(runtimes).plan(&spec, &job.path);
        self.run_candidates(runtimes, job, device, &candidates)
    }

    /// Compile `job` on exactly `device` (no fallback) and time it.
    pub fn run_exact(
        &self,
        runtimes: &Mutex<Runtimes>,
        job: &Job,
        device: Device,
    ) -> Result<RunResult> {
        let cand = Candidate {
            device,
            fell_back: false,
            note: None,
        };
        self.run_candidates(runtimes, job, &device.to_string(), &[cand])
    }

    fn run_candidates(
        &self,
        runtimes: &Mutex<Runtimes>,
        job: &Job,
        requested: &str,
        candidates: &[Candidate],
    ) -> Result<RunResult> {
        if self.image_count() == 0 {
            bail!("no benchmark images");
        }
        self.check_cancel()?;
        self.report(job, requested, Phase::Loading, 0, 1, "");
        let req = LoadRequest {
            path: job.path.clone(),
            requested: requested.to_string(),
            gpu_precision: job.gpu_precision.clone(),
        };
        let before = self.cache_dir.map(cache_files);
        let t = Instant::now();
        let label = format!("model {}", job.name);
        // Compile under the runtimes lock (workers compile through the same lock), create the
        // inference request outside it.
        let backend = crate::backend::try_candidates(&label, candidates, |cand| {
            let compiled = lock(runtimes).compile(cand, &req)?;
            compiled.into_backend()
        })
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
        self.time(job, requested, backend, compile_ms, cache)
    }

    /// Warm up and time `backend` over every image, then score its detections.
    fn time(
        &self,
        job: &Job,
        requested: &str,
        mut backend: Backend,
        compile_ms: f64,
        cache: String,
    ) -> Result<RunResult> {
        let info = backend.info();
        let dev = info.device.clone();
        let (in_w, in_h) = info.input_size;
        let family =
            crate::model::make_family(job.family, &info.inputs, &info.outputs, job.classes.len())?;
        let mut pre = Preprocessor::for_family(in_w, in_h, family.as_ref());
        let threshold = job
            .confidence_threshold
            .unwrap_or(self.params.confidence_threshold);
        let params = PostParams {
            confidence_threshold: threshold,
            nms_iou: self.params.nms_iou,
        };
        let eval_params = PostParams {
            confidence_threshold: EVAL_CONFIDENCE.min(threshold),
            nms_iou: self.params.nms_iou,
        };
        let filter: Option<&[String]> = match &job.object_filter {
            Some(f) if !f.is_empty() => Some(f),
            Some(_) => None,
            None => (!self.object_filter.is_empty()).then_some(self.object_filter),
        };

        // One request exactly as `WorkerCtx::process` runs it, with per-stage timings; with
        // `eval`, the same outputs are post-processed again at the evaluation threshold
        // (untimed, unfiltered).
        let mut iteration = |bytes: &[u8], eval: bool| -> Result<Sample> {
            let t = Instant::now();
            let img = crate::image::decode(bytes)?;
            let decode = ms(t.elapsed());

            let t = Instant::now();
            let (chw, ctx) = pre.run(&img.rgb, img.width, img.height)?;
            let extra = family.extra_inputs(&ctx);
            let preprocess = ms(t.elapsed());

            let t = Instant::now();
            let outputs = backend.infer(chw, &extra)?;
            let infer = ms(t.elapsed());

            let t = Instant::now();
            let dets = family.postprocess(&outputs, &ctx, &params)?;
            let preds = crate::model::to_predictions(&dets, &ctx, &job.classes, filter);
            let postprocess = ms(t.elapsed());
            let eval = if eval {
                let dets = family.postprocess(&outputs, &ctx, &eval_params)?;
                crate::model::to_predictions(&dets, &ctx, &job.classes, None)
            } else {
                Vec::new()
            };
            Ok(Sample {
                stages: [decode, preprocess, infer, postprocess],
                predictions: preds,
                eval,
            })
        };

        let all: Vec<(&ImageSet, &images::SetImage)> = self
            .sets
            .iter()
            .flat_map(|s| s.images.iter().map(move |i| (s, i)))
            .collect();
        let t = Instant::now();
        self.report(job, requested, Phase::Warmup, 0, self.warmup, "");
        if self.warmup > 0 {
            let first = all[0].1.bytes()?;
            for i in 0..self.warmup {
                self.check_cancel()?;
                iteration(&first, false).context("warm-up inference")?;
                self.report(job, requested, Phase::Warmup, i + 1, self.warmup, "");
            }
        }
        let warmup_ms = ms(t.elapsed());

        let repeat = self.repeat.max(1);
        let total_iters = all.len() * repeat;
        let mut samples: [Vec<f64>; 5] = Default::default();
        let mut by_res: BTreeMap<&'static str, ([Vec<f64>; 4], usize)> = BTreeMap::new();
        let mut per_image: Vec<ImageRun> = Vec::with_capacity(all.len());
        let mut eval_preds: Vec<Vec<PredBox>> = Vec::with_capacity(all.len());
        let mut done = 0;
        let mut loop_time = Duration::ZERO;
        self.report(job, requested, Phase::Timing, 0, total_iters, "");
        for (set, img) in &all {
            let bytes = img.bytes()?;
            let mut totals = Vec::with_capacity(repeat);
            let mut infers = Vec::with_capacity(repeat);
            let mut last: Option<Sample> = None;
            let bucket = img.resolution();
            for r in 0..repeat {
                self.check_cancel()?;
                let t = Instant::now();
                let sample = iteration(&bytes, r + 1 == repeat)
                    .with_context(|| format!("image {}", img.file))?;
                loop_time += t.elapsed();
                let total: f64 = sample.stages.iter().sum();
                for (k, v) in sample.stages.iter().enumerate() {
                    samples[k].push(*v);
                }
                samples[4].push(total);
                let e = by_res.entry(bucket).or_default();
                e.0[0].push(sample.stages[0] + sample.stages[1]);
                e.0[1].push(sample.stages[2]);
                e.0[2].push(sample.stages[3]);
                e.0[3].push(total);
                totals.push(total);
                infers.push(sample.stages[2]);
                last = Some(sample);
                done += 1;
                self.report(job, requested, Phase::Timing, done, total_iters, &img.file);
            }
            by_res.entry(bucket).or_default().1 += 1;
            let last = last.expect("repeat >= 1");
            let mut preds: Vec<ImagePred> = last
                .predictions
                .iter()
                .map(|p| {
                    let b = to_pred(p);
                    ImagePred {
                        label: b.label,
                        confidence: b.confidence,
                        bbox: b.bbox,
                        tp: None,
                    }
                })
                .collect();
            preds.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
            preds.truncate(MAX_STORED_PREDS);
            eval_preds.push(last.eval.iter().map(to_pred).collect());
            per_image.push(ImageRun {
                set: set.id.clone(),
                file: img.file.clone(),
                total_ms: Stats::from_samples(&totals).p50,
                infer_ms: Stats::from_samples(&infers).p50,
                preds,
                gt_matched: Vec::new(),
                counts: None,
            });
        }
        let loop_s = loop_time.as_secs_f64();

        let total = Stats::from_samples(&samples[4]);
        let (accuracy, accuracy_note) = score(job, &all, &mut per_image, &eval_preds, threshold);
        let accuracy_score = accuracy.as_ref().and_then(|a| {
            let small = a
                .overall
                .by_size
                .iter()
                .find(|b| b.bucket == "small" && b.gt > 0)
                .and_then(|b| b.recall);
            a.overall.ap50.map(|ap| grade::accuracy_score(ap, small))
        });
        let relative = accuracy.as_ref().is_some_and(|a| a.relative);
        let grades = Grades::new(accuracy_score, total.p50, self.weights, relative);
        let first_dets = per_image
            .first()
            .map(|r| r.preds.iter().map(Detection::from).collect())
            .unwrap_or_default();
        Ok(RunResult {
            model: job.name.clone(),
            path: job.path.display().to_string(),
            family: family.kind().to_string(),
            requested_device: requested.to_string(),
            device: dev.actual.clone(),
            device_name: dev.full_name.clone(),
            spec: dev.spec.clone(),
            fell_back: dev.fell_back,
            input: format!("{in_w}x{in_h}"),
            image: self.describe_sets(),
            image_size: if all.len() == 1 {
                format!("{}x{}", all[0].1.width, all[0].1.height)
            } else {
                format!("{} images", all.len())
            },
            images: all.len(),
            compile_ms,
            cache,
            warmup_iterations: self.warmup,
            warmup_ms,
            repeat,
            stages_ms: StageStats {
                decode: Stats::from_samples(&samples[0]),
                preprocess: Stats::from_samples(&samples[1]),
                infer: Stats::from_samples(&samples[2]),
                postprocess: Stats::from_samples(&samples[3]),
                total,
            },
            throughput_fps: if loop_s > 0.0 {
                total_iters as f64 / loop_s
            } else {
                0.0
            },
            detections: first_dets,
            by_resolution: images::RESOLUTION_BUCKETS
                .iter()
                .filter_map(|b| {
                    let (s, n) = by_res.get(b)?;
                    let t = Stats::from_samples(&s[3]);
                    Some(BucketSpeed {
                        bucket: b.to_string(),
                        images: *n,
                        pre_p50: Stats::from_samples(&s[0]).p50,
                        infer_p50: Stats::from_samples(&s[1]).p50,
                        post_p50: Stats::from_samples(&s[2]).p50,
                        total_p50: t.p50,
                        total_p95: t.p95,
                    })
                })
                .collect(),
            accuracy,
            accuracy_note,
            grades: Some(grades),
            agreement: None,
            per_image,
        })
    }
}

/// Score the run: overall accuracy (over the images with real ground truth, else those with
/// pseudo ground truth), per dataset and per tag; marks each stored prediction TP/FP and each
/// ground-truth object matched/missed in `per_image`.
fn score(
    job: &Job,
    all: &[(&ImageSet, &images::SetImage)],
    per_image: &mut [ImageRun],
    eval_preds: &[Vec<PredBox>],
    threshold: f32,
) -> (Option<AccuracyReport>, Option<String>) {
    let map = ClassMap::for_model(&job.classes);
    if map.is_empty() {
        return (
            None,
            Some("the model has none of the scored CCTV classes".to_string()),
        );
    }
    if !all.iter().any(|(_, i)| i.gt.is_some()) {
        return (
            None,
            Some("no ground truth (and no pseudo ground truth) for these images".to_string()),
        );
    }
    // Per image: mapped ground truth (with the original index) and the eval input.
    let mut evals: Vec<Option<EvalImage>> = Vec::with_capacity(all.len());
    for (k, (set, img)) in all.iter().enumerate() {
        let Some(gt) = &img.gt else {
            evals.push(None);
            continue;
        };
        let (mapped, eval, allowed) = eval_input(&map, set, gt, &eval_preds[k]);
        let gt_list = eval.gt.clone();
        // Drill-down flags on the stored (production) predictions.
        let run = &mut per_image[k];
        let scored: Vec<(usize, PredBox)> = run
            .preds
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                map.pred_class(&p.label)
                    .filter(|c| allowed.as_ref().is_none_or(|a| a.iter().any(|x| x == c)))
                    .map(|c| {
                        (
                            i,
                            PredBox {
                                label: c.to_string(),
                                confidence: p.confidence,
                                bbox: p.bbox,
                            },
                        )
                    })
            })
            .collect();
        let preds: Vec<PredBox> = scored.iter().map(|(_, p)| p.clone()).collect();
        let m = metrics::match_image(&preds, &gt_list, 0.5);
        for ((i, _), r) in scored.iter().zip(&m.pred) {
            run.preds[*i].tp = *r;
        }
        let mut j = 0;
        run.gt_matched = mapped
            .iter()
            .map(|g| match g {
                Some(g) => {
                    let hit = m.gt_matched[j];
                    j += 1;
                    (!g.ignore).then_some(hit)
                }
                None => None,
            })
            .collect();
        run.counts = Some(metrics::image_counts(&eval, threshold).0);
        evals.push(Some(eval));
    }
    let real: Vec<usize> = (0..all.len())
        .filter(|&k| evals[k].is_some() && all[k].0.ground_truth.is_real())
        .collect();
    let (chosen, relative) = if real.is_empty() {
        (
            (0..all.len())
                .filter(|&k| evals[k].is_some())
                .collect::<Vec<_>>(),
            true,
        )
    } else {
        (real, false)
    };
    let subset =
        |ks: &[usize]| -> Vec<EvalImage> { ks.iter().filter_map(|&k| evals[k].clone()).collect() };
    let overall = metrics::evaluate(&subset(&chosen), &map.classes, threshold);
    let all_map = ClassMap::all_classes(&job.classes);
    let all_classes = (all_map.classes.len() > map.classes.len()).then(|| {
        let imgs: Vec<EvalImage> = chosen
            .iter()
            .filter_map(|&k| {
                let gt = all[k].1.gt.as_ref()?;
                Some(eval_input(&all_map, all[k].0, gt, &eval_preds[k]).1)
            })
            .collect();
        let s = metrics::evaluate(&imgs, &all_map.classes, threshold);
        AllClasses {
            classes: s.per_class.iter().filter(|c| c.gt > 0).count(),
            ap50: s.ap50,
            ap50_95: s.ap50_95,
        }
    });
    let ground_truth = if relative {
        let refs: Vec<String> = all
            .iter()
            .filter_map(|(s, _)| match &s.ground_truth {
                GroundTruth::Pseudo(r) => Some(r.clone()),
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        format!("relative to {}, not ground truth", refs.join(", "))
    } else {
        "ground truth".to_string()
    };
    // Per dataset (each with its own kind of ground truth).
    let mut by_dataset = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for (s, _) in all {
        if seen.contains(&s.id.as_str()) {
            continue;
        }
        seen.push(&s.id);
        let ks: Vec<usize> = (0..all.len())
            .filter(|&k| all[k].0.id == s.id && evals[k].is_some())
            .collect();
        if ks.is_empty() {
            continue;
        }
        let sum = metrics::evaluate(&subset(&ks), &map.classes, threshold);
        by_dataset.push(Breakdown::of(&s.id, &sum, !s.ground_truth.is_real()));
    }
    // Per tag, over the images of the overall score.
    let mut tags: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for &k in &chosen {
        for t in &all[k].1.tags {
            tags.entry(t.as_str()).or_default().push(k);
        }
    }
    let by_tag = tags
        .into_iter()
        .map(|(t, ks)| {
            let sum = metrics::evaluate(&subset(&ks), &map.classes, threshold);
            Breakdown::of(t, &sum, relative)
        })
        .collect();
    (
        Some(AccuracyReport {
            ground_truth,
            relative,
            classes: map.classes.clone(),
            threshold,
            overall,
            by_dataset,
            by_tag,
            all_classes,
        }),
        None,
    )
}

/// Fill the ground truth of every set without it with the detections of `reference` on
/// `device`: confidence >= [`PSEUDO_GT_CONFIDENCE`] are objects, down to
/// [`PSEUDO_IGNORE_CONFIDENCE`] ignore regions. Returns the reference description
/// ("rt-detrv2-x on openvino:cpu"), or None when every set already had ground truth.
pub fn pseudo_ground_truth(
    bench: &Bench,
    runtimes: &Mutex<Runtimes>,
    reference: &Job,
    device: Device,
    sets: &mut [ImageSet],
) -> Result<Option<String>> {
    let todo: Vec<usize> = (0..sets.len()).filter(|&i| !sets[i].annotated()).collect();
    if todo.is_empty() {
        return Ok(None);
    }
    let subset: Vec<ImageSet> = todo.iter().map(|&i| sets[i].clone()).collect();
    let mut job = reference.clone();
    job.confidence_threshold = Some(PSEUDO_IGNORE_CONFIDENCE);
    job.object_filter = Some(Vec::new());
    let pb = Bench {
        sets: &subset,
        warmup: 0,
        repeat: 1,
        params: PostParams {
            confidence_threshold: PSEUDO_IGNORE_CONFIDENCE,
            nms_iou: bench.params.nms_iou,
        },
        object_filter: &[],
        cache_dir: bench.cache_dir,
        cancel: bench.cancel,
        progress: bench.progress,
        weights: bench.weights,
    };
    let run = pb
        .run_exact(runtimes, &job, device)
        .with_context(|| format!("pseudo ground truth with {} on {device}", job.name))?;
    let desc = format!("{} on {device}", job.name);
    let mut k = 0;
    for &i in &todo {
        for img in &mut sets[i].images {
            let r = &run.per_image[k];
            k += 1;
            img.gt = Some(
                r.preds
                    .iter()
                    .map(|p| GtBox {
                        label: metrics::canonical_label(&p.label),
                        bbox: p.bbox,
                        ignore: p.confidence < PSEUDO_GT_CONFIDENCE,
                    })
                    .collect(),
            );
        }
        sets[i].ground_truth = GroundTruth::Pseudo(desc.clone());
    }
    Ok(Some(desc))
}

/// CPU devices whose detections serve as the agreement reference, in order of preference.
pub const REFERENCE_DEVICES: [Device; 2] = [
    Device::OPENVINO_CPU,
    Device {
        runtime: Runtime::Ort,
        target: Target::Cpu,
        index: None,
    },
];

/// The first runnable CPU device of `sel` (pseudo ground truth runs there).
pub fn reference_device(sel: &Selection) -> Option<Device> {
    REFERENCE_DEVICES
        .into_iter()
        .find(|d| sel.option(d).is_some_and(|o| o.runnable))
}

/// Which devices a sweep runs.
#[derive(Debug, Clone, Default)]
pub struct SweepOptions {
    /// Exactly these devices (an option that cannot run for the model becomes a failed row);
    /// None = every runnable option of the model's selection.
    pub devices: Option<Vec<Device>>,
    /// The device the model is configured to run on (kept when within the noise margin of the
    /// best; marked in the UI).
    pub configured: Option<String>,
}

/// Run `job` on every chosen device, compare detections with the CPU reference device (OpenVINO
/// CPU, else ONNX Runtime CPU, whichever ran) and recommend a device. The reference CPU devices
/// run first so each later result can be judged as it arrives; the result lists devices in the
/// selection's order. `on_device` sees the partial result after each device (live progress).
///
/// Errors only with [`Cancelled`]; device failures are recorded in the result.
pub fn sweep(
    bench: &Bench,
    runtimes: &Mutex<Runtimes>,
    job: &Job,
    opts: &SweepOptions,
    on_device: &mut dyn FnMut(&ModelResult),
) -> Result<ModelResult> {
    let selection = lock(runtimes).selection(Some(&job.path));
    let mut result = ModelResult::new(job, bench, opts.configured.clone());

    // (selection order, device, label, why it cannot run).
    let mut planned: Vec<(usize, Device, String, Option<String>)> = Vec::new();
    match &opts.devices {
        None => {
            for (i, o) in selection.options.iter().enumerate() {
                if o.runnable {
                    planned.push((i, o.spec, o.label.clone(), None));
                } else {
                    result.skipped.push(format!(
                        "{}: {}",
                        o.spec,
                        o.reason.as_deref().unwrap_or("not runnable")
                    ));
                }
            }
        }
        Some(devices) => {
            for d in devices {
                match selection.options.iter().position(|o| o.spec == *d) {
                    Some(i) => {
                        let o = &selection.options[i];
                        let why = (!o.runnable)
                            .then(|| o.reason.clone().unwrap_or_else(|| "not runnable".into()));
                        planned.push((i, o.spec, o.label.clone(), why));
                    }
                    None => planned.push((
                        usize::MAX,
                        *d,
                        d.to_string(),
                        Some("not a device option on this machine".into()),
                    )),
                }
            }
        }
    }
    planned.sort_by_key(|p| p.0);
    // Run order: the reference candidates first.
    let mut order: Vec<usize> = (0..planned.len()).collect();
    order.sort_by_key(|&k| {
        REFERENCE_DEVICES
            .iter()
            .position(|r| *r == planned[k].1)
            .unwrap_or(REFERENCE_DEVICES.len())
    });

    let mut slots: Vec<Option<DeviceResult>> = vec![None; planned.len()];
    for k in order {
        let (_, device, label, why) = &planned[k];
        let entry = match why {
            Some(reason) => DeviceResult::failed(device, label, reason.clone()),
            None => match bench.run_exact(runtimes, job, *device) {
                Ok(run) => DeviceResult::ran(device, label, run),
                Err(e) if is_cancelled(&e) => return Err(e),
                Err(e) => {
                    tracing::warn!(model = %job.name, device = %device, "benchmark run failed: {e:#}");
                    DeviceResult::failed(device, label, format!("{e:#}"))
                }
            },
        };
        slots[k] = Some(entry);
        result.devices = slots.iter().flatten().cloned().collect();
        result.finish();
        on_device(&result);
    }
    result.devices = slots.into_iter().flatten().collect();
    result.finish();
    Ok(result)
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
    fn config_models_are_the_enabled_entries() {
        let m = |n: &str, enabled: bool| ModelConfig {
            path: format!("models/{n}.onnx").into(),
            enabled,
            ..Default::default()
        };
        let mut config = Config {
            models: vec![m("a", false), m("b", true), m("c", false), m("d", true)],
            ..Default::default()
        };
        let names: Vec<String> = config_models(&config)
            .unwrap()
            .iter()
            .map(|m| m.effective_name())
            .collect();
        assert_eq!(names, ["b", "d"]);
        assert_eq!(find_model(&config, "C.onnx").unwrap().effective_name(), "c");
        assert!(find_model(&config, "zzz").is_none());

        for model in &mut config.models {
            model.enabled = false;
        }
        let err = config_models(&config).unwrap_err();
        assert!(format!("{err:#}").contains("disabled"), "{err:#}");
        let err = config_models(&Config::default()).unwrap_err();
        assert!(format!("{err:#}").contains("no models"), "{err:#}");
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

        // Merging per-image comparisons weights the mean by matches.
        let mut a = Comparison {
            matched: 1,
            mean_confidence_diff: 0.1,
            max_confidence_diff: 0.1,
            ..Default::default()
        };
        a.merge(&Comparison {
            matched: 3,
            only_cpu: 2,
            mean_confidence_diff: 0.02,
            max_confidence_diff: 0.05,
            ..Default::default()
        });
        assert_eq!((a.matched, a.only_cpu), (4, 2));
        assert!((a.mean_confidence_diff - 0.04).abs() < 1e-6);
        assert!((a.max_confidence_diff - 0.1).abs() < 1e-6);
    }

    #[test]
    fn embedded_image_decodes() {
        let img = crate::image::decode(DEFAULT_IMAGE).unwrap();
        assert!(img.width > 0 && img.height > 0);
        assert_eq!(img.rgb.len(), (img.width * img.height * 3) as usize);
    }

    #[test]
    fn cancelled_is_detected_through_context() {
        let e = anyhow::Error::from(Cancelled).context("loading x");
        assert!(is_cancelled(&e));
        assert!(!is_cancelled(&anyhow::anyhow!("other")));
    }

    #[test]
    fn configured_device_resolves_auto_and_force_cpu() {
        use crate::backend::select::{RuntimeProbe, select};
        let hw = crate::backend::HardwareInfo::this_platform_without_gpus();
        let sel = select(&hw, &RuntimeProbe::openvino_only(&["CPU".into()]), true);
        let m = ModelConfig::default();
        let mut c = Config::default();
        assert_eq!(
            configured_device(&c, &m, &sel).as_deref(),
            Some("openvino:cpu")
        );
        let over = ModelConfig {
            device: Some("ort:coreml".into()),
            ..Default::default()
        };
        assert_eq!(
            configured_device(&c, &over, &sel).as_deref(),
            Some("ort:coreml")
        );
        c.force_cpu = true;
        assert_eq!(
            configured_device(&c, &over, &sel).as_deref(),
            Some("openvino:cpu")
        );
        assert_eq!(reference_device(&sel), Some(Device::OPENVINO_CPU));
    }

    #[test]
    fn reference_model_choice() {
        let dir = std::env::temp_dir().join(format!("bop-ref-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["IPcam-general.onnx", "dfine-s.onnx", "rt-detrv2-l.onnx"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        std::fs::write(
            dir.join("IPcam-general.yaml"),
            "NAMES:\n  - person\n  - vehicle\n  - unknown\n",
        )
        .unwrap();
        let m = |n: &str| ModelConfig {
            name: Some(n.into()),
            path: dir.join(format!("{n}.onnx")),
            ..Default::default()
        };
        let c = Config {
            models: vec![
                m("IPcam-general"),
                m("dfine-s"),
                m("rt-detrv2-x"),
                m("rt-detrv2-l"),
            ],
            ..Default::default()
        };
        // rt-detrv2-x is configured but its file is missing: the next best present one.
        assert_eq!(
            pick_reference(&c, None).unwrap().effective_name(),
            "rt-detrv2-l"
        );
        assert_eq!(
            pick_reference(&c, Some("DFINE-S"))
                .unwrap()
                .effective_name(),
            "dfine-s"
        );
        let e = pick_reference(&c, Some("rt-detrv2-x")).unwrap_err();
        assert!(format!("{e:#}").contains("not found"), "{e:#}");
        assert!(pick_reference(&c, Some("zzz")).is_err());
        // Only a non-COCO model: no reference.
        let only = Config {
            models: vec![m("IPcam-general")],
            ..Default::default()
        };
        assert!(pick_reference(&only, None).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A score over synthetic per-image predictions: ground truth, pseudo ground truth and the
    /// drill-down flags.
    #[test]
    fn scoring_real_and_pseudo_ground_truth() {
        let mut set = ImageSet::sample();
        set.id = "real".into();
        set.ground_truth = GroundTruth::Manifest;
        set.images[0].gt = Some(vec![
            GtBox {
                label: "dog".into(),
                bbox: [0.0, 0.0, 100.0, 100.0],
                ignore: false,
            },
            GtBox {
                label: "umbrella".into(),
                bbox: [0.0, 0.0, 10.0, 10.0],
                ignore: false,
            },
            GtBox {
                label: "car".into(),
                bbox: [300.0, 300.0, 400.0, 400.0],
                ignore: false,
            },
        ]);
        let mut pseudo = set.clone();
        pseudo.id = "pseudo".into();
        pseudo.ground_truth = GroundTruth::Pseudo("ref on openvino:cpu".into());
        let job = Job {
            name: "m".into(),
            path: "m.onnx".into(),
            family: ModelFamilyKind::Yolo5,
            classes: crate::model::classes::coco80(),
            device: "auto".into(),
            gpu_precision: None,
            confidence_threshold: None,
            object_filter: None,
        };
        let pred = |l: &str, c: f32, b: [f32; 4]| PredBox {
            label: l.into(),
            confidence: c,
            bbox: b,
        };
        let eval = vec![
            pred("dog", 0.9, [1.0, 1.0, 100.0, 100.0]),
            pred("person", 0.2, [500.0, 0.0, 600.0, 100.0]),
        ];
        let stored = |e: &[PredBox]| {
            e.iter()
                .filter(|p| p.confidence >= 0.5)
                .map(|p| ImagePred {
                    label: p.label.clone(),
                    confidence: p.confidence,
                    bbox: p.bbox,
                    tp: None,
                })
                .collect::<Vec<_>>()
        };
        let all = vec![(&set, &set.images[0]), (&pseudo, &pseudo.images[0])];
        let mut per_image: Vec<ImageRun> = all
            .iter()
            .map(|(s, i)| ImageRun {
                set: s.id.clone(),
                file: i.file.clone(),
                total_ms: 1.0,
                infer_ms: 1.0,
                preds: stored(&eval),
                gt_matched: vec![],
                counts: None,
            })
            .collect();
        let (acc, note) = score(&job, &all, &mut per_image, &[eval.clone(), eval], 0.5);
        assert!(note.is_none());
        let acc = acc.unwrap();
        // Real ground truth wins for the overall score.
        assert!(!acc.relative);
        assert_eq!(acc.ground_truth, "ground truth");
        assert_eq!(acc.overall.images, 1);
        assert_eq!(
            (
                acc.overall.counts.tp,
                acc.overall.counts.fp,
                acc.overall.counts.fn_
            ),
            (1, 0, 1)
        );
        // dog AP 1, car AP 0 -> 0.5.
        assert!((acc.overall.ap50.unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(acc.by_dataset.len(), 2);
        assert!(acc.by_dataset[1].relative);
        assert!(acc.by_tag.iter().any(|t| t.key.starts_with("res:")));
        // Drill-down: the dog is a TP, the umbrella is not scored, the car is missed.
        assert_eq!(per_image[0].preds[0].tp, Some(true));
        assert_eq!(per_image[0].gt_matched, [Some(true), None, Some(false)]);

        // Pseudo only: relative, labelled with the reference.
        let all = vec![(&pseudo, &pseudo.images[0])];
        let mut pi = vec![per_image[1].clone()];
        let ev = vec![pred("dog", 0.9, [1.0, 1.0, 100.0, 100.0])];
        let (acc, _) = score(&job, &all, &mut pi, &[ev], 0.5);
        let acc = acc.unwrap();
        assert!(acc.relative);
        assert_eq!(
            acc.ground_truth,
            "relative to ref on openvino:cpu, not ground truth"
        );

        // No scored classes / no ground truth.
        let pkg = Job {
            classes: vec!["package".into()],
            ..job.clone()
        };
        let (acc, note) = score(&pkg, &all, &mut pi, &[vec![]], 0.5);
        assert!(acc.is_none() && note.unwrap().contains("CCTV"));
        let bare = ImageSet::sample();
        let all = vec![(&bare, &bare.images[0])];
        let (acc, note) = score(&job, &all, &mut pi, &[vec![]], 0.5);
        assert!(acc.is_none() && note.unwrap().contains("no ground truth"));
    }

    #[test]
    fn scored_labels_limit_a_set() {
        // A vehicles-only set: the person prediction is neither TP nor FP, person ground truth
        // is not a miss.
        let mut set = ImageSet::sample();
        set.ground_truth = GroundTruth::Manifest;
        set.scored_labels = Some(vec!["car".into(), "motorbike".into()]);
        set.images[0].gt = Some(vec![
            GtBox {
                label: "car".into(),
                bbox: [0.0, 0.0, 100.0, 100.0],
                ignore: false,
            },
            GtBox {
                label: "person".into(),
                bbox: [200.0, 0.0, 300.0, 100.0],
                ignore: false,
            },
        ]);
        let job = Job {
            name: "m".into(),
            path: "m.onnx".into(),
            family: ModelFamilyKind::Yolo5,
            classes: crate::model::classes::coco80(),
            device: "auto".into(),
            gpu_precision: None,
            confidence_threshold: None,
            object_filter: None,
        };
        let p = |l: &str, b: [f32; 4]| PredBox {
            label: l.into(),
            confidence: 0.9,
            bbox: b,
        };
        let eval = vec![
            p("car", [0.0, 0.0, 100.0, 100.0]),
            p("person", [500.0, 0.0, 600.0, 100.0]),
        ];
        let all = vec![(&set, &set.images[0])];
        let mut pi = vec![ImageRun {
            set: set.id.clone(),
            file: "f".into(),
            total_ms: 1.0,
            infer_ms: 1.0,
            preds: eval
                .iter()
                .map(|e| ImagePred {
                    label: e.label.clone(),
                    confidence: e.confidence,
                    bbox: e.bbox,
                    tp: None,
                })
                .collect(),
            gt_matched: vec![],
            counts: None,
        }];
        let (acc, _) = score(&job, &all, &mut pi, &[eval], 0.5);
        let acc = acc.unwrap();
        assert_eq!(
            (
                acc.overall.counts.tp,
                acc.overall.counts.fp,
                acc.overall.counts.fn_
            ),
            (1, 0, 0)
        );
        assert_eq!(pi[0].preds[1].tp, None);
        assert_eq!(pi[0].gt_matched, [Some(true), None]);
        // All-class AP is reported for a COCO model (more classes than the CCTV ones).
        let all_c = acc.all_classes.unwrap();
        assert_eq!(all_c.classes, 1);
        assert!((all_c.ap50.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sweep_without_runtimes_records_failures() {
        // No OpenVINO and no ONNX Runtime: every explicitly requested device fails with its
        // reason, nothing is recommended, and the sweep itself succeeds.
        let dir = std::env::temp_dir().join(format!("bop-sweep-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.onnx");
        std::fs::write(&model, b"not a model").unwrap();
        let rt = Runtimes::new_with(
            &crate::backend::CoreOptions {
                openvino_dir: Some(dir.join("no-openvino")),
                ..Default::default()
            },
            &crate::backend::OrtOptions {
                onnxruntime_dir: Some(dir.join("no-ort")),
                default_dir: Some(dir.join("no-ort")),
                cuda_libs_dir: None,
            },
        );
        if rt.openvino().is_some() || rt.has_ort() {
            // A system-wide runtime was found; this test is about the failure path only.
            return;
        }
        let rt = Mutex::new(rt);
        let job = Job {
            name: "m".into(),
            path: model,
            family: ModelFamilyKind::Yolo5,
            classes: vec!["a".into()],
            device: "auto".into(),
            gpu_precision: None,
            confidence_threshold: None,
            object_filter: None,
        };
        let params = PostParams {
            confidence_threshold: 0.5,
            nms_iou: 0.5,
        };
        let sets = [ImageSet::sample()];
        let bench = Bench::new(&sets, 0, 1, params);
        let opts = SweepOptions {
            devices: Some(REFERENCE_DEVICES.to_vec()),
            configured: None,
        };
        let mut seen = 0;
        let r = sweep(&bench, &rt, &job, &opts, &mut |_| seen += 1).unwrap();
        assert_eq!(seen, 2);
        assert_eq!(r.devices.len(), 2);
        assert!(
            r.devices
                .iter()
                .all(|d| d.error.is_some() && d.run.is_none())
        );
        assert_eq!(r.recommended, None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
