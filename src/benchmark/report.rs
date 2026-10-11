//! Benchmark results: per-device agreement with the CPU reference, the recommended device per
//! model, and the persisted `benchmark.json` (the latest run of every model, next to the config
//! file) that the CLI and the web UI both read and write.

use super::grade::{AccuracyMetric, Grade, Grades, Weights};
use super::images::{GroundTruth, ImageSet, SetKind};
use super::metrics::GtBox;
use super::threshold::{Objective, ThresholdAdvice};
use super::{Bench, Comparison, Detection, Job, MATCH_IOU, REFERENCE_DEVICES, RunResult, compare};
use crate::backend::spec::Device;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// File name of the persisted results, in the config file's directory.
pub const RESULTS_FILE: &str = "benchmark.json";
/// Devices within this fraction of the fastest total p50 count as equally fast (timing noise);
/// among them the configured device is kept.
pub const NOISE_MARGIN: f64 = 0.05;

/// `benchmark.json` beside `config_path`.
pub fn results_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .map(|d| d.join(RESULTS_FILE))
        .unwrap_or_else(|| PathBuf::from(RESULTS_FILE))
}

/// How a device's detections compare with the CPU reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// This device is the reference.
    Reference,
    /// Same objects, confidences within a few hundredths.
    Agrees,
    /// Mostly the same objects (e.g. one borderline detection more or less).
    Minor,
    /// Different objects or confidences: the device computes something else (precision
    /// problem, broken kernel). Never recommended.
    Disagrees,
}

impl Verdict {
    /// Can be recommended.
    pub fn acceptable(self) -> bool {
        self != Verdict::Disagrees
    }
}

/// Agreement of one device's detections with the reference device's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Agreement {
    /// Reference device spec ("openvino:cpu" or "ort:cpu").
    pub reference: String,
    pub matched: usize,
    /// Detections only this device found.
    pub only_device: usize,
    /// Reference detections this device missed.
    pub only_reference: usize,
    pub max_confidence_diff: f32,
    pub mean_confidence_diff: f32,
    /// `matched / max(device, reference)` detections (1 when both found nothing).
    pub score: f32,
    pub verdict: Verdict,
}

/// Score at or above which (with small confidence differences) a device agrees.
const AGREE_SCORE: f32 = 0.8;
const AGREE_CONF: f32 = 0.05;
/// Below this score (or above this mean confidence difference) a device disagrees.
const MINOR_SCORE: f32 = 0.5;
const MINOR_CONF: f32 = 0.15;

/// Compare `device` detections with `reference` detections (IoU >= [`MATCH_IOU`], same label).
pub fn agreement(device: &[Detection], reference: &[Detection], reference_spec: &str) -> Agreement {
    let c: Comparison = compare(device, reference, MATCH_IOU);
    verdict_of(&c, device.len().max(reference.len()), reference_spec)
}

/// Agreement of `device`'s detections with `reference`'s over every image both ran.
pub fn run_agreement(device: &RunResult, reference: &RunResult, reference_spec: &str) -> Agreement {
    let dets = |r: &RunResult, k: usize| -> Vec<Detection> {
        r.per_image[k].preds.iter().map(Detection::from).collect()
    };
    if device.per_image.is_empty() || device.per_image.len() != reference.per_image.len() {
        return agreement(&device.detections, &reference.detections, reference_spec);
    }
    let mut total = Comparison::default();
    let (mut n_dev, mut n_ref) = (0, 0);
    for k in 0..device.per_image.len() {
        let (a, b) = (dets(device, k), dets(reference, k));
        n_dev += a.len();
        n_ref += b.len();
        total.merge(&compare(&a, &b, MATCH_IOU));
    }
    verdict_of(&total, n_dev.max(n_ref), reference_spec)
}

fn verdict_of(c: &Comparison, n: usize, reference_spec: &str) -> Agreement {
    let score = if n == 0 {
        1.0
    } else {
        c.matched as f32 / n as f32
    };
    let verdict = if score >= AGREE_SCORE && c.mean_confidence_diff <= AGREE_CONF {
        Verdict::Agrees
    } else if score >= MINOR_SCORE && c.mean_confidence_diff <= MINOR_CONF {
        Verdict::Minor
    } else {
        Verdict::Disagrees
    };
    Agreement {
        reference: reference_spec.to_string(),
        matched: c.matched,
        only_device: c.only_primary,
        only_reference: c.only_cpu,
        max_confidence_diff: c.max_confidence_diff,
        mean_confidence_diff: c.mean_confidence_diff,
        score,
        verdict,
    }
}

/// One device of a model's sweep: the run, or why it did not run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceResult {
    /// Device spec ("openvino:cpu", "ort:coreml").
    pub device: String,
    /// Option label ("ONNX Runtime CoreML (Apple M1)").
    pub label: String,
    /// Why it could not run (not runnable, compile or inference error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<RunResult>,
}

impl DeviceResult {
    pub fn ran(device: &Device, label: &str, run: RunResult) -> Self {
        Self {
            device: device.to_string(),
            label: label.to_string(),
            error: None,
            run: Some(run),
        }
    }

    pub fn failed(device: &Device, label: &str, error: String) -> Self {
        Self {
            device: device.to_string(),
            label: label.to_string(),
            error: Some(error),
            run: None,
        }
    }

    /// Ran on the requested device itself (a fallback run does not count).
    pub fn ok(&self) -> bool {
        self.run.as_ref().is_some_and(|r| !r.fell_back)
    }

    /// Full per-request p50 (ms) of a successful run.
    pub fn total_p50(&self) -> Option<f64> {
        self.run
            .as_ref()
            .filter(|_| self.ok())
            .map(|r| r.stages_ms.total.p50)
    }

    pub fn agreement(&self) -> Option<&Agreement> {
        self.run.as_ref().and_then(|r| r.agreement.as_ref())
    }

    pub fn grades(&self) -> Option<&Grades> {
        self.run
            .as_ref()
            .filter(|_| self.ok())
            .and_then(|r| r.grades.as_ref())
    }

    /// Ran and its detections are acceptable (or could not be checked: no reference).
    pub fn eligible(&self) -> bool {
        self.ok() && self.agreement().is_none_or(|a| a.verdict.acceptable())
    }
}

/// AP50 accuracy scores (0..1) this close to a model's best count as equally accurate when
/// picking its device. Each metric has its own tie on its own scale
/// ([`AccuracyMetric::tie`]); this is the AP50 one.
pub const ACCURACY_TIE: f64 = 0.015;

/// The recommended device of a model and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recommendation {
    pub device: Option<String>,
    pub reason: String,
}

/// The best device among those that ran and whose detections agree with the CPU reference:
/// the fastest full-request p50 among those as accurate as the best (within the grading
/// metric's tie, [`AccuracyMetric::tie`]: [`ACCURACY_TIE`] for AP50, 0.01 for ROC AUC). When the
/// `configured` device is eligible, as accurate ([`AccuracyMetric::keep_tie`]) and within
/// `margin` (fraction, e.g. [`NOISE_MARGIN`]) of the best p50, it is kept: the difference is
/// noise.
pub fn recommend(
    devices: &[DeviceResult],
    configured: Option<&str>,
    margin: f64,
) -> Recommendation {
    let eligible: Vec<(&DeviceResult, Grades)> = devices
        .iter()
        .filter(|d| d.eligible())
        .filter_map(|d| {
            let g = d.grades().copied().or_else(|| {
                d.total_p50()
                    .map(|p| Grades::new(None, p, Default::default(), false))
            })?;
            Some((d, g))
        })
        .collect();
    // Devices of one model whose detections agree differ in accuracy only by numeric noise
    // (OpenVINO's f16 CPU inference on ARM moves AP by ~1 point), so accuracy within
    // ACCURACY_TIE of the best counts as equal and the fastest of those wins. Letter grades are
    // too coarse here: 14 ms and 35 ms are both speed A.
    // Every run of one sweep is graded on one metric; the tie is on its scale.
    let metric = eligible
        .iter()
        .find(|(_, g)| g.accuracy_score.is_some())
        .map_or(AccuracyMetric::Ap50, |(_, g)| g.metric);
    let top_acc = eligible
        .iter()
        .filter_map(|(_, g)| g.accuracy_score)
        .max_by(f64::total_cmp);
    let as_accurate = |g: &Grades| match (top_acc, g.accuracy_score) {
        (Some(top), Some(a)) => a >= top - metric.tie(),
        (Some(_), None) => false,
        (None, _) => true,
    };
    let Some((best, bg)) = eligible
        .iter()
        .filter(|(_, g)| as_accurate(g))
        .min_by(|a, b| {
            a.1.p50_ms
                .total_cmp(&b.1.p50_ms)
                .then_with(|| Grades::rank_cmp(&a.1, &b.1))
        })
        .map(|(d, g)| (*d, *g))
    else {
        let reason = if devices.iter().any(DeviceResult::ok) {
            "no device's detections agree with the CPU reference".to_string()
        } else {
            "no device could run this model".to_string()
        };
        return Recommendation {
            device: None,
            reason,
        };
    };
    let excluded: Vec<&str> = devices
        .iter()
        .filter(|d| d.ok() && !d.eligible())
        .map(|d| d.device.as_str())
        .collect();
    let excluded = if excluded.is_empty() {
        String::new()
    } else {
        format!(
            "; excluded (detections disagree with the reference): {}",
            excluded.join(", ")
        )
    };
    let grade_text = |g: &Grades| match g.accuracy {
        Some(a) => format!(
            "overall {} (accuracy {a} by {}, speed {})",
            g.overall,
            g.metric.label(),
            g.speed
        ),
        None => format!("speed {}", g.speed),
    };
    if let Some(cur) = configured
        && cur != best.device
        && let Some((d, g)) = eligible.iter().find(|(d, _)| d.device == cur)
        && g.p50_ms <= bg.p50_ms * (1.0 + margin)
        && match (g.accuracy_score, bg.accuracy_score) {
            (Some(a), Some(b)) => a >= b - metric.keep_tie(),
            (None, None) => true,
            _ => false,
        }
    {
        return Recommendation {
            device: Some(d.device.clone()),
            reason: format!(
                "the configured device {} ({:.1} ms) is within {:.0}% of the best, {} ({:.1} ms){excluded}",
                d.device,
                g.p50_ms,
                margin * 100.0,
                best.device,
                bg.p50_ms
            ),
        };
    }
    // Compared with the fastest other eligible device.
    let runner_up = eligible
        .iter()
        .filter(|(d, _)| d.device != best.device)
        .min_by(|a, b| a.1.p50_ms.total_cmp(&b.1.p50_ms));
    let vs = match runner_up {
        Some((d, g)) if g.p50_ms >= bg.p50_ms => format!(
            ", {:.2}x faster than {} ({:.1} ms)",
            g.p50_ms / bg.p50_ms.max(1e-9),
            d.device,
            g.p50_ms
        ),
        Some((d, g)) => format!(
            "; {} is {:.2}x faster ({:.1} ms) but less accurate",
            d.device,
            bg.p50_ms / g.p50_ms.max(1e-9),
            g.p50_ms
        ),
        None => String::new(),
    };
    Recommendation {
        device: Some(best.device.clone()),
        reason: format!(
            "{}; full request p50 {:.1} ms{vs}{excluded}",
            grade_text(&bg),
            bg.p50_ms
        ),
    }
}

/// One model's sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResult {
    /// Model name (config `name` / file stem).
    pub model: String,
    pub path: String,
    /// Configured family ("auto" when detected at load time; each run reports the actual one).
    pub family: String,
    /// RFC 3339 time the sweep started.
    pub timestamp: String,
    /// Dataset ids, comma separated.
    pub image: String,
    /// Dataset ids.
    #[serde(default)]
    pub datasets: Vec<String>,
    #[serde(default)]
    pub images: usize,
    pub warmup: usize,
    pub repeat: usize,
    /// Device the model was configured to run on at benchmark time.
    #[serde(default)]
    pub configured: Option<String>,
    /// Device whose detections the others were compared with.
    #[serde(default)]
    pub reference: Option<String>,
    pub devices: Vec<DeviceResult>,
    /// Options not tried because they cannot run here (sweeps over every runnable option).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped: Vec<String>,
    pub recommended: Option<String>,
    #[serde(default)]
    pub recommendation: String,
    /// Why the model could not be benchmarked at all (e.g. the file is missing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// What the best confidence threshold optimizes (config `benchmark.threshold_objective`).
    #[serde(default)]
    pub threshold_objective: Objective,
    /// Best confidence threshold (from the recommended device's curves).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<ThresholdAdvice>,
}

impl ModelResult {
    /// An empty result for `job` (devices are added by the sweep).
    pub fn new(job: &Job, bench: &Bench, configured: Option<String>) -> Self {
        Self {
            model: job.name.clone(),
            path: job.path.display().to_string(),
            family: job.family.to_string(),
            timestamp: now(),
            image: bench.describe_sets(),
            datasets: bench.sets.iter().map(|s| s.id.clone()).collect(),
            images: bench.image_count(),
            warmup: bench.warmup,
            repeat: bench.repeat,
            configured,
            reference: None,
            devices: Vec::new(),
            skipped: Vec::new(),
            recommended: None,
            recommendation: String::new(),
            error: None,
            threshold_objective: Objective::default(),
            threshold: None,
        }
    }

    /// A model that could not be benchmarked.
    pub fn failed(model: &str, path: &str, error: String) -> Self {
        Self {
            model: model.to_string(),
            path: path.to_string(),
            family: String::new(),
            timestamp: now(),
            image: String::new(),
            datasets: Vec::new(),
            images: 0,
            warmup: 0,
            repeat: 0,
            configured: None,
            reference: None,
            devices: Vec::new(),
            skipped: Vec::new(),
            recommended: None,
            recommendation: String::new(),
            error: Some(error),
            threshold_objective: Objective::default(),
            threshold: None,
        }
    }

    /// Recompute the reference, each device's agreement and the recommendation.
    pub fn finish(&mut self) {
        let reference = REFERENCE_DEVICES.iter().find_map(|r| {
            let spec = r.to_string();
            self.devices
                .iter()
                .find(|d| d.device == spec && d.ok())
                .and_then(|d| d.run.clone())
                .map(|run| (spec, run))
        });
        self.reference = reference.as_ref().map(|(s, _)| s.clone());
        for d in &mut self.devices {
            let ok = d.ok();
            let Some(run) = d.run.as_mut() else { continue };
            run.agreement = match &reference {
                Some((spec, _)) if ok && *spec == d.device => Some(Agreement {
                    reference: spec.clone(),
                    matched: run.detections.len(),
                    only_device: 0,
                    only_reference: 0,
                    max_confidence_diff: 0.0,
                    mean_confidence_diff: 0.0,
                    score: 1.0,
                    verdict: Verdict::Reference,
                }),
                Some((spec, reference)) if ok => Some(run_agreement(run, reference, spec)),
                _ => None,
            };
        }
        let rec = recommend(&self.devices, self.configured.as_deref(), NOISE_MARGIN);
        self.recommended = rec.device;
        self.recommendation = match &self.reference {
            Some(_) => rec.reason,
            None if self.devices.iter().any(DeviceResult::ok) => format!(
                "{} (no CPU reference ran, detections not checked)",
                rec.reason
            ),
            None => rec.reason,
        };
        self.threshold = super::threshold::advise(
            &self.devices,
            self.recommended.as_deref(),
            self.threshold_objective,
        );
        if self.recommended.is_some()
            && let Some(t) = self.threshold.as_ref().and_then(ThresholdAdvice::summary)
        {
            self.recommendation.push_str("; ");
            self.recommendation.push_str(&t);
        }
    }

    /// Grades of the recommended device.
    pub fn recommended_grades(&self) -> Option<&Grades> {
        let rec = self.recommended.as_deref()?;
        self.devices.iter().find(|d| d.device == rec)?.grades()
    }

    /// A copy without per-image details and per-device threshold curves (for polling; the
    /// threshold advice keeps the chosen device's curve).
    pub fn summary(&self) -> Self {
        let mut m = self.clone();
        for d in &mut m.devices {
            if let Some(r) = d.run.as_mut() {
                r.per_image.clear();
                r.eval_preds.clear();
                if let Some(a) = r.accuracy.as_mut() {
                    a.sweep = None;
                }
            }
        }
        m
    }

    /// The device row with the lowest total p50 among successful runs.
    pub fn fastest(&self) -> Option<&DeviceResult> {
        self.devices
            .iter()
            .filter_map(|d| d.total_p50().map(|p| (d, p)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(d, _)| d)
    }
}

/// Host facts recorded with the results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct HardwareSummary {
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub gpus: Vec<String>,
}

impl HardwareSummary {
    pub fn current() -> Self {
        let hw = crate::backend::detect::hardware();
        Self {
            os: hw.os.clone(),
            arch: hw.arch.clone(),
            cpu: crate::system_info::cpu_name(),
            gpus: hw.gpus.iter().map(|g| g.to_string()).collect(),
        }
    }
}

/// Runtime versions recorded with the results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RuntimeVersions {
    pub openvino: Option<String>,
    pub onnxruntime: Option<String>,
    pub onnxruntime_flavor: Option<String>,
}

impl RuntimeVersions {
    pub fn of(rt: &crate::backend::Runtimes) -> Self {
        let info = rt.info();
        Self {
            openvino: (!info.openvino_version.is_empty()).then_some(info.openvino_version),
            onnxruntime: rt.onnxruntime_version(),
            onnxruntime_flavor: rt
                .has_ort()
                .then(|| rt.probe().ort.flavor.clone())
                .flatten(),
        }
    }
}

/// One image of a dataset as recorded in the results (for the drill-down).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageInfo {
    pub file: String,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Ground truth (or pseudo ground truth) used for scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objects: Option<Vec<GtBox>>,
}

/// A dataset as recorded in the results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageSetInfo {
    pub id: String,
    pub title: String,
    pub kind: SetKind,
    /// Directory of the images (None for the embedded sample).
    #[serde(default)]
    pub dir: Option<std::path::PathBuf>,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub attribution: String,
    #[serde(default)]
    pub source: String,
    pub ground_truth: GroundTruth,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scored_labels: Option<Vec<String>>,
    pub images: Vec<ImageInfo>,
}

impl ImageSetInfo {
    pub fn of(s: &ImageSet) -> Self {
        Self {
            id: s.id.clone(),
            title: s.title.clone(),
            kind: s.kind,
            dir: s.dir.clone(),
            license: s.license.clone(),
            attribution: s.attribution.clone(),
            source: s.source.clone(),
            ground_truth: s.ground_truth.clone(),
            scored_labels: s.scored_labels.clone(),
            images: s
                .images
                .iter()
                .map(|i| ImageInfo {
                    file: i.file.clone(),
                    width: i.width,
                    height: i.height,
                    tags: i.tags.clone(),
                    objects: i.gt.clone(),
                })
                .collect(),
        }
    }

    /// The file of image `file`, when it is one of this set's images (no other path is ever
    /// served). None for the embedded sample.
    pub fn image_path(&self, file: &str) -> Option<std::path::PathBuf> {
        if !super::images::safe_file_name(file) || !self.images.iter().any(|i| i.file == file) {
            return None;
        }
        self.dir.as_ref().map(|d| d.join(file))
    }
}

/// One row of the cross-model ranking: a model on its recommended device.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RankRow {
    pub rank: usize,
    pub model: String,
    pub device: String,
    pub overall: Grade,
    pub overall_points: f64,
    pub accuracy: Option<Grade>,
    pub speed: Grade,
    pub p50_ms: f64,
    pub ap50: Option<f64>,
    /// Macro frame ROC AUC.
    pub roc_auc: Option<f64>,
    /// What the accuracy grade (and the ranking) was computed from.
    pub metric: AccuracyMetric,
    /// The accuracy score of `metric`.
    pub accuracy_score: Option<f64>,
    pub relative: bool,
    pub datasets: Vec<String>,
}

/// The persisted results file: the latest sweep of every benchmarked model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Blue Onyx Prism version that wrote the file.
    pub version: String,
    /// RFC 3339 time of the last update.
    pub timestamp: String,
    pub hardware: HardwareSummary,
    pub runtimes: RuntimeVersions,
    pub models: Vec<ModelResult>,
    /// Datasets the models were run on (images, tags, ground truth).
    #[serde(default)]
    pub image_sets: Vec<ImageSetInfo>,
}

/// `name` is in `only` (normalized names), or `only` is None.
fn wanted(only: Option<&[String]>, name: &str) -> bool {
    only.is_none_or(|o| {
        o.iter()
            .any(|n| crate::registry::normalize_name(n) == crate::registry::normalize_name(name))
    })
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

impl BenchmarkResults {
    pub fn new(
        hardware: HardwareSummary,
        runtimes: RuntimeVersions,
        models: Vec<ModelResult>,
    ) -> Self {
        Self {
            version: crate::VERSION.to_string(),
            timestamp: now(),
            hardware,
            runtimes,
            models,
            image_sets: Vec::new(),
        }
    }

    /// With the datasets the models ran on.
    pub fn with_sets(mut self, sets: &[ImageSet]) -> Self {
        self.image_sets = sets.iter().map(ImageSetInfo::of).collect();
        self
    }

    pub fn image_set(&self, id: &str) -> Option<&ImageSetInfo> {
        self.image_sets.iter().find(|s| s.id == id)
    }

    /// Models ranked by the grades of their recommended device (best first): "the best model
    /// for this machine" is the first row.
    pub fn ranking(&self) -> Vec<RankRow> {
        let mut rows: Vec<(RankRow, Grades)> = self
            .models
            .iter()
            .filter_map(|m| {
                let g = *m.recommended_grades()?;
                let acc = m
                    .devices
                    .iter()
                    .find(|d| Some(&d.device) == m.recommended.as_ref())
                    .and_then(|d| d.run.as_ref())
                    .and_then(|r| r.accuracy.as_ref());
                let ap50 = acc.and_then(|a| a.overall.ap50);
                let roc_auc = acc.and_then(|a| a.roc_auc());
                Some((
                    RankRow {
                        rank: 0,
                        model: m.model.clone(),
                        device: m.recommended.clone()?,
                        overall: g.overall,
                        overall_points: g.overall_points,
                        accuracy: g.accuracy,
                        speed: g.speed,
                        p50_ms: g.p50_ms,
                        ap50,
                        roc_auc,
                        metric: g.metric,
                        accuracy_score: g.accuracy_score,
                        relative: g.relative,
                        datasets: m.datasets.clone(),
                    },
                    g,
                ))
            })
            .collect();
        rows.sort_by(|a, b| Grades::rank_cmp(&a.1, &b.1));
        rows.into_iter()
            .enumerate()
            .map(|(i, (mut r, _))| {
                r.rank = i + 1;
                r
            })
            .collect()
    }

    /// The metric the grades were computed from (the first graded run's; AP50 without any).
    pub fn accuracy_metric(&self) -> AccuracyMetric {
        self.models
            .iter()
            .flat_map(|m| m.devices.iter())
            .find_map(|d| d.run.as_ref()?.grades.filter(|g| g.accuracy.is_some()))
            .map_or(AccuracyMetric::Ap50, |g| g.metric)
    }

    /// Re-grade every run on `metric` with `weights` from its stored accuracy and speed, and
    /// recompute each model's recommendation: models kept from earlier runs (merged results)
    /// are then graded like the new ones, so the ranking compares like with like.
    pub fn regrade(&mut self, metric: AccuracyMetric, weights: Weights) {
        for m in &mut self.models {
            let mut changed = false;
            for d in &mut m.devices {
                let Some(run) = d.run.as_mut() else { continue };
                let Some(old) = run.grades else { continue };
                let score = run.accuracy.as_ref().and_then(|a| a.score(metric));
                let g = Grades::with_metric(metric, score, old.p50_ms, weights, old.relative);
                if g != old {
                    run.grades = Some(g);
                    changed = true;
                }
            }
            if changed {
                m.finish();
            }
        }
    }

    /// A copy without per-image details (for polling).
    pub fn summary(&self) -> Self {
        let mut r = self.clone();
        r.models = r.models.iter().map(ModelResult::summary).collect();
        for s in &mut r.image_sets {
            for i in &mut s.images {
                i.objects = None;
            }
        }
        r
    }

    /// The file at `path`; None when it does not exist.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        serde_json::from_str(&text)
            .map(Some)
            .with_context(|| format!("parsing {}", path.display()))
    }

    /// [`Self::load`], logging (not returning) errors.
    pub fn load_or_warn(path: &Path) -> Option<Self> {
        Self::load(path)
            .map_err(|e| tracing::warn!("{e:#}"))
            .ok()
            .flatten()
    }

    /// Write pretty JSON through a temporary file and a rename (readers never see half a file).
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
    }

    /// Fold `new` into the previous results: models in `new` replace their previous entries
    /// (matched by normalized name), other models are kept, unless the hardware or runtime
    /// versions changed (then the previous results are dropped as not comparable).
    pub fn merge(previous: Option<Self>, new: Self) -> Self {
        let Some(prev) = previous else { return new };
        if prev.hardware != new.hardware || prev.runtimes != new.runtimes {
            return new;
        }
        let key = |m: &ModelResult| crate::registry::normalize_name(&m.model);
        let mut models: Vec<ModelResult> = prev.models;
        for m in new.models {
            match models.iter_mut().find(|p| key(p) == key(&m)) {
                Some(slot) => *slot = m,
                None => models.push(m),
            }
        }
        let mut image_sets = prev.image_sets;
        for s in new.image_sets {
            match image_sets.iter_mut().find(|p| p.id == s.id) {
                Some(slot) => *slot = s,
                None => image_sets.push(s),
            }
        }
        // Keep only the sets some model still refers to.
        image_sets.retain(|s| models.iter().any(|m| m.datasets.contains(&s.id)));
        Self {
            models,
            image_sets,
            ..new
        }
    }

    /// The latest result for model `name` (normalized match).
    pub fn find(&self, name: &str) -> Option<&ModelResult> {
        let key = crate::registry::normalize_name(name);
        self.models
            .iter()
            .find(|m| crate::registry::normalize_name(&m.model) == key)
    }

    /// `(model, best confidence threshold)` of every model with threshold advice, optionally
    /// limited to `only` (normalized names).
    pub fn thresholds(&self, only: Option<&[String]>) -> Vec<(String, f32)> {
        self.models
            .iter()
            .filter(|m| wanted(only, &m.model))
            .filter_map(|m| {
                let t = m.threshold.as_ref()?.apply_value()?;
                Some((m.model.clone(), t))
            })
            .collect()
    }

    /// `(model, recommended device)` of every model with a recommendation, optionally limited
    /// to `only` (normalized names).
    pub fn recommendations(&self, only: Option<&[String]>) -> Vec<(String, String)> {
        let wanted = |name: &str| {
            only.is_none_or(|o| {
                o.iter().any(|n| {
                    crate::registry::normalize_name(n) == crate::registry::normalize_name(name)
                })
            })
        };
        self.models
            .iter()
            .filter(|m| wanted(&m.model))
            .filter_map(|m| m.recommended.as_ref().map(|d| (m.model.clone(), d.clone())))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{StageStats, Stats};
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

    fn scene() -> Vec<Detection> {
        vec![
            det("dog", 0.90, [0, 0, 100, 100]),
            det("car", 0.80, [200, 200, 300, 300]),
            det("bicycle", 0.70, [400, 0, 500, 100]),
        ]
    }

    fn run(spec: &str, p50: f64, dets: Vec<Detection>) -> RunResult {
        let total = Stats {
            min: p50,
            mean: p50,
            p50,
            p95: p50,
            max: p50,
        };
        RunResult {
            model: "m".into(),
            path: "m.onnx".into(),
            family: "yolo5".into(),
            requested_device: spec.into(),
            device: spec.into(),
            device_name: String::new(),
            spec: spec.into(),
            fell_back: false,
            input: "640x640".into(),
            image: "x".into(),
            image_size: "1x1".into(),
            images: 1,
            compile_ms: 10.0,
            cache: "disabled".into(),
            warmup_iterations: 0,
            warmup_ms: 0.0,
            repeat: 1,
            stages_ms: StageStats {
                infer: total,
                total,
                ..Default::default()
            },
            throughput_fps: 1000.0 / p50,
            detections: dets,
            by_resolution: Vec::new(),
            accuracy: None,
            accuracy_note: None,
            grades: Some(Grades::new(None, p50, Default::default(), false)),
            agreement: None,
            per_image: Vec::new(),
            eval_preds: Vec::new(),
        }
    }

    fn dev(spec: &str, p50: f64, dets: Vec<Detection>) -> DeviceResult {
        let d = crate::backend::spec::parse(spec).unwrap();
        DeviceResult::ran(d.device().unwrap(), spec, run(spec, p50, dets))
    }

    fn failed(spec: &str) -> DeviceResult {
        let d = crate::backend::spec::parse(spec).unwrap();
        DeviceResult::failed(d.device().unwrap(), spec, "compile failed".into())
    }

    fn model(devices: Vec<DeviceResult>, configured: Option<&str>) -> ModelResult {
        let mut m = ModelResult {
            model: "m".into(),
            path: "m.onnx".into(),
            family: "yolo5".into(),
            timestamp: now(),
            image: "x".into(),
            datasets: vec!["sample".into()],
            images: 1,
            warmup: 0,
            repeat: 1,
            configured: configured.map(str::to_string),
            reference: None,
            devices,
            skipped: Vec::new(),
            recommended: None,
            recommendation: String::new(),
            error: None,
            threshold_objective: Objective::default(),
            threshold: None,
        };
        m.finish();
        m
    }

    #[test]
    fn agreement_verdicts() {
        let r = scene();
        // Identical: agrees.
        let a = agreement(&r, &r, "openvino:cpu");
        assert_eq!((a.matched, a.score, a.verdict), (3, 1.0, Verdict::Agrees));
        // Slightly shifted boxes and confidences: still agrees.
        let near: Vec<Detection> = r
            .iter()
            .map(|d| Detection {
                confidence: d.confidence - 0.01,
                x_min: d.x_min + 2,
                ..d.clone()
            })
            .collect();
        assert_eq!(agreement(&near, &r, "x").verdict, Verdict::Agrees);
        // One of three missing: 2/3 matched -> minor.
        let a = agreement(&r[..2], &r, "x");
        assert_eq!(a.only_reference, 1);
        assert_eq!(a.verdict, Verdict::Minor);
        // Nothing found where the reference found three: disagrees.
        assert_eq!(agreement(&[], &r, "x").verdict, Verdict::Disagrees);
        // Wrong labels: disagrees.
        let wrong: Vec<Detection> = r
            .iter()
            .map(|d| Detection {
                label: "person".into(),
                ..d.clone()
            })
            .collect();
        assert_eq!(agreement(&wrong, &r, "x").verdict, Verdict::Disagrees);
        // Same boxes, confidences far off (precision problem): disagrees.
        let off: Vec<Detection> = r
            .iter()
            .map(|d| Detection {
                confidence: d.confidence - 0.3,
                ..d.clone()
            })
            .collect();
        assert_eq!(agreement(&off, &r, "x").verdict, Verdict::Disagrees);
        // Both empty: agrees.
        assert_eq!(agreement(&[], &[], "x").verdict, Verdict::Agrees);
    }

    #[test]
    fn recommends_fastest_agreeing_device() {
        let m = model(
            vec![
                dev("ort:coreml", 105.0, scene()),
                dev("openvino:cpu", 37.0, scene()),
                dev("ort:cpu", 60.0, scene()),
            ],
            Some("ort:coreml"),
        );
        assert_eq!(m.reference.as_deref(), Some("openvino:cpu"));
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
        assert!(
            m.recommendation.contains("1.62x faster than ort:cpu"),
            "{}",
            m.recommendation
        );
        let cpu = &m.devices[1];
        assert_eq!(cpu.agreement().unwrap().verdict, Verdict::Reference);
        assert_eq!(m.devices[0].agreement().unwrap().verdict, Verdict::Agrees);
        assert_eq!(m.fastest().unwrap().device, "openvino:cpu");
    }

    #[test]
    fn disagreeing_device_is_never_recommended() {
        let m = model(
            vec![
                dev("ort:coreml", 20.0, vec![]),
                dev("openvino:cpu", 89.0, scene()),
                dev("ort:cpu", 70.0, scene()),
            ],
            None,
        );
        assert_eq!(
            m.devices[0].agreement().unwrap().verdict,
            Verdict::Disagrees
        );
        assert!(!m.devices[0].eligible());
        assert_eq!(m.recommended.as_deref(), Some("ort:cpu"));
        assert!(
            m.recommendation.contains("excluded"),
            "{}",
            m.recommendation
        );
        // ...even though it is the fastest row.
        assert_eq!(m.fastest().unwrap().device, "ort:coreml");
    }

    #[test]
    fn failed_and_fallback_runs_are_excluded() {
        let mut fb = dev("openvino:gpu", 5.0, scene());
        fb.run.as_mut().unwrap().fell_back = true;
        let m = model(
            vec![failed("ort:coreml"), fb, dev("openvino:cpu", 40.0, scene())],
            None,
        );
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
        let m = model(vec![failed("ort:coreml"), failed("openvino:cpu")], None);
        assert_eq!(m.recommended, None);
        assert_eq!(m.reference, None);
        assert!(m.recommendation.contains("no device could run"));
    }

    #[test]
    fn noise_margin_keeps_the_configured_device() {
        // Configured ort:coreml is 4% slower than the fastest: kept.
        let devices = || {
            vec![
                dev("ort:coreml", 41.6, scene()),
                dev("openvino:cpu", 40.0, scene()),
            ]
        };
        let m = model(devices(), Some("ort:coreml"));
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        assert!(
            m.recommendation.contains("within 5%"),
            "{}",
            m.recommendation
        );
        // Without a configured device (or configured elsewhere) the fastest wins.
        assert_eq!(
            model(devices(), None).recommended.as_deref(),
            Some("openvino:cpu")
        );
        // 6% slower: switch.
        let m = model(
            vec![
                dev("ort:coreml", 42.4, scene()),
                dev("openvino:cpu", 40.0, scene()),
            ],
            Some("ort:coreml"),
        );
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
        // A configured device that disagrees is not kept even when within the margin.
        let m = model(
            vec![
                dev("ort:coreml", 40.5, vec![]),
                dev("openvino:cpu", 40.0, scene()),
            ],
            Some("ort:coreml"),
        );
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
    }

    #[test]
    fn ort_cpu_is_the_reference_without_openvino() {
        let m = model(
            vec![
                dev("ort:coreml", 44.0, scene()),
                dev("ort:cpu", 89.0, scene()),
            ],
            None,
        );
        assert_eq!(m.reference.as_deref(), Some("ort:cpu"));
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        // No CPU device at all: unchecked, still recommended, with a note.
        let m = model(vec![dev("ort:coreml", 44.0, scene())], None);
        assert_eq!(m.reference, None);
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        assert!(
            m.recommendation.contains("not checked"),
            "{}",
            m.recommendation
        );
    }

    #[test]
    fn results_round_trip_and_merge() {
        let dir = std::env::temp_dir().join(format!("bop-bench-{}", uuid::Uuid::new_v4()));
        let cfg = dir.join("blue_onyx_prism_config.json");
        let path = results_path(&cfg);
        assert_eq!(path, dir.join(RESULTS_FILE));
        assert!(BenchmarkResults::load(&path).unwrap().is_none());

        let hw = HardwareSummary {
            os: "macos".into(),
            arch: "aarch64".into(),
            cpu: "Apple M1".into(),
            gpus: vec!["#0 Apple M1".into()],
        };
        let rtv = RuntimeVersions {
            openvino: Some("2026.0.0".into()),
            onnxruntime: Some("1.24.4".into()),
            onnxruntime_flavor: Some("coreml".into()),
        };
        let mut a = model(vec![dev("openvino:cpu", 37.0, scene())], None);
        a.model = "IPcam-general".into();
        let mut b = model(
            vec![
                dev("ort:coreml", 44.0, scene()),
                dev("openvino:cpu", 89.0, scene()),
            ],
            None,
        );
        b.model = "dfine-s".into();
        let first = BenchmarkResults::new(hw.clone(), rtv.clone(), vec![a.clone(), b.clone()]);
        first.save(&path).unwrap();
        let back = BenchmarkResults::load(&path).unwrap().unwrap();
        assert_eq!(back, first);
        assert_eq!(
            back.find("ipcam-general.onnx").unwrap().model,
            "IPcam-general"
        );
        assert_eq!(
            back.recommendations(None),
            [
                ("IPcam-general".to_string(), "openvino:cpu".to_string()),
                ("dfine-s".to_string(), "ort:coreml".to_string())
            ]
        );
        assert_eq!(
            back.recommendations(Some(&["DFINE-S".to_string()])).len(),
            1
        );

        // A new run of one model replaces only that model.
        let mut a2 = model(vec![dev("ort:cpu", 50.0, scene())], None);
        a2.model = "ipcam-general".into();
        let second = BenchmarkResults::new(hw.clone(), rtv.clone(), vec![a2.clone()]);
        let merged = BenchmarkResults::merge(Some(back.clone()), second);
        assert_eq!(merged.models.len(), 2);
        assert_eq!(merged.models[0], a2);
        assert_eq!(merged.models[1], b);
        // Different runtimes: the old results are dropped.
        let other = RuntimeVersions {
            onnxruntime: Some("1.25.0".into()),
            ..rtv
        };
        let third = BenchmarkResults::new(hw, other, vec![a2]);
        assert_eq!(BenchmarkResults::merge(Some(back), third).models.len(), 1);

        // Garbage is an error, not a panic.
        std::fs::write(&path, "{not json").unwrap();
        assert!(BenchmarkResults::load(&path).is_err());
        assert!(BenchmarkResults::load_or_warn(&path).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    fn graded(spec: &str, p50: f64, acc: Option<f64>) -> DeviceResult {
        let mut d = dev(spec, p50, scene());
        d.run.as_mut().unwrap().grades = Some(Grades::new(acc, p50, Default::default(), false));
        d
    }

    #[test]
    fn grades_drive_recommendation_and_ranking() {
        // ort:coreml is faster (A speed) but less accurate (C): 0.6*2 + 0.4*4 = 2.8;
        // openvino:cpu: accuracy A, speed B: 0.6*4 + 0.4*3 = 3.6 -> wins.
        let m = model(
            vec![
                graded("ort:coreml", 40.0, Some(0.55)),
                graded("openvino:cpu", 60.0, Some(0.75)),
            ],
            None,
        );
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
        assert!(
            m.recommendation.starts_with("overall A"),
            "{}",
            m.recommendation
        );
        assert_eq!(m.recommended_grades().unwrap().overall, Grade::A);
        // Configured device kept only when as accurate.
        let m = model(
            vec![
                graded("ort:coreml", 61.0, Some(0.745)),
                graded("openvino:cpu", 60.0, Some(0.75)),
            ],
            Some("ort:coreml"),
        );
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));

        // Same letter grades and accuracy within noise (yolo26n: OpenVINO f16 on ARM is ~1 AP
        // off): the clearly faster device wins, and the reason reads as a speedup.
        let m = model(
            vec![
                graded("ort:coreml", 14.4, Some(0.481)),
                graded("openvino:cpu", 35.0, Some(0.490)),
                graded("ort:cpu", 33.7, Some(0.481)),
            ],
            None,
        );
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        assert!(
            m.recommendation.contains("2.34x faster than ort:cpu"),
            "{}",
            m.recommendation
        );
        // A faster but clearly less accurate device loses, and the reason says so.
        let m = model(
            vec![
                graded("ort:coreml", 40.0, Some(0.55)),
                graded("openvino:cpu", 60.0, Some(0.75)),
            ],
            None,
        );
        assert!(
            m.recommendation
                .contains("ort:coreml is 1.50x faster (40.0 ms) but less accurate"),
            "{}",
            m.recommendation
        );

        let mut a = model(vec![graded("openvino:cpu", 30.0, Some(0.5))], None);
        a.model = "small".into();
        let mut b = model(vec![graded("ort:coreml", 45.0, Some(0.8))], None);
        b.model = "big".into();
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![a, b]);
        let rank = r.ranking();
        assert_eq!(rank.len(), 2);
        assert_eq!((rank[0].rank, rank[0].model.as_str()), (1, "big"));
        assert_eq!(rank[0].overall, Grade::A);
        assert_eq!(rank[1].device, "openvino:cpu");
    }

    fn graded_auc(spec: &str, p50: f64, auc: Option<f64>) -> DeviceResult {
        let mut d = dev(spec, p50, scene());
        d.run.as_mut().unwrap().grades = Some(Grades::with_metric(
            AccuracyMetric::RocAuc,
            auc,
            p50,
            Default::default(),
            false,
        ));
        d
    }

    #[test]
    fn roc_auc_metric_drives_tie_ranking_and_text() {
        // AUC within 0.01 of the best counts as equal (the AUC scale): the faster device wins.
        let m = model(
            vec![
                graded_auc("ort:coreml", 14.0, Some(0.952)),
                graded_auc("openvino:cpu", 35.0, Some(0.960)),
            ],
            None,
        );
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        assert!(
            m.recommendation.contains("accuracy A by ROC AUC"),
            "{}",
            m.recommendation
        );
        // 0.015 apart is a real difference on the AUC scale (it would tie for AP50).
        let m = model(
            vec![
                graded_auc("ort:coreml", 14.0, Some(0.945)),
                graded_auc("openvino:cpu", 35.0, Some(0.960)),
            ],
            None,
        );
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
        let ap = model(
            vec![
                graded("ort:coreml", 14.0, Some(0.745)),
                graded("openvino:cpu", 35.0, Some(0.760)),
            ],
            None,
        );
        assert_eq!(ap.recommended.as_deref(), Some("ort:coreml"));
        // Ranking rows say which metric graded them.
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![m.clone()]);
        let rows = r.ranking();
        assert_eq!(rows[0].metric, AccuracyMetric::RocAuc);
        assert_eq!(rows[0].accuracy_score, Some(0.960));
        assert_eq!(r.accuracy_metric(), AccuracyMetric::RocAuc);
    }

    #[test]
    fn regrade_switches_metric_from_stored_accuracy() {
        use super::super::AccuracyReport;
        use super::super::metrics::{ClassMetrics, Summary};
        use super::super::roc::FrameRoc;
        // Model a: AP50 0.80, AUC 0.85; model b: AP50 0.55, AUC 0.97. AP50 ranks a first,
        // ROC AUC ranks b first.
        let with_acc = |name: &str, ap: f64, auc: f64| {
            let mut d = graded("openvino:cpu", 40.0, Some(ap));
            d.run.as_mut().unwrap().accuracy = Some(AccuracyReport {
                ground_truth: "ground truth".into(),
                relative: false,
                classes: vec!["person".into()],
                threshold: 0.5,
                overall: Summary {
                    ap50: Some(ap),
                    per_class: vec![ClassMetrics {
                        class: "person".into(),
                        gt: 3,
                        ap50: Some(ap),
                        ap50_95: None,
                        precision: None,
                        recall: None,
                    }],
                    ..Default::default()
                },
                by_dataset: vec![],
                by_tag: vec![],
                all_classes: None,
                sweep: None,
                frame_roc: Some(FrameRoc {
                    auc: Some(auc),
                    ..Default::default()
                }),
            });
            let mut m = model(vec![d], None);
            m.model = name.into();
            m
        };
        let mut r = BenchmarkResults::new(
            Default::default(),
            Default::default(),
            vec![with_acc("a", 0.80, 0.85), with_acc("b", 0.55, 0.97)],
        );
        let names = |r: &BenchmarkResults| -> Vec<String> {
            r.ranking().into_iter().map(|x| x.model).collect()
        };
        assert_eq!(names(&r), ["a", "b"]);
        assert_eq!(r.accuracy_metric(), AccuracyMetric::Ap50);
        r.regrade(AccuracyMetric::RocAuc, Default::default());
        assert_eq!(r.accuracy_metric(), AccuracyMetric::RocAuc);
        assert_eq!(names(&r), ["b", "a"]);
        let g = r.models[1].recommended_grades().unwrap();
        assert_eq!((g.accuracy, g.accuracy_score), (Some(Grade::A), Some(0.97)));
        assert!(r.models[1].recommendation.contains("by ROC AUC"));
        // And back.
        r.regrade(AccuracyMetric::Ap50, Default::default());
        assert_eq!(names(&r), ["a", "b"]);
        // A run without frame ROC (older results) is unscored under ROC AUC.
        let mut old = with_acc("old", 0.7, 0.9);
        old.devices[0]
            .run
            .as_mut()
            .unwrap()
            .accuracy
            .as_mut()
            .unwrap()
            .frame_roc = None;
        let mut r = BenchmarkResults::new(Default::default(), Default::default(), vec![old]);
        r.regrade(AccuracyMetric::RocAuc, Default::default());
        assert_eq!(r.models[0].recommended_grades().unwrap().accuracy, None);
    }

    #[test]
    fn agreement_over_many_images() {
        let img = |dets: Vec<Detection>| super::super::ImageRun {
            set: "s".into(),
            file: "f".into(),
            total_ms: 1.0,
            infer_ms: 1.0,
            preds: dets
                .iter()
                .map(|d| super::super::ImagePred {
                    label: d.label.clone(),
                    confidence: d.confidence,
                    bbox: [
                        d.x_min as f32,
                        d.y_min as f32,
                        d.x_max as f32,
                        d.y_max as f32,
                    ],
                    tp: None,
                })
                .collect(),
            gt_matched: vec![],
            counts: None,
        };
        let mut reference = run("openvino:cpu", 10.0, vec![]);
        reference.per_image = vec![img(scene()), img(scene())];
        let mut same = run("ort:coreml", 5.0, vec![]);
        same.per_image = vec![img(scene()), img(scene())];
        let a = run_agreement(&same, &reference, "openvino:cpu");
        assert_eq!((a.matched, a.verdict), (6, Verdict::Agrees));
        // Second image empty: 3 of 6 -> minor.
        let mut half = run("ort:coreml", 5.0, vec![]);
        half.per_image = vec![img(scene()), img(vec![])];
        let a = run_agreement(&half, &reference, "openvino:cpu");
        assert_eq!(
            (a.matched, a.only_reference, a.verdict),
            (3, 3, Verdict::Minor)
        );
        let mut none = run("ort:coreml", 5.0, vec![]);
        none.per_image = vec![img(vec![]), img(vec![])];
        assert_eq!(
            run_agreement(&none, &reference, "x").verdict,
            Verdict::Disagrees
        );
    }

    #[test]
    fn image_paths_are_restricted_to_the_set() {
        let info = ImageSetInfo {
            id: "s".into(),
            title: "s".into(),
            kind: SetKind::Dir,
            dir: Some("/data/s".into()),
            license: String::new(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: GroundTruth::None,
            scored_labels: None,
            images: vec![ImageInfo {
                file: "a.jpg".into(),
                width: 1,
                height: 1,
                tags: vec![],
                objects: None,
            }],
        };
        assert_eq!(info.image_path("a.jpg"), Some("/data/s/a.jpg".into()));
        assert_eq!(info.image_path("b.jpg"), None);
        assert_eq!(info.image_path("../a.jpg"), None);
    }

    /// A run with an accuracy report whose sweep comes from `preds` (confidence, TP) over
    /// `objects` objects.
    fn with_sweep(mut d: DeviceResult, objects: usize, preds: &[(f32, bool)]) -> DeviceResult {
        use super::super::metrics::{Breakpoints, SweepImage, SweepPred, SweepScope};
        use super::super::threshold::{Curve, ThresholdSweep, exact_pick, objectives};
        let img = SweepImage {
            gt: vec![("person".into(), false); objects],
            preds: preds
                .iter()
                .map(|&(confidence, tp)| SweepPred {
                    confidence,
                    class: "person".into(),
                    tp,
                    small: false,
                })
                .collect(),
            frames: vec![],
        };
        let bp = Breakpoints::new(&[&img], SweepScope::All);
        let curve = |key: &str| Curve {
            key: key.into(),
            images: 1,
            gt: objects,
            relative: false,
            points: super::super::metrics::SWEEP_THRESHOLDS
                .iter()
                .map(|&t| bp.point_at(t))
                .collect(),
            configured: bp.point_at(0.5),
            picks: objectives(Objective::F1)
                .iter()
                .filter_map(|o| exact_pick(&bp, *o))
                .collect(),
            exact: bp.decimated(50),
            roc_auc: None,
        };
        let run = d.run.as_mut().unwrap();
        run.accuracy = Some(super::super::AccuracyReport {
            ground_truth: "ground truth".into(),
            relative: false,
            classes: vec!["person".into()],
            threshold: 0.5,
            overall: Default::default(),
            by_dataset: vec![],
            by_tag: vec![],
            all_classes: None,
            sweep: Some(ThresholdSweep {
                configured: 0.5,
                overall: curve("overall"),
                by_dataset: vec![curve("exdark-night")],
                by_tag: vec![],
                per_class: vec![curve("person")],
                roc: None,
            }),
            frame_roc: None,
        });
        d
    }

    #[test]
    fn threshold_advice_from_the_recommended_device() {
        // CPU (reference) and a faster CoreML that agrees: CoreML is recommended and its curve
        // is used. Kept sets: CPU best keeps down to 0.45, CoreML down to 0.35.
        let cpu = with_sweep(
            dev("openvino:cpu", 50.0, scene()),
            3,
            &[(0.9, true), (0.6, true), (0.45, true), (0.2, false)],
        );
        let coreml = with_sweep(
            dev("ort:coreml", 10.0, scene()),
            3,
            &[
                (0.9, true),
                (0.6, true),
                (0.55, false),
                (0.35, true),
                (0.2, false),
            ],
        );
        let m = model(vec![cpu, coreml], None);
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        let a = m.threshold.as_ref().unwrap();
        assert_eq!(a.device, "ort:coreml");
        assert_eq!(a.objective, "f1");
        let best = a.best.as_ref().unwrap();
        // F1: {.9,.6} .8, {+.55 FP} .667, {+.35} .857, {+.2 FP} .75 -> (0.2, 0.35] -> 0.27.
        assert_eq!(best.exact_threshold, Some(0.35));
        assert_eq!(best.threshold, 0.27);
        assert_eq!(a.configured.threshold, 0.5);
        assert_eq!(a.by_device.len(), 2);
        let cpu_best = a.by_device[0].best.as_ref().unwrap();
        assert_eq!(cpu_best.exact_threshold, Some(0.45));
        assert_eq!(a.alternatives.len(), 3);
        assert_eq!(a.by_dataset[0].key, "exdark-night");
        assert_eq!(a.per_class[0].key, "person");
        assert!(!a.exact.is_empty() && a.curve.len() == 19);
        assert!(
            m.recommendation
                .contains("P/R at the best confidence threshold 0.27 (F1)"),
            "{}",
            m.recommendation
        );
        assert!(m.recommendation.contains("at the configured 0.50"));
        // Polling summary: no per-device curves, the advice stays.
        let s = m.summary();
        assert!(s.devices.iter().all(|d| {
            d.run
                .as_ref()
                .unwrap()
                .accuracy
                .as_ref()
                .unwrap()
                .sweep
                .is_none()
        }));
        assert_eq!(s.threshold, m.threshold);
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![m.clone()]);
        assert_eq!(r.thresholds(None), [("m".to_string(), 0.27)]);
        assert!(r.thresholds(Some(&["other".into()])).is_empty());

        // Another objective: F2 favors the lower threshold that finds everything.
        let mut m2 = m.clone();
        m2.threshold_objective = Objective::F2;
        m2.finish();
        let b2 = m2.threshold.as_ref().unwrap().best.as_ref().unwrap();
        assert_eq!(b2.objective, "f2");
        assert!(b2.point.recall.unwrap() >= best.point.recall.unwrap());

        // Results of older versions (no sweep, no advice fields) still load.
        let mut v = serde_json::to_value(&m).unwrap();
        let o = v.as_object_mut().unwrap();
        o.remove("threshold");
        o.remove("threshold_objective");
        for d in o["devices"].as_array_mut().unwrap() {
            d["run"]["accuracy"]
                .as_object_mut()
                .unwrap()
                .remove("sweep");
        }
        let mut old: ModelResult = serde_json::from_value(v).unwrap();
        assert!(old.threshold.is_none());
        assert_eq!(old.threshold_objective, Objective::F1);
        old.finish();
        assert!(old.threshold.is_none());
        // A round trip keeps everything.
        let back: ModelResult = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back, m);
    }
}
