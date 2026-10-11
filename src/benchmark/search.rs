//! Confidence-threshold search over stored predictions: no inference.
//!
//! Every benchmark run keeps, per model and device, the predictions it made at the evaluation
//! threshold ([`super::EVAL_CONFIDENCE`]) in a scored class, per image, in post-processing
//! order. They are persisted in [`PREDS_FILE`] beside `benchmark.json` (kept out of the main
//! file so polling stays light) as `[class index, confidence, x_min, y_min, x_max, y_max]`
//! (integer pixel boxes, exactly what was scored). The ground truth (or pseudo ground truth) is
//! the one recorded in `benchmark.json`'s `image_sets`.
//!
//! [`search`] re-runs the exact sorted sweep ([`super::metrics::Breakpoints`],
//! [`super::threshold::exact_pick`]) over any subset: datasets, images carrying given tags
//! (e.g. `night`), scored classes, IoU and objective. Filtering the stored predictions at a
//! higher threshold is exact for every family (see [`super::metrics`]), so the search equals a
//! benchmark run at that threshold, without running one.
//!
//! The same stored predictions give the frame-level ROC ([`super::roc`]): a frame's score for a
//! class is its most confident stored prediction of that class, so the search reports frame ROC
//! AUC (overall, per dataset, per class) and can optimize the frame objectives (`youden`,
//! `fpr:<x>`) without inference.

use super::images::GroundTruth;
use super::metrics::{
    Breakpoints, ClassMap, EvalImage, GtBox, PredBox, SWEEP_THRESHOLDS, SweepImage, SweepScope,
    ThresholdPoint, canonical_label,
};
use super::report::{BenchmarkResults, ImageInfo, ImageSetInfo, ModelResult};
use super::roc::{FrameRoc, ROC_POINTS};
use super::threshold::{GroupPick, Objective, Pick, exact_pick};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// File name of the stored predictions, beside `benchmark.json`.
pub const PREDS_FILE: &str = "benchmark-preds.json";
/// Points of the exact curve a search returns.
pub const SEARCH_EXACT_POINTS: usize = 200;

/// `benchmark-preds.json` beside `config_path`.
pub fn preds_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .map(|d| d.join(PREDS_FILE))
        .unwrap_or_else(|| PathBuf::from(PREDS_FILE))
}

/// One stored prediction: `[class index, confidence, x_min, y_min, x_max, y_max]`, the box in
/// integer pixels (the scored predictions are integer pixel boxes, so this is lossless).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(
    from = "(u32, f32, u32, u32, u32, u32)",
    into = "(u32, f32, u32, u32, u32, u32)"
)]
pub struct StoredPred {
    pub class: u32,
    pub confidence: f32,
    pub bbox: [f32; 4],
}

impl From<(u32, f32, u32, u32, u32, u32)> for StoredPred {
    fn from(t: (u32, f32, u32, u32, u32, u32)) -> Self {
        Self {
            class: t.0,
            confidence: t.1,
            bbox: [t.2 as f32, t.3 as f32, t.4 as f32, t.5 as f32],
        }
    }
}

impl From<StoredPred> for (u32, f32, u32, u32, u32, u32) {
    fn from(p: StoredPred) -> Self {
        let c = |v: f32| v.max(0.0).round() as u32;
        (
            p.class,
            p.confidence,
            c(p.bbox[0]),
            c(p.bbox[1]),
            c(p.bbox[2]),
            c(p.bbox[3]),
        )
    }
}

/// The stored predictions of one image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredImage {
    /// Dataset id.
    pub set: String,
    pub file: String,
    pub preds: Vec<StoredPred>,
}

/// The stored predictions of one model on one device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredRun {
    pub model: String,
    /// Device spec ("ort:coreml").
    pub device: String,
    pub timestamp: String,
    /// Scored classes in the model's label space; predictions refer to them by index.
    pub classes: Vec<String>,
    /// The run's configured confidence threshold.
    pub configured: f32,
    /// Lowest stored confidence (the evaluation threshold of the run).
    pub eval_confidence: f32,
    pub images: Vec<StoredImage>,
}

impl StoredRun {
    /// The stored form of a device run (None without accuracy or stored predictions).
    pub fn of(m: &ModelResult, device: &str, run: &super::RunResult) -> Option<Self> {
        let acc = run.accuracy.as_ref()?;
        if run.eval_preds.len() != run.per_image.len() || run.per_image.is_empty() {
            return None;
        }
        let index: HashMap<&str, u32> = acc
            .classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i as u32))
            .collect();
        Some(Self {
            model: m.model.clone(),
            device: device.to_string(),
            timestamp: m.timestamp.clone(),
            classes: acc.classes.clone(),
            configured: acc.threshold,
            eval_confidence: super::EVAL_CONFIDENCE.min(acc.threshold),
            images: run
                .per_image
                .iter()
                .zip(&run.eval_preds)
                .map(|(img, preds)| StoredImage {
                    set: img.set.clone(),
                    file: img.file.clone(),
                    preds: preds
                        .iter()
                        .filter_map(|p| {
                            Some(StoredPred {
                                class: *index.get(p.label.as_str())?,
                                confidence: p.confidence,
                                bbox: p.bbox,
                            })
                        })
                        .collect(),
                })
                .collect(),
        })
    }

    /// The predictions of image `k` as scored boxes.
    fn preds(&self, k: usize) -> Vec<PredBox> {
        self.images[k]
            .preds
            .iter()
            .filter_map(|p| {
                Some(PredBox {
                    label: self.classes.get(p.class as usize)?.clone(),
                    confidence: p.confidence,
                    bbox: p.bbox,
                })
            })
            .collect()
    }
}

/// The stored-predictions file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StoredPreds {
    pub version: String,
    pub runs: Vec<StoredRun>,
}

impl StoredPreds {
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

    /// Compact JSON through a temporary file and a rename.
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string(self)?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
    }

    /// The previous runs of models still in `merged` and not re-run in `new`, plus the runs of
    /// `new` (like [`BenchmarkResults::merge`]).
    pub fn merge(previous: Option<Self>, merged: &BenchmarkResults, new: &[ModelResult]) -> Self {
        let key = crate::registry::normalize_name;
        let mut runs: Vec<StoredRun> = previous
            .map(|p| p.runs)
            .unwrap_or_default()
            .into_iter()
            .filter(|r| {
                !new.iter().any(|m| key(&m.model) == key(&r.model))
                    && merged.find(&r.model).is_some()
            })
            .collect();
        for m in new {
            for d in &m.devices {
                if let Some(run) = d.run.as_ref().filter(|_| d.ok())
                    && let Some(s) = StoredRun::of(m, &d.device, run)
                {
                    runs.push(s);
                }
            }
        }
        Self {
            version: crate::VERSION.to_string(),
            runs,
        }
    }

    /// The stored runs of `model` (normalized match).
    pub fn runs_of(&self, model: &str) -> Vec<&StoredRun> {
        let key = crate::registry::normalize_name(model);
        self.runs
            .iter()
            .filter(|r| crate::registry::normalize_name(&r.model) == key)
            .collect()
    }
}

/// Merge the predictions of `new` into the file beside `results_file` (logs, does not fail the
/// benchmark, on error).
pub fn save_beside(results_file: &Path, merged: &BenchmarkResults, new: &[ModelResult]) {
    let path = results_file
        .parent()
        .map(|d| d.join(PREDS_FILE))
        .unwrap_or_else(|| PathBuf::from(PREDS_FILE));
    let prev = StoredPreds::load(&path)
        .map_err(|e| tracing::warn!("{e:#}"))
        .ok()
        .flatten();
    if let Err(e) = StoredPreds::merge(prev, merged, new).save(&path) {
        tracing::warn!("saving the stored predictions: {e:#}");
    }
}

/// What to search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Model names; empty = every model with stored predictions.
    pub models: Vec<String>,
    /// Device spec; None = each model's recommended device.
    pub device: Option<String>,
    /// Dataset ids; empty = all.
    pub datasets: Vec<String>,
    /// Only images carrying every one of these tags (e.g. `night`).
    pub tags: Vec<String>,
    /// Scored classes to count (model label space, e.g. `person`); empty = all.
    pub classes: Vec<String>,
    pub objective: Objective,
    /// Matching IoU.
    pub iou: f32,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            device: None,
            datasets: Vec::new(),
            tags: Vec::new(),
            classes: Vec::new(),
            objective: Objective::F1,
            iou: 0.5,
        }
    }
}

impl SearchRequest {
    /// Form fields: `model` (repeated, or `all`), `device` ("" = recommended), `dataset`, `tag`,
    /// `class` (repeated), `objective` (`f1`, `f2`, `precision:<p>`, `recall:<r>`), `iou`.
    pub fn from_form(form: &[(String, String)], default_objective: Objective) -> Result<Self> {
        let all = |k: &str| -> Vec<String> {
            form.iter()
                .filter(|(key, _)| key == k)
                .flat_map(|(_, v)| v.split(','))
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .collect()
        };
        let one = |k: &str| {
            form.iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.trim())
                .filter(|v| !v.is_empty())
        };
        let mut models = all("model");
        if models.iter().any(|m| m.eq_ignore_ascii_case("all")) {
            models.clear();
        }
        let req = Self {
            models,
            device: one("device")
                .filter(|d| !d.eq_ignore_ascii_case("recommended"))
                .map(str::to_string),
            datasets: all("dataset"),
            tags: all("tag"),
            classes: all("class"),
            objective: match one("objective") {
                Some(o) => o.parse().map_err(|e: String| anyhow::anyhow!(e))?,
                None => default_objective,
            },
            iou: match one("iou") {
                Some(v) => v
                    .parse()
                    .map_err(|_| anyhow::anyhow!("iou: '{v}' is not a number"))?,
                None => 0.5,
            },
        };
        req.validate()?;
        Ok(req)
    }

    pub fn validate(&self) -> Result<()> {
        if !(self.iou.is_finite() && self.iou > 0.0 && self.iou < 1.0) {
            bail!("iou: must be between 0 and 1 (e.g. 0.5)");
        }
        if let Some(d) = &self.device {
            crate::backend::spec::parse(d).map_err(|e| anyhow::anyhow!("device: {e}"))?;
        }
        Ok(())
    }
}

/// The search result of one model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub model: String,
    pub device: String,
    pub objective: String,
    pub iou: f32,
    /// "ground truth" or "relative to <reference>, not ground truth".
    pub ground_truth: String,
    pub relative: bool,
    pub images: usize,
    /// Scored ground-truth objects.
    pub gt: usize,
    /// Classes counted.
    pub classes: Vec<String>,
    /// At the threshold the run was configured with.
    pub configured: ThresholdPoint,
    /// At the model's threshold in the config now (when it differs from the run's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<ThresholdPoint>,
    pub best: Option<Pick>,
    /// The best for each standard objective (comparison).
    pub alternatives: Vec<Pick>,
    /// At [`SWEEP_THRESHOLDS`].
    pub grid: Vec<ThresholdPoint>,
    /// The exact curve, thinned to [`SEARCH_EXACT_POINTS`].
    pub exact: Vec<ThresholdPoint>,
    pub by_dataset: Vec<GroupPick>,
    pub per_class: Vec<GroupPick>,
    /// Frame-level ROC of the searched images (macro / micro / per class AUC and the curve).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roc: Option<FrameRoc>,
}

/// A search over several models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchOutput {
    pub results: Vec<SearchResult>,
    /// Models that could not be searched, and why.
    pub errors: Vec<String>,
    pub elapsed_ms: f64,
}

/// What can be searched (the page's controls).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchOptions {
    /// (model, devices with stored predictions, recommended device).
    pub models: Vec<SearchModel>,
    pub datasets: Vec<String>,
    pub tags: Vec<String>,
    pub classes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchModel {
    pub model: String,
    pub devices: Vec<String>,
    pub recommended: Option<String>,
}

/// The controls of the search page from the stored predictions and results.
pub fn options(results: &BenchmarkResults, preds: &StoredPreds) -> SearchOptions {
    let mut models: Vec<SearchModel> = Vec::new();
    let mut datasets: Vec<String> = Vec::new();
    let mut classes: Vec<String> = Vec::new();
    for r in &preds.runs {
        match models.iter_mut().find(|m| m.model == r.model) {
            Some(m) => m.devices.push(r.device.clone()),
            None => models.push(SearchModel {
                model: r.model.clone(),
                devices: vec![r.device.clone()],
                recommended: results.find(&r.model).and_then(|m| m.recommended.clone()),
            }),
        }
        for i in &r.images {
            if !datasets.contains(&i.set) {
                datasets.push(i.set.clone());
            }
        }
        for c in &r.classes {
            if !classes.contains(c) {
                classes.push(c.clone());
            }
        }
    }
    let mut tags: Vec<String> = results
        .image_sets
        .iter()
        .filter(|s| datasets.contains(&s.id))
        .flat_map(|s| s.images.iter().flat_map(|i| i.tags.iter().cloned()))
        .collect();
    tags.sort();
    tags.dedup();
    SearchOptions {
        models,
        datasets,
        tags,
        classes,
    }
}

/// The run of `model` to search: on `device`, else the recommended device, else the CPU
/// reference, else the first stored one.
fn pick_run<'a>(
    results: &BenchmarkResults,
    runs: &[&'a StoredRun],
    model: &str,
    device: Option<&str>,
) -> Result<&'a StoredRun> {
    if let Some(d) = device {
        let want = crate::backend::spec::parse(d)
            .map(|s| s.to_string())
            .unwrap_or_else(|_| d.to_string());
        return runs
            .iter()
            .find(|r| r.device == want)
            .copied()
            .with_context(|| {
                format!(
                    "no stored predictions on {want} (stored: {})",
                    runs.iter()
                        .map(|r| r.device.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
    }
    let rec = results.find(model).and_then(|m| m.recommended.clone());
    let reference = super::REFERENCE_DEVICES.map(|d| d.to_string());
    rec.iter()
        .chain(reference.iter())
        .find_map(|d| runs.iter().find(|r| r.device == *d).copied())
        .or_else(|| runs.first().copied())
        .context("no stored predictions")
}

/// Search the stored predictions (see the module docs). `current` gives a model's threshold in
/// the config now. Errors when there are no stored predictions at all.
pub fn search(
    results: &BenchmarkResults,
    preds: &StoredPreds,
    req: &SearchRequest,
    current: &dyn Fn(&str) -> Option<f32>,
) -> Result<SearchOutput> {
    let started = Instant::now();
    if preds.runs.is_empty() {
        bail!("no stored predictions; run a benchmark first");
    }
    req.validate()?;
    let mut names: Vec<String> = if req.models.is_empty() {
        let mut v: Vec<String> = Vec::new();
        for r in &preds.runs {
            if !v.contains(&r.model) {
                v.push(r.model.clone());
            }
        }
        v
    } else {
        req.models.clone()
    };
    names.dedup();
    let sets: HashMap<&str, &ImageSetInfo> = results
        .image_sets
        .iter()
        .map(|s| (s.id.as_str(), s))
        .collect();
    let mut out = SearchOutput {
        results: Vec::new(),
        errors: Vec::new(),
        elapsed_ms: 0.0,
    };
    for name in &names {
        let runs = preds.runs_of(name);
        if runs.is_empty() {
            out.errors.push(format!(
                "{name}: no stored predictions; run a benchmark of this model first"
            ));
            continue;
        }
        let run = match pick_run(results, &runs, name, req.device.as_deref()) {
            Ok(r) => r,
            Err(e) => {
                out.errors.push(format!("{name}: {e:#}"));
                continue;
            }
        };
        match search_run(run, &sets, req, current(&run.model)) {
            Ok(r) => out.results.push(r),
            Err(e) => out.errors.push(format!("{name}: {e:#}")),
        }
    }
    out.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok(out)
}

/// Search one stored run.
fn search_run(
    run: &StoredRun,
    sets: &HashMap<&str, &ImageSetInfo>,
    req: &SearchRequest,
    current: Option<f32>,
) -> Result<SearchResult> {
    let map = ClassMap::for_model(&run.classes);
    let wanted: Vec<String> = if req.classes.is_empty() {
        map.classes.clone()
    } else {
        let mut v = Vec::new();
        for c in &req.classes {
            let canon = canonical_label(c);
            let class = map
                .classes
                .iter()
                .find(|k| **k == canon)
                .or_else(|| {
                    map.gt_class(&canon)
                        .and_then(|g| map.classes.iter().find(|k| *k == g))
                })
                .with_context(|| {
                    format!(
                        "class '{c}' is not scored for this model (scored: {})",
                        map.classes.join(", ")
                    )
                })?;
            if !v.contains(class) {
                v.push(class.clone());
            }
        }
        v
    };
    // (set, image info, eval input, classes annotated) of every searched image.
    let mut inputs: Vec<(&ImageSetInfo, &ImageInfo, EvalImage, Vec<String>)> = Vec::new();
    for (k, img) in run.images.iter().enumerate() {
        if !req.datasets.is_empty() && !req.datasets.contains(&img.set) {
            continue;
        }
        let Some(set) = sets.get(img.set.as_str()) else {
            continue;
        };
        let Some(info) = set.images.iter().find(|i| i.file == img.file) else {
            continue;
        };
        if !req.tags.iter().all(|t| info.tags.iter().any(|x| x == t)) {
            continue;
        }
        let Some(gt) = &info.objects else { continue };
        let (_, mut eval, allowed) =
            super::eval_input(&map, set.scored_labels.as_deref(), gt, &run.preds(k));
        eval.gt.retain(|g| wanted.contains(&g.label));
        eval.preds.retain(|p| wanted.contains(&p.label));
        let mut fc = super::frame_classes(&map, allowed.as_deref());
        fc.retain(|c| wanted.contains(c));
        inputs.push((set, info, eval, fc));
    }
    if inputs.is_empty() {
        bail!("no scored images match these filters");
    }
    let real: Vec<usize> = (0..inputs.len())
        .filter(|&k| inputs[k].0.ground_truth.is_real())
        .collect();
    let (chosen, relative) = if real.is_empty() {
        ((0..inputs.len()).collect::<Vec<_>>(), true)
    } else {
        (real, false)
    };
    let matched: Vec<SweepImage> = inputs
        .iter()
        .map(|(_, _, e, fc)| SweepImage::with_iou(e, req.iou).with_frames(e, fc))
        .collect();
    let roc_of = |ks: &[usize], classes: &[String], points: usize| {
        FrameRoc::of(
            ks.iter().flat_map(|&k| matched[k].frames.iter()),
            classes,
            points,
        )
    };
    let refs = |ks: &[usize]| -> Vec<&SweepImage> { ks.iter().map(|&k| &matched[k]).collect() };
    let overall = Breakpoints::new(&refs(&chosen), SweepScope::All);
    if overall.npos == 0 {
        bail!("no ground-truth objects of these classes in the matching images");
    }
    let group = |key: &str, bp: &Breakpoints, relative: bool, roc_auc: Option<f64>| GroupPick {
        key: key.to_string(),
        images: bp.images,
        gt: bp.npos,
        relative,
        configured: bp.point_at(run.configured),
        best: exact_pick(bp, req.objective),
        roc_auc,
    };
    let mut by_dataset = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for (s, _, _, _) in &inputs {
        if seen.contains(&s.id.as_str()) {
            continue;
        }
        seen.push(&s.id);
        let ks: Vec<usize> = (0..inputs.len())
            .filter(|&k| inputs[k].0.id == s.id)
            .collect();
        let bp = Breakpoints::new(&refs(&ks), SweepScope::All);
        if bp.npos > 0 {
            let auc = roc_of(&ks, &wanted, 0).auc;
            by_dataset.push(group(&s.id, &bp, !s.ground_truth.is_real(), auc));
        }
    }
    let roc = roc_of(&chosen, &wanted, ROC_POINTS);
    let per_class = wanted
        .iter()
        .map(|c| (c, Breakpoints::new(&refs(&chosen), SweepScope::Class(c))))
        .filter(|(_, bp)| bp.npos > 0)
        .map(|(c, bp)| group(c, &bp, relative, roc.class_auc(c)))
        .collect();
    let ground_truth = if relative {
        let refs: std::collections::BTreeSet<String> = chosen
            .iter()
            .filter_map(|&k| match &inputs[k].0.ground_truth {
                GroundTruth::Pseudo(r) => Some(r.clone()),
                _ => None,
            })
            .collect();
        format!(
            "relative to {}, not ground truth",
            refs.into_iter().collect::<Vec<_>>().join(", ")
        )
    } else {
        "ground truth".to_string()
    };
    Ok(SearchResult {
        model: run.model.clone(),
        device: run.device.clone(),
        objective: req.objective.to_string(),
        iou: req.iou,
        ground_truth,
        relative,
        images: overall.images,
        gt: overall.npos,
        classes: wanted.clone(),
        configured: overall.point_at(run.configured),
        current: current
            .filter(|c| (c - run.configured).abs() > 1e-6)
            .map(|c| overall.point_at(c)),
        best: exact_pick(&overall, req.objective),
        alternatives: Objective::STANDARD
            .iter()
            .filter_map(|o| exact_pick(&overall, *o))
            .collect(),
        grid: SWEEP_THRESHOLDS
            .iter()
            .map(|&t| overall.point_at(t))
            .collect(),
        exact: overall.decimated(SEARCH_EXACT_POINTS),
        by_dataset,
        per_class,
        roc: (roc.positives + roc.negatives > 0).then_some(roc),
    })
}

/// CSV of the search summary and each model's 0.05 grid.
pub fn to_csv(out: &SearchOutput) -> String {
    let f = |v: Option<f64>| v.map_or(String::new(), |v| format!("{v:.4}"));
    let ft = |p: Option<&ThresholdPoint>| p.and_then(|p| p.frame).and_then(|f| f.tpr);
    let ff = |p: Option<&ThresholdPoint>| p.and_then(|p| p.frame).and_then(|f| f.fpr);
    let mut s = String::from(
        "model,device,objective,configured,configured_precision,configured_recall,configured_f1,best,best_precision,best_recall,best_f1,f1_gain,roc_auc,roc_auc_micro,configured_frame_tpr,configured_frame_fpr,best_frame_tpr,best_frame_fpr\n",
    );
    for r in &out.results {
        let b = r.best.as_ref();
        let roc = r.roc.as_ref();
        s.push_str(&format!(
            "{},{},{},{:.2},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            csv_field(&r.model),
            r.device,
            r.objective,
            r.configured.threshold,
            f(r.configured.precision),
            f(r.configured.recall),
            f(r.configured.f1),
            b.map_or(String::new(), |b| format!("{:.2}", b.threshold)),
            f(b.and_then(|b| b.point.precision)),
            f(b.and_then(|b| b.point.recall)),
            f(b.and_then(|b| b.point.f1)),
            f(b.and_then(|b| Some(b.point.f1? - r.configured.f1?))),
            f(roc.and_then(|x| x.auc)),
            f(roc.and_then(|x| x.micro_auc)),
            f(ft(Some(&r.configured))),
            f(ff(Some(&r.configured))),
            f(ft(b.map(|b| &b.point))),
            f(ff(b.map(|b| &b.point))),
        ));
    }
    s.push_str(
        "\nmodel,threshold,tp,fp,fn,precision,recall,f1,f2,fp_per_image,frame_tpr,frame_fpr\n",
    );
    for r in &out.results {
        for p in &r.grid {
            s.push_str(&format!(
                "{},{:.2},{},{},{},{},{},{},{},{:.4},{},{}\n",
                csv_field(&r.model),
                p.threshold,
                p.counts.tp,
                p.counts.fp,
                p.counts.fn_,
                f(p.precision),
                f(p.recall),
                f(p.f1),
                f(p.f2),
                p.fp_per_image,
                f(ft(Some(p))),
                f(ff(Some(p))),
            ));
        }
    }
    s.push_str("\nmodel,class,positive_frames,negative_frames,roc_auc\n");
    for r in &out.results {
        for c in r.roc.iter().flat_map(|x| &x.per_class) {
            s.push_str(&format!(
                "{},{},{},{},{}\n",
                csv_field(&r.model),
                csv_field(&c.class),
                c.positives,
                c.negatives,
                f(c.auc)
            ));
        }
    }
    s
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Ground truth of a stored image (tests and tools).
pub fn image_gt<'a>(results: &'a BenchmarkResults, set: &str, file: &str) -> Option<&'a [GtBox]> {
    results
        .image_set(set)?
        .images
        .iter()
        .find(|i| i.file == file)?
        .objects
        .as_deref()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::benchmark::images::SetKind;

    fn gt(label: &str, b: [f32; 4], ignore: bool) -> GtBox {
        GtBox {
            label: label.into(),
            bbox: b,
            ignore,
        }
    }

    /// Results with two datasets (one real, night-tagged; one pseudo) and stored predictions of
    /// one model on two devices.
    pub(crate) fn fixture() -> (BenchmarkResults, StoredPreds) {
        let info = |file: &str, tags: &[&str], objects: Vec<GtBox>| ImageInfo {
            file: file.into(),
            width: 640,
            height: 480,
            tags: tags.iter().map(|t| t.to_string()).collect(),
            objects: Some(objects),
        };
        let set = |id: &str, gt: GroundTruth, images: Vec<ImageInfo>| ImageSetInfo {
            id: id.into(),
            title: id.into(),
            kind: SetKind::Builtin,
            dir: None,
            license: String::new(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: gt,
            scored_labels: None,
            images,
        };
        let mut results = BenchmarkResults::new(Default::default(), Default::default(), vec![]);
        results.image_sets = vec![
            set(
                "night-set",
                GroundTruth::Coco,
                vec![
                    info(
                        "a.jpg",
                        &["night"],
                        vec![
                            gt("person", [0.0, 0.0, 100.0, 100.0], false),
                            gt("car", [200.0, 200.0, 300.0, 300.0], false),
                            gt("person", [400.0, 0.0, 500.0, 100.0], true),
                        ],
                    ),
                    info(
                        "b.jpg",
                        &["day"],
                        vec![gt("person", [0.0, 0.0, 50.0, 50.0], false)],
                    ),
                ],
            ),
            set(
                "pseudo-set",
                GroundTruth::Pseudo("ref on openvino:cpu".into()),
                vec![info(
                    "c.jpg",
                    &["day"],
                    vec![gt("person", [0.0, 0.0, 100.0, 100.0], false)],
                )],
            ),
        ];
        let mut m = ModelResult::failed("m", "m.onnx", String::new());
        m.error = None;
        m.recommended = Some("ort:coreml".into());
        results.models.push(m);
        let p = |class: u32, confidence: f32, b: [f32; 4]| StoredPred {
            class,
            confidence,
            bbox: b,
        };
        let run = |device: &str, shift: f32| StoredRun {
            model: "m".into(),
            device: device.into(),
            timestamp: String::new(),
            classes: vec!["person".into(), "car".into()],
            configured: 0.5,
            eval_confidence: 0.05,
            images: vec![
                StoredImage {
                    set: "night-set".into(),
                    file: "a.jpg".into(),
                    preds: vec![
                        p(0, 0.9 - shift, [0.0, 0.0, 100.0, 100.0]), // TP person
                        p(1, 0.4, [200.0, 200.0, 300.0, 300.0]),     // TP car
                        p(0, 0.7, [400.0, 0.0, 500.0, 100.0]),       // ignore region
                        p(0, 0.3, [600.0, 300.0, 640.0, 400.0]),     // FP
                    ],
                },
                StoredImage {
                    set: "night-set".into(),
                    file: "b.jpg".into(),
                    preds: vec![p(0, 0.6, [0.0, 0.0, 50.0, 60.0])], // IoU 0.83: TP at 0.5
                },
                StoredImage {
                    set: "pseudo-set".into(),
                    file: "c.jpg".into(),
                    preds: vec![p(0, 0.8, [0.0, 0.0, 100.0, 100.0])],
                },
            ],
        };
        let preds = StoredPreds {
            version: "test".into(),
            runs: vec![run("openvino:cpu", 0.0), run("ort:coreml", 0.05)],
        };
        (results, preds)
    }

    #[test]
    fn search_hand_computed_and_filters() {
        let (results, preds) = fixture();
        let none = |_: &str| None;
        let out = search(&results, &preds, &SearchRequest::default(), &none).unwrap();
        assert!(out.errors.is_empty(), "{:?}", out.errors);
        let r = &out.results[0];
        // Recommended device; only the real-ground-truth set counts overall.
        assert_eq!(r.device, "ort:coreml");
        assert!(!r.relative);
        assert_eq!((r.images, r.gt), (2, 3));
        // At 0.5: person .85 TP, .7 ignored, b .6 TP; car .4 and FP .3 below -> 2 TP, 0 FP, 1 FN.
        let c = r.configured.counts;
        assert_eq!((c.tp, c.fp, c.fn_), (2, 0, 1));
        // F1 by kept set: down to .6: 2/(2+0+1) -> .8; +car .4: 1.0; +FP .3: 6/7.
        let b = r.best.as_ref().unwrap();
        assert_eq!(b.exact_threshold, Some(0.4));
        assert_eq!(b.plateau, Some([0.3, 0.4]));
        assert_eq!(b.threshold, 0.35);
        assert_eq!(b.point.f1, Some(1.0));
        assert_eq!(r.grid.len(), 19);
        assert!(!r.exact.is_empty());
        assert_eq!(r.by_dataset.len(), 2);
        assert!(r.by_dataset[1].relative);
        let classes: Vec<&str> = r.per_class.iter().map(|g| g.key.as_str()).collect();
        assert_eq!(classes, ["person", "car"]);

        // Tag filter: night only (a.jpg): person TP, car TP, FP.
        let req = SearchRequest {
            tags: vec!["night".into()],
            device: Some("openvino:cpu".into()),
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        assert_eq!((r.device.as_str(), r.images, r.gt), ("openvino:cpu", 1, 2));
        // Class subset: cars only.
        let req = SearchRequest {
            classes: vec!["Car".into()],
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        assert_eq!((r.gt, r.classes.clone()), (1, vec!["car".to_string()]));
        // Pseudo set only: relative.
        let req = SearchRequest {
            datasets: vec!["pseudo-set".into()],
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        assert!(r.relative && r.ground_truth.contains("ref on openvino:cpu"));
        // Stricter IoU: b.jpg (IoU 0.83) no longer matches at 0.9.
        let req = SearchRequest {
            iou: 0.9,
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        assert_eq!(r.configured.counts.fp, 1);
        // The config's current threshold is evaluated too.
        let cur = |_: &str| Some(0.3);
        let r = &search(&results, &preds, &SearchRequest::default(), &cur)
            .unwrap()
            .results[0];
        assert_eq!(r.current.unwrap().counts.fp, 1);

        // Errors: unknown model / device / class / nothing left; no stored predictions at all.
        for req in [
            SearchRequest {
                models: vec!["zzz".into()],
                ..Default::default()
            },
            SearchRequest {
                device: Some("ort:cpu".into()),
                ..Default::default()
            },
            SearchRequest {
                classes: vec!["dog".into()],
                ..Default::default()
            },
            SearchRequest {
                tags: vec!["fog".into()],
                ..Default::default()
            },
        ] {
            let out = search(&results, &preds, &req, &none).unwrap();
            assert!(out.results.is_empty() && out.errors.len() == 1, "{req:?}");
        }
        let err = search(
            &results,
            &StoredPreds::default(),
            &SearchRequest::default(),
            &none,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("run a benchmark first"));
        let csv = to_csv(&out_of(&results, &preds));
        assert!(csv.starts_with("model,device,objective,configured"));
        assert!(csv.contains("\nm,0.05,"), "{csv}");
    }

    /// Frame ROC and the frame objectives from the stored predictions alone.
    #[test]
    fn search_frame_roc_and_youden() {
        let (results, preds) = fixture();
        let none = |_: &str| None;
        let req = SearchRequest {
            objective: Objective::Youden,
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        // Frames (real set, coreml): a person+ .85 (max of .85, the .7 in the ignore region and
        // the .3 FP), a car+ .4, b person+ .6, b car- 0. Person has no negative frame:
        // undefined; car 1.0.
        let roc = r.roc.as_ref().unwrap();
        let person = roc.per_class.iter().find(|c| c.class == "person").unwrap();
        assert_eq!(
            (person.positives, person.negatives, person.auc),
            (2, 0, None)
        );
        assert_eq!(roc.class_auc("car"), Some(1.0));
        assert_eq!((roc.auc, roc.micro_auc), (Some(1.0), Some(1.0)));
        assert_eq!((roc.positives, roc.negatives), (3, 1));
        let b = r.best.as_ref().unwrap();
        assert_eq!((b.threshold, b.exact_threshold), (0.22, Some(0.4)));
        assert_eq!(b.plateau, Some([0.05, 0.4]));
        let f = b.point.frame.unwrap();
        assert_eq!((f.tpr, f.fpr), (Some(1.0), Some(0.0)));
        // At the configured 0.5 the car frame (.4) does not alert: TPR 2/3.
        let c = r.configured.frame.unwrap();
        assert_eq!((c.tp, c.positives), (2, 3));
        // Youden is among the alternatives of every search.
        assert!(r.alternatives.iter().any(|p| p.objective == "youden"));
        // Per class / per dataset AUC.
        let car = r.per_class.iter().find(|g| g.key == "car").unwrap();
        assert_eq!(car.roc_auc, Some(1.0));
        assert_eq!(r.by_dataset[0].roc_auc, Some(1.0));
        // Class filter: person only has no negative frames -> no Youden pick, no AUC.
        let req = SearchRequest {
            objective: Objective::Youden,
            classes: vec!["person".into()],
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        assert!(r.best.is_none());
        assert_eq!(r.roc.as_ref().unwrap().auc, None);
        // fpr:0 -> the highest TPR without false alerts: everything (the negative scores 0).
        let req = SearchRequest {
            objective: Objective::Fpr(0.0),
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        let f = r.best.as_ref().unwrap().point.frame.unwrap();
        assert_eq!((f.tpr, f.fpr), (Some(1.0), Some(0.0)));
        // CSV carries the AUC and the frame rates.
        let csv = to_csv(&out_of(&results, &preds));
        assert!(csv.contains(",roc_auc,roc_auc_micro,"), "{csv}");
        assert!(csv.contains("\nm,car,1,1,1.0000\n"), "{csv}");
    }

    fn out_of(results: &BenchmarkResults, preds: &StoredPreds) -> SearchOutput {
        search(results, preds, &SearchRequest::default(), &|_| None).unwrap()
    }

    #[test]
    fn form_parsing() {
        let f = |pairs: &[(&str, &str)]| -> Vec<(String, String)> {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        let r = SearchRequest::from_form(
            &f(&[
                ("model", "a"),
                ("model", "b"),
                ("device", ""),
                ("dataset", "x,y"),
                ("tag", "night"),
                ("objective", "recall:0.9"),
                ("iou", "0.6"),
            ]),
            Objective::F1,
        )
        .unwrap();
        assert_eq!(r.models, ["a", "b"]);
        assert_eq!(r.device, None);
        assert_eq!(r.datasets, ["x", "y"]);
        assert_eq!(r.objective, Objective::Recall(0.9));
        assert_eq!(r.iou, 0.6);
        let r = SearchRequest::from_form(&f(&[("model", "all")]), Objective::F2).unwrap();
        assert!(r.models.is_empty());
        assert_eq!(r.objective, Objective::F2);
        for bad in [
            ("iou", "1.2"),
            ("objective", "f9"),
            ("device", "vulkan"),
            ("iou", "x"),
        ] {
            assert!(
                SearchRequest::from_form(&f(&[bad]), Objective::F1).is_err(),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn stored_predictions_round_trip_and_merge() {
        let (results, preds) = fixture();
        let dir = std::env::temp_dir().join(format!("bop-preds-{}", uuid::Uuid::new_v4()));
        let path = dir.join(PREDS_FILE);
        assert!(StoredPreds::load(&path).unwrap().is_none());
        preds.save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("[0,0.9,0,0,100,100]"), "{text}");
        let back = StoredPreds::load(&path).unwrap().unwrap();
        assert_eq!(back, preds);
        // The search over the loaded file equals the one over the original.
        assert_eq!(
            out_of(&results, &back).results,
            out_of(&results, &preds).results
        );
        // Merge: a model no longer in the results is dropped; a re-run model is replaced.
        let mut other = preds.runs[0].clone();
        other.model = "gone".into();
        let prev = StoredPreds {
            version: "x".into(),
            runs: vec![preds.runs[0].clone(), other],
        };
        let merged = StoredPreds::merge(Some(prev.clone()), &results, &[]);
        assert_eq!(merged.runs.len(), 1);
        assert_eq!(merged.runs[0].model, "m");
        let rerun = results.models[0].clone();
        let merged = StoredPreds::merge(Some(prev), &results, &[rerun]);
        assert!(
            merged.runs.is_empty(),
            "re-run without stored runs replaces the old ones"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
