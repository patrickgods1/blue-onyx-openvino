//! Detection accuracy: class mapping between ground truth and a model's labels, COCO-style
//! matching, AP@0.5 and AP@[0.5:0.95] (101-point interpolation), precision / recall / F1 at a
//! confidence threshold, and recall by object size. Pure functions over boxes; no runtime.
//!
//! Class mapping. Every label goes through [`canonical_label`] (lowercase; VOC/YOLO aliases such
//! as `motorbike`, `aeroplane`, `tvmonitor` become the COCO spelling). Accuracy is scored on the
//! CCTV classes ([`CCTV_CLASSES`]) the model can output, plus the class groups of
//! [`CLASS_GROUPS`]: a model with a `vehicle` class (IPcam-general) is scored on ground-truth
//! car/truck/bus boxes relabelled `vehicle`. Ground truth of classes the model cannot output is
//! dropped (not a miss), and so are predictions of classes outside the scored set.
//!
//! Matching follows pycocotools: detections in descending confidence order take the unmatched
//! ground-truth box of the same class with the highest IoU at or above the threshold; a
//! detection that only overlaps an `ignore` region (crowd, "other" class) is neither a true nor
//! a false positive; ignored ground truth is never a miss. AP is the mean, over the 101 recall
//! points 0, 0.01, .., 1, of the interpolated (monotone envelope) precision; classes without
//! ground truth are left out of the mean.
//!
//! Threshold sweep ([`SweepImage`], [`threshold_curve`]): TP / FP / FN, precision, recall, F1,
//! F2 and false positives per image at every confidence threshold of [`SWEEP_THRESHOLDS`],
//! from one matching of the low-threshold predictions. This is exact, not an approximation:
//! greedy matching visits predictions by descending confidence, so whether a prediction is a
//! true positive (and which object it takes) depends only on the predictions at least as
//! confident as it is, all of which survive any threshold it survives. Matching the
//! predictions at or above `t` therefore gives the same outcome for each of them as matching
//! everything and keeping those at or above `t`. The same holds for the families' post-processing
//! (see `Bench::time`): greedy NMS keeps or drops a box based only on more confident boxes, and
//! the DETR top-Q selection keeps a prefix of the confidence order.

use super::roc::{FrameRates, FrameSample, FrameSteps, frame_samples};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Classes the accuracy score is computed on (COCO spelling), when the model has them.
pub const CCTV_CLASSES: [&str; 10] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "cat",
    "bird",
    "horse",
];

/// Model labels that stand for several ground-truth classes.
pub const CLASS_GROUPS: [(&str, &[&str]); 1] = [("vehicle", &["car", "truck", "bus"])];

/// Fine-grained model labels that stand for one scored class: a label any of whose words
/// (split on spaces and hyphens) is listed counts as that class. ipcam-bird labels bird
/// species ("Blue Jay", "Eastern Screech-Owl", ...); "Purple Squirrel" matches nothing.
pub const SPECIES: [(&str, &[&str]); 1] = [(
    "bird",
    &[
        "bird",
        "blackbird",
        "bluebird",
        "bunting",
        "cardinal",
        "chickadee",
        "crow",
        "dove",
        "duck",
        "eagle",
        "falcon",
        "finch",
        "flicker",
        "goldfinch",
        "goose",
        "grackle",
        "gull",
        "hawk",
        "heron",
        "hummingbird",
        "jay",
        "junco",
        "kestrel",
        "magpie",
        "mockingbird",
        "nuthatch",
        "oriole",
        "owl",
        "parrot",
        "pigeon",
        "robin",
        "sparrow",
        "starling",
        "swallow",
        "tanager",
        "thrush",
        "towhee",
        "warbler",
        "woodpecker",
        "wren",
    ],
)];

/// The scored class a fine-grained label belongs to ([`SPECIES`]), if any.
pub fn species_class(label: &str) -> Option<&'static str> {
    let l = canonical_label(label);
    let words: Vec<&str> = l.split([' ', '-']).filter(|w| !w.is_empty()).collect();
    SPECIES
        .iter()
        .find(|(_, names)| words.iter().any(|w| names.contains(w)))
        .map(|(class, _)| *class)
}

/// IoU thresholds of AP@[0.5:0.95].
pub const IOU_THRESHOLDS: [f32; 10] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95];
/// Detections kept per image and class for AP (COCO `maxDets`).
pub const MAX_DETS: usize = 100;
/// COCO object-size buckets (box area in original-image pixels): small < 32^2 <= medium < 96^2.
pub const SMALL_AREA: f32 = 32.0 * 32.0;
pub const MEDIUM_AREA: f32 = 96.0 * 96.0;

/// Lowercase, trimmed, with VOC/YOLO spellings mapped to COCO's.
pub fn canonical_label(label: &str) -> String {
    let l = label.trim().to_ascii_lowercase().replace('_', " ");
    match l.as_str() {
        "motorbike" => "motorcycle",
        "aeroplane" => "airplane",
        "tvmonitor" | "tv monitor" => "tv",
        "sofa" => "couch",
        "diningtable" => "dining table",
        "pottedplant" => "potted plant",
        "pedestrian" | "people" => "person",
        "bike" => "bicycle",
        _ => return l,
    }
    .to_string()
}

/// A box `[x_min, y_min, x_max, y_max]` in original-image pixels.
pub type BBox = [f32; 4];

pub fn area(b: &BBox) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

pub fn iou(a: &BBox, b: &BBox) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let union = area(a) + area(b) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// A ground-truth object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GtBox {
    pub label: String,
    pub bbox: BBox,
    /// Region that is neither required nor penalized (crowd, "other").
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ignore: bool,
}

/// A prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredBox {
    pub label: String,
    pub confidence: f32,
    pub bbox: BBox,
}

/// COCO size bucket of a box.
pub fn size_bucket(b: &BBox) -> &'static str {
    let a = area(b);
    if a < SMALL_AREA {
        "small"
    } else if a < MEDIUM_AREA {
        "medium"
    } else {
        "large"
    }
}

/// How a model's labels relate to the scored ground-truth classes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ClassMap {
    /// Scored classes in the model's label space ("person", "vehicle", ...).
    pub classes: Vec<String>,
    /// Canonical ground-truth label -> scored class.
    gt: HashMap<String, String>,
    /// Canonical model label -> scored class.
    pred: HashMap<String, String>,
}

impl ClassMap {
    /// The scored classes of a model with these labels (see the module docs).
    pub fn for_model(model_labels: &[String]) -> Self {
        let labels: Vec<String> = model_labels.iter().map(|l| canonical_label(l)).collect();
        let has = |c: &str| labels.iter().any(|l| l == c);
        let mut m = Self::default();
        for c in CCTV_CLASSES {
            if has(c) {
                m.classes.push(c.to_string());
                m.gt.insert(c.to_string(), c.to_string());
                m.pred.insert(c.to_string(), c.to_string());
            }
        }
        for l in &labels {
            if m.pred.contains_key(l) {
                continue;
            }
            if let Some(class) = species_class(l) {
                if !m.classes.iter().any(|c| c == class) {
                    m.classes.push(class.to_string());
                    m.gt.insert(class.to_string(), class.to_string());
                }
                m.pred.insert(l.clone(), class.to_string());
            }
        }
        for (group, members) in CLASS_GROUPS {
            if has(group) {
                m.classes.push(group.to_string());
                m.pred.insert(group.to_string(), group.to_string());
                for member in members {
                    m.gt.entry(member.to_string())
                        .or_insert_with(|| group.to_string());
                }
            }
        }
        m
    }

    /// Every label of the model (canonical), plus the [`CLASS_GROUPS`] it has: for AP over all
    /// classes, not just the CCTV ones.
    pub fn all_classes(model_labels: &[String]) -> Self {
        let mut m = Self::for_model(model_labels);
        for l in model_labels {
            let c = canonical_label(l);
            if c.is_empty() || c == "unknown" || m.pred.contains_key(&c) {
                continue;
            }
            m.classes.push(c.clone());
            m.gt.entry(c.clone()).or_insert_with(|| c.clone());
            m.pred.insert(c.clone(), c);
        }
        m
    }

    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// The scored classes of ground-truth labels `labels` (a set's `scored_labels`).
    pub fn restrict(&self, labels: &[String]) -> Vec<String> {
        let mut v: Vec<String> = labels
            .iter()
            .filter_map(|l| self.gt_class(l).map(str::to_string))
            .collect();
        v.sort();
        v.dedup();
        v
    }

    /// Scored class of a ground-truth label (None = not scored).
    pub fn gt_class(&self, label: &str) -> Option<&str> {
        self.gt.get(&canonical_label(label)).map(String::as_str)
    }

    /// Scored class of a model label (None = not scored).
    pub fn pred_class(&self, label: &str) -> Option<&str> {
        self.pred.get(&canonical_label(label)).map(String::as_str)
    }

    /// Ground truth in the scored label space (unscored boxes dropped).
    pub fn map_gt(&self, gt: &[GtBox]) -> Vec<GtBox> {
        gt.iter()
            .filter_map(|g| {
                self.gt_class(&g.label).map(|c| GtBox {
                    label: c.to_string(),
                    ..g.clone()
                })
            })
            .collect()
    }

    /// Predictions in the scored label space (unscored ones dropped).
    pub fn map_preds(&self, preds: &[PredBox]) -> Vec<PredBox> {
        preds
            .iter()
            .filter_map(|p| {
                self.pred_class(&p.label).map(|c| PredBox {
                    label: c.to_string(),
                    ..p.clone()
                })
            })
            .collect()
    }
}

/// Outcome of matching one image's predictions against its ground truth.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImageMatch {
    /// Per prediction (input order): Some(true) TP, Some(false) FP, None ignored.
    pub pred: Vec<Option<bool>>,
    /// Per ground-truth box (input order): matched. Ignored boxes are always false.
    pub gt_matched: Vec<bool>,
    /// Per prediction (input order): index of the ground-truth box it matched (true positives).
    pub pred_gt: Vec<Option<usize>>,
}

/// Greedy COCO matching of `preds` (any order; matched by descending confidence) against `gt`,
/// same class, IoU >= `iou_thr`. Labels must already be in one label space.
pub fn match_image(preds: &[PredBox], gt: &[GtBox], iou_thr: f32) -> ImageMatch {
    let mut order: Vec<usize> = (0..preds.len()).collect();
    order.sort_by(|&a, &b| preds[b].confidence.total_cmp(&preds[a].confidence));
    let mut out = ImageMatch {
        pred: vec![Some(false); preds.len()],
        gt_matched: vec![false; gt.len()],
        pred_gt: vec![None; preds.len()],
    };
    for i in order {
        let p = &preds[i];
        let mut best: Option<(usize, f32)> = None;
        // Unmatched, non-ignored boxes first; an ignored region only if nothing else matches.
        for (j, g) in gt.iter().enumerate() {
            if g.ignore || out.gt_matched[j] || g.label != p.label {
                continue;
            }
            let v = iou(&p.bbox, &g.bbox);
            if v >= iou_thr && best.is_none_or(|(_, b)| v > b) {
                best = Some((j, v));
            }
        }
        match best {
            Some((j, _)) => {
                out.gt_matched[j] = true;
                out.pred[i] = Some(true);
                out.pred_gt[i] = Some(j);
            }
            None => {
                let in_ignored = gt
                    .iter()
                    .any(|g| g.ignore && g.label == p.label && iou(&p.bbox, &g.bbox) >= iou_thr);
                if in_ignored {
                    out.pred[i] = None;
                }
            }
        }
    }
    out
}

/// AP of one class: 101-point interpolated precision. `scored` holds (confidence, is TP) of
/// every non-ignored detection of the class over all images; `npos` the non-ignored ground
/// truth. None when `npos` is 0.
pub fn average_precision(scored: &mut [(f32, bool)], npos: usize) -> Option<f64> {
    if npos == 0 {
        return None;
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut recall = Vec::with_capacity(scored.len());
    let mut precision = Vec::with_capacity(scored.len());
    for &(_, is_tp) in scored.iter() {
        if is_tp {
            tp += 1;
        } else {
            fp += 1;
        }
        recall.push(tp as f64 / npos as f64);
        precision.push(tp as f64 / (tp + fp) as f64);
    }
    // Monotone envelope from the right.
    for i in (0..precision.len().saturating_sub(1)).rev() {
        if precision[i + 1] > precision[i] {
            precision[i] = precision[i + 1];
        }
    }
    let mut sum = 0.0;
    for k in 0..=100 {
        let r = k as f64 / 100.0;
        // First index with recall >= r (recall is non-decreasing).
        let idx = recall.partition_point(|&x| x < r - 1e-12);
        if idx < precision.len() {
            sum += precision[idx];
        }
    }
    Some(sum / 101.0)
}

/// One image's inputs to [`evaluate`], already in the scored label space.
#[derive(Debug, Clone, Default)]
pub struct EvalImage {
    pub gt: Vec<GtBox>,
    /// Predictions at the low evaluation threshold.
    pub preds: Vec<PredBox>,
}

/// AP and counts of one class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub class: String,
    /// Non-ignored ground-truth objects.
    pub gt: usize,
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
}

/// Recall at the confidence threshold for one object-size bucket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SizeRecall {
    pub bucket: String,
    pub gt: usize,
    pub recall: Option<f64>,
}

/// Counts and rates at the confidence threshold, IoU 0.5.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Counts {
    pub tp: usize,
    pub fp: usize,
    #[serde(rename = "fn")]
    pub fn_: usize,
}

impl Counts {
    pub fn add(&mut self, o: Counts) {
        self.tp += o.tp;
        self.fp += o.fp;
        self.fn_ += o.fn_;
    }

    /// None without detections.
    pub fn precision(&self) -> Option<f64> {
        let d = self.tp + self.fp;
        (d > 0).then(|| self.tp as f64 / d as f64)
    }

    /// None without ground truth.
    pub fn recall(&self) -> Option<f64> {
        let g = self.tp + self.fn_;
        (g > 0).then(|| self.tp as f64 / g as f64)
    }

    /// Harmonic mean of precision and recall; 0 when either is 0, None when either is unknown.
    pub fn f1(&self) -> Option<f64> {
        let (p, r) = (self.precision()?, self.recall()?);
        Some(if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        })
    }
}

/// Summary metrics over a set of images.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Summary {
    pub images: usize,
    pub counts: Counts,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    /// Mean over classes with ground truth.
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    pub per_class: Vec<ClassMetrics>,
    pub by_size: Vec<SizeRecall>,
}

/// Counts of one image at `threshold` (IoU 0.5), and its match (for drawing).
pub fn image_counts(img: &EvalImage, threshold: f32) -> (Counts, Vec<PredBox>, ImageMatch) {
    let kept: Vec<PredBox> = img
        .preds
        .iter()
        .filter(|p| p.confidence >= threshold)
        .cloned()
        .collect();
    let m = match_image(&kept, &img.gt, 0.5);
    let tp = m.pred.iter().filter(|x| **x == Some(true)).count();
    let fp = m.pred.iter().filter(|x| **x == Some(false)).count();
    let npos = img.gt.iter().filter(|g| !g.ignore).count();
    (
        Counts {
            tp,
            fp,
            fn_: npos - tp,
        },
        kept,
        m,
    )
}

/// Keep the `MAX_DETS` most confident predictions per class.
fn cap(preds: &[PredBox]) -> Vec<PredBox> {
    let mut by: BTreeMap<&str, Vec<&PredBox>> = BTreeMap::new();
    for p in preds {
        by.entry(p.label.as_str()).or_default().push(p);
    }
    let mut out = Vec::new();
    for (_, mut v) in by {
        v.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        out.extend(v.into_iter().take(MAX_DETS).cloned());
    }
    out
}

/// Metrics over `images` for `classes`, with P/R/F1 at `threshold`.
pub fn evaluate(images: &[EvalImage], classes: &[String], threshold: f32) -> Summary {
    let capped: Vec<Vec<PredBox>> = images.iter().map(|i| cap(&i.preds)).collect();
    let mut per_class = Vec::new();
    let mut ap50s = Vec::new();
    let mut ap_all = Vec::new();
    // Per-class counts at the threshold.
    let mut class_counts: HashMap<&str, Counts> = HashMap::new();
    let mut counts = Counts::default();
    let mut size: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for img in images {
        let (c, kept, m) = image_counts(img, threshold);
        counts.add(c);
        for (j, g) in img.gt.iter().enumerate() {
            if g.ignore {
                continue;
            }
            let e = size.entry(size_bucket(&g.bbox)).or_default();
            e.0 += 1;
            e.1 += usize::from(m.gt_matched[j]);
            let cc = class_counts.entry(g.label.as_str()).or_default();
            if m.gt_matched[j] {
                cc.tp += 1;
            } else {
                cc.fn_ += 1;
            }
        }
        for (p, r) in kept.iter().zip(&m.pred) {
            if *r == Some(false)
                && let Some(c) = classes.iter().find(|c| **c == p.label)
            {
                class_counts.entry(c.as_str()).or_default().fp += 1;
            }
        }
    }
    for class in classes {
        let npos: usize = images
            .iter()
            .map(|i| {
                i.gt.iter()
                    .filter(|g| !g.ignore && g.label == *class)
                    .count()
            })
            .sum();
        let mut aps = Vec::new();
        for &t in &IOU_THRESHOLDS {
            let mut scored = Vec::new();
            for (img, preds) in images.iter().zip(&capped) {
                let preds: Vec<PredBox> = preds
                    .iter()
                    .filter(|p| p.label == *class)
                    .cloned()
                    .collect();
                let gt: Vec<GtBox> = img
                    .gt
                    .iter()
                    .filter(|g| g.label == *class)
                    .cloned()
                    .collect();
                let m = match_image(&preds, &gt, t);
                for (p, r) in preds.iter().zip(m.pred) {
                    if let Some(tp) = r {
                        scored.push((p.confidence, tp));
                    }
                }
            }
            aps.push(average_precision(&mut scored, npos));
        }
        let ap50 = aps[0];
        let ap50_95 = ap50.map(|_| aps.iter().flatten().sum::<f64>() / IOU_THRESHOLDS.len() as f64);
        if let Some(a) = ap50 {
            ap50s.push(a);
        }
        if let Some(a) = ap50_95 {
            ap_all.push(a);
        }
        let cc = class_counts
            .get(class.as_str())
            .copied()
            .unwrap_or_default();
        per_class.push(ClassMetrics {
            class: class.clone(),
            gt: npos,
            ap50,
            ap50_95,
            precision: cc.precision(),
            recall: cc.recall(),
        });
    }
    let mean = |v: &[f64]| (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64);
    Summary {
        images: images.len(),
        counts,
        precision: counts.precision(),
        recall: counts.recall(),
        f1: counts.f1(),
        ap50: mean(&ap50s),
        ap50_95: mean(&ap_all),
        per_class,
        by_size: ["small", "medium", "large"]
            .iter()
            .filter_map(|b| {
                size.get(b).map(|&(gt, hit)| SizeRecall {
                    bucket: b.to_string(),
                    gt,
                    recall: (gt > 0).then(|| hit as f64 / gt as f64),
                })
            })
            .collect(),
    }
}

/// Confidence thresholds of the threshold sweep: 0.05 to 0.95 by 0.05.
pub const SWEEP_THRESHOLDS: [f32; 19] = [
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
    0.9, 0.95,
];

impl Counts {
    /// F-beta from the counts, `(1 + b^2) TP / ((1 + b^2) TP + b^2 FN + FP)` (beta 1 = F1,
    /// beta 2 weighs recall four times as much as precision). None without ground truth. Unlike
    /// [`Counts::f1`], 0 (not None) when nothing is detected: every object was missed.
    pub fn f_beta(&self, beta: f64) -> Option<f64> {
        if self.tp + self.fn_ == 0 {
            return None;
        }
        let b2 = beta * beta;
        let tp = (1.0 + b2) * self.tp as f64;
        Some(tp / (tp + b2 * self.fn_ as f64 + self.fp as f64))
    }
}

/// Counts and rates at one confidence threshold (IoU 0.5).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThresholdPoint {
    pub threshold: f32,
    #[serde(flatten)]
    pub counts: Counts,
    /// None without detections.
    pub precision: Option<f64>,
    /// None without ground truth.
    pub recall: Option<f64>,
    /// [`Counts::f_beta`] with beta 1 and 2.
    pub f1: Option<f64>,
    pub f2: Option<f64>,
    /// False positives per image (0 without images).
    pub fp_per_image: f64,
    /// Frame-level alert rates at this threshold (see [`super::roc`]); None when the curve has
    /// no frame samples (small-object scope, results of older versions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<FrameRates>,
}

impl ThresholdPoint {
    pub fn new(threshold: f32, counts: Counts, images: usize) -> Self {
        Self {
            threshold,
            counts,
            precision: counts.precision(),
            recall: counts.recall(),
            f1: counts.f_beta(1.0),
            f2: counts.f_beta(2.0),
            fp_per_image: if images == 0 {
                0.0
            } else {
                counts.fp as f64 / images as f64
            },
            frame: None,
        }
    }

    /// With frame-level rates.
    pub fn with_frame(mut self, frame: Option<FrameRates>) -> Self {
        self.frame = frame;
        self
    }
}

/// A scored prediction after matching, for the threshold sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepPred {
    pub confidence: f32,
    /// Scored class.
    pub class: String,
    pub tp: bool,
    /// Counts in the small-object scope: a true positive of a small object, or a small false
    /// positive.
    pub small: bool,
}

/// One image matched once (at IoU 0.5) for the threshold sweep: its non-ignored ground truth
/// and its scored (non-ignored) predictions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SweepImage {
    /// Non-ignored ground truth: (class, small).
    pub gt: Vec<(String, bool)>,
    pub preds: Vec<SweepPred>,
    /// Frame samples of the classes annotated in the image ([`super::roc`]); empty unless added
    /// with [`SweepImage::with_frames`].
    pub frames: Vec<FrameSample>,
}

impl SweepImage {
    /// Match `img` (predictions at the low evaluation threshold) once; see the module docs for
    /// why every higher threshold follows exactly.
    pub fn of(img: &EvalImage) -> Self {
        Self::with_iou(img, 0.5)
    }

    /// [`SweepImage::of`] with matches at IoU >= `iou`.
    pub fn with_iou(img: &EvalImage, iou: f32) -> Self {
        let m = match_image(&img.preds, &img.gt, iou);
        let is_small = |b: &BBox| area(b) < SMALL_AREA;
        let preds = img
            .preds
            .iter()
            .zip(m.pred.iter().zip(&m.pred_gt))
            .filter_map(|(p, (r, j))| {
                let tp = (*r)?;
                let small = match j {
                    Some(j) => is_small(&img.gt[*j].bbox),
                    None => is_small(&p.bbox),
                };
                Some(SweepPred {
                    confidence: p.confidence,
                    class: p.label.clone(),
                    tp,
                    small,
                })
            })
            .collect();
        Self {
            gt: img
                .gt
                .iter()
                .filter(|g| !g.ignore)
                .map(|g| (g.label.clone(), is_small(&g.bbox)))
                .collect(),
            preds,
            frames: Vec::new(),
        }
    }

    /// With the frame samples of `img` for `classes` (the scored classes its dataset
    /// annotates).
    pub fn with_frames(mut self, img: &EvalImage, classes: &[String]) -> Self {
        self.frames = frame_samples(img, classes);
        self
    }
}

/// Which objects and predictions a curve counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepScope<'a> {
    /// Every scored class.
    All,
    /// One scored class.
    Class(&'a str),
    /// Small objects (< 32x32 px): recall over small ground truth; a true positive of a larger
    /// object is not counted, and a false positive only when its box is small (COCO's area
    /// range rule).
    Small,
}

impl SweepScope<'_> {
    fn gt(&self, class: &str, small: bool) -> bool {
        match self {
            SweepScope::All => true,
            SweepScope::Class(c) => *c == class,
            SweepScope::Small => small,
        }
    }

    fn pred(&self, p: &SweepPred) -> bool {
        self.gt(&p.class, p.small)
    }

    /// Frame samples in scope: every class, or one; none for small objects (a frame-level
    /// alert has no object size).
    fn frame(&self, f: &FrameSample) -> bool {
        match self {
            SweepScope::All => true,
            SweepScope::Class(c) => *c == f.class,
            SweepScope::Small => false,
        }
    }
}

/// Counts over `images` at `threshold` (predictions with confidence >= `threshold`).
pub fn sweep_counts(images: &[&SweepImage], scope: SweepScope, threshold: f32) -> Counts {
    let mut c = Counts::default();
    for img in images {
        let npos = img.gt.iter().filter(|(l, s)| scope.gt(l, *s)).count();
        let mut tp = 0;
        for p in img
            .preds
            .iter()
            .filter(|p| p.confidence >= threshold && scope.pred(p))
        {
            if p.tp {
                tp += 1;
            } else {
                c.fp += 1;
            }
        }
        c.tp += tp;
        c.fn_ += npos - tp;
    }
    c
}

/// The curve over `thresholds`.
pub fn threshold_curve(
    images: &[&SweepImage],
    scope: SweepScope,
    thresholds: &[f32],
) -> Vec<ThresholdPoint> {
    thresholds
        .iter()
        .map(|&t| ThresholdPoint::new(t, sweep_counts(images, scope, t), images.len()))
        .collect()
}

/// Cumulative counts at every distinct prediction confidence of a scope: precision, recall
/// and F only change at those values, so this is the exact curve. Built with one sort and one
/// pass with running TP / FP counts (FN = scored ground truth - TP; predictions in ignore
/// regions are not in [`SweepImage::preds`]).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Breakpoints {
    /// Scored (non-ignored) ground truth in the scope.
    pub npos: usize,
    pub images: usize,
    /// `(confidence, TP, FP)` with every prediction at or above `confidence`, by descending
    /// confidence (one entry per distinct confidence).
    pub steps: Vec<(f32, usize, usize)>,
    /// Frame-level steps of the scope's frame samples, pooled over classes (None without
    /// samples).
    pub frames: Option<FrameSteps>,
}

impl Breakpoints {
    pub fn new(images: &[&SweepImage], scope: SweepScope) -> Self {
        let mut preds: Vec<(f32, bool)> = Vec::new();
        let mut npos = 0;
        for img in images {
            npos += img.gt.iter().filter(|(l, s)| scope.gt(l, *s)).count();
            preds.extend(
                img.preds
                    .iter()
                    .filter(|p| scope.pred(p))
                    .map(|p| (p.confidence, p.tp)),
            );
        }
        preds.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut steps: Vec<(f32, usize, usize)> = Vec::new();
        let (mut tp, mut fp) = (0, 0);
        for (i, &(c, is_tp)) in preds.iter().enumerate() {
            if is_tp {
                tp += 1;
            } else {
                fp += 1;
            }
            // Close the step after the last prediction of this confidence.
            if preds.get(i + 1).is_none_or(|n| n.0 != c) {
                steps.push((c, tp, fp));
            }
        }
        let frames = FrameSteps::new(
            images
                .iter()
                .flat_map(|i| i.frames.iter())
                .filter(|f| scope.frame(f)),
        );
        Self {
            npos,
            images: images.len(),
            steps,
            frames,
        }
    }

    /// Frame rates at `t` (None without frame samples).
    pub fn frame_at(&self, t: f32) -> Option<FrameRates> {
        self.frames.as_ref().map(|f| f.rates_at(t))
    }

    /// Counts with the predictions at or above `t` (binary search over the steps).
    pub fn counts_at(&self, t: f32) -> Counts {
        let n = self.steps.partition_point(|s| s.0 >= t);
        let (tp, fp) = match n {
            0 => (0, 0),
            n => (self.steps[n - 1].1, self.steps[n - 1].2),
        };
        Counts {
            tp,
            fp,
            fn_: self.npos - tp,
        }
    }

    pub fn point_at(&self, t: f32) -> ThresholdPoint {
        ThresholdPoint::new(t, self.counts_at(t), self.images).with_frame(self.frame_at(t))
    }

    /// The exact curve (one point per distinct confidence, ascending threshold) thinned to at
    /// most `max` points, keeping the first and the last (for charts).
    pub fn decimated(&self, max: usize) -> Vec<ThresholdPoint> {
        let pts: Vec<ThresholdPoint> = self
            .steps
            .iter()
            .rev()
            .map(|&(c, tp, fp)| {
                ThresholdPoint::new(
                    c,
                    Counts {
                        tp,
                        fp,
                        fn_: self.npos - tp,
                    },
                    self.images,
                )
            })
            .collect();
        if pts.len() <= max || max < 2 {
            return pts;
        }
        let last = pts.len() - 1;
        (0..max)
            .map(|i| pts[(i * last + (max - 1) / 2) / (max - 1)])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gt(label: &str, b: BBox) -> GtBox {
        GtBox {
            label: label.into(),
            bbox: b,
            ignore: false,
        }
    }

    fn p(label: &str, c: f32, b: BBox) -> PredBox {
        PredBox {
            label: label.into(),
            confidence: c,
            bbox: b,
        }
    }

    #[test]
    fn canonical_labels_and_class_map() {
        assert_eq!(canonical_label(" Motorbike "), "motorcycle");
        assert_eq!(canonical_label("tvmonitor"), "tv");
        assert_eq!(canonical_label("aeroplane"), "airplane");
        assert_eq!(canonical_label("Car"), "car");
        // COCO-80 as this repo spells it (VOC style) maps onto the CCTV classes.
        let coco = ClassMap::for_model(&crate::model::classes::coco80());
        assert_eq!(coco.classes.len(), CCTV_CLASSES.len());
        assert_eq!(coco.gt_class("motorbike"), Some("motorcycle"));
        assert_eq!(coco.pred_class("motorcycle"), Some("motorcycle"));
        assert_eq!(coco.gt_class("umbrella"), None);
        // IPcam-general: person + vehicle (car/truck/bus).
        let ipcam = ClassMap::for_model(&["person".into(), "vehicle".into(), "unknown".into()]);
        assert_eq!(ipcam.classes, ["person", "vehicle"]);
        assert_eq!(ipcam.gt_class("truck"), Some("vehicle"));
        assert_eq!(ipcam.gt_class("motorcycle"), None);
        assert_eq!(ipcam.pred_class("unknown"), None);
        let mapped = ipcam.map_gt(&[gt("bus", [0.0; 4]), gt("dog", [0.0; 4])]);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].label, "vehicle");
        // A model with exact classes and a group keeps the exact ones.
        let both = ClassMap::for_model(&["car".into(), "vehicle".into()]);
        assert_eq!(both.gt_class("car"), Some("car"));
        assert_eq!(both.gt_class("bus"), Some("vehicle"));
        // No scored classes (package model).
        assert!(ClassMap::for_model(&["package".into()]).is_empty());
        // ipcam-bird: species labels count as bird; a squirrel does not.
        let birds = ClassMap::for_model(&[
            "Blue Jay".into(),
            "Eastern Screech-Owl".into(),
            "House Finch Female".into(),
            "Purple Squirrel".into(),
        ]);
        assert_eq!(birds.classes, ["bird"]);
        assert_eq!(birds.pred_class("Blue Jay"), Some("bird"));
        assert_eq!(birds.pred_class("Eastern Screech-Owl"), Some("bird"));
        assert_eq!(birds.pred_class("Purple Squirrel"), None);
        assert_eq!(birds.gt_class("bird"), Some("bird"));
    }

    #[test]
    fn iou_and_buckets() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(iou(&a, &a), 1.0);
        assert!((iou(&a, &[5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(size_bucket(&[0.0, 0.0, 31.0, 31.0]), "small");
        assert_eq!(size_bucket(&[0.0, 0.0, 32.0, 32.0]), "medium");
        assert_eq!(size_bucket(&[0.0, 0.0, 96.0, 96.0]), "large");
    }

    #[test]
    fn greedy_matching_by_confidence() {
        let g = vec![gt("car", [0.0, 0.0, 100.0, 100.0])];
        // Two detections on one object: the more confident one is the TP even though the other
        // overlaps better.
        let preds = vec![
            p("car", 0.6, [0.0, 0.0, 100.0, 100.0]),
            p("car", 0.9, [10.0, 0.0, 100.0, 100.0]),
        ];
        let m = match_image(&preds, &g, 0.5);
        assert_eq!(m.pred, [Some(false), Some(true)]);
        assert_eq!(m.gt_matched, [true]);
        // A detection takes the best-IoU unmatched box.
        let g2 = vec![
            gt("car", [0.0, 0.0, 100.0, 100.0]),
            gt("car", [20.0, 0.0, 120.0, 100.0]),
        ];
        let m = match_image(&[p("car", 0.9, [18.0, 0.0, 118.0, 100.0])], &g2, 0.5);
        assert_eq!(m.gt_matched, [false, true]);
        // Wrong class never matches.
        let m = match_image(&[p("dog", 0.9, [0.0, 0.0, 100.0, 100.0])], &g, 0.5);
        assert_eq!(m.pred, [Some(false)]);
        // Inside an ignored region: neither TP nor FP; the region is no miss.
        let ig = vec![GtBox {
            ignore: true,
            ..gt("person", [0.0, 0.0, 100.0, 100.0])
        }];
        let m = match_image(&[p("person", 0.9, [5.0, 5.0, 95.0, 95.0])], &ig, 0.5);
        assert_eq!(m.pred, [None]);
        let (c, _, _) = image_counts(
            &EvalImage {
                gt: ig,
                preds: vec![],
            },
            0.5,
        );
        assert_eq!(c, Counts::default());
    }

    #[test]
    fn ap_known_values() {
        // Perfect ranking: AP 1.
        let mut s = vec![(0.9, true), (0.8, true)];
        assert!((average_precision(&mut s, 2).unwrap() - 1.0).abs() < 1e-12);
        // TP, FP, TP with 2 GT: recall 0.5 at precision 1, recall 1 at precision 2/3.
        // 51 points (r = 0..0.5) at 1, 50 points (0.51..1) at 2/3.
        let mut s = vec![(0.9, true), (0.8, false), (0.7, true)];
        let want = (51.0 + 50.0 * 2.0 / 3.0) / 101.0;
        assert!((average_precision(&mut s, 2).unwrap() - want).abs() < 1e-12);
        // FP first: the envelope lifts precision at low recall to the later 2/3 and 1/2.
        // Order: FP (p 0), TP (r .5, p .5), TP (r 1, p 2/3) -> envelope 2/3 everywhere.
        let mut s = vec![(0.9, false), (0.8, true), (0.7, true)];
        assert!((average_precision(&mut s, 2).unwrap() - 2.0 / 3.0).abs() < 1e-12);
        // Half the objects never found: recall points above 0.5 score 0.
        let mut s = vec![(0.9, true)];
        assert!((average_precision(&mut s, 2).unwrap() - 51.0 / 101.0).abs() < 1e-12);
        assert_eq!(average_precision(&mut [], 0), None);
        assert_eq!(average_precision(&mut [], 3), Some(0.0));
    }

    #[test]
    fn evaluate_counts_ap_and_sizes() {
        let images = vec![
            EvalImage {
                gt: vec![
                    gt("person", [0.0, 0.0, 100.0, 200.0]),
                    gt("car", [300.0, 300.0, 320.0, 320.0]), // small, missed
                ],
                preds: vec![
                    p("person", 0.9, [2.0, 2.0, 100.0, 200.0]),
                    p("person", 0.1, [500.0, 0.0, 600.0, 100.0]), // low-confidence FP
                ],
            },
            EvalImage {
                gt: vec![gt("car", [0.0, 0.0, 200.0, 100.0])],
                preds: vec![
                    p("car", 0.8, [0.0, 0.0, 200.0, 100.0]),
                    p("car", 0.7, [400.0, 0.0, 500.0, 100.0]), // FP
                ],
            },
        ];
        let classes: Vec<String> = vec!["person".into(), "car".into()];
        let s = evaluate(&images, &classes, 0.5);
        assert_eq!(s.images, 2);
        assert_eq!((s.counts.tp, s.counts.fp, s.counts.fn_), (2, 1, 1));
        assert!((s.precision.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert!((s.recall.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert!((s.f1.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        // person: TP then FP -> AP 1. car: TP (r .5), FP -> AP 51/101.
        let person = &s.per_class[0];
        assert!((person.ap50.unwrap() - 1.0).abs() < 1e-12);
        let car = &s.per_class[1];
        assert_eq!(car.gt, 2);
        assert!((car.ap50.unwrap() - 51.0 / 101.0).abs() < 1e-12);
        assert!((s.ap50.unwrap() - (1.0 + 51.0 / 101.0) / 2.0).abs() < 1e-12);
        // AP@[.5:.95] <= AP@.5; the person box (IoU 0.97) matches at every threshold.
        assert!(s.ap50_95.unwrap() <= s.ap50.unwrap());
        assert!((person.ap50_95.unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(car.recall, Some(0.5));
        assert_eq!(car.precision, Some(0.5));
        let small = s.by_size.iter().find(|b| b.bucket == "small").unwrap();
        assert_eq!((small.gt, small.recall), (1, Some(0.0)));
        let large = s.by_size.iter().find(|b| b.bucket == "large").unwrap();
        assert_eq!((large.gt, large.recall), (2, Some(1.0)));
        // A class without ground truth is left out of the mean.
        let s = evaluate(&images, &["person".into(), "dog".into()], 0.5);
        assert_eq!(s.per_class[1].ap50, None);
        assert!((s.ap50.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn counts_rates() {
        let c = Counts {
            tp: 0,
            fp: 0,
            fn_: 3,
        };
        assert_eq!((c.precision(), c.recall(), c.f1()), (None, Some(0.0), None));
        let c = Counts {
            tp: 0,
            fp: 2,
            fn_: 3,
        };
        assert_eq!(c.f1(), Some(0.0));
    }

    #[test]
    fn threshold_curve_hand_computed() {
        let img = EvalImage {
            gt: vec![
                gt("person", [0.0, 0.0, 100.0, 200.0]),     // large
                gt("person", [300.0, 300.0, 320.0, 320.0]), // small
                GtBox {
                    ignore: true,
                    ..gt("person", [500.0, 0.0, 600.0, 100.0])
                },
            ],
            preds: vec![
                p("person", 0.9, [0.0, 0.0, 100.0, 200.0]),   // TP (large)
                p("person", 0.7, [505.0, 5.0, 600.0, 100.0]), // in the ignore region
                p("person", 0.6, [700.0, 0.0, 800.0, 100.0]), // FP, large box
                p("person", 0.3, [300.0, 300.0, 320.0, 320.0]), // TP (small)
                p("person", 0.2, [2.0, 0.0, 100.0, 200.0]),   // duplicate: FP, large box
            ],
        };
        let s = SweepImage::of(&img);
        assert_eq!(s.gt.len(), 2);
        assert_eq!(s.preds.len(), 4, "the ignored prediction is dropped");
        let imgs = [&s];
        let at = |scope, t| sweep_counts(&imgs, scope, t);
        let c = |tp, fp, fn_| Counts { tp, fp, fn_ };
        assert_eq!(at(SweepScope::All, 0.1), c(2, 2, 0));
        assert_eq!(at(SweepScope::All, 0.25), c(2, 1, 0));
        assert_eq!(at(SweepScope::All, 0.5), c(1, 1, 1));
        assert_eq!(at(SweepScope::All, 0.95), c(0, 0, 2));
        // Small objects: only the small object and small false positives (none) count.
        assert_eq!(at(SweepScope::Small, 0.1), c(1, 0, 0));
        assert_eq!(at(SweepScope::Small, 0.5), c(0, 0, 1));
        assert_eq!(at(SweepScope::Class("car"), 0.1), c(0, 0, 0));

        let curve = threshold_curve(&imgs, SweepScope::All, &SWEEP_THRESHOLDS);
        assert_eq!(curve.len(), 19);
        let pt = |t: f32| *curve.iter().find(|p| p.threshold == t).unwrap();
        let p25 = pt(0.25);
        assert!((p25.precision.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(p25.recall, Some(1.0));
        assert!((p25.f1.unwrap() - 0.8).abs() < 1e-12);
        // F2 = 5 TP / (5 TP + 4 FN + FP) = 10 / 11.
        assert!((p25.f2.unwrap() - 10.0 / 11.0).abs() < 1e-12);
        assert_eq!(p25.fp_per_image, 1.0);
        let p50 = pt(0.5);
        assert_eq!((p50.precision, p50.recall), (Some(0.5), Some(0.5)));
        assert!((p50.f2.unwrap() - 0.5).abs() < 1e-12);
        // Nothing detected: precision unknown, F1 0 (every object missed).
        let p95 = pt(0.95);
        assert_eq!(
            (p95.precision, p95.recall, p95.f1),
            (None, Some(0.0), Some(0.0))
        );
        // No ground truth: no recall, no F.
        let car = threshold_curve(&imgs, SweepScope::Class("car"), &[0.5]);
        assert_eq!((car[0].recall, car[0].f1), (None, None));
    }

    /// The sweep from one matching equals matching at each threshold (`image_counts`, what the
    /// P/R at the configured threshold uses), on pseudo-random scenes with duplicates,
    /// overlapping objects, ignore regions and several classes.
    #[test]
    fn threshold_curve_equals_rematching_at_each_threshold() {
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        let classes = ["person", "car"];
        let mut images = Vec::new();
        for _ in 0..60 {
            let mut img = EvalImage::default();
            for _ in 0..(rnd() * 6.0) as usize {
                let (x, y, w) = (rnd() * 300.0, rnd() * 300.0, 10.0 + rnd() * 80.0);
                img.gt.push(GtBox {
                    label: classes[(rnd() * 2.0) as usize % 2].into(),
                    bbox: [x, y, x + w, y + w],
                    ignore: rnd() < 0.15,
                });
            }
            for _ in 0..(rnd() * 10.0) as usize {
                let (x, y, w) = match img.gt.get((rnd() * 8.0) as usize) {
                    // Near an object (sometimes a duplicate), else anywhere.
                    Some(g) => (
                        g.bbox[0] + rnd() * 12.0 - 6.0,
                        g.bbox[1] + rnd() * 12.0 - 6.0,
                        g.bbox[2] - g.bbox[0] + rnd() * 10.0 - 5.0,
                    ),
                    None => (rnd() * 300.0, rnd() * 300.0, 10.0 + rnd() * 80.0),
                };
                // Coarse confidences so ties on the grid happen.
                let conf = ((0.05 + rnd() * 0.95) * 40.0).round() / 40.0;
                img.preds.push(p(
                    classes[(rnd() * 2.0) as usize % 2],
                    conf.max(0.05),
                    [x, y, x + w, y + w],
                ));
            }
            images.push(img);
        }
        let sweep: Vec<SweepImage> = images.iter().map(SweepImage::of).collect();
        let refs: Vec<&SweepImage> = sweep.iter().collect();
        let mut nonzero = 0;
        for t in SWEEP_THRESHOLDS.into_iter().chain([0.42, 0.5, 0.125]) {
            let mut want = Counts::default();
            for img in &images {
                want.add(image_counts(img, t).0);
            }
            assert_eq!(sweep_counts(&refs, SweepScope::All, t), want, "t = {t}");
            nonzero += usize::from(want.tp > 0 && want.fp > 0);
            // Per class: the per-class P/R of `evaluate` at the same threshold.
            let s = evaluate(&images, &["person".into(), "car".into()], t);
            for cm in &s.per_class {
                let c = sweep_counts(&refs, SweepScope::Class(&cm.class), t);
                assert_eq!(
                    (c.precision(), c.recall()),
                    (cm.precision, cm.recall),
                    "t = {t}"
                );
            }
        }
        assert!(nonzero > 5, "the scenes exercise true and false positives");
    }

    /// The breakpoint sweep equals brute-force counting at every breakpoint and in between,
    /// for every scope.
    #[test]
    fn breakpoints_match_brute_force() {
        let mut seed: u64 = 7;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        let mut imgs = Vec::new();
        for _ in 0..40 {
            let mut img = SweepImage::default();
            for _ in 0..(rnd() * 5.0) as usize {
                img.gt.push((
                    ["person", "car"][(rnd() * 2.0) as usize % 2].into(),
                    rnd() < 0.3,
                ));
            }
            for _ in 0..(rnd() * 8.0) as usize {
                img.preds.push(SweepPred {
                    // Coarse values so equal confidences occur.
                    confidence: 0.05 + ((rnd() * 0.95) * 60.0).floor() / 60.0,
                    class: ["person", "car"][(rnd() * 2.0) as usize % 2].into(),
                    tp: rnd() < 0.5,
                    small: rnd() < 0.3,
                });
            }
            // Keep TP <= objects per class, as matching guarantees.
            for class in ["person", "car"] {
                for small in [false, true] {
                    let objects = img
                        .gt
                        .iter()
                        .filter(|g| g.0 == class && (!small || g.1))
                        .count();
                    let mut tps = 0;
                    for p in img
                        .preds
                        .iter_mut()
                        .filter(|p| p.class == class && (!small || p.small))
                    {
                        if p.tp {
                            tps += 1;
                            if tps > objects {
                                p.tp = false;
                            }
                        }
                    }
                }
            }
            imgs.push(img);
        }
        let refs: Vec<&SweepImage> = imgs.iter().collect();
        for scope in [SweepScope::All, SweepScope::Class("car"), SweepScope::Small] {
            let bp = Breakpoints::new(&refs, scope);
            assert!(bp.steps.len() > 20);
            assert!(bp.steps.windows(2).all(|w| w[0].0 > w[1].0));
            for &(c, _, _) in &bp.steps {
                assert_eq!(
                    bp.counts_at(c),
                    sweep_counts(&refs, scope, c),
                    "{scope:?} {c}"
                );
            }
            for t in [0.0, 0.051, 0.33, 0.5, 0.999, 1.5] {
                assert_eq!(
                    bp.counts_at(t),
                    sweep_counts(&refs, scope, t),
                    "{scope:?} {t}"
                );
            }
            let d = bp.decimated(10);
            assert_eq!(d.len(), 10);
            assert_eq!(d[0].threshold, bp.steps.last().unwrap().0);
            assert_eq!(d[9].threshold, bp.steps[0].0);
            assert!(d.windows(2).all(|w| w[0].threshold < w[1].threshold));
        }
    }
}
