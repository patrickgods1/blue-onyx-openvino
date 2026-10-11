//! Frame-level (image-level) ROC: how well a model's per-frame confidence separates frames that
//! contain an object of a class from frames that do not, which is the decision Blue Iris makes
//! per frame (alert or not). Box-level ROC is undefined for detection (there are no true
//! negative boxes), so the samples are frames.
//!
//! For each scored class `c` (the [`super::metrics::ClassMap`] classes, so `vehicle` stands for
//! car/truck/bus ground truth and bird species for `bird`), every image is one binary sample:
//!
//! - **positive** when it has at least one non-ignored ground-truth box of `c`;
//! - **negative** when it has no ground truth of `c` at all;
//! - **excluded** when its only boxes of `c` are `ignore` regions (crowd, pseudo ground truth
//!   between 0.3 and 0.5), and for classes its dataset does not annotate (`scored_labels`).
//!
//! The sample's score is the highest confidence of a prediction of `c` in the image, from the
//! stored low-threshold pass ([`super::EVAL_CONFIDENCE`], 0.05); an image without a prediction
//! of `c` scores 0, below every threshold. Scores under 0.05 are therefore unknown (clipped to
//! 0), which ties those negatives and positives at the bottom and can only understate the AUC
//! slightly.
//!
//! The AUC is exact: the Mann-Whitney U statistic over the samples (a positive scoring above a
//! negative counts 1, a tie 0.5), divided by `positives * negatives` ([`auc`]); this equals the
//! trapezoidal area under the ROC curve through every distinct score. It is undefined (None)
//! without positives or without negatives. Reported per class, as the macro mean over the
//! classes with a defined AUC (the headline `roc_auc`) and micro (all class samples pooled).
//!
//! The alert decision at a confidence threshold `t` is "max confidence >= t": frame TPR (alert
//! rate on positive frames) and FPR (false-alert rate on negative frames) at `t` come from the
//! same samples ([`FrameSteps`]), pooled over classes for a single threshold.

use serde::{Deserialize, Serialize};

use super::metrics::EvalImage;

/// Points of a stored (decimated) ROC curve.
pub const ROC_POINTS: usize = 100;

/// One frame sample of one class.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSample {
    /// Scored class.
    pub class: String,
    pub positive: bool,
    /// Highest confidence of a prediction of the class in the frame (0 = none).
    pub score: f32,
}

/// The frame samples of one image (labels in the scored label space) for `classes`, the
/// classes annotated in its dataset. Classes whose only boxes are ignore regions are excluded.
pub fn frame_samples(img: &EvalImage, classes: &[String]) -> Vec<FrameSample> {
    classes
        .iter()
        .filter_map(|c| {
            let positive = img.gt.iter().any(|g| !g.ignore && g.label == *c);
            let any = img.gt.iter().any(|g| g.label == *c);
            if any && !positive {
                return None;
            }
            let score = img
                .preds
                .iter()
                .filter(|p| p.label == *c)
                .map(|p| p.confidence)
                .fold(0.0f32, f32::max);
            Some(FrameSample {
                class: c.clone(),
                positive,
                score,
            })
        })
        .collect()
}

/// Exact ROC AUC of `(score, positive)` samples by the Mann-Whitney U statistic: the share of
/// (positive, negative) pairs where the positive scores higher, ties counting one half. None
/// without positives or without negatives.
pub fn auc(samples: &[(f32, bool)]) -> Option<f64> {
    let pos = samples.iter().filter(|s| s.1).count();
    let neg = samples.len() - pos;
    if pos == 0 || neg == 0 {
        return None;
    }
    let mut s: Vec<(f32, bool)> = samples.to_vec();
    s.sort_by(|a, b| a.0.total_cmp(&b.0));
    // U = sum over positives of (negatives below + half the negatives tied with it), computed
    // per group of equal scores; twice U is an integer, so the sum is exact in f64.
    let mut u2: u128 = 0;
    let mut neg_below: u128 = 0;
    let mut i = 0;
    while i < s.len() {
        let mut j = i;
        let (mut p, mut n) = (0u128, 0u128);
        while j < s.len() && s[j].0 == s[i].0 {
            if s[j].1 {
                p += 1;
            } else {
                n += 1;
            }
            j += 1;
        }
        u2 += p * (2 * neg_below + n);
        neg_below += n;
        i = j;
    }
    Some(u2 as f64 / (2.0 * pos as f64 * neg as f64))
}

/// Frame-level counts and rates at one threshold (frames alert when their score >= it).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameRates {
    /// Positive frames that alert.
    pub tp: usize,
    /// Negative frames that alert (false alerts).
    pub fp: usize,
    pub positives: usize,
    pub negatives: usize,
    /// Alert rate on positive frames (None without positives).
    pub tpr: Option<f64>,
    /// False-alert rate on negative frames (None without negatives).
    pub fpr: Option<f64>,
}

impl FrameRates {
    pub fn new(tp: usize, fp: usize, positives: usize, negatives: usize) -> Self {
        Self {
            tp,
            fp,
            positives,
            negatives,
            tpr: (positives > 0).then(|| tp as f64 / positives as f64),
            fpr: (negatives > 0).then(|| fp as f64 / negatives as f64),
        }
    }

    /// Youden's J, TPR - FPR (None when either is undefined).
    pub fn youden(&self) -> Option<f64> {
        Some(self.tpr? - self.fpr?)
    }
}

/// Cumulative frame counts at every distinct score (descending): TPR and FPR only change there.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FrameSteps {
    pub positives: usize,
    pub negatives: usize,
    /// `(score, TP, FP)` with every frame at or above `score` alerting, by descending score
    /// (one entry per distinct score, frames without a prediction at score 0).
    pub steps: Vec<(f32, usize, usize)>,
}

impl FrameSteps {
    /// None without samples.
    pub fn new<'a>(samples: impl IntoIterator<Item = &'a FrameSample>) -> Option<Self> {
        let mut v: Vec<(f32, bool)> = samples.into_iter().map(|s| (s.score, s.positive)).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        let positives = v.iter().filter(|s| s.1).count();
        let mut steps = Vec::new();
        let (mut tp, mut fp) = (0, 0);
        for (i, &(c, p)) in v.iter().enumerate() {
            if p {
                tp += 1;
            } else {
                fp += 1;
            }
            if v.get(i + 1).is_none_or(|n| n.0 != c) {
                steps.push((c, tp, fp));
            }
        }
        Some(Self {
            positives,
            negatives: v.len() - positives,
            steps,
        })
    }

    /// Rates when frames scoring at or above `t` alert.
    pub fn rates_at(&self, t: f32) -> FrameRates {
        let n = self.steps.partition_point(|s| s.0 >= t);
        let (tp, fp) = match n {
            0 => (0, 0),
            n => (self.steps[n - 1].1, self.steps[n - 1].2),
        };
        FrameRates::new(tp, fp, self.positives, self.negatives)
    }

    /// The ROC curve: one point per distinct score, by descending threshold (rising FPR / TPR),
    /// ending at (1, 1); (0, 0) is implied. Thinned to at most `max` points keeping the ends.
    pub fn curve(&self, max: usize) -> Vec<RocPoint> {
        if self.positives == 0 || self.negatives == 0 {
            return Vec::new();
        }
        let pts: Vec<RocPoint> = self
            .steps
            .iter()
            .map(|&(t, tp, fp)| RocPoint {
                threshold: t,
                tpr: tp as f64 / self.positives as f64,
                fpr: fp as f64 / self.negatives as f64,
            })
            .collect();
        thin(pts, max)
    }

    /// The trapezoidal area under the full curve (equals [`auc`]).
    pub fn trapezoid_auc(&self) -> Option<f64> {
        if self.positives == 0 || self.negatives == 0 {
            return None;
        }
        let (mut area, mut x, mut y) = (0.0, 0.0, 0.0);
        for &(_, tp, fp) in &self.steps {
            let nx = fp as f64 / self.negatives as f64;
            let ny = tp as f64 / self.positives as f64;
            area += (nx - x) * (y + ny) / 2.0;
            x = nx;
            y = ny;
        }
        Some(area)
    }
}

/// Keep at most `max` points of `pts`, evenly by index, keeping the first and the last.
fn thin<T: Copy>(pts: Vec<T>, max: usize) -> Vec<T> {
    if pts.len() <= max || max < 2 {
        return pts;
    }
    let last = pts.len() - 1;
    (0..max)
        .map(|i| pts[(i * last + (max - 1) / 2) / (max - 1)])
        .collect()
}

/// One point of a ROC curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RocPoint {
    /// Confidence threshold (frames scoring at or above it alert).
    #[serde(rename = "t")]
    pub threshold: f32,
    pub tpr: f64,
    pub fpr: f64,
}

/// Frame ROC of one class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassRoc {
    pub class: String,
    /// Frames with the class.
    pub positives: usize,
    /// Frames without it.
    pub negatives: usize,
    pub auc: Option<f64>,
}

/// Frame-level ROC of a set of images.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FrameRoc {
    /// Macro mean of the per-class AUCs that are defined (the headline `roc_auc`).
    pub auc: Option<f64>,
    /// AUC over every class's samples pooled.
    pub micro_auc: Option<f64>,
    /// Pooled sample counts (a frame counts once per class).
    pub positives: usize,
    pub negatives: usize,
    pub per_class: Vec<ClassRoc>,
    /// Pooled ROC curve, thinned (absent in summaries that do not chart it).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub curve: Vec<RocPoint>,
}

impl FrameRoc {
    /// The ROC of `samples` over `classes` (in that order), with a pooled curve of at most
    /// `curve_points` points (0 = none).
    pub fn of<'a>(
        samples: impl IntoIterator<Item = &'a FrameSample>,
        classes: &[String],
        curve_points: usize,
    ) -> Self {
        let samples: Vec<&FrameSample> = samples.into_iter().collect();
        let per_class: Vec<ClassRoc> = classes
            .iter()
            .map(|c| {
                let v: Vec<(f32, bool)> = samples
                    .iter()
                    .filter(|s| s.class == *c)
                    .map(|s| (s.score, s.positive))
                    .collect();
                let positives = v.iter().filter(|s| s.1).count();
                ClassRoc {
                    class: c.clone(),
                    positives,
                    negatives: v.len() - positives,
                    auc: auc(&v),
                }
            })
            .filter(|c| c.positives + c.negatives > 0)
            .collect();
        let defined: Vec<f64> = per_class.iter().filter_map(|c| c.auc).collect();
        let pooled: Vec<(f32, bool)> = samples
            .iter()
            .filter(|s| classes.contains(&s.class))
            .map(|s| (s.score, s.positive))
            .collect();
        let positives = pooled.iter().filter(|s| s.1).count();
        let curve = if curve_points > 0 {
            FrameSteps::new(
                samples
                    .iter()
                    .copied()
                    .filter(|s| classes.contains(&s.class)),
            )
            .map(|f| f.curve(curve_points))
            .unwrap_or_default()
        } else {
            Vec::new()
        };
        Self {
            auc: (!defined.is_empty()).then(|| defined.iter().sum::<f64>() / defined.len() as f64),
            micro_auc: auc(&pooled),
            positives,
            negatives: pooled.len() - positives,
            per_class,
            curve,
        }
    }

    /// The AUC of `class`.
    pub fn class_auc(&self, class: &str) -> Option<f64> {
        self.per_class
            .iter()
            .find(|c| c.class == class)
            .and_then(|c| c.auc)
    }
}

#[cfg(test)]
mod tests {
    use super::super::metrics::{GtBox, PredBox};
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut seed = seed;
        move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        }
    }

    /// Brute force over every (positive, negative) pair.
    fn brute(s: &[(f32, bool)]) -> Option<f64> {
        let (mut num, mut den) = (0.0, 0.0);
        for a in s.iter().filter(|x| x.1) {
            for b in s.iter().filter(|x| !x.1) {
                den += 1.0;
                num += if a.0 > b.0 {
                    1.0
                } else if a.0 == b.0 {
                    0.5
                } else {
                    0.0
                };
            }
        }
        (den > 0.0).then(|| num / den)
    }

    #[test]
    fn mann_whitney_equals_brute_force_and_trapezoid_with_ties() {
        for seed in 1..60u64 {
            let mut r = rng(seed);
            let n = 5 + (r() * 80.0) as usize;
            // Coarse scores (many ties), some zeros (no prediction).
            let s: Vec<(f32, bool)> = (0..n)
                .map(|_| {
                    let pos = r() < 0.4;
                    let raw = if r() < 0.2 {
                        0.0
                    } else {
                        ((r() + if pos { 0.3 } else { 0.0 }).min(1.0) * 12.0).round() / 12.0
                    };
                    (raw, pos)
                })
                .collect();
            let a = auc(&s);
            let b = brute(&s);
            assert_eq!(a.is_some(), b.is_some(), "seed {seed}");
            if let (Some(a), Some(b)) = (a, b) {
                assert!((a - b).abs() < 1e-12, "seed {seed}: {a} vs {b}");
                let samples: Vec<FrameSample> = s
                    .iter()
                    .map(|&(score, positive)| FrameSample {
                        class: "person".into(),
                        positive,
                        score,
                    })
                    .collect();
                let fs = FrameSteps::new(&samples).unwrap();
                assert!(
                    (fs.trapezoid_auc().unwrap() - a).abs() < 1e-12,
                    "seed {seed}"
                );
                let c = fs.curve(1000);
                assert_eq!((c.last().unwrap().tpr, c.last().unwrap().fpr), (1.0, 1.0));
            }
        }
    }

    #[test]
    fn separability_extremes_and_undefined() {
        let perfect = [(0.9, true), (0.8, true), (0.3, false), (0.0, false)];
        assert_eq!(auc(&perfect), Some(1.0));
        let inverse = [(0.1, true), (0.0, true), (0.7, false), (0.6, false)];
        assert_eq!(auc(&inverse), Some(0.0));
        // All tied: 0.5.
        assert_eq!(auc(&[(0.5, true), (0.5, false), (0.5, false)]), Some(0.5));
        // Random scores independent of the label: ~0.5.
        let mut r = rng(99);
        let s: Vec<(f32, bool)> = (0..4000).map(|_| (r(), r() < 0.5)).collect();
        assert!((auc(&s).unwrap() - 0.5).abs() < 0.03);
        // No positives / no negatives / nothing.
        assert_eq!(auc(&[(0.5, false), (0.2, false)]), None);
        assert_eq!(auc(&[(0.5, true)]), None);
        assert_eq!(auc(&[]), None);
        // 2 positives, 2 negatives, one tie: (1 + 1 + 0.5 + 1) / 4.
        let s = [(0.9, true), (0.5, true), (0.5, false), (0.2, false)];
        assert!((auc(&s).unwrap() - 0.875).abs() < 1e-12);
    }

    fn gt(label: &str, ignore: bool) -> GtBox {
        GtBox {
            label: label.into(),
            bbox: [0.0, 0.0, 10.0, 10.0],
            ignore,
        }
    }

    fn pred(label: &str, c: f32) -> PredBox {
        PredBox {
            label: label.into(),
            confidence: c,
            bbox: [0.0, 0.0, 10.0, 10.0],
        }
    }

    #[test]
    fn frame_samples_positive_negative_excluded() {
        let classes: Vec<String> = vec!["person".into(), "vehicle".into(), "dog".into()];
        let img = EvalImage {
            gt: vec![gt("person", false), gt("person", true), gt("vehicle", true)],
            preds: vec![
                pred("person", 0.3),
                pred("person", 0.8),
                pred("vehicle", 0.6),
            ],
        };
        let s = frame_samples(&img, &classes);
        // vehicle: only an ignore region -> excluded; dog: negative without prediction -> 0.
        assert_eq!(
            s,
            [
                FrameSample {
                    class: "person".into(),
                    positive: true,
                    score: 0.8
                },
                FrameSample {
                    class: "dog".into(),
                    positive: false,
                    score: 0.0
                },
            ]
        );
        // A class not annotated in the dataset is not passed in: no sample.
        assert_eq!(frame_samples(&img, &["person".to_string()]).len(), 1);
    }

    #[test]
    fn frame_roc_macro_micro_and_rates() {
        let mk = |class: &str, positive: bool, score: f32| FrameSample {
            class: class.into(),
            positive,
            score,
        };
        let samples = vec![
            mk("person", true, 0.9),
            mk("person", false, 0.4),
            mk("person", true, 0.6),
            mk("person", false, 0.0),
            mk("vehicle", true, 0.3),
            mk("vehicle", false, 0.5),
            // No negatives for dog: undefined, left out of the macro mean.
            mk("dog", true, 0.7),
        ];
        let classes: Vec<String> = vec!["person".into(), "vehicle".into(), "dog".into()];
        let r = FrameRoc::of(&samples, &classes, 50);
        assert_eq!(r.class_auc("person"), Some(1.0));
        assert_eq!(r.class_auc("vehicle"), Some(0.0));
        assert_eq!(r.class_auc("dog"), None);
        assert_eq!(r.auc, Some(0.5));
        let pooled: Vec<(f32, bool)> = samples.iter().map(|s| (s.score, s.positive)).collect();
        assert_eq!(r.micro_auc, auc(&pooled));
        assert_eq!((r.positives, r.negatives), (4, 3));
        assert!(!r.curve.is_empty());
        let fs = FrameSteps::new(&samples).unwrap();
        // At 0.5: positives .9 .6 .7 alert (3 of 4), negatives .5 (1 of 3).
        let at = fs.rates_at(0.5);
        assert_eq!((at.tp, at.fp), (3, 1));
        assert!((at.youden().unwrap() - (0.75 - 1.0 / 3.0)).abs() < 1e-12);
        // Above everything nothing alerts; 0.05 leaves out the score-0 negative.
        assert_eq!(fs.rates_at(0.95).fpr, Some(0.0));
        assert_eq!(fs.rates_at(0.05).fp, 2);
        // Thinning keeps the ends.
        let c = fs.curve(3);
        assert_eq!(c.len(), 3);
        assert_eq!((c[2].tpr, c[2].fpr), (1.0, 1.0));
    }
}
