//! Confidence-threshold sweep: the best confidence threshold of a model for an objective.
//!
//! Every device run stores, next to its AP, the precision / recall / F1 / F2 / false positives
//! per image curve over [`SWEEP_THRESHOLDS`] ([`ThresholdSweep`]: overall, per dataset, per
//! tag group and per class), computed from the predictions the run already made at the
//! evaluation threshold (no extra inference; exact, see [`super::metrics`]). [`advise`] picks
//! the best threshold of a model from its recommended device's curves for the configured
//! [`Objective`] (config `benchmark.threshold_objective`):
//!
//! - `f1` (default): the highest F1, precision and recall balanced;
//! - `f2`: the highest F2, recall weighted 4x (fewer missed people, more false alerts);
//! - `precision:<p>`: the highest recall with precision >= p (e.g. `precision:0.9` to limit false
//!   alerts). When no threshold reaches p, the most precise threshold is picked, with a note;
//! - `recall:<r>`: the highest precision with recall >= r (e.g. `recall:0.9`: miss at most 10%).
//!   When no threshold reaches r, the threshold with the highest recall is picked, with a note.
//!
//! The pick is exact, not limited to the 0.05 grid ([`exact_pick`]): precision, recall and F
//! only change at the predictions' confidence values, so one pass over the predictions sorted
//! by confidence ([`Breakpoints`]) evaluates the objective at every distinct confidence. Among
//! the thresholds within [`PLATEAU`] of the best score, the widest contiguous range wins and
//! its midpoint is taken (robust to noise: a lone spike between two drops is not chosen over a
//! broad flat top). The threshold is then rounded down to 0.01 (keeping a superset of the
//! predictions; when that leaves the range, the range's top rounded down is used), at least
//! [`MIN_THRESHOLD`] (the evaluation pass's floor), and P/R/F are re-evaluated at the rounded
//! value. The 0.05 grid ([`SWEEP_THRESHOLDS`]) is only used for the curve tables and charts.
//!
//! Ties go to the higher threshold (fewer detections for the same score). The server's
//! threshold is what the model reports; Blue Iris can send its own `min_confidence` per request
//! (it then replaces the server threshold for that request) and filters the returned objects
//! again with each camera's minimum confidence.

use super::metrics::{Breakpoints, SWEEP_THRESHOLDS, ThresholdPoint};
use super::report::DeviceResult;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Scores this close count as equal (the higher threshold wins).
const TIE: f64 = 1e-12;
/// Thresholds scoring within this of the best form the plateau whose midpoint is picked.
pub const PLATEAU: f64 = 0.002;
/// Lowest threshold a pick can have: the floor of the evaluation pass.
pub const MIN_THRESHOLD: f32 = super::EVAL_CONFIDENCE;

/// What the best threshold optimizes.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Objective {
    /// Highest F1.
    #[default]
    F1,
    /// Highest F2 (recall-weighted).
    F2,
    /// Highest recall with precision at or above the target (0..1].
    Precision(f64),
    /// Highest precision with recall at or above the target (0..1].
    Recall(f64),
}

impl Objective {
    /// The objectives every report shows for comparison.
    pub const STANDARD: [Objective; 3] = [Objective::F1, Objective::F2, Objective::Precision(0.9)];

    /// The value maximized at `p`: F1, F2, or recall when the precision target is met.
    pub fn score(&self, p: &ThresholdPoint) -> Option<f64> {
        match self {
            Objective::F1 => p.f1,
            Objective::F2 => p.f2,
            Objective::Precision(target) => {
                p.precision.filter(|&v| v >= target - TIE).and(p.recall)
            }
            Objective::Recall(target) => p.recall.filter(|&v| v >= target - TIE).and(p.precision),
        }
    }

    /// Human-readable: "F1", "F2 (favor recall)", "recall at precision >= 90%".
    pub fn describe(&self) -> String {
        match self {
            Objective::F1 => "F1".into(),
            Objective::F2 => "F2 (favor recall)".into(),
            Objective::Precision(p) => format!("recall at precision \u{2265} {}%", pct0(*p)),
            Objective::Recall(r) => format!("precision at recall \u{2265} {}%", pct0(*r)),
        }
    }
}

fn pct0(v: f64) -> String {
    let s = format!("{:.1}", v * 100.0);
    s.strip_suffix(".0").map(str::to_string).unwrap_or(s)
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Objective::F1 => f.write_str("f1"),
            Objective::F2 => f.write_str("f2"),
            Objective::Precision(p) => write!(f, "precision:{p}"),
            Objective::Recall(r) => write!(f, "recall:{r}"),
        }
    }
}

impl FromStr for Objective {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let t = s.trim().to_ascii_lowercase();
        match t.as_str() {
            "f1" | "" => Ok(Objective::F1),
            "f2" => Ok(Objective::F2),
            _ => {
                let (kind, p) = if let Some(p) = t
                    .strip_prefix("precision:")
                    .or_else(|| t.strip_prefix("precision="))
                {
                    ("precision", p)
                } else if let Some(p) = t
                    .strip_prefix("recall:")
                    .or_else(|| t.strip_prefix("recall="))
                {
                    ("recall", p)
                } else {
                    return Err(format!(
                        "threshold objective '{s}': use f1, f2, precision:<0..1> or recall:<0..1>"
                    ));
                };
                let v: f64 = p.trim().parse().map_err(|_| {
                    format!("threshold objective '{s}': '{p}' is not a number (e.g. {kind}:0.9)")
                })?;
                if !(v > 0.0 && v <= 1.0) {
                    return Err(format!(
                        "threshold objective '{s}': the {kind} target must be in (0, 1]"
                    ));
                }
                Ok(if kind == "precision" {
                    Objective::Precision(v)
                } else {
                    Objective::Recall(v)
                })
            }
        }
    }
}

impl TryFrom<String> for Objective {
    type Error = String;
    fn try_from(s: String) -> Result<Self, String> {
        s.parse()
    }
}

impl From<Objective> for String {
    fn from(o: Objective) -> String {
        o.to_string()
    }
}

/// The best threshold of a curve for an objective.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pick {
    /// Objective (`f1`, `f2`, `precision:0.9`).
    pub objective: String,
    pub threshold: f32,
    pub point: ThresholdPoint,
    /// False when a precision target is reached at no threshold (the most precise one was
    /// picked instead).
    pub met: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Exact optimum before the plateau and the rounding: the confidence value it sits at
    /// (None for a pick from the 0.05 grid only, e.g. results of older versions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_threshold: Option<f32>,
    /// Objective score of the exact optimum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_score: Option<f64>,
    /// Threshold range `[lo, hi]` within [`PLATEAU`] of the best score the pick was taken from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plateau: Option<[f32; 2]>,
}

/// Round down to a multiple of 0.01 (a hair of tolerance so 0.35 stays 0.35 in f32).
pub fn floor_cent(t: f64) -> f32 {
    ((t * 100.0 + 1e-4).floor() / 100.0) as f32
}

/// A threshold range over which the kept predictions do not change: thresholds in `(lo, hi]`
/// (`[lo, hi]` for the lowest) keep the predictions at or above `hi`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub lo: f32,
    pub hi: f32,
    /// Objective score (None = undefined or a precision target missed).
    pub score: Option<f64>,
}

/// Index of the best-scoring segment (ties: the higher one). `segments` ascend.
pub fn best_segment(segments: &[Segment]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, sg) in segments.iter().enumerate() {
        if let Some(v) = sg.score
            && best.is_none_or(|(_, b)| v >= b - TIE)
        {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i)
}

/// The widest contiguous run `(first, last)` of segments scoring at least `best - tol`
/// (ties: the higher run). `segments` ascend and are contiguous.
pub fn plateau(segments: &[Segment], tol: f64) -> Option<(usize, usize)> {
    let best = segments[best_segment(segments)?].score?;
    let ok = |s: &Segment| s.score.is_some_and(|v| v >= best - tol - TIE);
    let mut out: Option<(usize, usize, f32)> = None;
    let mut i = 0;
    while i < segments.len() {
        if !ok(&segments[i]) {
            i += 1;
            continue;
        }
        let a = i;
        while i + 1 < segments.len() && ok(&segments[i + 1]) {
            i += 1;
        }
        let width = segments[i].hi - segments[a].lo;
        if out.is_none_or(|(_, _, w)| width >= w - 1e-6) {
            out = Some((a, i, width));
        }
        i += 1;
    }
    out.map(|(a, b, _)| (a, b))
}

/// The reported threshold of the run `(a, b)`: its midpoint rounded down to 0.01; when that
/// falls out of the run, the run's top rounded down; at least [`MIN_THRESHOLD`].
fn round_in(segments: &[Segment], a: usize, b: usize) -> f32 {
    let (lo, hi) = (segments[a].lo, segments[b].hi);
    let r = floor_cent((lo as f64 + hi as f64) / 2.0);
    // The lowest segment includes its lower end.
    let inside = r > lo || (a == 0 && r >= lo);
    let r = if inside { r } else { floor_cent(hi as f64) };
    r.max(MIN_THRESHOLD)
}

/// Segments of `bp` over `[MIN_THRESHOLD, 1]`, ascending, scored for `objective`.
pub fn segments(bp: &Breakpoints, objective: Objective) -> Vec<Segment> {
    let mut cs: Vec<f32> = bp
        .steps
        .iter()
        .rev()
        .map(|s| s.0)
        .filter(|c| (MIN_THRESHOLD..=1.0).contains(c))
        .collect();
    // Above the most confident prediction nothing is kept.
    if cs.last().is_none_or(|&c| c < 1.0) {
        cs.push(1.0);
    }
    let mut out = Vec::with_capacity(cs.len());
    let mut lo = MIN_THRESHOLD;
    for &hi in &cs {
        // The added top segment (above every prediction) keeps nothing.
        let p = bp.point_at(hi);
        out.push(Segment {
            lo,
            hi,
            score: objective.score(&p),
        });
        lo = hi;
    }
    out
}

/// The exact best threshold of `bp` for `objective` (see the module docs): None without
/// ground truth (or, for a precision target, without any detection).
pub fn exact_pick(bp: &Breakpoints, objective: Objective) -> Option<Pick> {
    if bp.npos == 0 {
        return None;
    }
    let segs = segments(bp, objective);
    let make =
        |t: f32, met: bool, note: Option<String>, exact: (f32, Option<f64>), run: [f32; 2]| Pick {
            objective: objective.to_string(),
            threshold: t,
            point: bp.point_at(t),
            met,
            note,
            exact_threshold: Some(exact.0),
            exact_score: exact.1,
            plateau: Some(run),
        };
    if let Some(best) = best_segment(&segs) {
        let (a, b) = plateau(&segs, PLATEAU).unwrap_or((best, best));
        let mut t = round_in(&segs, a, b);
        let mut note = None;
        if let Objective::Precision(target) = objective {
            // Rounding down may let in enough false positives to miss the target: take the
            // next 0.01 step up that meets it.
            let meets = |t: f32| bp.point_at(t).precision.is_some_and(|p| p >= target - TIE);
            if !meets(t) {
                let start = (t as f64 * 100.0).round() as u32 + 1;
                match (start..=100).map(|k| k as f32 / 100.0).find(|&u| meets(u)) {
                    Some(up) => t = up,
                    None => {
                        note = Some(format!(
                            "precision {}% is met only between 0.01 steps (exact optimum {:.3})",
                            pct0(target),
                            segs[best].hi
                        ))
                    }
                }
            }
        }
        let p = bp.point_at(t);
        if p.counts.tp == 0 {
            note = Some("no threshold finds any object".to_string());
        }
        return Some(make(
            t,
            true,
            note,
            (segs[best].hi, segs[best].score),
            [segs[a].lo, segs[b].hi],
        ));
    }
    let (_, primary, secondary) = fallback_keys(objective)?;
    // Target not reached: the best range on the constrained metric (then the other metric,
    // then the higher threshold).
    let prim = |sg: &Segment| primary(&bp.point_at(sg.hi));
    let sec = |sg: &Segment| secondary(&bp.point_at(sg.hi)).unwrap_or(0.0);
    let mut best: Option<usize> = None;
    for (i, sg) in segs.iter().enumerate() {
        let Some(v) = prim(sg) else { continue };
        let better = match best {
            None => true,
            Some(b) => {
                let bv = prim(&segs[b]).unwrap_or(0.0);
                v > bv + TIE || ((v - bv).abs() <= TIE && sec(sg) >= sec(&segs[b]) - TIE)
            }
        };
        if better {
            best = Some(i);
        }
    }
    let i = best?;
    let t = round_in(&segs, i, i);
    Some(make(
        t,
        false,
        Some(unreached_note(objective, &bp.point_at(segs[i].hi))),
        (segs[i].hi, None),
        [segs[i].lo, segs[i].hi],
    ))
}

/// The best point of `points` for `objective`; ties go to the higher threshold. None without
/// ground truth (or, for a precision target, without any detection).
pub fn pick(points: &[ThresholdPoint], objective: Objective) -> Option<Pick> {
    let mut sorted: Vec<&ThresholdPoint> = points.iter().filter(|p| p.recall.is_some()).collect();
    sorted.sort_by(|a, b| a.threshold.total_cmp(&b.threshold));
    let best_by = |score: &dyn Fn(&ThresholdPoint) -> Option<f64>| {
        let mut best: Option<(&ThresholdPoint, f64)> = None;
        for p in &sorted {
            if let Some(s) = score(p)
                && best.is_none_or(|(_, b)| s >= b - TIE)
            {
                best = Some((p, s));
            }
        }
        best.map(|(p, _)| *p)
    };
    let make = |p: &ThresholdPoint, met: bool, note: Option<String>| Pick {
        objective: objective.to_string(),
        threshold: p.threshold,
        point: *p,
        met,
        note,
        exact_threshold: None,
        exact_score: None,
        plateau: None,
    };
    if let Some(p) = best_by(&|p| objective.score(p)) {
        let note = (p.counts.tp == 0).then(|| "no threshold finds any object".to_string());
        return Some(make(&p, true, note));
    }
    let (_, primary, secondary) = fallback_keys(objective)?;
    // Target not reached: the best on the constrained metric (then the other one).
    let mut best: Option<&ThresholdPoint> = None;
    for p in &sorted {
        let Some(v) = primary(p) else { continue };
        let better = match best {
            None => true,
            Some(b) => {
                let bv = primary(b).unwrap_or(0.0);
                v > bv + TIE
                    || ((v - bv).abs() <= TIE
                        && secondary(p).unwrap_or(0.0) >= secondary(b).unwrap_or(0.0) - TIE)
            }
        };
        if better {
            best = Some(p);
        }
    }
    let p = best?;
    Some(make(p, false, Some(unreached_note(objective, p))))
}

type Metric = fn(&ThresholdPoint) -> Option<f64>;

/// For a target objective: (target, the constrained metric, the maximized one).
fn fallback_keys(objective: Objective) -> Option<(f64, Metric, Metric)> {
    match objective {
        Objective::Precision(t) => Some((t, |p| p.precision, |p| p.recall)),
        Objective::Recall(t) => Some((t, |p| p.recall, |p| p.precision)),
        _ => None,
    }
}

/// "precision 95% is not reached at any threshold; the most precise is 80% at 0.30".
fn unreached_note(objective: Objective, p: &ThresholdPoint) -> String {
    let Some((target, metric, _)) = fallback_keys(objective) else {
        return String::new();
    };
    let (name, most) = match objective {
        Objective::Recall(_) => ("recall", "the highest recall"),
        _ => ("precision", "the most precise"),
    };
    format!(
        "{name} {}% is not reached at any threshold; {most} is {}% at {:.2}",
        pct0(target),
        pct0(metric(p).unwrap_or(0.0)),
        p.threshold
    )
}

/// A curve of one group of images or objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Curve {
    /// "overall", dataset id, tag, `size:small` or class.
    pub key: String,
    pub images: usize,
    /// Scored ground-truth objects.
    pub gt: usize,
    /// Against pseudo ground truth.
    #[serde(default)]
    pub relative: bool,
    /// At [`SWEEP_THRESHOLDS`].
    pub points: Vec<ThresholdPoint>,
    /// At the run's configured threshold (also when it is not on the grid).
    pub configured: ThresholdPoint,
    /// Exact picks ([`exact_pick`]) for the run's objective and [`Objective::STANDARD`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub picks: Vec<Pick>,
    /// The exact curve, thinned to at most [`EXACT_POINTS`] points (overall curve only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exact: Vec<ThresholdPoint>,
}

/// Points of the thinned exact curve.
pub const EXACT_POINTS: usize = 150;

impl Curve {
    /// The best threshold for `objective`: the exact pick when the run computed it, else from
    /// the 0.05 grid.
    pub fn best(&self, objective: Objective) -> Option<Pick> {
        let key = objective.to_string();
        self.picks
            .iter()
            .find(|p| p.objective == key)
            .cloned()
            .or_else(|| pick(&self.points, objective))
    }
}

/// The objectives a run computes exact picks for: `objective` and [`Objective::STANDARD`].
pub fn objectives(objective: Objective) -> Vec<Objective> {
    let mut v = vec![objective];
    for o in Objective::STANDARD {
        if !v.contains(&o) {
            v.push(o);
        }
    }
    v
}

/// Threshold curves of one device run (in its [`super::AccuracyReport`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdSweep {
    /// The run's configured confidence threshold.
    pub configured: f32,
    pub overall: Curve,
    #[serde(default)]
    pub by_dataset: Vec<Curve>,
    /// Day / night / small-objects tags and `size:small` (objects under 32x32 px).
    #[serde(default)]
    pub by_tag: Vec<Curve>,
    #[serde(default)]
    pub per_class: Vec<Curve>,
}

/// Tags that get their own curve (when present).
pub const SWEEP_TAGS: [&str; 3] = ["day", "night", "small-objects"];
/// Key of the small-object curve.
pub const SMALL_KEY: &str = "size:small";

/// The best threshold of one group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupPick {
    pub key: String,
    pub images: usize,
    pub gt: usize,
    #[serde(default)]
    pub relative: bool,
    pub configured: ThresholdPoint,
    pub best: Option<Pick>,
}

impl GroupPick {
    fn of(c: &Curve, objective: Objective) -> Self {
        Self {
            key: c.key.clone(),
            images: c.images,
            gt: c.gt,
            relative: c.relative,
            configured: c.configured,
            best: c.best(objective),
        }
    }
}

/// The best threshold on one device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DevicePick {
    pub device: String,
    pub best: Option<Pick>,
}

/// A model's threshold advice (on [`super::ModelResult`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdAdvice {
    /// Objective of [`ThresholdAdvice::best`].
    pub objective: String,
    /// Device whose curves were used (the recommended one when it has them).
    pub device: String,
    /// Scored against pseudo ground truth.
    #[serde(default)]
    pub relative: bool,
    /// The configured threshold of the run, and its overall counts.
    pub configured: ThresholdPoint,
    pub best: Option<Pick>,
    /// The best threshold for each of [`Objective::STANDARD`] (comparison).
    #[serde(default)]
    pub alternatives: Vec<Pick>,
    /// Overall curve on [`ThresholdAdvice::device`] at [`SWEEP_THRESHOLDS`] (tables, chart).
    pub curve: Vec<ThresholdPoint>,
    /// The exact overall curve, thinned (chart).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exact: Vec<ThresholdPoint>,
    #[serde(default)]
    pub by_dataset: Vec<GroupPick>,
    #[serde(default)]
    pub by_tag: Vec<GroupPick>,
    #[serde(default)]
    pub per_class: Vec<GroupPick>,
    /// Best threshold per device that ran.
    #[serde(default)]
    pub by_device: Vec<DevicePick>,
}

fn pct(v: Option<f64>) -> String {
    v.map_or("-".into(), |v| format!("{:.1}%", v * 100.0))
}

/// "P 82.1% R 74.0% F1 78.0%".
pub fn prf(p: &ThresholdPoint) -> String {
    format!(
        "P {} R {} F1 {}",
        pct(p.precision),
        pct(p.recall),
        pct(p.f1)
    )
}

impl ThresholdAdvice {
    /// The objective, parsed.
    pub fn objective(&self) -> Objective {
        self.objective.parse().unwrap_or_default()
    }

    /// "best confidence threshold 0.35 (F1): P 82.1% R 74.0% F1 78.0%; configured 0.50: P ..".
    pub fn summary(&self) -> Option<String> {
        let b = self.best.as_ref()?;
        let mut s = format!(
            "P/R at the best confidence threshold {:.2} ({}): {}",
            b.threshold,
            self.objective().describe(),
            prf(&b.point)
        );
        if (b.threshold - self.configured.threshold).abs() > 1e-6 {
            s.push_str(&format!(
                " vs {} at the configured {:.2}",
                prf(&self.configured),
                self.configured.threshold
            ));
        } else {
            s.push_str(" (the configured threshold)");
        }
        if let Some(n) = &b.note {
            s.push_str(&format!(" ({n})"));
        }
        Some(s)
    }

    /// The threshold to write into the config: the best one, when the objective was met and it
    /// finds objects (an unreachable precision target or a model that finds nothing changes
    /// nothing).
    pub fn apply_value(&self) -> Option<f32> {
        self.best
            .as_ref()
            .filter(|b| b.met && b.point.counts.tp > 0)
            .map(|b| b.threshold)
    }
}

/// The sweep of a device row, when it ran and was scored.
fn sweep_of(d: &DeviceResult) -> Option<&ThresholdSweep> {
    d.run
        .as_ref()
        .filter(|_| d.ok())
        .and_then(|r| r.accuracy.as_ref())
        .and_then(|a| a.sweep.as_ref())
}

/// The threshold advice of a model from its device rows: curves of `recommended` when it has
/// them, else of the CPU reference device, else of the first scored device.
pub fn advise(
    devices: &[DeviceResult],
    recommended: Option<&str>,
    objective: Objective,
) -> Option<ThresholdAdvice> {
    let reference = super::REFERENCE_DEVICES.map(|d| d.to_string());
    let chosen = recommended
        .and_then(|r| {
            devices
                .iter()
                .find(|d| d.device == r && sweep_of(d).is_some())
        })
        .or_else(|| {
            reference.iter().find_map(|r| {
                devices
                    .iter()
                    .find(|d| d.device == *r && sweep_of(d).is_some())
            })
        })
        .or_else(|| devices.iter().find(|d| sweep_of(d).is_some()))?;
    let sweep = sweep_of(chosen)?;
    let groups = |cs: &[Curve]| -> Vec<GroupPick> {
        cs.iter().map(|c| GroupPick::of(c, objective)).collect()
    };
    Some(ThresholdAdvice {
        objective: objective.to_string(),
        device: chosen.device.clone(),
        relative: sweep.overall.relative,
        configured: sweep.overall.configured,
        best: sweep.overall.best(objective),
        alternatives: Objective::STANDARD
            .iter()
            .filter_map(|o| sweep.overall.best(*o))
            .collect(),
        curve: sweep.overall.points.clone(),
        exact: sweep.overall.exact.clone(),
        by_dataset: groups(&sweep.by_dataset),
        by_tag: groups(&sweep.by_tag),
        per_class: groups(&sweep.per_class),
        by_device: devices
            .iter()
            .filter_map(|d| {
                let s = sweep_of(d)?;
                Some(DevicePick {
                    device: d.device.clone(),
                    best: s.overall.best(objective),
                })
            })
            .collect(),
    })
}

/// Rows of the compact threshold table: 0.20 .. 0.80, plus the configured and best thresholds
/// when outside that range or off the grid.
pub fn table_points(curve: &[ThresholdPoint], extra: &[ThresholdPoint]) -> Vec<ThresholdPoint> {
    let mut v: Vec<ThresholdPoint> = curve
        .iter()
        .filter(|p| p.threshold >= 0.2 - 1e-6 && p.threshold <= 0.8 + 1e-6)
        .copied()
        .collect();
    for e in extra {
        if !v.iter().any(|p| (p.threshold - e.threshold).abs() < 1e-6) {
            v.push(*e);
        }
    }
    v.sort_by(|a, b| a.threshold.total_cmp(&b.threshold));
    v
}

/// Whether `t` is one of [`SWEEP_THRESHOLDS`].
pub fn on_grid(t: f32) -> bool {
    SWEEP_THRESHOLDS.iter().any(|g| (g - t).abs() < 1e-6)
}

/// Advice whose best threshold is `t` (tests of the apply paths).
#[cfg(test)]
pub(crate) fn test_advice(t: f32) -> ThresholdAdvice {
    use super::metrics::Counts;
    let p = |t: f32| {
        ThresholdPoint::new(
            t,
            Counts {
                tp: 3,
                fp: 1,
                fn_: 1,
            },
            2,
        )
    };
    ThresholdAdvice {
        objective: "f1".into(),
        device: "openvino:cpu".into(),
        relative: false,
        configured: p(0.5),
        best: Some(Pick {
            objective: "f1".into(),
            threshold: t,
            point: p(t),
            met: true,
            note: None,
            exact_threshold: Some(t),
            exact_score: p(t).f1,
            plateau: Some([t, t]),
        }),
        alternatives: vec![],
        curve: SWEEP_THRESHOLDS.iter().map(|&t| p(t)).collect(),
        exact: vec![],
        by_dataset: vec![],
        by_tag: vec![],
        per_class: vec![],
        by_device: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::super::metrics::Counts;
    use super::*;

    fn pt(t: f32, tp: usize, fp: usize, fn_: usize) -> ThresholdPoint {
        ThresholdPoint::new(t, Counts { tp, fp, fn_ }, 10)
    }

    /// A typical curve over 10 objects: recall falls and precision rises with the threshold.
    fn curve() -> Vec<ThresholdPoint> {
        vec![
            pt(0.1, 10, 30, 0), // P .25  R 1   F1 .4    F2 .625
            pt(0.2, 9, 6, 1),   // P .6   R .9  F1 .72   F2 .818
            pt(0.3, 8, 2, 2),   // P .8   R .8  F1 .8    F2 .8
            pt(0.4, 7, 1, 3),   // P .875 R .7  F1 .778  F2 .729
            pt(0.5, 6, 0, 4),   // P 1    R .6  F1 .75   F2 .652
            pt(0.6, 0, 0, 10),  // nothing detected
        ]
    }

    #[test]
    fn objectives_parse_and_print() {
        for (s, o) in [
            ("f1", Objective::F1),
            (" F2 ", Objective::F2),
            ("precision:0.9", Objective::Precision(0.9)),
            ("Precision:1", Objective::Precision(1.0)),
            ("", Objective::F1),
        ] {
            assert_eq!(s.parse::<Objective>().unwrap(), o, "{s}");
        }
        for bad in [
            "f3",
            "precision:",
            "precision:abc",
            "precision:0",
            "precision:1.5",
        ] {
            assert!(bad.parse::<Objective>().is_err(), "{bad}");
        }
        assert_eq!(Objective::Precision(0.9).to_string(), "precision:0.9");
        assert_eq!(
            "recall:0.8".parse::<Objective>().unwrap(),
            Objective::Recall(0.8)
        );
        assert_eq!(Objective::Recall(0.8).to_string(), "recall:0.8");
        assert!("recall:2".parse::<Objective>().is_err());
        assert_eq!(
            serde_json::to_string(&Objective::Precision(0.85)).unwrap(),
            "\"precision:0.85\""
        );
        let o: Objective = serde_json::from_str("\"f2\"").unwrap();
        assert_eq!(o, Objective::F2);
        assert!(serde_json::from_str::<Objective>("\"recall\"").is_err());
        assert_eq!(
            Objective::Precision(0.9).describe(),
            "recall at precision \u{2265} 90%"
        );
    }

    #[test]
    fn picks_by_objective() {
        let c = curve();
        let f1 = pick(&c, Objective::F1).unwrap();
        assert_eq!((f1.threshold, f1.met, f1.note.is_none()), (0.3, true, true));
        let f2 = pick(&c, Objective::F2).unwrap();
        assert_eq!(f2.threshold, 0.2);
        // Highest recall with precision >= 0.85: 0.4 (P .875 R .7).
        let p = pick(&c, Objective::Precision(0.85)).unwrap();
        assert_eq!((p.threshold, p.met), (0.4, true));
        assert_eq!(p.objective, "precision:0.85");
        // Highest precision with recall >= 0.8: 0.3 (P .8 R .8); recall 1 only at 0.1.
        assert_eq!(pick(&c, Objective::Recall(0.8)).unwrap().threshold, 0.3);
        assert_eq!(pick(&c, Objective::Recall(1.0)).unwrap().threshold, 0.1);
        // Precision 1 is reached at 0.5 only.
        assert_eq!(pick(&c, Objective::Precision(1.0)).unwrap().threshold, 0.5);
    }

    #[test]
    fn ties_go_to_the_higher_threshold() {
        // Same counts at 0.3 and 0.35: same F1, the higher threshold wins.
        let c = vec![
            pt(0.25, 5, 5, 5),
            pt(0.3, 8, 2, 2),
            pt(0.35, 8, 2, 2),
            pt(0.4, 1, 0, 9),
        ];
        assert_eq!(pick(&c, Objective::F1).unwrap().threshold, 0.35);
        assert_eq!(pick(&c, Objective::Precision(0.8)).unwrap().threshold, 0.35);
        // Order of the input does not matter.
        let mut r = c.clone();
        r.reverse();
        assert_eq!(pick(&r, Objective::F1).unwrap().threshold, 0.35);
    }

    #[test]
    fn unreachable_precision_falls_back_with_a_note() {
        let c = vec![pt(0.1, 10, 30, 0), pt(0.2, 9, 6, 1), pt(0.3, 8, 2, 2)];
        let p = pick(&c, Objective::Precision(0.95)).unwrap();
        assert!(!p.met);
        assert_eq!(p.threshold, 0.3, "the most precise threshold");
        let note = p.note.unwrap();
        assert!(note.contains("95%") && note.contains("80%"), "{note}");
        // Equal precision: the higher recall.
        let c = vec![pt(0.1, 4, 4, 6), pt(0.2, 2, 2, 8)];
        let p = pick(&c, Objective::Precision(0.9)).unwrap();
        assert_eq!(p.threshold, 0.1);
        // No ground truth, or no detections at all.
        let none = vec![ThresholdPoint::new(
            0.5,
            Counts {
                tp: 0,
                fp: 3,
                fn_: 0,
            },
            2,
        )];
        assert!(pick(&none, Objective::F1).is_none());
        let silent = vec![pt(0.5, 0, 0, 3), pt(0.6, 0, 0, 3)];
        assert!(pick(&silent, Objective::Precision(0.9)).is_none());
        // F1 0 everywhere: still a pick (highest threshold), with a note.
        let f = pick(&silent, Objective::F1).unwrap();
        assert_eq!(f.threshold, 0.6);
        assert!(f.note.is_some());
    }

    #[test]
    fn table_rows_cover_the_middle_and_the_marks() {
        let full: Vec<ThresholdPoint> = SWEEP_THRESHOLDS.iter().map(|&t| pt(t, 1, 1, 1)).collect();
        let rows = table_points(&full, &[]);
        assert_eq!(rows.len(), 13);
        assert_eq!((rows[0].threshold, rows[12].threshold), (0.2, 0.8));
        let rows = table_points(
            &full,
            &[pt(0.1, 1, 1, 1), pt(0.42, 1, 1, 1), pt(0.5, 1, 1, 1)],
        );
        assert_eq!(rows.len(), 15);
        assert_eq!(rows[0].threshold, 0.1);
        assert!(rows.iter().any(|p| p.threshold == 0.42));
        assert!(on_grid(0.35) && !on_grid(0.42));
    }

    /// NMS families post-processed at 0.05 and filtered at `t` give exactly the detections of
    /// post-processing at `t` (greedy NMS keeps a box based on more confident boxes only).
    #[test]
    fn nms_after_low_threshold_then_filter_is_exact() {
        use crate::model::yolo5::Yolo5;
        use crate::model::{Family, NamedOutput, OutputBuf, PostParams, PreprocessCtx, ResizeMode};
        let mut seed: u64 = 42;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        let classes = 3;
        let rows = 400;
        let mut data = Vec::with_capacity(rows * (5 + classes));
        for _ in 0..rows {
            // Many boxes on a small canvas: plenty overlap.
            let (cx, cy) = (20.0 + rnd() * 600.0, 20.0 + rnd() * 600.0);
            data.extend([cx, cy, 30.0 + rnd() * 40.0, 30.0 + rnd() * 40.0, rnd()]);
            for _ in 0..classes {
                data.push(rnd());
            }
        }
        let out = vec![NamedOutput {
            name: "output0".into(),
            shape: vec![1, rows, 5 + classes],
            data: OutputBuf::F32(data),
        }];
        let ctx = PreprocessCtx {
            orig_w: 640,
            orig_h: 640,
            input_w: 640,
            input_h: 640,
            mode: ResizeMode::Letterbox,
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        };
        let fam = Yolo5::new(classes);
        let at = |t: f32| {
            fam.postprocess(
                &out,
                &ctx,
                &PostParams {
                    confidence_threshold: t,
                    nms_iou: 0.45,
                },
            )
            .unwrap()
        };
        let low = at(0.05);
        assert!(low.len() > 20);
        for t in SWEEP_THRESHOLDS {
            let filtered: Vec<_> = low.iter().filter(|d| d.score >= t).cloned().collect();
            assert_eq!(filtered, at(t), "t = {t}");
        }
    }

    fn seg(lo: f32, hi: f32, score: Option<f64>) -> Segment {
        Segment { lo, hi, score }
    }

    #[test]
    fn plateau_prefers_the_widest_near_best_range() {
        // A lone spike (0.80 at (0.30, 0.31]) vs a broad top within 0.002 ((0.40, 0.70]).
        let s = [
            seg(0.05, 0.30, Some(0.70)),
            seg(0.30, 0.31, Some(0.800)),
            seg(0.31, 0.40, Some(0.70)),
            seg(0.40, 0.55, Some(0.7985)),
            seg(0.55, 0.70, Some(0.7990)),
            seg(0.70, 1.00, Some(0.0)),
        ];
        assert_eq!(best_segment(&s), Some(1));
        assert_eq!(plateau(&s, PLATEAU), Some((3, 4)));
        // Midpoint 0.55 of (0.40, 0.70].
        assert_eq!(round_in(&s, 3, 4), 0.55);
        // Without tolerance: the spike only.
        assert_eq!(plateau(&s, 0.0), Some((1, 1)));
        // The spike's midpoint 0.305 rounds down to 0.30, out of (0.30, 0.31]: its top, 0.31.
        assert_eq!(round_in(&s, 1, 1), 0.31);
        // Equal widths: the higher run; equal scores: the higher segment.
        let t = [
            seg(0.05, 0.25, Some(0.5)),
            seg(0.25, 0.30, Some(0.1)),
            seg(0.30, 0.50, Some(0.5)),
        ];
        assert_eq!(best_segment(&t), Some(2));
        assert_eq!(plateau(&t, PLATEAU), Some((2, 2)));
        // The lowest segment includes its lower end; never below the minimum.
        let u = [seg(0.05, 0.059, Some(1.0)), seg(0.059, 1.0, Some(0.0))];
        assert_eq!(round_in(&u, 0, 0), 0.05);
        assert!(plateau(&[seg(0.05, 1.0, None)], PLATEAU).is_none());
        assert_eq!(floor_cent(0.35f32 as f64), 0.35);
        assert_eq!(floor_cent(0.3549), 0.35);
        assert_eq!(floor_cent(0.0599), 0.05);
    }

    fn random_breakpoints(seed: u64, n: usize) -> Breakpoints {
        use super::super::metrics::{SweepImage, SweepPred, SweepScope};
        let mut seed = seed;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        let mut img = SweepImage::default();
        let objects = n / 2;
        for _ in 0..objects {
            img.gt.push(("person".into(), false));
        }
        let mut tps = 0;
        for _ in 0..n {
            let c = 0.05 + rnd() * 0.95;
            // More confident detections are more often right.
            let tp = tps < objects && rnd() < c;
            tps += usize::from(tp);
            img.preds.push(SweepPred {
                confidence: c,
                class: "person".into(),
                tp,
                small: false,
            });
        }
        Breakpoints::new(&[&img], SweepScope::All)
    }

    /// The exact optimum equals brute force over every breakpoint and is at least the grid's;
    /// the reported threshold is on 0.01, at least 0.05, and its numbers are re-evaluated.
    #[test]
    fn exact_pick_matches_brute_force_and_beats_the_grid() {
        for seed in 1..40u64 {
            let bp = random_breakpoints(seed, 30 + seed as usize * 3);
            for o in [
                Objective::F1,
                Objective::F2,
                Objective::Precision(0.8),
                Objective::Precision(0.9),
                Objective::Recall(0.7),
                Objective::Recall(0.95),
            ] {
                let brute = bp
                    .steps
                    .iter()
                    .filter_map(|s| o.score(&bp.point_at(s.0)))
                    .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
                let grid: Vec<ThresholdPoint> =
                    SWEEP_THRESHOLDS.iter().map(|&t| bp.point_at(t)).collect();
                let g = pick(&grid, o);
                let Some(p) = exact_pick(&bp, o) else {
                    assert!(brute.is_none() && g.is_none(), "seed {seed} {o}");
                    continue;
                };
                if !p.met {
                    assert!(brute.is_none(), "seed {seed} {o}");
                    continue;
                }
                let exact = p.exact_score.unwrap();
                assert!((exact - brute.unwrap()).abs() < 1e-12, "seed {seed} {o}");
                if let Some(g) = g.filter(|g| g.met) {
                    assert!(
                        exact >= o.score(&g.point).unwrap() - 1e-12,
                        "seed {seed} {o}"
                    );
                }
                // Reported: a 0.01 step >= 0.05, its own (re-evaluated) counts, inside or near
                // the plateau.
                assert!(p.threshold >= MIN_THRESHOLD);
                assert!(((p.threshold * 100.0).round() / 100.0 - p.threshold).abs() < 1e-6);
                assert_eq!(p.point, bp.point_at(p.threshold));
                let [lo, hi] = p.plateau.unwrap();
                assert!(lo < hi && lo >= MIN_THRESHOLD && hi <= 1.0);
                // Every threshold of the plateau scores within PLATEAU of the optimum.
                for k in 0..20 {
                    let t = lo + (hi - lo) * (k as f32 + 0.5) / 20.0;
                    let v = o.score(&bp.point_at(t)).unwrap();
                    assert!(v >= exact - PLATEAU - 1e-9, "seed {seed} {o} t {t}");
                }
                if let Objective::Precision(target) = o
                    && p.point.precision.is_some_and(|v| v < target)
                {
                    // Only when no 0.01 step up meets the target.
                    let up = (1..=100)
                        .map(|k| k as f32 / 100.0)
                        .filter(|&t| t > p.threshold)
                        .any(|t| bp.point_at(t).precision.is_some_and(|v| v >= target));
                    assert!(!up, "seed {seed} {o}");
                }
            }
        }
    }

    #[test]
    fn exact_pick_hand_computed() {
        use super::super::metrics::{SweepImage, SweepPred, SweepScope};
        // 4 objects; detections by confidence: TP .92, TP .81, FP .63, TP .47, FP .12.
        let mut img = SweepImage::default();
        for _ in 0..4 {
            img.gt.push(("person".into(), false));
        }
        for (c, tp) in [
            (0.92, true),
            (0.81, true),
            (0.63, false),
            (0.47, true),
            (0.12, false),
        ] {
            img.preds.push(SweepPred {
                confidence: c,
                class: "person".into(),
                tp,
                small: false,
            });
        }
        let bp = Breakpoints::new(&[&img], SweepScope::All);
        // F1 by kept set: {.92} 2/5=.4, {..81} 4/6=.667, {..63} 4/7=.571, {..47} 6/8=.75,
        // {..12} 6/9=.667. Best: keep down to .47, i.e. thresholds (0.12, 0.47].
        let p = exact_pick(&bp, Objective::F1).unwrap();
        assert_eq!(p.exact_threshold, Some(0.47));
        assert!((p.exact_score.unwrap() - 0.75).abs() < 1e-12);
        assert_eq!(p.plateau, Some([0.12, 0.47]));
        // Midpoint 0.295 -> 0.29, same kept set.
        assert_eq!(p.threshold, 0.29);
        assert!((p.point.f1.unwrap() - 0.75).abs() < 1e-12);
        // The grid would say 0.15..0.45 (equal F1), tie -> 0.45.
        let grid: Vec<ThresholdPoint> = SWEEP_THRESHOLDS.iter().map(|&t| bp.point_at(t)).collect();
        assert_eq!(pick(&grid, Objective::F1).unwrap().threshold, 0.45);
        // Precision 1: highest recall is keeping {.92, .81} (R .5): (0.63, 0.81] -> 0.72.
        let p = exact_pick(&bp, Objective::Precision(1.0)).unwrap();
        assert_eq!((p.threshold, p.point.recall), (0.72, Some(0.5)));
    }
}
