//! Letter grades for a model on a device.
//!
//! - **Accuracy** from the accuracy score: AP@0.5 over the scored CCTV classes, blended with
//!   small-object recall when the images have small objects
//!   (`score = (1 - SMALL_RECALL_WEIGHT) * AP50 + SMALL_RECALL_WEIGHT * small recall`), against
//!   [`ACCURACY_THRESHOLDS`]: A >= 0.70, B >= 0.60, C >= 0.50, D >= 0.40, else F.
//! - **Speed** from the full-request p50 (decode + preprocess + inference + postprocess) against
//!   [`SPEED_THRESHOLDS_MS`]: A < 50 ms, B < 100, C < 200, D < 400, else F.
//! - **Overall**: the weighted mean of the grade points (A = 4 .. F = 0), default 60% accuracy /
//!   40% speed (config `benchmark.weights`), back to a letter at 3.5 / 2.5 / 1.5 / 0.5. Without
//!   an accuracy score the overall grade is the speed grade.

use serde::{Deserialize, Serialize};

/// Accuracy score lower bounds of A, B, C, D.
pub const ACCURACY_THRESHOLDS: [f64; 4] = [0.70, 0.60, 0.50, 0.40];
/// Full-request p50 upper bounds (ms, exclusive) of A, B, C, D.
pub const SPEED_THRESHOLDS_MS: [f64; 4] = [50.0, 100.0, 200.0, 400.0];
/// Share of small-object recall in the accuracy score (when there are small objects).
pub const SMALL_RECALL_WEIGHT: f64 = 0.2;
/// Default share of accuracy in the overall grade (speed gets the rest).
pub const DEFAULT_ACCURACY_WEIGHT: f64 = 0.6;
/// Overall-points lower bounds of A, B, C, D.
const OVERALL_THRESHOLDS: [f64; 4] = [3.5, 2.5, 1.5, 0.5];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub enum Grade {
    A,
    B,
    C,
    D,
    F,
}

impl Grade {
    const ALL: [Grade; 5] = [Grade::A, Grade::B, Grade::C, Grade::D, Grade::F];

    pub fn points(self) -> f64 {
        match self {
            Grade::A => 4.0,
            Grade::B => 3.0,
            Grade::C => 2.0,
            Grade::D => 1.0,
            Grade::F => 0.0,
        }
    }

    pub fn letter(self) -> &'static str {
        match self {
            Grade::A => "A",
            Grade::B => "B",
            Grade::C => "C",
            Grade::D => "D",
            Grade::F => "F",
        }
    }

    /// Grade of a higher-is-better value against descending lower bounds.
    fn at_least(v: f64, bounds: &[f64; 4]) -> Self {
        bounds
            .iter()
            .position(|&b| v >= b)
            .map_or(Grade::F, |i| Self::ALL[i])
    }

    pub fn accuracy(score: f64) -> Self {
        Self::at_least(score, &ACCURACY_THRESHOLDS)
    }

    pub fn speed(p50_ms: f64) -> Self {
        SPEED_THRESHOLDS_MS
            .iter()
            .position(|&b| p50_ms < b)
            .map_or(Grade::F, |i| Self::ALL[i])
    }

    pub fn from_points(p: f64) -> Self {
        Self::at_least(p, &OVERALL_THRESHOLDS)
    }
}

impl std::fmt::Display for Grade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.letter())
    }
}

/// Accuracy vs speed weights of the overall grade (normalized when used).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Weights {
    pub accuracy: f64,
    pub speed: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            accuracy: DEFAULT_ACCURACY_WEIGHT,
            speed: 1.0 - DEFAULT_ACCURACY_WEIGHT,
        }
    }
}

impl Weights {
    /// Accuracy share in 0..=1 (the default when both weights are 0 or invalid).
    pub fn accuracy_share(&self) -> f64 {
        let (a, s) = (self.accuracy.max(0.0), self.speed.max(0.0));
        if !(a + s).is_finite() || a + s <= 0.0 {
            DEFAULT_ACCURACY_WEIGHT
        } else {
            a / (a + s)
        }
    }
}

/// Accuracy score: AP@0.5 blended with small-object recall (see the module docs).
pub fn accuracy_score(ap50: f64, small_recall: Option<f64>) -> f64 {
    match small_recall {
        Some(r) => (1.0 - SMALL_RECALL_WEIGHT) * ap50 + SMALL_RECALL_WEIGHT * r,
        None => ap50,
    }
}

/// The grades of one model on one device.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Grades {
    pub accuracy: Option<Grade>,
    pub accuracy_score: Option<f64>,
    pub speed: Grade,
    /// Full-request p50 the speed grade is based on.
    pub p50_ms: f64,
    pub overall: Grade,
    /// Weighted grade points, 0..4 (ranking key).
    pub overall_points: f64,
    /// Accuracy was measured against pseudo ground truth.
    #[serde(default)]
    pub relative: bool,
}

impl Grades {
    pub fn new(accuracy_score: Option<f64>, p50_ms: f64, weights: Weights, relative: bool) -> Self {
        let speed = Grade::speed(p50_ms);
        let accuracy = accuracy_score.map(Grade::accuracy);
        let points = match accuracy {
            Some(a) => {
                let w = weights.accuracy_share();
                w * a.points() + (1.0 - w) * speed.points()
            }
            None => speed.points(),
        };
        Self {
            accuracy,
            accuracy_score,
            speed,
            p50_ms,
            overall: Grade::from_points(points),
            overall_points: points,
            relative,
        }
    }

    /// Ranking order: overall points, then accuracy score, then speed (best first).
    pub fn rank_cmp(a: &Grades, b: &Grades) -> std::cmp::Ordering {
        b.overall_points
            .total_cmp(&a.overall_points)
            .then_with(|| {
                b.accuracy_score
                    .unwrap_or(-1.0)
                    .total_cmp(&a.accuracy_score.unwrap_or(-1.0))
            })
            .then_with(|| a.p50_ms.total_cmp(&b.p50_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_thresholds() {
        assert_eq!(Grade::speed(0.0), Grade::A);
        assert_eq!(Grade::speed(49.9), Grade::A);
        assert_eq!(Grade::speed(50.0), Grade::B);
        assert_eq!(Grade::speed(99.0), Grade::B);
        assert_eq!(Grade::speed(150.0), Grade::C);
        assert_eq!(Grade::speed(399.0), Grade::D);
        assert_eq!(Grade::speed(400.0), Grade::F);
    }

    #[test]
    fn accuracy_thresholds() {
        assert_eq!(Grade::accuracy(0.95), Grade::A);
        assert_eq!(Grade::accuracy(0.70), Grade::A);
        assert_eq!(Grade::accuracy(0.69), Grade::B);
        assert_eq!(Grade::accuracy(0.55), Grade::C);
        assert_eq!(Grade::accuracy(0.40), Grade::D);
        assert_eq!(Grade::accuracy(0.1), Grade::F);
        assert!((accuracy_score(0.8, Some(0.3)) - 0.70).abs() < 1e-12);
        assert_eq!(accuracy_score(0.8, None), 0.8);
    }

    #[test]
    fn overall_weighting() {
        let w = Weights::default();
        assert!((w.accuracy_share() - 0.6).abs() < 1e-12);
        // A accuracy (4), C speed (2): 0.6*4 + 0.4*2 = 3.2 -> B.
        let g = Grades::new(Some(0.75), 150.0, w, false);
        assert_eq!((g.accuracy, g.speed), (Some(Grade::A), Grade::C));
        assert!((g.overall_points - 3.2).abs() < 1e-12);
        assert_eq!(g.overall, Grade::B);
        // Speed only.
        let g = Grades::new(None, 30.0, w, false);
        assert_eq!((g.overall, g.overall_points), (Grade::A, 4.0));
        // Accuracy-only weighting.
        let g = Grades::new(
            Some(0.75),
            1000.0,
            Weights {
                accuracy: 1.0,
                speed: 0.0,
            },
            false,
        );
        assert_eq!(g.overall, Grade::A);
        // Degenerate weights fall back to the default.
        assert!(
            (Weights {
                accuracy: 0.0,
                speed: 0.0
            }
            .accuracy_share()
                - 0.6)
                .abs()
                < 1e-12
        );
        assert_eq!(Grade::from_points(3.5), Grade::A);
        assert_eq!(Grade::from_points(0.49), Grade::F);
    }

    #[test]
    fn ranking() {
        let w = Weights::default();
        let mut v = [
            Grades::new(Some(0.5), 40.0, w, false),
            Grades::new(Some(0.75), 40.0, w, false),
            Grades::new(Some(0.75), 30.0, w, false),
        ];
        v.sort_by(Grades::rank_cmp);
        assert_eq!(v[0].p50_ms, 30.0);
        assert_eq!(v[2].accuracy_score, Some(0.5));
    }
}
