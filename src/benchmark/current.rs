//! The saved results seen through the config now.
//!
//! `benchmark.json` records each model's threshold advice at the confidence threshold the run
//! was configured with ([`ThresholdAdvice::configured`]) and the device the config resolved to
//! then ([`ModelResult::configured`]). After "Apply threshold" or a device change those are
//! history: the Benchmark page, `/v1/benchmark` and the exported reports compare the best
//! threshold with the one configured *now*. [`rebase`] fills [`ThresholdAdvice::now`] (and each
//! group's `now`) at the model's current effective threshold, rewrites the threshold part of
//! the recommendation text, and sets [`ModelResult::configured`] to the device the config
//! resolves to now. Nothing rebased is saved.
//!
//! The counts at the new threshold are exact: from the stored predictions
//! ([`super::search::points_at`], checked against the run's own counts at its threshold, so
//! predictions of another run are never mixed in), else from a point the results store at
//! exactly that threshold (0.05 grid, picks), else left unknown ([`NowSource::None`]).

use super::report::{BenchmarkResults, ImageSetInfo, ModelResult};
use super::search::{StoredPreds, points_at};
use super::threshold::{NowSource, NowThreshold, ThresholdAdvice};
use crate::config::{Config, ModelConfig};
use crate::registry::normalize_name;
use std::collections::HashMap;

/// What the config says about a model now.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelNow {
    /// Effective confidence threshold (the model's own, else the global one).
    pub threshold: f32,
    /// What the loaded model runs with, when known (differs while a restart is pending).
    pub running: Option<f32>,
    /// Canonical spec of the device the config resolves to ([`super::configured_device`]).
    pub device: Option<String>,
}

/// Normalized model name -> [`ModelNow`] of every configured model. `device` resolves a model's
/// configured device (usually [`super::configured_device`] with the model's [`crate::backend::Selection`]);
/// `running` gives the threshold a loaded model runs with.
pub fn from_config(
    cfg: &Config,
    device: &dyn Fn(&ModelConfig) -> Option<String>,
    running: &dyn Fn(&str) -> Option<f32>,
) -> HashMap<String, ModelNow> {
    cfg.models
        .iter()
        .map(|m| {
            let name = m.effective_name();
            (
                normalize_name(&name),
                ModelNow {
                    threshold: m.confidence_threshold.unwrap_or(cfg.confidence_threshold),
                    running: running(&name),
                    device: device(m),
                },
            )
        })
        .collect()
}

/// Rebase every model of `results` that the config has (see the module docs).
pub fn rebase(
    results: &mut BenchmarkResults,
    preds: Option<&StoredPreds>,
    now: &HashMap<String, ModelNow>,
) {
    let BenchmarkResults {
        models, image_sets, ..
    } = results;
    for m in models {
        if let Some(n) = now.get(&normalize_name(&m.model)) {
            rebase_model(m, image_sets, preds, n);
        }
    }
}

/// Rebase one model: `sets` (the results' image sets, with ground truth) and `preds` give the
/// exact counts at a threshold the run did not use.
pub fn rebase_model(
    m: &mut ModelResult,
    sets: &[ImageSetInfo],
    preds: Option<&StoredPreds>,
    n: &ModelNow,
) {
    if let Some(d) = &n.device {
        m.configured = Some(d.clone());
    }
    if let Some(a) = m.threshold.as_mut() {
        m.recommendation = rebase_advice(a, &m.model, &m.recommendation, sets, preds, n);
    }
}

/// Fill [`ThresholdAdvice::now`] of `model`'s advice for `n`; returns `recommendation` (which
/// ends with the advice's summary, see `ModelResult::finish`) with that summary rewritten.
pub fn rebase_advice(
    a: &mut ThresholdAdvice,
    model: &str,
    recommendation: &str,
    sets: &[ImageSetInfo],
    preds: Option<&StoredPreds>,
    n: &ModelNow,
) -> String {
    let old = a.summary();
    let t = n.threshold;
    let running = n.running.filter(|r| (r - t).abs() > 1e-6);
    a.now = None;
    if (t - a.configured.threshold).abs() < 1e-6 {
        a.now = Some(NowThreshold {
            threshold: t,
            running,
            point: Some(a.configured),
            source: NowSource::Run,
        });
        for g in a
            .by_dataset
            .iter_mut()
            .chain(&mut a.by_tag)
            .chain(&mut a.per_class)
        {
            g.now = Some(g.configured);
        }
    } else {
        let exact = preds
            .and_then(|p| p.runs_of(model).into_iter().find(|r| r.device == a.device))
            .and_then(|r| points_at(sets, r, t))
            .filter(|p| {
                p.at_run.counts == a.configured.counts
                    && (p.at_run.threshold - a.configured.threshold).abs() < 1e-6
            });
        let (point, source) = match &exact {
            Some(p) => (Some(p.overall), NowSource::Preds),
            None => match a.known_point_at(t) {
                Some(p) => (Some(p), NowSource::Curve),
                None => (None, NowSource::None),
            },
        };
        for (groups, found) in [
            (&mut a.by_dataset, exact.as_ref().map(|p| &p.by_dataset)),
            (&mut a.by_tag, exact.as_ref().map(|p| &p.by_tag)),
            (&mut a.per_class, exact.as_ref().map(|p| &p.per_class)),
        ] {
            for g in groups {
                g.now = found.and_then(|f| f.get(&g.key)).copied();
            }
        }
        a.now = Some(NowThreshold {
            threshold: t,
            running,
            point,
            source,
        });
    }
    match (old, a.summary()) {
        (Some(old), Some(new)) => match recommendation.strip_suffix(&old) {
            Some(head) => format!("{head}{new}"),
            None => recommendation.to_string(),
        },
        _ => recommendation.to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::benchmark::search::{SearchRequest, search};
    use crate::benchmark::threshold::{Objective, ThresholdAdvice};

    /// The search fixture with model `m`'s threshold advice on its recommended device built as
    /// a benchmark run builds it (configured 0.50, best 0.35 for F1) and a recommendation that
    /// ends with the advice summary.
    pub(crate) fn advised_fixture() -> (BenchmarkResults, StoredPreds) {
        let (mut results, preds) = crate::benchmark::search::tests::fixture();
        let r = search(&results, &preds, &SearchRequest::default(), &|_| None)
            .unwrap()
            .results
            .remove(0);
        let a = ThresholdAdvice {
            objective: Objective::F1.to_string(),
            device: r.device.clone(),
            relative: r.relative,
            configured: r.configured,
            best: r.best.clone(),
            alternatives: r.alternatives.clone(),
            curve: r.grid.clone(),
            exact: r.exact.clone(),
            by_dataset: r
                .by_dataset
                .iter()
                .map(|g| crate::benchmark::threshold::GroupPick {
                    now: None,
                    ..g.clone()
                })
                .collect(),
            by_tag: vec![],
            per_class: r
                .per_class
                .iter()
                .map(|g| crate::benchmark::threshold::GroupPick {
                    now: None,
                    ..g.clone()
                })
                .collect(),
            by_device: vec![],
            roc: r.roc.clone(),
            now: None,
        };
        let m = &mut results.models[0];
        m.recommendation = format!(
            "fastest agreeing device; {}",
            a.summary().expect("best threshold")
        );
        m.threshold = Some(a);
        (results, preds)
    }

    fn now(t: f32) -> ModelNow {
        ModelNow {
            threshold: t,
            running: None,
            device: None,
        }
    }

    #[test]
    fn rebase_evaluates_the_configured_threshold_now_exactly() {
        let (results, preds) = advised_fixture();
        // Unchanged config: the run's own counts.
        let mut r = results.clone();
        rebase_model(
            &mut r.models[0],
            &results.image_sets,
            Some(&preds),
            &now(0.5),
        );
        let a = r.models[0].threshold.as_ref().unwrap();
        let n = a.now.as_ref().unwrap();
        assert_eq!((n.source, n.threshold), (NowSource::Run, 0.5));
        assert_eq!(n.point, Some(a.configured));
        assert!(a.by_dataset.iter().all(|g| g.now == Some(g.configured)));
        assert_eq!(r.models[0].recommendation, results.models[0].recommendation);

        // After "Apply threshold" (0.35, the best): exact from the stored predictions, equal
        // to the search's evaluation at the threshold configured now.
        let s = search(&results, &preds, &SearchRequest::default(), &|_| Some(0.35))
            .unwrap()
            .results
            .remove(0);
        let mut r = results.clone();
        let n35 = ModelNow {
            running: Some(0.5),
            ..now(0.35)
        };
        rebase_model(&mut r.models[0], &results.image_sets, Some(&preds), &n35);
        let a = r.models[0].threshold.as_ref().unwrap();
        let n = a.now.as_ref().unwrap();
        assert_eq!(
            (n.source, n.threshold, n.running),
            (NowSource::Preds, 0.35, Some(0.5))
        );
        assert_eq!(n.point, Some(s.now));
        // At 0.35 every object is found and the FP (.3) is filtered: P = R = F1 = 1.
        let p = n.point.unwrap();
        assert_eq!((p.counts.tp, p.counts.fp, p.counts.fn_), (3, 0, 0));
        assert_eq!(p.f1, Some(1.0));
        assert_eq!(a.now_point(), Some(&p));
        for (g, sg) in a.by_dataset.iter().zip(&s.by_dataset) {
            assert_eq!(g.now, sg.now, "{}", g.key);
        }
        for (g, sg) in a.per_class.iter().zip(&s.per_class) {
            assert_eq!(g.now, sg.now, "{}", g.key);
        }
        // The recommendation now compares with the configured 0.35 (= the best).
        let rec = &r.models[0].recommendation;
        assert!(rec.starts_with("fastest agreeing device; "), "{rec}");
        assert!(rec.contains("(the configured threshold)"), "{rec}");
        assert!(rec.contains("running 0.50 until a restart"), "{rec}");
        assert!(!rec.contains("at the configured 0.50"), "{rec}");

        // Without stored predictions: a stored grid point is exact (0.35 is on the grid) ...
        let mut r = results.clone();
        rebase_model(&mut r.models[0], &results.image_sets, None, &now(0.35));
        let n = r.models[0].threshold.as_ref().unwrap().now.clone().unwrap();
        assert_eq!(
            (n.source, n.point.unwrap().counts),
            (NowSource::Curve, p.counts)
        );
        assert!(
            r.models[0]
                .threshold
                .as_ref()
                .unwrap()
                .by_dataset
                .iter()
                .all(|g| g.now.is_none())
        );
        // ... an off-grid threshold is not evaluated (never interpolated).
        let mut r = results.clone();
        rebase_model(&mut r.models[0], &results.image_sets, None, &now(0.37));
        let a = r.models[0].threshold.as_ref().unwrap();
        let n = a.now.as_ref().unwrap();
        assert_eq!((n.source, n.point), (NowSource::None, None));
        assert_eq!(a.now_point(), None);
        assert!(
            r.models[0]
                .recommendation
                .contains("(configured 0.37, not evaluated)"),
            "{}",
            r.models[0].recommendation
        );
        // An off-grid threshold with predictions: exact.
        let mut r = results.clone();
        rebase_model(
            &mut r.models[0],
            &results.image_sets,
            Some(&preds),
            &now(0.42),
        );
        let n = r.models[0].threshold.as_ref().unwrap().now.clone().unwrap();
        assert_eq!(n.source, NowSource::Preds);
        // .85 and .6 TP, .4 car below: 2 TP, 0 FP, 1 FN.
        let c = n.point.unwrap().counts;
        assert_eq!((c.tp, c.fp, c.fn_), (2, 0, 1));

        // Stored predictions of another run (counts at the run's threshold differ) are not
        // mixed in.
        let mut other = preds.clone();
        for run in &mut other.runs {
            for img in &mut run.images {
                img.preds.clear();
            }
        }
        let mut r = results.clone();
        rebase_model(
            &mut r.models[0],
            &results.image_sets,
            Some(&other),
            &now(0.37),
        );
        let n = r.models[0].threshold.as_ref().unwrap().now.clone().unwrap();
        assert_eq!(n.source, NowSource::None);
    }

    #[test]
    fn rebase_from_config_sets_device_and_threshold() {
        let (mut results, preds) = advised_fixture();
        // What is saved never carries `now`.
        let text = serde_json::to_string(&results).unwrap();
        assert!(!text.contains("\"now\""));
        let mut cfg = Config {
            confidence_threshold: 0.5,
            ..Default::default()
        };
        cfg.models.push(ModelConfig {
            name: Some("M".into()),
            path: "m.onnx".into(),
            confidence_threshold: Some(0.35),
            ..Default::default()
        });
        let map = from_config(&cfg, &|_| Some("openvino:cpu".into()), &|_| Some(0.5));
        rebase(&mut results, Some(&preds), &map);
        let m = &results.models[0];
        assert_eq!(m.configured.as_deref(), Some("openvino:cpu"));
        let n = m.threshold.as_ref().unwrap().now.as_ref().unwrap();
        assert_eq!((n.threshold, n.running), (0.35, Some(0.5)));
        // Served: `now` on the advice and its groups.
        let v = serde_json::to_value(m).unwrap();
        assert_eq!(v["threshold"]["now"]["source"], "preds");
        assert!(v["threshold"]["by_dataset"][0]["now"]["f1"].is_number());
    }
}
