//! Standalone benchmark reports from [`BenchmarkResults`]: Markdown (`--report out.md`) and a
//! self-contained HTML page (`--report out.html`, inline CSS, light/dark, no scripts or
//! external files). Both show the machine, the datasets, the cross-model ranking, and per model
//! the device table with grades plus the recommended device's breakdown by dataset, tag and
//! resolution.

use super::grade::{ACCURACY_THRESHOLDS, SMALL_RECALL_WEIGHT, SPEED_THRESHOLDS_MS};
use super::report::{BenchmarkResults, DeviceResult, ModelResult};
use super::{Breakdown, RunResult};
use anyhow::{Context, Result, bail};
use std::fmt::Write as _;
use std::path::Path;

/// A table: title, header, rows of cells.
struct Table {
    title: String,
    head: Vec<String>,
    rows: Vec<Vec<String>>,
}

/// A report section: heading, paragraphs, tables.
struct Section {
    heading: String,
    level: u8,
    paragraphs: Vec<String>,
    tables: Vec<Table>,
}

fn pct(v: Option<f64>) -> String {
    v.map_or("-".into(), |v| format!("{:.1}", v * 100.0))
}

fn msf(v: f64) -> String {
    format!("{v:.1}")
}

fn grade_cells(r: &RunResult) -> [String; 3] {
    match &r.grades {
        Some(g) => [
            g.overall.to_string(),
            g.accuracy.map_or("-".into(), |a| {
                if g.relative {
                    format!("{a}*")
                } else {
                    a.to_string()
                }
            }),
            g.speed.to_string(),
        ],
        None => ["-".into(), "-".into(), "-".into()],
    }
}

fn device_row(m: &ModelResult, d: &DeviceResult) -> Vec<String> {
    let mut marks = Vec::new();
    if m.recommended.as_deref() == Some(d.device.as_str()) {
        marks.push("recommended");
    }
    if m.configured.as_deref() == Some(d.device.as_str()) {
        marks.push("configured");
    }
    if m.fastest().is_some_and(|f| f.device == d.device) {
        marks.push("fastest");
    }
    let mut device = d.device.clone();
    if !marks.is_empty() {
        device.push_str(&format!(" ({})", marks.join(", ")));
    }
    match (&d.run, &d.error) {
        (Some(r), _) if d.ok() => {
            let acc = r.accuracy.as_ref().map(|a| &a.overall);
            let [o, a, s] = grade_cells(r);
            vec![
                device,
                format!("{:.0}", r.compile_ms),
                msf(r.stages_ms.infer.p50),
                msf(r.stages_ms.total.p50),
                msf(r.stages_ms.total.p95),
                format!("{:.1}", r.throughput_fps),
                pct(acc.and_then(|s| s.ap50)),
                pct(acc.and_then(|s| s.ap50_95)),
                pct(acc.and_then(|s| s.precision)),
                pct(acc.and_then(|s| s.recall)),
                o,
                a,
                s,
                super::cli::agreement_text(r),
            ]
        }
        (Some(r), _) => {
            let mut v = vec![device, format!("fell back to {}", r.device)];
            v.resize(14, String::new());
            v
        }
        (None, e) => {
            let mut v = vec![
                device,
                format!("failed: {}", e.as_deref().unwrap_or("did not run")),
            ];
            v.resize(14, String::new());
            v
        }
    }
}

fn breakdown_table(title: &str, rows: &[Breakdown]) -> Table {
    Table {
        title: title.to_string(),
        head: ["", "images", "objects", "AP50", "AP50-95", "P", "R", "F1"]
            .map(String::from)
            .to_vec(),
        rows: rows
            .iter()
            .map(|b| {
                vec![
                    if b.relative {
                        format!("{} (relative)", b.key)
                    } else {
                        b.key.clone()
                    },
                    b.images.to_string(),
                    b.gt.to_string(),
                    pct(b.ap50),
                    pct(b.ap50_95),
                    pct(b.precision),
                    pct(b.recall),
                    pct(b.f1),
                ]
            })
            .collect(),
    }
}

fn sections(r: &BenchmarkResults) -> Vec<Section> {
    let mut out = Vec::new();
    let hw = &r.hardware;
    let rt = &r.runtimes;
    out.push(Section {
        heading: "Machine".into(),
        level: 2,
        paragraphs: vec![
            format!(
                "{} {} | CPU {} | GPUs: {}",
                hw.os,
                hw.arch,
                if hw.cpu.is_empty() { "?" } else { &hw.cpu },
                if hw.gpus.is_empty() {
                    "none".to_string()
                } else {
                    hw.gpus.join("; ")
                }
            ),
            format!(
                "OpenVINO {} | ONNX Runtime {}{} | Blue Onyx Prism {} | updated {}",
                rt.openvino.as_deref().unwrap_or("not loaded"),
                rt.onnxruntime.as_deref().unwrap_or("not loaded"),
                rt.onnxruntime_flavor
                    .as_deref()
                    .map(|f| format!(" ({f})"))
                    .unwrap_or_default(),
                r.version,
                r.timestamp
            ),
        ],
        tables: vec![],
    });
    out.push(Section {
        heading: "Datasets".into(),
        level: 2,
        paragraphs: vec![],
        tables: vec![Table {
            title: String::new(),
            head: [
                "dataset",
                "images",
                "scored against",
                "license",
                "attribution",
            ]
            .map(String::from)
            .to_vec(),
            rows: r
                .image_sets
                .iter()
                .map(|s| {
                    vec![
                        format!("{} ({})", s.id, s.title),
                        s.images.len().to_string(),
                        s.ground_truth.describe(),
                        s.license.clone(),
                        s.attribution.clone(),
                    ]
                })
                .collect(),
        }],
    });
    let ranking = r.ranking();
    if !ranking.is_empty() {
        out.push(Section {
            heading: "Best model for this machine".into(),
            level: 2,
            paragraphs: vec![format!(
                "Each model on its recommended device, ranked by overall grade, then accuracy, then \
                 speed. Models with no class in the datasets are not scored for accuracy and \
                 rank last. Best: {} on {}.",
                ranking[0].model, ranking[0].device
            )],
            tables: vec![Table {
                title: String::new(),
                head: ["#", "model", "device", "overall", "accuracy", "speed", "p50 ms", "AP50"]
                    .map(String::from)
                    .to_vec(),
                rows: ranking
                    .iter()
                    .map(|k| {
                        vec![
                            k.rank.to_string(),
                            k.model.clone(),
                            k.device.clone(),
                            if k.accuracy.is_none() {
                                format!("{} (speed only)", k.overall)
                            } else {
                                k.overall.to_string()
                            },
                            k.accuracy.map_or("not scored".into(), |a| {
                                if k.relative {
                                    format!("{a}*")
                                } else {
                                    a.to_string()
                                }
                            }),
                            k.speed.to_string(),
                            msf(k.p50_ms),
                            pct(k.ap50),
                        ]
                    })
                    .collect(),
            }],
        });
    }
    for m in &r.models {
        let mut paragraphs = vec![format!(
            "{} | datasets {} ({} images) | {} timed run(s) per image after {} warm-up | {}",
            m.path,
            m.datasets.join(", "),
            m.images,
            m.repeat,
            m.warmup,
            m.timestamp
        )];
        if let Some(e) = &m.error {
            paragraphs.push(format!("Not benchmarked: {e}"));
        }
        paragraphs.push(match &m.recommended {
            Some(d) => format!("Recommended device: {d} ({})", m.recommendation),
            None => format!("No recommendation: {}", m.recommendation),
        });
        if !m.skipped.is_empty() {
            paragraphs.push(format!("Not runnable here: {}", m.skipped.join("; ")));
        }
        let mut tables = vec![Table {
            title: "Devices".into(),
            head: [
                "device",
                "load ms",
                "infer p50",
                "total p50",
                "total p95",
                "req/s",
                "AP50",
                "AP50-95",
                "P",
                "R",
                "overall",
                "accuracy",
                "speed",
                "agreement",
            ]
            .map(String::from)
            .to_vec(),
            rows: m.devices.iter().map(|d| device_row(m, d)).collect(),
        }];
        let best = m
            .recommended
            .as_ref()
            .and_then(|rec| m.devices.iter().find(|d| &d.device == rec))
            .and_then(|d| d.run.as_ref());
        if let Some(run) = best {
            if let Some(a) = &run.accuracy {
                paragraphs.push(format!(
                    "Accuracy of {} ({}; classes {}; P/R at confidence {:.2}, IoU 0.5).",
                    run.requested_device,
                    a.ground_truth,
                    a.classes.join(", "),
                    a.threshold
                ));
                tables.push(breakdown_table("By dataset", &a.by_dataset));
                tables.push(breakdown_table("By tag", &a.by_tag));
                tables.push(Table {
                    title: "By class".into(),
                    head: ["class", "objects", "AP50", "AP50-95", "P", "R"]
                        .map(String::from)
                        .to_vec(),
                    rows: a
                        .overall
                        .per_class
                        .iter()
                        .map(|c| {
                            vec![
                                c.class.clone(),
                                c.gt.to_string(),
                                pct(c.ap50),
                                pct(c.ap50_95),
                                pct(c.precision),
                                pct(c.recall),
                            ]
                        })
                        .collect(),
                });
                tables.push(Table {
                    title: "Recall by object size".into(),
                    head: ["size", "objects", "recall"].map(String::from).to_vec(),
                    rows: a
                        .overall
                        .by_size
                        .iter()
                        .map(|b| vec![b.bucket.clone(), b.gt.to_string(), pct(b.recall)])
                        .collect(),
                });
            } else if let Some(n) = &run.accuracy_note {
                paragraphs.push(format!("Accuracy not scored: {n}."));
            }
            if !run.by_resolution.is_empty() {
                tables.push(Table {
                    title: format!("Speed by resolution on {} (ms)", run.requested_device),
                    head: [
                        "resolution",
                        "images",
                        "decode+pre p50",
                        "infer p50",
                        "post p50",
                        "total p50",
                        "total p95",
                    ]
                    .map(String::from)
                    .to_vec(),
                    rows: super::images::RESOLUTION_BUCKETS
                        .iter()
                        .map(|bucket| {
                            match run.by_resolution.iter().find(|b| b.bucket == *bucket) {
                                Some(b) => vec![
                                    b.bucket.clone(),
                                    b.images.to_string(),
                                    msf(b.pre_p50),
                                    msf(b.infer_p50),
                                    msf(b.post_p50),
                                    msf(b.total_p50),
                                    msf(b.total_p95),
                                ],
                                None => {
                                    let mut v = vec![bucket.to_string(), "0".into()];
                                    v.resize(7, "n/a".into());
                                    v
                                }
                            }
                        })
                        .collect(),
                });
            }
        }
        out.push(Section {
            heading: m.model.clone(),
            level: 2,
            paragraphs,
            tables,
        });
    }
    out.push(Section {
        heading: "How grades are computed".into(),
        level: 2,
        paragraphs: vec![
            format!(
                "Speed: full request p50 (decode, preprocess, inference, postprocess) A < {} ms, B < {}, C < {}, D < {}, else F.",
                SPEED_THRESHOLDS_MS[0], SPEED_THRESHOLDS_MS[1], SPEED_THRESHOLDS_MS[2], SPEED_THRESHOLDS_MS[3]
            ),
            format!(
                "Accuracy: AP@0.5 over the CCTV classes the model has (person, bicycle, car, motorcycle, bus, truck, dog, cat, bird, horse; IPcam 'vehicle' = car/truck/bus), blended {:.0}% with small-object recall when there are small objects: A >= {:.2}, B >= {:.2}, C >= {:.2}, D >= {:.2}, else F. * = relative to a reference model, not ground truth.",
                SMALL_RECALL_WEIGHT * 100.0,
                ACCURACY_THRESHOLDS[0], ACCURACY_THRESHOLDS[1], ACCURACY_THRESHOLDS[2], ACCURACY_THRESHOLDS[3]
            ),
            "Overall: weighted mean of the grade points (A=4 .. F=0), default 60% accuracy, 40% speed (config benchmark.weights). Devices whose detections disagree with the CPU reference are never recommended.".into(),
        ],
        tables: vec![],
    });
    out
}

/// Markdown report.
pub fn markdown(r: &BenchmarkResults) -> String {
    let esc = |s: &str| s.replace('|', "\\|").replace('\n', " ");
    let mut out = String::from("# Blue Onyx Prism benchmark report\n\n");
    for s in sections(r) {
        let _ = writeln!(out, "{} {}\n", "#".repeat(s.level as usize), s.heading);
        for p in &s.paragraphs {
            let _ = writeln!(out, "{}\n", p);
        }
        for t in &s.tables {
            if t.rows.is_empty() {
                continue;
            }
            if !t.title.is_empty() {
                let _ = writeln!(out, "**{}**\n", t.title);
            }
            let _ = writeln!(
                out,
                "| {} |",
                t.head
                    .iter()
                    .map(|h| esc(h))
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
            let _ = writeln!(out, "|{}", " --- |".repeat(t.head.len()));
            for row in &t.rows {
                let _ = writeln!(
                    out,
                    "| {} |",
                    row.iter().map(|c| esc(c)).collect::<Vec<_>>().join(" | ")
                );
            }
            out.push('\n');
        }
    }
    out
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Self-contained HTML report.
pub fn html(r: &BenchmarkResults) -> String {
    let mut out = String::from(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>Blue Onyx Prism benchmark report</title><style>\
         :root{--bg:#f6f7f9;--fg:#1d2330;--muted:#5d6677;--card:#fff;--border:#d9dde5}\
         @media (prefers-color-scheme: dark){:root{--bg:#12151b;--fg:#dfe4ec;--muted:#949db0;--card:#1b2029;--border:#2d3442}}\
         body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}\
         main{max-width:1200px;margin:0 auto;padding:1.5rem}p{color:var(--muted)}\
         .scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--border);margin:.5rem 0 1rem}\
         th,td{padding:.35rem .6rem;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}\
         th{font-size:.85rem;color:var(--muted)}h3{font-size:1rem}\
         </style></head><body><main>\n<h1>Blue Onyx Prism benchmark report</h1>\n",
    );
    for s in sections(r) {
        let _ = writeln!(out, "<h{l}>{}</h{l}>", html_escape(&s.heading), l = s.level);
        for p in &s.paragraphs {
            let _ = writeln!(out, "<p>{}</p>", html_escape(p));
        }
        for t in &s.tables {
            if t.rows.is_empty() {
                continue;
            }
            if !t.title.is_empty() {
                let _ = writeln!(out, "<h3>{}</h3>", html_escape(&t.title));
            }
            out.push_str("<div class=\"scroll\"><table><thead><tr>");
            for h in &t.head {
                let _ = write!(out, "<th>{}</th>", html_escape(h));
            }
            out.push_str("</tr></thead><tbody>\n");
            for row in &t.rows {
                out.push_str("<tr>");
                for c in row {
                    let _ = write!(out, "<td>{}</td>", html_escape(c));
                }
                out.push_str("</tr>\n");
            }
            out.push_str("</tbody></table></div>\n");
        }
    }
    out.push_str("</main></body></html>\n");
    out
}

/// Write the report for `path` by its extension (`.html`/`.htm` or `.md`/`.markdown`).
pub fn write(r: &BenchmarkResults, path: &Path) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let text = match ext.as_str() {
        "html" | "htm" => html(r),
        "md" | "markdown" => markdown(r),
        _ => bail!("--report {}: use a .html or .md file name", path.display()),
    };
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_render_and_escape() {
        let mut m = ModelResult::failed("<b>m</b>", "m.onnx", "file | missing".into());
        m.datasets = vec!["sample".into()];
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![m])
            .with_sets(&[super::super::images::ImageSet::sample()]);
        let md = markdown(&r);
        assert!(md.starts_with("# Blue Onyx Prism benchmark report"));
        assert!(md.contains("## <b>m</b>"));
        assert!(md.contains("Not benchmarked: file | missing"));
        assert!(md.contains("| sample (Embedded sample"), "{md}");
        assert!(md.contains("How grades are computed"));
        let h = html(&r);
        assert!(h.contains("&lt;b&gt;m&lt;/b&gt;") && !h.contains("<b>m</b>"));
        assert!(h.contains("prefers-color-scheme"));
        let dir = std::env::temp_dir().join(format!("bop-rep-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&r, &dir.join("r.md")).unwrap();
        write(&r, &dir.join("r.HTML")).unwrap();
        assert!(write(&r, &dir.join("r.txt")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
