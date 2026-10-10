//! Hugging Face model catalog and downloader.
//!
//! Downloaded files are untrusted data: file names come only from the static catalog below
//! and everything is written strictly under the destination directory.

use crate::config::{Config, ModelConfig};
use crate::model::ModelFamilyKind;
use anyhow::{Context, Result, anyhow, bail};
use hf_hub::HFClientSync;
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub struct ModelEntry {
    pub name: &'static str,
    pub repo: &'static str,
    pub files: &'static [&'static str],
    pub family: ModelFamilyKind,
    pub description: &'static str,
}

const RTDETR_REPO: &str = "xnorpx/rt-detr2-onnx";
const YOLO5_REPO: &str = "xnorpx/blue-onyx-yolo5";

static CATALOG: &[ModelEntry] = &[
    ModelEntry {
        name: "rt-detrv2-s",
        repo: RTDETR_REPO,
        files: &["rt-detrv2-s.onnx", "rt-detrv2-s.yaml"],
        family: ModelFamilyKind::RtDetr,
        description: "RT-DETRv2 small, general COCO",
    },
    ModelEntry {
        name: "rt-detrv2-ms",
        repo: RTDETR_REPO,
        files: &["rt-detrv2-ms.onnx", "rt-detrv2-ms.yaml"],
        family: ModelFamilyKind::RtDetr,
        description: "RT-DETRv2 medium-small, general COCO",
    },
    ModelEntry {
        name: "rt-detrv2-m",
        repo: RTDETR_REPO,
        files: &["rt-detrv2-m.onnx", "rt-detrv2-m.yaml"],
        family: ModelFamilyKind::RtDetr,
        description: "RT-DETRv2 medium, general COCO",
    },
    ModelEntry {
        name: "rt-detrv2-l",
        repo: RTDETR_REPO,
        files: &["rt-detrv2-l.onnx", "rt-detrv2-l.yaml"],
        family: ModelFamilyKind::RtDetr,
        description: "RT-DETRv2 large, general COCO",
    },
    ModelEntry {
        name: "rt-detrv2-x",
        repo: RTDETR_REPO,
        files: &["rt-detrv2-x.onnx", "rt-detrv2-x.yaml"],
        family: ModelFamilyKind::RtDetr,
        description: "RT-DETRv2 extra large, general COCO",
    },
    ModelEntry {
        name: "delivery",
        repo: YOLO5_REPO,
        files: &["delivery.onnx", "delivery.yaml"],
        family: ModelFamilyKind::Yolo5,
        description: "YOLOv5 delivery (vehicles, people, packages)",
    },
    ModelEntry {
        name: "IPcam-animal",
        repo: YOLO5_REPO,
        files: &["IPcam-animal.onnx", "IPcam-animal.yaml"],
        family: ModelFamilyKind::Yolo5,
        description: "YOLOv5 IP camera animals",
    },
    ModelEntry {
        name: "ipcam-bird",
        repo: YOLO5_REPO,
        files: &["ipcam-bird.onnx", "ipcam-bird.yaml"],
        family: ModelFamilyKind::Yolo5,
        description: "YOLOv5 IP camera birds",
    },
    ModelEntry {
        name: "IPcam-combined",
        repo: YOLO5_REPO,
        files: &["IPcam-combined.onnx", "IPcam-combined.yaml"],
        family: ModelFamilyKind::Yolo5,
        description: "YOLOv5 IP camera combined",
    },
    ModelEntry {
        name: "IPcam-dark",
        repo: YOLO5_REPO,
        files: &["IPcam-dark.onnx", "IPcam-dark.yaml"],
        family: ModelFamilyKind::Yolo5,
        description: "YOLOv5 IP camera night/dark scenes",
    },
    ModelEntry {
        name: "IPcam-general",
        repo: YOLO5_REPO,
        files: &["IPcam-general.onnx", "IPcam-general.yaml"],
        family: ModelFamilyKind::Yolo5,
        description: "YOLOv5 IP camera general purpose",
    },
    ModelEntry {
        name: "package",
        repo: YOLO5_REPO,
        files: &["package.onnx", "package.yaml"],
        family: ModelFamilyKind::Yolo5,
        description: "YOLOv5 package detection",
    },
];

pub fn catalog() -> &'static [ModelEntry] {
    CATALOG
}

/// Case-insensitive lookup by name (a trailing `.onnx` is accepted).
pub fn find(name: &str) -> Option<&'static ModelEntry> {
    let n = name.trim();
    let n = n
        .strip_suffix(".onnx")
        .or_else(|| n.strip_suffix(".ONNX"))
        .unwrap_or(n);
    CATALOG.iter().find(|e| e.name.eq_ignore_ascii_case(n))
}

struct BarHandler {
    bar: ProgressBar,
}

impl ProgressHandler for BarHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        if let ProgressEvent::Download(ev) = event {
            match ev {
                DownloadEvent::Start { total_bytes, .. } => self.bar.set_length(*total_bytes),
                DownloadEvent::Progress { files } => {
                    if let Some(f) = files.last() {
                        if f.total_bytes > 0 {
                            self.bar.set_length(f.total_bytes);
                        }
                        self.bar.set_position(f.bytes_completed);
                    }
                }
                DownloadEvent::AggregateProgress {
                    bytes_completed,
                    total_bytes,
                    ..
                } => {
                    self.bar.set_length(*total_bytes);
                    self.bar.set_position(*bytes_completed);
                }
                DownloadEvent::Complete => {
                    if let Some(len) = self.bar.length() {
                        self.bar.set_position(len);
                    }
                }
            }
        }
    }
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg:30} [{bar:30}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=> ")
}

/// Download the named models (or the whole catalog when `all`) into `dest_dir`.
/// Returns the paths of every file written (or already present and refreshed).
pub fn download(names: &[String], all: bool, dest_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries: Vec<&'static ModelEntry> = Vec::new();
    if all {
        entries.extend(CATALOG.iter());
    } else {
        for n in names {
            let e = find(n).ok_or_else(|| {
                anyhow!("unknown model '{n}'; run `list-models` to see available names")
            })?;
            if !entries.iter().any(|x| x.name == e.name) {
                entries.push(e);
            }
        }
    }
    if entries.is_empty() {
        bail!("no models selected: pass --name <NAME> or --all");
    }

    std::fs::create_dir_all(dest_dir)
        .with_context(|| format!("creating {}", dest_dir.display()))?;
    let dest_dir = dest_dir
        .canonicalize()
        .with_context(|| format!("resolving {}", dest_dir.display()))?;
    let client = HFClientSync::new().map_err(|e| anyhow!("creating Hugging Face client: {e}"))?;

    let mut written = Vec::new();
    for entry in entries {
        let (owner, repo_name) = entry
            .repo
            .split_once('/')
            .ok_or_else(|| anyhow!("bad repo id {}", entry.repo))?;
        let repo = client.model(owner, repo_name);
        for file in entry.files {
            let bar = ProgressBar::new(0);
            bar.set_style(bar_style());
            bar.set_message((*file).to_string());
            let handler = BarHandler { bar: bar.clone() };
            let path = repo
                .download_file()
                .filename((*file).to_string())
                .local_dir(dest_dir.clone())
                .progress(Progress::new(handler))
                .send()
                .map_err(|e| anyhow!("downloading {}/{file}: {e}", entry.repo));
            match path {
                Ok(p) => {
                    bar.finish();
                    let canon = p.canonicalize().unwrap_or(p.clone());
                    if !canon.starts_with(&dest_dir) {
                        bail!(
                            "refusing file outside destination directory: {}",
                            canon.display()
                        );
                    }
                    written.push(p);
                }
                Err(e) => {
                    bar.abandon();
                    // A missing .yaml is not fatal: the model falls back to default classes.
                    if file.ends_with(".yaml") {
                        tracing::warn!("{e}");
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }
    Ok(written)
}

/// Build config entries for the downloaded `.onnx` files (name = catalog name, family from the
/// catalog, classes = the sibling `.yaml` when it was downloaded). Paths under `exe_dir` are
/// stored relative to it, others absolute. Files not in the catalog are ignored.
pub fn model_configs(files: &[PathBuf], exe_dir: &Path) -> Vec<ModelConfig> {
    let exe_canon = exe_dir.canonicalize().ok();
    let rel = |p: &Path| -> PathBuf {
        for base in [Some(exe_dir), exe_canon.as_deref()].into_iter().flatten() {
            if let Ok(r) = p.strip_prefix(base) {
                return r.to_path_buf();
            }
        }
        p.to_path_buf()
    };
    let mut out = Vec::new();
    for f in files {
        if f.extension().and_then(|e| e.to_str()) != Some("onnx") {
            continue;
        }
        let Some(entry) = f.file_stem().and_then(|s| s.to_str()).and_then(find) else {
            continue;
        };
        let yaml = f.with_extension("yaml");
        out.push(ModelConfig {
            name: Some(entry.name.to_string()),
            path: rel(f),
            family: entry.family,
            classes: files.contains(&yaml).then(|| rel(&yaml)),
            ..Default::default()
        });
    }
    out
}

/// What [`add_to_config`] did with one downloaded model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddOutcome {
    Added { name: String, enabled: bool },
    Skipped { name: String },
}

impl AddOutcome {
    /// One line for the `download-models --add-to-config` output.
    pub fn describe(&self, config_path: &Path) -> String {
        match self {
            AddOutcome::Added {
                name,
                enabled: true,
            } => format!("added '{name}' to {} (enabled)", config_path.display()),
            AddOutcome::Added {
                name,
                enabled: false,
            } => format!(
                "added '{name}' to {} (disabled; enable it on the Config page)",
                config_path.display()
            ),
            AddOutcome::Skipped { name } => {
                format!("'{name}' already in {}, skipped", config_path.display())
            }
        }
    }
}

/// Append downloaded models to `config` (duplicates by name or path are skipped). New entries
/// are disabled when the config already has an enabled model; otherwise the first new entry is
/// enabled and the rest disabled, so a fresh config loads exactly one model.
pub fn add_to_config(config: &mut Config, models: Vec<ModelConfig>) -> Vec<AddOutcome> {
    let mut have_enabled = config.models.iter().any(|m| m.enabled);
    models
        .into_iter()
        .map(|m| {
            let name = m.effective_name();
            let enabled = !have_enabled;
            if config.add_model_if_absent(ModelConfig { enabled, ..m }) {
                have_enabled |= enabled;
                AddOutcome::Added { name, enabled }
            } else {
                AddOutcome::Skipped { name }
            }
        })
        .collect()
}

/// Print the catalog and whether each model is already in `dest_dir`.
pub fn print_list(dest_dir: &Path) {
    println!("Available models (destination: {})", dest_dir.display());
    println!("{:<16} {:<8} {:<9} DESCRIPTION", "NAME", "FAMILY", "LOCAL");
    for e in CATALOG {
        let present = e.files.iter().all(|f| dest_dir.join(f).exists());
        let onnx = e.files.iter().any(|f| dest_dir.join(f).exists());
        let status = if present {
            "yes"
        } else if onnx {
            "partial"
        } else {
            "no"
        };
        println!(
            "{:<16} {:<8} {:<9} {}",
            e.name, e.family, status, e.description
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_shape() {
        assert_eq!(catalog().len(), 12);
        for e in catalog() {
            assert_eq!(e.files.len(), 2);
            assert!(e.files[0].ends_with(".onnx") && e.files[1].ends_with(".yaml"));
            assert!(
                e.files
                    .iter()
                    .all(|f| !f.contains('/') && !f.contains(".."))
            );
        }
    }

    #[test]
    fn find_is_case_insensitive() {
        assert_eq!(find("ipcam-general").unwrap().name, "IPcam-general");
        assert_eq!(find("IPCAM-BIRD.onnx").unwrap().name, "ipcam-bird");
        assert_eq!(find("RT-DETRV2-S").unwrap().family, ModelFamilyKind::RtDetr);
        assert!(find("nope").is_none());
    }

    #[test]
    fn model_configs_from_files() {
        let exe = Path::new("/exe");
        let files = vec![
            PathBuf::from("/exe/models/IPcam-general.onnx"),
            PathBuf::from("/exe/models/IPcam-general.yaml"),
            PathBuf::from("/elsewhere/rt-detrv2-s.onnx"),
            PathBuf::from("/elsewhere/unknown.onnx"),
        ];
        let c = model_configs(&files, exe);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].name.as_deref(), Some("IPcam-general"));
        assert_eq!(c[0].path, Path::new("models/IPcam-general.onnx"));
        assert_eq!(c[0].family, ModelFamilyKind::Yolo5);
        assert_eq!(
            c[0].classes.as_deref(),
            Some(Path::new("models/IPcam-general.yaml"))
        );
        assert_eq!(c[1].path, Path::new("/elsewhere/rt-detrv2-s.onnx"));
        assert_eq!(c[1].family, ModelFamilyKind::RtDetr);
        assert_eq!(c[1].classes, None);
    }

    #[test]
    fn add_to_config_enables_only_the_first_model_of_an_empty_config() {
        let m = |n: &str| ModelConfig {
            name: Some(n.into()),
            path: format!("models/{n}.onnx").into(),
            ..Default::default()
        };
        let mut c = Config::default();
        let out = add_to_config(&mut c, vec![m("a"), m("b"), m("a")]);
        assert_eq!(
            out,
            [
                AddOutcome::Added {
                    name: "a".into(),
                    enabled: true
                },
                AddOutcome::Added {
                    name: "b".into(),
                    enabled: false
                },
                AddOutcome::Skipped { name: "a".into() },
            ]
        );
        let on: Vec<bool> = c.models.iter().map(|m| m.enabled).collect();
        assert_eq!(on, [true, false]);
        assert!(
            out[1]
                .describe(Path::new("cfg.json"))
                .contains("(disabled; enable it on the Config page)")
        );

        // Existing enabled model: everything new is disabled.
        let out = add_to_config(&mut c, vec![m("c")]);
        assert_eq!(
            out,
            [AddOutcome::Added {
                name: "c".into(),
                enabled: false
            }]
        );

        // Only disabled models present: the first new one is enabled.
        let mut c = Config {
            models: vec![ModelConfig {
                enabled: false,
                ..m("x")
            }],
            ..Default::default()
        };
        add_to_config(&mut c, vec![m("y"), m("z")]);
        let on: Vec<bool> = c.models.iter().map(|m| m.enabled).collect();
        assert_eq!(on, [false, true, false]);
    }

    #[test]
    fn add_to_config_with_an_enabled_model_appends_disabled_and_skips_known_paths() {
        let existing_off = ModelConfig {
            name: Some("old-off".into()),
            path: "models/old-off.onnx".into(),
            enabled: false,
            ..Default::default()
        };
        let mut c = Config {
            models: vec![
                ModelConfig {
                    name: Some("main".into()),
                    path: "models/main.onnx".into(),
                    ..Default::default()
                },
                existing_off.clone(),
            ],
            ..Default::default()
        };
        let new = |n: &str, path: &str| ModelConfig {
            name: Some(n.into()),
            path: path.into(),
            ..Default::default()
        };
        let out = add_to_config(
            &mut c,
            vec![
                new("x", "models/x.onnx"),
                // Same path as the disabled entry, other name: skipped, flag untouched.
                new("renamed", "models/old-off.onnx"),
                // Same path as the enabled entry: skipped, stays enabled.
                new("again", "models/main.onnx"),
                new("y", "models/y.onnx"),
            ],
        );
        assert_eq!(
            out,
            [
                AddOutcome::Added {
                    name: "x".into(),
                    enabled: false
                },
                AddOutcome::Skipped {
                    name: "renamed".into()
                },
                AddOutcome::Skipped {
                    name: "again".into()
                },
                AddOutcome::Added {
                    name: "y".into(),
                    enabled: false
                },
            ]
        );
        let flags: Vec<(String, bool)> = c
            .models
            .iter()
            .map(|m| (m.effective_name(), m.enabled))
            .collect();
        assert_eq!(
            flags,
            [
                ("main".to_string(), true),
                ("old-off".to_string(), false),
                ("x".to_string(), false),
                ("y".to_string(), false),
            ]
        );
        assert_eq!(c.models[1], existing_off);

        let cfg = Path::new("cfg.json");
        assert_eq!(
            AddOutcome::Added {
                name: "a".into(),
                enabled: true
            }
            .describe(cfg),
            "added 'a' to cfg.json (enabled)"
        );
        assert_eq!(
            out[0].describe(cfg),
            "added 'x' to cfg.json (disabled; enable it on the Config page)"
        );
        assert_eq!(
            out[1].describe(cfg),
            "'renamed' already in cfg.json, skipped"
        );
    }

    #[test]
    fn unknown_name_errors_without_network() {
        let dir = std::env::temp_dir().join(format!("bo_dl_{}", uuid::Uuid::new_v4()));
        assert!(download(&["nope".to_string()], false, &dir).is_err());
        assert!(download(&[], false, &dir).is_err());
    }
}
