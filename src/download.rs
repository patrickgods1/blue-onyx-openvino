//! Model downloads (`download-models`, `list-models`): the model resources of the catalog
//! (`resources::catalog::MODELS`, Hugging Face files pinned to a commit with SHA-256) installed
//! through the download manager. File names come only from the static catalog and everything is
//! written strictly under the destination directory.

use crate::config::{Config, ModelConfig};
use crate::resources::catalog::{self, Provides, Resource};
use crate::resources::manager::{self, Job, ManagerOptions};
use anyhow::{Result, anyhow, bail};
use std::path::{Path, PathBuf};

/// The downloadable models.
pub fn catalog() -> &'static [Resource] {
    catalog::MODELS
}

/// Case-insensitive lookup by name (a trailing `.onnx` is accepted).
pub fn find(name: &str) -> Option<&'static Resource> {
    catalog::model(name)
}

/// Catalog name of a model resource.
fn name_of(r: &Resource) -> &'static str {
    r.model_name().unwrap_or(r.id)
}

/// Download the named models (or the whole catalog when `all`) into `dest_dir`, using the
/// download root `root` for the lock (default: the exe dir). Files already present with the
/// pinned hash are kept. Returns the paths of every model file.
pub fn download(names: &[String], all: bool, dest_dir: &Path) -> Result<Vec<PathBuf>> {
    download_with_root(names, all, dest_dir, &crate::exe_dir())
}

/// [`download`] with an explicit download root.
pub fn download_with_root(
    names: &[String],
    all: bool,
    dest_dir: &Path,
    root: &Path,
) -> Result<Vec<PathBuf>> {
    let mut entries: Vec<&'static Resource> = Vec::new();
    if all {
        entries.extend(catalog::MODELS.iter());
    } else {
        for n in names {
            let e = find(n).ok_or_else(|| {
                anyhow!("unknown model '{n}'; run `list-models` to see available names")
            })?;
            if !entries.iter().any(|x| x.id == e.id) {
                entries.push(e);
            }
        }
    }
    if entries.is_empty() {
        bail!("no models selected: pass --name <NAME> or --all");
    }
    let jobs: Vec<Job> = entries
        .iter()
        .map(|r| Job::into_dir(r, dest_dir.to_path_buf()))
        .collect();
    let mut opts = ManagerOptions::new(root);
    opts.on_progress = Some(manager::cli_progress());
    manager::install_all(opts, jobs)?;
    Ok(entries
        .iter()
        .flat_map(|r| r.parts.iter().map(|p| dest_dir.join(p.file_name)))
        .collect())
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
        let Provides::Model { name, family } = entry.provides else {
            continue;
        };
        let yaml = f.with_extension("yaml");
        out.push(ModelConfig {
            name: Some(name.to_string()),
            path: rel(f),
            family,
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
    println!(
        "{:<16} {:<8} {:>7} {:<9} DESCRIPTION",
        "NAME", "FAMILY", "SIZE", "LOCAL"
    );
    for e in catalog::MODELS {
        let present = e.parts.iter().all(|p| dest_dir.join(p.file_name).exists());
        let some = e.parts.iter().any(|p| dest_dir.join(p.file_name).exists());
        let status = if present {
            "yes"
        } else if some {
            "partial"
        } else {
            "no"
        };
        let family = match e.provides {
            Provides::Model { family, .. } => family.to_string(),
            _ => String::new(),
        };
        println!(
            "{:<16} {:<8} {:>7} {:<9} {}",
            name_of(e),
            family,
            catalog::format_size(e.size()),
            status,
            e.description
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelFamilyKind;

    #[test]
    fn catalog_shape() {
        assert_eq!(catalog().len(), 21);
        for e in catalog() {
            // YOLOv5 / RT-DETRv2 ship a class file; D-FINE / RF-DETR use COCO-80.
            let detr = matches!(
                e.provides,
                Provides::Model {
                    family: ModelFamilyKind::Detr | ModelFamilyKind::RfDetr,
                    ..
                }
            );
            assert_eq!(e.parts.len(), if detr { 1 } else { 2 }, "{}", e.id);
            assert!(e.parts[0].file_name.ends_with(".onnx"));
            if !detr {
                assert!(e.parts[1].file_name.ends_with(".yaml"));
            }
            assert!(!e.description.is_empty());
            assert!(
                e.parts
                    .iter()
                    .all(|p| !p.file_name.contains('/') && !p.file_name.contains(".."))
            );
        }
    }

    #[test]
    fn find_is_case_insensitive() {
        assert_eq!(
            find("ipcam-general").unwrap().model_name(),
            Some("IPcam-general")
        );
        assert_eq!(
            find("IPCAM-BIRD.onnx").unwrap().model_name(),
            Some("ipcam-bird")
        );
        assert!(matches!(
            find("RT-DETRV2-S").unwrap().provides,
            Provides::Model {
                family: ModelFamilyKind::RtDetr,
                ..
            }
        ));
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
