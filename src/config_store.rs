//! The one in-memory config of the service with a revision number and a short change log.
//!
//! Every change goes through [`ConfigStore::update`] (web UI forms, Benchmark apply, Resources
//! "Add to config", the log level, YOLO26 export) or is picked up from the file
//! ([`ConfigStore::check_disk`]: the CLI or an editor changed it) or a restart
//! ([`ConfigStore::adopt`]). Each change bumps the revision and is logged with its source and a
//! summary, which the pages poll through `GET /v1/config`.
//!
//! The store is shared by every registry generation of the process (restarts keep the revision
//! and the log).

use crate::config::Config;
use crate::config_merge::{FieldChange, describe_changes};
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{RwLock, RwLockReadGuard};
use std::time::SystemTime;

/// Changes kept for `GET /v1/config`.
pub const CHANGE_LOG_LEN: usize = 50;

/// Source of changes found in the file.
pub const SOURCE_DISK: &str = "file changed on disk";

/// One logged change.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChangeRecord {
    pub revision: u64,
    /// RFC 3339.
    pub time: String,
    /// "config page", "benchmark apply", "file changed on disk", ...
    pub source: String,
    /// "IPcam-general device → openvino:cpu, threshold → 0.35".
    pub summary: String,
    pub fields: Vec<FieldChange>,
}

/// File identity used to notice changes made by other processes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DiskStamp {
    modified: Option<SystemTime>,
    len: u64,
    hash: u64,
}

fn hash_bytes(b: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    b.hash(&mut h);
    h.finish()
}

fn stat(path: &Path) -> Option<(Option<SystemTime>, u64)> {
    let md = std::fs::metadata(path).ok()?;
    Some((md.modified().ok(), md.len()))
}

fn stamp_of(path: &Path, bytes: &[u8]) -> Option<DiskStamp> {
    let (modified, len) = stat(path)?;
    Some(DiskStamp {
        modified,
        len,
        hash: hash_bytes(bytes),
    })
}

struct Inner {
    config: Config,
    revision: u64,
    generation: u64,
    changes: VecDeque<ChangeRecord>,
    disk: Option<DiskStamp>,
    disk_error: Option<String>,
}

/// Read access to the current config.
pub struct ConfigRef<'a>(RwLockReadGuard<'a, Inner>);

impl std::ops::Deref for ConfigRef<'_> {
    type Target = Config;
    fn deref(&self) -> &Config {
        &self.0.config
    }
}

/// Outcome of [`ConfigStore::update`].
#[derive(Debug)]
pub struct Updated<R> {
    /// What the edit returned.
    pub value: R,
    /// Revision after the update.
    pub revision: u64,
    /// The config changed (and the revision was bumped).
    pub changed: bool,
    /// The config before the edit.
    pub before: Config,
    /// The logged change, when something changed.
    pub record: Option<ChangeRecord>,
}

pub struct ConfigStore {
    path: PathBuf,
    /// Process start in Unix milliseconds: revisions restart at 1 in a new process.
    epoch: u64,
    inner: RwLock<Inner>,
}

impl ConfigStore {
    /// A store holding `config` (normalized, see [`Config::normalize`]) for the file at `path`.
    pub fn new(mut config: Config, path: PathBuf) -> Self {
        config.normalize();
        let disk = std::fs::read(&path).ok().and_then(|b| stamp_of(&path, &b));
        let epoch = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            path,
            epoch,
            inner: RwLock::new(Inner {
                config,
                revision: 1,
                generation: 1,
                changes: VecDeque::new(),
                disk,
                disk_error: None,
            }),
        }
    }

    fn lock_read(&self) -> RwLockReadGuard<'_, Inner> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_write(&self) -> std::sync::RwLockWriteGuard<'_, Inner> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The current config (hold the guard briefly).
    pub fn read(&self) -> ConfigRef<'_> {
        ConfigRef(self.lock_read())
    }

    /// Revision and a copy of the config, consistent with each other.
    pub fn snapshot(&self) -> (u64, Config) {
        let g = self.lock_read();
        (g.revision, g.config.clone())
    }

    pub fn revision(&self) -> u64 {
        self.lock_read().revision
    }

    /// Registry generation (1 for the first, +1 per restart).
    pub fn generation(&self) -> u64 {
        self.lock_read().generation
    }

    /// The last `n` changes, oldest first.
    pub fn changes(&self, n: usize) -> Vec<ChangeRecord> {
        let g = self.lock_read();
        let skip = g.changes.len().saturating_sub(n);
        g.changes.iter().skip(skip).cloned().collect()
    }

    /// Why the file on disk could not be used (it is not valid JSON, ...), if so.
    pub fn disk_error(&self) -> Option<String> {
        self.lock_read().disk_error.clone()
    }

    fn record(inner: &mut Inner, source: &str, before: &Config) -> Option<ChangeRecord> {
        if *before == inner.config {
            return None;
        }
        inner.revision += 1;
        let (fields, summary) = describe_changes(before, &inner.config);
        let rec = ChangeRecord {
            revision: inner.revision,
            time: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            source: source.to_string(),
            summary,
            fields,
        };
        inner.changes.push_back(rec.clone());
        while inner.changes.len() > CHANGE_LOG_LEN {
            inner.changes.pop_front();
        }
        Some(rec)
    }

    /// Reload the file when another process changed it (call with the write lock held).
    fn sync_disk(&self, inner: &mut Inner) -> Option<ChangeRecord> {
        let (modified, len) = stat(&self.path)?;
        if let Some(d) = &inner.disk
            && d.modified == modified
            && d.len == len
        {
            return None;
        }
        let bytes = std::fs::read(&self.path).ok()?;
        let stamp = stamp_of(&self.path, &bytes)?;
        if inner.disk.as_ref().is_some_and(|d| d.hash == stamp.hash) {
            inner.disk = Some(stamp);
            return None;
        }
        inner.disk = Some(stamp);
        let parsed = std::str::from_utf8(&bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .and_then(|t| serde_json::from_str::<Config>(t).map_err(Into::into));
        match parsed {
            Ok(mut c) => {
                c.normalize();
                inner.disk_error = None;
                if c == inner.config {
                    return None;
                }
                let before = std::mem::replace(&mut inner.config, c);
                let rec = Self::record(inner, SOURCE_DISK, &before);
                if let Some(r) = &rec {
                    tracing::info!(
                        revision = r.revision,
                        "config file changed on disk ({}): {}",
                        self.path.display(),
                        r.summary
                    );
                }
                rec
            }
            Err(e) => {
                let msg = format!("{} is not a valid config: {e:#}", self.path.display());
                tracing::warn!("{msg}; keeping the config in memory");
                inner.disk_error = Some(msg);
                None
            }
        }
    }

    /// Pick up changes other processes made to the file (cheap when it did not change: one
    /// `stat`). Returns the logged change.
    pub fn check_disk(&self) -> Option<ChangeRecord> {
        // Cheap pre-check under the read lock.
        {
            let g = self.lock_read();
            let now = stat(&self.path);
            match (&g.disk, now) {
                (_, None) => return None,
                (Some(d), Some((m, l))) if d.modified == m && d.len == l => return None,
                _ => {}
            }
        }
        let mut g = self.lock_write();
        self.sync_disk(&mut g)
    }

    /// Apply `edit` to a copy of the current config (after picking up external file changes),
    /// write the file and make it current. The revision is bumped and the change logged when
    /// the config changed; the file is written when it changed or does not exist yet. On error
    /// nothing changes.
    pub fn update<R>(
        &self,
        source: &str,
        edit: impl FnOnce(&mut Config) -> Result<R>,
    ) -> Result<Updated<R>> {
        let mut g = self.lock_write();
        self.sync_disk(&mut g);
        let before = g.config.clone();
        let mut new = before.clone();
        let value = edit(&mut new)?;
        new.normalize();
        let changed = new != before;
        if changed || !self.path.exists() {
            new.save(&self.path)?;
            let bytes = std::fs::read(&self.path)
                .with_context(|| format!("reading back {}", self.path.display()))?;
            g.disk = stamp_of(&self.path, &bytes);
            g.disk_error = None;
        }
        g.config = new;
        let record = Self::record(&mut g, source, &before);
        if let Some(r) = &record {
            tracing::info!(
                revision = r.revision,
                source,
                "config changed: {}",
                r.summary
            );
        }
        Ok(Updated {
            value,
            revision: g.revision,
            changed,
            before,
            record,
        })
    }

    /// Start a new registry generation with `config` (a restart reloaded the file or restored
    /// the previous config). The in-memory config becomes `config` when `source` is given;
    /// a difference is logged under `source`.
    pub fn adopt(&self, config: Option<Config>, source: &str) -> Option<ChangeRecord> {
        let mut g = self.lock_write();
        g.generation += 1;
        let mut config = config?;
        config.normalize();
        let before = std::mem::replace(&mut g.config, config);
        if let Ok(bytes) = std::fs::read(&self.path) {
            g.disk = stamp_of(&self.path, &bytes);
        }
        Self::record(&mut g, source, &before)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("bo_store_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d.join("cfg.json")
    }

    fn cfg() -> Config {
        Config {
            models: vec![ModelConfig {
                name: Some("IPcam-general".into()),
                path: "models/IPcam-general.onnx".into(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn update_bumps_revision_and_logs() {
        let path = tmp();
        let s = ConfigStore::new(cfg(), path.clone());
        assert_eq!(s.revision(), 1);
        // No-op edit: file written once (did not exist), no bump.
        let u = s.update("test", |_| Ok(())).unwrap();
        assert!(!u.changed);
        assert_eq!(u.revision, 1);
        assert!(path.exists());
        let u = s
            .update("benchmark apply", |c| {
                c.models[0].device = Some("openvino:cpu".into());
                c.models[0].confidence_threshold = Some(0.3449 + 0.0052);
                Ok(())
            })
            .unwrap();
        assert!(u.changed);
        assert_eq!(u.revision, 2);
        let log = s.changes(10);
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].source, "benchmark apply");
        assert_eq!(
            log[0].summary,
            "IPcam-general device \u{2192} openvino:cpu, threshold \u{2192} 0.35"
        );
        // Stored rounded, in memory and in the file.
        assert_eq!(s.read().models[0].confidence_threshold, Some(0.35));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"confidence_threshold\": 0.35"), "{text}");
        // Our own write is not reported as an external change.
        assert!(s.check_disk().is_none());
        // A failing edit changes nothing.
        assert!(
            s.update("x", |c| -> Result<()> {
                c.port = 1;
                anyhow::bail!("no")
            })
            .is_err()
        );
        assert_eq!(s.revision(), 2);
        assert_eq!(s.read().port, crate::DEFAULT_PORT);
    }

    #[test]
    fn external_file_changes_are_detected() {
        let path = tmp();
        let s = ConfigStore::new(cfg(), path.clone());
        s.update("test", |_| Ok(())).unwrap();
        assert!(s.check_disk().is_none());
        // Another process (the CLI, an editor) rewrites the file.
        let mut other = Config::load(&path).unwrap();
        other.port = 4321;
        // Same length and possibly the same mtime tick: the content hash catches it.
        std::thread::sleep(std::time::Duration::from_millis(20));
        other.save(&path).unwrap();
        let rec = s.check_disk().expect("change detected");
        assert_eq!(rec.source, SOURCE_DISK);
        assert_eq!(rec.revision, 2);
        assert!(
            rec.summary.contains("Port \u{2192} 4321"),
            "{}",
            rec.summary
        );
        assert_eq!(s.read().port, 4321);
        assert!(s.check_disk().is_none());
        // A broken file is reported and ignored.
        std::fs::write(&path, "{ not json").unwrap();
        assert!(s.check_disk().is_none());
        assert!(s.disk_error().is_some());
        assert_eq!(s.read().port, 4321);
        // An update after an external change starts from the file's content.
        let mut other = s.read().clone();
        other.nms_iou = 0.45;
        other.save(&path).unwrap();
        s.update("config page", |c| {
            c.port = 4000;
            Ok(())
        })
        .unwrap();
        let now = Config::load(&path).unwrap();
        assert_eq!((now.port, now.nms_iou), (4000, 0.45));
        let log = s.changes(10);
        assert_eq!(
            log.iter().map(|c| c.source.as_str()).collect::<Vec<_>>(),
            [SOURCE_DISK, SOURCE_DISK, "config page"]
        );
        assert_eq!(s.revision(), 4);
    }

    #[test]
    fn adopt_on_restart() {
        let path = tmp();
        let s = ConfigStore::new(cfg(), path.clone());
        assert_eq!(s.generation(), 1);
        // Resource restart: same config, new generation, no bump.
        assert!(s.adopt(None, "restart").is_none());
        assert_eq!((s.generation(), s.revision()), (2, 1));
        let mut c = cfg();
        c.port = 5000;
        let rec = s.adopt(Some(c), "restart (config file reloaded)").unwrap();
        assert_eq!(rec.revision, 2);
        assert_eq!(s.generation(), 3);
        assert_eq!(s.read().port, 5000);
    }
}
