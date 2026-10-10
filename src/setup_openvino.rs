//! `setup-openvino`: install the pinned OpenVINO runtime (the `openvino-runtime` catalog
//! resource) into `<exe_dir>/openvino`, preserving the archive's `runtime/...` layout so that
//! `openvino-finder` (via `OPENVINO_INSTALL_DIR`, see `backend::libs`) finds it.
//!
//! A thin wrapper over the download manager (`resources::manager`): verified, resumable
//! download, whitelist-only extraction (`resources::extract`), atomic install and a
//! `.installed.json` manifest. Nothing from the archive is executed.

use crate::resources::catalog;
use crate::resources::manager::{self, Job, ManagerOptions};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Pinned OpenVINO runtime version (the catalog in `resources::catalog` owns the pin).
pub use crate::resources::catalog::{ArchiveKind, OPENVINO_VERSION};

/// Where to get the runtime for one platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Package {
    pub url: &'static str,
    pub kind: ArchiveKind,
    /// Pinned SHA-256 and size (from the resource catalog).
    pub sha256: &'static str,
    pub size: u64,
}

/// Download URL and archive kind for `(std::env::consts::OS, std::env::consts::ARCH)`, from the
/// `openvino-runtime` entries of the resource catalog.
pub fn package_for(os: &str, arch: &str) -> Option<Package> {
    let part = catalog::openvino_runtime(os, arch)?.parts.first()?;
    Some(Package {
        url: part.url,
        kind: part.archive?,
        sha256: part.sha256,
        size: part.size,
    })
}

#[derive(Debug, Clone, Default)]
pub struct SetupOptions {
    /// Destination directory (default `<root>/openvino`).
    pub dest: Option<PathBuf>,
    /// Version label; only the pinned [`OPENVINO_VERSION`] can be downloaded.
    pub version: Option<String>,
    /// Use this local archive instead of downloading.
    pub archive: Option<PathBuf>,
    /// Keep the downloaded archive in `<root>/.downloads`.
    pub keep_archive: bool,
    /// Download root (lock file, `.downloads/`); default the exe dir.
    pub root: Option<PathBuf>,
}

pub(crate) fn dir_size(p: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(p) else {
        return 0;
    };
    rd.filter_map(|e| e.ok())
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map(|m| m.len()).unwrap_or(0),
            _ => 0,
        })
        .sum()
}

/// Download (or take `opts.archive`), install the runtime subset into the destination and
/// return the destination directory.
pub fn run(opts: &SetupOptions) -> Result<PathBuf> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let version = opts.version.as_deref().unwrap_or(OPENVINO_VERSION);
    let root = opts.root.clone().unwrap_or_else(crate::exe_dir);
    let resource = catalog::openvino_runtime(os, arch).with_context(|| {
        format!("no OpenVINO {OPENVINO_VERSION} runtime package known for {os}/{arch}")
    })?;
    let dest = opts
        .dest
        .clone()
        .unwrap_or_else(|| root.join(resource.dest));

    let mut job = Job::into_dir(resource, dest.clone());
    job.keep_downloads = opts.keep_archive;
    match &opts.archive {
        Some(a) => {
            if !a.is_file() {
                anyhow::bail!("archive {} does not exist", a.display());
            }
            let name = a.to_string_lossy().to_ascii_lowercase();
            if !(name.ends_with(".zip") || name.ends_with(".tgz") || name.ends_with(".tar.gz")) {
                anyhow::bail!(
                    "unknown archive type for {} (expected .zip/.tgz)",
                    a.display()
                );
            }
            println!("Using local OpenVINO archive {}", a.display());
            job.local_archives = Some(vec![a.clone()]);
            job.version = Some(version.to_string());
        }
        None => {
            if version != OPENVINO_VERSION {
                anyhow::bail!(
                    "only OpenVINO {OPENVINO_VERSION} can be downloaded (requested {version}); \
                     pass a local archive for other versions"
                );
            }
            if manager::is_installed(resource, &dest) {
                println!(
                    "OpenVINO {OPENVINO_VERSION} is already installed at {}",
                    dest.display()
                );
                return Ok(dest);
            }
            println!(
                "Downloading OpenVINO {OPENVINO_VERSION} for {os}/{arch} ({})\n  {}",
                catalog::format_size(resource.size()),
                resource.parts[0].url
            );
        }
    }
    let mut mopts = ManagerOptions::new(root);
    mopts.on_progress = Some(manager::cli_progress());
    manager::install_all(mopts, vec![job])?;
    println!(
        "OpenVINO {version} runtime ({:.1} MB) installed at {}",
        dir_size(&dest) as f64 / 1e6,
        dest.display()
    );
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_per_platform() {
        assert!(
            package_for("windows", "x86_64")
                .unwrap()
                .url
                .ends_with("_x86_64.zip")
        );
        assert_eq!(
            package_for("linux", "aarch64").unwrap().kind,
            ArchiveKind::TarGz
        );
        assert!(
            package_for("macos", "aarch64")
                .unwrap()
                .url
                .contains("macos_12_6")
        );
        assert!(package_for("macos", "x86_64").is_none());
        for (o, a) in [
            ("windows", "x86_64"),
            ("linux", "x86_64"),
            ("linux", "aarch64"),
            ("macos", "aarch64"),
        ] {
            assert!(package_for(o, a).unwrap().url.contains(OPENVINO_VERSION));
        }
    }

    #[test]
    fn rejects_other_versions_without_an_archive() {
        let err = run(&SetupOptions {
            version: Some("2025.1.0".into()),
            dest: Some(std::env::temp_dir().join("bop-ov-never")),
            ..SetupOptions::default()
        })
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("pass a local archive"),
            "{err:#}"
        );
    }
}
