//! `setup-onnxruntime`: install the pinned ONNX Runtime package(s) for this OS/arch/flavor (the
//! `onnxruntime-<flavor>` catalog resource). By default it goes to
//! `<exe_dir>/onnxruntime/<flavor>/` and becomes the active flavor (`onnxruntime/active.txt`,
//! see `backend::libs`); `--dir` installs flat into that directory instead (point
//! `onnxruntime_dir` at it). A process can load only one ONNX Runtime library, so one flavor is
//! active at a time.
//!
//! A thin wrapper over the download manager: verified download, whitelist-only extraction
//! (`resources::extract`), atomic install, `flavor.txt` + `.installed.json`. Nothing from an
//! archive is executed.

use crate::resources::catalog;
use crate::resources::manager::{self, Job, ManagerOptions};
use crate::setup_openvino::{ArchiveKind, dir_size};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Pinned versions and the flavor type live in the resource catalog.
pub use crate::resources::catalog::{DIRECTML_VERSION, Flavor, Layout, ONNXRUNTIME_VERSION};
/// Default folder name next to the executable.
pub const DIR_NAME: &str = "onnxruntime";
/// File recording the installed flavor.
pub const FLAVOR_FILE: &str = "flavor.txt";

/// `--flavor` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum FlavorChoice {
    /// Pick from the detected hardware.
    #[default]
    Auto,
    Cpu,
    Cuda,
    Directml,
}

/// One archive to download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub url: String,
    pub kind: ArchiveKind,
    pub layout: Layout,
    /// Pinned SHA-256 and size (from the resource catalog).
    pub sha256: &'static str,
    pub size: u64,
}

/// Flavor `--flavor auto` installs on this hardware.
pub fn auto_flavor(hw: &crate::backend::detect::HardwareInfo) -> Flavor {
    use crate::backend::detect::GpuVendor;
    match (hw.os.as_str(), hw.arch.as_str()) {
        ("windows", "x86_64") if hw.has_vendor(GpuVendor::Nvidia) => Flavor::Cuda,
        ("windows", _) => Flavor::DirectMl,
        ("linux", "x86_64") if hw.has_vendor(GpuVendor::Nvidia) => Flavor::Cuda,
        ("macos", "aarch64") => Flavor::CoreMl,
        _ => Flavor::Cpu,
    }
}

/// Resolve `--flavor` to a flavor that exists for `os`/`arch`.
pub fn resolve_flavor(
    hw: &crate::backend::detect::HardwareInfo,
    choice: FlavorChoice,
) -> Result<Flavor> {
    let flavor = match choice {
        FlavorChoice::Auto => auto_flavor(hw),
        FlavorChoice::Cpu => Flavor::Cpu,
        FlavorChoice::Cuda => Flavor::Cuda,
        FlavorChoice::Directml => Flavor::DirectMl,
    };
    packages_for(&hw.os, &hw.arch, flavor)?;
    Ok(flavor)
}

/// Archives to download for `(os, arch, flavor)` (the `onnxruntime-<flavor>` catalog entry); an
/// error when the combination is not shipped.
pub fn packages_for(os: &str, arch: &str, flavor: Flavor) -> Result<Vec<Part>> {
    let res = crate::resources::catalog::onnxruntime(os, arch, flavor).with_context(|| {
        format!(
            "no ONNX Runtime '{}' package for {os}/{arch}",
            flavor.as_str()
        )
    })?;
    res.parts
        .iter()
        .map(|p| {
            Ok(Part {
                url: p.url.to_string(),
                kind: p
                    .archive
                    .with_context(|| format!("catalog part {} is not an archive", p.url))?,
                layout: p.layout,
                sha256: p.sha256,
                size: p.size,
            })
        })
        .collect()
}

/// An install found in a directory (by `VERSION` and `flavor.txt`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledOrt {
    pub version: String,
    pub flavor: String,
}

/// What `setup-onnxruntime` left in `dir`; None when there is no install.
pub fn installed(dir: &Path) -> Option<InstalledOrt> {
    let read = |f: &str| {
        std::fs::read_to_string(dir.join(f))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let flavor = read(FLAVOR_FILE)?;
    Some(InstalledOrt {
        version: read("VERSION").unwrap_or_else(|| "unknown version".to_string()),
        flavor,
    })
}

#[derive(Debug, Clone, Default)]
pub struct SetupOptions {
    /// Destination directory (default `<root>/onnxruntime/<flavor>`, made active).
    pub dest: Option<PathBuf>,
    pub flavor: FlavorChoice,
    /// Keep the downloaded archives in `<root>/.downloads`.
    pub keep_archive: bool,
    /// Download root (lock file, `.downloads/`); default the exe dir.
    pub root: Option<PathBuf>,
}

/// Download and install the runtime; returns `(destination, flavor)`.
pub fn run(opts: &SetupOptions) -> Result<(PathBuf, Flavor)> {
    let hw = crate::backend::detect::hardware();
    let flavor = resolve_flavor(hw, opts.flavor)?;
    let resource = catalog::onnxruntime(&hw.os, &hw.arch, flavor).with_context(|| {
        format!(
            "no ONNX Runtime '{flavor}' package for {}/{}",
            hw.os, hw.arch
        )
    })?;
    let root = opts.root.clone().unwrap_or_else(crate::exe_dir);
    let mut job = match &opts.dest {
        Some(d) => Job::into_dir(resource, d.clone()),
        None => Job {
            activate_ort: true,
            ..Job::new(resource, &root)
        },
    };
    job.keep_downloads = opts.keep_archive;
    let dest = job.target.clone();
    println!(
        "Downloading ONNX Runtime ({flavor}) for {}/{} ({})",
        hw.os,
        hw.arch,
        catalog::format_size(resource.size())
    );
    for p in resource.parts {
        println!("  {}", p.url);
    }
    let mut mopts = ManagerOptions::new(root);
    mopts.on_progress = Some(manager::cli_progress());
    manager::install_all(mopts, vec![job])?;
    println!(
        "ONNX Runtime {ONNXRUNTIME_VERSION} ({flavor}; {:.1} MB) installed at {}{}",
        dir_size(&dest) as f64 / 1e6,
        dest.display(),
        if opts.dest.is_none() {
            " (active flavor)"
        } else {
            ""
        }
    );
    if let Some(note) = crate::resources::extract::install_note(resource) {
        println!("note: {note}");
    }
    Ok((dest, flavor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::detect::{GpuAdapter, GpuVendor, HardwareInfo};

    fn gpu(vendor: GpuVendor) -> GpuAdapter {
        GpuAdapter {
            vendor,
            name: "test".into(),
            vram_mb: 0,
            index: 0,
            discrete: true,
        }
    }

    fn hw(os: &str, arch: &str, vendors: &[GpuVendor]) -> HardwareInfo {
        HardwareInfo::new(os, arch, vendors.iter().map(|v| gpu(*v)).collect())
    }

    #[test]
    fn auto_flavor_follows_hardware() {
        let n = GpuVendor::Nvidia;
        let i = GpuVendor::Intel;
        assert_eq!(auto_flavor(&hw("windows", "x86_64", &[i, n])), Flavor::Cuda);
        assert_eq!(
            auto_flavor(&hw("windows", "x86_64", &[i])),
            Flavor::DirectMl
        );
        assert_eq!(auto_flavor(&hw("windows", "x86_64", &[])), Flavor::DirectMl);
        assert_eq!(auto_flavor(&hw("linux", "x86_64", &[n])), Flavor::Cuda);
        assert_eq!(auto_flavor(&hw("linux", "x86_64", &[i])), Flavor::Cpu);
        assert_eq!(auto_flavor(&hw("linux", "aarch64", &[n])), Flavor::Cpu);
        assert_eq!(auto_flavor(&hw("macos", "aarch64", &[])), Flavor::CoreMl);
    }

    #[test]
    fn packages_per_platform() {
        let v = ONNXRUNTIME_VERSION;
        let one = |os, arch, f| {
            let p = packages_for(os, arch, f).unwrap();
            assert_eq!(p.len(), 1);
            p[0].url.clone()
        };
        assert!(
            one("windows", "x86_64", Flavor::Cuda)
                .ends_with(&format!("onnxruntime-win-x64-gpu-{v}.zip"))
        );
        assert!(
            one("linux", "x86_64", Flavor::Cuda)
                .ends_with(&format!("onnxruntime-linux-x64-gpu-{v}.tgz"))
        );
        assert!(
            one("linux", "x86_64", Flavor::Cpu)
                .ends_with(&format!("onnxruntime-linux-x64-{v}.tgz"))
        );
        assert!(
            one("linux", "aarch64", Flavor::Cpu)
                .ends_with(&format!("onnxruntime-linux-aarch64-{v}.tgz"))
        );
        assert!(
            one("macos", "aarch64", Flavor::CoreMl)
                .ends_with(&format!("onnxruntime-osx-arm64-{v}.tgz"))
        );
        let dml = packages_for("windows", "x86_64", Flavor::DirectMl).unwrap();
        assert_eq!(dml.len(), 2);
        assert!(
            dml[0]
                .url
                .contains("microsoft.ml.onnxruntime.directml.1.24.4.nupkg")
        );
        assert!(dml[1].url.contains("microsoft.ai.directml.1.15.4.nupkg"));
        assert_eq!(dml[1].layout, Layout::NugetDirectMl);
        assert!(packages_for("macos", "aarch64", Flavor::Cuda).is_err());
        assert!(packages_for("linux", "aarch64", Flavor::Cuda).is_err());
        assert!(packages_for("linux", "x86_64", Flavor::DirectMl).is_err());
        assert!(packages_for("freebsd", "x86_64", Flavor::Cpu).is_err());
    }

    #[test]
    fn explicit_flavor_must_exist_for_platform() {
        let mac = hw("macos", "aarch64", &[]);
        assert!(resolve_flavor(&mac, FlavorChoice::Cuda).is_err());
        assert_eq!(
            resolve_flavor(&mac, FlavorChoice::Auto).unwrap(),
            Flavor::CoreMl
        );
        let lin = hw("linux", "x86_64", &[]);
        assert_eq!(
            resolve_flavor(&lin, FlavorChoice::Cuda).unwrap(),
            Flavor::Cuda
        );
    }

    #[test]
    fn flavor_names_round_trip() {
        for f in [Flavor::Cpu, Flavor::Cuda, Flavor::DirectMl, Flavor::CoreMl] {
            assert_eq!(Flavor::parse(f.as_str()), Some(f));
        }
        assert_eq!(Flavor::parse("nope"), None);
    }
}
