//! Download the pinned ONNX Runtime package(s) for this OS/arch/flavor and copy only the shared
//! libraries, flat, into `<exe_dir>/onnxruntime` (or `--dir`), plus a `flavor.txt` naming the
//! installed flavor (`cpu`, `cuda`, `directml` or `coreml`). The ONNX Runtime backend reads that
//! layout (a process can load only one ORT library, so one flavor is installed at a time).
//!
//! Like `setup_openvino`, archive entries are untrusted data: only whitelisted file names are
//! written, to destination names built from our own whitelist (never from the entry path), and
//! nothing from the archive is executed. Download and file-writing helpers are shared with it.

use crate::setup_openvino::{
    ArchiveKind, dir_size, download_to_file, same_dir_link_target, write_entry,
};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
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

/// Library stem of a file name (`onnxruntime.dll` -> `onnxruntime`,
/// `libonnxruntime.so.1.24.4` -> `onnxruntime`, `libonnxruntime.1.24.4.dylib` -> `onnxruntime`).
/// None when the name is not a shared library of the target OS.
fn lib_stem(name: &str, windows: bool) -> Option<&str> {
    if windows {
        return name.strip_suffix(".dll");
    }
    if !(name.contains(".so") || name.ends_with(".dylib")) {
        return None;
    }
    name.strip_prefix("lib")?.split('.').next()
}

/// Libraries every flavor needs.
const CORE_STEMS: &[&str] = &["onnxruntime", "onnxruntime_providers_shared"];
/// Provider libraries of the CUDA/TensorRT package.
const GPU_STEMS: &[&str] = &[
    "onnxruntime_providers_cuda",
    "onnxruntime_providers_tensorrt",
];
const DIRECTML_DLL: &str = "DirectML.dll";

fn stem_allowed(stem: &str, flavor: Flavor) -> bool {
    CORE_STEMS.contains(&stem) || (flavor == Flavor::Cuda && GPU_STEMS.contains(&stem))
}

/// Split an archive entry name into clean components; None for suspicious ones.
fn components(entry: &str) -> Option<Vec<String>> {
    let e = entry.replace('\\', "/");
    let parts: Vec<String> = e
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .map(str::to_string)
        .collect();
    if parts.is_empty() || parts.iter().any(|p| p == ".." || p.contains(':')) {
        return None;
    }
    Some(parts)
}

/// Decide whether an archive entry belongs to the runtime subset. Returns the flat destination
/// file name (taken from the entry's last component, which must itself be whitelisted).
pub fn wanted(layout: Layout, flavor: Flavor, windows: bool, entry: &str) -> Option<String> {
    let parts = components(entry)?;
    let (name, dirs) = parts.split_last()?;
    let dirs: Vec<&str> = dirs.iter().map(String::as_str).collect();
    match layout {
        Layout::OrtRelease => {
            // `<top>/lib/<file>` (some Windows packages use `bin`).
            if dirs.len() != 2 || !matches!(dirs[1], "lib" | "bin") {
                return None;
            }
            let stem = lib_stem(name, windows)?;
            stem_allowed(stem, flavor).then(|| name.clone())
        }
        Layout::NugetOrt => {
            if dirs != ["runtimes", "win-x64", "native"] {
                return None;
            }
            let stem = lib_stem(name, true)?;
            stem_allowed(stem, flavor).then(|| name.clone())
        }
        Layout::NugetDirectMl => {
            (dirs == ["bin", "x64-win"] && name == DIRECTML_DLL).then(|| name.clone())
        }
        Layout::OpenVino | Layout::NvidiaWheel | Layout::File => None,
    }
}

/// True for a file in the destination that a previous install may have left (removed before
/// installing so switching flavors never mixes libraries).
fn is_ours(name: &str, windows: bool) -> bool {
    name == FLAVOR_FILE
        || name == "VERSION"
        || name == DIRECTML_DLL
        || lib_stem(name, windows)
            .is_some_and(|s| CORE_STEMS.contains(&s) || GPU_STEMS.contains(&s))
}

fn clean_dest(dest: &Path, windows: bool) {
    let Ok(rd) = std::fs::read_dir(dest) else {
        return;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let is_file = e
            .file_type()
            .map(|t| t.is_file() || t.is_symlink())
            .unwrap_or(false);
        if is_file && is_ours(&name, windows) {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

fn extract_zip(
    archive: &Path,
    part: &Part,
    flavor: Flavor,
    dest: &Path,
) -> Result<BTreeSet<String>> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("opening archive {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("reading zip {}", archive.display()))?;
    let mut got = BTreeSet::new();
    for i in 0..zip.len() {
        let mut entry = zip
            .by_index(i)
            .with_context(|| format!("reading zip entry #{i} of {}", archive.display()))?;
        if entry.is_dir() || entry.is_symlink() {
            continue;
        }
        let Some(name) = wanted(part.layout, flavor, true, entry.name()) else {
            continue;
        };
        write_entry(dest, &name, &mut entry)
            .with_context(|| format!("extracting {}", entry.name()))?;
        got.insert(name);
    }
    Ok(got)
}

fn extract_tgz(
    archive: &Path,
    part: &Part,
    flavor: Flavor,
    dest: &Path,
) -> Result<BTreeSet<String>> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("opening archive {}", archive.display()))?;
    let gz = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
    let mut tar = tar::Archive::new(gz);
    let mut got = BTreeSet::new();
    // (link file name, target file name in the same directory)
    let mut links: Vec<(String, String)> = Vec::new();
    for entry in tar
        .entries()
        .with_context(|| format!("reading tar {}", archive.display()))?
    {
        let mut entry =
            entry.with_context(|| format!("reading tar entry in {}", archive.display()))?;
        let path = entry.path()?.to_string_lossy().into_owned();
        let Some(name) = wanted(part.layout, flavor, false, &path) else {
            continue;
        };
        let ty = entry.header().entry_type();
        if ty.is_symlink() || ty.is_hard_link() {
            if let Some(target) = entry.link_name()? {
                let t = target.to_string_lossy().into_owned();
                // Both the link and its target must be same-directory whitelisted names.
                let rel = format!("x/{name}");
                if let Some(tn) = same_dir_link_target(&rel, &t, ty.is_hard_link())
                    && lib_stem(&tn, false).is_some_and(|s| stem_allowed(s, flavor))
                {
                    links.push((name, tn));
                } else {
                    tracing::debug!("skipping link {path} -> {t}");
                }
            }
            continue;
        }
        if !ty.is_file() {
            continue;
        }
        write_entry(dest, &name, &mut entry).with_context(|| format!("extracting {path}"))?;
        got.insert(name);
    }
    // Resolve link chains (a -> b -> real file) by repeated passes.
    for _ in 0..4 {
        let mut progressed = false;
        for (name, target) in &links {
            if got.contains(name) || !got.contains(target) {
                continue;
            }
            let link_path = dest.join(name);
            if link_path.symlink_metadata().is_ok() {
                std::fs::remove_file(&link_path).ok();
            }
            #[cfg(unix)]
            std::os::unix::fs::symlink(target, &link_path)
                .with_context(|| format!("creating symlink {}", link_path.display()))?;
            #[cfg(not(unix))]
            std::fs::copy(dest.join(target), &link_path)
                .with_context(|| format!("copying {}", link_path.display()))?;
            got.insert(name.clone());
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    Ok(got)
}

/// Warn about missing expected files; fail when the core library is missing.
fn check(flavor: Flavor, windows: bool, got: &BTreeSet<String>) -> Result<Vec<String>> {
    let has = |stem: &str| got.iter().any(|n| lib_stem(n, windows) == Some(stem));
    if !has("onnxruntime") {
        anyhow::bail!("the archive does not contain the onnxruntime shared library");
    }
    let mut warnings = Vec::new();
    if flavor == Flavor::Cuda {
        for stem in GPU_STEMS {
            if !has(stem) {
                warnings.push(format!("ONNX Runtime package is missing '{stem}'"));
            }
        }
    }
    if flavor == Flavor::DirectMl && !got.contains(DIRECTML_DLL) {
        warnings.push(format!("DirectML package is missing '{DIRECTML_DLL}'"));
    }
    for w in &warnings {
        tracing::warn!("{w}");
        eprintln!("warning: {w}");
    }
    Ok(warnings)
}

/// What the install needs the user to know afterwards.
fn flavor_note(flavor: Flavor) -> Option<&'static str> {
    match flavor {
        Flavor::Cuda => Some(
            "CUDA 12.x and cuDNN 9 runtimes are not bundled; install them separately \
             (or use the OpenVINO runtime).",
        ),
        _ => None,
    }
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
    /// Destination directory (default `<exe_dir>/onnxruntime`).
    pub dest: Option<PathBuf>,
    pub flavor: FlavorChoice,
    /// Keep the downloaded archives (their path is printed).
    pub keep_archive: bool,
}

/// Download and extract the runtime; returns `(destination, flavor)`.
pub fn run(opts: &SetupOptions) -> Result<(PathBuf, Flavor)> {
    let hw = crate::backend::detect::hardware();
    let flavor = resolve_flavor(hw, opts.flavor)?;
    let parts = packages_for(&hw.os, &hw.arch, flavor)?;
    let windows = hw.is_windows();
    let dest = opts
        .dest
        .clone()
        .unwrap_or_else(|| crate::exe_dir().join(DIR_NAME));

    let tmp =
        std::env::temp_dir().join(format!("blue-onyx-prism-setup-ort-{}", std::process::id()));
    std::fs::create_dir_all(&tmp)
        .with_context(|| format!("creating temp dir {}", tmp.display()))?;
    let mut archives = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        let ext = match part.kind {
            ArchiveKind::Zip => "zip",
            ArchiveKind::TarGz => "tgz",
        };
        let path = tmp.join(format!("part{i}.{ext}"));
        println!(
            "Downloading ONNX Runtime ({}) for {}/{}\n  {}",
            flavor.as_str(),
            hw.os,
            hw.arch,
            part.url
        );
        let n = download_to_file(&part.url, &path)?;
        println!("Downloaded {:.1} MB", n as f64 / 1e6);
        archives.push(path);
    }

    std::fs::create_dir_all(&dest)
        .with_context(|| format!("creating destination {}", dest.display()))?;
    clean_dest(&dest, windows);
    println!("Extracting runtime libraries into {}", dest.display());
    let mut got = BTreeSet::new();
    for (part, archive) in parts.iter().zip(&archives) {
        got.extend(match part.kind {
            ArchiveKind::Zip => extract_zip(archive, part, flavor, &dest)?,
            ArchiveKind::TarGz => extract_tgz(archive, part, flavor, &dest)?,
        });
    }
    check(flavor, windows, &got)?;
    std::fs::write(dest.join(FLAVOR_FILE), format!("{}\n", flavor.as_str()))
        .with_context(|| format!("writing {}", dest.join(FLAVOR_FILE).display()))?;
    std::fs::write(dest.join("VERSION"), format!("{ONNXRUNTIME_VERSION}\n"))
        .with_context(|| format!("writing {}", dest.join("VERSION").display()))?;

    if opts.keep_archive {
        println!("Archives kept in {}", tmp.display());
    } else if let Err(e) = std::fs::remove_dir_all(&tmp) {
        tracing::warn!("could not remove temp dir {}: {e}", tmp.display());
    }
    println!(
        "ONNX Runtime {ONNXRUNTIME_VERSION} ({}; {} files, {:.1} MB) installed at {}",
        flavor.as_str(),
        got.len(),
        dir_size(&dest) as f64 / 1e6,
        dest.display()
    );
    if let Some(note) = flavor_note(flavor) {
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

    #[test]
    fn release_whitelist_unix() {
        let w = |f, e: &str| wanted(Layout::OrtRelease, f, false, e);
        let top = "onnxruntime-linux-x64-gpu-1.24.4";
        assert_eq!(
            w(Flavor::Cuda, &format!("{top}/lib/libonnxruntime.so.1.24.4")).as_deref(),
            Some("libonnxruntime.so.1.24.4")
        );
        assert_eq!(
            w(Flavor::Cuda, &format!("{top}/lib/libonnxruntime.so")).as_deref(),
            Some("libonnxruntime.so")
        );
        assert!(
            w(
                Flavor::Cuda,
                &format!("{top}/lib/libonnxruntime_providers_cuda.so")
            )
            .is_some()
        );
        assert!(
            w(
                Flavor::Cuda,
                &format!("{top}/lib/libonnxruntime_providers_tensorrt.so")
            )
            .is_some()
        );
        assert!(
            w(
                Flavor::Cuda,
                &format!("{top}/lib/libonnxruntime_providers_shared.so")
            )
            .is_some()
        );
        // Providers only for the cuda flavor.
        assert!(
            w(
                Flavor::Cpu,
                &format!("{top}/lib/libonnxruntime_providers_cuda.so")
            )
            .is_none()
        );
        // Not libraries / wrong places.
        assert!(
            w(
                Flavor::Cpu,
                &format!("{top}/lib/pkgconfig/libonnxruntime.pc")
            )
            .is_none()
        );
        assert!(w(Flavor::Cpu, &format!("{top}/include/onnxruntime_c_api.h")).is_none());
        assert!(
            w(
                Flavor::Cpu,
                &format!("{top}/testdata/libcustom_op_library.so")
            )
            .is_none()
        );
        assert!(w(Flavor::Cpu, "lib/libonnxruntime.so").is_none());
        assert!(w(Flavor::Cpu, &format!("{top}/lib/../../libonnxruntime.so")).is_none());
        assert!(w(Flavor::Cpu, &format!("{top}/lib/libcustom.so")).is_none());
    }

    #[test]
    fn release_whitelist_mac_and_windows() {
        let top = "onnxruntime-osx-arm64-1.24.4";
        let m = |e: &str| wanted(Layout::OrtRelease, Flavor::CoreMl, false, e);
        assert!(m(&format!("{top}/lib/libonnxruntime.1.24.4.dylib")).is_some());
        assert!(m(&format!("{top}/lib/libonnxruntime.dylib")).is_some());
        assert!(
            m(&format!(
                "{top}/lib/libonnxruntime.1.24.4.dylib.dSYM/Contents/Info.plist"
            ))
            .is_none()
        );
        let w = |f, e: &str| wanted(Layout::OrtRelease, f, true, e);
        let top = "onnxruntime-win-x64-gpu-1.24.4";
        assert_eq!(
            w(Flavor::Cuda, &format!("{top}\\lib\\onnxruntime.dll")).as_deref(),
            Some("onnxruntime.dll")
        );
        assert!(
            w(
                Flavor::Cuda,
                &format!("{top}/lib/onnxruntime_providers_cuda.dll")
            )
            .is_some()
        );
        assert!(w(Flavor::Cuda, &format!("{top}/lib/onnxruntime.lib")).is_none());
        assert!(w(Flavor::Cuda, &format!("{top}/lib/onnxruntime.pdb")).is_none());
        assert!(
            w(
                Flavor::Cpu,
                &format!("{top}/lib/onnxruntime_providers_cuda.dll")
            )
            .is_none()
        );
    }

    #[test]
    fn nuget_whitelist() {
        let o = |e: &str| wanted(Layout::NugetOrt, Flavor::DirectMl, true, e);
        assert!(o("runtimes/win-x64/native/onnxruntime.dll").is_some());
        assert!(o("runtimes/win-x64/native/onnxruntime_providers_shared.dll").is_some());
        assert!(o("runtimes/win-x64/native/onnxruntime.lib").is_none());
        assert!(o("runtimes/win-arm64/native/onnxruntime.dll").is_none());
        let d = |e: &str| wanted(Layout::NugetDirectMl, Flavor::DirectMl, true, e);
        assert_eq!(
            d("bin/x64-win/DirectML.dll").as_deref(),
            Some("DirectML.dll")
        );
        assert!(d("bin/x64-win/DirectML.Debug.dll").is_none());
        assert!(d("bin/arm64-win/DirectML.dll").is_none());
        assert!(d("../bin/x64-win/DirectML.dll").is_none());
    }

    #[test]
    fn check_requires_core_library() {
        let set = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<BTreeSet<_>>();
        assert!(
            check(Flavor::Cpu, false, &set(&["libonnxruntime.so.1"]))
                .unwrap()
                .is_empty()
        );
        assert!(check(Flavor::Cpu, false, &set(&[])).is_err());
        let w = check(Flavor::Cuda, false, &set(&["libonnxruntime.so.1"])).unwrap();
        assert_eq!(w.len(), 2);
        let w = check(Flavor::DirectMl, true, &set(&["onnxruntime.dll"])).unwrap();
        assert_eq!(w.len(), 1);
    }
}
