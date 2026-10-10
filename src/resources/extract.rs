//! Whitelist-only extraction of downloaded archives, shared by the download manager and the
//! `setup-*` commands. Dispatches on the catalog [`Layout`] of each part.
//!
//! Archive entries are untrusted data: only whitelisted file names are written, to destination
//! paths built from our own whitelist (never taken verbatim from the entry path), links are
//! only recreated when both ends are same-directory whitelisted files, and nothing from an
//! archive is executed.
//!
//! Layouts:
//! - [`Layout::OpenVino`]: the toolkit's `runtime/...` libraries, layout preserved (what
//!   `openvino-finder` expects under `OPENVINO_INSTALL_DIR`).
//! - [`Layout::OrtRelease`], [`Layout::NugetOrt`], [`Layout::NugetDirectMl`]: ONNX Runtime and
//!   DirectML shared libraries, flat.
//! - [`Layout::NvidiaWheel`]: CUDA/cuDNN shared libraries from `nvidia/<component>/{bin,lib}/`,
//!   flat.
//! - [`Layout::File`]: not an archive; the manager moves the verified file into place.

use super::catalog::{ArchiveKind, Flavor, Layout, Resource, ResourceKind};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::Path;

/// Target platform of an extraction (the resource's platform, normally this machine's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target<'a> {
    pub os: &'a str,
    pub arch: &'a str,
}

impl Target<'_> {
    fn windows(&self) -> bool {
        self.os == "windows"
    }
}

/// Extract the whitelisted entries of one downloaded `archive` (a part with `layout`) into
/// `dest`. Returns the written paths relative to `dest` (`/`-separated).
pub fn extract_part(
    archive: &Path,
    kind: ArchiveKind,
    layout: Layout,
    flavor: Option<Flavor>,
    target: Target<'_>,
    dest: &Path,
) -> Result<BTreeSet<String>> {
    std::fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;
    let windows = target.windows();
    let flavor = flavor.unwrap_or(Flavor::Cpu);
    match layout {
        Layout::OpenVino => {
            let sel = selection_for(target.os)?;
            let wanted = |e: &str| strip_top(e).and_then(|r| sel.wanted(&r));
            let link = |rel: &str, t: &str, hard: bool| {
                let name = same_dir_link_target(rel, t, hard)?;
                let dir = rel.rsplit_once('/').map_or("", |(d, _)| d);
                Some(format!("{dir}/{name}"))
            };
            match kind {
                ArchiveKind::Zip => extract_zip(archive, dest, wanted),
                ArchiveKind::TarGz => extract_tgz(archive, dest, wanted, link),
            }
        }
        Layout::OrtRelease | Layout::NugetOrt | Layout::NugetDirectMl => {
            let wanted = |e: &str| ort_wanted(layout, flavor, windows, e);
            // Flat destination: the link must name a same-directory, whitelisted library.
            let link = |rel: &str, t: &str, hard: bool| {
                let tn = same_dir_link_target(&format!("x/{rel}"), t, hard)?;
                lib_stem(&tn, windows)
                    .is_some_and(|s| stem_allowed(s, flavor))
                    .then_some(tn)
            };
            match kind {
                ArchiveKind::Zip => extract_zip(archive, dest, wanted),
                ArchiveKind::TarGz => extract_tgz(archive, dest, wanted, link),
            }
        }
        Layout::NvidiaWheel => {
            let wanted = |e: &str| wheel_wanted(windows, e);
            match kind {
                ArchiveKind::Zip => extract_zip(archive, dest, wanted),
                ArchiveKind::TarGz => extract_tgz(archive, dest, wanted, |_, _, _| None),
            }
        }
        Layout::File => anyhow::bail!("{} is not an archive", archive.display()),
    }
}

/// After every part of `resource` is extracted into `dest`: platform fix-ups, sanity checks
/// (errors when the core library is missing, warnings otherwise) and the `VERSION` /
/// `flavor.txt` marker files. Returns the warnings; `got` gains the files written here.
pub fn finalize(
    resource: &Resource,
    version: &str,
    target: Target<'_>,
    dest: &Path,
    got: &mut BTreeSet<String>,
) -> Result<Vec<String>> {
    let warnings = match resource.kind {
        ResourceKind::OpenVinoRuntime => {
            let sel = selection_for(target.os)?;
            mirror_tbb_into_lib_dir(&sel, dest, got)?;
            sel.check(target.arch, got)?
        }
        ResourceKind::OnnxRuntime(flavor) => {
            let w = ort_check(flavor, target.windows(), got)?;
            write_marker(dest, ORT_FLAVOR_FILE, flavor.as_str(), got)?;
            w
        }
        ResourceKind::CudaLibs => wheel_check(target.windows(), dest, got)?,
        ResourceKind::Model => Vec::new(),
    };
    if resource.kind != ResourceKind::Model {
        write_marker(dest, "VERSION", version, got)?;
    }
    Ok(warnings)
}

fn write_marker(dest: &Path, name: &str, value: &str, got: &mut BTreeSet<String>) -> Result<()> {
    std::fs::write(dest.join(name), format!("{value}\n"))
        .with_context(|| format!("writing {}", dest.join(name).display()))?;
    got.insert(name.to_string());
    Ok(())
}

/// What the user should know after installing `resource`.
pub fn install_note(resource: &Resource) -> Option<&'static str> {
    match resource.kind {
        ResourceKind::OnnxRuntime(Flavor::Cuda) => Some(
            "CUDA 12.x and cuDNN 9 runtimes are not part of this package; install them, or \
             fetch `nvidia-cuda-libs` (large, opt-in), or use the OpenVINO runtime.",
        ),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

/// Strip the archive's single top-level folder (`openvino_toolkit_.../`) and normalise slashes.
/// Returns None for entries with suspicious components.
pub(crate) fn strip_top(entry: &str) -> Option<String> {
    let e = entry.replace('\\', "/");
    let mut parts = e.split('/').filter(|p| !p.is_empty() && *p != ".");
    let first = parts.next()?;
    let rest: Vec<&str> = if first == "runtime" {
        std::iter::once(first).chain(parts).collect()
    } else {
        parts.collect()
    };
    if rest.is_empty() || rest.iter().any(|p| *p == ".." || p.contains(':')) {
        return None;
    }
    Some(rest.join("/"))
}

/// File name of the target of a tar link entry at `rel` (top folder already stripped) when the
/// target lives in the same directory; None otherwise. Symlink targets are relative to the
/// link's directory (`libfoo.so -> libfoo.so.2026.4.0`); hard-link targets are full archive
/// paths (`openvino_toolkit_.../runtime/lib/.../libfoo.2026.4.0.dylib`) and go through
/// [`strip_top`] like regular entries.
pub(crate) fn same_dir_link_target(rel: &str, target: &str, hard: bool) -> Option<String> {
    let (dir, _) = rel.rsplit_once('/')?;
    let t = target.replace('\\', "/");
    let t = t.strip_prefix("./").unwrap_or(&t);
    let name = if !t.contains('/') {
        t.to_string()
    } else if hard {
        let stripped = strip_top(t)?;
        let (tdir, name) = stripped.rsplit_once('/')?;
        if tdir != dir {
            return None;
        }
        name.to_string()
    } else {
        return None;
    };
    if name.is_empty() || name.starts_with('.') || name.contains(':') {
        return None;
    }
    Some(name)
}

/// Write `reader` to `dest/rel` through a `.partial` file.
pub(crate) fn write_entry(dest: &Path, rel: &str, reader: &mut dyn Read) -> Result<u64> {
    let out = dest.join(rel);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    let tmp = out.with_extension("partial");
    let mut f =
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let n = std::io::copy(reader, &mut f).with_context(|| format!("writing {}", out.display()))?;
    f.flush()?;
    drop(f);
    if out.symlink_metadata().is_ok() {
        std::fs::remove_file(&out).with_context(|| format!("replacing {}", out.display()))?;
    }
    std::fs::rename(&tmp, &out).with_context(|| format!("renaming to {}", out.display()))?;
    Ok(n)
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

/// Zip driver: regular files whose name `wanted` maps to a destination path.
fn extract_zip(
    archive: &Path,
    dest: &Path,
    wanted: impl Fn(&str) -> Option<String>,
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
        let Some(rel) = wanted(entry.name()) else {
            continue;
        };
        write_entry(dest, &rel, &mut entry)
            .with_context(|| format!("extracting {}", entry.name()))?;
        got.insert(rel);
    }
    Ok(got)
}

/// Tar.gz driver: regular files mapped by `wanted`; links are recreated when `link(rel,
/// target, hard)` maps them to a same-directory destination path that was extracted.
fn extract_tgz(
    archive: &Path,
    dest: &Path,
    wanted: impl Fn(&str) -> Option<String>,
    link: impl Fn(&str, &str, bool) -> Option<String>,
) -> Result<BTreeSet<String>> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("opening archive {}", archive.display()))?;
    let gz = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
    let mut tar = tar::Archive::new(gz);
    let mut got = BTreeSet::new();
    // (link rel path, target rel path in the same directory)
    let mut links: Vec<(String, String)> = Vec::new();
    for entry in tar
        .entries()
        .with_context(|| format!("reading tar {}", archive.display()))?
    {
        let mut entry =
            entry.with_context(|| format!("reading tar entry in {}", archive.display()))?;
        let path = entry.path()?.to_string_lossy().into_owned();
        let Some(rel) = wanted(&path) else {
            continue;
        };
        let ty = entry.header().entry_type();
        if ty.is_symlink() || ty.is_hard_link() {
            if let Some(target) = entry.link_name()? {
                let t = target.to_string_lossy().into_owned();
                match link(&rel, &t, ty.is_hard_link()) {
                    Some(target_rel) => links.push((rel, target_rel)),
                    None => tracing::debug!("skipping link {path} -> {t}"),
                }
            }
            continue;
        }
        if !ty.is_file() {
            continue;
        }
        write_entry(dest, &rel, &mut entry).with_context(|| format!("extracting {path}"))?;
        got.insert(rel);
    }
    // Resolve link chains (a -> b -> real file) by repeated passes.
    for _ in 0..4 {
        let mut progressed = false;
        for (rel, target_rel) in &links {
            if got.contains(rel) || !got.contains(target_rel) {
                continue;
            }
            let link_path = dest.join(rel);
            if link_path.symlink_metadata().is_ok() {
                std::fs::remove_file(&link_path).ok();
            }
            #[cfg(unix)]
            {
                let name = target_rel.rsplit('/').next().unwrap_or(target_rel);
                std::os::unix::fs::symlink(name, &link_path)
                    .with_context(|| format!("creating symlink {}", link_path.display()))?;
            }
            #[cfg(not(unix))]
            std::fs::copy(dest.join(target_rel), &link_path)
                .with_context(|| format!("copying {}", link_path.display()))?;
            got.insert(rel.clone());
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    Ok(got)
}

// ---------------------------------------------------------------------------------------------
// OpenVINO
// ---------------------------------------------------------------------------------------------

/// Which files of the archive make up the runtime on one OS.
pub(crate) struct Selection {
    os: &'static str,
    /// Directory of the OpenVINO libs inside the archive (relative to the top folder).
    lib_dirs: &'static [&'static str],
    tbb_dir: &'static str,
}

/// Libraries copied on every OS (base names without prefix/suffix) when the archive has them.
/// Plugins on macOS use a `.so` suffix while the other libs are `.dylib`; both are matched.
const CORE_LIBS: &[&str] = &[
    "openvino",
    "openvino_c",
    CPU_PLUGIN_X86,
    CPU_PLUGIN_ARM,
    GPU_PLUGIN,
    "openvino_auto_plugin",
    "openvino_hetero_plugin",
    "openvino_ir_frontend",
    "openvino_onnx_frontend",
];
/// x86 CPU plugin (Windows, Linux x86_64).
const CPU_PLUGIN_X86: &str = "openvino_intel_cpu_plugin";
/// ARM CPU plugin (macOS arm64, Linux aarch64).
const CPU_PLUGIN_ARM: &str = "openvino_arm_cpu_plugin";
/// Intel GPU plugin (only shipped for Windows and Linux x86_64).
pub(crate) const GPU_PLUGIN: &str = "openvino_intel_gpu_plugin";
/// Libraries whose absence only deserves a warning (the CPU and GPU plugins are checked
/// separately because which ones exist depends on the platform).
const EXPECTED_LIBS: &[&str] = &[
    "openvino",
    "openvino_auto_plugin",
    "openvino_hetero_plugin",
    "openvino_ir_frontend",
    "openvino_onnx_frontend",
];
/// Plugin registry files. Shipped in the Windows archive; 2026.x macOS/Linux archives do not
/// have them (OpenVINO then discovers plugins by file name), so they are optional there.
const EXTRA_FILES: &[&str] = &["plugins.xml", "cache.json"];
const WINDOWS_TBB: &[&str] = &["tbb12.dll", "tbbbind_2_5.dll"];

pub(crate) fn selection_for(os: &str) -> Result<Selection> {
    Ok(match os {
        "windows" => Selection {
            os: "windows",
            lib_dirs: &["runtime/bin/intel64/Release"],
            tbb_dir: "runtime/3rdparty/tbb/bin",
        },
        "linux" => Selection {
            os: "linux",
            lib_dirs: &[
                "runtime/lib/intel64",
                "runtime/lib/aarch64",
                "runtime/lib/arm64",
            ],
            tbb_dir: "runtime/3rdparty/tbb/lib",
        },
        "macos" => Selection {
            os: "macos",
            lib_dirs: &["runtime/lib/arm64/Release", "runtime/lib/intel64/Release"],
            tbb_dir: "runtime/3rdparty/tbb/lib",
        },
        other => anyhow::bail!("unsupported OS '{other}' for OpenVINO runtime setup"),
    })
}

/// Base library name of a unix shared-library file name: `libopenvino_c.so.2026.4.0` ->
/// `openvino_c`, `libopenvino.2026.4.0.dylib` -> `openvino`. None if not a shared library.
pub(crate) fn unix_lib_base(name: &str) -> Option<&str> {
    let is_lib = name.contains(".so") || name.ends_with(".dylib");
    if !is_lib {
        return None;
    }
    let stem = name.strip_prefix("lib")?;
    stem.split('.').next()
}

impl Selection {
    /// Decide whether `rel` (path inside the archive with the top folder stripped, `/`-separated)
    /// belongs to the runtime subset. Returns the (validated) relative destination path.
    pub(crate) fn wanted(&self, rel: &str) -> Option<String> {
        let (dir, name) = rel.rsplit_once('/')?;
        if name.is_empty() || name.contains('\\') {
            return None;
        }
        if self.lib_dirs.contains(&dir) {
            let keep = if self.os == "windows" {
                EXTRA_FILES.contains(&name)
                    || name
                        .strip_suffix(".dll")
                        .is_some_and(|b| CORE_LIBS.contains(&b))
            } else {
                EXTRA_FILES.contains(&name)
                    || unix_lib_base(name).is_some_and(|b| CORE_LIBS.contains(&b))
            };
            return keep.then(|| format!("{dir}/{name}"));
        }
        if dir == self.tbb_dir {
            let keep = if self.os == "windows" {
                WINDOWS_TBB.contains(&name)
            } else {
                // libhwloc is a dependency of libtbbbind.
                (name.starts_with("libtbb") || name.starts_with("libhwloc"))
                    && !name.contains("debug")
                    && unix_lib_base(name).is_some()
            };
            return keep.then(|| format!("{dir}/{name}"));
        }
        None
    }

    /// Warn about missing expected files; fail if `openvino_c` is missing. Returns the
    /// warnings (also logged) so tests can inspect them.
    pub(crate) fn check(&self, arch: &str, got: &BTreeSet<String>) -> Result<Vec<String>> {
        let names: Vec<&str> = got
            .iter()
            .filter_map(|p| p.rsplit_once('/').map(|(_, n)| n))
            .collect();
        let has_lib = |base: &str| {
            names.iter().any(|n| {
                if self.os == "windows" {
                    n.strip_suffix(".dll") == Some(base)
                } else {
                    unix_lib_base(n) == Some(base)
                }
            })
        };
        if !has_lib("openvino_c") {
            anyhow::bail!(
                "the archive does not contain the openvino_c library under {:?}",
                self.lib_dirs
            );
        }
        let mut warnings = Vec::new();
        for base in EXPECTED_LIBS {
            if !has_lib(base) {
                warnings.push(format!("OpenVINO archive is missing library '{base}'"));
            }
        }
        if !has_lib(CPU_PLUGIN_X86) && !has_lib(CPU_PLUGIN_ARM) {
            warnings.push(format!(
                "OpenVINO archive has no CPU plugin ('{CPU_PLUGIN_X86}' or '{CPU_PLUGIN_ARM}'); \
                 CPU inference will not work"
            ));
        }
        let gpu_expected = matches!((self.os, arch), ("windows", "x86_64") | ("linux", "x86_64"));
        if gpu_expected && !has_lib(GPU_PLUGIN) {
            warnings.push(format!(
                "OpenVINO archive is missing library '{GPU_PLUGIN}'; GPU inference will not work"
            ));
        }
        let tbb: &[&str] = if self.os == "windows" {
            WINDOWS_TBB
        } else {
            &[]
        };
        for f in tbb {
            if !names.contains(f) {
                warnings.push(format!("OpenVINO archive is missing '{f}' (continuing)"));
            }
        }
        for f in EXTRA_FILES {
            if !names.contains(f) {
                if self.os == "windows" {
                    warnings.push(format!("OpenVINO archive is missing '{f}' (continuing)"));
                } else {
                    tracing::debug!("OpenVINO archive has no '{f}' (not needed on {})", self.os);
                }
            }
        }
        if self.os != "windows" && !names.iter().any(|n| n.starts_with("libtbb")) {
            warnings.push(
                "OpenVINO archive has no bundled TBB; the system libtbb will be used".to_string(),
            );
        }
        for w in &warnings {
            tracing::warn!("{w}");
        }
        Ok(warnings)
    }
}

/// On macOS, `(source rel, destination rel)` pairs that copy the bundled TBB/hwloc libraries
/// next to `libopenvino_c` (empty elsewhere).
///
/// `libopenvino.dylib` and the plugins reference `@rpath/libtbb.12.dylib`; the only LC_RPATH of
/// libopenvino is `@loader_path/../../../3rdparty/tbb` (not the `tbb/lib` folder the archive
/// actually uses), while `libopenvino_c` has `@loader_path/`. Having TBB in the lib dir makes
/// `@rpath` resolve through `libopenvino_c` (which is what we load first).
fn tbb_mirror_plan(sel: &Selection, got: &BTreeSet<String>) -> Vec<(String, String)> {
    if sel.os != "macos" {
        return Vec::new();
    }
    let Some(lib_dir) = got.iter().find_map(|rel| {
        let (dir, name) = rel.rsplit_once('/')?;
        (sel.lib_dirs.contains(&dir) && unix_lib_base(name) == Some("openvino_c")).then_some(dir)
    }) else {
        return Vec::new();
    };
    let prefix = format!("{}/", sel.tbb_dir);
    got.iter()
        .filter_map(|rel| {
            let name = rel.strip_prefix(&prefix)?;
            (!name.contains('/')).then(|| (rel.clone(), format!("{lib_dir}/{name}")))
        })
        .collect()
}

/// Apply [`tbb_mirror_plan`]: regular files are copied, symlinks recreated with the same
/// (same-directory) target.
fn mirror_tbb_into_lib_dir(sel: &Selection, dest: &Path, got: &mut BTreeSet<String>) -> Result<()> {
    let plan = tbb_mirror_plan(sel, got);
    // Regular files first so symlinks never dangle, even transiently.
    let is_link = |rel: &str| {
        dest.join(rel)
            .symlink_metadata()
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false)
    };
    let (links, files): (Vec<_>, Vec<_>) = plan.into_iter().partition(|(src, _)| is_link(src));
    for (src, dst) in files.iter().chain(links.iter()) {
        let from = dest.join(src);
        let to = dest.join(dst);
        if to.symlink_metadata().is_ok() {
            std::fs::remove_file(&to).with_context(|| format!("replacing {}", to.display()))?;
        }
        let link_target = std::fs::read_link(&from).ok();
        match link_target {
            #[cfg(unix)]
            Some(target) => std::os::unix::fs::symlink(&target, &to)
                .with_context(|| format!("creating symlink {}", to.display()))?,
            _ => {
                std::fs::copy(&from, &to)
                    .with_context(|| format!("copying {} to {}", from.display(), to.display()))?;
            }
        }
        got.insert(dst.clone());
    }
    if !files.is_empty() {
        tracing::debug!(
            "installed {} TBB libraries next to libopenvino for @rpath resolution",
            files.len() + links.len()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// ONNX Runtime / DirectML
// ---------------------------------------------------------------------------------------------

/// File naming the flavor in an ONNX Runtime install directory.
pub const ORT_FLAVOR_FILE: &str = "flavor.txt";

/// Library stem of a file name (`onnxruntime.dll` -> `onnxruntime`,
/// `libonnxruntime.so.1.24.4` -> `onnxruntime`, `libonnxruntime.1.24.4.dylib` -> `onnxruntime`).
/// None when the name is not a shared library of the target OS.
pub(crate) fn lib_stem(name: &str, windows: bool) -> Option<&str> {
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

/// Decide whether an ONNX Runtime / DirectML archive entry belongs to the runtime subset.
/// Returns the flat destination file name (the entry's last component, itself whitelisted).
pub fn ort_wanted(layout: Layout, flavor: Flavor, windows: bool, entry: &str) -> Option<String> {
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

/// True for a file a previous ONNX Runtime install may have left in a directory (removed before
/// installing into a shared directory so flavors never mix).
pub(crate) fn is_ort_file(name: &str, windows: bool) -> bool {
    name == ORT_FLAVOR_FILE
        || name == "VERSION"
        || name == DIRECTML_DLL
        || lib_stem(name, windows)
            .is_some_and(|s| CORE_STEMS.contains(&s) || GPU_STEMS.contains(&s))
}

/// Warn about missing expected files; fail when the core library is missing.
pub(crate) fn ort_check(
    flavor: Flavor,
    windows: bool,
    got: &BTreeSet<String>,
) -> Result<Vec<String>> {
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
    }
    Ok(warnings)
}

// ---------------------------------------------------------------------------------------------
// NVIDIA wheels
// ---------------------------------------------------------------------------------------------

/// `nvidia/<component>/bin/<x>.dll` (Windows) or `nvidia/<component>/lib/lib<x>.so[.N]`
/// (Linux) -> the flat file name.
pub fn wheel_wanted(windows: bool, entry: &str) -> Option<String> {
    let parts = components(entry)?;
    let (name, dirs) = parts.split_last()?;
    let [top, component, sub] = dirs else {
        return None;
    };
    let component_ok = !component.is_empty()
        && component
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_');
    if top != "nvidia" || !component_ok {
        return None;
    }
    let is_lib = if windows {
        sub == "bin" && name.to_ascii_lowercase().ends_with(".dll")
    } else {
        sub == "lib" && name.starts_with("lib") && name.contains(".so")
    };
    (is_lib && !name.starts_with('.')).then(|| name.clone())
}

/// Exact library names ONNX Runtime 1.24's CUDA provider links (`DT_NEEDED` / imports, checked
/// on `libonnxruntime_providers_cuda.so`): they must exist after extraction.
pub fn cuda_provider_needs(windows: bool) -> &'static [&'static str] {
    if windows {
        &[
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cudnn64_9.dll",
            "cufft64_11.dll",
            "curand64_10.dll",
        ]
    } else {
        &[
            "libcudart.so.12",
            "libcublas.so.12",
            "libcublasLt.so.12",
            "libcudnn.so.9",
            "libcufft.so.11",
            "libcurand.so.10",
        ]
    }
}

/// Soname links missing among extracted Linux libraries: `libfoo.so.12.8.90` without
/// `libfoo.so.12` gives `("libfoo.so.12", "libfoo.so.12.8.90")` (the shortest longer name wins
/// when there are several). The NVIDIA wheels normally ship the soname itself.
pub fn soname_links(names: &BTreeSet<String>) -> Vec<(String, String)> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for n in names {
        let Some((base, ver)) = n.split_once(".so.") else {
            continue;
        };
        let mut parts = ver.split('.');
        let Some(major) = parts
            .next()
            .filter(|m| m.chars().all(|c| c.is_ascii_digit()))
        else {
            continue;
        };
        if parts.next().is_none() || !base.starts_with("lib") || base.contains('/') {
            continue;
        }
        let link = format!("{base}.so.{major}");
        if names.contains(&link) {
            continue;
        }
        match out.get(&link) {
            Some(t) if t.len() <= n.len() => {}
            _ => {
                out.insert(link, n.clone());
            }
        }
    }
    out.into_iter().collect()
}

/// Create the [`soname_links`] in `dest` (symlinks on Unix, copies elsewhere).
fn add_soname_links(dest: &Path, got: &mut BTreeSet<String>) -> Result<()> {
    for (link, target) in soname_links(got) {
        let path = dest.join(&link);
        if path.symlink_metadata().is_ok() {
            std::fs::remove_file(&path).ok();
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &path)
            .with_context(|| format!("creating symlink {}", path.display()))?;
        #[cfg(not(unix))]
        std::fs::copy(dest.join(&target), &path)
            .with_context(|| format!("copying {}", path.display()))?;
        got.insert(link);
    }
    Ok(())
}

/// Linux: add soname links; then every library the CUDA provider links must be there.
fn wheel_check(windows: bool, dest: &Path, got: &mut BTreeSet<String>) -> Result<Vec<String>> {
    if !windows {
        add_soname_links(dest, got)?;
    }
    let missing: Vec<&str> = cuda_provider_needs(windows)
        .iter()
        .copied()
        .filter(|n| !got.contains(*n))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!("the CUDA wheels are missing {}", missing.join(", "));
    }
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_selection() {
        let s = selection_for("windows").unwrap();
        let top = "openvino_toolkit_windows_vc_mt_2026.4.0.22959.99c81491cc3_x86_64/";
        let w = |p: &str| strip_top(&format!("{top}{p}")).and_then(|r| s.wanted(&r));
        assert_eq!(
            w("runtime/bin/intel64/Release/openvino_c.dll").as_deref(),
            Some("runtime/bin/intel64/Release/openvino_c.dll")
        );
        assert!(w("runtime/bin/intel64/Release/cache.json").is_some());
        assert!(w("runtime/3rdparty/tbb/bin/tbb12.dll").is_some());
        assert!(w("runtime/3rdparty/tbb/bin/tbb12_debug.dll").is_none());
        assert!(w("runtime/bin/intel64/Debug/openvino_cd.dll").is_none());
        assert!(w("runtime/bin/intel64/Release/openvino_intel_npu_plugin.dll").is_none());
        assert!(w("runtime/bin/intel64/Release/openvino_pytorch_frontend.dll").is_none());
        assert!(w("runtime/bin/intel64/Release/../../../evil.dll").is_none());
    }

    #[test]
    fn unix_selection() {
        let s = selection_for("linux").unwrap();
        let w = |p: &str| strip_top(&format!("top/{p}")).and_then(|r| s.wanted(&r));
        assert!(w("runtime/lib/intel64/libopenvino_c.so.2026.4.0").is_some());
        assert!(w("runtime/lib/intel64/libopenvino_c.so").is_some());
        assert!(w("runtime/lib/intel64/libopenvino_intel_npu_plugin.so").is_none());
        assert!(w("runtime/3rdparty/tbb/lib/libtbb.so.12").is_some());
        assert!(w("runtime/3rdparty/tbb/lib/libtbb_debug.so.12").is_none());
        let m = selection_for("macos").unwrap();
        assert!(
            strip_top("top/runtime/lib/arm64/Release/libopenvino.2026.4.0.dylib")
                .and_then(|r| m.wanted(&r))
                .is_some()
        );
        assert_eq!(
            unix_lib_base("libopenvino.2026.4.0.dylib"),
            Some("openvino")
        );
        assert_eq!(unix_lib_base("libopenvino_c.so.2640"), Some("openvino_c"));
        // macOS plugins are `.so`, the ARM CPU plugin is selected; hwloc (tbbbind dep) too.
        let mw = |p: &str| strip_top(&format!("top/{p}")).and_then(|r| m.wanted(&r));
        assert!(mw("runtime/lib/arm64/Release/libopenvino_arm_cpu_plugin.so").is_some());
        assert!(mw("runtime/lib/arm64/Release/libopenvino_auto_batch_plugin.so").is_none());
        assert!(mw("runtime/lib/arm64/Release/libopenvino_pytorch_frontend.2640.dylib").is_none());
        assert!(mw("runtime/3rdparty/tbb/lib/libhwloc.15.dylib").is_some());
        assert!(mw("runtime/3rdparty/tbb/lib/pkgconfig/tbb.pc").is_none());
        assert!(w("runtime/lib/aarch64/libopenvino_arm_cpu_plugin.so").is_some());
    }

    #[test]
    fn link_targets() {
        let rel = "runtime/lib/arm64/Release/libopenvino.2640.dylib";
        // Symlinks: bare same-dir names only.
        assert_eq!(
            same_dir_link_target(rel, "libopenvino.2026.4.0.dylib", false).as_deref(),
            Some("libopenvino.2026.4.0.dylib")
        );
        assert_eq!(same_dir_link_target(rel, "../x.dylib", false), None);
        assert_eq!(same_dir_link_target(rel, "/usr/lib/x.dylib", false), None);
        // Hard links: full archive path, mapped through strip_top and required in the same dir.
        assert_eq!(
            same_dir_link_target(
                rel,
                "openvino_toolkit_macos_12_6_2026.4.0_arm64/runtime/lib/arm64/Release/libopenvino.2026.4.0.dylib",
                true
            )
            .as_deref(),
            Some("libopenvino.2026.4.0.dylib")
        );
        assert_eq!(
            same_dir_link_target(rel, "top/runtime/3rdparty/tbb/lib/libtbb.12.dylib", true),
            None
        );
        assert_eq!(
            same_dir_link_target(rel, "top/runtime/lib/arm64/Release/../../../../evil", true),
            None
        );
    }

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn check_warnings_per_platform() {
        let mac = set(&[
            "runtime/lib/arm64/Release/libopenvino.2640.dylib",
            "runtime/lib/arm64/Release/libopenvino_c.2640.dylib",
            "runtime/lib/arm64/Release/libopenvino_arm_cpu_plugin.so",
            "runtime/lib/arm64/Release/libopenvino_auto_plugin.so",
            "runtime/lib/arm64/Release/libopenvino_hetero_plugin.so",
            "runtime/lib/arm64/Release/libopenvino_ir_frontend.2640.dylib",
            "runtime/lib/arm64/Release/libopenvino_onnx_frontend.2640.dylib",
            "runtime/3rdparty/tbb/lib/libtbb.12.dylib",
        ]);
        let m = selection_for("macos").unwrap();
        assert!(m.check("aarch64", &mac).unwrap().is_empty());
        // Without any CPU plugin there is a warning.
        let mut no_cpu = mac.clone();
        no_cpu.remove("runtime/lib/arm64/Release/libopenvino_arm_cpu_plugin.so");
        let w = m.check("aarch64", &no_cpu).unwrap();
        assert!(w.len() == 1 && w[0].contains("CPU plugin"), "{w:?}");
        // openvino_c is mandatory.
        let mut no_c = mac.clone();
        no_c.remove("runtime/lib/arm64/Release/libopenvino_c.2640.dylib");
        assert!(m.check("aarch64", &no_c).is_err());

        // Linux x86_64 expects the GPU plugin; aarch64 does not.
        let lin = set(&[
            "runtime/lib/intel64/libopenvino.so.2640",
            "runtime/lib/intel64/libopenvino_c.so.2640",
            "runtime/lib/intel64/libopenvino_intel_cpu_plugin.so",
            "runtime/lib/intel64/libopenvino_auto_plugin.so",
            "runtime/lib/intel64/libopenvino_hetero_plugin.so",
            "runtime/lib/intel64/libopenvino_ir_frontend.so.2640",
            "runtime/lib/intel64/libopenvino_onnx_frontend.so.2640",
            "runtime/3rdparty/tbb/lib/libtbb.so.12",
        ]);
        let l = selection_for("linux").unwrap();
        let w = l.check("x86_64", &lin).unwrap();
        assert!(w.len() == 1 && w[0].contains(GPU_PLUGIN), "{w:?}");
        assert!(l.check("aarch64", &lin).unwrap().is_empty());
    }

    #[test]
    fn macos_tbb_is_mirrored_next_to_libopenvino() {
        let got = set(&[
            "runtime/lib/arm64/Release/libopenvino_c.2640.dylib",
            "runtime/3rdparty/tbb/lib/libtbb.12.13.dylib",
            "runtime/3rdparty/tbb/lib/libtbb.12.dylib",
        ]);
        let plan = tbb_mirror_plan(&selection_for("macos").unwrap(), &got);
        assert_eq!(
            plan,
            vec![
                (
                    "runtime/3rdparty/tbb/lib/libtbb.12.13.dylib".to_string(),
                    "runtime/lib/arm64/Release/libtbb.12.13.dylib".to_string()
                ),
                (
                    "runtime/3rdparty/tbb/lib/libtbb.12.dylib".to_string(),
                    "runtime/lib/arm64/Release/libtbb.12.dylib".to_string()
                ),
            ]
        );
        assert!(tbb_mirror_plan(&selection_for("linux").unwrap(), &got).is_empty());
    }

    #[test]
    fn ort_release_whitelist_unix() {
        let w = |f, e: &str| ort_wanted(Layout::OrtRelease, f, false, e);
        let top = "onnxruntime-linux-x64-gpu-1.24.4";
        assert_eq!(
            w(Flavor::Cuda, &format!("{top}/lib/libonnxruntime.so.1.24.4")).as_deref(),
            Some("libonnxruntime.so.1.24.4")
        );
        assert_eq!(
            w(Flavor::Cuda, &format!("{top}/lib/libonnxruntime.so")).as_deref(),
            Some("libonnxruntime.so")
        );
        for p in ["cuda", "tensorrt", "shared"] {
            let e = format!("{top}/lib/libonnxruntime_providers_{p}.so");
            assert!(w(Flavor::Cuda, &e).is_some(), "{e}");
        }
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
    fn ort_release_whitelist_mac_and_windows() {
        let top = "onnxruntime-osx-arm64-1.24.4";
        let m = |e: &str| ort_wanted(Layout::OrtRelease, Flavor::CoreMl, false, e);
        assert!(m(&format!("{top}/lib/libonnxruntime.1.24.4.dylib")).is_some());
        assert!(m(&format!("{top}/lib/libonnxruntime.dylib")).is_some());
        assert!(
            m(&format!(
                "{top}/lib/libonnxruntime.1.24.4.dylib.dSYM/Contents/Info.plist"
            ))
            .is_none()
        );
        let w = |f, e: &str| ort_wanted(Layout::OrtRelease, f, true, e);
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
        let o = |e: &str| ort_wanted(Layout::NugetOrt, Flavor::DirectMl, true, e);
        assert!(o("runtimes/win-x64/native/onnxruntime.dll").is_some());
        assert!(o("runtimes/win-x64/native/onnxruntime_providers_shared.dll").is_some());
        assert!(o("runtimes/win-x64/native/onnxruntime.lib").is_none());
        assert!(o("runtimes/win-arm64/native/onnxruntime.dll").is_none());
        let d = |e: &str| ort_wanted(Layout::NugetDirectMl, Flavor::DirectMl, true, e);
        assert_eq!(
            d("bin/x64-win/DirectML.dll").as_deref(),
            Some("DirectML.dll")
        );
        assert!(d("bin/x64-win/DirectML.Debug.dll").is_none());
        assert!(d("bin/arm64-win/DirectML.dll").is_none());
        assert!(d("../bin/x64-win/DirectML.dll").is_none());
    }

    #[test]
    fn ort_check_requires_core_library() {
        assert!(
            ort_check(Flavor::Cpu, false, &set(&["libonnxruntime.so.1"]))
                .unwrap()
                .is_empty()
        );
        assert!(ort_check(Flavor::Cpu, false, &set(&[])).is_err());
        let w = ort_check(Flavor::Cuda, false, &set(&["libonnxruntime.so.1"])).unwrap();
        assert_eq!(w.len(), 2);
        let w = ort_check(Flavor::DirectMl, true, &set(&["onnxruntime.dll"])).unwrap();
        assert_eq!(w.len(), 1);
        assert!(is_ort_file("libonnxruntime_providers_cuda.so", false));
        assert!(!is_ort_file("libcudnn.so.9", false));
    }

    #[test]
    fn nvidia_wheel_whitelist() {
        let w = |e: &str| wheel_wanted(true, e);
        assert_eq!(
            w("nvidia/cudnn/bin/cudnn64_9.dll").as_deref(),
            Some("cudnn64_9.dll")
        );
        assert!(w("nvidia/cublas/bin/cublasLt64_12.dll").is_some());
        assert!(w("nvidia/cudnn/include/cudnn.h").is_none());
        assert!(w("nvidia/cudnn/lib/x64/cudnn.lib").is_none());
        assert!(w("nvidia_cudnn_cu12-9.8.0.87.dist-info/RECORD").is_none());
        assert!(w("nvidia/../bin/evil.dll").is_none());
        let l = |e: &str| wheel_wanted(false, e);
        assert_eq!(
            l("nvidia/cudnn/lib/libcudnn.so.9").as_deref(),
            Some("libcudnn.so.9")
        );
        assert!(l("nvidia/cuda_runtime/lib/libcudart.so.12").is_some());
        assert!(l("nvidia/cuda_runtime/lib/libcudart_static.a").is_none());
        assert!(l("nvidia/cudnn/bin/libcudnn.so.9").is_none());
        let dir = std::env::temp_dir().join(format!("bop-wheelchk-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut got = set(&["libcudart.so.12", "libcublas.so.12"]);
        let err = wheel_check(false, &dir, &mut got).unwrap_err();
        assert!(format!("{err}").contains("libcudnn.so.9"), "{err}");
        let mut got: BTreeSet<String> = cuda_provider_needs(true)
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(wheel_check(true, &dir, &mut got).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn soname_links_fill_in_missing_names() {
        let names = set(&[
            "libcudart.so.12.8.90",
            "libcudnn.so.9",
            "libcudnn.so.9.8.0",
            "libnvrtc.so.12.8.93",
            "libnvrtc.so.12.8",
            "libweird.so.x.1",
            "cudart64_12.dll",
        ]);
        assert_eq!(
            soname_links(&names),
            [
                (
                    "libcudart.so.12".to_string(),
                    "libcudart.so.12.8.90".to_string()
                ),
                ("libnvrtc.so.12".to_string(), "libnvrtc.so.12.8".to_string()),
            ]
        );
        assert!(soname_links(&set(&["libcudnn.so.9"])).is_empty());
    }

    /// Write a zip (a wheel) with `entries` (name, bytes).
    fn zip_with(path: &Path, entries: &[(&str, &[u8])]) {
        let f = std::fs::File::create(path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        for (name, data) in entries {
            z.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            z.write_all(data).unwrap();
        }
        z.finish().unwrap();
    }

    /// Synthetic Linux wheels: only shared libraries land (flat), headers / static libs /
    /// metadata do not, and a missing soname gets its link.
    #[test]
    fn linux_wheels_extract_flat_with_soname_links() {
        let dir = std::env::temp_dir().join(format!("bop-wheel-l-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let w1 = dir.join("rt.whl");
        zip_with(
            &w1,
            &[
                ("nvidia/__init__.py", b""),
                ("nvidia/cuda_runtime/include/cuda.h", b"h"),
                ("nvidia/cuda_runtime/lib/libcudart.so.12.8.90", b"cudart"),
                ("nvidia/cuda_runtime/lib/libcudart_static.a", b"a"),
                ("nvidia_cuda_runtime_cu12-12.8.90.dist-info/RECORD", b"r"),
            ],
        );
        let w2 = dir.join("rest.whl");
        zip_with(
            &w2,
            &[
                ("nvidia/cublas/lib/libcublas.so.12", b"cublas"),
                ("nvidia/cublas/lib/libcublasLt.so.12", b"lt"),
                ("nvidia/cudnn/lib/libcudnn.so.9", b"cudnn"),
                ("nvidia/cudnn/lib/libcudnn_graph.so.9", b"graph"),
                ("nvidia/cudnn/include/cudnn.h", b"h"),
                ("nvidia/cufft/lib/libcufft.so.11", b"fft"),
                ("nvidia/curand/lib/libcurand.so.10", b"rand"),
                ("nvidia/../evil/lib/libevil.so.1", b"x"),
            ],
        );
        let dest = dir.join("cuda-libs");
        let t = Target {
            os: "linux",
            arch: "x86_64",
        };
        let mut got = BTreeSet::new();
        for w in [&w1, &w2] {
            got.extend(
                extract_part(w, ArchiveKind::Zip, Layout::NvidiaWheel, None, t, &dest).unwrap(),
            );
        }
        let res = crate::resources::catalog::cuda_libs_for("linux", "x86_64").unwrap();
        finalize(res, res.version, t, &dest, &mut got).unwrap();
        for n in cuda_provider_needs(false) {
            assert!(dest.join(n).exists(), "{n}");
        }
        assert_eq!(
            std::fs::read(dest.join("libcudart.so.12")).unwrap(),
            b"cudart"
        );
        #[cfg(unix)]
        assert!(
            dest.join("libcudart.so.12")
                .symlink_metadata()
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(dest.join("libcudnn_graph.so.9").is_file());
        assert!(dest.join("VERSION").is_file());
        for no in [
            "cuda.h",
            "cudnn.h",
            "libcudart_static.a",
            "RECORD",
            "libevil.so.1",
        ] {
            assert!(!dest.join(no).exists(), "{no}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Synthetic Windows wheels: DLLs from `bin/`, never `.lib`, headers or other dirs.
    #[test]
    fn windows_wheels_extract_only_dlls() {
        let dir = std::env::temp_dir().join(format!("bop-wheel-w-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let w = dir.join("all.whl");
        let mut entries: Vec<(String, Vec<u8>)> = cuda_provider_needs(true)
            .iter()
            .map(|n| (format!("nvidia/x/bin/{n}"), n.as_bytes().to_vec()))
            .collect();
        entries.push(("nvidia/cudnn/bin/cudnn_graph64_9.dll".into(), b"g".to_vec()));
        entries.push(("nvidia/cudnn/lib/x64/cudnn.lib".into(), b"l".to_vec()));
        entries.push(("nvidia/cudnn/include/cudnn.h".into(), b"h".to_vec()));
        entries.push((
            "nvidia/cudnn/lib/cudnn64_9.dll".into(),
            b"wrong dir".to_vec(),
        ));
        let refs: Vec<(&str, &[u8])> = entries
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        zip_with(&w, &refs);
        let dest = dir.join("cuda-libs");
        let t = Target {
            os: "windows",
            arch: "x86_64",
        };
        let mut got =
            extract_part(&w, ArchiveKind::Zip, Layout::NvidiaWheel, None, t, &dest).unwrap();
        let res = crate::resources::catalog::cuda_libs_for("windows", "x86_64").unwrap();
        finalize(res, res.version, t, &dest, &mut got).unwrap();
        assert!(got.contains("cudnn_graph64_9.dll"));
        assert_eq!(
            std::fs::read(dest.join("cudnn64_9.dll")).unwrap(),
            b"cudnn64_9.dll"
        );
        for no in ["cudnn.lib", "cudnn.h"] {
            assert!(!dest.join(no).exists(), "{no}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Build a small tar.gz with a library, a symlink to it, a link escaping the directory and
    /// a non-whitelisted file, and extract it with the ORT release layout.
    #[test]
    fn tgz_links_and_whitelist() {
        let dir = std::env::temp_dir().join(format!("bop-extract-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("a.tgz");
        {
            let f = std::fs::File::create(&archive).unwrap();
            let gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
            let mut tar = tar::Builder::new(gz);
            let mut add = |path: &str, data: &[u8]| {
                let mut h = tar::Header::new_gnu();
                h.set_size(data.len() as u64);
                h.set_mode(0o644);
                h.set_cksum();
                tar.append_data(&mut h, path, data).unwrap();
            };
            add("ort-1/lib/libonnxruntime.so.1.24.4", b"lib");
            add("ort-1/lib/libcustom.so", b"no");
            add("ort-1/include/onnxruntime_c_api.h", b"no");
            let mut link = |path: &str, target: &str| {
                let mut h = tar::Header::new_gnu();
                h.set_entry_type(tar::EntryType::Symlink);
                h.set_size(0);
                h.set_mode(0o777);
                tar.append_link(&mut h, path, target).unwrap();
            };
            link("ort-1/lib/libonnxruntime.so", "libonnxruntime.so.1.24.4");
            link(
                "ort-1/lib/libonnxruntime_providers_shared.so",
                "../../etc/passwd",
            );
            tar.into_inner().unwrap().finish().unwrap();
        }
        let dest = dir.join("out");
        let got = extract_part(
            &archive,
            ArchiveKind::TarGz,
            Layout::OrtRelease,
            Some(Flavor::Cpu),
            Target {
                os: "linux",
                arch: "x86_64",
            },
            &dest,
        )
        .unwrap();
        assert_eq!(got, set(&["libonnxruntime.so", "libonnxruntime.so.1.24.4"]));
        assert_eq!(
            std::fs::read(dest.join("libonnxruntime.so")).unwrap(),
            b"lib"
        );
        assert!(!dest.join("libcustom.so").exists());
        assert!(!dest.join("libonnxruntime_providers_shared.so").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
