//! Download the pinned OpenVINO runtime archive for this OS/arch and copy only the runtime
//! libraries into `<exe_dir>/openvino`, preserving the archive's `runtime/...` layout so that
//! `openvino-finder` (via `OPENVINO_INSTALL_DIR`, see `backend::libs`) finds them.
//!
//! Archive entries are treated as untrusted data: only whitelisted file names are written, to
//! destination paths built from our own whitelist (never from the entry path), and nothing from
//! the archive is executed.

use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Pinned OpenVINO runtime version.
pub const OPENVINO_VERSION: &str = "2026.4.0";

/// Archive container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveKind {
    Zip,
    TarGz,
}

/// Where to get the runtime for one platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Package {
    pub url: &'static str,
    pub kind: ArchiveKind,
}

/// Download URL and archive kind for `(std::env::consts::OS, std::env::consts::ARCH)`.
pub fn package_for(os: &str, arch: &str) -> Option<Package> {
    Some(match (os, arch) {
        ("windows", "x86_64") => Package {
            url: "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.4/windows_vc_mt/openvino_toolkit_windows_vc_mt_2026.4.0.22959.99c81491cc3_x86_64.zip",
            kind: ArchiveKind::Zip,
        },
        ("linux", "x86_64") => Package {
            url: "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.4/linux/openvino_toolkit_ubuntu24_2026.4.0.22959.99c81491cc3_x86_64.tgz",
            kind: ArchiveKind::TarGz,
        },
        ("linux", "aarch64") => Package {
            url: "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.4/linux/openvino_toolkit_ubuntu22_2026.4.0.22959.99c81491cc3_arm64.tgz",
            kind: ArchiveKind::TarGz,
        },
        ("macos", "aarch64") => Package {
            url: "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.4/macos/openvino_toolkit_macos_12_6_2026.4.0.22959.99c81491cc3_arm64.tgz",
            kind: ArchiveKind::TarGz,
        },
        _ => return None,
    })
}

#[derive(Debug, Clone, Default)]
pub struct SetupOptions {
    /// Destination directory (default `<exe_dir>/openvino`).
    pub dest: Option<PathBuf>,
    /// Version label; only the pinned [`OPENVINO_VERSION`] can be downloaded.
    pub version: Option<String>,
    /// Use this local archive instead of downloading.
    pub archive: Option<PathBuf>,
    /// Keep the downloaded archive (its path is printed).
    pub keep_archive: bool,
}

/// Which files of the archive make up the runtime on one OS.
struct Selection {
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
const GPU_PLUGIN: &str = "openvino_intel_gpu_plugin";
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

fn selection_for(os: &str) -> Result<Selection> {
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
fn unix_lib_base(name: &str) -> Option<&str> {
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
    fn wanted(&self, rel: &str) -> Option<String> {
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
    /// warnings (also printed) so tests can inspect them.
    fn check(&self, arch: &str, got: &BTreeSet<String>) -> Result<Vec<String>> {
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
            eprintln!("warning: {w}");
        }
        Ok(warnings)
    }
}

/// Strip the archive's single top-level folder (`openvino_toolkit_.../`) and normalise slashes.
/// Returns None for entries with suspicious components.
fn strip_top(entry: &str) -> Option<String> {
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
fn same_dir_link_target(rel: &str, target: &str, hard: bool) -> Option<String> {
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

fn write_entry(dest: &Path, rel: &str, reader: &mut dyn Read) -> Result<u64> {
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
    if out.exists() {
        std::fs::remove_file(&out).with_context(|| format!("replacing {}", out.display()))?;
    }
    std::fs::rename(&tmp, &out).with_context(|| format!("renaming to {}", out.display()))?;
    Ok(n)
}

fn extract_zip(archive: &Path, sel: &Selection, dest: &Path) -> Result<BTreeSet<String>> {
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
        let Some(rel) = strip_top(entry.name()).and_then(|r| sel.wanted(&r)) else {
            continue;
        };
        write_entry(dest, &rel, &mut entry)
            .with_context(|| format!("extracting {}", entry.name()))?;
        got.insert(rel);
    }
    Ok(got)
}

fn extract_tgz(archive: &Path, sel: &Selection, dest: &Path) -> Result<BTreeSet<String>> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("opening archive {}", archive.display()))?;
    let gz = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
    let mut tar = tar::Archive::new(gz);
    let mut got = BTreeSet::new();
    // (link rel path, target file name in the same directory)
    let mut links: Vec<(String, String)> = Vec::new();
    for entry in tar
        .entries()
        .with_context(|| format!("reading tar {}", archive.display()))?
    {
        let mut entry =
            entry.with_context(|| format!("reading tar entry in {}", archive.display()))?;
        let path = entry.path()?.to_string_lossy().into_owned();
        let Some(rel) = strip_top(&path).and_then(|r| sel.wanted(&r)) else {
            continue;
        };
        let ty = entry.header().entry_type();
        if ty.is_symlink() || ty.is_hard_link() {
            if let Some(target) = entry.link_name()? {
                let t = target.to_string_lossy().into_owned();
                if let Some(name) = same_dir_link_target(&rel, &t, ty.is_hard_link()) {
                    links.push((rel, name));
                } else {
                    tracing::debug!("skipping link {path} -> {t} (not a same-directory link)");
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
        for (rel, target) in &links {
            if got.contains(rel) {
                continue;
            }
            let (dir, _) = rel.rsplit_once('/').unwrap_or(("", ""));
            let target_rel = format!("{dir}/{target}");
            if !got.contains(&target_rel) {
                continue;
            }
            let link_path = dest.join(rel);
            if link_path.exists() || link_path.symlink_metadata().is_ok() {
                std::fs::remove_file(&link_path).ok();
            }
            #[cfg(unix)]
            std::os::unix::fs::symlink(target, &link_path)
                .with_context(|| format!("creating symlink {}", link_path.display()))?;
            #[cfg(not(unix))]
            std::fs::copy(dest.join(&target_rel), &link_path)
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

fn run_async<F, T>(fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    let go = move || -> Result<T> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("creating tokio runtime for download")?
            .block_on(fut)
    };
    if tokio::runtime::Handle::try_current().is_ok() {
        // Called from inside a runtime: block_on would panic, so use a helper thread.
        std::thread::spawn(go)
            .join()
            .map_err(|_| anyhow::anyhow!("download thread panicked"))?
    } else {
        go()
    }
}

async fn download_async(url: String, dest: PathBuf) -> Result<u64> {
    use futures_util::StreamExt;
    let client = reqwest::Client::builder()
        .user_agent(concat!("blue-onyx-prism/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("creating HTTP client")?;
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("GET {url}"))?;
    let total = resp.content_length();
    let pb = match total {
        Some(n) => indicatif::ProgressBar::new(n),
        None => indicatif::ProgressBar::new_spinner(),
    };
    if let Ok(style) = indicatif::ProgressStyle::with_template(
        "{msg} [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
    ) {
        pb.set_style(style.progress_chars("=> "));
    }
    let name = url.rsplit('/').next().unwrap_or(&url).to_string();
    pb.set_message(name);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    let tmp = dest.with_extension("partial");
    let mut f =
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut n = 0u64;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("downloading {url}"))?;
        f.write_all(&chunk)
            .with_context(|| format!("writing {}", tmp.display()))?;
        n += chunk.len() as u64;
        pb.inc(chunk.len() as u64);
    }
    f.flush()?;
    drop(f);
    pb.finish_and_clear();
    if let Some(t) = total
        && t != n
    {
        anyhow::bail!("download of {url} truncated: {n} of {t} bytes");
    }
    if dest.exists() {
        std::fs::remove_file(&dest).ok();
    }
    std::fs::rename(&tmp, &dest).with_context(|| format!("renaming to {}", dest.display()))?;
    Ok(n)
}

/// Blocking download of `url` to `dest` with a progress bar. Returns the byte count.
pub fn download_to_file(url: &str, dest: &Path) -> Result<u64> {
    run_async(download_async(url.to_string(), dest.to_path_buf()))
}

fn dir_size(p: &Path) -> u64 {
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

/// Download (or take `opts.archive`), extract the runtime subset into the destination and
/// return the destination directory.
pub fn run(opts: &SetupOptions) -> Result<PathBuf> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let version = opts.version.as_deref().unwrap_or(OPENVINO_VERSION);
    let dest = opts
        .dest
        .clone()
        .unwrap_or_else(|| crate::exe_dir().join(crate::backend::libs::BUNDLED_DIR_NAME));
    let sel = selection_for(os)?;

    let mut temp_dir: Option<PathBuf> = None;
    let (archive, kind) = match &opts.archive {
        Some(a) => {
            let name = a.to_string_lossy().to_ascii_lowercase();
            let kind = if name.ends_with(".zip") {
                ArchiveKind::Zip
            } else if name.ends_with(".tgz") || name.ends_with(".tar.gz") {
                ArchiveKind::TarGz
            } else {
                anyhow::bail!(
                    "unknown archive type for {} (expected .zip/.tgz)",
                    a.display()
                );
            };
            if !a.is_file() {
                anyhow::bail!("archive {} does not exist", a.display());
            }
            println!("Using local OpenVINO archive {}", a.display());
            (a.clone(), kind)
        }
        None => {
            if version != OPENVINO_VERSION {
                anyhow::bail!(
                    "only OpenVINO {OPENVINO_VERSION} can be downloaded (requested {version}); \
                     pass a local archive for other versions"
                );
            }
            let pkg = package_for(os, arch).with_context(|| {
                format!("no OpenVINO {OPENVINO_VERSION} runtime package known for {os}/{arch}")
            })?;
            let tmp =
                std::env::temp_dir().join(format!("blue-onyx-prism-setup-{}", std::process::id()));
            std::fs::create_dir_all(&tmp)
                .with_context(|| format!("creating temp dir {}", tmp.display()))?;
            let file_name = pkg.url.rsplit('/').next().unwrap_or("openvino_archive");
            let path = tmp.join(file_name);
            println!(
                "Downloading OpenVINO {OPENVINO_VERSION} for {os}/{arch}\n  {}",
                pkg.url
            );
            let n = download_to_file(pkg.url, &path)?;
            println!("Downloaded {:.1} MB", n as f64 / 1e6);
            temp_dir = Some(tmp);
            (path, pkg.kind)
        }
    };

    std::fs::create_dir_all(&dest)
        .with_context(|| format!("creating destination {}", dest.display()))?;
    println!("Extracting runtime libraries into {}", dest.display());
    let got = match kind {
        ArchiveKind::Zip => extract_zip(&archive, &sel, &dest)?,
        ArchiveKind::TarGz => extract_tgz(&archive, &sel, &dest)?,
    };
    let mut got = got;
    mirror_tbb_into_lib_dir(&sel, &dest, &mut got)?;
    sel.check(arch, &got)?;
    std::fs::write(dest.join("VERSION"), format!("{version}\n"))
        .with_context(|| format!("writing {}", dest.join("VERSION").display()))?;

    if let Some(tmp) = temp_dir {
        if opts.keep_archive {
            println!("Archive kept at {}", archive.display());
        } else if let Err(e) = std::fs::remove_dir_all(&tmp) {
            tracing::warn!("could not remove temp dir {}: {e}", tmp.display());
        }
    }
    let size = dir_size(&dest);
    println!(
        "OpenVINO {version} runtime ({} files, {:.1} MB) installed at {}",
        got.len(),
        size as f64 / 1e6,
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
}
