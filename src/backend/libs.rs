//! Locate the OpenVINO runtime libraries before `Core::new()`.
//!
//! `openvino-finder` searches `OPENVINO_INSTALL_DIR`, `INTEL_OPENVINO_DIR`, then the OS library
//! path (`PATH` / `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`) and `/opt/intel/openvino*`. It never
//! looks in the executable directory, so we point `OPENVINO_INSTALL_DIR` at `<exe_dir>/openvino`
//! (archive layout: `runtime/{bin|lib}/<arch>/Release`) when that folder exists.
//!
//! On Windows `openvino_c.dll` is opened by absolute path, which does not make the loader search
//! that DLL's directory for its own dependencies (`openvino.dll`, `tbb12.dll`, plugins), so the
//! runtime bin directories are also prepended to `PATH`.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const BUNDLED_DIR_NAME: &str = "openvino";

const ENV_INSTALL_DIR: &str = "OPENVINO_INSTALL_DIR";
const ENV_INTEL_DIR: &str = "INTEL_OPENVINO_DIR";

#[cfg(target_os = "windows")]
const ENV_LIBRARY_PATH: &str = "PATH";
#[cfg(target_os = "macos")]
const ENV_LIBRARY_PATH: &str = "DYLD_LIBRARY_PATH";
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const ENV_LIBRARY_PATH: &str = "LD_LIBRARY_PATH";

/// Sub-directories of an install dir that `openvino-finder` probes (same list, same order).
const KNOWN_SUBDIRS: &[&str] = &[
    "runtime/lib/intel64/Release",
    "runtime/lib/intel64",
    "runtime/lib/arm64/Release",
    "runtime/lib/arm64",
    "runtime/lib/aarch64/Release",
    "runtime/lib/aarch64",
    "runtime/lib/armv7l",
    "runtime/lib/armv7l/Release",
    "runtime/bin/intel64/Release",
    "runtime/bin/intel64",
    "runtime/bin/arm64/Release",
    "runtime/bin/arm64",
    "runtime/3rdparty/tbb/bin",
    "runtime/3rdparty/tbb/lib",
];

/// Windows directories holding DLLs that `openvino_c.dll` depends on.
#[cfg(windows)]
const WINDOWS_DLL_SUBDIRS: &[&str] = &["runtime/bin/intel64/Release", "runtime/3rdparty/tbb/bin"];

/// What `prepare_environment` decided, for `diagnostics()`.
static NOTES: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn note(s: String) {
    tracing::debug!("{s}");
    if let Ok(mut n) = NOTES.lock() {
        n.push(s);
    }
}

/// Set up environment variables so the finder can locate the runtime.
/// `explicit` (from config `openvino_dir`) wins over `<exe_dir>/openvino`; an already-set
/// `OPENVINO_INSTALL_DIR` is left untouched.
pub fn prepare_environment(explicit: Option<&Path>) {
    if let Ok(mut n) = NOTES.lock() {
        n.clear();
    }
    let existing = std::env::var_os(ENV_INSTALL_DIR).filter(|v| !v.is_empty());
    let install_dir: Option<PathBuf> = if let Some(v) = existing {
        let p = PathBuf::from(&v);
        note(format!(
            "{ENV_INSTALL_DIR} already set to {} (left untouched)",
            p.display()
        ));
        if let Some(e) = explicit {
            note(format!(
                "configured openvino_dir {} ignored because {ENV_INSTALL_DIR} is set",
                e.display()
            ));
        }
        Some(p)
    } else {
        let chosen = match explicit {
            Some(e) => {
                let e = crate::resolve_path(e);
                if !e.is_dir() {
                    note(format!(
                        "configured openvino_dir {} does not exist",
                        e.display()
                    ));
                }
                Some(e)
            }
            None => {
                let b = crate::exe_dir().join(BUNDLED_DIR_NAME);
                if b.is_dir() {
                    Some(b)
                } else {
                    note(format!("no bundled runtime at {}", b.display()));
                    None
                }
            }
        };
        if let Some(dir) = &chosen {
            // SAFETY: called from `OvCore::new` at startup, before OpenVINO starts its threads;
            // no other thread of ours mutates the environment concurrently.
            unsafe { std::env::set_var(ENV_INSTALL_DIR, dir) };
            note(format!("set {ENV_INSTALL_DIR}={}", dir.display()));
        }
        chosen
    };

    #[cfg(windows)]
    if let Some(dir) = &install_dir {
        let dirs: Vec<PathBuf> = WINDOWS_DLL_SUBDIRS
            .iter()
            .map(|s| dir.join(s))
            .filter(|d| d.is_dir())
            .collect();
        prepend_to_path(&dirs);
    }
    #[cfg(not(windows))]
    let _ = &install_dir;
}

/// Prepend `dirs` to `PATH`, skipping entries already present.
#[cfg(windows)]
fn prepend_to_path(dirs: &[PathBuf]) {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let existing: Vec<PathBuf> = std::env::split_paths(&current).collect();
    let same = |a: &Path, b: &Path| a.as_os_str().eq_ignore_ascii_case(b.as_os_str());
    let new: Vec<PathBuf> = dirs
        .iter()
        .filter(|d| !existing.iter().any(|e| same(e, d)))
        .cloned()
        .collect();
    if new.is_empty() {
        return;
    }
    let joined = match std::env::join_paths(new.iter().chain(existing.iter())) {
        Ok(j) => j,
        Err(e) => {
            note(format!("could not extend PATH: {e}"));
            return;
        }
    };
    // SAFETY: see `prepare_environment`.
    unsafe { std::env::set_var("PATH", joined) };
    for d in &new {
        note(format!("prepended {} to PATH", d.display()));
    }
}

/// Directory that would be used for the bundled runtime, if present.
pub fn bundled_dir() -> Option<PathBuf> {
    let d = crate::exe_dir().join(BUNDLED_DIR_NAME);
    d.is_dir().then_some(d)
}

/// File name of the OpenVINO C library on this OS (`openvino_c.dll`, `libopenvino_c.so`, ...).
pub fn openvino_c_file_name() -> String {
    format!(
        "{}openvino_c{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    )
}

/// Mirror of the env-var based part of the `openvino-finder` search; returns the first hit.
fn find_openvino_c(checked: &mut Vec<String>) -> Option<PathBuf> {
    let file = openvino_c_file_name();
    for var in [ENV_INSTALL_DIR, ENV_INTEL_DIR] {
        match std::env::var_os(var).filter(|v| !v.is_empty()) {
            Some(v) => {
                let root = PathBuf::from(v);
                checked.push(format!(
                    "{var}={} (runtime/lib|bin/<arch>[/Release])",
                    root.display()
                ));
                for sub in KNOWN_SUBDIRS {
                    let p = root.join(sub).join(&file);
                    if p.is_file() {
                        return Some(p);
                    }
                }
            }
            None => checked.push(format!("{var} (unset)")),
        }
    }
    match std::env::var_os(ENV_LIBRARY_PATH) {
        Some(path) => {
            checked.push(format!("{ENV_LIBRARY_PATH} entries"));
            for d in std::env::split_paths(&path) {
                let p = d.join(&file);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
        None => checked.push(format!("{ENV_LIBRARY_PATH} (unset)")),
    }
    None
}

/// Human readable summary of where the runtime was looked for (used in error messages).
pub fn diagnostics() -> String {
    let mut checked = Vec::new();
    let found = find_openvino_c(&mut checked);
    let notes = NOTES.lock().map(|n| n.clone()).unwrap_or_default();
    let mut s = format!(
        "OpenVINO runtime lookup for {}: checked {}.",
        openvino_c_file_name(),
        checked.join(", ")
    );
    if !notes.is_empty() {
        s.push_str(&format!(" Environment setup: {}.", notes.join("; ")));
    }
    match found {
        Some(p) => s.push_str(&format!(
            " Found {} (if loading still fails, a dependent library or plugin next to it is missing).",
            p.display()
        )),
        None => s.push_str(&format!(
            " {} was NOT found there (system dirs such as /opt/intel/openvino may still be probed). \
             Run `blue-onyx-openvino setup-openvino` to download the runtime into {}.",
            openvino_c_file_name(),
            crate::exe_dir().join(BUNDLED_DIR_NAME).display()
        )),
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_mentions_library() {
        let d = diagnostics();
        assert!(d.contains(&openvino_c_file_name()));
        assert!(d.contains("OPENVINO_INSTALL_DIR"));
    }
}
