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
//!
//! On Linux the archive's libraries have no `$ORIGIN` RUNPATH, so `libopenvino_c.so`'s
//! dependency `libopenvino.so.2640` (and its `libtbb.so.12`) only resolve through
//! `LD_LIBRARY_PATH`, which glibc reads once at process start. We therefore preload the
//! dependency chain by absolute path (TBB, then libopenvino) with `RTLD_NOW | RTLD_GLOBAL` and
//! keep the handles for the process lifetime; later loads (`libopenvino_c`, plugins, frontends)
//! find those dependencies among the already-loaded objects by soname. On macOS the same
//! preload is harmless (dyld matches loaded images by install name) and `setup-openvino`
//! additionally installs TBB next to libopenvino for `@rpath` resolution.

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
    #[cfg(unix)]
    if let Some(dir) = &install_dir {
        preload_dependencies(dir);
    }
    #[cfg(not(any(windows, unix)))]
    let _ = &install_dir;
}

/// Directory holding `openvino_c` inside an install dir (same probe order as the finder).
#[cfg_attr(not(unix), allow(dead_code))]
fn lib_dir_of(install_dir: &Path) -> Option<PathBuf> {
    let file = openvino_c_file_name();
    KNOWN_SUBDIRS
        .iter()
        .filter(|s| !s.contains("3rdparty"))
        .map(|s| install_dir.join(s))
        .find(|d| d.join(&file).is_file())
}

/// Pick the file to preload for library `base` (`"openvino"`, `"tbb"`) among the file names
/// of one directory. Versioned soname-style names (`libtbb.so.12`, `libopenvino.2640.dylib`)
/// are preferred, shortest first (the soname rather than the full version); the unversioned
/// development name (`libtbb.so`, `libopenvino.dylib`) is the fallback. Names of other
/// libraries sharing the prefix (`libopenvino_c.so`, `libtbbmalloc.so.2`) never match.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) fn pick_library<'a>(names: &[&'a str], base: &str) -> Option<&'a str> {
    let prefix = format!("lib{base}.");
    let mut versioned: Vec<&'a str> = Vec::new();
    let mut plain: Option<&'a str> = None;
    for &n in names {
        let Some(rest) = n.strip_prefix(&prefix) else {
            continue;
        };
        // rest: "so", "so.12", "so.12.13", "dylib", "12.dylib", "2026.4.0.dylib"
        if rest == "so" || rest == "dylib" {
            plain = Some(n);
            continue;
        }
        let version = if let Some(v) = rest.strip_prefix("so.") {
            v
        } else if let Some(v) = rest.strip_suffix(".dylib") {
            v
        } else {
            continue;
        };
        if !version.is_empty()
            && version
                .split('.')
                .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        {
            versioned.push(n);
        }
    }
    versioned.sort_by_key(|n| (n.len(), *n));
    versioned.first().copied().or(plain)
}

/// Absolute paths to preload, in order: TBB (from the lib dir when `setup-openvino` mirrored
/// it there, else `runtime/3rdparty/tbb/lib`), then libopenvino. Missing pieces are skipped.
#[cfg_attr(not(unix), allow(dead_code))]
fn preload_plan(install_dir: &Path) -> Vec<PathBuf> {
    let Some(lib_dir) = lib_dir_of(install_dir) else {
        return Vec::new();
    };
    let list = |d: &Path| -> Vec<String> {
        std::fs::read_dir(d)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter_map(|e| e.file_name().to_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let pick = |d: &Path, base: &str| -> Option<PathBuf> {
        let names = list(d);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        pick_library(&refs, base).map(|n| d.join(n))
    };
    let mut plan = Vec::new();
    let tbb = pick(&lib_dir, "tbb")
        .or_else(|| pick(&install_dir.join("runtime/3rdparty/tbb/lib"), "tbb"));
    plan.extend(tbb);
    plan.extend(pick(&lib_dir, "openvino"));
    plan
}

/// Handles of preloaded libraries; never dropped so the objects stay loaded for the process.
#[cfg(unix)]
static PRELOADED: Mutex<Vec<libloading::os::unix::Library>> = Mutex::new(Vec::new());

/// Load the OpenVINO dependency chain by absolute path (see the module docs). Failures are
/// only noted: if the libraries are reachable some other way, `Core::new()` still works, and
/// if not, its error plus `diagnostics()` tell the story.
#[cfg(unix)]
fn preload_dependencies(install_dir: &Path) {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
    let Ok(mut loaded) = PRELOADED.lock() else {
        return;
    };
    if !loaded.is_empty() {
        return;
    }
    for path in preload_plan(install_dir) {
        // SAFETY: loading OpenVINO's own runtime libraries (TBB, libopenvino) from the
        // configured install dir; their initializers are the same ones that would run when
        // openvino-sys loads libopenvino_c. The handles are kept for the process lifetime, so
        // no symbol from them can outlive its library.
        match unsafe { Library::open(Some(&path), RTLD_NOW | RTLD_GLOBAL) } {
            Ok(lib) => {
                note(format!("preloaded {}", path.display()));
                loaded.push(lib);
            }
            Err(e) => note(format!("could not preload {}: {e}", path.display())),
        }
    }
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

/// `ov_core_set_property(core, device, key, value)` called with its real, variadic C signature.
///
/// The C API declares `ov_core_set_property(const ov_core_t*, const char* device, ...)`, but
/// openvino-sys 0.11 binds (and, with runtime linking, calls) it as a fixed 4-argument function.
/// On x86_64 and Linux/Windows aarch64 both conventions pass the extra pointers in the same
/// registers, but Apple arm64 passes variadic arguments on the stack, so the plugin reads
/// garbage keys ("Unsupported property ...") and can crash. This looks the symbol up in the
/// already-loaded `openvino_c` and calls it as variadic.
#[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
pub(crate) fn core_set_property_variadic(
    core: &openvino::Core,
    device: &str,
    key: &str,
    value: &str,
) -> Result<(), String> {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    type SetProperty = unsafe extern "C" fn(*const c_void, *const c_char, ...) -> c_int;
    type LastErr = unsafe extern "C" fn() -> *const c_char;
    // `openvino::Core` is a struct whose only field is the `*mut ov_core_t`.
    const _: () =
        assert!(std::mem::size_of::<openvino::Core>() == std::mem::size_of::<*const c_void>());

    let path = find_openvino_c(&mut Vec::new())
        .ok_or_else(|| format!("{} not found", openvino_c_file_name()))?;
    let cstr = |s: &str| CString::new(s).map_err(|e| format!("invalid string {s:?}: {e}"));
    let (device, key, value) = (cstr(device)?, cstr(key)?, cstr(value)?);
    // SAFETY: `path` is the openvino_c library openvino-sys already loaded (Core exists), so
    // this only bumps its reference count. The symbol types match the C declarations in
    // ov_core.h / ov_common.h. The core pointer is read from `openvino::Core`, whose single
    // field is that pointer (size asserted above); `core` outlives the call and the C strings
    // are NUL-terminated and alive for the call.
    unsafe {
        let lib = libloading::Library::new(&path).map_err(|e| e.to_string())?;
        let set: libloading::Symbol<SetProperty> = lib
            .get(b"ov_core_set_property\0")
            .map_err(|e| e.to_string())?;
        let raw: *const c_void = std::mem::transmute_copy(core);
        let status = set(raw, device.as_ptr(), key.as_ptr(), value.as_ptr());
        if status == 0 {
            return Ok(());
        }
        let msg = lib
            .get::<LastErr>(b"ov_get_last_err_msg\0")
            .ok()
            .map(|f| f())
            .filter(|p| !p.is_null())
            .map(|p| CStr::from_ptr(p).to_string_lossy().into_owned())
            .unwrap_or_default();
        Err(format!("status {status}: {msg}"))
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
             Run `blue-onyx-prism setup-openvino` to download the runtime into {}.",
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
    fn picks_soname_style_files() {
        let linux = [
            "libopenvino.so",
            "libopenvino.so.2026.4.0",
            "libopenvino.so.2640",
            "libopenvino_c.so",
            "libopenvino_c.so.2640",
            "libopenvino_intel_cpu_plugin.so",
            "libopenvino_onnx_frontend.so.2640",
            "cache.json",
        ];
        assert_eq!(
            pick_library(&linux, "openvino"),
            Some("libopenvino.so.2640")
        );
        assert_eq!(
            pick_library(&linux, "openvino_c"),
            Some("libopenvino_c.so.2640")
        );
        let tbb = [
            "libtbb.so",
            "libtbb.so.12",
            "libtbb.so.12.13",
            "libtbbmalloc.so.2",
            "libtbbmalloc_proxy.so.2",
            "libtbbbind_2_5.so.3",
            "libhwloc.so.15",
        ];
        assert_eq!(pick_library(&tbb, "tbb"), Some("libtbb.so.12"));
        let mac = [
            "libopenvino.dylib",
            "libopenvino.2026.4.0.dylib",
            "libopenvino.2640.dylib",
            "libopenvino_c.2640.dylib",
            "libopenvino_arm_cpu_plugin.so",
            "libtbb.12.13.dylib",
            "libtbb.12.dylib",
            "libtbb.dylib",
        ];
        assert_eq!(
            pick_library(&mac, "openvino"),
            Some("libopenvino.2640.dylib")
        );
        assert_eq!(pick_library(&mac, "tbb"), Some("libtbb.12.dylib"));
        // Unversioned fallback; nothing when absent; never a sibling library.
        assert_eq!(pick_library(&["libtbb.so"], "tbb"), Some("libtbb.so"));
        assert_eq!(
            pick_library(&["libtbbmalloc.so.2", "libtbb.so.x"], "tbb"),
            None
        );
        assert_eq!(pick_library(&[], "openvino"), None);
    }

    #[cfg(unix)]
    #[test]
    fn preload_plan_orders_tbb_then_openvino() {
        let root = std::env::temp_dir().join(format!("bo_libs_{}", std::process::id()));
        let lib = root.join(if cfg!(target_os = "macos") {
            "runtime/lib/arm64/Release"
        } else {
            "runtime/lib/intel64"
        });
        let tbb = root.join("runtime/3rdparty/tbb/lib");
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::create_dir_all(&tbb).unwrap();
        let (ov, ovc, t) = if cfg!(target_os = "macos") {
            (
                "libopenvino.2640.dylib",
                "libopenvino_c.dylib",
                "libtbb.12.dylib",
            )
        } else {
            ("libopenvino.so.2640", "libopenvino_c.so", "libtbb.so.12")
        };
        for (d, n) in [(&lib, ov), (&lib, ovc), (&tbb, t)] {
            std::fs::write(d.join(n), b"").unwrap();
        }
        assert_eq!(preload_plan(&root), vec![tbb.join(t), lib.join(ov)]);
        // Without openvino_c the dir is not an install dir.
        std::fs::remove_file(lib.join(ovc)).unwrap();
        assert!(preload_plan(&root).is_empty());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn diagnostics_mentions_library() {
        let d = diagnostics();
        assert!(d.contains(&openvino_c_file_name()));
        assert!(d.contains("OPENVINO_INSTALL_DIR"));
    }
}
