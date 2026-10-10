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

/// The `openvino_c` library loaded by explicit path (Unix), for helpers that need its path.
static LOADED_OPENVINO_C: Mutex<Option<PathBuf>> = Mutex::new(None);

/// The `openvino_c` library this process loaded: the explicit path from
/// [`prepare_environment`], else what the finder's environment search sees.
pub fn loaded_openvino_c() -> Option<PathBuf> {
    let loaded = LOADED_OPENVINO_C.lock().ok().and_then(|g| g.clone());
    loaded.or_else(|| find_openvino_c(&mut Vec::new()))
}

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
        // Windows: the finder reads OPENVINO_INSTALL_DIR. Unix: the library is loaded by
        // explicit path below instead, because this also runs in later registry generations
        // (after a runtime download) while other threads exist, where `setenv` is not sound.
        #[cfg(windows)]
        if let Some(dir) = &chosen {
            // SAFETY: the Windows environment block is protected by a lock (SetEnvironmentVariableW),
            // so this is sound even with other threads running.
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
        match lib_dir_of(dir) {
            Some(lib_dir) => {
                let lib = lib_dir.join(openvino_c_file_name());
                match openvino_sys::library::load_from(&lib) {
                    Ok(()) => {
                        note(format!("loaded {}", lib.display()));
                        if let Ok(mut g) = LOADED_OPENVINO_C.lock()
                            && g.is_none()
                        {
                            *g = Some(lib.clone());
                        }
                    }
                    Err(e) => note(format!("loading {} failed: {e}", lib.display())),
                }
            }
            None => note(format!(
                "no {} under {}",
                openvino_c_file_name(),
                dir.display()
            )),
        }
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

    let path =
        loaded_openvino_c().ok_or_else(|| format!("{} not found", openvino_c_file_name()))?;
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

// ---------------------------------------------------------------------------------------------
// ONNX Runtime library lookup (pure apart from directory listings; shared with
// `setup-onnxruntime` and the download manager). Layout under `<exe_dir>/onnxruntime/`:
// - one directory per installed flavor (`cpu/`, `cuda/`, `directml/`, `coreml/`), each holding
//   the shared library (plus provider libraries / DirectML.dll) flat and a `flavor.txt`;
// - `active.txt` naming the flavor this process loads (a process can load only one);
// - legacy (before phase 7.2): the libraries flat in `onnxruntime/` itself, with `flavor.txt`.
// ---------------------------------------------------------------------------------------------

/// Folder next to the executable that `setup-onnxruntime` installs into.
pub const ORT_DIR_NAME: &str = "onnxruntime";
/// File in the ONNX Runtime folder naming the installed flavor (`cpu`, `cuda`, `directml`,
/// `coreml`).
pub const ORT_FLAVOR_FILE: &str = "flavor.txt";
/// Environment variable `ort` itself honors: the path of the ONNX Runtime library (or, for us, a
/// directory holding it).
pub const ENV_ORT_DYLIB_PATH: &str = "ORT_DYLIB_PATH";

/// File name of the ONNX Runtime shared library on this OS (`onnxruntime.dll`,
/// `libonnxruntime.so`, `libonnxruntime.dylib`). Installs may carry only a versioned name
/// (`libonnxruntime.so.1.24.4`, `libonnxruntime.1.24.4.dylib`); [`find_ort_library_in`] accepts
/// those too.
pub fn ort_library_file_name() -> &'static str {
    if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_vendor = "apple") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// `<exe_dir>/onnxruntime`.
pub fn default_onnxruntime_dir() -> PathBuf {
    crate::exe_dir().join(ORT_DIR_NAME)
}

/// Pick the ONNX Runtime library among the file names of one directory: the plain name
/// ([`ort_library_file_name`]) when present, else the shortest versioned one
/// (`libonnxruntime.so.1`, `libonnxruntime.1.24.4.dylib`). Provider libraries
/// (`libonnxruntime_providers_cuda.so`) never match.
pub fn pick_ort_library<'a>(names: &[&'a str], plain: &str) -> Option<&'a str> {
    if let Some(n) = names.iter().find(|n| n.eq_ignore_ascii_case(plain)) {
        return Some(n);
    }
    // Versioned names exist only on Unix-like systems ("lib" prefix).
    let base = plain
        .strip_prefix("lib")
        .and_then(|b| b.split('.').next())
        .filter(|_| !plain.ends_with(".dll"))?;
    pick_library(names, base)
}

/// The ONNX Runtime library inside `dir`, if any.
pub fn find_ort_library_in(dir: &Path) -> Option<PathBuf> {
    let names: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    pick_ort_library(&refs, ort_library_file_name()).map(|n| dir.join(n))
}

/// Where the ONNX Runtime library was looked for, and what was found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OrtLookup {
    /// The library to load (None = not found anywhere).
    pub library: Option<PathBuf>,
    /// Every place checked, in order, for error messages.
    pub checked: Vec<String>,
}

impl OrtLookup {
    /// Error text when nothing was found: where we looked and how to install it.
    pub fn not_found_message(&self) -> String {
        format!(
            "ONNX Runtime library ({}) not found; checked {}. Run `blue-onyx-prism \
             setup-onnxruntime` to install it into {}",
            ort_library_file_name(),
            if self.checked.is_empty() {
                "nothing".to_string()
            } else {
                self.checked.join(", ")
            },
            default_onnxruntime_dir().display()
        )
    }
}

/// File in the ONNX Runtime folder naming the active flavor's subdirectory.
pub const ORT_ACTIVE_FILE: &str = "active.txt";
/// Per-flavor subdirectories, in the order they are tried when `active.txt` is absent.
pub const ORT_FLAVOR_DIRS: &[&str] = &["cuda", "directml", "coreml", "cpu"];

/// The flavor named by `<ort_root>/active.txt` (trimmed, lowercase), when it is a known flavor.
pub fn read_active_flavor(ort_root: &Path) -> Option<String> {
    let f = std::fs::read_to_string(ort_root.join(ORT_ACTIVE_FILE)).ok()?;
    let f = f.trim().to_ascii_lowercase();
    ORT_FLAVOR_DIRS.contains(&f.as_str()).then_some(f)
}

/// Make `flavor` the active one (`<ort_root>/active.txt`, replaced atomically). It is used from
/// the next process start (a process keeps the ONNX Runtime library it loaded).
pub fn write_active_flavor(ort_root: &Path, flavor: &str) -> std::io::Result<()> {
    if !ORT_FLAVOR_DIRS.contains(&flavor) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unknown ONNX Runtime flavor '{flavor}'"),
        ));
    }
    std::fs::create_dir_all(ort_root)?;
    let tmp = ort_root.join(format!("{ORT_ACTIVE_FILE}.tmp"));
    std::fs::write(&tmp, format!("{flavor}\n"))?;
    std::fs::rename(&tmp, ort_root.join(ORT_ACTIVE_FILE))
}

/// [`find_onnxruntime`] with another install root than `<exe_dir>/onnxruntime` (config
/// `download_dir`).
pub fn find_onnxruntime_with(explicit: Option<&Path>, default_dir: &Path) -> OrtLookup {
    let env = std::env::var_os(ENV_ORT_DYLIB_PATH)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    find_onnxruntime_from(explicit, env.as_deref(), default_dir)
}

/// Locate the ONNX Runtime library. Order (first hit wins):
/// 1. config `onnxruntime_dir` (`explicit`, a directory or the library file itself);
/// 2. `ORT_DYLIB_PATH` (file or directory);
/// 3. `<exe_dir>/onnxruntime/<flavor>/` for the flavor in `onnxruntime/active.txt`;
/// 4. `<exe_dir>/onnxruntime/` itself (the legacy flat layout);
/// 5. the first installed `<exe_dir>/onnxruntime/<flavor>/` in [`ORT_FLAVOR_DIRS`] order.
pub fn find_onnxruntime(explicit: Option<&Path>) -> OrtLookup {
    let env = std::env::var_os(ENV_ORT_DYLIB_PATH)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    find_onnxruntime_from(explicit, env.as_deref(), &default_onnxruntime_dir())
}

/// [`find_onnxruntime`] with the environment and the default folder passed in (testable).
pub fn find_onnxruntime_from(
    explicit: Option<&Path>,
    env_path: Option<&Path>,
    default_dir: &Path,
) -> OrtLookup {
    let mut out = OrtLookup::default();
    let mut try_path = |label: &str, p: &Path| -> Option<PathBuf> {
        let hit = if p.is_file() {
            Some(p.to_path_buf())
        } else if p.is_dir() {
            find_ort_library_in(p)
        } else {
            None
        };
        let state = match (&hit, p.exists()) {
            (Some(_), _) => "found",
            (None, true) => "no ONNX Runtime library inside",
            (None, false) => "does not exist",
        };
        out.checked
            .push(format!("{label} {} ({state})", p.display()));
        hit
    };
    let explicit = explicit.map(crate::resolve_path);
    let active = read_active_flavor(default_dir);
    let found = explicit
        .as_deref()
        .and_then(|p| try_path("onnxruntime_dir", p))
        .or_else(|| env_path.and_then(|p| try_path(ENV_ORT_DYLIB_PATH, p)))
        .or_else(|| {
            let f = active.as_deref()?;
            try_path(
                &format!("active flavor ({ORT_ACTIVE_FILE})"),
                &default_dir.join(f),
            )
        })
        .or_else(|| try_path("bundled", default_dir))
        .or_else(|| {
            ORT_FLAVOR_DIRS
                .iter()
                .filter(|f| Some(**f) != active.as_deref())
                .map(|f| default_dir.join(f))
                .filter(|d| d.is_dir())
                .find_map(|d| try_path("bundled flavor", &d))
        });
    out.library = found;
    out
}

/// Installed flavor from `flavor.txt` next to the library (trimmed, lowercase), if present.
pub fn read_ort_flavor(lib_dir: &Path) -> Option<String> {
    std::fs::read_to_string(lib_dir.join(ORT_FLAVOR_FILE))
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
}

// ---------------------------------------------------------------------------------------------
// NVIDIA CUDA libraries (`nvidia-cuda-libs`, extracted flat into `onnxruntime/cuda-libs`).
//
// ONNX Runtime's CUDA provider links cudart, cuBLAS(Lt), cuDNN 9, cuFFT and cuRAND by name
// (`DT_NEEDED` / PE imports), and cuDNN loads its sub-libraries by name at run time. The
// directory cannot be put on `LD_LIBRARY_PATH` after the process started, so every library is
// preloaded by absolute path: a later by-name load then finds the already loaded module (glibc
// matches loaded objects by soname; Windows uses an already loaded DLL of the same module
// name). On Windows the directory is also prepended to `PATH` for anything loaded later.
// ---------------------------------------------------------------------------------------------

/// Subdirectory of the ONNX Runtime folder holding the NVIDIA libraries.
pub const CUDA_LIBS_DIR_NAME: &str = "cuda-libs";

/// Libraries that make a usable CUDA 12 / cuDNN 9 setup, by platform (any system install that
/// can load all of these does not need `nvidia-cuda-libs`).
pub fn cuda_core_libs(windows: bool) -> &'static [&'static str] {
    if windows {
        &["cudart64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"]
    } else {
        &["libcudart.so.12", "libcublas.so.12", "libcudnn.so.9"]
    }
}

/// Library family of a CUDA library file name: `libcublasLt.so.12` / `cublasLt64_12.dll` ->
/// `cublaslt`, `cudnn_graph64_9.dll` -> `cudnn_graph`, `nvrtc-builtins64_128.dll` ->
/// `nvrtc-builtins`. None when the name is not a shared library of that platform.
pub fn cuda_lib_family(name: &str, windows: bool) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    let base = if windows {
        lower.strip_suffix(".dll")?
    } else {
        let (b, _) = lower.split_once(".so")?;
        b.strip_prefix("lib")?
    };
    let family = base.trim_end_matches(|c: char| c.is_ascii_digit() || c == '_');
    (!family.is_empty()).then(|| family.to_string())
}

/// Preload order: libraries before the ones that link them (cudart, nvJitLink and NVRTC, then
/// cuBLASLt before cuBLAS, cuFFT, cuRAND, the cuDNN sub-libraries, cuDNN itself last).
/// Unknown libraries go between the known ones and cuDNN; ties sort by name. The loader also
/// retries failures after each pass, so this only has to be right in the common case.
pub fn cuda_preload_order(names: &[&str], windows: bool) -> Vec<String> {
    const ORDER: &[&str] = &[
        "cudart",
        "nvjitlink",
        "nvrtc-builtins",
        "nvrtc",
        "cublaslt",
        "cublas",
        "cufft",
        "curand",
        "cudnn_graph",
        "cudnn_engines_precompiled",
        "cudnn_engines_runtime_compiled",
        "cudnn_heuristic",
        "cudnn_ops",
        "cudnn_cnn",
        "cudnn_adv",
    ];
    let rank = |family: &str| -> usize {
        if family == "cudnn" {
            ORDER.len() + 1
        } else {
            ORDER
                .iter()
                .position(|f| *f == family)
                .unwrap_or(ORDER.len())
        }
    };
    let mut out: Vec<(usize, String)> = names
        .iter()
        .filter_map(|n| cuda_lib_family(n, windows).map(|f| (rank(&f), n.to_string())))
        .collect();
    out.sort();
    out.dedup_by(|a, b| a.1 == b.1);
    out.into_iter().map(|(_, n)| n).collect()
}

/// What [`preload_cuda_libs`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CudaPreload {
    pub loaded: Vec<PathBuf>,
    /// Libraries that would not load, with the last error.
    pub failed: Vec<(PathBuf, String)>,
}

/// Directory preloaded so far (once per process).
static CUDA_PRELOADED: Mutex<Option<(PathBuf, CudaPreload)>> = Mutex::new(None);

/// Preload every CUDA library in `dir` by absolute path (see the section comment), in
/// [`cuda_preload_order`], retrying failures until no pass makes progress. Runs once per
/// process; later calls return the first result. The libraries stay loaded for the process.
pub fn preload_cuda_libs(dir: &Path) -> CudaPreload {
    let mut slot = CUDA_PRELOADED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((d, r)) = slot.as_ref() {
        if d != dir {
            tracing::debug!(
                "CUDA libraries already preloaded from {}; {} ignored",
                d.display(),
                dir.display()
            );
        }
        return r.clone();
    }
    let windows = cfg!(windows);
    let names: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_file())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut pending: Vec<PathBuf> = cuda_preload_order(&refs, windows)
        .into_iter()
        .map(|n| dir.join(n))
        .collect();
    #[cfg(windows)]
    prepend_to_path(&[dir.to_path_buf()]);
    let mut report = CudaPreload::default();
    let mut errors: std::collections::HashMap<PathBuf, String> = Default::default();
    loop {
        let before = pending.len();
        pending.retain(|p| match open_cuda_lib(p) {
            Ok(()) => {
                report.loaded.push(p.clone());
                false
            }
            Err(e) => {
                errors.insert(p.clone(), e);
                true
            }
        });
        if pending.is_empty() || pending.len() == before {
            break;
        }
    }
    report.failed = pending
        .into_iter()
        .map(|p| {
            let e = errors.remove(&p).unwrap_or_default();
            (p, e)
        })
        .collect();
    *slot = Some((dir.to_path_buf(), report.clone()));
    report
}

/// Load one CUDA library by absolute path and keep it loaded.
fn open_cuda_lib(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
        // SAFETY: NVIDIA's CUDA/cuDNN redistributable libraries (verified by SHA-256 when
        // downloaded) that ONNX Runtime's CUDA provider would load itself; their initializers
        // run as they would then. The handle is leaked so they stay loaded for the process.
        let lib = unsafe { Library::open(Some(path), RTLD_NOW | RTLD_GLOBAL) }
            .map_err(|e| e.to_string())?;
        std::mem::forget(lib);
        Ok(())
    }
    #[cfg(windows)]
    {
        use libloading::os::windows::{LOAD_WITH_ALTERED_SEARCH_PATH, Library};
        // SAFETY: as above. LOAD_WITH_ALTERED_SEARCH_PATH resolves the DLL's own imports from
        // its directory first (the other CUDA DLLs next to it).
        let lib = unsafe { Library::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) }
            .map_err(|e| e.to_string())?;
        std::mem::forget(lib);
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err(format!("cannot load {} on this platform", path.display()))
    }
}

/// Whether the system (driver/toolkit install, `PATH` / `LD_LIBRARY_PATH`) already provides
/// CUDA 12 and cuDNN 9: every [`cuda_core_libs`] name loads by name.
pub fn system_cuda_present() -> bool {
    cuda_core_libs(cfg!(windows)).iter().all(|n| {
        // SAFETY: probing CUDA libraries by name, which the CUDA provider would load itself; the
        // handle is dropped right away and no symbol is used.
        unsafe { libloading::Library::new(n) }.is_ok()
    })
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

    #[test]
    fn picks_onnxruntime_library() {
        let mac = [
            "libonnxruntime.1.24.4.dylib",
            "libonnxruntime_providers_shared.dylib",
            "flavor.txt",
        ];
        assert_eq!(
            pick_ort_library(&mac, "libonnxruntime.dylib"),
            Some("libonnxruntime.1.24.4.dylib")
        );
        let mac_plain = ["libonnxruntime.1.24.4.dylib", "libonnxruntime.dylib"];
        assert_eq!(
            pick_ort_library(&mac_plain, "libonnxruntime.dylib"),
            Some("libonnxruntime.dylib")
        );
        let linux = [
            "libonnxruntime.so.1.24.4",
            "libonnxruntime.so.1",
            "libonnxruntime_providers_cuda.so",
            "libonnxruntime_providers_shared.so",
        ];
        assert_eq!(
            pick_ort_library(&linux, "libonnxruntime.so"),
            Some("libonnxruntime.so.1")
        );
        let win = [
            "onnxruntime.dll",
            "onnxruntime_providers_shared.dll",
            "DirectML.dll",
        ];
        assert_eq!(
            pick_ort_library(&win, "onnxruntime.dll"),
            Some("onnxruntime.dll")
        );
        assert_eq!(
            pick_ort_library(&["onnxruntime_providers_cuda.dll"], "onnxruntime.dll"),
            None
        );
        assert_eq!(
            pick_ort_library(&["libonnxruntime_providers_cuda.so"], "libonnxruntime.so"),
            None
        );
    }

    #[test]
    fn onnxruntime_lookup_order() {
        let root = std::env::temp_dir().join(format!("bop_ortlib_{}", uuid::Uuid::new_v4()));
        let (a, b, c) = (root.join("a"), root.join("b"), root.join("c"));
        for d in [&a, &b, &c] {
            std::fs::create_dir_all(d).unwrap();
        }
        let lib = ort_library_file_name();
        std::fs::write(b.join(lib), b"").unwrap();
        std::fs::write(c.join(lib), b"").unwrap();
        std::fs::write(c.join(ORT_FLAVOR_FILE), " CoreML\n").unwrap();

        // Explicit dir without a library falls through to the env path.
        let r = find_onnxruntime_from(Some(&a), Some(&b), &c);
        assert_eq!(r.library, Some(b.join(lib)));
        assert_eq!(r.checked.len(), 2);
        assert!(r.checked[0].contains("no ONNX Runtime library inside"));
        // Explicit dir with a library wins; a file path works too.
        assert_eq!(
            find_onnxruntime_from(Some(&c), Some(&b), &a).library,
            Some(c.join(lib))
        );
        assert_eq!(
            find_onnxruntime_from(None, Some(&b.join(lib)), &c).library,
            Some(b.join(lib))
        );
        // Default dir last; nothing found anywhere.
        assert_eq!(
            find_onnxruntime_from(None, None, &c).library,
            Some(c.join(lib))
        );
        let none = find_onnxruntime_from(Some(&root.join("missing")), None, &a);
        assert_eq!(none.library, None);
        assert!(
            none.checked[0].contains("does not exist"),
            "{:?}",
            none.checked
        );
        assert_eq!(read_ort_flavor(&c).as_deref(), Some("coreml"));
        assert_eq!(read_ort_flavor(&a), None);
        assert!(default_onnxruntime_dir().ends_with(ORT_DIR_NAME));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn cuda_library_families_and_order() {
        assert_eq!(
            cuda_lib_family("libcublasLt.so.12", false).as_deref(),
            Some("cublaslt")
        );
        assert_eq!(
            cuda_lib_family("cublasLt64_12.dll", true).as_deref(),
            Some("cublaslt")
        );
        assert_eq!(
            cuda_lib_family("cudnn_graph64_9.dll", true).as_deref(),
            Some("cudnn_graph")
        );
        assert_eq!(
            cuda_lib_family("libcudnn.so.9", false).as_deref(),
            Some("cudnn")
        );
        assert_eq!(
            cuda_lib_family("nvrtc-builtins64_128.dll", true).as_deref(),
            Some("nvrtc-builtins")
        );
        assert_eq!(
            cuda_lib_family("nvJitLink_120_0.dll", true).as_deref(),
            Some("nvjitlink")
        );
        assert_eq!(cuda_lib_family("VERSION", false), None);
        assert_eq!(cuda_lib_family("cudnn.h", true), None);
        assert_eq!(cuda_lib_family(".installed.json", true), None);

        let linux = [
            "libcudnn.so.9",
            "libcudnn_ops.so.9",
            "libcublas.so.12",
            "libcublasLt.so.12",
            "libcudnn_graph.so.9",
            "libcufft.so.11",
            "libnvJitLink.so.12",
            "libcudart.so.12",
            "libcurand.so.10",
            "libnvrtc.so.12",
            "libcudnn_engines_precompiled.so.9",
            "VERSION",
            ".installed.json",
        ];
        assert_eq!(
            cuda_preload_order(&linux, false),
            [
                "libcudart.so.12",
                "libnvJitLink.so.12",
                "libnvrtc.so.12",
                "libcublasLt.so.12",
                "libcublas.so.12",
                "libcufft.so.11",
                "libcurand.so.10",
                "libcudnn_graph.so.9",
                "libcudnn_engines_precompiled.so.9",
                "libcudnn_ops.so.9",
                "libcudnn.so.9",
            ]
        );
        let win = [
            "cudnn64_9.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cudart64_12.dll",
            "cudnn_cnn64_9.dll",
            "zlibwapi.dll",
        ];
        assert_eq!(
            cuda_preload_order(&win, true),
            [
                "cudart64_12.dll",
                "cublasLt64_12.dll",
                "cublas64_12.dll",
                "cudnn_cnn64_9.dll",
                "zlibwapi.dll",
                "cudnn64_9.dll",
            ]
        );
        assert_eq!(cuda_core_libs(false)[2], "libcudnn.so.9");
        // The catalog installs `nvidia-cuda-libs` where the loader looks.
        assert_eq!(
            crate::resources::catalog::CUDA_LIBS_DEST,
            format!("{ORT_DIR_NAME}/{CUDA_LIBS_DIR_NAME}")
        );
    }

    #[test]
    fn cuda_preload_of_an_empty_dir_reports_nothing() {
        let dir = std::env::temp_dir().join(format!("bop_cudalibs_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let r = preload_cuda_libs(&dir);
        assert!(r.loaded.is_empty() && r.failed.is_empty(), "{r:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn per_flavor_dirs_and_active_file() {
        let root = std::env::temp_dir().join(format!("bop_ortflv_{}", uuid::Uuid::new_v4()));
        let lib = ort_library_file_name();
        for f in ["cpu", "cuda"] {
            std::fs::create_dir_all(root.join(f)).unwrap();
            std::fs::write(root.join(f).join(lib), b"").unwrap();
        }
        // No active.txt, no flat library: the first per-flavor dir in order (cuda).
        assert_eq!(
            find_onnxruntime_from(None, None, &root).library,
            Some(root.join("cuda").join(lib))
        );
        // A legacy flat install wins over unmarked per-flavor dirs...
        std::fs::write(root.join(lib), b"").unwrap();
        assert_eq!(
            find_onnxruntime_from(None, None, &root).library,
            Some(root.join(lib))
        );
        // ...but active.txt wins over the flat install.
        write_active_flavor(&root, "cpu").unwrap();
        assert_eq!(read_active_flavor(&root).as_deref(), Some("cpu"));
        assert_eq!(
            find_onnxruntime_from(None, None, &root).library,
            Some(root.join("cpu").join(lib))
        );
        // An active flavor that is not installed falls through.
        write_active_flavor(&root, "directml").unwrap();
        assert_eq!(
            find_onnxruntime_from(None, None, &root).library,
            Some(root.join(lib))
        );
        assert!(write_active_flavor(&root, "../evil").is_err());
        std::fs::write(root.join(ORT_ACTIVE_FILE), "nonsense").unwrap();
        assert_eq!(read_active_flavor(&root), None);
        std::fs::remove_dir_all(&root).ok();
    }
}
