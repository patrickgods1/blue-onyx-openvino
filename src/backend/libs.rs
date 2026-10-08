//! Locate the OpenVINO runtime libraries before `Core::new()`.
//!
//! `openvino-finder` searches `OPENVINO_INSTALL_DIR`, `INTEL_OPENVINO_DIR`, then the OS library
//! path (`PATH` / `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`) and `/opt/intel/openvino*`. It never
//! looks in the executable directory, so we point `OPENVINO_INSTALL_DIR` at `<exe_dir>/openvino`
//! (archive layout: `runtime/{bin|lib}/<arch>/Release`) when that folder exists.
//! TODO(backend agent): implement `prepare_environment` and `diagnostics`.

use std::path::{Path, PathBuf};

pub const BUNDLED_DIR_NAME: &str = "openvino";

/// Set up environment variables so the finder can locate the runtime.
/// `explicit` (from config `openvino_dir`) wins over `<exe_dir>/openvino`; an already-set
/// `OPENVINO_INSTALL_DIR` is left untouched.
pub fn prepare_environment(explicit: Option<&Path>) {
    let _ = explicit;
}

/// Directory that would be used for the bundled runtime, if present.
pub fn bundled_dir() -> Option<PathBuf> {
    let d = crate::exe_dir().join(BUNDLED_DIR_NAME);
    d.is_dir().then_some(d)
}

/// Human readable summary of where the runtime was looked for (used in error messages).
pub fn diagnostics() -> String {
    String::from("Run `blue-onyx-openvino setup-openvino` to download the OpenVINO runtime.")
}
