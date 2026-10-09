//! STUB: replaced by the backend agent.
//! Minimal placeholder with the agreed signature so the main binary compiles.

use anyhow::Result;
use std::path::PathBuf;

/// STUB: pinned OpenVINO runtime version (the backend agent owns the real constant).
pub const OPENVINO_VERSION: &str = "2026.1.0";

/// STUB: replaced by the backend agent.
#[derive(Debug, Clone)]
pub struct SetupOptions {
    /// Destination directory (normally `<exe_dir>/openvino`).
    pub dest: PathBuf,
    /// OpenVINO version to install.
    pub version: String,
    /// Use an already downloaded archive instead of downloading.
    pub archive: Option<PathBuf>,
    /// Keep the downloaded archive after extraction.
    pub keep_archive: bool,
}

/// STUB: replaced by the backend agent. Returns the directory the runtime was installed into.
pub fn run(opts: &SetupOptions) -> Result<PathBuf> {
    anyhow::bail!(
        "setup_openvino::run not implemented (stub): dest {}, version {}",
        opts.dest.display(),
        opts.version
    )
}
