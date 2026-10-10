//! On-demand YOLO26 export: **the one place in Blue Onyx Prism that executes downloaded code.**
//!
//! Everything else the service downloads is data (libraries are loaded, never run as programs;
//! archives are extracted by whitelist). YOLO26 weights are AGPL-3.0 and cannot be redistributed,
//! so the only way to offer them is to fetch Ultralytics' official weights and run Ultralytics'
//! exporter on the user's machine. That is a deliberate exception with these limits:
//!
//! - **User-triggered only**: the Config page's Export button (with a confirm naming the size and
//!   the AGPL-3.0 license) or `blue-onyx-prism fetch --resource model:yolo26<size>`. A config entry
//!   pointing at a missing YOLO26 file is reported as "needs export"; `auto_download` never
//!   exports.
//! - **Pinned toolchain**: `tool:uv` (catalog, SHA-256 checked), Python 3.11 installed by that uv,
//!   and the packages of `scripts/yolo26-export/locks/*.txt`, installed with
//!   `--require-hashes --no-deps --only-binary :all:` (every wheel hash-checked, nothing built from
//!   source). On Linux and Windows torch comes from the PyTorch CPU index.
//! - **Pinned weights**: the catalog's `.pt` part, downloaded and verified by the download manager.
//! - **Pinned script**: `scripts/export_yolo26.py`, embedded in the binary, run as
//!   `python -u -I` with `YOLO_OFFLINE=1` and `YOLO_AUTOINSTALL=false` (Ultralytics neither reaches
//!   the network nor pip-installs anything) and its settings in our own `YOLO_CONFIG_DIR`.
//! - **Contained**: everything lives under `<data root>/tools/` (see [`Paths`]); "Remove export
//!   toolchain" deletes it. Inherited `PYTHON*`, `UV_*`, `PIP_*`, `CONDA_*`, `VIRTUAL_ENV` and
//!   `YOLO_*` variables are dropped, uv runs with `--no-config`.
//! - Child processes run with captured output (streamed into the log), a timeout per stage and
//!   cancellation (the child is killed). One export at a time ([`Exporter`] queues the rest; a
//!   lock file serializes processes).
//!
//! Stages ([`Stage`]): uv -> Python -> packages -> weights -> export -> install. The result,
//! `models/yolo26<size>.onnx` + `.yaml`, is moved into place from a staging directory and recorded
//! in `models/.yolo26<size>.installed.json` like a downloaded model.
//!
//! [`run_export`] is the stage machine over two seams, [`CommandRunner`] (processes) and
//! [`Fetcher`] (verified downloads), so it is tested without network or Python.

use super::catalog::{self, Platform, Resource, ResourceKind};
use super::manager::{self, Job, Manager, Manifest, State as DownloadState};
use anyhow::{Context, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::ffi::OsString;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The export script (`scripts/export_yolo26.py`), written into the work directory before it runs.
pub static EXPORT_SCRIPT: &str = include_str!("../../scripts/export_yolo26.py");
/// Hashed package lock for macOS arm64 (PyPI).
pub static LOCK_MACOS_ARM64: &str =
    include_str!("../../scripts/yolo26-export/locks/macos-arm64.txt");
/// Hashed package lock for Linux x86_64 / aarch64 and Windows x86_64 (torch from the PyTorch CPU
/// index).
pub static LOCK_TORCH_CPU_INDEX: &str =
    include_str!("../../scripts/yolo26-export/locks/torch-cpu-index.txt");

/// Python version of the export environment.
pub const PYTHON_VERSION: &str = "3.11";
/// Directory under the data root that holds the whole toolchain.
pub const TOOLS_DIR: &str = "tools";
/// Marker written into the environment once every package is installed.
pub const ENV_MARKER: &str = ".blue-onyx-prism-env.json";

/// Stage timeouts (the child process is killed when one passes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    pub python: Duration,
    pub packages: Duration,
    pub export: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            python: Duration::from_secs(15 * 60),
            packages: Duration::from_secs(60 * 60),
            export: Duration::from_secs(60 * 60),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Paths, locks, toolchain state
// ---------------------------------------------------------------------------------------------

/// Where the export toolchain and its outputs live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    /// `<data root>/tools`.
    pub tools: PathBuf,
    /// `tools/uv` (the `tool:uv` install).
    pub uv_dir: PathBuf,
    /// `tools/yolo26-env`: the Python virtual environment.
    pub env: PathBuf,
    /// `tools/python`: Python installations made by uv.
    pub python_installs: PathBuf,
    /// `tools/uv-cache` (deleted after the packages are installed).
    pub uv_cache: PathBuf,
    /// `tools/yolo26-weights`: verified `.pt` files.
    pub weights: PathBuf,
    /// `tools/yolo26-work`: the script and Ultralytics' intermediate files.
    pub work: PathBuf,
    /// `tools/yolo26-config`: `YOLO_CONFIG_DIR` (Ultralytics settings).
    pub yolo_config: PathBuf,
    /// The configured models directory (where the `.onnx` / `.yaml` go).
    pub models: PathBuf,
}

impl Paths {
    pub fn new(root: &Path, models: PathBuf) -> Self {
        let tools = root.join(TOOLS_DIR);
        Self {
            uv_dir: root.join(catalog::UV_DEST),
            env: tools.join("yolo26-env"),
            python_installs: tools.join("python"),
            uv_cache: tools.join("uv-cache"),
            weights: tools.join("yolo26-weights"),
            work: tools.join("yolo26-work"),
            yolo_config: tools.join("yolo26-config"),
            tools,
            models,
        }
    }

    /// Paths for `config`: data root and `models_dir`.
    pub fn for_config(config: &crate::config::Config) -> Self {
        Self::new(&config.data_root(), config.data_path(&config.models_dir))
    }

    pub fn uv_exe(&self, windows: bool) -> PathBuf {
        self.uv_dir.join(super::extract::uv_binary_name(windows))
    }

    /// The environment's interpreter.
    pub fn env_python(&self, windows: bool) -> PathBuf {
        if windows {
            self.env.join("Scripts").join("python.exe")
        } else {
            self.env.join("bin").join("python")
        }
    }

    /// Cross-process export lock.
    pub fn lock_file(&self) -> PathBuf {
        self.tools.join(".yolo26-export.lock")
    }

    /// Staging directory of one export, next to the models (same filesystem: atomic renames).
    pub fn staging(&self, name: &str) -> PathBuf {
        self.models.join(format!(".{name}.export-staging"))
    }

    /// Everything "Remove export toolchain" deletes.
    pub fn toolchain_dirs(&self) -> Vec<PathBuf> {
        vec![
            self.uv_dir.clone(),
            self.env.clone(),
            self.python_installs.clone(),
            self.uv_cache.clone(),
            self.weights.clone(),
            self.work.clone(),
            self.yolo_config.clone(),
        ]
    }
}

/// The package lock for a platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackageLock {
    /// File name under `scripts/yolo26-export/locks/`.
    pub name: &'static str,
    pub text: &'static str,
    /// torch / torchvision come from the PyTorch CPU index (`--torch-backend cpu`).
    pub torch_cpu_index: bool,
}

/// The package lock for `(os, arch)`; None = YOLO26 cannot be exported there (no PyTorch wheels).
pub fn package_lock(os: &str, arch: &str) -> Option<PackageLock> {
    match (os, arch) {
        ("macos", "aarch64") => Some(PackageLock {
            name: "macos-arm64.txt",
            text: LOCK_MACOS_ARM64,
            torch_cpu_index: false,
        }),
        ("linux", "x86_64") | ("linux", "aarch64") | ("windows", "x86_64") => Some(PackageLock {
            name: "torch-cpu-index.txt",
            text: LOCK_TORCH_CPU_INDEX,
            torch_cpu_index: true,
        }),
        _ => None,
    }
}

/// Why YOLO26 cannot be exported on `(os, arch)`, if it cannot.
pub fn unsupported_reason(os: &str, arch: &str) -> Option<String> {
    let ok = package_lock(os, arch).is_some()
        && catalog::uv_for(os, arch).is_some()
        && catalog::toolchain_estimate(os, arch).is_some();
    (!ok).then(|| {
        format!(
            "YOLO26 export is not available on {os}/{arch} (no PyTorch 2.14 CPU wheels); export \
             on another machine with scripts/export_yolo26.py and copy the .onnx and .yaml"
        )
    })
}

/// `env/.blue-onyx-prism-env.json`: what the environment was built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct EnvMarker {
    lock: String,
    lock_sha256: String,
    python: String,
    uv: String,
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn marker_for(lock: &PackageLock) -> EnvMarker {
    EnvMarker {
        lock: lock.name.to_string(),
        lock_sha256: sha256_hex(lock.text.as_bytes()),
        python: PYTHON_VERSION.to_string(),
        uv: catalog::UV_VERSION.to_string(),
    }
}

/// The environment is complete and built from this binary's lock.
fn env_ready(paths: &Paths, platform: Platform, lock: &PackageLock) -> bool {
    let windows = platform.os == "windows";
    let marker: Option<EnvMarker> = std::fs::read(paths.env.join(ENV_MARKER))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    marker.is_some_and(|m| m == marker_for(lock)) && paths.env_python(windows).is_file()
}

/// What of the toolchain is on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ToolchainState {
    /// `tool:uv` is installed (manifest).
    pub uv: bool,
    /// The Python environment is complete for this binary's lock.
    pub env_ready: bool,
    /// Any toolchain directory exists (something to remove).
    pub present: bool,
}

impl ToolchainState {
    /// An export needs no large download (only the weights).
    pub fn ready(&self) -> bool {
        self.uv && self.env_ready
    }
}

pub fn toolchain_state(paths: &Paths, platform: Platform) -> ToolchainState {
    let uv = catalog::uv_for(platform.os, platform.arch)
        .is_some_and(|r| manager::is_installed(r, &paths.uv_dir))
        && paths.uv_exe(platform.os == "windows").is_file();
    let env_ready =
        package_lock(platform.os, platform.arch).is_some_and(|l| env_ready(paths, platform, &l));
    let present = paths.toolchain_dirs().iter().any(|d| d.exists());
    ToolchainState {
        uv,
        env_ready,
        present,
    }
}

/// `res` is exported into `models`: its `.onnx` is there (with our manifest, or exported by hand
/// with the script).
pub fn is_exported(res: &Resource, models: &Path) -> bool {
    catalog::export_files(res).is_some_and(|(onnx, _)| models.join(onnx).is_file())
}

/// Our manifest of an export of `res` in `models`.
pub fn manifest(res: &Resource, models: &Path) -> Option<Manifest> {
    manager::read_manifest(&manager::manifest_path(res, models)).filter(|m| m.id == res.id)
}

/// Delete the export toolchain (uv, Python, the environment, weights, work files).
pub fn remove_toolchain(paths: &Paths) -> Result<()> {
    for d in paths.toolchain_dirs() {
        remove_any(&d)?;
    }
    let _ = std::fs::remove_file(paths.lock_file());
    let _ = std::fs::remove_file(paths.tools.join(LOCKFILE_NAME));
    // Only when nothing else is in there.
    let _ = std::fs::remove_dir(&paths.tools);
    Ok(())
}

/// Delete the exported model files of `res` (and our manifest).
pub fn remove_export(res: &Resource, models: &Path) -> Result<()> {
    let (onnx, yaml) =
        catalog::export_files(res).ok_or_else(|| anyhow!("{} is not exportable", res.id))?;
    for f in [onnx, yaml] {
        let p = models.join(f);
        if p.exists() {
            std::fs::remove_file(&p).with_context(|| format!("removing {}", p.display()))?;
        }
    }
    let _ = std::fs::remove_file(manager::manifest_path(res, models));
    Ok(())
}

fn remove_any(p: &Path) -> Result<()> {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => {
            std::fs::remove_dir_all(p).with_context(|| format!("removing {}", p.display()))
        }
        Ok(_) => std::fs::remove_file(p).with_context(|| format!("removing {}", p.display())),
        Err(_) => Ok(()),
    }
}

/// Exclusive cross-process export lock; released on drop.
struct ExportLock {
    _file: std::fs::File,
}

impl ExportLock {
    fn acquire(path: &Path, cancel: &AtomicBool, mut waiting: impl FnMut()) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let mut told = false;
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(Self { _file: file }),
                Err(std::fs::TryLockError::WouldBlock) => {}
                Err(std::fs::TryLockError::Error(e)) => {
                    return Err(e).with_context(|| format!("locking {}", path.display()));
                }
            }
            if !told {
                waiting();
                told = true;
            }
            if cancel.load(Ordering::Relaxed) {
                return Err(anyhow!(Cancelled));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Stages and progress
// ---------------------------------------------------------------------------------------------

/// Export stages, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Download and verify `tool:uv`.
    Uv,
    /// `uv python install` + `uv venv`.
    Python,
    /// `uv pip install --require-hashes` of the lock.
    Packages,
    /// Download and verify the `.pt` weights.
    Weights,
    /// Run the export script.
    Export,
    /// Check the output and move it into the models directory.
    Install,
}

impl Stage {
    pub const ALL: [Stage; 6] = [
        Stage::Uv,
        Stage::Python,
        Stage::Packages,
        Stage::Weights,
        Stage::Export,
        Stage::Install,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Stage::Uv => "downloading uv",
            Stage::Python => "installing Python 3.11",
            Stage::Packages => "installing the pinned Python packages",
            Stage::Weights => "downloading the weights",
            Stage::Export => "exporting to ONNX",
            Stage::Install => "installing the model files",
        }
    }

    /// 1-based position.
    pub fn number(self) -> usize {
        Stage::ALL.iter().position(|s| *s == self).unwrap_or(0) + 1
    }

    /// Share of the overall progress bar: `[start, end)` percent.
    fn span(self) -> (u8, u8) {
        match self {
            Stage::Uv => (0, 4),
            Stage::Python => (4, 12),
            Stage::Packages => (12, 70),
            Stage::Weights => (70, 78),
            Stage::Export => (78, 97),
            Stage::Install => (97, 100),
        }
    }

    /// Overall percent when this stage is `pct` percent done (None = just started).
    pub fn overall(self, pct: Option<u8>) -> u8 {
        let (a, b) = self.span();
        let p = u32::from(pct.unwrap_or(0).min(100));
        a + ((u32::from(b - a) * p) / 100) as u8
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// What [`run_export`] reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Progress<'a> {
    /// Entered or advanced a stage (`percent` of the stage when known).
    Stage {
        stage: Stage,
        percent: Option<u8>,
        detail: &'a str,
    },
    /// One line of a child process's output.
    Line { stage: Stage, line: &'a str },
}

/// The export was cancelled.
#[derive(Debug, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

/// Why an export failed, and in which stage.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ExportError {
    pub stage: Option<Stage>,
    pub message: String,
    pub cancelled: bool,
}

/// Result of a successful export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOutput {
    pub onnx: PathBuf,
    pub yaml: PathBuf,
    /// Number of class names written.
    pub classes: usize,
}

// ---------------------------------------------------------------------------------------------
// Seams: processes and downloads
// ---------------------------------------------------------------------------------------------

/// One child process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cmd {
    pub stage: Stage,
    pub program: PathBuf,
    pub args: Vec<OsString>,
    /// Set on top of the (sanitized) inherited environment.
    pub env: Vec<(String, OsString)>,
    pub cwd: PathBuf,
    pub timeout: Duration,
}

impl Cmd {
    /// `program arg arg ...` for logs and tests.
    pub fn display(&self) -> String {
        std::iter::once(self.program.to_string_lossy().into_owned())
            .chain(self.args.iter().map(|a| a.to_string_lossy().into_owned()))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Runs child processes (the real one spawns them; tests fake it).
pub trait CommandRunner: Send + Sync {
    /// Run `cmd` to completion, passing each output line (stdout and stderr) to `on_line`. Fails
    /// on a non-zero exit, on timeout and when `cancel` is set (the child is killed).
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()>;
}

/// Installs a verified resource (the real one goes through the download manager).
pub trait Fetcher: Send + Sync {
    /// Install `res` into `dir` (archives extracted, plain files placed), reporting
    /// `(bytes, total)`. Returns when installed; fails on a download error or `cancel`.
    fn fetch(
        &self,
        res: &'static Resource,
        dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
        cancel: &AtomicBool,
    ) -> Result<()>;
}

/// Variables that could redirect Python, uv, pip or Ultralytics; dropped from the child's
/// environment.
fn sanitized(key: &str) -> bool {
    let k = key.to_ascii_uppercase();
    [
        "PYTHON",
        "UV_",
        "PIP_",
        "CONDA_",
        "YOLO_",
        "VIRTUAL_ENV",
        "__PYVENV",
    ]
    .iter()
    .any(|p| k.starts_with(p))
}

/// Spawns real processes.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemRunner;

impl CommandRunner for SystemRunner {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        use std::process::{Command, Stdio};
        let mut c = Command::new(&cmd.program);
        c.args(&cmd.args)
            .current_dir(&cmd.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, _) in std::env::vars_os() {
            if sanitized(&k.to_string_lossy()) {
                c.env_remove(&k);
            }
        }
        for (k, v) in &cmd.env {
            c.env(k, v);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // CREATE_NO_WINDOW: no console window when the service runs it.
            c.creation_flags(0x0800_0000);
        }
        tracing::info!("running {}", cmd.display());
        let mut child = c
            .spawn()
            .with_context(|| format!("starting {}", cmd.program.display()))?;
        let (tx, rx) = crossbeam_channel::unbounded::<String>();
        let mut readers = Vec::new();
        if let Some(out) = child.stdout.take() {
            readers.push(spawn_reader(out, tx.clone()));
        }
        if let Some(err) = child.stderr.take() {
            readers.push(spawn_reader(err, tx.clone()));
        }
        drop(tx);
        let deadline = Instant::now() + cmd.timeout;
        let mut tail: VecDeque<String> = VecDeque::new();
        let mut keep = |line: String, on_line: &mut dyn FnMut(&str)| {
            on_line(&line);
            if tail.len() == 8 {
                tail.pop_front();
            }
            tail.push_back(line);
        };
        let status = loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(line) => keep(line, on_line),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Output closed: wait for the exit (still honoring cancel and the timeout).
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            if cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(anyhow!(Cancelled));
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                bail!(
                    "{} timed out after {} min and was stopped",
                    cmd.program.display(),
                    cmd.timeout.as_secs() / 60
                );
            }
            if let Some(status) = child.try_wait()? {
                break status;
            }
        };
        for r in readers {
            let _ = r.join();
        }
        while let Ok(line) = rx.try_recv() {
            keep(line, on_line);
        }
        if !status.success() {
            let tail: Vec<String> = tail.into_iter().collect();
            bail!(
                "{} failed ({status}); last output: {}",
                cmd.program.file_name().map_or_else(
                    || cmd.program.display().to_string(),
                    |n| n.to_string_lossy().into_owned()
                ),
                tail.join(" | ")
            );
        }
        Ok(())
    }
}

fn spawn_reader(
    stream: impl std::io::Read + Send + 'static,
    tx: Sender<String>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let reader = std::io::BufReader::new(stream);
        // Split on \n and \r (progress bars rewrite one line with \r).
        for chunk in reader.split(b'\n') {
            let Ok(chunk) = chunk else { break };
            for part in chunk.split(|b| *b == b'\r') {
                let line = strip_ansi(&String::from_utf8_lossy(part))
                    .trim_end()
                    .to_string();
                if !line.is_empty() && tx.send(line).is_err() {
                    return;
                }
            }
        }
    })
}

/// `text` without ANSI escape sequences (Ultralytics colors its log lines).
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // CSI: ESC [ params final-byte (0x40..=0x7e); other escapes: drop ESC + one char.
            if chars.peek() == Some(&'[') {
                chars.next();
                for n in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&n) {
                        break;
                    }
                }
            } else {
                chars.next();
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Verified downloads through the shared [`Manager`] (same queue, lock, resume and SHA-256 checks
/// as every other download).
pub struct ManagerFetcher {
    manager: Arc<Manager>,
}

impl ManagerFetcher {
    pub fn new(manager: Arc<Manager>) -> Self {
        Self { manager }
    }
}

impl Fetcher for ManagerFetcher {
    fn fetch(
        &self,
        res: &'static Resource,
        dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
        cancel: &AtomicBool,
    ) -> Result<()> {
        let job = Job::into_dir(res, dir.to_path_buf());
        let key = job.key();
        // A remembered "installed" from before a removal would make enqueue a no-op.
        self.manager.forget(&key);
        self.manager.enqueue(job);
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(anyhow!(Cancelled));
            }
            match self.manager.status(&key).map(|s| s.state) {
                Some(DownloadState::Installed) => return Ok(()),
                Some(DownloadState::Failed { error, .. }) => {
                    bail!("download of {} failed: {error}", res.title)
                }
                Some(DownloadState::Downloading { bytes, total }) => progress(bytes, total),
                _ => {}
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The stage machine
// ---------------------------------------------------------------------------------------------

/// One export to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportJob {
    pub resource: &'static Resource,
    pub paths: Paths,
    pub platform: Platform,
    pub timeouts: Timeouts,
}

impl ExportJob {
    pub fn new(resource: &'static Resource, paths: Paths) -> Self {
        Self {
            resource,
            paths,
            platform: Platform::current(),
            timeouts: Timeouts::default(),
        }
    }
}

/// Name of the lock file written into `tools/` for `uv pip install -r`.
const LOCKFILE_NAME: &str = "yolo26-requirements.lock.txt";

/// Global uv flags: no user/project config, no colors, our cache.
fn uv_cmd(job: &ExportJob, stage: Stage, timeout: Duration, args: &[&std::ffi::OsStr]) -> Cmd {
    let p = &job.paths;
    let windows = job.platform.os == "windows";
    let mut all: Vec<OsString> = vec!["--no-config".into(), "--color".into(), "never".into()];
    all.push("--cache-dir".into());
    all.push(p.uv_cache.clone().into_os_string());
    all.extend(args.iter().map(|a| a.to_os_string()));
    Cmd {
        stage,
        program: p.uv_exe(windows),
        args: all,
        env: vec![
            (
                "UV_PYTHON_INSTALL_DIR".into(),
                p.python_installs.clone().into_os_string(),
            ),
            ("UV_NO_CONFIG".into(), "1".into()),
            ("UV_NO_MODIFY_PATH".into(), "1".into()),
        ],
        cwd: p.tools.clone(),
        timeout,
    }
}

/// Progress of `uv pip install` from its output: "Downloading torch (75.3MiB)", " Downloaded
/// torch", "Prepared 48 packages in 40s", "Installed 48 packages in 2s".
#[derive(Debug, Default)]
struct PackagesProgress {
    downloading: usize,
    downloaded: usize,
}

impl PackagesProgress {
    fn line(&mut self, line: &str) -> Option<(u8, String)> {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("Downloading ") {
            self.downloading += 1;
            return Some((self.percent(), format!("downloading {rest}")));
        }
        if let Some(rest) = l.strip_prefix("Downloaded ") {
            self.downloaded += 1;
            return Some((self.percent(), format!("downloaded {rest}")));
        }
        if l.starts_with("Prepared ") {
            return Some((90, l.to_string()));
        }
        if l.starts_with("Installed ") {
            return Some((100, l.to_string()));
        }
        None
    }

    fn percent(&self) -> u8 {
        (self.downloaded.min(self.downloading) * 85)
            .checked_div(self.downloading)
            .unwrap_or(0) as u8
    }
}

fn python_line(report: &mut dyn FnMut(Progress<'_>), line: &str) {
    report(Progress::Line {
        stage: Stage::Python,
        line,
    });
}

fn mb(b: u64) -> String {
    format!("{:.1}", b as f64 / 1e6)
}

/// Run every stage of `job` (see the module docs). Progress and child output go to `report`.
pub fn run_export(
    job: &ExportJob,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    report: &mut dyn FnMut(Progress<'_>),
    cancel: &AtomicBool,
) -> Result<ExportOutput, ExportError> {
    let mut stage: Option<Stage> = None;
    let result = run_stages(job, runner, fetcher, report, cancel, &mut stage);
    result.map_err(|e| {
        let cancelled = e.downcast_ref::<Cancelled>().is_some() || cancel.load(Ordering::Relaxed);
        ExportError {
            stage,
            message: if cancelled {
                "cancelled".to_string()
            } else {
                format!("{e:#}")
            },
            cancelled,
        }
    })
}

fn run_stages(
    job: &ExportJob,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    report: &mut dyn FnMut(Progress<'_>),
    cancel: &AtomicBool,
    current: &mut Option<Stage>,
) -> Result<ExportOutput> {
    let res = job.resource;
    let p = &job.paths;
    let (os, arch) = (job.platform.os, job.platform.arch);
    let windows = os == "windows";
    if res.kind != ResourceKind::ExportModel {
        bail!("{} is not an exportable model", res.id);
    }
    let name = res
        .model_name()
        .ok_or_else(|| anyhow!("{} has no model name", res.id))?;
    let (onnx_name, yaml_name) =
        catalog::export_files(res).ok_or_else(|| anyhow!("{} is not exportable", res.id))?;
    if let Some(why) = unsupported_reason(os, arch) {
        bail!(why);
    }
    let lock = package_lock(os, arch).ok_or_else(|| anyhow!("no package lock"))?;
    let check = |cancel: &AtomicBool| -> Result<()> {
        if cancel.load(Ordering::Relaxed) {
            Err(anyhow!(Cancelled))
        } else {
            Ok(())
        }
    };
    macro_rules! enter {
        ($stage:expr, $pct:expr, $detail:expr) => {{
            *current = Some($stage);
            report(Progress::Stage {
                stage: $stage,
                percent: $pct,
                detail: $detail,
            });
        }};
    }

    std::fs::create_dir_all(&p.tools).with_context(|| format!("creating {}", p.tools.display()))?;
    enter!(Stage::Uv, None, "checking the toolchain");
    let _lock = ExportLock::acquire(&p.lock_file(), cancel, || {
        tracing::info!("another process is exporting a YOLO26 model; waiting");
    })?;

    // 1. uv
    let uv_res = catalog::uv_for(os, arch).ok_or_else(|| anyhow!("no uv for {os}/{arch}"))?;
    let uv = p.uv_exe(windows);
    if manager::is_installed(uv_res, &p.uv_dir) && uv.is_file() {
        enter!(Stage::Uv, Some(100), "uv is installed");
    } else {
        enter!(Stage::Uv, Some(0), uv_res.title);
        fetcher.fetch(
            uv_res,
            &p.uv_dir,
            &mut |b, t| {
                let pct = (b * 100).checked_div(t).unwrap_or(0) as u8;
                let d = format!("{} of {} MB", mb(b), mb(t));
                report(Progress::Stage {
                    stage: Stage::Uv,
                    percent: Some(pct),
                    detail: &d,
                });
            },
            cancel,
        )?;
        if !uv.is_file() {
            bail!("{} is missing after installing uv", uv.display());
        }
    }
    check(cancel)?;

    // 2 + 3. Python and the packages
    if env_ready(p, job.platform, &lock) {
        enter!(Stage::Python, Some(100), "reusing the Python environment");
        enter!(Stage::Packages, Some(100), "packages already installed");
    } else {
        let _ = std::fs::remove_file(p.env.join(ENV_MARKER));
        enter!(Stage::Python, Some(0), "uv python install 3.11");
        let mut py_args: Vec<&std::ffi::OsStr> = vec![
            "python".as_ref(),
            "install".as_ref(),
            PYTHON_VERSION.as_ref(),
            "--install-dir".as_ref(),
            p.python_installs.as_os_str(),
            "--no-bin".as_ref(),
        ];
        if windows {
            py_args.push("--no-registry".as_ref());
        }
        let cmd = uv_cmd(job, Stage::Python, job.timeouts.python, &py_args);
        runner.run(&cmd, &mut |l| python_line(report, l), cancel)?;
        check(cancel)?;
        enter!(Stage::Python, Some(60), "uv venv");
        let cmd = uv_cmd(
            job,
            Stage::Python,
            job.timeouts.python,
            &[
                "venv".as_ref(),
                "--python".as_ref(),
                PYTHON_VERSION.as_ref(),
                "--managed-python".as_ref(),
                "--clear".as_ref(),
                p.env.as_os_str(),
            ],
        );
        runner.run(&cmd, &mut |l| python_line(report, l), cancel)?;
        check(cancel)?;

        enter!(Stage::Packages, Some(0), lock.name);
        let lockfile = p.tools.join(LOCKFILE_NAME);
        std::fs::write(&lockfile, lock.text)
            .with_context(|| format!("writing {}", lockfile.display()))?;
        let python = p.env_python(windows);
        let mut args: Vec<&std::ffi::OsStr> = vec![
            "pip".as_ref(),
            "install".as_ref(),
            "--python".as_ref(),
            python.as_os_str(),
            "--require-hashes".as_ref(),
            "--no-deps".as_ref(),
            "--only-binary".as_ref(),
            ":all:".as_ref(),
            "-r".as_ref(),
            lockfile.as_os_str(),
        ];
        if lock.torch_cpu_index {
            args.push(std::ffi::OsStr::new("--torch-backend"));
            args.push(std::ffi::OsStr::new("cpu"));
        }
        let cmd = uv_cmd(job, Stage::Packages, job.timeouts.packages, &args);
        let mut pp = PackagesProgress::default();
        runner.run(
            &cmd,
            &mut |l| {
                report(Progress::Line {
                    stage: Stage::Packages,
                    line: l,
                });
                if let Some((pct, detail)) = pp.line(l) {
                    report(Progress::Stage {
                        stage: Stage::Packages,
                        percent: Some(pct),
                        detail: &detail,
                    });
                }
            },
            cancel,
        )?;
        check(cancel)?;
        if !python.is_file() {
            bail!(
                "{} is missing after creating the environment",
                python.display()
            );
        }
        // The wheels are installed (copied, hard-linked or cloned); the cache is not needed.
        let _ = std::fs::remove_dir_all(&p.uv_cache);
        let marker = serde_json::to_vec_pretty(&marker_for(&lock)).map_err(|e| anyhow!(e))?;
        std::fs::write(p.env.join(ENV_MARKER), marker)
            .with_context(|| format!("writing the marker in {}", p.env.display()))?;
        enter!(Stage::Packages, Some(100), "packages installed");
    }
    check(cancel)?;

    // 4. Weights
    let part = res
        .parts
        .first()
        .ok_or_else(|| anyhow!("{} has no weights", res.id))?;
    let weights = p.weights.join(part.file_name);
    if manager::is_installed(res, &p.weights) && weights.is_file() {
        enter!(Stage::Weights, Some(100), "weights already downloaded");
    } else {
        enter!(Stage::Weights, Some(0), part.file_name);
        fetcher.fetch(
            res,
            &p.weights,
            &mut |b, t| {
                let pct = (b * 100).checked_div(t).unwrap_or(0) as u8;
                let d = format!("{} {} of {} MB", part.file_name, mb(b), mb(t));
                report(Progress::Stage {
                    stage: Stage::Weights,
                    percent: Some(pct),
                    detail: &d,
                });
            },
            cancel,
        )?;
        if !weights.is_file() {
            bail!("{} is missing after the download", weights.display());
        }
    }
    check(cancel)?;

    // 5. Export
    enter!(Stage::Export, Some(0), "starting Ultralytics' exporter");
    std::fs::create_dir_all(&p.work).with_context(|| format!("creating {}", p.work.display()))?;
    std::fs::create_dir_all(&p.yolo_config)
        .with_context(|| format!("creating {}", p.yolo_config.display()))?;
    let script = p.work.join("export_yolo26.py");
    std::fs::write(&script, EXPORT_SCRIPT)
        .with_context(|| format!("writing {}", script.display()))?;
    std::fs::create_dir_all(&p.models)
        .with_context(|| format!("creating {}", p.models.display()))?;
    let staging = p.staging(name);
    remove_any(&staging)?;
    std::fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;
    let size = name.trim_start_matches("yolo26");
    let cmd = Cmd {
        stage: Stage::Export,
        program: p.env_python(windows),
        args: vec![
            "-u".into(),
            "-I".into(),
            script.clone().into_os_string(),
            "--sizes".into(),
            size.into(),
            "--weights".into(),
            weights.clone().into_os_string(),
            "--no-openvino".into(),
            "--out-dir".into(),
            staging.clone().into_os_string(),
            "--work".into(),
            p.work.clone().into_os_string(),
        ],
        env: vec![
            ("YOLO_OFFLINE".into(), "1".into()),
            ("YOLO_AUTOINSTALL".into(), "false".into()),
            (
                "YOLO_CONFIG_DIR".into(),
                p.yolo_config.clone().into_os_string(),
            ),
            ("MPLBACKEND".into(), "Agg".into()),
        ],
        cwd: p.work.clone(),
        timeout: job.timeouts.export,
    };
    let mut lines = 0usize;
    let export = runner.run(
        &cmd,
        &mut |l| {
            report(Progress::Line {
                stage: Stage::Export,
                line: l,
            });
            lines += 1;
            // No real progress from Ultralytics: creep towards 90% with the output.
            let pct = (lines.min(45) * 2) as u8;
            report(Progress::Stage {
                stage: Stage::Export,
                percent: Some(pct),
                detail: l,
            });
        },
        cancel,
    );
    // Ultralytics' copies in the work dir are not needed any more.
    for f in [format!("{name}.pt"), onnx_name.clone()] {
        let _ = std::fs::remove_file(p.work.join(f));
    }
    if let Err(e) = export {
        let _ = remove_any(&staging);
        return Err(e);
    }
    check(cancel)?;

    // 6. Install
    enter!(Stage::Install, Some(0), "checking the output");
    let out = install_outputs(res, &staging, &p.models, &onnx_name, &yaml_name);
    let _ = remove_any(&staging);
    let out = out?;
    enter!(Stage::Install, Some(100), "installed");
    Ok(out)
}

/// Check the staged `.onnx` / `.yaml`, move them into `models` and write the manifest.
fn install_outputs(
    res: &Resource,
    staging: &Path,
    models: &Path,
    onnx_name: &str,
    yaml_name: &str,
) -> Result<ExportOutput> {
    let onnx = staging.join(onnx_name);
    let yaml = staging.join(yaml_name);
    let onnx_len = std::fs::metadata(&onnx)
        .with_context(|| format!("the exporter wrote no {onnx_name}"))?
        .len();
    if onnx_len < 1024 {
        bail!("{onnx_name} is only {onnx_len} bytes");
    }
    let text = std::fs::read_to_string(&yaml)
        .with_context(|| format!("the exporter wrote no {yaml_name}"))?;
    let classes = class_count(&text)?;
    if classes != 80 {
        bail!("{yaml_name} lists {classes} classes; YOLO26 COCO models have 80");
    }
    let shas = vec![manager::sha256_file(&onnx)?, manager::sha256_file(&yaml)?];
    let mut files = Vec::new();
    for (src, file) in [(&onnx, onnx_name), (&yaml, yaml_name)] {
        let dst = models.join(file);
        if std::fs::symlink_metadata(&dst).is_ok() {
            std::fs::remove_file(&dst).with_context(|| format!("replacing {}", dst.display()))?;
        }
        std::fs::rename(src, &dst).with_context(|| format!("moving to {}", dst.display()))?;
        files.push(file.to_string());
    }
    let m = Manifest {
        id: res.id.to_string(),
        version: res.version.to_string(),
        sha256: shas,
        files,
    };
    let path = manager::manifest_path(res, models);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&m).map_err(|e| anyhow!(e))?)
        .with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("writing {}", path.display()))?;
    Ok(ExportOutput {
        onnx: models.join(onnx_name),
        yaml: models.join(yaml_name),
        classes,
    })
}

/// Number of names in a `NAMES:` class file.
fn class_count(text: &str) -> Result<usize> {
    #[derive(Deserialize)]
    struct Names {
        #[serde(rename = "NAMES")]
        names: Vec<String>,
    }
    let n: Names = serde_yaml_ng::from_str(text).context("parsing the class file")?;
    Ok(n.names.len())
}

// ---------------------------------------------------------------------------------------------
// Exporter: one export at a time, in the background (service)
// ---------------------------------------------------------------------------------------------

/// Called with the result after a successful export (e.g. add it to the config).
pub type OnDone = Box<dyn FnOnce(&ExportOutput) + Send>;

/// One queued export.
pub struct ExportRequest {
    pub job: ExportJob,
    pub on_done: Option<OnDone>,
}

impl std::fmt::Debug for ExportRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportRequest")
            .field("job", &self.job)
            .field("on_done", &self.on_done.is_some())
            .finish()
    }
}

/// Where an export is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ExportState {
    Queued,
    Running {
        stage: Stage,
        /// 1-based stage number of [`Stage::ALL`].
        step: usize,
        steps: usize,
        /// Overall percent.
        percent: u8,
        detail: String,
    },
    Installed,
    Failed {
        stage: Option<Stage>,
        error: String,
    },
    Cancelled,
}

impl ExportState {
    pub fn is_busy(&self) -> bool {
        matches!(self, ExportState::Queued | ExportState::Running { .. })
    }

    /// "exporting: installing the pinned Python packages (3/6): downloading torch".
    pub fn describe(&self) -> String {
        match self {
            ExportState::Queued => "export queued".to_string(),
            ExportState::Running {
                stage,
                step,
                steps,
                detail,
                ..
            } => {
                if detail.is_empty() {
                    format!("exporting: {} ({step}/{steps})", stage.label())
                } else {
                    format!("exporting: {} ({step}/{steps}): {detail}", stage.label())
                }
            }
            ExportState::Installed => "exported".to_string(),
            ExportState::Failed {
                stage: Some(s),
                error,
            } => format!("export failed while {}: {error}", s.label()),
            ExportState::Failed { stage: None, error } => format!("export failed: {error}"),
            ExportState::Cancelled => "export cancelled".to_string(),
        }
    }
}

/// Shared status of one export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExportStatus {
    pub id: &'static str,
    pub title: &'static str,
    #[serde(flatten)]
    pub state: ExportState,
    /// Last lines of the child processes' output.
    pub log: Vec<String>,
}

/// Lines of output kept per export.
const LOG_TAIL: usize = 40;

struct Shared {
    runner: Arc<dyn CommandRunner>,
    fetcher: Arc<dyn Fetcher>,
    statuses: Mutex<BTreeMap<&'static str, ExportStatus>>,
    changed: Condvar,
    /// The running export's id and cancel flag.
    current: Mutex<Option<(&'static str, Arc<AtomicBool>)>>,
    /// Queued exports cancelled before they started.
    cancelled: Mutex<HashSet<&'static str>>,
    stop: AtomicBool,
}

impl Shared {
    fn update(&self, id: &'static str, f: impl FnOnce(&mut ExportStatus)) {
        let mut map = self.statuses.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = map.get_mut(id) {
            f(s);
        }
        drop(map);
        self.changed.notify_all();
    }

    fn set_state(&self, id: &'static str, state: ExportState) {
        self.update(id, |s| s.state = state);
    }
}

enum Msg {
    Job(Box<ExportRequest>),
    Stop,
}

/// Background exporter: one thread, one export at a time, the rest queued.
pub struct Exporter {
    shared: Arc<Shared>,
    tx: Sender<Msg>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Exporter {
    pub fn start(runner: Arc<dyn CommandRunner>, fetcher: Arc<dyn Fetcher>) -> Self {
        let shared = Arc::new(Shared {
            runner,
            fetcher,
            statuses: Mutex::new(BTreeMap::new()),
            changed: Condvar::new(),
            current: Mutex::new(None),
            cancelled: Mutex::new(HashSet::new()),
            stop: AtomicBool::new(false),
        });
        let (tx, rx) = crossbeam_channel::unbounded();
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("yolo26-export".into())
            .spawn(move || worker_loop(&worker, &rx))
            .expect("spawning the export thread");
        Self {
            shared,
            tx,
            thread: Mutex::new(Some(thread)),
        }
    }

    /// Queue an export. Refused while the same model is queued or exporting.
    pub fn enqueue(&self, req: ExportRequest) -> Result<(), String> {
        let res = req.job.resource;
        {
            let mut map = self
                .shared
                .statuses
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if map.get(res.id).is_some_and(|s| s.state.is_busy()) {
                return Err(format!("{} is already being exported", res.title));
            }
            map.insert(
                res.id,
                ExportStatus {
                    id: res.id,
                    title: res.title,
                    state: ExportState::Queued,
                    log: Vec::new(),
                },
            );
        }
        self.shared
            .cancelled
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(res.id);
        self.shared.changed.notify_all();
        self.tx
            .send(Msg::Job(Box::new(req)))
            .map_err(|_| "the export thread has stopped".to_string())
    }

    /// Cancel a queued or running export. Returns whether there was one.
    pub fn cancel(&self, id: &str) -> bool {
        let Some(st) = self.status(id) else {
            return false;
        };
        match st.state {
            ExportState::Queued => {
                self.shared
                    .cancelled
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(st.id);
                self.shared.set_state(st.id, ExportState::Cancelled);
                true
            }
            ExportState::Running { .. } => {
                let cur = self
                    .shared
                    .current
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some((cid, flag)) = cur.as_ref()
                    && cid.eq_ignore_ascii_case(id)
                {
                    flag.store(true, Ordering::Relaxed);
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    pub fn status(&self, id: &str) -> Option<ExportStatus> {
        let map = self
            .shared
            .statuses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.values()
            .find(|s| s.id.eq_ignore_ascii_case(id))
            .cloned()
    }

    pub fn statuses(&self) -> Vec<ExportStatus> {
        let map = self
            .shared
            .statuses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.values().cloned().collect()
    }

    /// An export is queued or running.
    pub fn busy(&self) -> bool {
        self.statuses().iter().any(|s| s.state.is_busy())
    }

    /// Drop a finished status (after its files were removed).
    pub fn forget(&self, id: &str) {
        let mut map = self
            .shared
            .statuses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.retain(|k, s| !(k.eq_ignore_ascii_case(id) && !s.state.is_busy()));
    }

    /// Block until `id` is no longer busy or `timeout` passes.
    pub fn wait(&self, id: &str, timeout: Option<Duration>) -> Option<ExportStatus> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut map = self
            .shared
            .statuses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        loop {
            let st = map
                .values()
                .find(|s| s.id.eq_ignore_ascii_case(id))
                .cloned();
            if st.as_ref().is_none_or(|s| !s.state.is_busy()) {
                return st;
            }
            let wait = match deadline {
                Some(d) => match d.checked_duration_since(Instant::now()) {
                    Some(left) => left,
                    None => return st,
                },
                None => Duration::from_secs(1),
            };
            map = self
                .shared
                .changed
                .wait_timeout(map, wait)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Stop the worker (a running export is cancelled: its child process is killed) and join.
    pub fn shutdown(&self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some((_, flag)) = self
            .shared
            .current
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            flag.store(true, Ordering::Relaxed);
        }
        let _ = self.tx.send(Msg::Stop);
        let handle = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }
}

impl Drop for Exporter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(shared: &Shared, rx: &Receiver<Msg>) {
    while let Ok(msg) = rx.recv() {
        let req = match msg {
            Msg::Stop => return,
            Msg::Job(r) => *r,
        };
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        let res = req.job.resource;
        let id = res.id;
        let skipped = shared
            .cancelled
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id);
        if skipped {
            continue;
        }
        let flag = Arc::new(AtomicBool::new(false));
        *shared.current.lock().unwrap_or_else(|e| e.into_inner()) = Some((id, flag.clone()));
        tracing::info!(model = %res.model_name().unwrap_or(id), "YOLO26 export started ({})", catalog::YOLO26_LICENSE);
        let started = Instant::now();
        let mut last_stage: Option<Stage> = None;
        let mut report = |p: Progress<'_>| match p {
            Progress::Stage {
                stage,
                percent,
                detail,
            } => {
                if last_stage != Some(stage) {
                    tracing::info!(model = %res.model_name().unwrap_or(id), "export stage {}/{}: {}", stage.number(), Stage::ALL.len(), stage.label());
                    last_stage = Some(stage);
                }
                shared.set_state(
                    id,
                    ExportState::Running {
                        stage,
                        step: stage.number(),
                        steps: Stage::ALL.len(),
                        percent: stage.overall(percent),
                        detail: detail.chars().take(200).collect(),
                    },
                );
            }
            Progress::Line { line, .. } => {
                tracing::info!(target: "blue_onyx_prism::resources::export::child", model = %res.model_name().unwrap_or(id), "{line}");
                shared.update(id, |s| {
                    if s.log.len() == LOG_TAIL {
                        s.log.remove(0);
                    }
                    s.log.push(line.chars().take(300).collect());
                });
            }
        };
        let result = run_export(
            &req.job,
            shared.runner.as_ref(),
            shared.fetcher.as_ref(),
            &mut report,
            &flag,
        );
        *shared.current.lock().unwrap_or_else(|e| e.into_inner()) = None;
        match result {
            Ok(out) => {
                tracing::info!(
                    model = %res.model_name().unwrap_or(id),
                    "YOLO26 export finished in {:.0} s: {} ({} classes)",
                    started.elapsed().as_secs_f64(),
                    out.onnx.display(),
                    out.classes
                );
                // Before the state flips, so a waiter sees the callback's effect.
                if let Some(cb) = req.on_done {
                    cb(&out);
                }
                shared.set_state(id, ExportState::Installed);
            }
            Err(e) if e.cancelled => {
                tracing::warn!(model = %res.model_name().unwrap_or(id), "YOLO26 export cancelled");
                shared.set_state(id, ExportState::Cancelled);
            }
            Err(e) => {
                tracing::warn!(
                    model = %res.model_name().unwrap_or(id),
                    "YOLO26 export failed{}: {}",
                    e.stage.map(|s| format!(" while {}", s.label())).unwrap_or_default(),
                    e.message
                );
                shared.set_state(
                    id,
                    ExportState::Failed {
                        stage: e.stage,
                        error: e.message,
                    },
                );
            }
        }
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
    }
}

/// Run one export now (CLI): progress lines and the child output go to stderr.
pub fn export_now(
    job: &ExportJob,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    verbose: bool,
) -> Result<ExportOutput> {
    let cancel = AtomicBool::new(false);
    let mut last: Option<(Stage, u8)> = None;
    let name = job.resource.model_name().unwrap_or(job.resource.id);
    let mut report = |p: Progress<'_>| {
        let mut err = std::io::stderr().lock();
        match p {
            Progress::Stage {
                stage,
                percent,
                detail,
            } => {
                let overall = stage.overall(percent);
                // One line per stage change and per 10% of overall progress.
                let show = match last {
                    Some((s, o)) => s != stage || overall / 10 != o / 10,
                    None => true,
                };
                if show {
                    let _ = writeln!(
                        err,
                        "[{name}] {overall:>3}% {}/{} {}: {detail}",
                        stage.number(),
                        Stage::ALL.len(),
                        stage.label()
                    );
                    last = Some((stage, overall));
                }
            }
            Progress::Line { line, .. } => {
                if verbose {
                    let _ = writeln!(err, "    {line}");
                }
            }
        }
    };
    run_export(job, runner, fetcher, &mut report, &cancel).map_err(|e| {
        anyhow!(
            "export of {name} failed{}: {}",
            e.stage
                .map(|s| format!(" while {}", s.label()))
                .unwrap_or_default(),
            e.message
        )
    })
}

/// A [`ManagerFetcher`] on a fresh download manager for `root` (CLI).
pub fn cli_fetcher(root: &Path) -> ManagerFetcher {
    let mut opts = manager::ManagerOptions::new(root);
    opts.on_progress = Some(manager::cli_progress());
    ManagerFetcher::new(Arc::new(Manager::start(opts)))
}

/// "needs export" text for a configured model file that an export would produce.
pub fn needs_export_message(res: &Resource, path: &Path) -> String {
    format!(
        "needs export: {} is not there yet. YOLO26 ({}) is exported on this machine on request: \
         click Export for {} on the Config page (Resources, YOLO26 section) or run \
         `blue-onyx-prism fetch --resource {}`",
        path.display(),
        catalog::YOLO26_LICENSE,
        res.title,
        res.id
    )
}

/// A fake toolchain for tests (no network, no Python): records commands and writes what the real
/// tools would (manifests of fetched resources, the venv's interpreter, the exporter's output).
#[doc(hidden)]
pub mod fake {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Writes what the real tools would: manifests for fetched resources, the venv's python, the
    /// exporter's output. Records every command.
    #[derive(Default)]
    pub struct Fake {
        pub commands: Mutex<Vec<Cmd>>,
        pub fetched: Mutex<Vec<String>>,
        /// Fail the command whose display contains this.
        pub fail_on: Option<&'static str>,
        /// Block in the command whose display contains this until cancelled.
        pub hang_on: Option<&'static str>,
        /// Classes the fake exporter writes.
        pub classes: usize,
        pub fetch_fails: Option<&'static str>,
        pub runs: AtomicUsize,
    }

    impl Fake {
        pub fn new() -> Self {
            Self {
                classes: 80,
                ..Default::default()
            }
        }

        pub fn cmds(&self) -> Vec<String> {
            self.commands
                .lock()
                .unwrap()
                .iter()
                .map(|c| c.display())
                .collect()
        }
    }

    pub fn arg_after(cmd: &Cmd, flag: &str) -> Option<PathBuf> {
        let i = cmd.args.iter().position(|a| a == flag)?;
        cmd.args.get(i + 1).map(PathBuf::from)
    }

    impl CommandRunner for Fake {
        fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            self.commands.lock().unwrap().push(cmd.clone());
            let d = cmd.display();
            if self.hang_on.is_some_and(|h| d.contains(h)) {
                while !cancel.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(10));
                }
                return Err(anyhow!(Cancelled));
            }
            if self.fail_on.is_some_and(|f| d.contains(f)) {
                on_line("error: something broke");
                bail!(
                    "{} failed (exit status: 2); last output: error: something broke",
                    d
                );
            }
            if d.contains(" venv ") {
                let env = PathBuf::from(cmd.args.last().unwrap());
                let windows = cmd.program.extension().is_some_and(|e| e == "exe");
                let py = if windows {
                    env.join("Scripts/python.exe")
                } else {
                    env.join("bin/python")
                };
                std::fs::create_dir_all(py.parent().unwrap()).unwrap();
                std::fs::write(py, b"#!python").unwrap();
            }
            if d.contains(" pip install ") {
                on_line("Resolved 48 packages in 3ms");
                on_line("Downloading torch (75.3MiB)");
                on_line(" Downloaded torch");
                on_line("Prepared 48 packages in 40s");
                on_line("Installed 48 packages in 2s");
            }
            if d.contains("export_yolo26.py") {
                let size = arg_after(cmd, "--sizes").unwrap();
                let name = format!("yolo26{}", size.display());
                on_line(&format!(
                    "[export_yolo26] {name}: exporting ONNX (opset 17)"
                ));
                let out = arg_after(cmd, "--out-dir").unwrap();
                let names: String = (0..self.classes).map(|i| format!("  - c{i}\n")).collect();
                std::fs::write(out.join(format!("{name}.onnx")), vec![7u8; 4096]).unwrap();
                std::fs::write(out.join(format!("{name}.yaml")), format!("NAMES:\n{names}"))
                    .unwrap();
            }
            Ok(())
        }
    }

    impl Fetcher for Fake {
        fn fetch(
            &self,
            res: &'static Resource,
            dir: &Path,
            progress: &mut dyn FnMut(u64, u64),
            _cancel: &AtomicBool,
        ) -> Result<()> {
            self.fetched.lock().unwrap().push(res.id.to_string());
            if self.fetch_fails == Some(res.id) {
                bail!("hash mismatch for {}", res.parts[0].file_name);
            }
            progress(res.size() / 2, res.size());
            std::fs::create_dir_all(dir).unwrap();
            let mut files = Vec::new();
            if res.kind == ResourceKind::Tool {
                std::fs::write(dir.join("uv"), b"uv").unwrap();
                std::fs::write(dir.join("uv.exe"), b"uv").unwrap();
                files.push("uv".to_string());
            } else {
                for p in res.parts {
                    std::fs::write(dir.join(p.file_name), b"weights").unwrap();
                    files.push(p.file_name.to_string());
                }
            }
            let m = Manifest {
                id: res.id.to_string(),
                version: res.version.to_string(),
                sha256: vec![],
                files,
            };
            std::fs::write(
                manager::manifest_path(res, dir),
                serde_json::to_vec(&m).unwrap(),
            )
            .unwrap();
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{Fake, arg_after};
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-export-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn yolo26n() -> &'static Resource {
        catalog::export_model("yolo26n").unwrap()
    }

    const MAC: Platform = Platform::MACOS_ARM64;
    const LINUX: Platform = Platform::LINUX_X64;

    fn job(root: &Path, platform: Platform) -> ExportJob {
        ExportJob {
            resource: yolo26n(),
            paths: Paths::new(root, root.join("models")),
            platform,
            timeouts: Timeouts::default(),
        }
    }

    fn run(fake: &Fake, j: &ExportJob) -> (Result<ExportOutput, ExportError>, Vec<(Stage, u8)>) {
        let mut stages = Vec::new();
        let cancel = AtomicBool::new(false);
        let r = run_export(
            j,
            fake,
            fake,
            &mut |p| {
                if let Progress::Stage { stage, percent, .. } = p {
                    stages.push((stage, stage.overall(percent)));
                }
            },
            &cancel,
        );
        (r, stages)
    }

    #[test]
    fn full_export_runs_every_stage_in_order() {
        let root = tmp("full");
        let fake = Fake::new();
        let j = job(&root, MAC);
        let (r, stages) = run(&fake, &j);
        let out = r.unwrap();
        assert_eq!(out.classes, 80);
        assert_eq!(out.onnx, root.join("models/yolo26n.onnx"));
        assert!(out.onnx.is_file() && out.yaml.is_file());
        // Stages appear in order and the overall percent never goes back.
        let order: Vec<Stage> = stages.iter().map(|s| s.0).fold(Vec::new(), |mut v, s| {
            if v.last() != Some(&s) {
                v.push(s);
            }
            v
        });
        assert_eq!(order, Stage::ALL);
        assert!(stages.windows(2).all(|w| w[0].1 <= w[1].1), "{stages:?}");
        assert_eq!(stages.last().unwrap().1, 100);
        // Downloads went through the fetcher: uv and the weights.
        assert_eq!(*fake.fetched.lock().unwrap(), ["tool:uv", "model:yolo26n"]);
        let cmds = fake.cmds();
        assert_eq!(cmds.len(), 4, "{cmds:#?}");
        assert!(
            cmds[0].contains(" python install 3.11 --install-dir "),
            "{}",
            cmds[0]
        );
        assert!(cmds[0].contains("--no-bin") && !cmds[0].contains("--no-registry"));
        assert!(cmds[0].contains("--no-config"));
        assert!(cmds[1].contains(" venv --python 3.11 --managed-python --clear "));
        assert!(cmds[2].contains(" pip install --python "));
        assert!(cmds[2].contains("--require-hashes --no-deps --only-binary :all: -r "));
        assert!(!cmds[2].contains("--torch-backend"), "macOS uses PyPI");
        assert!(cmds[3].contains(" -u -I "));
        assert!(cmds[3].contains("--sizes n --weights "));
        assert!(cmds[3].contains("--no-openvino --out-dir "));
        let all = fake.commands.lock().unwrap().clone();
        let export = &all[3];
        assert_eq!(export.program, j.paths.env_python(false));
        for (k, v) in [("YOLO_OFFLINE", "1"), ("YOLO_AUTOINSTALL", "false")] {
            assert!(export.env.iter().any(|(a, b)| a == k && b == v), "{k}");
        }
        assert_eq!(
            arg_after(export, "--weights").unwrap(),
            root.join("tools/yolo26-weights/yolo26n.pt")
        );
        assert_eq!(
            std::fs::read_to_string(root.join("tools/yolo26-work/export_yolo26.py")).unwrap(),
            EXPORT_SCRIPT
        );
        // Manifest, marker, staging gone, cache gone.
        let m = manifest(yolo26n(), &root.join("models")).unwrap();
        assert_eq!(m.version, catalog::YOLO26_EXPORT_VERSION);
        assert_eq!(m.files, ["yolo26n.onnx", "yolo26n.yaml"]);
        assert_eq!(m.sha256.len(), 2);
        assert!(is_exported(yolo26n(), &root.join("models")));
        assert!(!j.paths.staging("yolo26n").exists());
        assert!(toolchain_state(&j.paths, MAC).ready());
        assert!(
            std::fs::read_to_string(root.join("tools/yolo26-requirements.lock.txt"))
                .unwrap()
                .contains("ultralytics==8.4.163")
        );

        // Second export: toolchain reused; only the export runs (weights are kept too).
        let fake2 = Fake::new();
        let (r, _) = run(&fake2, &j);
        r.unwrap();
        assert!(fake2.fetched.lock().unwrap().is_empty());
        let cmds = fake2.cmds();
        assert_eq!(cmds.len(), 1);
        assert!(cmds[0].contains("export_yolo26.py"));

        // Remove: model files and toolchain.
        remove_export(yolo26n(), &root.join("models")).unwrap();
        assert!(!is_exported(yolo26n(), &root.join("models")));
        assert!(manifest(yolo26n(), &root.join("models")).is_none());
        remove_toolchain(&j.paths).unwrap();
        assert!(!toolchain_state(&j.paths, MAC).present);
        assert!(!root.join("tools").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn linux_uses_the_torch_cpu_index_and_windows_paths() {
        let root = tmp("linux");
        let fake = Fake::new();
        run(&fake, &job(&root, LINUX)).0.unwrap();
        let cmds = fake.cmds();
        assert!(cmds[2].contains("--torch-backend cpu"), "{}", cmds[2]);

        let root = tmp("win");
        let fake = Fake::new();
        let j = job(&root, Platform::WINDOWS_X64);
        run(&fake, &j).0.unwrap();
        let all = fake.commands.lock().unwrap().clone();
        assert!(all[0].display().contains("--no-registry"));
        assert_eq!(all[0].program, root.join("tools/uv/uv.exe"));
        assert_eq!(
            all[3].program,
            root.join("tools/yolo26-env/Scripts/python.exe")
        );
    }

    #[test]
    fn unsupported_platform_fails_before_doing_anything() {
        let root = tmp("unsup");
        let fake = Fake::new();
        let (r, _) = run(&fake, &job(&root, Platform::new("macos", "x86_64")));
        let e = r.unwrap_err();
        assert!(
            e.message.contains("not available on macos/x86_64"),
            "{}",
            e.message
        );
        assert!(!e.cancelled);
        assert!(fake.cmds().is_empty() && fake.fetched.lock().unwrap().is_empty());
        assert!(unsupported_reason("macos", "aarch64").is_none());
        assert!(unsupported_reason("linux", "riscv64").is_some());
    }

    #[test]
    fn failures_name_the_stage_and_leave_no_partial_install() {
        // Package install fails: no marker, so the next export rebuilds the environment.
        let root = tmp("fail-pkgs");
        let fake = Fake {
            fail_on: Some(" pip install "),
            ..Fake::new()
        };
        let j = job(&root, MAC);
        let e = run(&fake, &j).0.unwrap_err();
        assert_eq!(e.stage, Some(Stage::Packages));
        assert!(e.message.contains("something broke"), "{}", e.message);
        assert!(!toolchain_state(&j.paths, MAC).env_ready);
        assert!(!is_exported(yolo26n(), &root.join("models")));

        // Exporter fails: staging removed, nothing installed.
        let fake = Fake {
            fail_on: Some("export_yolo26.py"),
            ..Fake::new()
        };
        let e = run(&fake, &j).0.unwrap_err();
        assert_eq!(e.stage, Some(Stage::Export));
        assert!(!j.paths.staging("yolo26n").exists());
        assert!(!is_exported(yolo26n(), &root.join("models")));

        // Wrong output: refused at install.
        let fake = Fake {
            classes: 3,
            ..Fake::new()
        };
        let e = run(&fake, &j).0.unwrap_err();
        assert_eq!(e.stage, Some(Stage::Install));
        assert!(e.message.contains("3 classes"), "{}", e.message);
        assert!(!is_exported(yolo26n(), &root.join("models")));

        // Weights hash mismatch from the fetcher (fresh root: no weights yet).
        let root = tmp("fail-weights");
        let j = job(&root, MAC);
        let fake = Fake {
            fetch_fails: Some("model:yolo26n"),
            ..Fake::new()
        };
        let e = run(&fake, &j).0.unwrap_err();
        assert_eq!(e.stage, Some(Stage::Weights));
        assert!(e.message.contains("hash mismatch"), "{}", e.message);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn exporter_queues_cancels_and_reports() {
        let root = tmp("exporter");
        let fake = Arc::new(Fake {
            hang_on: Some("export_yolo26.py"),
            ..Fake::new()
        });
        let ex = Exporter::start(fake.clone(), fake.clone());
        let mut j = job(&root, MAC);
        ex.enqueue(ExportRequest {
            job: j.clone(),
            on_done: None,
        })
        .unwrap();
        // Same model again while busy: refused.
        assert!(
            ex.enqueue(ExportRequest {
                job: j.clone(),
                on_done: None
            })
            .is_err()
        );
        // A second model queues behind it.
        j.resource = catalog::export_model("yolo26s").unwrap();
        ex.enqueue(ExportRequest {
            job: j.clone(),
            on_done: None,
        })
        .unwrap();
        assert!(ex.busy());
        // Wait until the first export hangs in the exporter.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let st = ex.status("model:yolo26n").unwrap();
            if let ExportState::Running {
                stage: Stage::Export,
                step,
                steps,
                ..
            } = &st.state
            {
                assert_eq!((*step, *steps), (5, 6));
                assert!(
                    st.state
                        .describe()
                        .starts_with("exporting: exporting to ONNX (5/6)")
                );
                assert!(
                    st.log.iter().any(|l| l.contains("Downloaded torch")),
                    "{:?}",
                    st.log
                );
                break;
            }
            assert!(Instant::now() < deadline, "{st:?}");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            ex.status("model:yolo26s").unwrap().state,
            ExportState::Queued
        );
        // Cancel the queued one, then the running one.
        assert!(ex.cancel("model:yolo26s"));
        assert_eq!(
            ex.status("model:yolo26s").unwrap().state,
            ExportState::Cancelled
        );
        assert!(ex.cancel("MODEL:yolo26n"));
        let st = ex
            .wait("model:yolo26n", Some(Duration::from_secs(10)))
            .unwrap();
        assert_eq!(st.state, ExportState::Cancelled);
        assert!(!ex.busy());
        assert!(!is_exported(yolo26n(), &root.join("models")));
        assert!(!j.paths.staging("yolo26n").exists());
        // Nothing to cancel any more.
        assert!(!ex.cancel("model:yolo26n"));
        ex.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn exporter_failure_and_success_with_callback() {
        let root = tmp("exporter2");
        let fake = Arc::new(Fake {
            fetch_fails: Some("model:yolo26n"),
            ..Fake::new()
        });
        let ex = Exporter::start(fake.clone(), fake.clone());
        let j = job(&root, MAC);
        ex.enqueue(ExportRequest {
            job: j.clone(),
            on_done: None,
        })
        .unwrap();
        let st = ex
            .wait("model:yolo26n", Some(Duration::from_secs(10)))
            .unwrap();
        match &st.state {
            ExportState::Failed { stage, error } => {
                assert_eq!(*stage, Some(Stage::Weights));
                assert!(error.contains("hash mismatch"));
            }
            other => panic!("{other:?}"),
        }
        assert!(
            st.state
                .describe()
                .starts_with("export failed while downloading the weights")
        );
        ex.shutdown();

        // Retry with a working fetcher: success and the callback runs.
        let fake = Arc::new(Fake::new());
        let ex = Exporter::start(fake.clone(), fake);
        let (tx, rx) = crossbeam_channel::bounded(1);
        ex.enqueue(ExportRequest {
            job: j,
            on_done: Some(Box::new(move |o: &ExportOutput| {
                let _ = tx.send(o.onnx.clone());
            })),
        })
        .unwrap();
        let st = ex
            .wait("model:yolo26n", Some(Duration::from_secs(10)))
            .unwrap();
        assert_eq!(st.state, ExportState::Installed);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5)).unwrap(),
            root.join("models/yolo26n.onnx")
        );
        ex.forget("model:yolo26n");
        assert!(ex.status("model:yolo26n").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn system_runner_streams_fails_times_out_and_cancels() {
        #[cfg(unix)]
        let (sh, flag) = ("/bin/sh", "-c");
        #[cfg(windows)]
        let (sh, flag) = ("cmd", "/C");
        let dir = tmp("runner");
        let cmd = |script: &str, timeout: Duration| Cmd {
            stage: Stage::Export,
            program: sh.into(),
            args: vec![flag.into(), script.into()],
            env: vec![("BOP_TEST".into(), "hello".into())],
            cwd: dir.clone(),
            timeout,
        };
        let never = AtomicBool::new(false);
        let mut lines = Vec::new();
        #[cfg(unix)]
        let ok = "echo one; echo two 1>&2; echo $BOP_TEST";
        #[cfg(windows)]
        let ok = "echo one& echo two 1>&2& echo %BOP_TEST%";
        SystemRunner
            .run(
                &cmd(ok, Duration::from_secs(30)),
                &mut |l| lines.push(l.trim().to_string()),
                &never,
            )
            .unwrap();
        lines.sort();
        assert_eq!(lines, ["hello", "one", "two"]);
        #[cfg(unix)]
        let bad = "echo bad output; exit 3";
        #[cfg(windows)]
        let bad = "echo bad output& exit 3";
        let e = SystemRunner
            .run(&cmd(bad, Duration::from_secs(30)), &mut |_| {}, &never)
            .unwrap_err();
        assert!(format!("{e:#}").contains("bad output"), "{e:#}");
        #[cfg(unix)]
        {
            let started = Instant::now();
            let e = SystemRunner
                .run(
                    &cmd("sleep 30", Duration::from_millis(300)),
                    &mut |_| {},
                    &never,
                )
                .unwrap_err();
            assert!(format!("{e:#}").contains("timed out"), "{e:#}");
            assert!(started.elapsed() < Duration::from_secs(10));
            let cancel = Arc::new(AtomicBool::new(false));
            let c2 = cancel.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                c2.store(true, Ordering::Relaxed);
            });
            let e = SystemRunner
                .run(
                    &cmd("sleep 30", Duration::from_secs(60)),
                    &mut |_| {},
                    &cancel,
                )
                .unwrap_err();
            assert!(e.downcast_ref::<Cancelled>().is_some(), "{e:#}");
            assert!(started.elapsed() < Duration::from_secs(10));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sanitized_variables_and_package_progress() {
        for k in [
            "PYTHONPATH",
            "PYTHONHOME",
            "UV_INDEX_URL",
            "PIP_INDEX_URL",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "YOLO_CONFIG_DIR",
            "pythonpath",
        ] {
            assert!(sanitized(k), "{k}");
        }
        for k in ["PATH", "HOME", "TMPDIR", "SYSTEMROOT"] {
            assert!(!sanitized(k), "{k}");
        }
        let mut p = PackagesProgress::default();
        assert_eq!(p.line("Resolved 48 packages in 3ms"), None);
        assert_eq!(p.line("Downloading torch (75.3MiB)").unwrap().0, 0);
        assert_eq!(p.line("Downloading opencv-python (40MiB)").unwrap().0, 0);
        assert_eq!(
            p.line(" Downloaded torch").unwrap(),
            (42, "downloaded torch".to_string())
        );
        assert_eq!(p.line("Prepared 48 packages in 40s").unwrap().0, 90);
        assert_eq!(p.line("Installed 48 packages in 2s").unwrap().0, 100);
        assert_eq!(Stage::Uv.overall(None), 0);
        assert_eq!(Stage::Packages.overall(Some(50)), 41);
        assert_eq!(Stage::Install.overall(Some(100)), 100);
        assert_eq!(class_count("NAMES:\n  - a\n  - b\n").unwrap(), 2);
        assert_eq!(
            strip_ansi("\u{1b}[34m\u{1b}[1mONNX:\u{1b}[0m export success \u{2705} 3.4s"),
            "ONNX: export success \u{2705} 3.4s"
        );
        assert_eq!(strip_ansi("plain"), "plain");
        assert!(class_count("nope: 1").is_err());
    }

    #[test]
    fn locks_match_the_requirements_and_platforms() {
        let reqs = include_str!("../../scripts/requirements-yolo26-export.txt");
        for lock in [LOCK_MACOS_ARM64, LOCK_TORCH_CPU_INDEX] {
            for line in reqs.lines().filter(|l| l.contains("==")) {
                let (pkg, ver) = line.split_once("==").unwrap();
                let pinned = lock
                    .lines()
                    .any(|l| l.starts_with(&format!("{pkg}=={ver}")));
                assert!(pinned, "{pkg}=={ver} missing from a lock");
            }
            // Every requirement is hashed.
            let reqs_in_lock = lock
                .lines()
                .filter(|l| l.contains("==") && !l.trim_start().starts_with('#'))
                .count();
            let hashed = lock.split("==").count() - 1;
            assert_eq!(reqs_in_lock, hashed);
            assert!(lock.contains("--hash=sha256:"));
            assert!(!lock.contains("openvino"), "no OpenVINO Python package");
        }
        assert!(LOCK_TORCH_CPU_INDEX.contains("torch==2.14.0+cpu"));
        assert!(LOCK_MACOS_ARM64.contains("torch==2.14.0 "));
        assert!(reqs.contains(&format!("ultralytics=={}", catalog::ULTRALYTICS_VERSION)));
        // Supported platforms agree across the lock, uv and the estimates.
        for p in crate::resources::catalog::SHIPPED_PLATFORMS {
            assert!(package_lock(p.os, p.arch).is_some(), "{p}");
            assert!(unsupported_reason(p.os, p.arch).is_none(), "{p}");
        }
        for e in catalog::TOOLCHAIN_ESTIMATES {
            assert!(package_lock(e.platform.os, e.platform.arch).is_some());
        }
        // The script has the flags the exporter passes.
        for f in [
            "--weights",
            "--no-openvino",
            "--out-dir",
            "--work",
            "--sizes",
        ] {
            assert!(EXPORT_SCRIPT.contains(f), "{f}");
        }
    }

    #[test]
    fn needs_export_text() {
        let t = needs_export_message(yolo26n(), Path::new("models/yolo26n.onnx"));
        assert!(t.starts_with("needs export: models/yolo26n.onnx"));
        assert!(t.contains("fetch --resource model:yolo26n"));
        assert!(t.contains("AGPL-3.0"));
    }
}
