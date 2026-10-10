//! Download manager (docs/PLAN.md, "Download manager"): one worker thread, a queue, one download
//! at a time, with shared progress.
//!
//! For each [`Job`]:
//! 1. Skip when the target already holds a matching [`Manifest`] (`.installed.json`).
//! 2. Take the cross-process lock `<root>/.downloads.lock` (`File::try_lock`, an OS-level lock
//!    that is released when the process dies), so the CLI and the service never download at the
//!    same time. Re-check step 1 once it is held.
//! 3. Stream each part over HTTPS to `<file>.partial` (archives under `<root>/.downloads/`,
//!    model files next to their destination), resuming with `Range` when the server allows it.
//! 4. Check the exact size and the pinned SHA-256. A mismatch deletes the file and fails the job
//!    with "hash mismatch" (not retried automatically).
//! 5. Archives: extract the whitelisted entries ([`super::extract`]) into a staging directory
//!    next to the target, write the manifest there, then rename it into place. Model files are
//!    renamed into place one by one.
//! 6. Optionally mark an ONNX Runtime flavor active (`onnxruntime/active.txt`).
//!
//! Failures are retried with a 30 s -> 5 min exponential backoff ([`backoff`]) when
//! [`ManagerOptions::retry`] is set (the service); the CLI fails fast. Every state change is
//! published in [`Status`] (shared, for `/stats` and `/v1/resources`), through the optional
//! [`ManagerOptions::on_progress`] hook (CLI progress bars) and, for terminal states, as an
//! [`Event`] to every [`Manager::subscribe`]r (the registry in phase 7.3).
//!
//! Only `https://` URLs are fetched (plus `http://` to the loopback address, used by tests).

use super::catalog::{ArchiveKind, Layout, Resource, ResourceKind};
use super::extract::{self, Target};
use anyhow::{Context, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// Manifest written into a completed install directory.
pub const MANIFEST_FILE: &str = ".installed.json";
/// Cross-process download lock, under the download root.
pub const LOCK_FILE: &str = ".downloads.lock";
/// Partial and verified archive downloads, under the download root.
pub const DOWNLOADS_DIR: &str = ".downloads";

/// First retry delay.
pub const BACKOFF_MIN: Duration = Duration::from_secs(30);
/// Longest retry delay.
pub const BACKOFF_MAX: Duration = Duration::from_secs(300);

/// Delay before retry number `attempt` (1-based): 30 s, 60 s, 120 s, 240 s, then 300 s.
pub fn backoff(attempt: u32) -> Duration {
    let shift = attempt.saturating_sub(1).min(16);
    BACKOFF_MIN
        .checked_mul(1u32 << shift)
        .unwrap_or(BACKOFF_MAX)
        .min(BACKOFF_MAX)
}

// ---------------------------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------------------------

/// `.installed.json`: what was installed into a directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub id: String,
    pub version: String,
    /// SHA-256 of every downloaded part, in catalog order (empty for local archives).
    pub sha256: Vec<String>,
    /// Installed files, relative to the directory.
    pub files: Vec<String>,
}

/// Manifest location for `resource` installed in `dir`: `<dir>/.installed.json` for runtimes
/// (their directory is exclusive), `<dir>/.<model>.installed.json` for model files (a models
/// directory holds many).
pub fn manifest_path(resource: &Resource, dir: &Path) -> PathBuf {
    match resource.model_name() {
        Some(name) => dir.join(format!(".{name}{MANIFEST_FILE}")),
        None => dir.join(MANIFEST_FILE),
    }
}

/// Read a manifest file; None when absent or unreadable.
pub fn read_manifest(path: &Path) -> Option<Manifest> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

fn write_manifest(path: &Path, m: &Manifest) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(m)?)
        .with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
}

/// `resource` (this version) is installed in `dir` according to its manifest; for models every
/// file must also still exist.
pub fn is_installed(resource: &Resource, dir: &Path) -> bool {
    let Some(m) = read_manifest(&manifest_path(resource, dir)) else {
        return false;
    };
    m.id == resource.id
        && m.version == resource.version
        && (resource.kind != ResourceKind::Model
            || resource
                .parts
                .iter()
                .all(|p| dir.join(p.file_name).is_file()))
}

// ---------------------------------------------------------------------------------------------
// Lock
// ---------------------------------------------------------------------------------------------

/// Exclusive cross-process download lock (`<root>/.downloads.lock`). Released on drop (and by
/// the OS when the process exits).
#[derive(Debug)]
pub struct DownloadLock {
    _file: std::fs::File,
}

impl DownloadLock {
    /// Take the lock if it is free; None when another process (or handle) holds it.
    pub fn try_acquire(root: &Path) -> Result<Option<Self>> {
        std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
        let path = root.join(LOCK_FILE);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        match file.try_lock() {
            Ok(()) => {
                // For humans looking at a stuck lock; the OS lock is what counts.
                let _ = file.set_len(0);
                let _ = writeln!(file, "{}", std::process::id());
                Ok(Some(Self { _file: file }))
            }
            Err(std::fs::TryLockError::WouldBlock) => Ok(None),
            Err(std::fs::TryLockError::Error(e)) => {
                Err(e).with_context(|| format!("locking {}", path.display()))
            }
        }
    }

    /// Take the lock, polling for up to `wait` (forever when None) unless `stop` is set.
    pub fn acquire(root: &Path, wait: Option<Duration>, stop: &AtomicBool) -> Result<Self> {
        let started = Instant::now();
        let mut logged = false;
        loop {
            if let Some(l) = Self::try_acquire(root)? {
                return Ok(l);
            }
            if !logged {
                tracing::info!(
                    "another process is downloading ({}); waiting",
                    root.join(LOCK_FILE).display()
                );
                logged = true;
            }
            if stop.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            if wait.is_some_and(|w| started.elapsed() >= w) {
                return Err(anyhow!(LockBusy));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }
}

/// The download lock is held by another process.
#[derive(Debug, thiserror::Error)]
#[error("another process holds the download lock")]
pub struct LockBusy;

/// A downloaded file does not match its pinned hash (not retried automatically).
#[derive(Debug, thiserror::Error)]
#[error("hash mismatch for {file}: expected sha256 {expected}, got {actual}; the file was deleted")]
pub struct HashMismatch {
    pub file: String,
    pub expected: String,
    pub actual: String,
}

/// Retry this error later? Hash mismatches and refused URLs are permanent.
fn retryable(e: &anyhow::Error) -> bool {
    !(e.downcast_ref::<HashMismatch>().is_some() || e.downcast_ref::<UrlRefused>().is_some())
}

#[derive(Debug, thiserror::Error)]
#[error("refusing to download {0}: only https URLs are allowed")]
struct UrlRefused(String);

// ---------------------------------------------------------------------------------------------
// Jobs, status, events
// ---------------------------------------------------------------------------------------------

/// One resource to install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    pub resource: &'static Resource,
    /// Install directory (runtimes: exclusive directory; models: the models directory).
    pub target: PathBuf,
    /// After installing an ONNX Runtime flavor, make it the active one (`active.txt` in the
    /// target's parent directory).
    pub activate_ort: bool,
    /// Install from these local archives (one per part) instead of downloading; no hash check.
    pub local_archives: Option<Vec<PathBuf>>,
    /// Version recorded in `VERSION` and the manifest (default: the resource's).
    pub version: Option<String>,
    /// Keep downloaded archives in `<root>/.downloads` after installing.
    pub keep_downloads: bool,
}

impl Job {
    /// Install `resource` into `<root>/<resource.dest>`.
    pub fn new(resource: &'static Resource, root: &Path) -> Self {
        Self::into_dir(resource, root.join(resource.dest))
    }

    /// Install `resource` into `dir` (models: the directory the files go in).
    pub fn into_dir(resource: &'static Resource, dir: PathBuf) -> Self {
        Self {
            resource,
            target: dir,
            activate_ort: false,
            local_archives: None,
            version: None,
            keep_downloads: false,
        }
    }

    /// Queue key: resource id and target directory.
    pub fn key(&self) -> String {
        format!("{}@{}", self.resource.id, self.target.display())
    }

    fn version(&self) -> &str {
        self.version.as_deref().unwrap_or(self.resource.version)
    }
}

/// Where a job is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum State {
    Queued,
    /// `bytes` of `total` (all parts) downloaded.
    Downloading {
        bytes: u64,
        total: u64,
    },
    Verifying,
    Extracting,
    Installed,
    /// `retry_at` (Unix seconds) is set when an automatic retry is scheduled.
    Failed {
        error: String,
        retry_at: Option<u64>,
        attempts: u32,
    },
}

impl State {
    /// Installed, or failed without a scheduled retry.
    pub fn is_final(&self) -> bool {
        matches!(
            self,
            State::Installed | State::Failed { retry_at: None, .. }
        )
    }

    /// "downloading 42% (35/83 MB)", "installed", ...
    pub fn describe(&self) -> String {
        match self {
            State::Queued => "queued".to_string(),
            State::Downloading { bytes, total } => {
                let pct = if *total > 0 { bytes * 100 / total } else { 0 };
                format!(
                    "downloading {pct}% ({}/{} MB)",
                    bytes / 1_000_000,
                    total.div_ceil(1_000_000)
                )
            }
            State::Verifying => "verifying".to_string(),
            State::Extracting => "extracting".to_string(),
            State::Installed => "installed".to_string(),
            State::Failed {
                error, retry_at, ..
            } => match retry_at.and_then(|t| {
                let now = unix_now();
                (t > now).then(|| t - now)
            }) {
                Some(s) => format!("failed: {error} (retrying in {s} s)"),
                None => format!("failed: {error}"),
            },
        }
    }
}

/// Shared status of one job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Status {
    /// Queue key ([`Job::key`]).
    pub key: String,
    pub id: &'static str,
    pub title: &'static str,
    pub size: u64,
    pub target: PathBuf,
    #[serde(flatten)]
    pub state: State,
}

/// Terminal outcomes, sent to subscribers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Installed {
        key: String,
        id: &'static str,
        target: PathBuf,
    },
    Failed {
        key: String,
        id: &'static str,
        error: String,
        /// When the next automatic attempt happens (None = no retry).
        retry_at: Option<SystemTime>,
    },
}

/// Progress hook: called on every state change (downloads at most every 100 ms).
pub type ProgressFn = Arc<dyn Fn(&Status) + Send + Sync>;

/// Manager settings.
#[derive(Clone)]
pub struct ManagerOptions {
    /// Download root: holds the lock file and `.downloads/`.
    pub root: PathBuf,
    /// Retry failed jobs with [`backoff`] (the service); the CLI fails fast.
    pub retry: bool,
    /// How long to wait for another process's lock before failing the attempt (None = wait).
    pub lock_wait: Option<Duration>,
    pub on_progress: Option<ProgressFn>,
    /// Retry schedule (tests shorten it).
    pub backoff: fn(u32) -> Duration,
}

impl ManagerOptions {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            retry: false,
            lock_wait: None,
            on_progress: None,
            backoff,
        }
    }
}

impl std::fmt::Debug for ManagerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagerOptions")
            .field("root", &self.root)
            .field("retry", &self.retry)
            .field("lock_wait", &self.lock_wait)
            .finish_non_exhaustive()
    }
}

struct Shared {
    opts: ManagerOptions,
    statuses: Mutex<BTreeMap<String, Status>>,
    changed: Condvar,
    subscribers: Mutex<Vec<Sender<Event>>>,
    stop: AtomicBool,
}

impl Shared {
    fn set(&self, job: &Job, state: State) {
        let status = Status {
            key: job.key(),
            id: job.resource.id,
            title: job.resource.title,
            size: job.resource.size(),
            target: job.target.clone(),
            state,
        };
        if let Some(hook) = &self.opts.on_progress {
            hook(&status);
        }
        let mut map = self.statuses.lock().unwrap_or_else(|e| e.into_inner());
        map.insert(status.key.clone(), status);
        drop(map);
        self.changed.notify_all();
    }

    fn state(&self, key: &str) -> Option<State> {
        let map = self.statuses.lock().unwrap_or_else(|e| e.into_inner());
        map.get(key).map(|s| s.state.clone())
    }

    fn emit(&self, ev: Event) {
        let mut subs = self.subscribers.lock().unwrap_or_else(|e| e.into_inner());
        subs.retain(|s| s.send(ev.clone()).is_ok());
    }
}

enum Msg {
    Job(Box<Job>),
    Stop,
}

/// The download manager. Cheap to share behind an `Arc`; dropping it stops the worker (an
/// unfinished download keeps its `.partial` file for the next start).
pub struct Manager {
    shared: Arc<Shared>,
    tx: Sender<Msg>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Manager {
    /// Start the worker thread.
    pub fn start(opts: ManagerOptions) -> Self {
        let shared = Arc::new(Shared {
            opts,
            statuses: Mutex::new(BTreeMap::new()),
            changed: Condvar::new(),
            subscribers: Mutex::new(Vec::new()),
            stop: AtomicBool::new(false),
        });
        let (tx, rx) = crossbeam_channel::unbounded();
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("downloads".into())
            .spawn(move || worker_loop(&worker, &rx))
            .expect("spawning the download thread");
        Self {
            shared,
            tx,
            thread: Mutex::new(Some(thread)),
        }
    }

    /// Queue `job` unless the same key is already queued, running or installed. Returns the key.
    pub fn enqueue(&self, job: Job) -> String {
        let key = job.key();
        let busy = matches!(
            self.shared.state(&key),
            Some(State::Queued | State::Downloading { .. } | State::Verifying | State::Extracting)
                | Some(State::Installed)
        );
        if !busy {
            self.shared.set(&job, State::Queued);
            let _ = self.tx.send(Msg::Job(Box::new(job)));
        }
        key
    }

    pub fn status(&self, key: &str) -> Option<Status> {
        let map = self
            .shared
            .statuses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.get(key).cloned()
    }

    /// Every job seen so far.
    pub fn statuses(&self) -> Vec<Status> {
        let map = self
            .shared
            .statuses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.values().cloned().collect()
    }

    /// Terminal events from now on (installed / failed).
    pub fn subscribe(&self) -> Receiver<Event> {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut subs = self
            .shared
            .subscribers
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        subs.push(tx);
        rx
    }

    /// Block until every key is final ([`State::is_final`]) or `timeout` passes. Returns the
    /// statuses of `keys`.
    pub fn wait(&self, keys: &[String], timeout: Option<Duration>) -> Vec<Status> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut map = self
            .shared
            .statuses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        loop {
            let done = keys
                .iter()
                .all(|k| map.get(k).is_some_and(|s| s.state.is_final()));
            if done {
                break;
            }
            let wait = match deadline {
                Some(d) => match d.checked_duration_since(Instant::now()) {
                    Some(left) => left,
                    None => break,
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
        keys.iter().filter_map(|k| map.get(k).cloned()).collect()
    }

    /// Stop the worker (the current download is abandoned, its `.partial` kept) and join it.
    pub fn shutdown(&self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Msg::Stop);
        let handle = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }
}

impl Drop for Manager {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Jobs for `needs` of a resolution: runtimes into `<root>/<dest>`, models into their
/// configured directory (resolved against the exe dir). The wanted ORT flavor is marked active
/// once installed (see [`super::resolve::Resolution::ort_activation`]).
pub fn jobs_for(
    resolution: &super::resolve::Resolution,
    needs: &[super::resolve::Need],
    root: &Path,
) -> Vec<Job> {
    needs
        .iter()
        .map(|n| {
            let mut job = match &n.model_dir {
                Some(dir) => Job::into_dir(n.resource, crate::resolve_path(dir)),
                None => Job::new(n.resource, root),
            };
            job.activate_ort =
                n.resource.flavor().is_some() && n.resource.flavor() == resolution.ort_activation();
            job
        })
        .collect()
}

/// Mark the resolution's wanted ORT flavor active when it is already installed under `root`
/// (no download needed). Returns the flavor marked.
pub fn apply_ort_activation(
    resolution: &super::resolve::Resolution,
    root: &Path,
) -> Result<Option<super::catalog::Flavor>> {
    let Some(f) = resolution.ort_activation() else {
        return Ok(None);
    };
    let dir = root.join(f.dest());
    if read_manifest(&dir.join(MANIFEST_FILE)).is_some_and(|m| m.id == f.resource_id()) {
        let ort_root = root.join(crate::backend::libs::ORT_DIR_NAME);
        crate::backend::libs::write_active_flavor(&ort_root, f.as_str())
            .with_context(|| format!("marking {f} active in {}", ort_root.display()))?;
        return Ok(Some(f));
    }
    Ok(None)
}

/// Install `jobs` now (CLI): one manager, no retries, wait for all. Errors list every failure.
pub fn install_all(opts: ManagerOptions, jobs: Vec<Job>) -> Result<Vec<Status>> {
    let manager = Manager::start(ManagerOptions {
        retry: false,
        ..opts
    });
    let keys: Vec<String> = jobs.into_iter().map(|j| manager.enqueue(j)).collect();
    let statuses = manager.wait(&keys, None);
    manager.shutdown();
    let failed: Vec<String> = statuses
        .iter()
        .filter_map(|s| match &s.state {
            State::Failed { error, .. } => Some(format!("{}: {error}", s.id)),
            _ => None,
        })
        .collect();
    if !failed.is_empty() {
        bail!("{}", failed.join("; "));
    }
    Ok(statuses)
}

// ---------------------------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------------------------

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn worker_loop(shared: &Shared, rx: &Receiver<Msg>) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!("download manager: creating tokio runtime: {e}");
            return;
        }
    };
    let client = match http_client() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("download manager: {e:#}");
            return;
        }
    };
    let mut queue: VecDeque<(Job, u32)> = VecDeque::new();
    // (due, job, attempts so far)
    let mut retries: Vec<(Instant, Job, u32)> = Vec::new();
    loop {
        // Collect messages; block only when nothing is runnable.
        let msg = if queue.is_empty() {
            let next_due = retries.iter().map(|(d, _, _)| *d).min();
            match next_due {
                Some(d) => rx
                    .recv_timeout(d.saturating_duration_since(Instant::now()))
                    .ok(),
                None => rx.recv().ok().or(Some(Msg::Stop)),
            }
        } else {
            rx.try_recv().ok()
        };
        match msg {
            Some(Msg::Stop) => return,
            Some(Msg::Job(job)) => {
                let key = job.key();
                retries.retain(|(_, j, _)| j.key() != key);
                if !queue.iter().any(|(j, _)| j.key() == key) {
                    queue.push_back((*job, 0));
                }
                continue;
            }
            None => {}
        }
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        let now = Instant::now();
        let (due, later): (Vec<_>, Vec<_>) = retries.drain(..).partition(|(d, _, _)| *d <= now);
        retries = later;
        for (_, job, attempts) in due {
            shared.set(&job, State::Queued);
            queue.push_back((job, attempts));
        }
        let Some((job, attempts)) = queue.pop_front() else {
            continue;
        };
        match run_job(shared, &client, &rt, &job) {
            Ok(()) => {
                shared.set(&job, State::Installed);
                shared.emit(Event::Installed {
                    key: job.key(),
                    id: job.resource.id,
                    target: job.target.clone(),
                });
            }
            Err(e) => {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                let attempts = attempts + 1;
                let error = format!("{e:#}");
                let retry = shared.opts.retry && retryable(&e);
                let delay = (shared.opts.backoff)(attempts);
                let retry_at = retry.then(|| SystemTime::now() + delay);
                tracing::warn!(
                    "download of {} failed (attempt {attempts}): {error}{}",
                    job.resource.id,
                    if retry {
                        format!("; retrying in {} s", delay.as_secs())
                    } else {
                        String::new()
                    }
                );
                shared.set(
                    &job,
                    State::Failed {
                        error: error.clone(),
                        retry_at: retry_at.map(|t| {
                            t.duration_since(SystemTime::UNIX_EPOCH)
                                .map_or(0, |d| d.as_secs())
                        }),
                        attempts,
                    },
                );
                shared.emit(Event::Failed {
                    key: job.key(),
                    id: job.resource.id,
                    error,
                    retry_at,
                });
                if retry {
                    retries.push((Instant::now() + delay, job, attempts));
                }
            }
        }
    }
}

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(concat!("blue-onyx-prism/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(60))
        .build()
        .context("creating HTTP client")
}

fn check_url(url: &str) -> Result<()> {
    let ok = url.starts_with("https://")
        || url.starts_with("http://127.0.0.1:")
        || url.starts_with("http://localhost:");
    if ok {
        Ok(())
    } else {
        Err(anyhow!(UrlRefused(url.to_string())))
    }
}

/// Install one job (steps 1-6 of the module docs).
fn run_job(
    shared: &Shared,
    client: &reqwest::Client,
    rt: &tokio::runtime::Runtime,
    job: &Job,
) -> Result<()> {
    let res = job.resource;
    let local = job.local_archives.is_some();
    if !local && is_installed(res, &job.target) {
        return Ok(());
    }
    let _lock = DownloadLock::acquire(&shared.opts.root, shared.opts.lock_wait, &shared.stop)?;
    if !local && is_installed(res, &job.target) {
        return Ok(());
    }
    let total = res.size();
    let mut done = 0u64;
    let mut last_report = Instant::now() - Duration::from_secs(1);
    let mut progress = |bytes: u64, force: bool| {
        if force || last_report.elapsed() >= Duration::from_millis(100) {
            last_report = Instant::now();
            shared.set(job, State::Downloading { bytes, total });
        }
    };

    // Model files: verified, then renamed into place next to the target.
    if res.parts.iter().all(|p| p.layout == Layout::File) {
        std::fs::create_dir_all(&job.target)
            .with_context(|| format!("creating {}", job.target.display()))?;
        let mut files = Vec::new();
        for part in res.parts {
            let dest = job.target.join(part.file_name);
            if file_matches(&dest, part.size, part.sha256)? {
                done += part.size;
            } else {
                let partial = job.target.join(format!("{}.partial", part.file_name));
                rt.block_on(download_part(client, part, &partial, &shared.stop, |b| {
                    progress(done + b, false)
                }))?;
                shared.set(job, State::Verifying);
                verify(&partial, part)?;
                std::fs::rename(&partial, &dest)
                    .with_context(|| format!("renaming to {}", dest.display()))?;
                done += part.size;
                progress(done, true);
            }
            files.push(part.file_name.to_string());
        }
        write_manifest(
            &manifest_path(res, &job.target),
            &Manifest {
                id: res.id.to_string(),
                version: job.version().to_string(),
                sha256: res.parts.iter().map(|p| p.sha256.to_string()).collect(),
                files,
            },
        )?;
        return Ok(());
    }

    // Archives: download (or take local ones), extract into staging, swap into place.
    let downloads = shared.opts.root.join(DOWNLOADS_DIR);
    let mut archives: Vec<PathBuf> = Vec::new();
    match &job.local_archives {
        Some(local) => {
            if local.len() != res.parts.len() {
                bail!(
                    "{} needs {} archive(s), got {}",
                    res.id,
                    res.parts.len(),
                    local.len()
                );
            }
            archives.extend(local.iter().cloned());
        }
        None => {
            std::fs::create_dir_all(&downloads)
                .with_context(|| format!("creating {}", downloads.display()))?;
            for part in res.parts {
                let file = downloads.join(part.file_name);
                if !file_matches(&file, part.size, part.sha256)? {
                    let partial = downloads.join(format!("{}.partial", part.file_name));
                    rt.block_on(download_part(client, part, &partial, &shared.stop, |b| {
                        progress(done + b, false)
                    }))?;
                    shared.set(job, State::Verifying);
                    verify(&partial, part)?;
                    std::fs::rename(&partial, &file)
                        .with_context(|| format!("renaming to {}", file.display()))?;
                }
                done += part.size;
                progress(done, true);
                archives.push(file);
            }
        }
    }

    shared.set(job, State::Extracting);
    let platform = res
        .platform
        .unwrap_or_else(super::catalog::Platform::current);
    let target = Target {
        os: platform.os,
        arch: platform.arch,
    };
    let staging = sibling(&job.target, "staging")?;
    remove_any(&staging)?;
    let result = (|| -> Result<BTreeSet<String>> {
        let mut got = BTreeSet::new();
        for (part, archive) in res.parts.iter().zip(&archives) {
            let kind = match &job.local_archives {
                Some(_) => archive_kind_of(archive).or(part.archive),
                None => part.archive,
            }
            .with_context(|| format!("{} is not an archive", archive.display()))?;
            got.extend(extract::extract_part(
                archive,
                kind,
                part.layout,
                res.flavor(),
                target,
                &staging,
            )?);
        }
        extract::finalize(res, job.version(), target, &staging, &mut got)?;
        let manifest = Manifest {
            id: res.id.to_string(),
            version: job.version().to_string(),
            sha256: if local {
                Vec::new()
            } else {
                res.parts.iter().map(|p| p.sha256.to_string()).collect()
            },
            files: got.iter().cloned().collect(),
        };
        write_manifest(&staging.join(MANIFEST_FILE), &manifest)?;
        Ok(got)
    })();
    if let Err(e) = result {
        let _ = remove_any(&staging);
        return Err(e);
    }
    put_in_place(res, &staging, &job.target)?;
    if !local && !job.keep_downloads {
        for a in &archives {
            let _ = std::fs::remove_file(a);
        }
    }
    if job.activate_ort
        && let (Some(flavor), Some(parent)) = (res.flavor(), job.target.parent())
    {
        crate::backend::libs::write_active_flavor(parent, flavor.as_str())
            .with_context(|| format!("marking {flavor} active in {}", parent.display()))?;
    }
    Ok(())
}

fn archive_kind_of(p: &Path) -> Option<ArchiveKind> {
    let n = p.to_string_lossy().to_ascii_lowercase();
    if n.ends_with(".zip") || n.ends_with(".nupkg") || n.ends_with(".whl") {
        Some(ArchiveKind::Zip)
    } else if n.ends_with(".tgz") || n.ends_with(".tar.gz") {
        Some(ArchiveKind::TarGz)
    } else {
        None
    }
}

/// `<parent>/.<name>.<what>` next to `dir` (same filesystem, so renames are atomic).
fn sibling(dir: &Path, what: &str) -> Result<PathBuf> {
    let name = dir
        .file_name()
        .with_context(|| format!("{} has no directory name", dir.display()))?
        .to_string_lossy();
    let parent = dir.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    Ok(parent.join(format!(".{name}.{what}")))
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

/// Move a finished `staging` directory to `target`:
/// - `target` absent: one rename.
/// - `target` is a previous install of the same resource (its manifest says so): rename it
///   aside, rename staging in, delete the old one (restored if the second rename fails).
/// - otherwise (a directory we do not own exclusively, e.g. a legacy flat
///   `onnxruntime/` that also holds per-flavor installs, or a user `--dir`): merge file by
///   file, first removing ONNX Runtime files of a previous install so flavors never mix.
fn put_in_place(res: &Resource, staging: &Path, target: &Path) -> Result<()> {
    let in_use_hint = |e: std::io::Error| {
        anyhow!(e).context(format!(
            "replacing {} (if a running process has its libraries loaded, stop it or restart \
             to finish the update)",
            target.display()
        ))
    };
    if std::fs::symlink_metadata(target).is_err() {
        return std::fs::rename(staging, target).map_err(in_use_hint);
    }
    let owned = read_manifest(&target.join(MANIFEST_FILE)).is_some_and(|m| m.id == res.id);
    if owned {
        let old = sibling(target, &format!("old-{}", std::process::id()))?;
        remove_any(&old)?;
        std::fs::rename(target, &old).map_err(in_use_hint)?;
        if let Err(e) = std::fs::rename(staging, target) {
            let _ = std::fs::rename(&old, target);
            return Err(in_use_hint(e));
        }
        if let Err(e) = remove_any(&old) {
            tracing::warn!(
                "could not remove the previous install {}: {e:#}",
                old.display()
            );
        }
        return Ok(());
    }
    if let ResourceKind::OnnxRuntime(_) = res.kind {
        let windows = res.platform.is_some_and(|p| p.os == "windows");
        if let Ok(rd) = std::fs::read_dir(target) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                let is_file = e.file_type().is_ok_and(|t| t.is_file() || t.is_symlink());
                if is_file && extract::is_ort_file(&name, windows) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
    }
    merge_dir(staging, target)?;
    remove_any(staging)
}

fn merge_dir(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to).with_context(|| format!("creating {}", to.display()))?;
    for e in std::fs::read_dir(from).with_context(|| format!("reading {}", from.display()))? {
        let e = e?;
        let src = e.path();
        let dst = to.join(e.file_name());
        let ty = e.file_type()?;
        if ty.is_dir() {
            merge_dir(&src, &dst)?;
        } else {
            if std::fs::symlink_metadata(&dst).is_ok() {
                remove_any(&dst)?;
            }
            std::fs::rename(&src, &dst).with_context(|| format!("moving {}", dst.display()))?;
        }
    }
    Ok(())
}

/// Lowercase hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// `path` exists with exactly `size` bytes and the pinned hash.
fn file_matches(path: &Path, size: u64, sha256: &str) -> Result<bool> {
    match std::fs::metadata(path) {
        Ok(m) if m.is_file() && m.len() == size => Ok(sha256_file(path)? == sha256),
        _ => Ok(false),
    }
}

/// Check size and hash of a finished download; delete it on mismatch.
fn verify(path: &Path, part: &super::catalog::Part) -> Result<()> {
    let len = std::fs::metadata(path)
        .with_context(|| format!("reading {}", path.display()))?
        .len();
    if len != part.size {
        let _ = std::fs::remove_file(path);
        bail!(
            "size mismatch for {}: expected {} bytes, got {len}; the file was deleted",
            part.file_name,
            part.size
        );
    }
    let actual = sha256_file(path)?;
    if actual != part.sha256 {
        let _ = std::fs::remove_file(path);
        return Err(anyhow!(HashMismatch {
            file: part.file_name.to_string(),
            expected: part.sha256.to_string(),
            actual,
        }));
    }
    Ok(())
}

/// Stream `part` into `partial`, resuming from its current length with a `Range` request.
/// `progress` gets the bytes of this part on disk.
async fn download_part(
    client: &reqwest::Client,
    part: &super::catalog::Part,
    partial: &Path,
    stop: &AtomicBool,
    mut progress: impl FnMut(u64),
) -> Result<()> {
    use futures_util::StreamExt;
    check_url(part.url)?;
    let mut have = std::fs::metadata(partial).map_or(0, |m| m.len());
    if have > part.size {
        std::fs::remove_file(partial).ok();
        have = 0;
    }
    if have == part.size {
        return Ok(());
    }
    let mut req = client.get(part.url);
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("GET {}", part.url))?;
    let status = resp.status();
    let append = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        // Only append when the server resumed exactly where we are.
        let start = resp
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("bytes "))
            .and_then(|v| v.split('-').next())
            .and_then(|v| v.parse::<u64>().ok());
        if start != Some(have) {
            bail!(
                "GET {}: server resumed at {start:?}, expected {have}",
                part.url
            );
        }
        true
    } else if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        std::fs::remove_file(partial).ok();
        bail!("GET {}: range not satisfiable; restarting", part.url);
    } else {
        resp.error_for_status_ref()
            .with_context(|| format!("GET {}", part.url))?;
        false
    };
    if !append {
        have = 0;
    }
    if let Some(parent) = partial.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(partial)
        .with_context(|| format!("opening {}", partial.display()))?;
    progress(have);
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if stop.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let chunk = chunk.with_context(|| format!("downloading {}", part.url))?;
        have += chunk.len() as u64;
        if have > part.size {
            drop(f);
            std::fs::remove_file(partial).ok();
            bail!(
                "{} is larger than the pinned {} bytes; the file was deleted",
                part.url,
                part.size
            );
        }
        f.write_all(&chunk)
            .with_context(|| format!("writing {}", partial.display()))?;
        progress(have);
    }
    f.flush()?;
    if have != part.size {
        bail!(
            "download of {} interrupted at {have} of {} bytes (will resume)",
            part.url,
            part.size
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// CLI progress
// ---------------------------------------------------------------------------------------------

/// Progress hook drawing one indicatif bar per job (CLI).
pub fn cli_progress() -> ProgressFn {
    let multi = indicatif::MultiProgress::new();
    let bars: Mutex<BTreeMap<String, indicatif::ProgressBar>> = Mutex::new(BTreeMap::new());
    let style = indicatif::ProgressStyle::with_template(
        "{msg:40} [{bar:30}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
    )
    .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
    .progress_chars("=> ");
    Arc::new(move |s: &Status| {
        let mut bars = bars.lock().unwrap_or_else(|e| e.into_inner());
        let bar = bars.entry(s.key.clone()).or_insert_with(|| {
            let b = multi.add(indicatif::ProgressBar::new(s.size));
            b.set_style(style.clone());
            b.set_message(s.id.to_string());
            b
        });
        match &s.state {
            State::Queued => {}
            State::Downloading { bytes, total } => {
                bar.set_length(*total);
                bar.set_position(*bytes);
            }
            State::Verifying => bar.set_message(format!("{} (verifying)", s.id)),
            State::Extracting => bar.set_message(format!("{} (extracting)", s.id)),
            State::Installed => {
                bar.set_message(format!("{} installed", s.id));
                bar.finish();
            }
            State::Failed { .. } => bar.abandon_with_message(format!("{} failed", s.id)),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelFamilyKind;
    use crate::resources::catalog::{Part, Platform, Provides};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn backoff_schedule() {
        let s: Vec<u64> = (1..=7).map(|a| backoff(a).as_secs()).collect();
        assert_eq!(s, [30, 60, 120, 240, 300, 300, 300]);
        assert_eq!(backoff(0).as_secs(), 30);
        assert_eq!(backoff(u32::MAX), BACKOFF_MAX);
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-mgr-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    fn sha_hex(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        Sha256::digest(data)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Minimal HTTP/1.1 file server: honors `Range: bytes=N-`; when `cut_first` is set, the
    /// first full response is cut off after half the body (to exercise resume). Counts
    /// requests and ranged requests.
    struct Server {
        url: String,
        requests: Arc<AtomicUsize>,
        ranged: Arc<AtomicUsize>,
    }

    fn serve(body: Vec<u8>, cut_first: bool) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/file.bin", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let ranged = Arc::new(AtomicUsize::new(0));
        let (rq, rg) = (Arc::clone(&requests), Arc::clone(&ranged));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = Vec::new();
                let mut byte = [0u8; 1];
                while !buf.ends_with(b"\r\n\r\n") {
                    if s.read(&mut byte).map_or(true, |n| n == 0) {
                        break;
                    }
                    buf.push(byte[0]);
                }
                let head = String::from_utf8_lossy(&buf).to_ascii_lowercase();
                let n = rq.fetch_add(1, Ordering::SeqCst);
                let start = head
                    .lines()
                    .find_map(|l| l.strip_prefix("range: bytes="))
                    .and_then(|r| r.trim().trim_end_matches('-').parse::<usize>().ok());
                let (status, slice, extra) = match start {
                    Some(st) => {
                        rg.fetch_add(1, Ordering::SeqCst);
                        (
                            "206 Partial Content",
                            &body[st..],
                            format!(
                                "Content-Range: bytes {st}-{}/{}\r\n",
                                body.len() - 1,
                                body.len()
                            ),
                        )
                    }
                    None => ("200 OK", &body[..], String::new()),
                };
                let _ = write!(
                    s,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
                    slice.len()
                );
                let send = if cut_first && n == 0 {
                    &slice[..slice.len() / 2]
                } else {
                    slice
                };
                let _ = s.write_all(send);
                let _ = s.flush();
            }
        });
        Server {
            url,
            requests,
            ranged,
        }
    }

    fn model_resource(url: &str, data: &[u8], sha: Option<String>) -> &'static Resource {
        let part = Part {
            url: leak(url.to_string()),
            sha256: leak(sha.unwrap_or_else(|| sha_hex(data))),
            size: data.len() as u64,
            file_name: "test-model.onnx",
            archive: None,
            layout: Layout::File,
        };
        Box::leak(Box::new(Resource {
            id: "model:test-model",
            kind: ResourceKind::Model,
            version: "1",
            platform: None,
            parts: Box::leak(vec![part].into_boxed_slice()),
            dest: "models",
            provides: Provides::Model {
                name: "test-model",
                family: ModelFamilyKind::Yolo5,
            },
            title: "Model test-model",
            description: "test",
            license: "MIT",
        }))
    }

    fn body() -> Vec<u8> {
        (0..300_000u32).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn download_resumes_after_an_interrupted_transfer() {
        let data = body();
        let server = serve(data.clone(), true);
        let res = model_resource(&server.url, &data, None);
        let root = tmp("resume");
        let opts = ManagerOptions {
            backoff: |_| Duration::from_millis(10),
            retry: true,
            ..ManagerOptions::new(&root)
        };
        let m = Manager::start(opts);
        let events = m.subscribe();
        let key = m.enqueue(Job::into_dir(res, root.join("models")));
        // First attempt fails half way (retry scheduled), the second resumes with Range.
        let first = events.recv_timeout(Duration::from_secs(20)).unwrap();
        assert!(
            matches!(
                first,
                Event::Failed {
                    retry_at: Some(_),
                    ..
                }
            ),
            "{first:?}"
        );
        let second = events.recv_timeout(Duration::from_secs(20)).unwrap();
        assert!(matches!(second, Event::Installed { .. }), "{second:?}");
        assert_eq!(m.status(&key).unwrap().state, State::Installed);
        assert_eq!(
            server.ranged.load(Ordering::SeqCst),
            1,
            "resumed with Range"
        );
        assert_eq!(
            std::fs::read(root.join("models/test-model.onnx")).unwrap(),
            data
        );
        assert!(!root.join("models/test-model.onnx.partial").exists());
        assert!(is_installed(res, &root.join("models")));
        // Manifest round trip.
        let man = read_manifest(&manifest_path(res, &root.join("models"))).unwrap();
        assert_eq!(man.id, "model:test-model");
        assert_eq!(man.files, ["test-model.onnx"]);
        assert_eq!(man.sha256, [res.parts[0].sha256]);
        // Installed: queueing again does nothing.
        let before = server.requests.load(Ordering::SeqCst);
        m.enqueue(Job::into_dir(res, root.join("models")));
        m.shutdown();
        assert_eq!(server.requests.load(Ordering::SeqCst), before);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn hash_mismatch_deletes_the_file_and_is_not_retried() {
        let data = body();
        let server = serve(data.clone(), false);
        let res = model_resource(&server.url, &data, Some("0".repeat(64)));
        let root = tmp("hash");
        let opts = ManagerOptions {
            retry: true,
            ..ManagerOptions::new(&root)
        };
        let m = Manager::start(opts);
        let key = m.enqueue(Job::into_dir(res, root.join("models")));
        let st = m.wait(std::slice::from_ref(&key), Some(Duration::from_secs(20)));
        match &st[0].state {
            State::Failed {
                error, retry_at, ..
            } => {
                assert!(error.contains("hash mismatch"), "{error}");
                assert_eq!(*retry_at, None, "no automatic retry");
            }
            other => panic!("{other:?}"),
        }
        let models = root.join("models");
        assert!(!models.join("test-model.onnx").exists());
        assert!(!models.join("test-model.onnx.partial").exists());
        assert!(!is_installed(res, &models));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn lock_contention_and_install_all() {
        let root = tmp("lock");
        let held = DownloadLock::try_acquire(&root)
            .unwrap()
            .expect("free lock");
        assert!(
            DownloadLock::try_acquire(&root).unwrap().is_none(),
            "a second handle must not get the lock"
        );
        let stop = AtomicBool::new(false);
        let err =
            DownloadLock::acquire(&root, Some(Duration::from_millis(300)), &stop).unwrap_err();
        assert!(err.downcast_ref::<LockBusy>().is_some());

        // A job waits for the lock and fails the attempt when it stays busy.
        let data = body();
        let server = serve(data.clone(), false);
        let res = model_resource(&server.url, &data, None);
        let opts = ManagerOptions {
            lock_wait: Some(Duration::from_millis(300)),
            ..ManagerOptions::new(&root)
        };
        let err = install_all(opts.clone(), vec![Job::into_dir(res, root.join("m"))]).unwrap_err();
        assert!(format!("{err:#}").contains("download lock"), "{err:#}");
        assert_eq!(server.requests.load(Ordering::SeqCst), 0);
        drop(held);
        let st = install_all(opts, vec![Job::into_dir(res, root.join("m"))]).unwrap();
        assert_eq!(st[0].state, State::Installed);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn https_only() {
        assert!(check_url("https://example.com/x").is_ok());
        assert!(check_url("http://127.0.0.1:8080/x").is_ok());
        let e = check_url("http://example.com/x").unwrap_err();
        assert!(!retryable(&e));
        assert!(retryable(&anyhow!("connection reset")));
    }

    /// An ONNX Runtime archive from local files: extracted into staging, swapped into place,
    /// manifest + flavor.txt written, marked active; a second install replaces the first.
    #[test]
    fn local_archive_install_and_swap() {
        let root = tmp("swap");
        let archive = root.join("ort.tgz");
        let make = |content: &[u8]| {
            let f = std::fs::File::create(&archive).unwrap();
            let gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
            let mut tar = tar::Builder::new(gz);
            let mut h = tar::Header::new_gnu();
            h.set_size(content.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            tar.append_data(
                &mut h,
                "onnxruntime-linux-x64-1.24.4/lib/libonnxruntime.so",
                content,
            )
            .unwrap();
            tar.into_inner().unwrap().finish().unwrap();
        };
        let res =
            crate::resources::catalog::onnxruntime("linux", "x86_64", super::super::Flavor::Cpu)
                .unwrap();
        assert_eq!(res.platform, Some(Platform::LINUX_X64));
        let job = || Job {
            local_archives: Some(vec![archive.clone()]),
            activate_ort: true,
            ..Job::new(res, &root)
        };
        make(b"v1");
        install_all(ManagerOptions::new(&root), vec![job()]).unwrap();
        let dir = root.join("onnxruntime/cpu");
        assert_eq!(std::fs::read(dir.join("libonnxruntime.so")).unwrap(), b"v1");
        assert_eq!(
            std::fs::read_to_string(dir.join("flavor.txt"))
                .unwrap()
                .trim(),
            "cpu"
        );
        let man = read_manifest(&dir.join(MANIFEST_FILE)).unwrap();
        assert_eq!(man.id, "onnxruntime-cpu");
        assert!(man.files.contains(&"libonnxruntime.so".to_string()));
        assert_eq!(
            crate::backend::libs::read_active_flavor(&root.join("onnxruntime")).as_deref(),
            Some("cpu")
        );
        make(b"v2");
        install_all(ManagerOptions::new(&root), vec![job()]).unwrap();
        assert_eq!(std::fs::read(dir.join("libonnxruntime.so")).unwrap(), b"v2");
        let leftovers: Vec<_> = std::fs::read_dir(root.join("onnxruntime"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with('.'))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
