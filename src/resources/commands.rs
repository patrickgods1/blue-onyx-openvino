//! `fetch` and `list-resources` subcommands, on top of the resolver and the download manager.
//! `fetch --resource model:yolo26<size>` runs the YOLO26 export ([`super::export`]) instead of a
//! download.

use super::catalog::{self, Resource, ResourceKind};
use super::manager::{self, Job, ManagerOptions};
use super::resolve::{self, Installed, Resolution};
use crate::backend::detect::HardwareInfo;
use crate::config::Config;
use anyhow::{Result, bail};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// What `fetch` should download.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FetchRequest {
    /// Everything the config needs (also the default when nothing else is asked for).
    pub for_config: bool,
    /// Resource ids.
    pub resources: Vec<String>,
    /// Every resource of this platform.
    pub all_for_platform: bool,
    /// Allow downloads over the large threshold (or config `allow_large_downloads`).
    pub allow_large: bool,
}

/// The jobs for `req`, plus notes to print (skipped large downloads, model errors).
pub fn fetch_jobs(
    config: &Config,
    hw: &HardwareInfo,
    installed: &Installed,
    root: &Path,
    req: &FetchRequest,
) -> Result<(Vec<Job>, Vec<String>)> {
    let allow_large = req.allow_large || config.allow_large_downloads;
    let mut jobs: Vec<Job> = Vec::new();
    let mut notes = Vec::new();
    let push = |jobs: &mut Vec<Job>, job: Job| {
        if !jobs.iter().any(|j| j.key() == job.key()) {
            jobs.push(job);
        }
    };
    let models_dir = config.data_path(&config.models_dir);
    let no_ort_active = installed.active_ort.is_none() && !installed.ort_pinned;

    if req.for_config || (req.resources.is_empty() && !req.all_for_platform) {
        let cfg = Config {
            allow_large_downloads: allow_large,
            ..config.clone()
        };
        let r = resolve::needed(&cfg, hw, installed);
        for m in &r.models {
            if let Some(e) = &m.error {
                notes.push(format!("model '{}': {e}", m.model));
            }
        }
        for n in &r.optional {
            notes.push(format!(
                "skipped {} ({}) for {}: pass --allow-large to download it",
                n.id(),
                catalog::format_size(n.size()),
                n.blocking_models.join(", ")
            ));
        }
        for j in manager::jobs_for(&r, &r.needs, root) {
            push(&mut jobs, j);
        }
    }
    for id in &req.resources {
        let Some(res) = catalog::find(id, &hw.os, &hw.arch) else {
            let ids: Vec<&str> = catalog::for_platform(&hw.os, &hw.arch)
                .map(|r| r.id)
                .collect();
            bail!(
                "unknown resource '{id}' for {}/{}; available: {}",
                hw.os,
                hw.arch,
                ids.join(", ")
            );
        };
        // Exports run after the downloads (see `fetch`), never as a download job.
        if res.kind == ResourceKind::ExportModel {
            continue;
        }
        // Naming a large resource explicitly is the opt-in (like --allow-large).
        if res.is_large() && !allow_large {
            notes.push(format!(
                "{} is {} ({})",
                res.id,
                catalog::format_size(res.size()),
                res.license
            ));
        }
        push(&mut jobs, job_for(res, root, &models_dir, no_ort_active));
    }
    if req.all_for_platform {
        let auto = crate::setup_onnxruntime::auto_flavor(hw);
        // Benchmark image sets are not part of a deployment; name them (`bench:<id>`) to fetch.
        // YOLO26 exports (and their toolchain) run code: only when named.
        for res in catalog::for_platform(&hw.os, &hw.arch).filter(|r| {
            !matches!(
                r.kind,
                ResourceKind::BenchImages | ResourceKind::ExportModel | ResourceKind::Tool
            )
        }) {
            if res.is_large() && !allow_large {
                notes.push(format!(
                    "skipped {} ({}): pass --allow-large to download it",
                    res.id,
                    catalog::format_size(res.size())
                ));
                continue;
            }
            let activate = no_ort_active && res.flavor() == Some(auto);
            push(&mut jobs, job_for(res, root, &models_dir, activate));
        }
    }
    Ok((jobs, notes))
}

fn job_for(res: &'static Resource, root: &Path, models_dir: &Path, activate: bool) -> Job {
    match res.kind {
        ResourceKind::Model => Job::into_dir(res, models_dir.to_path_buf()),
        _ => Job {
            activate_ort: activate && res.flavor().is_some(),
            ..Job::new(res, root)
        },
    }
}

/// Exportable models named in `req` (`model:yolo26s`).
pub fn requested_exports(req: &FetchRequest, os: &str, arch: &str) -> Vec<&'static Resource> {
    let mut out: Vec<&'static Resource> = Vec::new();
    for id in &req.resources {
        if let Some(r) = catalog::find(id, os, arch).filter(|r| r.kind == ResourceKind::ExportModel)
            && !out.iter().any(|o| o.id == r.id)
        {
            out.push(r);
        }
    }
    out
}

/// Export `res` now (CLI). The toolchain download needs `allow_large` unless it is installed.
pub fn export_model(config: &Config, res: &'static Resource, allow_large: bool) -> Result<()> {
    use super::export;
    let platform = catalog::Platform::current();
    let paths = export::Paths::for_config(config);
    if export::is_exported(res, &paths.models) {
        println!(
            "{} is already exported in {} (remove the .onnx to export it again)",
            res.title,
            paths.models.display()
        );
        return Ok(());
    }
    if let Some(why) = export::unsupported_reason(platform.os, platform.arch) {
        bail!(why);
    }
    let toolchain = export::toolchain_state(&paths, platform);
    if !toolchain.ready() && !allow_large {
        let cost =
            catalog::toolchain_estimate(platform.os, platform.arch).map_or_else(String::new, |e| {
                format!(
                    " (~{} download, ~{} on disk)",
                    catalog::format_size(e.download),
                    catalog::format_size(e.disk)
                )
            });
        bail!(
            "exporting {} first sets up the export toolchain in {}{cost} and runs Ultralytics' \
             exporter ({}); pass --allow-large to go ahead",
            res.id,
            paths.tools.display(),
            catalog::YOLO26_LICENSE
        );
    }
    println!("note: {}", catalog::YOLO26_NOTICE);
    println!(
        "Exporting {} into {} (toolchain in {}):",
        res.title,
        paths.models.display(),
        paths.tools.display()
    );
    let fetcher = export::cli_fetcher(&config.data_root());
    let started = std::time::Instant::now();
    let out = export::export_now(
        &export::ExportJob::new(res, paths),
        &export::SystemRunner,
        &fetcher,
        true,
    )?;
    println!(
        "Exported {} in {:.0} s: {} and {} ({} classes). Add it on the Config page (Add to \
         config) or to `models`: {{\"name\": \"{}\", \"path\": \"models/{}\", \"family\": \"yolo26\"}}",
        res.title,
        started.elapsed().as_secs_f64(),
        out.onnx.display(),
        out.yaml.display(),
        out.classes,
        res.model_name().unwrap_or_default(),
        out.onnx
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    );
    Ok(())
}

/// Run `fetch`: resolve, download with progress bars, mark the wanted ORT flavor active, then run
/// the YOLO26 exports that were named.
pub fn fetch(config: &Config, req: &FetchRequest) -> Result<()> {
    let root = resolve::download_root(config);
    let hw = crate::backend::detect::hardware();
    let installed = resolve::detect_installed(config, &root, None);
    let exports = requested_exports(req, &hw.os, &hw.arch);
    let (jobs, notes) = fetch_jobs(config, hw, &installed, &root, req)?;
    for n in &notes {
        println!("note: {n}");
    }
    if jobs.is_empty() {
        if exports.is_empty() {
            println!("Nothing to download.");
        }
    } else {
        let total: u64 = jobs.iter().map(|j| j.resource.size()).sum();
        println!(
            "Fetching {} resource(s), up to {}:",
            jobs.len(),
            catalog::format_size(total)
        );
        for j in &jobs {
            println!("  {} -> {}", j.resource.describe(), j.target.display());
        }
        let mut opts = ManagerOptions::new(&root);
        opts.on_progress = Some(manager::cli_progress());
        manager::install_all(opts, jobs.clone())?;
        for j in &jobs {
            if let Some(note) = super::extract::install_note(j.resource) {
                println!("note: {note}");
            }
        }
    }
    // The config's ORT flavor may already be installed but inactive.
    if req.for_config || (req.resources.is_empty() && !req.all_for_platform) {
        let installed = resolve::detect_installed(config, &root, None);
        let cfg = Config {
            allow_large_downloads: req.allow_large || config.allow_large_downloads,
            ..config.clone()
        };
        let r = resolve::needed(&cfg, hw, &installed);
        if let Some(f) = manager::apply_ort_activation(&r, &root)? {
            println!("ONNX Runtime flavor '{f}' marked active (used from the next start)");
        }
    }
    let allow_large = req.allow_large || config.allow_large_downloads;
    for res in exports {
        export_model(config, res, allow_large)?;
    }
    Ok(())
}

/// `list-resources` as text.
pub fn list_resources(config: &Config, hw: &HardwareInfo, installed: &Installed) -> String {
    let r: Resolution = resolve::needed(config, hw, installed);
    let models_dir = config.data_path(&config.models_dir);
    let mut out = String::new();
    let _ = writeln!(out, "Resources for {}/{}:", hw.os, hw.arch);
    let _ = writeln!(out, "  {:<24} {:>8}  {:<28} TITLE", "ID", "SIZE", "STATE");
    for res in catalog::for_platform(&hw.os, &hw.arch) {
        let state = state_of(res, &r, installed, &models_dir, &config.data_root());
        let _ = writeln!(
            out,
            "  {:<24} {:>8}  {:<28} {}",
            res.id,
            catalog::format_size(res.size()),
            state,
            res.title
        );
    }
    for m in &r.models {
        match (&m.device, &m.error) {
            (_, Some(e)) => {
                let _ = writeln!(out, "model '{}': {e}", m.model);
            }
            (Some(d), None) => {
                let _ = writeln!(out, "model '{}': will run on {d}", m.model);
            }
            _ => {}
        }
    }
    if r.needs.is_empty() {
        let _ = writeln!(out, "Everything the config needs is installed.");
    } else {
        let _ = writeln!(
            out,
            "Missing {} ({}); run `{}`{}.",
            r.needs.len(),
            catalog::format_size(r.total_size()),
            resolve::FETCH_FOR_CONFIG,
            if config.auto_download {
                " or start the service (auto_download is on)"
            } else {
                ""
            }
        );
    }
    out
}

fn state_of(
    res: &Resource,
    r: &Resolution,
    installed: &Installed,
    models_dir: &Path,
    root: &Path,
) -> String {
    let installed_now = match res.kind {
        ResourceKind::Model => {
            manager::is_installed(res, models_dir)
                || res
                    .parts
                    .iter()
                    .all(|p| models_dir.join(p.file_name).is_file())
        }
        ResourceKind::BenchImages | ResourceKind::Tool => {
            manager::is_installed(res, &root.join(res.dest))
        }
        ResourceKind::ExportModel => super::export::is_exported(res, models_dir),
        _ => installed.has(res.id),
    };
    if res.kind == ResourceKind::ExportModel && !installed_now {
        let users: Vec<&str> = r
            .models
            .iter()
            .filter(|m| m.needs_export == Some(res.id))
            .map(|m| m.model.as_str())
            .collect();
        return if users.is_empty() {
            "not exported (export)".to_string()
        } else {
            format!("needs export: {}", users.join(", "))
        };
    }
    if installed_now && res.kind == ResourceKind::ExportModel {
        return "exported".to_string();
    }
    if installed_now {
        let active = res.flavor().is_some() && res.flavor() == installed.active_ort;
        return if active {
            "installed (active)".to_string()
        } else {
            "installed".to_string()
        };
    }
    if let Some(n) = r.needs.iter().find(|n| n.id() == res.id) {
        return format!("needed: {}", n.blocking_models.join(", "));
    }
    if r.optional.iter().any(|n| n.id() == res.id) {
        return "optional (large, --allow-large)".to_string();
    }
    "available".to_string()
}

/// Paths a model resource writes, for `download-models --add-to-config`.
pub fn model_files(res: &Resource, dir: &Path) -> Vec<PathBuf> {
    res.parts.iter().map(|p| dir.join(p.file_name)).collect()
}

/// Total size a server reports for `url`: `Content-Range: bytes 0-0/<total>` of a one-byte
/// range request, else `Content-Length` when the range is ignored.
async fn remote_size(client: &reqwest::Client, url: &str) -> Result<u64> {
    let resp = client
        .get(url)
        .header(reqwest::header::RANGE, "bytes=0-0")
        .send()
        .await?
        .error_for_status()?;
    let total = resp
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.rsplit('/').next())
        .and_then(|v| v.trim().parse::<u64>().ok());
    match total.or(resp.content_length()) {
        Some(n) => Ok(n),
        None => bail!("no size in the response"),
    }
}

/// `list-resources --check-urls`: every distinct catalog URL (all platforms) must answer with
/// exactly the pinned size. Prints one line per URL; an error lists the failures.
pub fn check_urls() -> Result<()> {
    let mut parts: Vec<(&'static str, u64)> = catalog::all()
        .flat_map(|r| r.parts.iter().map(|p| (p.url, p.size)))
        .collect();
    parts.sort();
    parts.dedup();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let client = reqwest::Client::builder()
        .user_agent(concat!("blue-onyx-prism/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let mut failures = Vec::new();
    for (url, size) in &parts {
        let got = rt.block_on(remote_size(&client, url));
        match got {
            Ok(n) if n == *size => println!("ok    {size:>11}  {url}"),
            Ok(n) => {
                println!("FAIL  {size:>11}  {url} (server says {n} bytes)");
                failures.push(format!("{url}: size {n}, pinned {size}"));
            }
            Err(e) => {
                println!("FAIL  {size:>11}  {url} ({e:#})");
                failures.push(format!("{url}: {e:#}"));
            }
        }
    }
    println!("{} URLs checked, {} failed", parts.len(), failures.len());
    if !failures.is_empty() {
        bail!("catalog URLs failed: {}", failures.join("; "));
    }
    Ok(())
}
