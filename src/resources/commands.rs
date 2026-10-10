//! `fetch` and `list-resources` subcommands, on top of the resolver and the download manager.

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
        for res in catalog::for_platform(&hw.os, &hw.arch) {
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

/// Run `fetch`: resolve, download with progress bars, mark the wanted ORT flavor active.
pub fn fetch(config: &Config, req: &FetchRequest) -> Result<()> {
    let root = resolve::download_root(config);
    let hw = crate::backend::detect::hardware();
    let installed = resolve::detect_installed(config, &root, None);
    let (jobs, notes) = fetch_jobs(config, hw, &installed, &root, req)?;
    for n in &notes {
        println!("note: {n}");
    }
    if jobs.is_empty() {
        println!("Nothing to download.");
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
        let state = state_of(res, &r, installed, &models_dir);
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

fn state_of(res: &Resource, r: &Resolution, installed: &Installed, models_dir: &Path) -> String {
    let installed_now = match res.kind {
        ResourceKind::Model => {
            manager::is_installed(res, models_dir)
                || res
                    .parts
                    .iter()
                    .all(|p| models_dir.join(p.file_name).is_file())
        }
        _ => installed.has(res.id),
    };
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
