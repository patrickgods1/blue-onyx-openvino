//! On-demand resources (docs/PLAN.md, "On-demand resources"): the runtimes, provider libraries
//! and model files the service can download, and what the current config needs.
//!
//! - [`catalog`]: pinned URLs, SHA-256 and sizes per resource and platform (pure data).
//! - [`resolve`]: [`resolve::needed`] works out the missing resources for a config on given
//!   hardware (pure); [`resolve::detect_installed`] snapshots what is on disk.
//! - [`manager`]: the download manager (queue, verified resumable downloads, atomic installs,
//!   manifests, cross-process lock, backoff, progress and events).
//! - [`commands`]: the `fetch` and `list-resources` subcommands.
//! - [`extract`]: whitelist-only archive extraction shared by the manager and `setup-*`.

pub mod catalog;
pub mod commands;
pub mod extract;
pub mod manager;
pub mod resolve;

pub use catalog::{Flavor, LARGE_DOWNLOAD_BYTES, Platform, Resource, ResourceKind};
pub use manager::{Event, Job, Manager, ManagerOptions, State, Status};
pub use resolve::{
    Installed, ModelPlan, Need, Resolution, detect_installed, download_root, needed,
};
