//! On-demand resources (docs/PLAN.md, "On-demand resources"): the runtimes, provider libraries
//! and model files the service can download, and what the current config needs.
//!
//! - [`catalog`]: pinned URLs, SHA-256 and sizes per resource and platform (pure data).
//! - [`resolve`]: [`resolve::needed`] works out the missing resources for a config on given
//!   hardware (pure); [`resolve::detect_installed`] snapshots what is on disk.

pub mod catalog;
pub mod resolve;

pub use catalog::{Flavor, LARGE_DOWNLOAD_BYTES, Platform, Resource, ResourceKind};
pub use resolve::{
    Installed, ModelPlan, Need, Resolution, detect_installed, download_root, needed,
};
