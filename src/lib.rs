//! Blue Onyx OpenVINO: a Blue Iris / CodeProject.AI compatible object detection
//! service running on native OpenVINO.
//!
//! Module map (see CLAUDE.md):
//! - [`api`]       wire structs for the CodeProject.AI compatible JSON API
//! - [`config`]    JSON config next to the executable, merged with CLI flags
//! - [`backend`]   runtimes (OpenVINO today), device specs, load plans, inference backends
//! - [`model`]     model families (yolo26, yolo5, yolo8, rtdetr): pre/post-processing
//! - [`registry`]  loads all configured models and owns their worker threads
//! - [`worker`]    per-model inference thread
//! - [`startup`]   model readiness state shared with the HTTP layer
//! - [`server`]    axum HTTP server
//! - [`runner`]    server run loop (generations, restart) shared by the CLI and the service
//! - [`metrics`]   request/latency counters and Prometheus rendering

pub mod api;
pub mod backend;
pub mod cli;
pub mod config;
pub mod download;
pub mod image;
pub mod metrics;
pub mod model;
pub mod registry;
pub mod runner;
pub mod server;
pub mod setup_openvino;
pub mod startup;
pub mod system_info;
pub mod update;
pub mod worker;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const MODULE_ID: &str = "ObjectDetectionOpenVINO";
pub const MODULE_NAME: &str = "Object Detection (OpenVINO)";
pub const PROCESSED_BY: &str = "BlueOnyxOpenVINO";
pub const DEFAULT_PORT: u16 = 32168;

/// Directory containing the running executable (falls back to cwd).
pub fn exe_dir() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
}

/// Resolve a possibly relative path against the executable directory.
pub fn resolve_path(p: &std::path::Path) -> std::path::PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        exe_dir().join(p)
    }
}
