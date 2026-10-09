//! STUB: replaced by the model agent.
//! Minimal placeholders with the agreed signatures so the runtime glue compiles.

use anyhow::Result;
use std::path::Path;

/// STUB: replaced by the model agent.
pub fn load_class_names(path: &Path) -> Result<Vec<String>> {
    anyhow::bail!(
        "model::classes::load_class_names not implemented (stub): {}",
        path.display()
    )
}

/// STUB: replaced by the model agent.
pub fn coco80() -> Vec<String> {
    Vec::new()
}
