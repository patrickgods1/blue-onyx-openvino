//! STUB: replaced by the model agent.
//! Minimal placeholder with the agreed signature so the runtime glue compiles.

use super::{PreprocessCtx, ResizeMode};
use anyhow::Result;

/// STUB: replaced by the model agent.
pub struct Preprocessor {
    input_w: u32,
    input_h: u32,
    mode: ResizeMode,
}

impl Preprocessor {
    /// STUB: replaced by the model agent.
    pub fn new(input_w: u32, input_h: u32, mode: ResizeMode) -> Self {
        Self {
            input_w,
            input_h,
            mode,
        }
    }

    /// STUB: replaced by the model agent.
    pub fn run(&mut self, _rgb: &[u8], w: u32, h: u32) -> Result<(&[f32], PreprocessCtx)> {
        let _ = (w, h, self.input_w, self.input_h, self.mode);
        anyhow::bail!("model::preprocess::Preprocessor::run not implemented (stub)")
    }
}
