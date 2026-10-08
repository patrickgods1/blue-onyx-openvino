//! TODO(model agent): implement.
use super::{Detection, Family, ModelFamilyKind, NamedOutput, PostParams, PreprocessCtx, ResizeMode};

pub struct RtDetr {
    pub num_classes: usize,
}

impl RtDetr {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Family for RtDetr {
    fn kind(&self) -> ModelFamilyKind {
        ModelFamilyKind::RtDetr
    }
    fn resize_mode(&self) -> ResizeMode {
        ResizeMode::Stretch
    }
    fn postprocess(&self, _outputs: &[NamedOutput], _ctx: &PreprocessCtx, _params: &PostParams) -> anyhow::Result<Vec<Detection>> {
        anyhow::bail!("RtDetr postprocess not implemented")
    }
}
