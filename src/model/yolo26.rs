//! TODO(model agent): implement.
use super::{Detection, Family, ModelFamilyKind, NamedOutput, PostParams, PreprocessCtx, ResizeMode};

pub struct Yolo26 {
    pub num_classes: usize,
}

impl Yolo26 {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Family for Yolo26 {
    fn kind(&self) -> ModelFamilyKind {
        ModelFamilyKind::Yolo26
    }
    fn resize_mode(&self) -> ResizeMode {
        ResizeMode::Letterbox
    }
    fn postprocess(&self, _outputs: &[NamedOutput], _ctx: &PreprocessCtx, _params: &PostParams) -> anyhow::Result<Vec<Detection>> {
        anyhow::bail!("Yolo26 postprocess not implemented")
    }
}
