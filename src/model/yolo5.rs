//! TODO(model agent): implement.
use super::{Detection, Family, ModelFamilyKind, NamedOutput, PostParams, PreprocessCtx, ResizeMode};

pub struct Yolo5 {
    pub num_classes: usize,
}

impl Yolo5 {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Family for Yolo5 {
    fn kind(&self) -> ModelFamilyKind {
        ModelFamilyKind::Yolo5
    }
    fn resize_mode(&self) -> ResizeMode {
        ResizeMode::Letterbox
    }
    fn postprocess(&self, _outputs: &[NamedOutput], _ctx: &PreprocessCtx, _params: &PostParams) -> anyhow::Result<Vec<Detection>> {
        anyhow::bail!("Yolo5 postprocess not implemented")
    }
}
