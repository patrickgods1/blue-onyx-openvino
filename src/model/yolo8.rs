//! TODO(model agent): implement.
use super::{Detection, Family, ModelFamilyKind, NamedOutput, PostParams, PreprocessCtx, ResizeMode};

pub struct Yolo8 {
    pub num_classes: usize,
}

impl Yolo8 {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Family for Yolo8 {
    fn kind(&self) -> ModelFamilyKind {
        ModelFamilyKind::Yolo8
    }
    fn resize_mode(&self) -> ResizeMode {
        ResizeMode::Letterbox
    }
    fn postprocess(&self, _outputs: &[NamedOutput], _ctx: &PreprocessCtx, _params: &PostParams) -> anyhow::Result<Vec<Detection>> {
        anyhow::bail!("Yolo8 postprocess not implemented")
    }
}
