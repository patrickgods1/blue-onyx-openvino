//! Model families: how to build the input tensor(s) for a model and how to turn its raw
//! outputs into detections in original-image pixel coordinates.
//!
//! Everything here is pure CPU code with no OpenVINO dependency so it is unit-testable.

pub mod classes;
pub mod nms;
pub mod preprocess;
pub mod rtdetr;
pub mod yolo26;
pub mod yolo5;
pub mod yolo8;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ModelFamilyKind {
    /// Pick from the model's output shapes at load time.
    #[default]
    Auto,
    /// Ultralytics YOLO26 exported end-to-end (`nms=False`): output `[1, 300, 6]`.
    Yolo26,
    /// YOLOv5 (incl. MikeLud ipcam models): output `[1, N, 5+C]`, needs NMS.
    Yolo5,
    /// YOLOv8/11 raw: output `[1, 4+C, 8400]`, needs NMS.
    Yolo8,
    /// RT-DETRv2 ONNX: inputs `images` + `orig_target_sizes`, outputs `labels/boxes/scores`.
    RtDetr,
}

impl std::fmt::Display for ModelFamilyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ModelFamilyKind::Auto => "auto",
            ModelFamilyKind::Yolo26 => "yolo26",
            ModelFamilyKind::Yolo5 => "yolo5",
            ModelFamilyKind::Yolo8 => "yolo8",
            ModelFamilyKind::RtDetr => "rtdetr",
        };
        f.write_str(s)
    }
}

/// How the source image is fitted into the model input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Keep aspect ratio, pad with gray (114) and center. Used by YOLO families.
    Letterbox,
    /// Stretch to the input size. Used by RT-DETR.
    Stretch,
}

/// Geometry recorded during preprocessing so boxes can be mapped back to the original image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocessCtx {
    pub orig_w: u32,
    pub orig_h: u32,
    pub input_w: u32,
    pub input_h: u32,
    pub mode: ResizeMode,
    /// Letterbox: uniform scale applied to the source. Stretch: unused (1.0).
    pub scale: f32,
    /// Letterbox: left/top padding in input pixels. Stretch: 0.
    pub pad_x: f32,
    pub pad_y: f32,
}

impl PreprocessCtx {
    /// Map a box in model-input pixel coordinates back to original-image coordinates (unclamped).
    pub fn to_original(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> (f32, f32, f32, f32) {
        match self.mode {
            ResizeMode::Letterbox => {
                let s = self.scale.max(f32::EPSILON);
                (
                    (x1 - self.pad_x) / s,
                    (y1 - self.pad_y) / s,
                    (x2 - self.pad_x) / s,
                    (y2 - self.pad_y) / s,
                )
            }
            ResizeMode::Stretch => {
                let sx = self.orig_w as f32 / self.input_w.max(1) as f32;
                let sy = self.orig_h as f32 / self.input_h.max(1) as f32;
                (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            }
        }
    }
}

/// A raw output tensor copied out of the inference request.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputBuf {
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NamedOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: OutputBuf,
}

impl NamedOutput {
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            OutputBuf::F32(v) => Some(v),
            _ => None,
        }
    }
    /// Integer outputs as i64 regardless of storage width.
    pub fn as_i64(&self) -> Option<Vec<i64>> {
        match &self.data {
            OutputBuf::I64(v) => Some(v.clone()),
            OutputBuf::I32(v) => Some(v.iter().map(|&x| x as i64).collect()),
            OutputBuf::F32(_) => None,
        }
    }
}

/// An additional (non-image) input tensor a family needs per request.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtraInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: ExtraData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExtraData {
    I64(Vec<i64>),
    I32(Vec<i32>),
    F32(Vec<f32>),
}

/// Shape/type description of a model port, used for family auto-detection and validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortSpec {
    pub name: String,
    /// -1 for dynamic dimensions.
    pub shape: Vec<i64>,
    pub elem: PortElem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortElem {
    F32,
    F16,
    I64,
    I32,
    U8,
    Other,
}

/// A detection in original-image pixel coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: usize,
}

/// Thresholds applied during post-processing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostParams {
    pub confidence_threshold: f32,
    pub nms_iou: f32,
}

/// Behaviour of one model family. Implementations are stateless apart from the class count.
pub trait Family: Send + Sync {
    fn kind(&self) -> ModelFamilyKind;
    fn resize_mode(&self) -> ResizeMode;
    /// Extra inputs besides the image tensor (RT-DETR's `orig_target_sizes`).
    fn extra_inputs(&self, _ctx: &PreprocessCtx) -> Vec<ExtraInput> {
        Vec::new()
    }
    /// Convert raw outputs to detections in original-image coordinates (unclamped).
    fn postprocess(
        &self,
        outputs: &[NamedOutput],
        ctx: &PreprocessCtx,
        params: &PostParams,
    ) -> Result<Vec<Detection>>;
}

/// Pick a family implementation. `Auto` inspects the port specs.
pub fn make_family(
    kind: ModelFamilyKind,
    inputs: &[PortSpec],
    outputs: &[PortSpec],
    num_classes: usize,
) -> Result<Box<dyn Family>> {
    let kind = if kind == ModelFamilyKind::Auto {
        detect_family(inputs, outputs, num_classes)?
    } else {
        kind
    };
    Ok(match kind {
        ModelFamilyKind::Yolo26 => Box::new(yolo26::Yolo26::new(num_classes)),
        ModelFamilyKind::Yolo5 => Box::new(yolo5::Yolo5::new(num_classes)),
        ModelFamilyKind::Yolo8 => Box::new(yolo8::Yolo8::new(num_classes)),
        ModelFamilyKind::RtDetr => Box::new(rtdetr::RtDetr::new(num_classes)),
        ModelFamilyKind::Auto => unreachable!(),
    })
}

/// Heuristic family detection from port shapes:
/// - an input named `orig_target_sizes`, or 3 outputs including `boxes` -> RtDetr
/// - single output `[.., N, 6]` -> Yolo26 (end-to-end)
/// - single output `[.., 4+C, A]` with A > 4+C -> Yolo8
/// - single output `[.., A, 5+C]` -> Yolo5
pub fn detect_family(
    inputs: &[PortSpec],
    outputs: &[PortSpec],
    num_classes: usize,
) -> Result<ModelFamilyKind> {
    if inputs.iter().any(|p| p.name == "orig_target_sizes")
        || (outputs.len() == 3 && outputs.iter().any(|p| p.name == "boxes"))
    {
        return Ok(ModelFamilyKind::RtDetr);
    }
    if outputs.len() == 1 {
        let s = &outputs[0].shape;
        if s.len() >= 2 {
            let last = s[s.len() - 1];
            let prev = s[s.len() - 2];
            if last == 6 {
                return Ok(ModelFamilyKind::Yolo26);
            }
            if num_classes > 0 && prev == 4 + num_classes as i64 && last > prev {
                return Ok(ModelFamilyKind::Yolo8);
            }
            if num_classes > 0 && last == 5 + num_classes as i64 {
                return Ok(ModelFamilyKind::Yolo5);
            }
            // Fallbacks when the class count is unknown.
            if prev < last && (5..=1000).contains(&prev) {
                return Ok(ModelFamilyKind::Yolo8);
            }
            if (6..=1000).contains(&last) {
                return Ok(ModelFamilyKind::Yolo5);
            }
        }
    }
    anyhow::bail!(
        "could not auto-detect model family from outputs {:?}; set `family` explicitly",
        outputs
            .iter()
            .map(|o| (&o.name, &o.shape))
            .collect::<Vec<_>>()
    )
}

/// Common post-step: class filter (case-insensitive), clamp to the image, round to pixels.
/// Confidence thresholds are applied by the families themselves.
pub fn to_predictions(
    dets: &[Detection],
    ctx: &PreprocessCtx,
    class_names: &[String],
    filter: Option<&[String]>,
) -> Vec<crate::api::Prediction> {
    let w = ctx.orig_w.max(1) as f32;
    let h = ctx.orig_h.max(1) as f32;
    dets.iter()
        .filter_map(|d| {
            let label = class_names
                .get(d.class_id)
                .cloned()
                .unwrap_or_else(|| format!("class_{}", d.class_id));
            if let Some(f) = filter
                && !f.is_empty()
                && !f.iter().any(|x| x.eq_ignore_ascii_case(&label))
            {
                return None;
            }
            let x1 = d.x1.clamp(0.0, w);
            let y1 = d.y1.clamp(0.0, h);
            let x2 = d.x2.clamp(0.0, w);
            let y2 = d.y2.clamp(0.0, h);
            if x2 <= x1 || y2 <= y1 {
                return None;
            }
            Some(crate::api::Prediction {
                x_min: x1.round() as usize,
                y_min: y1.round() as usize,
                x_max: x2.round() as usize,
                y_max: y2.round() as usize,
                confidence: d.score,
                label,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn port(name: &str, shape: &[i64]) -> PortSpec {
        PortSpec {
            name: name.into(),
            shape: shape.to_vec(),
            elem: PortElem::F32,
        }
    }

    #[test]
    fn family_detection() {
        assert_eq!(
            detect_family(&[port("images", &[1, 3, 640, 640])], &[port("output0", &[1, 300, 6])], 80).unwrap(),
            ModelFamilyKind::Yolo26
        );
        assert_eq!(
            detect_family(&[port("images", &[1, 3, 640, 640])], &[port("output0", &[1, 84, 8400])], 80).unwrap(),
            ModelFamilyKind::Yolo8
        );
        assert_eq!(
            detect_family(&[port("images", &[1, 3, 640, 640])], &[port("output", &[1, 25200, 8])], 3).unwrap(),
            ModelFamilyKind::Yolo5
        );
        assert_eq!(
            detect_family(
                &[port("images", &[1, 3, 640, 640]), port("orig_target_sizes", &[1, 2])],
                &[port("labels", &[1, 300]), port("boxes", &[1, 300, 4]), port("scores", &[1, 300])],
                80
            )
            .unwrap(),
            ModelFamilyKind::RtDetr
        );
    }

    #[test]
    fn letterbox_unmapping() {
        // 1920x1080 into 640x640: scale 1/3, pad_y = (640-360)/2 = 140
        let ctx = PreprocessCtx {
            orig_w: 1920,
            orig_h: 1080,
            input_w: 640,
            input_h: 640,
            mode: ResizeMode::Letterbox,
            scale: 1.0 / 3.0,
            pad_x: 0.0,
            pad_y: 140.0,
        };
        let (x1, y1, x2, y2) = ctx.to_original(0.0, 140.0, 640.0, 500.0);
        assert!((x1 - 0.0).abs() < 1e-3 && (y1 - 0.0).abs() < 1e-3);
        assert!((x2 - 1920.0).abs() < 1e-2 && (y2 - 1080.0).abs() < 1e-2);
    }

    #[test]
    fn predictions_filter_and_clamp() {
        let ctx = PreprocessCtx {
            orig_w: 100,
            orig_h: 50,
            input_w: 640,
            input_h: 640,
            mode: ResizeMode::Stretch,
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        };
        let dets = vec![
            Detection { x1: -5.0, y1: 1.0, x2: 20.0, y2: 80.0, score: 0.9, class_id: 0 },
            Detection { x1: 1.0, y1: 1.0, x2: 2.0, y2: 2.0, score: 0.8, class_id: 1 },
        ];
        let names = vec!["person".to_string(), "car".to_string()];
        let filt = vec!["Person".to_string()];
        let p = to_predictions(&dets, &ctx, &names, Some(&filt));
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].label, "person");
        assert_eq!((p[0].x_min, p[0].y_max), (0, 50));
        assert_eq!(to_predictions(&dets, &ctx, &names, None).len(), 2);
    }
}
