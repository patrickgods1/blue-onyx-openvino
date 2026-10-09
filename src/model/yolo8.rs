//! YOLOv8/11 raw head: one output `[1, 4+C, A]`, channel-major (row 0..4 = cx, cy, w, h;
//! rows 4.. = class scores) in model-input pixels. Needs NMS.

use anyhow::Result;

use super::nms::nms;
use super::yolo26::{matrix_dims, meets, primary_f32_output};
use super::{
    Detection, Family, ModelFamilyKind, NamedOutput, PostParams, PreprocessCtx, ResizeMode,
};

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
    fn postprocess(
        &self,
        outputs: &[NamedOutput],
        ctx: &PreprocessCtx,
        params: &PostParams,
    ) -> Result<Vec<Detection>> {
        let (out, data) = primary_f32_output("yolo8", outputs)?;
        let (channels, anchors) = matrix_dims("yolo8", "[1, 4+C, A]", out, data.len())?;
        if channels < 5 {
            anyhow::bail!(
                "yolo8: output `{}` has shape {:?}, expected [1, 4+C, A] with C >= 1",
                out.name,
                out.shape
            );
        }
        if self.num_classes > 0 && channels != 4 + self.num_classes {
            anyhow::bail!(
                "yolo8: output `{}` has shape {:?}, expected [1, {}, A] for {} classes",
                out.name,
                out.shape,
                4 + self.num_classes,
                self.num_classes
            );
        }
        let num_classes = channels - 4;
        let thr = params.confidence_threshold;

        // Pass 1: per-anchor best class score, walking the class rows contiguously.
        let mut best = vec![f32::NEG_INFINITY; anchors];
        let mut best_cls = vec![0usize; anchors];
        for c in 0..num_classes {
            let row = &data[(4 + c) * anchors..(5 + c) * anchors];
            for (j, &v) in row.iter().enumerate() {
                if v > best[j] {
                    best[j] = v;
                    best_cls[j] = c;
                }
            }
        }

        let (cxs, rest) = data.split_at(anchors);
        let (cys, rest) = rest.split_at(anchors);
        let (ws, rest) = rest.split_at(anchors);
        let hs = &rest[..anchors];
        let mut dets = Vec::new();
        for j in 0..anchors {
            let conf = best[j];
            if !meets(conf, thr) {
                continue;
            }
            let (cx, cy, w, h) = (cxs[j], cys[j], ws[j], hs[j]);
            let (x1, y1, x2, y2) =
                ctx.to_original(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0);
            dets.push(Detection {
                x1,
                y1,
                x2,
                y2,
                score: conf,
                class_id: best_cls[j],
            });
        }
        nms(&mut dets, params.nms_iou);
        Ok(dets)
    }
}
