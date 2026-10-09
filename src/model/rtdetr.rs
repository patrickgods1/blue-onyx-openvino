//! RT-DETR(v2) ONNX: inputs `images` + i64 `orig_target_sizes` `[1, 2]`; outputs
//! `labels` `[1, N]` (i64/i32), `boxes` `[1, N, 4]` (xyxy, f32), `scores` `[1, N]` (f32).
//! NMS-free.
//!
//! We feed the *input* size as `orig_target_sizes`, so boxes come back in model-input pixels
//! and `PreprocessCtx::to_original` (stretch) maps them to the original image.

use anyhow::Result;

use super::yolo26::meets;
use super::{
    Detection, ExtraData, ExtraInput, Family, ModelFamilyKind, NamedOutput, PostParams,
    PreprocessCtx, ResizeMode,
};

pub struct RtDetr {
    pub num_classes: usize,
}

impl RtDetr {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

fn find<'a>(outputs: &'a [NamedOutput], name: &str) -> Result<&'a NamedOutput> {
    outputs.iter().find(|o| o.name == name).ok_or_else(|| {
        anyhow::anyhow!(
            "rtdetr: missing output `{name}`; got {:?}",
            outputs
                .iter()
                .map(|o| (&o.name, &o.shape))
                .collect::<Vec<_>>()
        )
    })
}

/// Number of queries for `[1, N, ..tail]` or `[N, ..tail]`.
fn queries(out: &NamedOutput, tail: &[usize], expected: &str) -> Result<usize> {
    let s = out.shape.as_slice();
    let n = if s.len() == tail.len() + 2 && s[0] == 1 && s[2..] == *tail {
        s[1]
    } else if s.len() == tail.len() + 1 && s[1..] == *tail {
        s[0]
    } else {
        anyhow::bail!(
            "rtdetr: output `{}` has shape {:?}, expected {expected}",
            out.name,
            out.shape
        );
    };
    Ok(n)
}

impl Family for RtDetr {
    fn kind(&self) -> ModelFamilyKind {
        ModelFamilyKind::RtDetr
    }
    fn resize_mode(&self) -> ResizeMode {
        ResizeMode::Stretch
    }
    fn extra_inputs(&self, ctx: &PreprocessCtx) -> Vec<ExtraInput> {
        vec![ExtraInput {
            name: "orig_target_sizes".into(),
            shape: vec![1, 2],
            data: ExtraData::I64(vec![ctx.input_w as i64, ctx.input_h as i64]),
        }]
    }
    fn postprocess(
        &self,
        outputs: &[NamedOutput],
        ctx: &PreprocessCtx,
        params: &PostParams,
    ) -> Result<Vec<Detection>> {
        let labels_o = find(outputs, "labels")?;
        let boxes_o = find(outputs, "boxes")?;
        let scores_o = find(outputs, "scores")?;

        let n_labels = queries(labels_o, &[], "[1, N]")?;
        let n_boxes = queries(boxes_o, &[4], "[1, N, 4]")?;
        let n_scores = queries(scores_o, &[], "[1, N]")?;
        if n_labels != n_boxes || n_labels != n_scores {
            anyhow::bail!(
                "rtdetr: inconsistent query counts: labels {:?}, boxes {:?}, scores {:?}",
                labels_o.shape,
                boxes_o.shape,
                scores_o.shape
            );
        }
        let n = n_labels;

        let labels = labels_o.as_i64().ok_or_else(|| {
            anyhow::anyhow!("rtdetr: output `labels` must be i64 or i32, got f32")
        })?;
        let boxes = boxes_o
            .as_f32()
            .ok_or_else(|| anyhow::anyhow!("rtdetr: output `boxes` must be f32"))?;
        let scores = scores_o
            .as_f32()
            .ok_or_else(|| anyhow::anyhow!("rtdetr: output `scores` must be f32"))?;
        if labels.len() != n || boxes.len() != 4 * n || scores.len() != n {
            anyhow::bail!(
                "rtdetr: buffer lengths (labels {}, boxes {}, scores {}) do not match shapes {:?} {:?} {:?}",
                labels.len(),
                boxes.len(),
                scores.len(),
                labels_o.shape,
                boxes_o.shape,
                scores_o.shape
            );
        }

        let mut dets = Vec::new();
        for i in 0..n {
            let score = scores[i];
            if !meets(score, params.confidence_threshold) || labels[i] < 0 {
                continue;
            }
            let b = &boxes[4 * i..4 * i + 4];
            let (x1, y1, x2, y2) = ctx.to_original(b[0], b[1], b[2], b[3]);
            dets.push(Detection {
                x1,
                y1,
                x2,
                y2,
                score,
                class_id: labels[i] as usize,
            });
        }
        Ok(dets)
    }
}
