//! Ultralytics YOLO26 exported end-to-end: one output `[1, N, 6]` with rows
//! `x1, y1, x2, y2, conf, cls` in model-input pixels. NMS-free.

use anyhow::Result;

use super::{
    Detection, Family, ModelFamilyKind, NamedOutput, PostParams, PreprocessCtx, ResizeMode,
};

pub struct Yolo26 {
    pub num_classes: usize,
}

impl Yolo26 {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

/// The single detection output of a YOLO-style model, as f32 data. When several outputs are
/// present, the first f32 one is used (some exports append auxiliary heads after the main one).
pub(crate) fn primary_f32_output<'a>(
    family: &str,
    outputs: &'a [NamedOutput],
) -> Result<(&'a NamedOutput, &'a [f32])> {
    let out = outputs
        .iter()
        .find(|o| o.as_f32().is_some())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{family}: expected an f32 output tensor, got {:?}",
                outputs
                    .iter()
                    .map(|o| (&o.name, &o.shape))
                    .collect::<Vec<_>>()
            )
        })?;
    let data = out.as_f32().unwrap_or_default();
    Ok((out, data))
}

/// `value >= threshold`; false for NaN scores.
#[inline]
pub(crate) fn meets(value: f32, threshold: f32) -> bool {
    value >= threshold
}

/// Interpret `[1, R, C]` or `[R, C]` as `(R, C)` and check the data length.
pub(crate) fn matrix_dims(
    family: &str,
    expected: &str,
    out: &NamedOutput,
    len: usize,
) -> Result<(usize, usize)> {
    let dims = match out.shape.as_slice() {
        [1, r, c] | [r, c] => (*r, *c),
        _ => anyhow::bail!(
            "{family}: output `{}` has shape {:?}, expected {expected}",
            out.name,
            out.shape
        ),
    };
    if dims.0 * dims.1 != len {
        anyhow::bail!(
            "{family}: output `{}` shape {:?} implies {} values but the buffer has {len}",
            out.name,
            out.shape,
            dims.0 * dims.1
        );
    }
    Ok(dims)
}

impl Family for Yolo26 {
    fn kind(&self) -> ModelFamilyKind {
        ModelFamilyKind::Yolo26
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
        let (out, data) = primary_f32_output("yolo26", outputs)?;
        let (_, cols) = matrix_dims("yolo26", "[1, N, 6]", out, data.len())?;
        if cols != 6 {
            anyhow::bail!(
                "yolo26: output `{}` has shape {:?}, expected [1, N, 6]",
                out.name,
                out.shape
            );
        }
        let mut dets = Vec::new();
        for row in data.as_chunks::<6>().0 {
            let conf = row[4];
            if !meets(conf, params.confidence_threshold) {
                continue;
            }
            let cls = row[5];
            if !cls.is_finite() || cls < 0.0 {
                continue;
            }
            let (x1, y1, x2, y2) = ctx.to_original(row[0], row[1], row[2], row[3]);
            dets.push(Detection {
                x1,
                y1,
                x2,
                y2,
                score: conf,
                class_id: cls.round() as usize,
            });
        }
        Ok(dets)
    }
}
