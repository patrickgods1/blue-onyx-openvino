//! Hugging Face DETR-style exports (D-FINE, RF-DETR; `onnx-community/*-ONNX`): input
//! `pixel_values [1,3,H,W]`; outputs `logits [1,Q,C]` (per-class logits, sigmoid scores) and
//! `pred_boxes [1,Q,4]` (`cx, cy, w, h` normalized to the input). NMS-free.
//!
//! Decoding follows the reference post-processors (`DFineForObjectDetection` / RF-DETR
//! `PostProcess`): sigmoid over all `Q x C` scores, keep the top `Q` (score, query, class) triples,
//! drop those under the threshold, scale the boxes by the input size and map them back to the
//! original image (stretch geometry).
//!
//! Class ids: D-FINE heads have `C = 80` in COCO-80 order (`classes::coco80`). RF-DETR COCO heads
//! have `C = 91`, indexed by COCO *category id* (1..=90 with gaps, 0 unused); with an 80-name
//! class list they are remapped to COCO-80 indices and the unused ids are skipped.
//!
//! Preprocessing: stretch to the input size, RGB x 1/255. RF-DETR (DINOv2 backbone) also needs
//! ImageNet mean/std normalization although the exports' `preprocessor_config.json` says
//! `do_normalize: false`: neither graph normalizes internally, and on real photos RF-DETR scores
//! are consistently higher with normalization while D-FINE's drop sharply (so it gets none).

use anyhow::{Result, bail};

use super::yolo26::meets;
use super::{
    Detection, Family, ModelFamilyKind, NamedOutput, Normalization, PortSpec, PostParams,
    PreprocessCtx, ResizeMode,
};

/// Class count of a head indexed by COCO category id (RF-DETR on COCO).
pub const COCO91_CLASSES: i64 = 91;
/// Input side used for RF-DETR-style models whose size is unknown (RF-DETR base).
pub const RFDETR_DEFAULT_SIZE: u32 = 560;

/// COCO category ids of the 80 COCO classes, in COCO-80 order.
pub const COCO80_CATEGORY_IDS: [u8; 80] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
];

/// COCO-80 index of a COCO category id (None for 0, the 10 unused ids and ids > 90).
pub fn coco80_index(category_id: usize) -> Option<usize> {
    COCO80_CATEGORY_IDS
        .iter()
        .position(|&c| c as usize == category_id)
}

/// Port names of the family.
pub const INPUT: &str = "pixel_values";
pub const LOGITS: &str = "logits";
pub const BOXES: &str = "pred_boxes";

/// Outputs `logits` + `pred_boxes` (with an input `pixel_values`, or any single input).
pub fn is_detr(inputs: &[PortSpec], outputs: &[PortSpec]) -> bool {
    let has = |n: &str| outputs.iter().any(|p| p.name == n);
    has(LOGITS) && has(BOXES) && (inputs.iter().any(|p| p.name == INPUT) || inputs.len() == 1)
}

/// Static class count `C` of the `logits` output, if known.
pub fn logits_classes(outputs: &[PortSpec]) -> Option<i64> {
    outputs
        .iter()
        .find(|p| p.name == LOGITS)
        .and_then(|p| p.shape.last().copied())
        .filter(|&c| c > 0)
}

pub struct Detr {
    /// Number of class names (80 for the COCO fallback).
    pub num_classes: usize,
    /// ImageNet mean/std normalization of the input.
    pub normalize: bool,
}

impl Detr {
    /// `normalize` as given.
    pub fn new(num_classes: usize, normalize: bool) -> Self {
        Self {
            num_classes,
            normalize,
        }
    }

    /// `rfdetr` forces normalization; otherwise it is on for 91-class (COCO category id)
    /// heads, which only RF-DETR uses among the supported exports.
    pub fn from_ports(outputs: &[PortSpec], num_classes: usize, rfdetr: bool) -> Self {
        Self::new(
            num_classes,
            rfdetr || logits_classes(outputs) == Some(COCO91_CLASSES),
        )
    }
}

fn find<'a>(outputs: &'a [NamedOutput], name: &str) -> Result<&'a NamedOutput> {
    outputs.iter().find(|o| o.name == name).ok_or_else(|| {
        anyhow::anyhow!(
            "detr: missing output `{name}`; got {:?}",
            outputs
                .iter()
                .map(|o| (&o.name, &o.shape))
                .collect::<Vec<_>>()
        )
    })
}

/// `(Q, last)` for `[1, Q, last]` or `[Q, last]`.
fn dims(out: &NamedOutput, expected: &str) -> Result<(usize, usize)> {
    match out.shape.as_slice() {
        [1, q, c] | [q, c] => Ok((*q, *c)),
        s => bail!(
            "detr: output `{}` has shape {s:?}, expected {expected}",
            out.name
        ),
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl Family for Detr {
    fn kind(&self) -> ModelFamilyKind {
        if self.normalize {
            ModelFamilyKind::RfDetr
        } else {
            ModelFamilyKind::Detr
        }
    }
    fn resize_mode(&self) -> ResizeMode {
        ResizeMode::Stretch
    }
    fn normalization(&self) -> Option<Normalization> {
        self.normalize.then_some(Normalization::IMAGENET)
    }
    fn postprocess(
        &self,
        outputs: &[NamedOutput],
        ctx: &PreprocessCtx,
        params: &PostParams,
    ) -> Result<Vec<Detection>> {
        let logits_o = find(outputs, LOGITS)?;
        let boxes_o = find(outputs, BOXES)?;
        let (q, c) = dims(logits_o, "[1, Q, C]")?;
        let (qb, four) = dims(boxes_o, "[1, Q, 4]")?;
        if four != 4 || qb != q {
            bail!(
                "detr: `pred_boxes` shape {:?} does not match `logits` {:?} (expected [1, Q, 4])",
                boxes_o.shape,
                logits_o.shape
            );
        }
        let logits = logits_o
            .as_f32()
            .ok_or_else(|| anyhow::anyhow!("detr: output `logits` must be f32"))?;
        let boxes = boxes_o
            .as_f32()
            .ok_or_else(|| anyhow::anyhow!("detr: output `pred_boxes` must be f32"))?;
        if logits.len() != q * c || boxes.len() != 4 * q {
            bail!(
                "detr: buffer lengths (logits {}, pred_boxes {}) do not match shapes {:?} {:?}",
                logits.len(),
                boxes.len(),
                logits_o.shape,
                boxes_o.shape
            );
        }
        // Class index -> reported class id. A 91-wide head with the 80 COCO names is indexed by
        // COCO category id: map to COCO-80 and never report the unused ids.
        let coco91 = c as i64 == COCO91_CLASSES && self.num_classes == 80;
        let class_map: Vec<Option<usize>> = (0..c)
            .map(|k| if coco91 { coco80_index(k) } else { Some(k) })
            .collect();

        // Top-Q over all reportable (query, class) scores, then the threshold. Equivalent:
        // everything at or above the threshold, best first, at most Q.
        let mut cand: Vec<(f32, usize)> = logits
            .iter()
            .enumerate()
            .filter_map(|(i, &l)| {
                class_map[i % c]?;
                let s = sigmoid(l);
                meets(s, params.confidence_threshold).then_some((s, i))
            })
            .collect();
        cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        cand.truncate(q);

        let (w, h) = (ctx.input_w as f32, ctx.input_h as f32);
        let mut dets = Vec::with_capacity(cand.len());
        for (score, i) in cand {
            let (query, class) = (i / c, i % c);
            let Some(class_id) = class_map[class] else {
                continue;
            };
            let b = &boxes[4 * query..4 * query + 4];
            let (cx, cy, bw, bh) = (b[0] * w, b[1] * h, b[2] * w, b[3] * h);
            let (x1, y1, x2, y2) =
                ctx.to_original(cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh);
            dets.push(Detection {
                x1,
                y1,
                x2,
                y2,
                score,
                class_id,
            });
        }
        Ok(dets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coco_category_ids() {
        assert_eq!(coco80_index(1), Some(0));
        assert_eq!(coco80_index(3), Some(2));
        assert_eq!(coco80_index(13), Some(11));
        assert_eq!(coco80_index(90), Some(79));
        for unused in [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91] {
            assert_eq!(coco80_index(unused), None, "{unused}");
        }
        assert!(COCO80_CATEGORY_IDS.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn sigmoid_is_stable() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!((sigmoid(2.0) - 0.880_797).abs() < 1e-6);
        assert!((sigmoid(-2.0) - 0.119_203).abs() < 1e-6);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(sigmoid(1000.0), 1.0);
    }
}
