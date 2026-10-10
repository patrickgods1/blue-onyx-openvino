//! Post-processing tests through the public API with synthetic output tensors.

use blue_onyx_prism::model::detr::Detr;
use blue_onyx_prism::model::preprocess::Preprocessor;
use blue_onyx_prism::model::rtdetr::RtDetr;
use blue_onyx_prism::model::yolo5::Yolo5;
use blue_onyx_prism::model::yolo8::Yolo8;
use blue_onyx_prism::model::yolo26::Yolo26;
use blue_onyx_prism::model::{
    Detection, ExtraData, Family, ModelFamilyKind, NamedOutput, Normalization, OutputBuf, PortElem,
    PortSpec, PostParams, PreprocessCtx, ResizeMode, make_family,
};

const PARAMS: PostParams = PostParams {
    confidence_threshold: 0.5,
    nms_iou: 0.45,
};

/// 1920x1080 letterboxed into 640x640: scale 1/3, content rows 140..500.
fn letterbox_ctx() -> PreprocessCtx {
    PreprocessCtx {
        orig_w: 1920,
        orig_h: 1080,
        input_w: 640,
        input_h: 640,
        mode: ResizeMode::Letterbox,
        scale: 1.0 / 3.0,
        pad_x: 0.0,
        pad_y: 140.0,
    }
}

fn stretch_ctx() -> PreprocessCtx {
    PreprocessCtx {
        orig_w: 100,
        orig_h: 50,
        input_w: 640,
        input_h: 640,
        mode: ResizeMode::Stretch,
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    }
}

fn f32_out(name: &str, shape: &[usize], data: Vec<f32>) -> NamedOutput {
    NamedOutput {
        name: name.into(),
        shape: shape.to_vec(),
        data: OutputBuf::F32(data),
    }
}

fn assert_box(d: &Detection, b: (f32, f32, f32, f32)) {
    let tol = 1e-2;
    assert!(
        (d.x1 - b.0).abs() < tol
            && (d.y1 - b.1).abs() < tol
            && (d.x2 - b.2).abs() < tol
            && (d.y2 - b.3).abs() < tol,
        "box {:?} != {:?}",
        (d.x1, d.y1, d.x2, d.y2),
        b
    );
}

// ---------------------------------------------------------------- yolo26

#[test]
fn yolo26_maps_and_thresholds() {
    let rows: Vec<f32> = vec![
        0.0, 140.0, 640.0, 500.0, 0.9, 2.0, // full image
        270.0, 270.0, 370.0, 370.0, 0.6, 0.0, // small box
        10.0, 150.0, 20.0, 160.0, 0.1, 1.0, // below threshold
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // padding row
    ];
    let out = f32_out("output0", &[1, 4, 6], rows);
    let fam = Yolo26::new(80);
    assert_eq!(fam.resize_mode(), ResizeMode::Letterbox);
    assert!(fam.extra_inputs(&letterbox_ctx()).is_empty());
    let dets = fam.postprocess(&[out], &letterbox_ctx(), &PARAMS).unwrap();
    assert_eq!(dets.len(), 2);
    assert_box(&dets[0], (0.0, 0.0, 1920.0, 1080.0));
    assert_eq!((dets[0].class_id, dets[0].score), (2, 0.9));
    assert_box(&dets[1], (810.0, 390.0, 1110.0, 690.0));
    assert_eq!(dets[1].class_id, 0);
}

#[test]
fn yolo26_no_nms() {
    // Two identical boxes: end-to-end models are NMS-free, so both are returned.
    let rows = vec![
        0.0, 140.0, 640.0, 500.0, 0.9, 2.0, 0.0, 140.0, 640.0, 500.0, 0.8, 2.0,
    ];
    let dets = Yolo26::new(80)
        .postprocess(&[f32_out("o", &[1, 2, 6], rows)], &letterbox_ctx(), &PARAMS)
        .unwrap();
    assert_eq!(dets.len(), 2);
}

#[test]
fn yolo26_shape_errors() {
    let fam = Yolo26::new(80);
    let e = fam
        .postprocess(
            &[f32_out("o", &[1, 2, 7], vec![0.0; 14])],
            &letterbox_ctx(),
            &PARAMS,
        )
        .unwrap_err()
        .to_string();
    assert!(e.contains("[1, 2, 7]"), "{e}");
    assert!(
        fam.postprocess(
            &[f32_out("o", &[1, 2, 6], vec![0.0; 6])],
            &letterbox_ctx(),
            &PARAMS
        )
        .is_err()
    );
    assert!(fam.postprocess(&[], &letterbox_ctx(), &PARAMS).is_err());
}

// ---------------------------------------------------------------- yolo5

fn yolo5_rows() -> Vec<f32> {
    vec![
        // cx, cy, w, h, obj, c0, c1, c2
        320.0, 320.0, 100.0, 100.0, 0.9, 0.1, 0.9, 0.2, // A: class 1, conf 0.81
        325.0, 322.0, 100.0, 100.0, 0.8, 0.1, 0.9, 0.2, // B: overlaps A, class 1, conf 0.72
        320.0, 320.0, 100.0, 100.0, 0.9, 0.95, 0.1, 0.0, // C: same place, class 0, conf 0.855
        100.0, 300.0, 20.0, 20.0, 0.2, 0.0, 1.0, 0.0, // D: obj too low
        100.0, 300.0, 20.0, 20.0, 0.9, 0.5, 0.5, 0.5, // E: conf 0.45 too low
    ]
}

#[test]
fn yolo5_decode_threshold_nms() {
    let out = f32_out("output", &[1, 5, 8], yolo5_rows());
    let dets = Yolo5::new(3)
        .postprocess(&[out], &letterbox_ctx(), &PARAMS)
        .unwrap();
    assert_eq!(dets.len(), 2, "{dets:?}");
    // Sorted by score: C (class 0) then A (class 1); B merged into A.
    assert_eq!(dets[0].class_id, 0);
    assert!((dets[0].score - 0.855).abs() < 1e-6);
    assert_eq!(dets[1].class_id, 1);
    assert!((dets[1].score - 0.81).abs() < 1e-6);
    assert_box(&dets[1], (810.0, 390.0, 1110.0, 690.0));
}

#[test]
fn yolo5_2d_shape_and_inferred_classes() {
    let out = f32_out("output", &[5, 8], yolo5_rows());
    let dets = Yolo5::new(0)
        .postprocess(&[out], &letterbox_ctx(), &PARAMS)
        .unwrap();
    assert_eq!(dets.len(), 2);
}

#[test]
fn yolo5_shape_errors() {
    let e = Yolo5::new(80)
        .postprocess(
            &[f32_out("output", &[1, 5, 8], yolo5_rows())],
            &letterbox_ctx(),
            &PARAMS,
        )
        .unwrap_err()
        .to_string();
    assert!(e.contains("[1, 5, 8]") && e.contains("85"), "{e}");
    assert!(
        Yolo5::new(0)
            .postprocess(
                &[f32_out("o", &[1, 2, 5], vec![0.0; 10])],
                &letterbox_ctx(),
                &PARAMS
            )
            .is_err()
    );
    assert!(
        Yolo5::new(0)
            .postprocess(
                &[f32_out("o", &[2, 1, 2, 8], vec![0.0; 32])],
                &letterbox_ctx(),
                &PARAMS
            )
            .is_err()
    );
}

// ---------------------------------------------------------------- yolo8

/// Channel-major `[1, 4+C, A]` from per-anchor rows `(cx, cy, w, h, scores...)`.
fn channel_major(anchors: &[Vec<f32>]) -> (Vec<usize>, Vec<f32>) {
    let ch = anchors[0].len();
    let a = anchors.len();
    let mut data = vec![0.0; ch * a];
    for (j, row) in anchors.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            data[c * a + j] = v;
        }
    }
    (vec![1, ch, a], data)
}

#[test]
fn yolo8_decode_threshold_nms() {
    let (shape, data) = channel_major(&[
        vec![320.0, 320.0, 100.0, 100.0, 0.1, 0.8], // A: class 1, 0.8
        vec![324.0, 318.0, 100.0, 100.0, 0.05, 0.7], // B: overlaps A, class 1 -> merged
        vec![320.0, 320.0, 100.0, 100.0, 0.6, 0.2], // C: same place, class 0 -> kept
        vec![50.0, 200.0, 20.0, 20.0, 0.3, 0.4],    // D: below threshold
        vec![600.0, 450.0, 40.0, 20.0, 0.0, 0.55],  // E: separate box, class 1
    ]);
    let out = f32_out("output0", &shape, data);
    let dets = Yolo8::new(2)
        .postprocess(&[out], &letterbox_ctx(), &PARAMS)
        .unwrap();
    assert_eq!(dets.len(), 3, "{dets:?}");
    assert_eq!((dets[0].class_id, dets[0].score), (1, 0.8));
    assert_box(&dets[0], (810.0, 390.0, 1110.0, 690.0));
    assert_eq!((dets[1].class_id, dets[1].score), (0, 0.6));
    assert_eq!((dets[2].class_id, dets[2].score), (1, 0.55));
    // (580, 440, 620, 460) in input pixels.
    assert_box(&dets[2], (1740.0, 900.0, 1860.0, 960.0));
}

#[test]
fn yolo8_shape_errors() {
    let e = Yolo8::new(80)
        .postprocess(
            &[f32_out("o", &[1, 6, 5], vec![0.0; 30])],
            &letterbox_ctx(),
            &PARAMS,
        )
        .unwrap_err()
        .to_string();
    assert!(e.contains("[1, 6, 5]") && e.contains("84"), "{e}");
    assert!(
        Yolo8::new(0)
            .postprocess(
                &[f32_out("o", &[1, 84], vec![0.0; 84])],
                &letterbox_ctx(),
                &PARAMS
            )
            .is_err()
    );
    assert!(
        Yolo8::new(0)
            .postprocess(
                &[f32_out("o", &[1, 4, 10], vec![0.0; 40])],
                &letterbox_ctx(),
                &PARAMS
            )
            .is_err()
    );
}

// ---------------------------------------------------------------- rtdetr

fn rtdetr_outputs(labels_i32: bool) -> Vec<NamedOutput> {
    let labels = vec![0i64, 2, 5];
    vec![
        NamedOutput {
            name: "labels".into(),
            shape: vec![1, 3],
            data: if labels_i32 {
                OutputBuf::I32(labels.iter().map(|&x| x as i32).collect())
            } else {
                OutputBuf::I64(labels)
            },
        },
        f32_out(
            "boxes",
            &[1, 3, 4],
            vec![
                0.0, 0.0, 640.0, 640.0, // full image
                320.0, 320.0, 640.0, 640.0, // bottom-right quarter
                0.0, 0.0, 64.0, 64.0, // low score
            ],
        ),
        f32_out("scores", &[1, 3], vec![0.95, 0.7, 0.3]),
    ]
}

fn check_rtdetr(dets: &[Detection]) {
    assert_eq!(dets.len(), 2, "{dets:?}");
    assert_box(&dets[0], (0.0, 0.0, 100.0, 50.0));
    assert_eq!((dets[0].class_id, dets[0].score), (0, 0.95));
    assert_box(&dets[1], (50.0, 25.0, 100.0, 50.0));
    assert_eq!((dets[1].class_id, dets[1].score), (2, 0.7));
}

#[test]
fn rtdetr_extra_inputs() {
    let fam = RtDetr::new(80);
    assert_eq!(fam.resize_mode(), ResizeMode::Stretch);
    let extra = fam.extra_inputs(&stretch_ctx());
    assert_eq!(extra.len(), 1);
    assert_eq!(extra[0].name, "orig_target_sizes");
    assert_eq!(extra[0].shape, vec![1, 2]);
    assert_eq!(extra[0].data, ExtraData::I64(vec![640, 640]));
}

#[test]
fn rtdetr_order_independent() {
    let fam = RtDetr::new(80);
    let base = rtdetr_outputs(false);
    for perm in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let outs: Vec<_> = perm.iter().map(|&i| base[i].clone()).collect();
        check_rtdetr(&fam.postprocess(&outs, &stretch_ctx(), &PARAMS).unwrap());
    }
    check_rtdetr(
        &fam.postprocess(&rtdetr_outputs(true), &stretch_ctx(), &PARAMS)
            .unwrap(),
    );
}

#[test]
fn rtdetr_errors() {
    let fam = RtDetr::new(80);
    let mut outs = rtdetr_outputs(false);
    outs.remove(1);
    let e = fam
        .postprocess(&outs, &stretch_ctx(), &PARAMS)
        .unwrap_err()
        .to_string();
    assert!(e.contains("boxes"), "{e}");

    let mut outs = rtdetr_outputs(false);
    outs[1] = f32_out("boxes", &[1, 3, 5], vec![0.0; 15]);
    let e = fam
        .postprocess(&outs, &stretch_ctx(), &PARAMS)
        .unwrap_err()
        .to_string();
    assert!(e.contains("[1, 3, 5]"), "{e}");

    let mut outs = rtdetr_outputs(false);
    outs[2] = f32_out("scores", &[1, 4], vec![0.0; 4]);
    assert!(fam.postprocess(&outs, &stretch_ctx(), &PARAMS).is_err());

    let mut outs = rtdetr_outputs(false);
    outs[0] = f32_out("labels", &[1, 3], vec![0.0; 3]);
    assert!(fam.postprocess(&outs, &stretch_ctx(), &PARAMS).is_err());
}

// ---------------------------------------------------------------- factory

#[test]
fn make_family_kinds() {
    for k in [
        ModelFamilyKind::Yolo26,
        ModelFamilyKind::Yolo5,
        ModelFamilyKind::Yolo8,
        ModelFamilyKind::RtDetr,
        ModelFamilyKind::Detr,
        ModelFamilyKind::RfDetr,
    ] {
        assert_eq!(make_family(k, &[], &[], 80).unwrap().kind(), k);
    }
}

// ---------------------------------------------------------------- detr (D-FINE / RF-DETR)

/// Inverse sigmoid: the logit whose score is `p`.
fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

/// `logits [1,Q,C]` (all -10 except `hot` = (query, class, score)) + `pred_boxes [1,Q,4]`.
fn detr_outputs(
    q: usize,
    c: usize,
    hot: &[(usize, usize, f32)],
    boxes: &[[f32; 4]],
) -> Vec<NamedOutput> {
    let mut logits = vec![-10.0f32; q * c];
    for &(qi, ci, p) in hot {
        logits[qi * c + ci] = logit(p);
    }
    assert_eq!(boxes.len(), q);
    // RF-DETR's export lists `pred_boxes` first; order must not matter.
    vec![
        f32_out(
            "pred_boxes",
            &[1, q, 4],
            boxes.iter().flatten().copied().collect(),
        ),
        f32_out("logits", &[1, q, c], logits),
    ]
}

fn sorted_by_score(d: &[Detection]) -> bool {
    d.windows(2).all(|w| w[0].score >= w[1].score)
}

#[test]
fn detr_decodes_cxcywh_to_original_coords() {
    // 100x50 image stretched to 640x640.
    let outs = detr_outputs(
        3,
        80,
        &[(0, 2, 0.88), (1, 0, 0.62), (1, 5, 0.73), (2, 7, 0.3)],
        &[
            [0.5, 0.5, 0.2, 0.4],
            [0.25, 0.75, 0.5, 0.5],
            [0.1, 0.1, 0.1, 0.1],
        ],
    );
    let fam = Detr::new(80, false);
    assert_eq!(fam.kind(), ModelFamilyKind::Detr);
    assert_eq!(fam.resize_mode(), ResizeMode::Stretch);
    assert_eq!(fam.normalization(), None);
    assert!(fam.extra_inputs(&stretch_ctx()).is_empty());
    let dets = fam.postprocess(&outs, &stretch_ctx(), &PARAMS).unwrap();
    // Query 1 yields two classes (top-k over Q x C); query 2 is under the threshold.
    assert_eq!(dets.len(), 3);
    assert!(sorted_by_score(&dets));
    assert_eq!(
        dets.iter().map(|d| d.class_id).collect::<Vec<_>>(),
        [2, 5, 0]
    );
    assert!((dets[0].score - 0.88).abs() < 1e-5);
    // cx,cy,w,h = 320,320,128,256 input px -> 256..384 x 192..448 -> x*100/640, y*50/640.
    assert_box(&dets[0], (40.0, 15.0, 60.0, 35.0));
    // 160,480,320,320 -> 0..320 x 320..640 -> 0..50 x 25..50
    assert_box(&dets[1], (0.0, 25.0, 50.0, 50.0));
    assert_box(&dets[2], (0.0, 25.0, 50.0, 50.0));
}

#[test]
fn detr_keeps_at_most_q_best() {
    // Q = 2 queries, every (query, class) pair above the threshold: only the best 2 remain.
    let outs = detr_outputs(
        2,
        3,
        &[
            (0, 0, 0.6),
            (0, 1, 0.95),
            (0, 2, 0.7),
            (1, 0, 0.9),
            (1, 1, 0.55),
            (1, 2, 0.8),
        ],
        &[[0.5; 4], [0.5; 4]],
    );
    let dets = Detr::new(3, false)
        .postprocess(&outs, &stretch_ctx(), &PARAMS)
        .unwrap();
    assert_eq!(dets.len(), 2);
    assert_eq!((dets[0].class_id, dets[1].class_id), (1, 0));
    assert!((dets[0].score - 0.95).abs() < 1e-5 && (dets[1].score - 0.9).abs() < 1e-5);
    // A 0.0 threshold still caps at Q.
    let all = PostParams {
        confidence_threshold: 0.0,
        nms_iou: 0.45,
    };
    let dets = Detr::new(3, false)
        .postprocess(&outs, &stretch_ctx(), &all)
        .unwrap();
    assert_eq!(dets.len(), 2);
}

#[test]
fn rfdetr_coco_category_ids_map_to_coco80() {
    // 1120x560 image stretched to RF-DETR's 560x560 input.
    let ctx = PreprocessCtx {
        orig_w: 1120,
        orig_h: 560,
        input_w: 560,
        input_h: 560,
        mode: ResizeMode::Stretch,
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };
    let outs = detr_outputs(
        3,
        91,
        // person (id 1), unused id 12 and background 0 (skipped), toothbrush (id 90), car (id 3)
        &[
            (0, 1, 0.9),
            (1, 12, 0.99),
            (1, 0, 0.98),
            (1, 90, 0.8),
            (2, 3, 0.7),
        ],
        &[
            [0.5, 0.5, 0.5, 0.5],
            [0.25, 0.25, 0.1, 0.2],
            [0.75, 0.5, 0.5, 1.0],
        ],
    );
    let ports = [
        PortSpec {
            name: "pred_boxes".into(),
            shape: vec![-1, 300, 4],
            elem: PortElem::F32,
        },
        PortSpec {
            name: "logits".into(),
            shape: vec![-1, 300, 91],
            elem: PortElem::F32,
        },
    ];
    let pix = [PortSpec {
        name: "pixel_values".into(),
        shape: vec![-1, 3, -1, -1],
        elem: PortElem::F32,
    }];
    let fam = make_family(ModelFamilyKind::Auto, &pix, &ports, 80).unwrap();
    assert_eq!(fam.kind(), ModelFamilyKind::RfDetr);
    assert_eq!(fam.normalization(), Some(Normalization::IMAGENET));
    let dets = fam.postprocess(&outs, &ctx, &PARAMS).unwrap();
    assert_eq!(
        dets.iter().map(|d| d.class_id).collect::<Vec<_>>(),
        [0, 79, 2],
        "person, toothbrush, car in COCO-80 order"
    );
    // 280,280,280,280 input px -> 140..420 both axes -> x*2, y*1.
    assert_box(&dets[0], (280.0, 140.0, 840.0, 420.0));
    // 140,140,56,112 -> 112..168 x 84..196
    assert_box(&dets[1], (224.0, 84.0, 336.0, 196.0));
    // 420,280,280,560 -> 280..560 x 0..560
    assert_box(&dets[2], (560.0, 0.0, 1120.0, 560.0));
    let names = blue_onyx_prism::model::classes::coco80();
    let preds = blue_onyx_prism::model::to_predictions(&dets, &ctx, &names, None);
    assert_eq!(
        preds.iter().map(|p| p.label.as_str()).collect::<Vec<_>>(),
        ["person", "toothbrush", "car"]
    );

    // With a 91-name class list the ids are used as they are (0 and 12 included), top Q = 3.
    let dets = Detr::new(91, true)
        .postprocess(&outs, &ctx, &PARAMS)
        .unwrap();
    assert_eq!(
        dets.iter().map(|d| d.class_id).collect::<Vec<_>>(),
        [12, 0, 1]
    );
}

#[test]
fn detr_2d_shapes_and_errors() {
    let fam = Detr::new(2, false);
    let ok = |outs: &[NamedOutput]| fam.postprocess(outs, &stretch_ctx(), &PARAMS);
    // [Q, C] / [Q, 4] without the batch dimension.
    let outs = vec![
        f32_out("logits", &[1, 2], vec![logit(0.9), -5.0]),
        f32_out("pred_boxes", &[1, 4], vec![0.5, 0.5, 1.0, 1.0]),
    ];
    let dets = ok(&outs).unwrap();
    assert_eq!(dets.len(), 1);
    assert_box(&dets[0], (0.0, 0.0, 100.0, 50.0));
    // Missing output.
    assert!(ok(&outs[..1]).is_err());
    // Query count mismatch.
    let bad = vec![
        f32_out("logits", &[1, 2, 2], vec![0.0; 4]),
        f32_out("pred_boxes", &[1, 1, 4], vec![0.0; 4]),
    ];
    assert!(ok(&bad).is_err());
    // Boxes not 4-wide.
    let bad = vec![
        f32_out("logits", &[1, 1, 2], vec![0.0; 2]),
        f32_out("pred_boxes", &[1, 1, 5], vec![0.0; 5]),
    ];
    assert!(ok(&bad).is_err());
    // Buffer length does not match the shape.
    let bad = vec![
        f32_out("logits", &[1, 1, 2], vec![0.0; 3]),
        f32_out("pred_boxes", &[1, 1, 4], vec![0.0; 4]),
    ];
    assert!(ok(&bad).is_err());
    // Wrong element type, wrong rank.
    let bad = vec![
        NamedOutput {
            name: "logits".into(),
            shape: vec![1, 1, 2],
            data: OutputBuf::I64(vec![0, 0]),
        },
        f32_out("pred_boxes", &[1, 1, 4], vec![0.0; 4]),
    ];
    assert!(ok(&bad).is_err());
    let bad = vec![
        f32_out("logits", &[2], vec![0.0; 2]),
        f32_out("pred_boxes", &[1, 1, 4], vec![0.0; 4]),
    ];
    assert!(ok(&bad).is_err());
    // NaN logits never pass the threshold.
    let nan = vec![
        f32_out("logits", &[1, 1, 2], vec![f32::NAN, f32::NAN]),
        f32_out("pred_boxes", &[1, 1, 4], vec![0.5; 4]),
    ];
    assert!(ok(&nan).unwrap().is_empty());
}

#[test]
fn detr_preprocessor_for_family() {
    // RF-DETR: 2x1 image stretched to 4x4, ImageNet-normalized; D-FINE: plain 0..1.
    let img = [255u8, 0, 0, 255, 0, 0];
    let rf = Detr::new(80, true);
    let mut p = Preprocessor::for_family(4, 4, &rf);
    assert_eq!(p.normalization(), Some(Normalization::IMAGENET));
    let (t, ctx) = p.run(&img, 2, 1).unwrap();
    assert_eq!(
        (ctx.mode, ctx.input_w, ctx.input_h),
        (ResizeMode::Stretch, 4, 4)
    );
    let n = Normalization::IMAGENET;
    for c in 0..3 {
        let v = if c == 0 { 1.0 } else { 0.0 };
        let want = (v - n.mean[c]) / n.std[c];
        for &x in &t[c * 16..(c + 1) * 16] {
            assert!((x - want).abs() < 1e-5, "c={c} {x} != {want}");
        }
    }
    let mut p = Preprocessor::for_family(4, 4, &Detr::new(80, false));
    assert_eq!(p.normalization(), None);
    let (t, _) = p.run(&img, 2, 1).unwrap();
    assert!(t[..16].iter().all(|&x| x == 1.0) && t[16..].iter().all(|&x| x == 0.0));
}
