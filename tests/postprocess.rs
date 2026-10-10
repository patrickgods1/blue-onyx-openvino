//! Post-processing tests through the public API with synthetic output tensors.

use blue_onyx_prism::model::rtdetr::RtDetr;
use blue_onyx_prism::model::yolo5::Yolo5;
use blue_onyx_prism::model::yolo8::Yolo8;
use blue_onyx_prism::model::yolo26::Yolo26;
use blue_onyx_prism::model::{
    Detection, ExtraData, Family, ModelFamilyKind, NamedOutput, OutputBuf, PostParams,
    PreprocessCtx, ResizeMode, make_family,
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
    ] {
        assert_eq!(make_family(k, &[], &[], 80).unwrap().kind(), k);
    }
}
