//! Class-aware greedy non-maximum suppression.

use super::Detection;

/// Intersection over union of two axis-aligned boxes (`x1,y1,x2,y2`). Inverted boxes have
/// zero area; returns 0 when the union is empty.
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let iw = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let ih = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    let inter = iw * ih;
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Greedy NMS: sort by score (descending) and drop every box whose IoU with a higher-scoring
/// kept box of the same class is `>= iou_threshold`. The result stays sorted by score.
pub fn nms(dets: &mut Vec<Detection>, iou_threshold: f32) {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let n = dets.len();
    let mut suppressed = vec![false; n];
    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !suppressed[j]
                && dets[j].class_id == dets[i].class_id
                && iou(&dets[i], &dets[j]) >= iou_threshold
            {
                suppressed[j] = true;
            }
        }
    }
    let mut flags = suppressed.into_iter();
    dets.retain(|_| !flags.next().unwrap_or(false));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: usize) -> Detection {
        Detection {
            x1,
            y1,
            x2,
            y2,
            score,
            class_id,
        }
    }

    #[test]
    fn iou_known_rects() {
        let a = d(0.0, 0.0, 10.0, 10.0, 1.0, 0);
        let b = d(5.0, 0.0, 15.0, 10.0, 1.0, 0);
        // inter 50, union 150
        assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(iou(&a, &a), 1.0);
        let c = d(20.0, 20.0, 30.0, 30.0, 1.0, 0);
        assert_eq!(iou(&a, &c), 0.0);
        let inner = d(0.0, 0.0, 5.0, 5.0, 1.0, 0);
        assert!((iou(&a, &inner) - 0.25).abs() < 1e-6);
        let degenerate = d(1.0, 1.0, 1.0, 1.0, 1.0, 0);
        assert_eq!(iou(&degenerate, &degenerate), 0.0);
    }

    #[test]
    fn same_class_suppressed() {
        let a = d(0.0, 0.0, 100.0, 100.0, 0.7, 1);
        let b = d(0.0, 11.0, 100.0, 111.0, 0.9, 1); // inter 8900, union 11100 -> ~0.80
        assert!((iou(&a, &b) - 8900.0 / 11100.0).abs() < 1e-5);
        let mut v = vec![a, b.clone()];
        nms(&mut v, 0.5);
        assert_eq!(v, vec![b]);
    }

    #[test]
    fn different_classes_kept() {
        let a = d(0.0, 0.0, 100.0, 100.0, 0.7, 1);
        let b = d(0.0, 11.0, 100.0, 111.0, 0.9, 2);
        let mut v = vec![a, b];
        nms(&mut v, 0.5);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].score, 0.9);
    }

    #[test]
    fn chain_and_threshold() {
        // b overlaps a strongly; c overlaps b but barely a -> c survives once b is suppressed.
        let a = d(0.0, 0.0, 10.0, 10.0, 0.9, 0);
        let b = d(2.0, 0.0, 12.0, 10.0, 0.8, 0);
        let c = d(9.0, 0.0, 19.0, 10.0, 0.7, 0);
        let mut v = vec![c.clone(), b, a.clone()];
        nms(&mut v, 0.5);
        assert_eq!(v, vec![a, c]);

        let x = d(0.0, 0.0, 10.0, 10.0, 0.9, 0);
        let y = d(5.0, 0.0, 15.0, 10.0, 0.8, 0); // IoU 1/3
        let mut v = vec![x.clone(), y.clone()];
        nms(&mut v, 0.34);
        assert_eq!(v.len(), 2);
        let mut v = vec![x, y];
        nms(&mut v, 0.33);
        assert_eq!(v.len(), 1);
    }
}
