//! Image preprocessing: fit an RGB8 image into the model input (letterbox or stretch) and
//! convert it to a CHW `f32` tensor in `0..1`, optionally mean/std normalized per channel.

use anyhow::{Context, Result};
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use super::{Family, Normalization, PreprocessCtx, ResizeMode};

/// Gray value used for letterbox padding (Ultralytics convention).
pub const PAD_VALUE: u8 = 114;

/// Reusable preprocessor. Buffers (resized image, output tensor, resizer scratch) are kept
/// across calls so steady-state operation does not allocate.
pub struct Preprocessor {
    input_w: u32,
    input_h: u32,
    mode: ResizeMode,
    resizer: Resizer,
    options: ResizeOptions,
    /// Resized HWC RGB8 content (only the content area, not the padding).
    resized: Vec<u8>,
    /// CHW f32 output tensor, `3 * input_w * input_h`.
    tensor: Vec<f32>,
    /// Per channel (R, G, B): `lut[c][v] == v as f32 / 255.0`, or with normalization
    /// `(v / 255 - mean[c]) / std[c]` (computed in f64, rounded once).
    lut: [[f32; 256]; 3],
    normalization: Option<Normalization>,
}

fn build_lut(norm: Option<Normalization>) -> [[f32; 256]; 3] {
    let mut lut = [[0f32; 256]; 3];
    for (c, table) in lut.iter_mut().enumerate() {
        for (i, v) in table.iter_mut().enumerate() {
            *v = match norm {
                None => i as f32 / 255.0,
                Some(n) => ((i as f64 / 255.0 - n.mean[c] as f64) / n.std[c] as f64) as f32,
            };
        }
    }
    lut
}

impl Preprocessor {
    pub fn new(input_w: u32, input_h: u32, mode: ResizeMode) -> Self {
        Self {
            input_w,
            input_h,
            mode,
            resizer: Resizer::new(),
            options: ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
            resized: Vec::new(),
            tensor: vec![0.0; 3 * input_w as usize * input_h as usize],
            lut: build_lut(None),
            normalization: None,
        }
    }

    /// Preprocessor for `family`: its resize mode and normalization.
    pub fn for_family(input_w: u32, input_h: u32, family: &dyn Family) -> Self {
        Self::new(input_w, input_h, family.resize_mode()).with_normalization(family.normalization())
    }

    /// Normalize each channel as `(v / 255 - mean) / std` (None = plain `0..1`).
    pub fn with_normalization(mut self, normalization: Option<Normalization>) -> Self {
        self.lut = build_lut(normalization);
        self.normalization = normalization;
        self
    }

    pub fn normalization(&self) -> Option<Normalization> {
        self.normalization
    }

    /// Model input size `(w, h)`.
    pub fn input_size(&self) -> (u32, u32) {
        (self.input_w, self.input_h)
    }

    pub fn mode(&self) -> ResizeMode {
        self.mode
    }

    /// `rgb` is tightly packed RGB8, `w*h*3` bytes. Returns CHW f32 in 0..1 (or normalized)
    /// (length `3*input_w*input_h`) and the geometry needed to map boxes back.
    pub fn run(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<(&[f32], PreprocessCtx)> {
        if w == 0 || h == 0 {
            anyhow::bail!("cannot preprocess an empty image ({w}x{h})");
        }
        if self.input_w == 0 || self.input_h == 0 {
            anyhow::bail!("invalid model input size {}x{}", self.input_w, self.input_h);
        }
        let expected = w as usize * h as usize * 3;
        if rgb.len() != expected {
            anyhow::bail!(
                "RGB buffer has {} bytes, expected {expected} for {w}x{h}x3",
                rgb.len()
            );
        }
        let (iw, ih) = (self.input_w, self.input_h);

        let (new_w, new_h, scale, pad_x, pad_y) = match self.mode {
            ResizeMode::Letterbox => {
                let scale = (iw as f64 / w as f64).min(ih as f64 / h as f64);
                let new_w = ((w as f64 * scale).round() as u32).clamp(1, iw);
                let new_h = ((h as f64 * scale).round() as u32).clamp(1, ih);
                let pad_x = (iw - new_w) / 2;
                let pad_y = (ih - new_h) / 2;
                (new_w, new_h, scale as f32, pad_x, pad_y)
            }
            ResizeMode::Stretch => (iw, ih, 1.0f32, 0, 0),
        };

        // Resize (or pass through) to the content size.
        let content: &[u8] = if new_w == w && new_h == h {
            rgb
        } else {
            let len = new_w as usize * new_h as usize * 3;
            self.resized.resize(len, 0);
            let src = ImageRef::new(w, h, rgb, PixelType::U8x3).context("source image view")?;
            let mut dst =
                Image::from_slice_u8(new_w, new_h, &mut self.resized[..len], PixelType::U8x3)
                    .context("resize destination")?;
            self.resizer
                .resize(&src, &mut dst, &self.options)
                .context("resizing image")?;
            &self.resized[..len]
        };

        let plane = iw as usize * ih as usize;
        self.tensor.resize(3 * plane, 0.0);
        if new_w != iw || new_h != ih {
            for (c, p) in self.tensor.chunks_exact_mut(plane).enumerate() {
                p.fill(self.lut[c][PAD_VALUE as usize]);
            }
        }

        // HWC u8 -> CHW f32 in one pass over the content rows.
        let (r_plane, rest) = self.tensor.split_at_mut(plane);
        let (g_plane, b_plane) = rest.split_at_mut(plane);
        let row_bytes = new_w as usize * 3;
        let [lr, lg, lb] = &self.lut;
        for (y, src_row) in content.chunks_exact(row_bytes).enumerate() {
            let start = (y + pad_y as usize) * iw as usize + pad_x as usize;
            let end = start + new_w as usize;
            let r = &mut r_plane[start..end];
            let g = &mut g_plane[start..end];
            let b = &mut b_plane[start..end];
            for (i, px) in src_row.as_chunks::<3>().0.iter().enumerate() {
                r[i] = lr[px[0] as usize];
                g[i] = lg[px[1] as usize];
                b[i] = lb[px[2] as usize];
            }
        }

        let ctx = PreprocessCtx {
            orig_w: w,
            orig_h: h,
            input_w: iw,
            input_h: ih,
            mode: self.mode,
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
        };
        Ok((&self.tensor, ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [u8; 3]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..w * h {
            v.extend_from_slice(&c);
        }
        v
    }

    #[test]
    fn letterbox_1080p() {
        let color = [200u8, 50, 10];
        let img = solid(1920, 1080, color);
        let mut p = Preprocessor::new(640, 640, ResizeMode::Letterbox);
        let (t, ctx) = p.run(&img, 1920, 1080).unwrap();
        assert_eq!(t.len(), 3 * 640 * 640);
        assert_eq!(ctx.pad_x, 0.0);
        assert_eq!(ctx.pad_y, 140.0);
        assert!((ctx.scale - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(
            (ctx.orig_w, ctx.orig_h, ctx.input_w, ctx.input_h),
            (1920, 1080, 640, 640)
        );
        let plane = 640 * 640;
        let pad = 114.0 / 255.0;
        for (c, &cv) in color.iter().enumerate() {
            let want = cv as f32 / 255.0;
            for y in 0..640 {
                for x in 0..640 {
                    let v = t[c * plane + y * 640 + x];
                    if !(140..500).contains(&y) {
                        assert_eq!(v, pad, "pad c={c} y={y} x={x}");
                    } else {
                        assert_eq!(v, want, "content c={c} y={y} x={x}");
                    }
                }
            }
        }
    }

    #[test]
    fn passthrough_same_size() {
        let img: Vec<u8> = (0..640 * 640 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let mut p = Preprocessor::new(640, 640, ResizeMode::Letterbox);
        let (t, ctx) = p.run(&img, 640, 640).unwrap();
        assert_eq!((ctx.scale, ctx.pad_x, ctx.pad_y), (1.0, 0.0, 0.0));
        let plane = 640 * 640;
        for (i, px) in img.as_chunks::<3>().0.iter().enumerate() {
            for (c, &v) in px.iter().enumerate() {
                assert_eq!(t[c * plane + i], v as f32 / 255.0);
            }
        }
    }

    #[test]
    fn stretch_fills_tensor() {
        let color = [10u8, 20, 30];
        let img = solid(100, 50, color);
        let mut p = Preprocessor::new(640, 640, ResizeMode::Stretch);
        let (t, ctx) = p.run(&img, 100, 50).unwrap();
        assert_eq!((ctx.scale, ctx.pad_x, ctx.pad_y), (1.0, 0.0, 0.0));
        assert_eq!(ctx.mode, ResizeMode::Stretch);
        assert_eq!((ctx.orig_w, ctx.orig_h), (100, 50));
        let plane = 640 * 640;
        for (c, &cv) in color.iter().enumerate() {
            let want = cv as f32 / 255.0;
            for &v in &t[c * plane..(c + 1) * plane] {
                assert_eq!(v, want);
            }
        }
    }

    #[test]
    fn imagenet_normalization_per_channel() {
        let n = Normalization::IMAGENET;
        let color = [255u8, 0, 128];
        let img = solid(8, 4, color);
        let mut p = Preprocessor::new(16, 16, ResizeMode::Letterbox).with_normalization(Some(n));
        assert_eq!(p.normalization(), Some(n));
        let (t, ctx) = p.run(&img, 8, 4).unwrap();
        assert_eq!((ctx.pad_x, ctx.pad_y), (0.0, 4.0));
        let plane = 16 * 16;
        let want =
            |c: usize, v: u8| ((v as f64 / 255.0 - n.mean[c] as f64) / n.std[c] as f64) as f32;
        for c in 0..3 {
            // Padding row 0 is normalized gray, content row 4 the normalized color.
            assert_eq!(t[c * plane], want(c, PAD_VALUE));
            assert_eq!(t[c * plane + 4 * 16 + 3], want(c, color[c]));
        }
        assert!((t[0] - (1.0 / 255.0 * 114.0 - 0.485) / 0.229).abs() < 1e-6);
        assert!((t[4 * 16] - (1.0 - 0.485) / 0.229).abs() < 1e-6);
        assert!((t[plane + 4 * 16] - (0.0 - 0.456) / 0.224).abs() < 1e-6);
        // Back to plain 0..1.
        let mut p = p.with_normalization(None);
        let (t, _) = p.run(&img, 8, 4).unwrap();
        assert_eq!(t[4 * 16], 1.0);
        assert_eq!(t[2 * plane + 4 * 16], 128.0 / 255.0);
    }

    #[test]
    fn buffers_reused_and_errors() {
        let mut p = Preprocessor::new(64, 64, ResizeMode::Letterbox);
        let a = solid(128, 32, [255, 255, 255]);
        let ptr1 = p.run(&a, 128, 32).unwrap().0.as_ptr();
        let b = solid(32, 128, [0, 0, 0]);
        let (t, ctx) = p.run(&b, 32, 128).unwrap();
        assert_eq!(t.as_ptr(), ptr1);
        assert_eq!((ctx.pad_x, ctx.pad_y), (24.0, 0.0));
        assert_eq!(ctx.scale, 0.5);
        // Left padding column is gray, content (x 24..40) is black.
        assert_eq!(t[0], 114.0 / 255.0);
        assert_eq!(t[30], 0.0);
        assert_eq!(t[40], 114.0 / 255.0);
        assert!(p.run(&b, 33, 128).is_err());
        assert!(p.run(&[], 0, 0).is_err());
    }
}
