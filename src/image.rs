//! Image decode/encode and annotation (bounding boxes + labels).
//!
//! Text is rasterized directly with `ab_glyph` and boxes are drawn on the raw RGB buffer, so
//! no optional `imageproc` features are required. Label font: Roboto Mono (Apache-2.0), see
//! `assets/roboto-mono-stripped.ttf` (taken from blue-onyx).

use crate::api::Prediction;
use ab_glyph::{Font, FontArc, PxScale, ScaleFont};
use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;
use zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
use zune_jpeg::JpegDecoder;

static FONT_BYTES: &[u8] = include_bytes!("../assets/roboto-mono-stripped.ttf");

/// Tightly packed RGB8 image (`rgb.len() == width * height * 3`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// Decode JPEG (zune-jpeg) or any format the `image` crate understands (PNG, ...).
pub fn decode(bytes: &[u8]) -> Result<RgbImage> {
    if bytes.starts_with(&[0xFF, 0xD8]) {
        match decode_jpeg(bytes) {
            Ok(img) => return Ok(img),
            Err(e) => {
                tracing::debug!("zune-jpeg failed ({e:#}), falling back to image crate");
            }
        }
    }
    let dynimg = image::load_from_memory(bytes).context("decoding image")?;
    let rgb = dynimg.to_rgb8();
    Ok(RgbImage {
        width: rgb.width(),
        height: rgb.height(),
        rgb: rgb.into_raw(),
    })
}

fn decode_jpeg(bytes: &[u8]) -> Result<RgbImage> {
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    let rgb = decoder
        .decode()
        .map_err(|e| anyhow!("JPEG decode failed: {e:?}"))?;
    let (w, h) = decoder
        .dimensions()
        .ok_or_else(|| anyhow!("JPEG has no dimensions"))?;
    if rgb.len() != w * h * 3 {
        bail!("unexpected JPEG output size {} for {w}x{h}", rgb.len());
    }
    Ok(RgbImage {
        width: w as u32,
        height: h as u32,
        rgb,
    })
}

/// Encode as baseline JPEG.
pub fn encode_jpeg(img: &RgbImage, quality: u8) -> Result<Vec<u8>> {
    if img.width == 0 || img.height == 0 {
        bail!("cannot encode an empty image");
    }
    if img.width > u16::MAX as u32 || img.height > u16::MAX as u32 {
        bail!("image too large for JPEG ({}x{})", img.width, img.height);
    }
    if img.rgb.len() != img.width as usize * img.height as usize * 3 {
        bail!("RGB buffer size does not match dimensions");
    }
    let mut out = Vec::new();
    let enc = jpeg_encoder::Encoder::new(&mut out, quality.clamp(1, 100));
    enc.encode(
        &img.rgb,
        img.width as u16,
        img.height as u16,
        jpeg_encoder::ColorType::Rgb,
    )
    .map_err(|e| anyhow!("JPEG encode failed: {e}"))?;
    Ok(out)
}

const BOX_COLOR: [u8; 3] = [255, 0, 0];
const BAR_COLOR: [u8; 3] = [170, 0, 0];
const TEXT_COLOR: [u8; 3] = [255, 255, 255];

fn fill_rect(img: &mut RgbImage, x0: i64, y0: i64, x1: i64, y1: i64, color: [u8; 3]) {
    // Half-open [x0, x1) x [y0, y1), clamped to the image.
    let x0 = x0.clamp(0, img.width as i64) as usize;
    let x1 = x1.clamp(0, img.width as i64) as usize;
    let y0 = y0.clamp(0, img.height as i64) as usize;
    let y1 = y1.clamp(0, img.height as i64) as usize;
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let w = img.width as usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * w + x) * 3;
            img.rgb[i..i + 3].copy_from_slice(&color);
        }
    }
}

fn blend_pixel(img: &mut RgbImage, x: i64, y: i64, color: [u8; 3], coverage: f32) {
    if x < 0 || y < 0 || x >= img.width as i64 || y >= img.height as i64 {
        return;
    }
    let i = (y as usize * img.width as usize + x as usize) * 3;
    let a = coverage.clamp(0.0, 1.0);
    for (px, &src) in img.rgb[i..i + 3].iter_mut().zip(color.iter()) {
        let dst = *px as f32;
        *px = (dst + (src as f32 - dst) * a).round() as u8;
    }
}

fn draw_text(img: &mut RgbImage, font: &FontArc, px: f32, x: i64, y: i64, max_x: i64, text: &str) {
    let scaled = font.as_scaled(PxScale::from(px));
    let mut caret = x as f32;
    let baseline = y as f32 + scaled.ascent();
    for ch in text.chars() {
        let gid = scaled.glyph_id(ch);
        let glyph =
            gid.with_scale_and_position(PxScale::from(px), ab_glyph::point(caret, baseline));
        caret += scaled.h_advance(gid);
        if caret as i64 > max_x {
            break;
        }
        if let Some(outlined) = font.outline_glyph(glyph) {
            let b = outlined.px_bounds();
            outlined.draw(|gx, gy, cov| {
                blend_pixel(
                    img,
                    b.min.x as i64 + gx as i64,
                    b.min.y as i64 + gy as i64,
                    TEXT_COLOR,
                    cov,
                );
            });
        }
    }
}

/// Return a copy of `img` with a red rectangle and a `label  NN%` bar for every prediction.
/// Boxes outside the image are clamped; degenerate boxes are skipped for the rectangle.
pub fn draw_predictions(img: &RgbImage, preds: &[Prediction]) -> RgbImage {
    let mut out = img.clone();
    if out.width == 0
        || out.height == 0
        || out.rgb.len() != out.width as usize * out.height as usize * 3
    {
        return out;
    }
    let font = FontArc::try_from_slice(FONT_BYTES).ok();
    let scale = (out.width.max(out.height) as f32 / 640.0).max(0.5);
    let thickness = scale.ceil().max(1.0) as i64;
    let font_px = (16.0 * scale).ceil().max(12.0);
    let bar_h = font_px as i64;

    for p in preds {
        let (x0, y0, x1, y1) = (
            p.x_min as i64,
            p.y_min as i64,
            p.x_max as i64,
            p.y_max as i64,
        );
        if x1 > x0 && y1 > y0 {
            let t = thickness
                .min((x1 - x0 + 1) / 2)
                .min((y1 - y0 + 1) / 2)
                .max(1);
            fill_rect(&mut out, x0, y0, x1, y0 + t, BOX_COLOR);
            fill_rect(&mut out, x0, y1 - t, x1, y1, BOX_COLOR);
            fill_rect(&mut out, x0, y0, x0 + t, y1, BOX_COLOR);
            fill_rect(&mut out, x1 - t, y0, x1, y1, BOX_COLOR);
        }
        if let Some(font) = &font {
            // Keep the bar on screen even when the box touches the top edge.
            let bar_w = (x1 - x0).max(font_px as i64 * 6);
            let bar_y = y0.clamp(0, (out.height as i64 - bar_h).max(0));
            fill_rect(&mut out, x0, bar_y, x0 + bar_w, bar_y + bar_h, BAR_COLOR);
            let label = format!("{}   {:.0}%", p.label, p.confidence * 100.0);
            draw_text(&mut out, font, font_px - 1.0, x0, bar_y, x0 + bar_w, &label);
        }
    }
    out
}

/// Write `<stem>_od.jpg` (annotated) into `dir`, plus `<stem>.jpg` (unannotated) when
/// `save_ref` is set. `name` is untrusted (client-supplied): only its file stem is used.
pub fn save_annotated(
    dir: &Path,
    name: &str,
    img: &RgbImage,
    preds: &[Prediction],
    save_ref: bool,
) -> Result<()> {
    let stem = Path::new(name)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.is_empty() && s != "." && s != "..")
        .unwrap_or_else(|| format!("image_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

    let annotated = draw_predictions(img, preds);
    let od_path = dir.join(format!("{stem}_od.jpg"));
    std::fs::write(&od_path, encode_jpeg(&annotated, 95)?)
        .with_context(|| format!("writing {}", od_path.display()))?;
    if save_ref {
        let ref_path = dir.join(format!("{stem}.jpg"));
        std::fs::write(&ref_path, encode_jpeg(img, 95)?)
            .with_context(|| format!("writing {}", ref_path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(w: u32, h: u32) -> RgbImage {
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&[(x * 4) as u8, (y * 4) as u8, 128]);
            }
        }
        RgbImage {
            width: w,
            height: h,
            rgb,
        }
    }

    #[test]
    fn jpeg_roundtrip() {
        let img = synthetic(64, 48);
        let bytes = encode_jpeg(&img, 95).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!((back.width, back.height), (64, 48));
        assert_eq!(back.rgb.len(), 64 * 48 * 3);
        // Lossy: check a mid pixel is close.
        let i = (24 * 64 + 32) * 3;
        for c in 0..3 {
            assert!((back.rgb[i + c] as i32 - img.rgb[i + c] as i32).abs() < 24);
        }
    }

    #[test]
    fn png_decodes() {
        let img = synthetic(8, 8);
        let buf = image::RgbImage::from_raw(8, 8, img.rgb.clone()).unwrap();
        let mut png = Vec::new();
        buf.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        assert_eq!(decode(&png).unwrap(), img);
    }

    #[test]
    fn draw_with_out_of_range_boxes() {
        let img = synthetic(32, 32);
        let preds = vec![
            Prediction {
                x_min: 0,
                y_min: 0,
                x_max: 5000,
                y_max: 5000,
                confidence: 0.9,
                label: "big".into(),
            },
            Prediction {
                x_min: 100,
                y_min: 100,
                x_max: 200,
                y_max: 200,
                confidence: 0.5,
                label: "off".into(),
            },
            Prediction {
                x_min: 10,
                y_min: 10,
                x_max: 10,
                y_max: 10,
                confidence: 0.1,
                label: "empty".into(),
            },
            Prediction {
                x_min: 20,
                y_min: 20,
                x_max: 5,
                y_max: 5,
                confidence: 0.1,
                label: "inverted".into(),
            },
        ];
        let out = draw_predictions(&img, &preds);
        assert_eq!(out.rgb.len(), img.rgb.len());
        assert_ne!(out.rgb, img.rgb);
        let none = draw_predictions(&RgbImage::default(), &preds);
        assert!(none.rgb.is_empty());
    }

    #[test]
    fn save_writes_files() {
        let dir = std::env::temp_dir().join(format!("bo_img_test_{}", uuid::Uuid::new_v4()));
        let img = synthetic(32, 32);
        save_annotated(&dir, "../evil/cam1.jpg", &img, &[], true).unwrap();
        assert!(dir.join("cam1_od.jpg").exists());
        assert!(dir.join("cam1.jpg").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
