//! Benchmark image sets ("datasets"): the built-in sets pinned by the manifests in
//! `assets/bench/<id>.json` (downloaded on demand to `<data_root>/bench/<id>/`), user directories
//! (with optional COCO JSON or YOLO txt ground truth and an optional `manifest.json` with tags),
//! and the embedded sample image (always available, works offline).
//!
//! Manifest (`assets/bench/<id>.json`, also accepted as `<dir>/manifest.json`):
//!
//! ```json
//! { "id": "coco-cctv", "title": "...", "description": "...", "license": "...",
//!   "attribution": "...", "source": "https://...", "revision": "...",
//!   "images": [ { "file": "000000123.jpg", "url": "https://...", "sha256": "<hex>",
//!                 "size": 123456, "width": 640, "height": 480,
//!                 "tags": ["day", "cctv", "complexity:easy", "res:sd", "small-objects"],
//!                 "objects": [ { "label": "person", "bbox": [x_min, y_min, x_max, y_max],
//!                                "ignore": false } ] } ] }
//! ```
//!
//! A user directory may hold just a JSON array of `images` entries in `manifest.json` (tags, and
//! `objects` as ground truth). Without ground truth a set is scored against pseudo ground truth
//! from a reference model (see `super::pseudo_ground_truth`).

use super::metrics::{BBox, GtBox};
use crate::config::DatasetRef;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// Built-in set ids the project curates (`assets/bench/<id>.json`). An id listed here whose
/// manifest is not embedded in this build is skipped with a warning.
pub const KNOWN_SET_IDS: [&str; 3] = ["coco-cctv", "bmd45-cctv", "exdark-night"];
/// Id of the embedded sample image.
pub const SAMPLE_ID: &str = "sample";
/// Prefix of a user-directory dataset reference (`dir:/path/to/images`).
pub const DIR_PREFIX: &str = "dir:";
/// Subdirectory of the data root that holds downloaded sets.
pub const BENCH_DIR: &str = "bench";
/// Image file extensions picked up from directories.
pub const IMAGE_EXTENSIONS: [&str; 3] = ["jpg", "jpeg", "png"];

/// Manifests embedded in this build: `(id, json)`. The curated manifests are added here (with
/// `include_str!`) once they land in `assets/bench/`.
pub const BUILTIN_MANIFESTS: &[(&str, &str)] = &[
    (
        "coco-cctv",
        include_str!("../../assets/bench/coco-cctv.json"),
    ),
    (
        "bmd45-cctv",
        include_str!("../../assets/bench/bmd45-cctv.json"),
    ),
    (
        "exdark-night",
        include_str!("../../assets/bench/exdark-night.json"),
    ),
];

/// One object of a manifest image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestObject {
    pub label: String,
    /// `[x_min, y_min, x_max, y_max]`, absolute pixels of the original image.
    pub bbox: BBox,
    #[serde(default)]
    pub ignore: bool,
}

/// One image of a manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestImage {
    pub file: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Ground truth; None = unannotated.
    #[serde(default)]
    pub objects: Option<Vec<ManifestObject>>,
}

/// A set manifest (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Manifest {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub attribution: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub revision: String,
    /// Only these labels are scored on this set: ground truth of other labels is dropped and
    /// predictions of other labels are neither true nor false positives (a set annotated for
    /// vehicles only). None = every label.
    #[serde(default)]
    pub scored_labels: Option<Vec<String>>,
    pub images: Vec<ManifestImage>,
}

impl Manifest {
    /// A full manifest object, or a bare array of images.
    pub fn parse(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)?;
        if v.is_array() {
            return Ok(Self {
                images: serde_json::from_value(v)?,
                ..Self::default()
            });
        }
        Ok(serde_json::from_value(v)?)
    }

    /// Total download size of the images.
    pub fn size(&self) -> u64 {
        self.images.iter().filter_map(|i| i.size).sum()
    }

    /// Every image has `objects`.
    pub fn has_ground_truth(&self) -> bool {
        !self.images.is_empty() && self.images.iter().all(|i| i.objects.is_some())
    }
}

/// The embedded built-in manifests, parsed once.
pub fn builtin_manifests() -> &'static [Manifest] {
    static M: std::sync::OnceLock<Vec<Manifest>> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        BUILTIN_MANIFESTS
            .iter()
            .filter_map(|(id, json)| match Manifest::parse(json) {
                Ok(mut m) => {
                    if m.id.is_empty() {
                        m.id = id.to_string();
                    }
                    Some(m)
                }
                Err(e) => {
                    tracing::error!("embedded benchmark manifest {id} is invalid: {e:#}");
                    None
                }
            })
            .collect()
    })
}

pub fn builtin_manifest(id: &str) -> Option<&'static Manifest> {
    builtin_manifests()
        .iter()
        .find(|m| m.id.eq_ignore_ascii_case(id.trim()))
}

/// Where a built-in set is downloaded to.
pub fn builtin_dir(data_root: &Path, id: &str) -> PathBuf {
    data_root.join(BENCH_DIR).join(id)
}

/// Where the ground truth of a set came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "kind", content = "reference")]
pub enum GroundTruth {
    /// None: scored against pseudo ground truth, or not at all.
    None,
    /// `objects` of the manifest.
    Manifest,
    /// COCO instances JSON.
    Coco,
    /// YOLO txt labels.
    Yolo,
    /// Detections of a reference model ("rt-detrv2-x on openvino:cpu"): relative, not ground
    /// truth.
    Pseudo(String),
}

impl GroundTruth {
    pub fn is_real(&self) -> bool {
        matches!(
            self,
            GroundTruth::Manifest | GroundTruth::Coco | GroundTruth::Yolo
        )
    }

    pub fn describe(&self) -> String {
        match self {
            GroundTruth::None => "no ground truth".into(),
            GroundTruth::Manifest => "ground truth (manifest)".into(),
            GroundTruth::Coco => "ground truth (COCO JSON)".into(),
            GroundTruth::Yolo => "ground truth (YOLO labels)".into(),
            GroundTruth::Pseudo(r) => format!("relative to {r}, not ground truth"),
        }
    }
}

/// One image of a set.
#[derive(Debug, Clone, PartialEq)]
pub struct SetImage {
    /// File name relative to the set directory.
    pub file: String,
    /// None = the embedded sample.
    pub path: Option<PathBuf>,
    pub width: u32,
    pub height: u32,
    /// Manifest tags plus the derived `res:` bucket.
    pub tags: Vec<String>,
    /// Ground truth (raw labels); None = unannotated.
    pub gt: Option<Vec<GtBox>>,
}

impl SetImage {
    /// Encoded image bytes.
    pub fn bytes(&self) -> Result<Cow<'static, [u8]>> {
        match &self.path {
            None => Ok(Cow::Borrowed(super::DEFAULT_IMAGE)),
            Some(p) => std::fs::read(p)
                .map(Cow::Owned)
                .with_context(|| format!("reading {}", p.display())),
        }
    }

    /// `res:` bucket of this image.
    /// `res:` bucket: the manifest's `res:` tag when it names a known bucket, else derived from
    /// the size.
    pub fn resolution(&self) -> &'static str {
        self.tags
            .iter()
            .filter_map(|t| t.strip_prefix("res:"))
            .find_map(|r| {
                RESOLUTION_BUCKETS
                    .iter()
                    .find(|b| b.eq_ignore_ascii_case(r))
            })
            .copied()
            .unwrap_or_else(|| resolution_bucket(self.width, self.height))
    }
}

/// Resolution bucket from the long side: `sd` (up to 960 px, e.g. 640x480, 640x640), `hd` (up
/// to 1280, 720p), `fhd` (up to 1920, 1080p), `4mp+` (larger: 2560x1440 and up). The manifests
/// use the same `res:` vocabulary.
pub fn resolution_bucket(w: u32, h: u32) -> &'static str {
    match w.max(h) {
        0..=960 => "sd",
        961..=1280 => "hd",
        1281..=1920 => "fhd",
        _ => "4mp+",
    }
}

/// Bucket order for tables.
pub const RESOLUTION_BUCKETS: [&str; 4] = ["sd", "hd", "fhd", "4mp+"];

fn add_resolution_tag(tags: &mut Vec<String>, w: u32, h: u32) {
    if w > 0 && h > 0 && !tags.iter().any(|t| t.starts_with("res:")) {
        tags.push(format!("res:{}", resolution_bucket(w, h)));
    }
}

/// Kind of a dataset source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SetKind {
    Builtin,
    Dir,
    Sample,
}

/// A loaded dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSet {
    /// `coco-cctv`, `sample`, `dir:<path>` (or the user's `name`).
    pub id: String,
    pub title: String,
    pub kind: SetKind,
    /// Directory the files are in (None for the sample).
    pub dir: Option<PathBuf>,
    pub license: String,
    pub attribution: String,
    pub source: String,
    pub ground_truth: GroundTruth,
    /// See [`Manifest::scored_labels`].
    pub scored_labels: Option<Vec<String>>,
    pub images: Vec<SetImage>,
}

impl ImageSet {
    /// The embedded sample image (no ground truth).
    pub fn sample() -> Self {
        let (width, height) = image::load_from_memory(super::DEFAULT_IMAGE)
            .map(|i| (i.width(), i.height()))
            .unwrap_or((0, 0));
        let mut tags = vec!["day".to_string(), "general".to_string()];
        add_resolution_tag(&mut tags, width, height);
        Self {
            id: SAMPLE_ID.into(),
            title: "Embedded sample (dog, bicycle, car)".into(),
            kind: SetKind::Sample,
            dir: None,
            license: "public domain test image".into(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: GroundTruth::None,
            scored_labels: None,
            images: vec![SetImage {
                file: super::DEFAULT_IMAGE_NAME.into(),
                path: None,
                width,
                height,
                tags,
                gt: None,
            }],
        }
    }

    /// One image file (`--image`), no ground truth.
    pub fn single_file(path: &Path) -> Result<Self> {
        let (w, h) =
            image::image_dimensions(path).with_context(|| format!("reading {}", path.display()))?;
        let mut tags = Vec::new();
        add_resolution_tag(&mut tags, w, h);
        Ok(Self {
            id: path.display().to_string(),
            title: path.display().to_string(),
            kind: SetKind::Dir,
            dir: path.parent().map(Path::to_path_buf),
            license: String::new(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: GroundTruth::None,
            scored_labels: None,
            images: vec![SetImage {
                file: path
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                path: Some(path.to_path_buf()),
                width: w,
                height: h,
                tags,
                gt: None,
            }],
        })
    }

    /// A built-in set from its manifest and the files present in `dir`. Missing files are
    /// skipped (`missing` counts them).
    pub fn from_manifest(m: &Manifest, dir: &Path) -> (Self, usize) {
        let mut images = Vec::new();
        let mut missing = 0;
        for i in &m.images {
            let path = dir.join(&i.file);
            if !safe_file_name(&i.file) || !path.is_file() {
                missing += 1;
                continue;
            }
            images.push(manifest_image(i, path));
        }
        let gt = if m.has_ground_truth() {
            GroundTruth::Manifest
        } else {
            GroundTruth::None
        };
        (
            Self {
                id: m.id.clone(),
                title: if m.title.is_empty() {
                    m.id.clone()
                } else {
                    m.title.clone()
                },
                kind: SetKind::Builtin,
                dir: Some(dir.to_path_buf()),
                license: m.license.clone(),
                attribution: m.attribution.clone(),
                source: m.source.clone(),
                ground_truth: gt,
                scored_labels: m.scored_labels.clone(),
                images,
            },
            missing,
        )
    }

    /// Keep at most `n` images (0 = all), evenly spread over the set so every part of it (and
    /// so every kind of scene, if the set is ordered) is represented.
    pub fn limit(&mut self, n: usize) {
        let len = self.images.len();
        if n == 0 || n >= len {
            return;
        }
        let picked: Vec<SetImage> = (0..n).map(|i| self.images[i * len / n].clone()).collect();
        self.images = picked;
    }

    /// Images with ground truth (real or pseudo).
    pub fn annotated(&self) -> bool {
        self.images.iter().all(|i| i.gt.is_some()) && !self.images.is_empty()
    }
}

/// A file name without directories (`a.jpg`, not `../a.jpg` or `x/a.jpg`).
pub fn safe_file_name(f: &str) -> bool {
    !f.is_empty() && !f.contains(['/', '\\']) && f != "." && f != ".." && !f.contains('\0')
}

fn manifest_image(i: &ManifestImage, path: PathBuf) -> SetImage {
    let (w, h) = match (i.width, i.height) {
        (Some(w), Some(h)) => (w, h),
        _ => image::image_dimensions(&path).unwrap_or((0, 0)),
    };
    let mut tags = i.tags.clone();
    add_resolution_tag(&mut tags, w, h);
    SetImage {
        file: i.file.clone(),
        path: Some(path),
        width: w,
        height: h,
        tags,
        gt: i.objects.as_ref().map(|objs| {
            objs.iter()
                .map(|o| GtBox {
                    label: o.label.clone(),
                    bbox: o.bbox,
                    ignore: o.ignore,
                })
                .collect()
        }),
    }
}

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Image files directly in `dir`, sorted by name.
fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_image(p))
        .collect();
    v.sort();
    Ok(v)
}

/// Load a user directory: the images in it (or in its `images/` subdirectory), tags and/or
/// ground truth from `manifest.json`, else ground truth from `gt` (a COCO JSON file, or a
/// directory of YOLO txt labels) or found automatically (`_annotations.coco.json`,
/// `annotations.json`, `instances*.json`; `labels/` or `<stem>.txt` beside the images).
pub fn load_dir(dir: &Path, gt: Option<&Path>, name: Option<&str>) -> Result<ImageSet> {
    if !dir.is_dir() {
        bail!("benchmark image directory {} does not exist", dir.display());
    }
    let mut img_dir = dir.to_path_buf();
    let mut files = list_images(dir)?;
    if files.is_empty() && dir.join("images").is_dir() {
        img_dir = dir.join("images");
        files = list_images(&img_dir)?;
    }
    if files.is_empty() {
        bail!(
            "no .jpg/.jpeg/.png images in {} (or its images/ subdirectory)",
            dir.display()
        );
    }
    let manifest = match dir.join("manifest.json") {
        p if p.is_file() => Some(
            Manifest::parse(&std::fs::read_to_string(&p)?)
                .with_context(|| format!("parsing {}", p.display()))?,
        ),
        _ => None,
    };
    let mut images = Vec::new();
    for path in files {
        let file = path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let entry = manifest
            .as_ref()
            .and_then(|m| m.images.iter().find(|i| i.file == file));
        images.push(match entry {
            Some(e) => manifest_image(e, path),
            None => {
                let (w, h) = image::image_dimensions(&path).unwrap_or((0, 0));
                let mut tags = Vec::new();
                add_resolution_tag(&mut tags, w, h);
                SetImage {
                    file,
                    path: Some(path),
                    width: w,
                    height: h,
                    tags,
                    gt: None,
                }
            }
        });
    }
    let mut ground_truth = if images.iter().all(|i| i.gt.is_some()) {
        GroundTruth::Manifest
    } else {
        GroundTruth::None
    };
    if ground_truth == GroundTruth::None {
        let coco = match gt {
            Some(p) if p.is_file() => Some(p.to_path_buf()),
            Some(p) if p.is_dir() => None,
            Some(p) => bail!("ground truth {} does not exist", p.display()),
            None => find_coco(dir),
        };
        if let Some(p) = coco {
            apply_coco(&mut images, &p)?;
            ground_truth = GroundTruth::Coco;
        } else {
            let labels = match gt {
                Some(p) if p.is_dir() => Some(p.to_path_buf()),
                _ => find_yolo_labels(dir, &img_dir, &images),
            };
            if let Some(l) = labels {
                apply_yolo(&mut images, &l, dir)?;
                ground_truth = GroundTruth::Yolo;
            }
        }
    }
    let id = match name {
        Some(n) if !n.trim().is_empty() => n.trim().to_string(),
        _ => format!("{DIR_PREFIX}{}", dir.display()),
    };
    let m = manifest.unwrap_or_default();
    Ok(ImageSet {
        title: if m.title.is_empty() {
            id.clone()
        } else {
            m.title.clone()
        },
        id,
        kind: SetKind::Dir,
        dir: Some(img_dir),
        license: m.license,
        attribution: m.attribution,
        source: m.source,
        ground_truth,
        scored_labels: m.scored_labels,
        images,
    })
}

fn find_coco(dir: &Path) -> Option<PathBuf> {
    for n in ["_annotations.coco.json", "annotations.json"] {
        let p = dir.join(n);
        if p.is_file() {
            return Some(p);
        }
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("instances") && n.ends_with(".json"))
        })
        .collect();
    v.sort();
    v.into_iter().next()
}

#[derive(Deserialize)]
struct CocoFile {
    images: Vec<CocoImage>,
    #[serde(default)]
    annotations: Vec<CocoAnn>,
    categories: Vec<CocoCat>,
}
#[derive(Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
}
#[derive(Deserialize)]
struct CocoAnn {
    image_id: u64,
    category_id: u64,
    bbox: [f32; 4],
    #[serde(default)]
    iscrowd: u8,
}
#[derive(Deserialize)]
struct CocoCat {
    id: u64,
    name: String,
}

/// Parse a COCO instances JSON into `file name -> boxes` (crowd annotations become `ignore`).
pub fn parse_coco(text: &str) -> Result<Vec<(String, Vec<GtBox>)>> {
    let f: CocoFile = serde_json::from_str(text).context("parsing COCO JSON")?;
    let cats: std::collections::HashMap<u64, &str> = f
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();
    let mut out: Vec<(String, Vec<GtBox>)> = Vec::new();
    let mut index = std::collections::HashMap::new();
    for img in &f.images {
        let base = Path::new(&img.file_name)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| img.file_name.clone());
        index.insert(img.id, out.len());
        out.push((base, Vec::new()));
    }
    for a in &f.annotations {
        let (Some(&i), Some(name)) = (index.get(&a.image_id), cats.get(&a.category_id)) else {
            continue;
        };
        let [x, y, w, h] = a.bbox;
        out[i].1.push(GtBox {
            label: name.to_string(),
            bbox: [x, y, x + w, y + h],
            ignore: a.iscrowd != 0,
        });
    }
    Ok(out)
}

fn apply_coco(images: &mut [SetImage], path: &Path) -> Result<()> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let parsed = parse_coco(&text).with_context(|| format!("in {}", path.display()))?;
    for img in images.iter_mut() {
        // Images not listed in the JSON have no objects.
        img.gt = Some(
            parsed
                .iter()
                .find(|(f, _)| *f == img.file)
                .map(|(_, b)| b.clone())
                .unwrap_or_default(),
        );
    }
    Ok(())
}

fn find_yolo_labels(dir: &Path, img_dir: &Path, images: &[SetImage]) -> Option<PathBuf> {
    let has_txt = |d: &Path| {
        images.iter().any(|i| {
            Path::new(&i.file)
                .file_stem()
                .is_some_and(|s| d.join(format!("{}.txt", s.to_string_lossy())).is_file())
        })
    };
    [dir.join("labels"), img_dir.to_path_buf()]
        .into_iter()
        .find(|d| d.is_dir() && has_txt(d))
}

/// YOLO class names: `classes.txt` (one per line) or `data.yaml` / `dataset.yaml` (`names` list
/// or index map) in any of `dirs`; else COCO-80.
pub fn yolo_class_names(dirs: &[&Path]) -> Result<Vec<String>> {
    for d in dirs {
        let txt = d.join("classes.txt");
        if txt.is_file() {
            return Ok(std::fs::read_to_string(&txt)?
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect());
        }
        for y in ["data.yaml", "dataset.yaml"] {
            let p = d.join(y);
            if p.is_file() {
                return parse_yolo_yaml(&std::fs::read_to_string(&p)?)
                    .with_context(|| format!("parsing {}", p.display()));
            }
        }
    }
    Ok(crate::model::classes::coco80())
}

/// `names:` of a YOLO dataset YAML, as a list or an `index: name` map.
pub fn parse_yolo_yaml(text: &str) -> Result<Vec<String>> {
    let v: serde_yaml_ng::Value = serde_yaml_ng::from_str(text)?;
    let names = v.get("names").context("no `names` in the dataset YAML")?;
    if let Some(seq) = names.as_sequence() {
        return Ok(seq
            .iter()
            .map(|n| n.as_str().unwrap_or_default().to_string())
            .collect());
    }
    if let Some(map) = names.as_mapping() {
        let mut pairs: Vec<(u64, String)> = map
            .iter()
            .filter_map(|(k, v)| Some((k.as_u64()?, v.as_str()?.to_string())))
            .collect();
        pairs.sort();
        let len = pairs.last().map_or(0, |p| p.0 as usize + 1);
        let mut out = vec![String::new(); len];
        for (i, n) in pairs {
            out[i as usize] = n;
        }
        return Ok(out);
    }
    bail!("`names` must be a list or a map")
}

/// Parse one YOLO label file (`class cx cy w h`, normalized) into boxes for a `w x h` image.
pub fn parse_yolo(text: &str, names: &[String], w: u32, h: u32) -> Result<Vec<GtBox>> {
    let (w, h) = (w as f32, h as f32);
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() {
            continue;
        }
        if f.len() < 5 {
            bail!("line {}: expected `class cx cy w h`", n + 1);
        }
        let cls: usize = f[0]
            .parse()
            .with_context(|| format!("line {}: bad class '{}'", n + 1, f[0]))?;
        let v: Vec<f32> = f[1..5]
            .iter()
            .map(|s| s.parse::<f32>())
            .collect::<Result<_, _>>()
            .with_context(|| format!("line {}: bad number", n + 1))?;
        let label = names
            .get(cls)
            .cloned()
            .unwrap_or_else(|| format!("class{cls}"));
        let (cx, cy, bw, bh) = (v[0] * w, v[1] * h, v[2] * w, v[3] * h);
        out.push(GtBox {
            label,
            bbox: [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
            ignore: false,
        });
    }
    Ok(out)
}

fn apply_yolo(images: &mut [SetImage], labels: &Path, dir: &Path) -> Result<()> {
    let names = yolo_class_names(&[labels, dir])?;
    for img in images.iter_mut() {
        let stem = Path::new(&img.file)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let p = labels.join(format!("{stem}.txt"));
        // No label file: an image without objects.
        let text = std::fs::read_to_string(&p).unwrap_or_default();
        img.gt = Some(
            parse_yolo(&text, &names, img.width, img.height)
                .with_context(|| format!("in {}", p.display()))?,
        );
    }
    Ok(())
}

/// Datasets resolved for a run.
#[derive(Debug, Clone, Default)]
pub struct Resolved {
    pub sets: Vec<ImageSet>,
    /// Problems that did not stop the run (sets skipped, files missing).
    pub warnings: Vec<String>,
    /// Built-in sets with no image on disk (download `bench:<id>`).
    pub not_downloaded: Vec<String>,
}

/// Every dataset id accepted besides `dir:<path>`: `sample` and the built-in sets.
pub fn valid_ids() -> Vec<String> {
    let mut v = vec![SAMPLE_ID.to_string()];
    for id in KNOWN_SET_IDS {
        v.push(id.to_string());
    }
    for m in builtin_manifests() {
        if !v.contains(&m.id) {
            v.push(m.id.clone());
        }
    }
    v
}

fn dir_of(d: &DatasetRef, data_root: &Path) -> Option<(PathBuf, Option<PathBuf>, Option<String>)> {
    let abs = |p: PathBuf| {
        if p.is_absolute() {
            p
        } else {
            data_root.join(p)
        }
    };
    d.dir().map(|(dir, gt, name)| (abs(dir), gt.map(abs), name))
}

/// Check dataset references without loading them: ids must be known, directories must exist
/// (relative ones are under `data_root`).
pub fn validate(refs: &[DatasetRef], data_root: &Path) -> Result<()> {
    let valid = valid_ids();
    for d in refs {
        match dir_of(d, data_root) {
            Some((dir, gt, _)) => {
                if !dir.is_dir() {
                    bail!(
                        "dataset {}: directory {} does not exist",
                        d.label(),
                        dir.display()
                    );
                }
                if let Some(g) = gt
                    && !g.exists()
                {
                    bail!(
                        "dataset {}: ground truth {} does not exist",
                        d.label(),
                        g.display()
                    );
                }
            }
            None => {
                let id = d.label();
                if !valid.iter().any(|v| v.eq_ignore_ascii_case(&id)) {
                    bail!(
                        "unknown dataset '{id}'; valid ids: {}, or dir:<path> for a folder of images",
                        valid.join(", ")
                    );
                }
            }
        }
    }
    Ok(())
}

/// Load the referenced datasets, each limited to `max_images` (0 = all). Built-in sets use the
/// files present under `<data_root>/bench/<id>/`; a set with none is listed in
/// [`Resolved::not_downloaded`]. When nothing is left, the embedded sample is used (with a
/// warning), so a run always has an image.
pub fn resolve(refs: &[DatasetRef], data_root: &Path, max_images: usize) -> Result<Resolved> {
    validate(refs, data_root)?;
    let mut out = Resolved::default();
    for d in refs {
        let mut set = match dir_of(d, data_root) {
            Some((dir, gt, name)) => {
                // The id stays as configured (`dir:<path>` or the name).
                let id = name.unwrap_or_else(|| d.label());
                load_dir(&dir, gt.as_deref(), Some(&id))
                    .with_context(|| format!("dataset {}", d.label()))?
            }
            None => {
                let id = d.label();
                if id.eq_ignore_ascii_case(SAMPLE_ID) {
                    ImageSet::sample()
                } else if let Some(m) = builtin_manifest(&id) {
                    let (set, missing) = ImageSet::from_manifest(m, &builtin_dir(data_root, &m.id));
                    if set.images.is_empty() {
                        out.not_downloaded.push(m.id.clone());
                        out.warnings.push(format!(
                            "dataset {} is not downloaded (resource bench:{})",
                            m.id, m.id
                        ));
                        continue;
                    }
                    if missing > 0 {
                        out.warnings.push(format!(
                            "dataset {}: {missing} of {} images are not downloaded; using the {} present",
                            m.id,
                            m.images.len(),
                            set.images.len()
                        ));
                    }
                    set
                } else {
                    out.warnings.push(format!(
                        "dataset {id} is not available in this build (its manifest is not embedded)"
                    ));
                    continue;
                }
            }
        };
        if out.sets.iter().any(|s| s.id == set.id) {
            continue;
        }
        set.limit(max_images);
        out.sets.push(set);
    }
    if out.sets.is_empty() {
        out.warnings
            .push("no dataset images available; using the embedded sample image".to_string());
        out.sets.push(ImageSet::sample());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/bench")
    }

    #[test]
    fn resolution_buckets() {
        assert_eq!(resolution_bucket(640, 480), "sd");
        assert_eq!(resolution_bucket(640, 640), "sd");
        assert_eq!(resolution_bucket(704, 576), "sd");
        // A manifest `res:` tag wins over the size.
        let mut img = ImageSet::sample().images.remove(0);
        img.width = 1280;
        img.height = 720;
        img.tags = vec!["res:sd".into()];
        assert_eq!(img.resolution(), "sd");
        img.tags = vec!["res:weird".into()];
        assert_eq!(img.resolution(), "hd");
        assert_eq!(resolution_bucket(1280, 720), "hd");
        assert_eq!(resolution_bucket(1920, 1080), "fhd");
        assert_eq!(resolution_bucket(2560, 1440), "4mp+");
        assert_eq!(resolution_bucket(3840, 2160), "4mp+");
    }

    #[test]
    fn sample_set_is_always_available() {
        let s = ImageSet::sample();
        assert_eq!(s.images.len(), 1);
        assert!(s.images[0].width > 0);
        assert!(s.images[0].tags.iter().any(|t| t.starts_with("res:")));
        assert_eq!(
            s.images[0].bytes().unwrap().len(),
            super::super::DEFAULT_IMAGE.len()
        );
        assert!(!s.annotated());
    }

    #[test]
    fn manifest_fixture_loads_present_files() {
        let dir = fixture().join("manifest-set");
        let m =
            Manifest::parse(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(m.id, "test-set");
        assert!(m.has_ground_truth());
        let (set, missing) = ImageSet::from_manifest(&m, &dir);
        assert_eq!(missing, 1, "the fixture lists one file that is not there");
        assert_eq!(set.images.len(), 2);
        assert_eq!(set.ground_truth, GroundTruth::Manifest);
        let a = &set.images[0];
        assert_eq!((a.width, a.height), (64, 48));
        assert!(a.tags.contains(&"night".to_string()));
        assert!(a.tags.contains(&"res:sd".to_string()));
        let gt = a.gt.as_ref().unwrap();
        assert_eq!(gt.len(), 2);
        assert!(gt[1].ignore);
        // Bare-array manifests work too.
        let m = Manifest::parse(r#"[{"file":"x.jpg","tags":["day"]}]"#).unwrap();
        assert_eq!(m.images.len(), 1);
        assert!(!m.has_ground_truth());
    }

    #[test]
    fn limit_spreads_evenly() {
        let mut s = ImageSet::sample();
        let img = s.images[0].clone();
        s.images = (0..10)
            .map(|i| SetImage {
                file: format!("{i}.jpg"),
                ..img.clone()
            })
            .collect();
        s.limit(3);
        let files: Vec<&str> = s.images.iter().map(|i| i.file.as_str()).collect();
        assert_eq!(files, ["0.jpg", "3.jpg", "6.jpg"]);
        s.limit(0);
        assert_eq!(s.images.len(), 3);
    }

    #[test]
    fn coco_parser() {
        let text = r#"{"images":[{"id":7,"file_name":"sub/a.jpg","width":100,"height":80}],
            "annotations":[{"image_id":7,"category_id":3,"bbox":[10,20,30,40],"iscrowd":0},
                           {"image_id":7,"category_id":1,"bbox":[0,0,50,50],"iscrowd":1},
                           {"image_id":99,"category_id":1,"bbox":[0,0,1,1]}],
            "categories":[{"id":1,"name":"person"},{"id":3,"name":"car"}]}"#;
        let p = parse_coco(text).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].0, "a.jpg");
        assert_eq!(p[0].1.len(), 2);
        assert_eq!(p[0].1[0].label, "car");
        assert_eq!(p[0].1[0].bbox, [10.0, 20.0, 40.0, 60.0]);
        assert!(p[0].1[1].ignore);
        assert!(parse_coco("{").is_err());
    }

    #[test]
    fn yolo_parser_and_names() {
        let names = vec!["person".to_string(), "car".to_string()];
        let b = parse_yolo("1 0.5 0.5 0.2 0.4\n\n0 0.1 0.1 0.1 0.1\n", &names, 200, 100).unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].label, "car");
        assert_eq!(b[0].bbox, [80.0, 30.0, 120.0, 70.0]);
        assert!(parse_yolo("0 0.1 0.1", &names, 1, 1).is_err());
        assert!(parse_yolo("x 0.1 0.1 0.1 0.1", &names, 1, 1).is_err());
        assert_eq!(
            parse_yolo("5 .5 .5 .1 .1", &names, 10, 10).unwrap()[0].label,
            "class5"
        );
        assert_eq!(parse_yolo_yaml("names: [a, b]").unwrap(), ["a", "b"]);
        assert_eq!(
            parse_yolo_yaml("names:\n  1: car\n  0: person\n").unwrap(),
            ["person", "car"]
        );
        assert!(parse_yolo_yaml("nc: 2").is_err());
    }

    #[test]
    fn user_dirs_with_coco_yolo_or_nothing() {
        let coco = load_dir(&fixture().join("coco-dir"), None, Some("mine")).unwrap();
        assert_eq!(coco.id, "mine");
        assert_eq!(coco.ground_truth, GroundTruth::Coco);
        assert_eq!(coco.images.len(), 2);
        // b.jpg has no annotations: annotated with zero objects.
        assert_eq!(coco.images[1].gt.as_ref().unwrap().len(), 0);
        assert_eq!(coco.images[0].gt.as_ref().unwrap()[0].label, "person");

        let yolo = load_dir(&fixture().join("yolo-dir"), None, None).unwrap();
        assert_eq!(yolo.ground_truth, GroundTruth::Yolo);
        assert!(yolo.id.starts_with(DIR_PREFIX));
        let gt = yolo.images[0].gt.as_ref().unwrap();
        assert_eq!(gt[0].label, "dog");
        // 64x48 image, box centered, half size.
        assert_eq!(gt[0].bbox, [16.0, 12.0, 48.0, 36.0]);

        let plain = load_dir(&fixture().join("plain-dir"), None, None).unwrap();
        assert_eq!(plain.ground_truth, GroundTruth::None);
        assert!(!plain.annotated());

        let err = load_dir(&fixture().join("nope"), None, None).unwrap_err();
        assert!(format!("{err:#}").contains("does not exist"));
        let err = load_dir(
            &fixture().join("plain-dir"),
            Some(&fixture().join("x.json")),
            None,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("does not exist"));
    }

    #[test]
    fn file_names_are_checked() {
        assert!(safe_file_name("a.jpg"));
        for bad in ["", "..", "../a.jpg", "x/a.jpg", "x\\a.jpg"] {
            assert!(!safe_file_name(bad), "{bad}");
        }
    }

    #[test]
    fn dataset_resolution_and_validation() {
        let root = fixture();
        let id = |s: &str| DatasetRef::Id(s.to_string());
        // Relative directories are under the data root; the sample always works.
        let r = resolve(
            &[
                id("sample"),
                id("dir:coco-dir"),
                DatasetRef::Dir {
                    dir: "yolo-dir".into(),
                    gt: None,
                    name: Some("yolo".into()),
                },
                id("sample"),
            ],
            &root,
            1,
        )
        .unwrap();
        let ids: Vec<&str> = r.sets.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["sample", "dir:coco-dir", "yolo"]);
        assert_eq!(r.sets[1].images.len(), 1, "limited to 1 image");
        // Known built-in ids without an embedded manifest are skipped with a warning, and the
        // sample stands in when nothing is left.
        let r = resolve(&[id("coco-cctv")], &root, 0).unwrap();
        assert_eq!(r.sets[0].id, "sample");
        assert!(r.warnings.len() >= 2, "{:?}", r.warnings);
        // Unknown ids and missing directories are errors naming the valid ids.
        let e = validate(&[id("nope")], &root).unwrap_err();
        let msg = format!("{e:#}");
        assert!(
            msg.contains("unknown dataset 'nope'") && msg.contains("coco-cctv"),
            "{msg}"
        );
        let e = validate(&[id("dir:missing")], &root).unwrap_err();
        assert!(format!("{e:#}").contains("does not exist"));
        let e = validate(
            &[DatasetRef::Dir {
                dir: "plain-dir".into(),
                gt: Some("nope.json".into()),
                name: None,
            }],
            &root,
        )
        .unwrap_err();
        assert!(format!("{e:#}").contains("ground truth"));
    }

    #[test]
    fn builtin_sets_are_catalog_resources() {
        let ids: Vec<&str> = builtin_manifests().iter().map(|m| m.id.as_str()).collect();
        for id in KNOWN_SET_IDS {
            assert!(ids.contains(&id), "{id} is embedded");
        }
        let coco: Vec<String> = crate::model::classes::coco80()
            .iter()
            .map(|l| super::super::metrics::canonical_label(l))
            .collect();
        for m in builtin_manifests() {
            let r = crate::resources::catalog::bench_set(&m.id).unwrap();
            assert_eq!(r.parts.len(), m.images.len(), "{}", m.id);
            assert_eq!(r.size(), m.size());
            assert_eq!(r.dest, format!("{BENCH_DIR}/{}", m.id));
            assert!(
                r.parts
                    .iter()
                    .all(|p| p.url.starts_with("https://") && p.sha256.len() == 64)
            );
            assert!(m.has_ground_truth());
            assert_eq!(crate::resources::status::group_of(r), "bench-images");
            assert_eq!(
                crate::resources::catalog::find(&format!("bench:{}", m.id), "linux", "x86_64")
                    .map(|r| r.id),
                Some(r.id)
            );
            // Every label is a COCO-80 label (any spelling), so models map onto it.
            for i in &m.images {
                assert!(safe_file_name(&i.file));
                for o in i.objects.iter().flatten() {
                    let c = super::super::metrics::canonical_label(&o.label);
                    assert!(coco.contains(&c), "{}: {}", m.id, o.label);
                }
            }
        }
    }
}
