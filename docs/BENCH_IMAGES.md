# Benchmark image sets

The benchmark grades every model x runtime for accuracy and speed on three pinned image sets.
Each set is a JSON manifest in `assets/bench/`. A manifest lists the image URL, SHA-256, size and
dimensions, tags and ground-truth boxes for each image. The repository holds no images: the
benchmark downloads them from the original hosts on demand and checks each one against its SHA-256.

| Set | Images | Download | Content | Resolution |
| --- | --- | --- | --- | --- |
| `bmd45-cctv` | 44 | 142 MB | Real Bengaluru Safe City CCTV, daytime, vehicles only | 1920x1080 PNG (uniform) |
| `exdark-night` | 160 | 7 MB | Low-light and night photos: people, vehicles, pets | 640x640 JPEG (uniform, stretched) |
| `coco-cctv` | 180 | 32 MB | COCO val2017 street/people/vehicle/animal/yard scenes | <= 640 px long side |

## Manifest schema

```json
{ "id", "title", "description", "license", "attribution", "source", "revision",
  "scored_labels": ["..."],            // optional: only these labels are scored on this set
  "images": [ { "file", "url", "sha256", "size", "width", "height", "tags": ["..."],
                "objects": [ { "label", "bbox": [x_min, y_min, x_max, y_max], "ignore": false } ] } ] }
```

- `label` is always a COCO-80 name spelled as in `assets/coco_classes.yaml` (for example
  `motorbike`, `diningtable`, `tvmonitor`).
- `bbox` is in absolute pixels of the original image and is clipped to the image bounds.
- `ignore: true` marks a region that is neither a true positive nor a false positive. The
  `exdark-night` and `bmd45-cctv` sets also set `scored_labels`, so the engine scores only
  those labels there. Detections of any other label are not penalised.
- Tags:
  - `day` / `night`
  - `cctv` / `general`
  - `complexity:easy` (1-2 scored objects), `complexity:medium` (3-9) or `complexity:hard`
    (10 or more scored objects, or at least 3 small ones, or at least 4 scored boxes that overlap
    another one at IoU >= 0.3, used as a stand-in for heavy occlusion)
  - `empty` (no scored objects)
  - `res:sd` / `res:hd` / `res:fhd` / `res:4mp+` (under 0.5 / 1.5 / 3 MP, or 3 MP and above)
  - `small-objects` (a scored box with area under 32x32 px)
  - `crowd` (8 or more persons)

## The sets

### bmd45-cctv: real CCTV (BMD-45)

- Source: [iisc-aim/BMD-45](https://huggingface.co/datasets/iisc-aim/BMD-45), pinned at revision
  `3e3f6a0b4834d282890c34986216c487914c5dff`, `BMD-45-Val` split.
- Images: frames from 3,679 Bengaluru Police Safe City cameras, captured in February 2025 between
  06:00 and 18:00, all 1920x1080 PNG (about 3.3 MB each). Every image is therefore tagged
  `res:fhd`, and `day/night` is set from mean luminance (all came out `day`). File names are
  anonymised and there is no camera ID, so camera diversity comes from seeded random sampling
  over the whole split.
- Class map:

  | BMD-45 class | Mapped to |
  | --- | --- |
  | Hatchback, Sedan, SUV, MUV, Van, Tempo-traveller | `car` |
  | Bus, Mini-bus | `bus` |
  | Truck, LCV (light goods carrier) | `truck` |
  | Two-wheeler | `motorbike` (the box includes the rider) |
  | Bicycle | `bicycle` (the box includes the rider) |
  | Three-wheeler (auto-rickshaw) | ignore region labelled `car` |
  | Other | ignore region labelled `truck` (no such annotations in val) |

- There is no person ground truth, so `scored_labels` is `car, bus, truck, motorbike, bicycle`.
- Very distant vehicles are often not annotated, and the dataset favours dense, occluded frames,
  so only 3 truly empty frames exist.
- The 150 MB budget limits this set to 44 images (363 scored objects). Use `--max-mb 500` for about
  150 images.

### exdark-night: low light (ExDark)

- Source: [dronefreak/ExDark](https://huggingface.co/datasets/dronefreak/ExDark), pinned at
  revision `980c106507f08ce9200b84b8308028eea1f41c97`, `data/test`.
- This copy is a Roboflow YOLO export of the original ExDark dataset. Roboflow resized every
  image to 640x640 with stretching, so aspect ratios are distorted and every image is tagged
  `res:sd`.
- Class map: `People` maps to `person`; the other classes map to their lower-case COCO-80 names,
  and `Table` maps to `diningtable`.
- `scored_labels`: person, bicycle, car, motorbike, bus, dog, cat. Boat, bottle, chair, cup and
  diningtable boxes are ignore regions.
- Selection favours street, vehicle, people and pet scenes over indoor table scenes.

### coco-cctv: COCO val2017 subset

- Source: COCO val2017. The annotations come from `annotations_trainval2017.zip` (SHA-256
  `113a836d...0268`, pinned in the script). Images are fetched individually from
  `https://s3.amazonaws.com/images.cocodataset.org/val2017/<file>`. The address
  `https://images.cocodataset.org` serves the same objects, but its TLS certificate does not
  match the host name.
- Candidate images contain at least one CCTV focus class (person, bicycle, car, motorbike, bus,
  truck, bird, cat, dog, horse). Images with indoor, food or sports-gear objects are excluded,
  and so are trains, aircraft and zoo animals. Street furniture (traffic light, stop sign,
  hydrant, bench, parking meter) and animals get extra weight. The `empty` negatives are val2017
  images that have no annotations at all.
- All 80 labels are scored. `iscrowd` boxes are ignore regions. COCO names map to the project's
  spellings: motorcycle -> motorbike, airplane -> aeroplane, couch -> sofa,
  potted plant -> pottedplant, dining table -> diningtable, tv -> tvmonitor.
- An image is tagged `night` when its mean luminance is below 50/255 or one of its COCO captions
  mentions night, dark, evening or dusk. Images with a night caption get a selection bonus.

## Licences and attribution

- **BMD-45**: CC BY 4.0, by the AI for Integrated Mobility (AIM) group, IISc Bengaluru. Imagery
  courtesy of the Bengaluru Traffic Police and Bengaluru Police Safe City. Cite: Sharma et al.,
  "BMD-45: Bengaluru Mobility Dataset for Large-Scale Vehicle Detection from Urban CCTV",
  CVPR Findings 2026.
- **ExDark**: BSD-3-Clause, by Yuen Peng Loh and Chee Seng Chan, Universiti Malaya. Cite: "Getting
  to Know Low-light Images with The Exclusively Dark Dataset", CVIU 178 (2019),
  doi:10.1016/j.cviu.2018.10.010. The authors ask that commercial use be cleared with Dr. Chee
  Seng Chan. The intermediate Roboflow export labels itself CC BY 4.0.
- **COCO**: annotations are CC BY 4.0 from the COCO Consortium (Lin et al., ECCV 2014). Images keep
  their individual Flickr licences (the CC 2.0 variants, listed per image in
  `instances_val2017.json`).
- The class remapping and subset selection were done by this project. The attribution lines are
  repeated in `LICENSE`, and each manifest carries its own `license` and `attribution` fields.

## Regenerating

```sh
uv venv --python 3.11 .venv && uv pip install -r scripts/requirements-bench.txt   # or reuse the export .venv
.venv/bin/python -I scripts/build_bench_sets.py --work /path/to/scratch            # all three sets
.venv/bin/python -I scripts/build_bench_sets.py --work /path/to/scratch --set coco-cctv
.venv/bin/python -I scripts/build_bench_sets.py --work /path/to/scratch --validate --head
```

- `--work` is a cache for downloads. It needs about 240 MB for the COCO annotations zip, 14 MB for
  the BMD-45 annotations, and room for the selected images. Treat everything in it as untrusted
  data.
- Other options:
  - `--seed` (default `20261010`)
  - `--count` (target number of images)
  - `--max-mb` (per-set budget, default 150; BMD-45 hits it first)
- Selection is deterministic: the same seed and pinned sources give byte-identical manifests.
- Each run downloads every selected image, decodes it with Pillow and checks the decoded size
  against the annotations. For Hugging Face files it also checks the LFS SHA-256. It then writes
  the manifest in stable order and validates every set: schema, labels within COCO-80, boxes
  inside the image, and the SHA-256 and size of the local copy. Add `--head` to also send an
  HTTP HEAD to every URL.
- Never commit the downloaded images.
