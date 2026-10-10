#!/usr/bin/env python3
"""Build the pinned benchmark image manifests in assets/bench/.

Three sets are curated from public datasets (images are NOT committed; the manifests pin
URL + SHA-256 + size and the ground truth, and the benchmark downloads on demand):

  bmd45-cctv    real Bengaluru Safe City CCTV frames (BMD-45 val, CC BY 4.0), vehicles only
  exdark-night  low-light / night photos (ExDark test split via dronefreak/ExDark, BSD-3-Clause)
  coco-cctv     COCO val2017 subset with CCTV-relevant content (annotations CC BY 4.0)

Selection is deterministic for a given --seed and pinned source revision: candidates are
sorted, shuffled with a seeded RNG and picked greedily per complexity bucket, favouring
under-represented labels and tags. Each selected image is downloaded into the work dir,
decoded with Pillow, and its sha256/size/width/height are written to the manifest.

Usage (Python 3.10+, needs Pillow; see scripts/requirements-bench.txt):

  .venv/bin/python -I scripts/build_bench_sets.py --work /tmp/bench-src            # all sets
  .venv/bin/python -I scripts/build_bench_sets.py --work /tmp/bench-src --set coco-cctv
  .venv/bin/python -I scripts/build_bench_sets.py --work /tmp/bench-src --validate  # re-check

Downloads are treated as untrusted data: file names are sanitised, archives are read by
member name only (nothing is extracted to paths from the archive), nothing is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

try:
    from PIL import Image, ImageStat
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: pip install -r scripts/requirements-bench.txt")

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "assets" / "bench"
COCO80_YAML = REPO / "assets" / "coco_classes.yaml"
USER_AGENT = "blue-onyx-prism-bench-builder/1.0"

BMD45_REPO = "iisc-aim/BMD-45"
BMD45_REV = "3e3f6a0b4834d282890c34986216c487914c5dff"
EXDARK_REPO = "dronefreak/ExDark"
EXDARK_REV = "980c106507f08ce9200b84b8308028eea1f41c97"
# images.cocodataset.org is an S3 bucket whose https certificate does not match the host name,
# so the https path-style S3 URL is used (same objects, verified byte-identical).
COCO_BASE = "https://s3.amazonaws.com/images.cocodataset.org"
COCO_ANN_ZIP = f"{COCO_BASE}/annotations/annotations_trainval2017.zip"
COCO_ANN_ZIP_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"

SMALL_AREA = 32 * 32
NIGHT_LUMA = 50.0  # mean Pillow "L" (0..255) below this => night / low-light
EMPTY_FRACTION = 0.07
CCTV_FOCUS = ["person", "bicycle", "car", "motorbike", "bus", "truck", "bird", "cat", "dog", "horse"]

DEFAULTS = {
    # BMD-45 frames are 1920x1080 PNG, ~3.3 MB each: the 150 MB budget caps this set (~45 images).
    "bmd45-cctv": {"count": 160, "max_mb": 150.0},
    "exdark-night": {"count": 160, "max_mb": 150.0},
    "coco-cctv": {"count": 180, "max_mb": 150.0},
}


# --------------------------------------------------------------------------------------------
# helpers


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_coco80() -> list[str]:
    names = []
    for line in COCO80_YAML.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if m:
            names.append(m.group(1))
    if len(names) != 80:
        raise SystemExit(f"expected 80 names in {COCO80_YAML}, got {len(names)}")
    return names


def http_get(url: str, *, retries: int = 4, timeout: float = 60.0) -> tuple[bytes, dict]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), dict(resp.headers.items())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404):
                raise
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def http_head(url: str) -> tuple[int, int | None]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        size = resp.headers.get("Content-Length")
        return resp.status, int(size) if size else None


def safe_name(name: str) -> str:
    base = name.replace("\\", "/").split("/")[-1]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}", base) or ".." in base:
        raise ValueError(f"unsafe file name from source: {name!r}")
    return base


def cached(path: Path, url: str, expect_sha: str | None = None) -> bytes:
    if path.exists():
        data = path.read_bytes()
        if expect_sha is None or hashlib.sha256(data).hexdigest() == expect_sha:
            return data
    log(f"  GET {url}")
    data, _ = http_get(url, timeout=600)
    if expect_sha and hashlib.sha256(data).hexdigest() != expect_sha:
        raise RuntimeError(f"sha256 mismatch for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data


def hf_resolve(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"


def hf_tree(repo: str, rev: str, path: str, cache: Path) -> list[dict]:
    """List a dataset folder at a pinned revision (follows Link: rel=next pagination)."""
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/{urllib.parse.quote(path)}"
    out: list[dict] = []
    while url:
        log(f"  LIST {url}")
        body, headers = http_get(url)
        out.extend(json.loads(body))
        link = headers.get("Link") or headers.get("link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else ""
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def clip_box(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> list[float] | None:
    x0, x1 = max(0.0, min(float(w), x0)), max(0.0, min(float(w), x1))
    y0, y1 = max(0.0, min(float(h), y0)), max(0.0, min(float(h), y1))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def res_tag(w: int, h: int) -> str:
    mp = w * h / 1e6
    if mp < 0.5:
        return "res:sd"
    if mp < 1.5:
        return "res:hd"
    if mp < 3.0:
        return "res:fhd"
    return "res:4mp+"


def structural_tags(cand: dict, scored: set[str] | None) -> list[str]:
    """Tags derivable from GT + size (complexity, res, small-objects, crowd, empty)."""
    objs = [o for o in cand["objects"] if not o["ignore"] and (scored is None or o["label"] in scored)]
    n = len(objs)
    areas = [(o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]) for o in objs]
    n_small = sum(1 for a in areas if a < SMALL_AREA)
    # occlusion proxy: scored boxes overlapping another scored box with IoU >= 0.3
    occluded = 0
    if 3 <= n <= 400:
        for i, a in enumerate(objs):
            if any(box_iou(a["bbox"], b["bbox"]) >= 0.3 for j, b in enumerate(objs) if j != i):
                occluded += 1
    tags = []
    if n == 0:
        tags.append("empty")
    elif n >= 10 or n_small >= 3 or occluded >= 4:
        tags.append("complexity:hard")
    elif n >= 3:
        tags.append("complexity:medium")
    else:
        tags.append("complexity:easy")
    tags.append(res_tag(cand["width"], cand["height"]))
    if n_small:
        tags.append("small-objects")
    if sum(1 for o in objs if o["label"] == "person") >= 8:
        tags.append("crowd")
    return tags


def complexity_of(tags: list[str]) -> str:
    for t in tags:
        if t.startswith("complexity:") or t == "empty":
            return t
    raise ValueError(tags)


# --------------------------------------------------------------------------------------------
# selection


def select(
    cands: list[dict],
    *,
    count: int,
    max_bytes: float,
    seed: int,
    bonus_tags: dict[str, float],
    scored: set[str] | None,
    est_size: float,
) -> list[dict]:
    """Seeded, stratified greedy pick.

    Buckets: empty (~7%), complexity easy/medium/hard (equal shares, shortfalls redistributed).
    Within a bucket the next pick maximises: static weight + label rarity + tag rarity bonus.
    Buckets are visited round-robin so a byte budget is shared fairly.
    """
    rng = random.Random(seed)
    pool = sorted(cands, key=lambda c: c["file"])
    rng.shuffle(pool)
    buckets: dict[str, list[dict]] = {k: [] for k in ("empty", "complexity:easy", "complexity:medium", "complexity:hard")}
    for c in pool:
        buckets[complexity_of(c["tags"])].append(c)

    sizes = sorted(c.get("size_hint", est_size) for c in pool)
    if sizes:  # a byte budget smaller than count * typical size lowers the effective count
        count = min(count, int(max_bytes // sizes[len(sizes) // 2]))
    n_empty = min(len(buckets["empty"]), max(1, round(count * EMPTY_FRACTION)))
    quotas = {"empty": n_empty}
    rest = count - n_empty
    names = ["complexity:easy", "complexity:medium", "complexity:hard"]
    avail = {k: len(buckets[k]) for k in names}
    remaining = list(names)
    left = rest
    for k in names:
        quotas[k] = 0
    while left > 0 and remaining:
        share = max(1, left // len(remaining))
        nxt = []
        for k in remaining:
            take = min(share, avail[k] - quotas[k], left)
            quotas[k] += take
            left -= take
            if avail[k] - quotas[k] > 0:
                nxt.append(k)
            if left <= 0:
                break
        if nxt == remaining and share == 0:
            break
        remaining = nxt

    label_imgs: Counter = Counter()
    tag_imgs: Counter = Counter()
    picked: list[dict] = []
    used_bytes = 0.0
    taken = {k: 0 for k in quotas}
    window = 400

    def gain(c: dict) -> float:
        g = c.get("weight", 0.0)
        labels = {o["label"] for o in c["objects"] if not o["ignore"] and (scored is None or o["label"] in scored)}
        g += sum(1.0 / (1 + label_imgs[lb]) for lb in labels)
        for t, w in bonus_tags.items():
            if t in c["tags"]:
                g += w / (1 + tag_imgs[t])
        return g

    order = ["complexity:hard", "complexity:medium", "complexity:easy", "empty"]
    progress = True
    while progress and len(picked) < count:
        progress = False
        for k in order:
            if taken[k] >= quotas[k] or not buckets[k]:
                continue
            best_i, best_g = -1, float("-inf")
            for i, c in enumerate(buckets[k][:window]):
                size = c.get("size_hint", est_size)
                if used_bytes + size > max_bytes:
                    continue
                g = gain(c)
                if g > best_g:
                    best_i, best_g = i, g
            if best_i < 0:
                quotas[k] = taken[k]  # budget exhausted for this bucket
                continue
            c = buckets[k].pop(best_i)
            picked.append(c)
            taken[k] += 1
            used_bytes += c.get("size_hint", est_size)
            label_imgs.update({o["label"] for o in c["objects"] if not o["ignore"]})
            tag_imgs.update(c["tags"])
            progress = True
    log(f"  selected {len(picked)} (quotas {quotas}, est {used_bytes / 1e6:.1f} MB)")
    return picked


# --------------------------------------------------------------------------------------------
# download + finalise


def fetch_image(c: dict, img_dir: Path) -> dict:
    path = img_dir / c["file"]
    data = cached(path, c["url"], c.get("sha_hint"))
    sha = hashlib.sha256(data).hexdigest()
    if c.get("sha_hint") and sha != c["sha_hint"]:
        raise RuntimeError(f"sha256 mismatch {c['file']}")
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        w, h = im.size
        luma = ImageStat.Stat(im.convert("L")).mean[0]
    if (w, h) != (c["width"], c["height"]):
        raise RuntimeError(f"{c['file']}: decoded {w}x{h}, annotations say {c['width']}x{c['height']}")
    return {"sha256": sha, "size": len(data), "width": w, "height": h, "luma": luma}


def finalise(set_id: str, meta: dict, picked: list[dict], work: Path, day_night: str) -> dict:
    img_dir = work / set_id / "images"
    images = []
    for i, c in enumerate(sorted(picked, key=lambda c: c["file"])):
        info = fetch_image(c, img_dir)
        if day_night == "night":
            dn = "night"
        else:
            dn = "night" if info["luma"] < NIGHT_LUMA or c.get("night_hint") else "day"
        tags = [dn, meta["kind_tag"]] + c["tags"]
        images.append(
            {
                "file": c["file"],
                "url": c["url"],
                "sha256": info["sha256"],
                "size": info["size"],
                "width": info["width"],
                "height": info["height"],
                "tags": tags,
                "objects": c["objects"],
            }
        )
        if (i + 1) % 25 == 0:
            log(f"  fetched {i + 1}/{len(picked)}")
    total = sum(im["size"] for im in images)
    manifest = {
        "id": set_id,
        "title": meta["title"],
        "description": meta["description"].format(n=len(images), mb=total / 1e6, bytes=total),
        "license": meta["license"],
        "attribution": meta["attribution"],
        "source": meta["source"],
        "revision": meta["revision"],
    }
    if meta.get("scored_labels"):
        manifest["scored_labels"] = meta["scored_labels"]
    manifest["images"] = images
    return manifest


def write_manifest(manifest: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{manifest['id']}.json"
    text = json.dumps(manifest, indent=1, ensure_ascii=False)
    # keep bbox arrays on one line for readable diffs
    text = re.sub(r"\[\s+([-0-9.e]+),\s+([-0-9.e]+),\s+([-0-9.e]+),\s+([-0-9.e]+)\s+\]", r"[\1, \2, \3, \4]", text)
    path.write_text(text + "\n", encoding="utf-8")
    log(f"  wrote {path} ({len(manifest['images'])} images)")
    return path


# --------------------------------------------------------------------------------------------
# BMD-45

BMD45_MAP = {
    "Hatchback": "car", "Sedan": "car", "SUV": "car", "MUV": "car", "Van": "car", "Tempo-traveller": "car",
    "Bus": "bus", "Mini-bus": "bus", "Truck": "truck",
    # LCV = light goods carriers (Tata Ace-style pickups, goods autos): COCO calls goods carriers "truck".
    "LCV": "truck",
    "Two-wheeler": "motorbike", "Bicycle": "bicycle",
}
# Ignored classes keep the nearest coco80 label so an engine matching ignore regions per class works.
BMD45_IGNORE = {"Three-wheeler": "car", "Other": "truck"}
BMD45_SCORED = ["car", "bus", "truck", "motorbike", "bicycle"]


def build_bmd45(work: Path, coco80: list[str], args) -> dict:
    d = work / "bmd45"
    ann = json.loads(cached(d / "ann.json", hf_resolve(BMD45_REPO, BMD45_REV, "BMD-45-Val/_annotations.coco.json")))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    unknown = set(cats.values()) - set(BMD45_MAP) - set(BMD45_IGNORE)
    if unknown:
        raise SystemExit(f"BMD-45: unmapped categories {unknown}")
    lfs: dict[str, dict] = {}
    for sub in ("images_000", "images_001", "images_002"):
        for e in hf_tree(BMD45_REPO, BMD45_REV, f"BMD-45-Val/{sub}", d / f"tree_{sub}.json"):
            if e.get("type") == "file" and e.get("lfs"):
                lfs[e["path"].removeprefix("BMD-45-Val/")] = e["lfs"]
    by_img: dict[int, list] = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    cands = []
    for im in ann["images"]:
        fn = im["file_name"]
        if fn not in lfs:
            continue
        w, h = im["width"], im["height"]
        objs = []
        for a in by_img.get(im["id"], []):
            name = cats[a["category_id"]]
            x, y, bw, bh = a["bbox"]
            box = clip_box(x, y, x + bw, y + bh, w, h)
            if box is None:
                continue
            if name in BMD45_MAP:
                objs.append({"label": BMD45_MAP[name], "bbox": box, "ignore": bool(a.get("iscrowd", 0))})
            else:
                objs.append({"label": BMD45_IGNORE[name], "bbox": box, "ignore": True})
        objs.sort(key=lambda o: (o["bbox"][1], o["bbox"][0], o["label"]))
        c = {
            "file": safe_name(fn),
            "url": hf_resolve(BMD45_REPO, BMD45_REV, f"BMD-45-Val/{fn}"),
            "width": w, "height": h, "objects": objs,
            "size_hint": lfs[fn]["size"], "sha_hint": lfs[fn]["oid"],
        }
        c["tags"] = structural_tags(c, set(BMD45_SCORED))
        cands.append(c)
    files = [c["file"] for c in cands]
    assert len(files) == len(set(files)), "BMD-45 file names are expected to be globally unique"
    log(f"  BMD-45 candidates: {len(cands)}")
    picked = select(cands, count=args.count, max_bytes=args.max_mb * 1e6, seed=args.seed,
                    bonus_tags={"small-objects": 1.0, "empty": 0.0}, scored=set(BMD45_SCORED), est_size=3.4e6)
    meta = {
        "title": "BMD-45 real CCTV (Bengaluru)",
        "kind_tag": "cctv",
        "description": (
            "{n} real CCTV frames from Bengaluru Police Safe City cameras (BMD-45 validation split, "
            "1920x1080 PNG, daytime 06:00-18:00, February 2025), {mb:.1f} MB ({bytes} bytes) in total. "
            "Vehicle-only ground truth: Hatchback/Sedan/SUV/MUV/Van/Tempo-traveller -> car, Bus/Mini-bus -> bus, "
            "Truck/LCV (light goods carrier) -> truck, Two-wheeler -> motorbike, Bicycle -> bicycle; Three-wheeler (auto-rickshaw) and Other are "
            "ignore regions. Two-wheeler and Bicycle boxes include the rider. There is no person ground truth, so only "
            "scored_labels are scored on this set. Resolution is uniform (res:fhd); the source has no per-image camera "
            "or timestamp metadata (file names are anonymised), so images are spread by seeded random sampling. "
            "Very distant vehicles are often unannotated."
        ),
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "attribution": (
            "BMD-45: Bengaluru Mobility Dataset, AI for Integrated Mobility (AIM) group, Indian Institute of Science "
            "(IISc), Bengaluru (Sharma, Mhatre, Gawali, Bokkasam, Kishore, Pattanaik, Rambha, Pinjari, Kovvali, "
            "Chakraborty, Rathore, Krishnapuram, Simmhan; CVPR Findings 2026). Imagery courtesy of Bengaluru Traffic "
            "Police / Bengaluru Police Safe City. Class mapping and subset selection by Blue Onyx Prism."
        ),
        "source": f"https://huggingface.co/datasets/{BMD45_REPO}",
        "revision": BMD45_REV,
        "scored_labels": BMD45_SCORED,
    }
    return finalise("bmd45-cctv", meta, picked, work, "luma")


# --------------------------------------------------------------------------------------------
# ExDark

EXDARK_NAMES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
EXDARK_MAP = {
    "Bicycle": "bicycle", "Boat": "boat", "Bottle": "bottle", "Bus": "bus", "Car": "car", "Cat": "cat",
    "Chair": "chair", "Cup": "cup", "Dog": "dog", "Motorbike": "motorbike", "People": "person",
    "Table": "diningtable",
}
EXDARK_SCORED = ["person", "bicycle", "car", "motorbike", "bus", "dog", "cat"]
EXDARK_INDOOR = {"bottle", "cup", "chair", "diningtable"}


def build_exdark(work: Path, coco80: list[str], args) -> dict:
    d = work / "exdark"
    yaml_txt = cached(d / "data.yaml", hf_resolve(EXDARK_REPO, EXDARK_REV, "data/data.yaml")).decode()
    m = re.search(r"names:\s*\[(.*?)\]", yaml_txt)
    names = [s.strip().strip("'\"") for s in m.group(1).split(",")] if m else []
    if names != EXDARK_NAMES:
        raise SystemExit(f"ExDark class list changed: {names}")
    tree = hf_tree(EXDARK_REPO, EXDARK_REV, "data/test/images", d / "tree_test_images.json")
    cands = []
    scored = set(EXDARK_SCORED)
    for e in sorted(tree, key=lambda e: e["path"]):
        if e.get("type") != "file" or not e["path"].lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        fn = safe_name(e["path"])
        stem = fn.rsplit(".", 1)[0]
        lbl_path = f"data/test/labels/{stem}.txt"
        try:
            txt = cached(d / "labels" / f"{stem}.txt", hf_resolve(EXDARK_REPO, EXDARK_REV, lbl_path)).decode()
        except urllib.error.HTTPError as err:
            if err.code == 404:
                continue
            raise
        w = h = 640  # Roboflow export: every image stretched to 640x640 (checked on download)
        objs = []
        for line in txt.splitlines():
            p = line.split()
            if not p:
                continue
            cls = int(p[0])
            vals = [float(v) for v in p[1:]]
            if len(vals) == 4:
                cx, cy, bw, bh = vals
                x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon -> enclosing box
                xs, ys = vals[0::2], vals[1::2]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                raise SystemExit(f"ExDark: bad label line in {lbl_path}: {line!r}")
            box = clip_box(x0 * w, y0 * h, x1 * w, y1 * h, w, h)
            if box is None:
                continue
            label = EXDARK_MAP[EXDARK_NAMES[cls]]
            objs.append({"label": label, "bbox": box, "ignore": label not in scored})
        objs.sort(key=lambda o: (o["bbox"][1], o["bbox"][0], o["label"]))
        labels = {o["label"] for o in objs}
        weight = 0.0
        weight -= 0.5 * len(labels & EXDARK_INDOOR)  # prefer outdoor / street content
        weight += 0.5 * bool(labels & {"car", "bus", "motorbike", "bicycle"})
        weight += 1.0 * bool(labels & {"dog", "cat"})  # pets are CCTV-relevant but rare in the split
        c = {
            "file": fn,
            "url": hf_resolve(EXDARK_REPO, EXDARK_REV, e["path"]),
            "width": w, "height": h, "objects": objs, "weight": weight,
            "size_hint": e.get("lfs", {}).get("size", e.get("size", 40000)),
            "sha_hint": e.get("lfs", {}).get("oid"),
        }
        c["tags"] = structural_tags(c, scored)
        cands.append(c)
    log(f"  ExDark candidates: {len(cands)}")
    picked = select(cands, count=args.count, max_bytes=args.max_mb * 1e6, seed=args.seed,
                    bonus_tags={"small-objects": 1.0, "crowd": 1.0}, scored=scored, est_size=4e4)
    meta = {
        "title": "ExDark low-light / night",
        "kind_tag": "general",
        "description": (
            "{n} low-light photos (very low light to twilight) from the ExDark test split, {mb:.1f} MB ({bytes} bytes) "
            "in total. Images were resized (stretched) to 640x640 by the Roboflow YOLO export, so resolution is uniform "
            "(res:sd) and aspect ratios are distorted. Scored labels: person (ExDark 'People'), bicycle, car, motorbike, "
            "bus, dog, cat; boat/bottle/chair/cup/diningtable boxes are ignore regions. Selection favours outdoor "
            "street, people, vehicle and animal scenes."
        ),
        "license": (
            "BSD-3-Clause (original ExDark, https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/LICENSE); "
            "authors ask that commercial use be cleared with Dr. Chee Seng Chan. Roboflow export labelled CC BY 4.0."
        ),
        "attribution": (
            "ExDark (Exclusively Dark Image Dataset), Yuen Peng Loh and Chee Seng Chan, Universiti Malaya: 'Getting to "
            "Know Low-light Images with The Exclusively Dark Dataset', CVIU 178 (2019) 30-42, "
            "doi:10.1016/j.cviu.2018.10.010. YOLO export by Roboflow Universe workspace my-workspace-ohnbt "
            "(exclusively-dark-image v2), redistributed by dronefreak/ExDark on Hugging Face. Subset selection and "
            "class mapping by Blue Onyx Prism."
        ),
        "source": f"https://huggingface.co/datasets/{EXDARK_REPO}",
        "revision": EXDARK_REV,
        "scored_labels": EXDARK_SCORED,
    }
    return finalise("exdark-night", meta, picked, work, "night")


# --------------------------------------------------------------------------------------------
# COCO

COCO_RENAME = {
    "motorcycle": "motorbike", "airplane": "aeroplane", "couch": "sofa", "potted plant": "pottedplant",
    "dining table": "diningtable", "tv": "tvmonitor",
}
COCO_EXCLUDE = {  # indoor close-ups, food and sports gear: not CCTV-like
    "bed", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "hair drier", "toothbrush", "scissors", "teddy bear", "sandwich",
    "hot dog", "pizza", "donut", "cake", "banana", "apple", "orange", "broccoli", "carrot", "bowl", "fork", "knife",
    "spoon", "wine glass", "cup", "skis", "snowboard", "surfboard", "tennis racket", "baseball bat",
    "baseball glove", "sports ball", "aeroplane", "giraffe", "zebra", "elephant", "bear", "train",
}
COCO_STREET = {"car", "bus", "truck", "motorbike", "bicycle", "traffic light", "stop sign", "fire hydrant",
               "parking meter", "bench"}
NIGHT_WORDS = re.compile(r"\b(night|nighttime|night-time|dark|darkness|evening|dusk|lit up|neon)\b", re.I)


def build_coco(work: Path, coco80: list[str], args) -> dict:
    d = work / "coco"
    zpath = d / "annotations_trainval2017.zip"
    cached(zpath, COCO_ANN_ZIP, COCO_ANN_ZIP_SHA256)
    with zipfile.ZipFile(zpath) as z:
        inst = json.loads(z.read("annotations/instances_val2017.json"))
        caps = json.loads(z.read("annotations/captions_val2017.json"))
    cats = {}
    for c in inst["categories"]:
        name = COCO_RENAME.get(c["name"], c["name"])
        if name not in coco80:
            raise SystemExit(f"COCO category {c['name']!r} -> {name!r} not in coco80")
        cats[c["id"]] = name
    night_hint = set()
    for c in caps["annotations"]:
        if NIGHT_WORDS.search(c["caption"]):
            night_hint.add(c["image_id"])
    by_img: dict[int, list] = {}
    for a in inst["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    focus = set(CCTV_FOCUS)
    cands = []
    for im in inst["images"]:
        anns = by_img.get(im["id"], [])
        w, h = im["width"], im["height"]
        objs = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            box = clip_box(x, y, x + bw, y + bh, w, h)
            if box is None:
                continue
            objs.append({"label": cats[a["category_id"]], "bbox": box, "ignore": bool(a["iscrowd"])})
        objs.sort(key=lambda o: (o["bbox"][1], o["bbox"][0], o["label"]))
        labels = {o["label"] for o in objs}
        if objs:
            if labels & COCO_EXCLUDE or not {o["label"] for o in objs if not o["ignore"]} & focus:
                continue
        weight = 1.0 * bool(labels & COCO_STREET) + 0.5 * bool(labels & {"dog", "cat", "horse", "bird"})
        fn = safe_name(im["file_name"])
        c = {
            "file": fn, "url": f"{COCO_BASE}/val2017/{fn}", "width": w, "height": h, "objects": objs,
            "weight": weight, "night_hint": im["id"] in night_hint,
        }
        c["tags"] = structural_tags(c, None)
        if c["night_hint"]:
            c["tags"].append("night-hint")
        cands.append(c)
    log(f"  COCO candidates: {len(cands)} ({sum(c['night_hint'] for c in cands)} with night captions)")
    picked = select(cands, count=args.count, max_bytes=args.max_mb * 1e6, seed=args.seed,
                    bonus_tags={"night-hint": 3.0, "small-objects": 1.0, "crowd": 1.5}, scored=None, est_size=1.7e5)
    for c in picked:
        c["tags"] = [t for t in c["tags"] if t != "night-hint"]
    meta = {
        "title": "COCO val2017 CCTV-relevant subset",
        "kind_tag": "general",
        "description": (
            "{n} COCO val2017 images with CCTV-relevant content (street scenes, people from single to crowds, "
            "vehicles, bicycles, motorbikes, animals, yards and parking), {mb:.1f} MB ({bytes} bytes) in total. "
            "All 80 COCO labels are scored (coco80 spellings); iscrowd regions are ignore. CCTV focus classes: "
            + ", ".join(CCTV_FOCUS) + ". COCO images are at most 640 px on the long side (res:sd). Day/night is "
            "estimated from mean luminance (< " + str(int(NIGHT_LUMA)) + "/255) or a night/dark caption. "
            "Empty images are val2017 images with no annotations."
        ),
        "license": (
            "Annotations: CC BY 4.0 (COCO Consortium). Images: individual Flickr licences as listed in "
            "instances_val2017.json (CC BY / BY-SA / BY-NC / BY-NC-SA / BY-ND / BY-NC-ND 2.0 and others); "
            "images are fetched from the COCO host, not redistributed."
        ),
        "attribution": (
            "Microsoft COCO: Common Objects in Context (Lin et al., ECCV 2014), https://cocodataset.org; images by "
            "their Flickr authors. Subset selection and label mapping by Blue Onyx Prism."
        ),
        "source": "https://cocodataset.org (val2017, annotations_trainval2017.zip)",
        "revision": f"val2017; annotations_trainval2017.zip sha256 {COCO_ANN_ZIP_SHA256}",
    }
    return finalise("coco-cctv", meta, picked, work, "luma")


# --------------------------------------------------------------------------------------------
# validation / stats


def validate(path: Path, coco80: list[str], work: Path | None, head: bool) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    for k in ("id", "title", "description", "license", "attribution", "source", "revision", "images"):
        assert k in m, f"{path.name}: missing {k}"
    names = set(coco80)
    if "scored_labels" in m:
        assert set(m["scored_labels"]) <= names, m["scored_labels"]
    files = set()
    errors = []
    for im in m["images"]:
        assert set(im) == {"file", "url", "sha256", "size", "width", "height", "tags", "objects"}, im.keys()
        assert im["file"] not in files, f"duplicate {im['file']}"
        files.add(im["file"])
        assert re.fullmatch(r"[0-9a-f]{64}", im["sha256"])
        for o in im["objects"]:
            assert set(o) == {"label", "bbox", "ignore"}
            assert o["label"] in names, o["label"]
            x0, y0, x1, y1 = o["bbox"]
            assert 0 <= x0 < x1 <= im["width"] and 0 <= y0 < y1 <= im["height"], (im["file"], o)
        if work is not None:
            p = work / m["id"] / "images" / im["file"]
            if p.exists():
                data = p.read_bytes()
                if hashlib.sha256(data).hexdigest() != im["sha256"] or len(data) != im["size"]:
                    errors.append(f"{im['file']}: local copy sha/size mismatch")
        if head:
            try:
                status, size = http_head(im["url"])
                if status != 200 or (size is not None and size != im["size"]):
                    errors.append(f"{im['file']}: HEAD {status} size {size} != {im['size']}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{im['file']}: HEAD failed {e}")
    if errors:
        raise SystemExit(f"{path.name}: " + "; ".join(errors[:10]))
    return m


def stats(m: dict) -> str:
    scored = set(m.get("scored_labels") or [])
    tags = Counter(t for im in m["images"] for t in im["tags"])
    lab = Counter(o["label"] for im in m["images"] for o in im["objects"] if not o["ignore"]
                  and (not scored or o["label"] in scored))
    ign = sum(1 for im in m["images"] for o in im["objects"] if o["ignore"] or (scored and o["label"] not in scored))
    areas = sorted((o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]) for im in m["images"]
                   for o in im["objects"] if not o["ignore"] and (not scored or o["label"] in scored))
    small = sum(1 for a in areas if a < 32 ** 2)
    medium = sum(1 for a in areas if 32 ** 2 <= a < 96 ** 2)
    large = len(areas) - small - medium
    total = sum(im["size"] for im in m["images"])
    dims = Counter(f"{im['width']}x{im['height']}" for im in m["images"])
    lines = [
        f"== {m['id']}: {len(m['images'])} images, {total / 1e6:.1f} MB",
        f"   tags: {dict(sorted(tags.items()))}",
        f"   scored objects: {sum(lab.values())} {dict(lab.most_common())}; ignored/unscored: {ign}",
        f"   object sizes (COCO area bins): small {small}, medium {medium}, large {large}",
        f"   image dims (top): {dims.most_common(5)}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="download/cache directory (untrusted data)")
    ap.add_argument("--set", choices=["all", *DEFAULTS], default="all")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--count", type=int, help="target image count (default per set)")
    ap.add_argument("--max-mb", type=float, help="byte budget in MB (default per set)")
    ap.add_argument("--validate", action="store_true", help="only validate existing manifests")
    ap.add_argument("--head", action="store_true", help="with --validate: HEAD every URL")
    args = ap.parse_args()
    coco80 = load_coco80()
    work = args.work.resolve()
    sets = list(DEFAULTS) if args.set == "all" else [args.set]
    builders = {"bmd45-cctv": build_bmd45, "exdark-night": build_exdark, "coco-cctv": build_coco}
    for s in sets:
        if not args.validate:
            log(f"building {s}")
            ns = argparse.Namespace(**vars(args))
            ns.count = args.count or DEFAULTS[s]["count"]
            ns.max_mb = args.max_mb or DEFAULTS[s]["max_mb"]
            write_manifest(builders[s](work, coco80, ns))
        m = validate(OUT_DIR / f"{s}.json", coco80, work, args.head)
        print(stats(m))


if __name__ == "__main__":
    main()
