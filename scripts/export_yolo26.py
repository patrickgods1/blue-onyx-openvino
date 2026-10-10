#!/usr/bin/env python
"""Export Ultralytics YOLO26 detection models to OpenVINO IR for Blue Onyx Prism.

Usage (inside the project venv: `.venv/Scripts/python.exe scripts/export_yolo26.py`):

    python scripts/export_yolo26.py                 # yolo26n, yolo26s, yolo26m -> models/
    python scripts/export_yolo26.py --sizes n s     # subset
    python scripts/export_yolo26.py --int8 --data coco128.yaml
    python scripts/export_yolo26.py --out C:/BlueOnyx/models

Produces `<name>.xml`, `<name>.bin`, `<name>.onnx` (for the ONNX Runtime devices, FP32, nms=False,
opset 17) and `<name>.yaml` (a `NAMES:` list) per model and prints the
config snippet to paste into `blue_onyx_prism_config.json`.

YOLO26 weights and the Ultralytics exporter are AGPL-3.0: export locally, do not commit weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", default=["n", "s", "m"], choices=list("nsmlx"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    p.add_argument("--int8", action="store_true", help="INT8 post-training quantization (needs --data)")
    p.add_argument("--data", default="coco128.yaml", help="dataset yaml used for INT8 calibration")
    p.add_argument("--fp32", action="store_true", help="keep FP32 weights instead of FP16")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset for the .onnx export")
    p.add_argument("--no-onnx", action="store_true", help="skip the <name>.onnx export")
    p.add_argument("--work", type=Path, default=Path(__file__).resolve().parent.parent / ".export_work")
    return p.parse_args()


def export_one(size: str, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    name = f"yolo26{size}"
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    # Ultralytics downloads `<name>.pt` and writes `<name>_openvino_model/` into the current
    # directory; run inside the work dir so the repo root stays clean and reruns are offline.
    os.chdir(args.work)
    weights = args.work / f"{name}.pt"
    model = YOLO(str(weights) if weights.exists() else f"{name}.pt")

    kwargs = dict(format="openvino", imgsz=args.imgsz, dynamic=False, nms=False, batch=1)
    if args.int8:
        kwargs.update(int8=True, data=args.data)
    elif not args.fp32:
        kwargs.update(half=True)
    # Ultralytics >= 8.4 replaced half/int8 with `quantize`; try the new spelling first.
    try:
        new_kwargs = {k: v for k, v in kwargs.items() if k not in ("half", "int8")}
        if args.int8:
            new_kwargs["quantize"] = 8
        elif not args.fp32:
            new_kwargs["quantize"] = 16
        out = model.export(**new_kwargs)
    except TypeError:
        out = model.export(**kwargs)
    return Path(out)


def export_onnx(size: str, args: argparse.Namespace) -> Path:
    """Export `<name>.onnx` (FP32, static 640x640, NMS-free [1,300,6] output) for ONNX Runtime."""
    from ultralytics import YOLO

    name = f"yolo26{size}"
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    os.chdir(args.work)
    weights = args.work / f"{name}.pt"
    model = YOLO(str(weights) if weights.exists() else f"{name}.pt")
    out = model.export(format="onnx", imgsz=args.imgsz, dynamic=False, nms=False, batch=1, opset=args.opset)
    return Path(out)


def output_shape(xml_path: Path) -> list[int]:
    root = ET.parse(xml_path).getroot()
    results = [l for l in root.iter("layer") if l.get("type") == "Result"]
    if not results:
        return []
    port = results[0].find("input/port")
    if port is None:
        return []
    return [int(d.text) for d in port.findall("dim")]


def names_from_metadata(export_dir: Path) -> list[str]:
    import yaml

    meta = export_dir / "metadata.yaml"
    if not meta.exists():
        raise SystemExit(f"metadata.yaml missing in {export_dir}")
    data = yaml.safe_load(meta.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def main() -> int:
    args = parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    snippets = []
    for size in args.sizes:
        name = f"yolo26{size}"
        export_dir = export_one(size, args)
        export_dir = export_dir if export_dir.is_dir() else export_dir.parent
        xml = next(export_dir.glob("*.xml"))
        binf = xml.with_suffix(".bin")
        shape = output_shape(xml)
        if len(shape) < 2 or shape[-1] != 6:
            raise SystemExit(
                f"{name}: unexpected output shape {shape}; expected [1, 300, 6] from an NMS-free export."
            )
        dst_xml = args.out / f"{name}.xml"
        shutil.copy2(xml, dst_xml)
        shutil.copy2(binf, args.out / f"{name}.bin")
        if not args.no_onnx:
            onnx = export_onnx(size, args)
            dst_onnx = args.out / f"{name}.onnx"
            shutil.copy2(onnx, dst_onnx)
            print(f"{name}: ONNX -> {dst_onnx}")
        names = names_from_metadata(export_dir)
        yaml_path = args.out / f"{name}.yaml"
        yaml_path.write_text("NAMES:\n" + "".join(f"  - {n}\n" for n in names), encoding="utf-8")
        print(f"{name}: output {shape}, {len(names)} classes -> {dst_xml}")
        snippets.append(
            {"name": name, "path": f"models/{name}.xml", "family": "yolo26", "classes": f"models/{name}.yaml"}
        )
    print("\nConfig snippet for blue_onyx_prism_config.json -> \"models\":")
    print(json.dumps(snippets, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
