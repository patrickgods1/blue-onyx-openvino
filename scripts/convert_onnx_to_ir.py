#!/usr/bin/env python
"""Convert an ONNX model to OpenVINO IR (`.xml` + `.bin`) for Blue Onyx Prism.

The service reads ONNX directly through OpenVINO's ONNX frontend. Use this fallback when a model
fails to load that way (an unsupported op, or a dynamic shape the frontend cannot resolve), or to
store FP16 weights for a smaller file and faster GPU compile.

Usage (inside the project venv, see scripts/requirements-export.txt):

    python scripts/convert_onnx_to_ir.py models/rt-detrv2-s.onnx
    python scripts/convert_onnx_to_ir.py models/IPcam-general.onnx --fp32 --out C:/BlueOnyx/models

Writes `<stem>.xml` and `<stem>.bin` next to the ONNX file (or into `--out`), copies the `NAMES:`
class yaml if present, and prints the config entry to use. The `family` stays the same as for the
ONNX model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("onnx", type=Path, help="path to the .onnx model")
    p.add_argument("--out", type=Path, help="output directory (default: next to the ONNX file)")
    p.add_argument("--fp32", action="store_true", help="keep FP32 weights instead of compressing to FP16")
    p.add_argument(
        "--input-shape",
        default="",
        help="static shape for the image input, e.g. 1,3,640,640 (default: keep the model's shape)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import openvino as ov

    src = args.onnx.resolve()
    if not src.is_file():
        raise SystemExit(f"{src} does not exist")
    out_dir = (args.out or src.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ov.convert_model(str(src))
    # The ONNX input name can end up as an alias behind an internal name (RT-DETR's
    # `orig_target_sizes` becomes `/postprocessor/Expand_output_0`); keep the public name only.
    for inp in model.inputs:
        public = sorted(n for n in inp.get_names() if not n.startswith("/"))
        if public:
            inp.get_tensor().set_names({public[0]})
    shapes = {}
    for i, inp in enumerate(model.inputs):
        ps = inp.get_partial_shape()
        if i == 0 and args.input_shape:
            shapes[inp.any_name] = ov.PartialShape([int(d) for d in args.input_shape.split(",")])
        elif ps.rank.is_static and ps.rank.get_length() > 0 and ps[0].is_dynamic:
            # Static batch of 1 (the service always sends one image).
            shapes[inp.any_name] = ov.PartialShape([1] + [ps[d] for d in range(1, ps.rank.get_length())])
    if shapes:
        model.reshape(shapes)

    dst = out_dir / f"{src.stem}.xml"
    ov.save_model(model, str(dst), compress_to_fp16=not args.fp32)

    for inp in model.inputs:
        print(f"input  {inp.any_name}: {inp.partial_shape} {inp.element_type}")
    for outp in model.outputs:
        print(f"output {outp.any_name}: {outp.partial_shape} {outp.element_type}")

    classes = src.with_suffix(".yaml")
    entry = {"name": src.stem, "path": f"models/{dst.name}", "family": "auto"}
    if classes.exists():
        dst_classes = out_dir / classes.name
        if dst_classes.resolve() != classes:
            shutil.copy2(classes, dst_classes)
        entry["classes"] = f"models/{classes.name}"
    print(f"\nwrote {dst} and {dst.with_suffix('.bin')}")
    print('Config entry for blue_onyx_prism_config.json -> "models" (set "family" explicitly if known):')
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
