#!/usr/bin/env python3
"""
Run inside a Docker container with ``--network none`` to execute YOLO inference
on validator-supplied images without loading untrusted weights on the host.

The validator prepares a job directory containing:

  - ``spec.json`` — see ``_SPEC_SCHEMA`` below
  - image files referenced by ``relpath``
  - ``model.pt`` — miner checkpoint (copied in by the host)

Writes ``predictions.json`` with raw box outputs for host-side scoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: flywheel_yolo_docker_worker.py /job", file=sys.stderr)
        return 2
    job = Path(sys.argv[1]).resolve()
    spec_path = job / "spec.json"
    if not spec_path.is_file():
        print(f"missing {spec_path}", file=sys.stderr)
        return 1
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    model_path = job / spec["model_relpath"]
    if not model_path.is_file():
        print(f"missing model {model_path}", file=sys.stderr)
        return 1

    from ultralytics import YOLO  # noqa: PLC0415 — loaded only inside the sandbox image
    from PIL import Image  # noqa: PLC0415

    model = YOLO(str(model_path))
    names = model.names
    out: dict = {"images": {}}

    for entry in spec["images"]:
        key = entry["key"]
        rel = entry["relpath"]
        img_path = job / rel
        if not img_path.is_file():
            raise FileNotFoundError(f"missing image {img_path}")
        with Image.open(img_path) as pil_img:
            pil = pil_img.convert("RGB")
        preds = model.predict(source=pil, verbose=False)
        if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0:
            out["images"][key] = {"xyxy": [], "classes": [], "confs": []}
            continue
        result = preds[0]
        xyxy = result.boxes.xyxy.cpu().tolist()
        clss = result.boxes.cls.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        classes: list[str] = []
        for cls_idx in clss:
            try:
                cls_int = int(cls_idx)
            except Exception:
                cls_int = 0
            raw_name = names.get(cls_int, str(cls_int)) if isinstance(names, dict) else (
                names[cls_int] if 0 <= cls_int < len(names) else str(cls_int)
            )
            classes.append(str(raw_name).lower().replace(" ", "_"))
        out["images"][key] = {
            "xyxy": [[float(v) for v in box] for box in xyxy],
            "classes": classes,
            "confs": [float(c) for c in confs],
        }

    (job / "predictions.json").write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
