from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from PIL import Image

from template.miner.vlm_client import VlmClient
from template.protocol import ImageAnnotationDocument, PerImageAnnotationItem, SeverityTier


def _expand_box_xyxy(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int, pad_frac: float = 0.08
) -> tuple[int, int, int, int]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_frac)
    pad_y = int(bh * pad_frac)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w, x2 + pad_x)
    ny2 = min(h, y2 + pad_y)
    if nx2 <= nx1 or ny2 <= ny1:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


_SEVERITY_ORDER = ("none", "low", "medium", "high", "critical")


def _parse_severity_tier(raw: object) -> SeverityTier:
    s = str(raw or "").strip().lower()
    if s not in _SEVERITY_ORDER:
        raise ValueError(
            f"VLM returned invalid severity tier {raw!r}; expected one of {_SEVERITY_ORDER}."
        )
    return s  # type: ignore[return-value]


def annotate_image_two_stage(
    *,
    checkpoint: Path,
    image_bytes: bytes,
    image_id: str,
    model_version: str,
    miner_uid: str,
    vlm: VlmClient,
) -> ImageAnnotationDocument:
    """
    Stage 1: YOLO detector (fine-tuned ``best.pt``).
    Stage 2: optional class/severity refinement per detection via the configured
    structured-output backend. No miner-supplied confidence field is emitted.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for two-stage annotation.") from exc

    model = YOLO(str(checkpoint))
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    results = model.predict(source=image, verbose=False)
    items: List[PerImageAnnotationItem] = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        clss = r.boxes.cls.cpu().tolist()
        for box, det_conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            raw_name = model.names.get(int(cls_id), str(int(cls_id)))
            hazard_class = str(raw_name).lower().replace(" ", "_")

            cx1, cy1, cx2, cy2 = _expand_box_xyxy(x1, y1, x2, y2, w, h)
            crop = image.crop((cx1, cy1, cx2, cy2))
            vlm_out = vlm.complete_safety_json(
                crop=crop,
                full_size=(w, h),
                hazard_class=hazard_class,
                detector_confidence=float(det_conf),
            )
            refined_class = str(vlm_out.get("hazard_class", hazard_class)).strip().lower()
            if refined_class:
                hazard_class = refined_class.replace(" ", "_")
            severity = _parse_severity_tier(vlm_out.get("severity"))

            items.append(
                PerImageAnnotationItem(
                    hazard_class=hazard_class,
                    bounding_box=[x1, y1, x2, y2],
                    severity=severity,
                )
            )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ImageAnnotationDocument(
        image_id=image_id,
        miner_uid=miner_uid,
        timestamp=ts,
        annotations=items,
        model_version=model_version,
    )


def annotate_image_detector_only(
    *,
    checkpoint: Path,
    image_bytes: bytes,
    image_id: str,
    model_version: str,
    miner_uid: str,
) -> ImageAnnotationDocument:
    """YOLO detector only (no VLM). Used for COCO localnet when VLM is not configured."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for detector-only annotation.") from exc

    model = YOLO(str(checkpoint))
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    results = model.predict(source=image, verbose=False)
    items: List[PerImageAnnotationItem] = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().tolist()
        clss = r.boxes.cls.cpu().tolist()
        for box, cls_id in zip(xyxy, clss):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            raw_name = model.names.get(int(cls_id), str(int(cls_id)))
            hazard_class = str(raw_name).lower().replace(" ", "_")
            items.append(
                PerImageAnnotationItem(
                    hazard_class=hazard_class,
                    bounding_box=[x1, y1, x2, y2],
                    severity="none",
                )
            )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ImageAnnotationDocument(
        image_id=image_id,
        miner_uid=miner_uid,
        timestamp=ts,
        annotations=items,
        model_version=model_version,
    )
