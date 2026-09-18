from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from template.protocol import ImageAnnotationDocument, PerImageAnnotationItem


def annotate_image_detector_only(
    *,
    checkpoint: Path,
    image_bytes: bytes,
    image_id: str,
    image_url: str,
    model_version: str,
    miner_uid: str,
) -> ImageAnnotationDocument:
    """Run YOLO-only detector on the provided image and return annotations."""
    try:
        from PIL import Image
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "pillow and ultralytics are required to run YOLO annotations."
        ) from exc

    # Load YOLO model
    model = YOLO(str(checkpoint))

    # Read image
    img = Image.open(io.BytesIO(image_bytes))

    # Run inference
    results = model(img, verbose=False)

    annotations: List[PerImageAnnotationItem] = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None:
            from template.miner.geometry import extract_canopy_geometry
            for b_idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                cls_idx = int(box.cls[0].item())
                cls_name = model.names[cls_idx]
                conf = float(box.conf[0].item()) if hasattr(box, "conf") and box.conf is not None else 1.0
                mask_xy = None
                if hasattr(result, "masks") and result.masks is not None and len(result.masks) > b_idx:
                    mask_xy = result.masks.xy[b_idx].tolist()
                ann_item = extract_canopy_geometry(
                    pil_img=img,
                    box_xyxy=xyxy,
                    hazard_class=cls_name,
                    confidence=conf,
                    mask_xy=mask_xy,
                )
                annotations.append(ann_item)

    from template.miner.geometry import compute_image_net_metrics, canonical_image_name
    img_w, img_h = img.size if hasattr(img, "size") else (1024, 1024)
    net_weight, coverage_pct, tree_count = compute_image_net_metrics(annotations, img_w, img_h)
    canonical_name = canonical_image_name(image_id, image_url)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ImageAnnotationDocument(
        image_id=image_id,
        image_url=image_url,
        miner_uid=miner_uid,
        timestamp=ts,
        annotations=annotations,
        model_version=model_version,
        image_name=canonical_name,
        net_weight=net_weight,
        tree_coverage_ratio=net_weight,
        tree_coverage_percentage=coverage_pct,
        tree_count=tree_count,
    )

