"""
Geometry and Net Weight utilities for flexible tree annotations.
Supports:
- Flexible polygons and Oriented Bounding Boxes (OBB)
- Per-object pixel area and coverage weight
- Whole-image net_weight, tree_coverage_percentage, tree_count
- Canonical image filename normalization
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
from template.protocol import PerImageAnnotationItem


def extract_canopy_geometry(
    pil_img: Any,
    box_xyxy: List[float],
    hazard_class: str,
    confidence: float = 1.0,
    mask_xy: Optional[List[List[float]]] = None,
) -> PerImageAnnotationItem:
    """Extract flexible polygon/OBB contour, pixel area, and area weight."""
    x1, y1, x2, y2 = [float(c) for c in box_xyxy]
    img_w, img_h = pil_img.width, pil_img.height
    img_area = float(max(1, img_w * img_h))

    poly: Optional[List[List[float]]] = None
    poly_area: Optional[float] = None

    # 1. If explicit mask coordinates are provided
    if mask_xy is not None and len(mask_xy) >= 3:
        poly = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in mask_xy]
        try:
            import cv2
            import numpy as np
            poly_area = float(cv2.contourArea(np.array(poly, dtype=np.float32)))
        except Exception:
            pass

    # 2. Extract vegetation contour / oriented bounding box with OpenCV if available
    if poly is None:
        try:
            import cv2
            import numpy as np
            np_img = np.array(pil_img)
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(img_w, int(x2)), min(img_h, int(y2))
            if ix2 > ix1 and iy2 > iy1:
                crop = np_img[iy1:iy2, ix1:ix2]
                if crop.ndim == 3 and crop.shape[2] >= 3:
                    r = crop[:, :, 0].astype(np.float32)
                    g = crop[:, :, 1].astype(np.float32)
                    b = crop[:, :, 2].astype(np.float32)
                    exg = 2.0 * g - r - b
                    exg_norm = np.clip((exg - exg.min()) / (exg.max() - exg.min() + 1e-5) * 255.0, 0, 255).astype(np.uint8)
                    _, thresh = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    c_area = float(cv2.contourArea(cnt))
                    if c_area > 8:
                        rect = cv2.minAreaRect(cnt)
                        box_pts = cv2.boxPoints(rect)
                        poly = [[round(float(p[0] + ix1), 2), round(float(p[1] + iy1), 2)] for p in box_pts]
                        poly_area = c_area
        except Exception:
            pass

    # 3. Fallback: 4-point polygon matching bounding box
    if poly is None:
        poly = [
            [round(x1, 2), round(y1, 2)],
            [round(x2, 2), round(y1, 2)],
            [round(x2, 2), round(y2, 2)],
            [round(x1, 2), round(y2, 2)],
        ]
    if poly_area is None or poly_area <= 0:
        poly_area = float(max(0.0, (x2 - x1) * (y2 - y1)))

    obj_weight = round(poly_area / img_area, 6)
    return PerImageAnnotationItem(
        hazard_class=hazard_class,
        bounding_box=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        polygon=poly,
        area=round(poly_area, 2),
        weight=obj_weight,
        confidence=round(confidence, 4),
    )


def compute_image_net_metrics(
    annotations: List[PerImageAnnotationItem],
    image_width: int = 1024,
    image_height: int = 1024,
) -> Tuple[float, float, int]:
    """Compute (net_weight, tree_coverage_percentage, tree_count) for an image.
    
    net_weight: ratio in [0.0, 1.0] representing total canopy coverage.
    tree_coverage_percentage: percentage in [0.0, 100.0]
    tree_count: total detections
    """
    tree_count = len(annotations)
    if tree_count == 0:
        return 0.0, 0.0, 0

    img_area = float(max(1, image_width * image_height))
    total_area = 0.0
    for ann in annotations:
        if ann.area is not None and ann.area > 0:
            total_area += ann.area
        elif ann.bounding_box and len(ann.bounding_box) == 4:
            x1, y1, x2, y2 = ann.bounding_box
            total_area += max(0.0, (x2 - x1) * (y2 - y1))

    net_weight = round(min(1.0, max(0.0, total_area / img_area)), 6)
    tree_coverage_percentage = round(net_weight * 100.0, 2)
    return net_weight, tree_coverage_percentage, tree_count


def canonical_image_name(image_id: str, image_url: str = "") -> str:
    """Derive canonical filename (e.g. climate_raw_042.jpg) from image_id or image_url."""
    raw = image_id.split(":")[-1] if ":" in image_id else image_id
    raw = raw.split("/")[-1]
    if raw.startswith("raw_climate_"):
        raw = raw[4:]
    elif raw.startswith("golden_climate_"):
        raw = raw[7:]
    raw = raw.strip()
    if not any(raw.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        raw = f"{raw}.jpg"
    return raw

