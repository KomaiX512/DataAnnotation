"""Localnet simulation helpers for annotation quality tiers (no fallback in prod path)."""

from __future__ import annotations

import random
from typing import List, Sequence

from template.protocol import PerImageAnnotationItem, SeverityTier


def random_annotations(
    *,
    width: int,
    height: int,
    rng: random.Random,
    count: int | None = None,
) -> List[PerImageAnnotationItem]:
    """Deliberately poor boxes for acceptance testing (``annotation_backend=random``)."""

    n = count if count is not None else max(1, rng.randint(1, 3))
    items: List[PerImageAnnotationItem] = []
    for _ in range(n):
        x1 = rng.randint(0, max(1, width // 2))
        y1 = rng.randint(0, max(1, height // 2))
        x2 = min(width, x1 + rng.randint(10, max(11, width // 3)))
        y2 = min(height, y1 + rng.randint(10, max(11, height // 3)))
        items.append(
            PerImageAnnotationItem(
                hazard_class="random_object",
                bounding_box=[float(x1), float(y1), float(x2), float(y2)],
                severity="none",
            )
        )
    return items


def perturb_annotations(
    items: Sequence[PerImageAnnotationItem],
    *,
    width: int,
    height: int,
    rng: random.Random,
    noise_px: int = 8,
) -> List[PerImageAnnotationItem]:
    """Shift boxes and occasionally flip class (``annotation_backend=yolo_medium``)."""

    out: List[PerImageAnnotationItem] = []
    for it in items:
        x1, y1, x2, y2 = [int(v) for v in it.bounding_box]
        dx = rng.randint(-noise_px, noise_px)
        dy = rng.randint(-noise_px, noise_px)
        nx1 = max(0, min(width - 2, x1 + dx))
        ny1 = max(0, min(height - 2, y1 + dy))
        nx2 = max(nx1 + 2, min(width, x2 + dx))
        ny2 = max(ny1 + 2, min(height, y2 + dy))
        cls = it.hazard_class
        if rng.random() < 0.15:
            cls = "ambiguous_object"
        out.append(
            PerImageAnnotationItem(
                hazard_class=cls,
                bounding_box=[float(nx1), float(ny1), float(nx2), float(ny2)],
                severity=it.severity,
            )
        )
    if rng.random() < 0.12:
        out.append(
            PerImageAnnotationItem(
                hazard_class="spurious_detection",
                bounding_box=[5.0, 5.0, 25.0, 25.0],
                severity="low",
            )
        )
    return out
