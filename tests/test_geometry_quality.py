"""Unit tests for geometry quality and anti-exploit verification."""

import math
import numpy as np
import pytest

from template.hazard.geometry_quality import (
    is_crude_box,
    is_spiky_polygon,
    isoperimetric_quotient,
    convexity_ratio,
    audit_annotation_geometry,
    raster_polygon_iou,
    verify_crop_vegetation,
)
from template.protocol import PerImageAnnotationItem


def test_crude_box_detection():
    # 4-point rectangle box
    box = [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]
    bbox = [100.0, 100.0, 200.0, 200.0]
    assert is_crude_box(box, bbox) is True

    # 8-point organic circle/canopy
    angles = [i * 2.0 * math.pi / 8.0 for i in range(8)]
    circle = [[150.0 + 45.0 * math.cos(a), 150.0 + 45.0 * math.sin(a)] for a in angles]
    assert is_crude_box(circle, bbox) is False


def test_isoperimetric_quotient_and_spikiness():
    # Regular rounded canopy
    angles = [i * 2.0 * math.pi / 16.0 for i in range(16)]
    canopy = [[200.0 + 50.0 * math.cos(a), 200.0 + 50.0 * math.sin(a)] for a in angles]
    q_canopy = isoperimetric_quotient(canopy)
    assert q_canopy > 0.70
    assert is_spiky_polygon(canopy) is False

    # Lightning-bolt / starburst spike
    spiky = [
        [100, 100], [150, 105], [105, 120], [180, 125], [110, 140],
        [190, 145], [115, 160], [200, 165], [120, 180], [100, 200],
        [80, 150], [95, 130], [70, 120], [90, 110]
    ]
    q_spiky = isoperimetric_quotient(spiky)
    assert q_spiky < 0.15
    assert is_spiky_polygon(spiky) is True


def test_audit_penalizes_excessive_boxes():
    # Miner submitting 10 crude boxes
    items = [
        PerImageAnnotationItem(
            hazard_class="ordinary_tree",
            bounding_box=[i * 20.0, i * 20.0, (i + 1) * 20.0, (i + 1) * 20.0],
            polygon=[
                [i * 20.0, i * 20.0],
                [(i + 1) * 20.0, i * 20.0],
                [(i + 1) * 20.0, (i + 1) * 20.0],
                [i * 20.0, (i + 1) * 20.0],
            ],
            confidence=0.9,
            area=400.0,
        )
        for i in range(10)
    ]
    audit = audit_annotation_geometry(items)
    assert audit.box_ratio == 1.0
    assert audit.quality_multiplier == 0.0
    assert audit.is_acceptable is False
    assert "excessive_box_polygons" in audit.rejection_reason


def test_audit_penalizes_spiky_polygons():
    spiky = [
        [100, 100], [150, 105], [105, 120], [180, 125], [110, 140],
        [190, 145], [115, 160], [200, 165], [120, 180], [100, 200],
        [80, 150], [95, 130], [70, 120], [90, 110]
    ]
    items = [
        PerImageAnnotationItem(
            hazard_class="ordinary_tree",
            bounding_box=[70, 100, 200, 200],
            polygon=spiky,
            confidence=0.9,
            area=5025.0,
        )
        for _ in range(5)
    ]
    audit = audit_annotation_geometry(items)
    assert audit.spiky_ratio == 1.0
    assert audit.quality_multiplier == 0.0
    assert audit.is_acceptable is False
    assert "spiky_erratic_polygons" in audit.rejection_reason


def test_vegetation_spectral_filter():
    # Pure ocean water (B=200, G=100, R=50) -> ExG = 200 - 50 - 200 = -50 <= 0
    ocean_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    ocean_crop[:, :, 0] = 50   # R
    ocean_crop[:, :, 1] = 100  # G
    ocean_crop[:, :, 2] = 200  # B
    assert verify_crop_vegetation(ocean_crop) is False

    # Healthy tree foliage (R=40, G=180, B=50) -> ExG = 360 - 40 - 50 = +270 > 0
    tree_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    tree_crop[:, :, 0] = 40
    tree_crop[:, :, 1] = 180
    tree_crop[:, :, 2] = 50
    assert verify_crop_vegetation(tree_crop) is True


def test_raster_polygon_iou_penalizes_box_vs_circle():
    angles = [i * 2.0 * math.pi / 16.0 for i in range(16)]
    circle = [[150.0 + 50.0 * math.cos(a), 150.0 + 50.0 * math.sin(a)] for a in angles]
    box = [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]
    bbox = [100.0, 100.0, 200.0, 200.0]

    # Perfect identical circle
    iou_identical = raster_polygon_iou(circle, circle, bbox, bbox)
    assert iou_identical > 0.98

    # Box vs Circle (box includes non-vegetated corners)
    iou_box = raster_polygon_iou(box, circle, bbox, bbox)
    # Area of circle is pi*r^2 = 7854, area of box is 10000 -> IoU ~ 7854 / 10000 = 0.785
    assert iou_box < 0.82
