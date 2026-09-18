"""
Tests for flexible tree geometry (polygons, OBB), net weight metrics,
canonical image naming, and commercial dataset export schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw

from template.miner.geometry import (
    canonical_image_name,
    compute_image_net_metrics,
    extract_canopy_geometry,
)
from template.protocol import ImageAnnotationDocument, PerImageAnnotationItem
from template.hazard.annotation_eval import AnnotationFidelityScorer, FidelityComponents
from template.hazard.image_corpus import GoldenAnnotation, GoldenImage


def test_canonical_image_name():
    assert canonical_image_name("climate_mrv_fallback:climate_raw_042") == "climate_raw_042.jpg"
    assert canonical_image_name("raw_climate_raw_123.jpg") == "climate_raw_123.jpg"
    assert canonical_image_name("golden_climate_tree_001.png") == "climate_tree_001.png"
    assert canonical_image_name("climate_raw_499") == "climate_raw_499.jpg"


def test_extract_canopy_geometry_fallback_and_area():
    img = Image.new("RGB", (1024, 1024), color=(30, 80, 30))
    box = [100.0, 100.0, 300.0, 300.0]
    ann = extract_canopy_geometry(img, box, "individual_tree", confidence=0.92)

    assert ann.hazard_class == "individual_tree"
    assert ann.bounding_box == [100.0, 100.0, 300.0, 300.0]
    assert ann.polygon is not None
    assert len(ann.polygon) == 4
    assert ann.area is not None
    assert ann.area > 0
    assert ann.weight is not None
    assert ann.weight == round(ann.area / (1024 * 1024), 6)
    assert ann.confidence == 0.92


def test_compute_image_net_metrics():
    ann1 = PerImageAnnotationItem(
        hazard_class="individual_tree",
        bounding_box=[0.0, 0.0, 200.0, 200.0],
        area=40000.0,
        weight=40000.0 / 1000000.0,
    )
    ann2 = PerImageAnnotationItem(
        hazard_class="group_of_trees",
        bounding_box=[200.0, 200.0, 400.0, 400.0],
        area=40000.0,
        weight=40000.0 / 1000000.0,
    )
    net_weight, coverage_pct, tree_cnt = compute_image_net_metrics(
        [ann1, ann2], image_width=1000, image_height=1000
    )
    assert tree_cnt == 2
    assert net_weight == 0.08  # 80000 / 1000000 = 0.08
    assert coverage_pct == 8.0


def test_image_annotation_document_schema():
    doc = ImageAnnotationDocument(
        image_id="climate_raw_042",
        image_url="http://example.com/test.jpg",
        model_version="test_v1.0",
        miner_uid="5CtC...",
        timestamp="2026-09-18T12:00:00Z",
        annotations=[
            PerImageAnnotationItem(
                hazard_class="individual_tree",
                bounding_box=[10.0, 10.0, 50.0, 50.0],
                polygon=[[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]],
                area=1600.0,
                weight=0.0016,
                confidence=0.88,
            )
        ],
        image_name="climate_raw_042.jpg",
        net_weight=0.0016,
        tree_coverage_ratio=0.0016,
        tree_coverage_percentage=0.16,
        tree_count=1,
    )
    data = json.loads(doc.model_dump_json())
    assert data["image_name"] == "climate_raw_042.jpg"
    assert data["net_weight"] == 0.0016
    assert data["tree_count"] == 1
    assert data["annotations"][0]["polygon"] is not None


def test_fidelity_scorer_with_net_weight():
    scorer = AnnotationFidelityScorer()
    golden = GoldenImage(
        image_id="test_golden_01",
        image_path=Path("/dev/null"),
        image_url="http://example.com/g.jpg",
        width=1000,
        height=1000,
        annotations=(
            GoldenAnnotation(
                hazard_class="individual_tree",
                bounding_box=(100, 100, 200, 200),
                severity="medium",
            ),
        ),
    )
    miner_items = [
        PerImageAnnotationItem(
            hazard_class="individual_tree",
            bounding_box=[100.0, 100.0, 200.0, 200.0],
            area=10000.0,
            weight=0.01,
        )
    ]
    res = scorer.score(miner_items, golden)
    assert res.fidelity > 0.8
    assert res.net_weight_agreement == 1.0
    assert res.gt_net_weight == 0.01
    assert res.miner_net_weight == 0.01
