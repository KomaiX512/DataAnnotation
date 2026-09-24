import pytest
import numpy as np
from PIL import Image

from template.miner.geometry import (
    CARBON_WEIGHT_MULTIPLIERS,
    canonical_annotation_class,
    canonical_carbon_class,
    extract_canopy_geometry,
    compute_image_net_metrics,
)
from template.miner.ecological_reasoning import EcologicalVisionEngine
from template.hazard.annotation_eval import AnnotationFidelityScorer, _ReliabilityAccumulator
from template.hazard.image_corpus import GoldenAnnotation, GoldenImage
from template.protocol import PerImageAnnotationItem


def test_carbon_multipliers_and_canonical_classes():
    assert CARBON_WEIGHT_MULTIPLIERS["mangrove"] == 3.5
    assert CARBON_WEIGHT_MULTIPLIERS["dense_tree"] == 1.8
    assert CARBON_WEIGHT_MULTIPLIERS["ordinary_tree"] == 1.0
    assert CARBON_WEIGHT_MULTIPLIERS["field"] == 0.7
    assert CARBON_WEIGHT_MULTIPLIERS["plant"] == 0.4

    assert canonical_carbon_class("mangrove_tree") == "mangrove"
    assert canonical_carbon_class("wetland") == "mangrove"
    assert canonical_carbon_class("intact_forest") == "dense_tree"
    assert canonical_carbon_class("group_of_trees") == "dense_tree"
    assert canonical_carbon_class("individual_tree") == "ordinary_tree"
    assert canonical_carbon_class("cropland") == "field"
    assert canonical_carbon_class("agriculture") == "field"
    assert canonical_carbon_class("field") == "field"
    assert canonical_carbon_class("shrub") == "plant"
    assert canonical_carbon_class("regrowth") == "plant"


def test_taxonomy_aliases_are_exact_and_unknowns_stay_distinct():
    assert canonical_annotation_class("tree") == "ordinary_tree"
    assert canonical_annotation_class("individual_tree") == "ordinary_tree"
    assert canonical_annotation_class("fire_scar") == "fire_scar"
    assert canonical_annotation_class("deforestation") == "deforestation"
    assert canonical_annotation_class("water") == "water"
    assert canonical_annotation_class("urban") == "urban"
    assert canonical_annotation_class("unlisted_species") == "unlisted_species"


def test_fidelity_and_reliability_share_alias_normalization():
    from pathlib import Path

    golden = GoldenImage(
        image_id="alias-golden",
        image_path=Path("/dev/null"),
        image_url="",
        width=100,
        height=100,
        annotations=(
            GoldenAnnotation(
                hazard_class="individual_tree",
                bounding_box=(10, 10, 50, 50),
                severity="none",
            ),
        ),
    )
    item = PerImageAnnotationItem(
        hazard_class="tree", bounding_box=[10, 10, 50, 50]
    )
    fidelity = AnnotationFidelityScorer().score([item], golden)
    reliability = _ReliabilityAccumulator()
    reliability.update(9, [item], golden)
    weights, f1, *_ = reliability.finalize_uid(9)
    assert fidelity.fidelity > 0.9
    assert 0.0 < weights["ordinary_tree"] < 0.2
    assert f1["ordinary_tree"] == pytest.approx(1.0)


def test_extract_canopy_geometry_carbon_weight():
    img = Image.new("RGB", (1000, 1000), color=(20, 80, 20))
    # 100x100 box -> area = 10,000 px, ratio = 10,000 / 1,000,000 = 0.01
    box = [100.0, 100.0, 200.0, 200.0]

    # 1. Ordinary tree
    ann_tree = extract_canopy_geometry(img, box, "ordinary_tree")
    assert ann_tree.area == 10000.0
    assert pytest.approx(ann_tree.weight, rel=1e-3) == 0.01 * 1.0

    # 2. Mangrove (3.5x multiplier)
    ann_mangrove = extract_canopy_geometry(img, box, "mangrove")
    assert pytest.approx(ann_mangrove.weight, rel=1e-3) == 0.01 * 3.5

    # 3. Dense tree (1.8x multiplier)
    ann_dense = extract_canopy_geometry(img, box, "dense_tree")
    assert pytest.approx(ann_dense.weight, rel=1e-3) == 0.01 * 1.8

    # 4. Field (0.7x multiplier)
    ann_field = extract_canopy_geometry(img, box, "field")
    assert pytest.approx(ann_field.weight, rel=1e-3) == 0.01 * 0.7

    # 5. Plant (0.4x multiplier)
    ann_plant = extract_canopy_geometry(img, box, "plant")
    assert pytest.approx(ann_plant.weight, rel=1e-3) == 0.01 * 0.4


def test_compute_image_net_metrics_carbon_sum():
    # 2 mangroves (0.01 * 3.5 = 0.035 each) + 1 ordinary tree (0.02 * 1.0 = 0.02)
    anns = [
        PerImageAnnotationItem(hazard_class="mangrove", bounding_box=[0, 0, 100, 100], area=10000, weight=0.035),
        PerImageAnnotationItem(hazard_class="mangrove", bounding_box=[100, 0, 200, 100], area=10000, weight=0.035),
        PerImageAnnotationItem(hazard_class="ordinary_tree", bounding_box=[200, 0, 300, 200], area=20000, weight=0.020),
    ]
    net_wt, cov_pct, cnt = compute_image_net_metrics(anns, image_width=1000, image_height=1000)
    assert cnt == 3
    assert pytest.approx(net_wt, rel=1e-3) == 0.035 + 0.035 + 0.020
    # Total physical area = 40,000 / 1,000,000 = 4.0%
    assert cov_pct == 4.0


def test_ecological_vision_engine_mock():
    # Test engine reasoning on a synthetic RGB image
    engine = EcologicalVisionEngine(checkpoint_path="models/tree_detection.pt")
    # Synthetic image with green vegetation
    img_data = np.zeros((256, 256, 3), dtype=np.uint8)
    img_data[20:100, 20:100] = [30, 160, 40]  # Green patch
    img = Image.fromarray(img_data)

    anns, metrics = engine.reason_and_annotate(img, image_id="test_chip_01")
    assert isinstance(anns, list)
    assert "net_weight" in metrics
    assert "tree_coverage_percentage" in metrics
    assert "dominant_class" in metrics
    assert "class_breakdown" in metrics


def test_annotation_fidelity_scorer_carbon_cross_verification():
    scorer = AnnotationFidelityScorer()
    from pathlib import Path
    golden = GoldenImage(
        image_id="golden_01",
        image_path=Path("/tmp/golden_01.jpg"),
        image_url="file:///tmp/golden_01.jpg",
        width=1000,
        height=1000,
        annotations=(
            GoldenAnnotation(hazard_class="mangrove", bounding_box=(100, 100, 200, 200), severity="none"),
        ),
    )
    # Miner with matching mangrove
    miner_items = [
        PerImageAnnotationItem(
            hazard_class="mangrove",
            bounding_box=[100.0, 100.0, 200.0, 200.0],
            area=10000.0,
            weight=0.035,
        )
    ]
    res = scorer.score(miner_items, golden)
    assert res.iou == 1.0
    assert res.class_severity == 1.0
    assert pytest.approx(res.net_weight_agreement, rel=1e-2) == 1.0
    assert res.fidelity > 0.95
