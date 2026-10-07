"""Test SegFormer MIT-B2 Frontier Vision Backend for miners."""

from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import pytest

from template.miner.backends.factory import get_backend
from template.miner.backends.base import InferImage
from template.hazard.geometry_quality import audit_annotation_geometry


def test_segformer_backend_inference(tmp_path):
    golden_img = Path("data/climate_mrv/samples/golden/tcd_forestry_golden_015.jpg")
    if not golden_img.exists():
        pytest.skip("Golden sample image not found")
    img_path = golden_img

    config = SimpleNamespace(
        miner=SimpleNamespace(
            model_backend="segformer",
            segformer_model_path="models/tcd-segformer-mit-b2",
            segformer_device="cpu",
            segformer_infer_res=256,
            segformer_min_area=20,
            segformer_peak_min_dist=10,
        )
    )

    backend = get_backend("segformer", config)
    results = backend.infer([InferImage(image_id="test_chip_1", image_path=img_path)])

    assert "test_chip_1" in results
    annos = results["test_chip_1"]
    assert len(annos) > 0

    # Ensure no crude boxes and no spikes
    report = audit_annotation_geometry(annos)
    assert report.box_count == 0
    assert report.spiky_count == 0
    assert report.is_acceptable is True

    # Ensure all polygons have at least 4 vertices (smooth organic)
    for a in annos:
        assert len(a.polygon) >= 4
        assert a.area > 0
        assert a.weight > 0
