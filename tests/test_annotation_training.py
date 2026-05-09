from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from template.hazard.r2_storage import upload_bytes_to_r2, upload_directory_to_r2
from template.miner.annotation_training import (
    AnnotationTrainingEngine,
    _annotate_deterministic,
    _severity_for_hazard_class,
    build_synthetic_labeled_png,
    fetch_url_bytes,
)
from template.miner.training import TrainingPipeline, TrainingSettings
from template.protocol import (
    AnnotationAndTrainingTask,
    AnnotationsFilePayload,
    ImageAnnotationDocument,
    LabeledTrainingImage,
    ModelCheckpoint,
    PerImageAnnotationItem,
    R2AccessCredentials,
    TrainingAnnotationLabel,
    UnlabeledAnnotationImage,
)


def test_training_annotation_label_rejects_bad_box():
    with pytest.raises(ValidationError):
        TrainingAnnotationLabel(
            hazard_class="x",
            bounding_box=[0, 0, 0, 10],
            severity="low",
        )


def test_severity_mapping():
    assert _severity_for_hazard_class("missing_hardhat") == "high"
    assert _severity_for_hazard_class("housekeeping_slip") == "medium"


def test_prepare_dataset_from_labeled_urls(tmp_path: Path):
    png = build_synthetic_labeled_png(80, 60)
    urls = {"u1": png}

    def fetch(url: str) -> bytes:
        return urls[url]

    labeled = [
        LabeledTrainingImage(
            image_url="u1",
            image_id="img_a",
            labels=[
                TrainingAnnotationLabel(
                    hazard_class="missing_hardhat",
                    bounding_box=[5, 5, 40, 50],
                    severity="high",
                    reasoning="worker without helmet",
                )
            ],
        ),
        LabeledTrainingImage(
            image_url="u1",
            image_id="img_b",
            labels=[
                TrainingAnnotationLabel(
                    hazard_class="trip_hazard",
                    bounding_box=[10, 10, 30, 30],
                    severity="medium",
                )
            ],
        ),
    ]
    pipeline = TrainingPipeline(
        TrainingSettings(
            workspace=tmp_path / "ws",
            private_dataset_root=None,
            enable_auto_hpo=False,
            autoresearch_max_iters=1,
            autoresearch_experiment_minutes=1,
            autoresearch_log_level="INFO",
            random_hpo_draw=False,
            hpo_seed=0,
        )
    )
    info = pipeline._prepare_dataset_from_labeled_urls(
        tmp_path / "ds",
        labeled,
        fetch,
        task_id="round-1",
    )
    assert "missing_hardhat" in info["class_names"]
    assert info["train_samples"] >= 1
    assert info["val_samples"] >= 1
    assert (info["yaml"]).is_file()


def test_fetch_file_url(tmp_path: Path):
    p = tmp_path / "x.png"
    p.write_bytes(build_synthetic_labeled_png())
    data = fetch_url_bytes(p.as_uri())
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_deterministic_annotation_document():
    from template.hazard.vector_db import OshaVectorDatabase

    png = build_synthetic_labeled_png(96, 96)
    doc = _annotate_deterministic(
        image_bytes=png,
        image_id="cs_000142",
        challenge_nonce="nonce-1",
        osha_db=OshaVectorDatabase.default(),
        model_version="m" * 16,
        miner_uid="miner_test_hk",
    )
    assert doc.image_id == "cs_000142"
    assert doc.miner_uid == "miner_test_hk"
    assert doc.model_version == "m" * 16
    assert isinstance(doc.annotations, list)


def test_annotations_file_payload_roundtrip():
    payload = AnnotationsFilePayload(
        task_id="t1",
        records=[
            ImageAnnotationDocument(
                image_id="i1",
                miner_uid="hk",
                timestamp="2026-05-01T12:00:00Z",
                annotations=[
                    PerImageAnnotationItem(
                        hazard_class="missing_fall_protection",
                        bounding_box=[245, 130, 412, 389],
                        severity="high",
                        confidence=0.94,
                        reasoning_chain="Worker near unprotected edge.",
                        osha_reference="1926.501(b)(1)",
                    )
                ],
                model_version="mv1" * 4,
            )
        ],
    )
    data = payload.model_dump()
    restored = AnnotationsFilePayload.model_validate(data)
    assert restored.records[0].annotations[0].hazard_class == "missing_fall_protection"


def test_annotation_training_engine_mocked_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    creds = R2AccessCredentials(
        account_id="abcd1234",
        bucket_name="testbucket",
        s3_endpoint="https://00000000000000000000000000000000.r2.cloudflarestorage.com",
        access_key_id="accesskey",
        secret_access_key="secretaccesssecret",
    )
    monkeypatch.setattr(
        "template.miner.annotation_training.load_r2_credentials_from_env",
        lambda: creds,
    )
    monkeypatch.setattr(
        "template.miner.annotation_training.upload_bytes_to_r2",
        lambda *a, **k: "r2://testbucket/miners/dual_flywheel/task-x/annotations.json",
    )
    monkeypatch.setattr(
        "template.miner.annotation_training.upload_directory_to_r2",
        lambda *a, **k: "r2://testbucket/miners/dual_flywheel/task-x/model_checkpoint/",
    )

    png = build_synthetic_labeled_png(64, 64)

    def fake_run_from_labeled_images(self, **kwargs):
        from template.protocol import TrainingManifest

        task_id = kwargs["task_id"]
        train_root = self.settings.workspace / task_id
        run_dir = train_root / "runs" / "yolov8s_construction"
        run_dir.mkdir(parents=True)
        (run_dir / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
        best = train_root / "best.pt"
        best.write_bytes(b"fake-weights")
        manifest = TrainingManifest(
            parent_model_hash="a" * 64,
            candidate_model_hash="b" * 64,
            candidate_model_uri="r2://testbucket/x/best.pt",
            config_hash="c" * 64,
            dataset_lineage_hash="d" * 64,
            recipe_uri=(train_root / "recipe.json").as_uri(),
            metrics={"uplift": 0.42},
        )
        (train_root / "recipe.json").write_text('{"ok": true}', encoding="utf-8")
        return manifest, best

    monkeypatch.setattr(
        TrainingPipeline,
        "run_from_labeled_images",
        fake_run_from_labeled_images,
    )

    class _Cfg:
        miner = MagicMock()
        miner.training_workspace = str(tmp_path / "train_ws")
        miner.private_dataset_root = ""
        miner.enable_auto_hpo = False
        miner.autoresearch = False
        miner.autoresearch_max_iters = 1
        miner.autoresearch_experiment_minutes = 1
        miner.autoresearch_log_level = "INFO"
        miner.annotation_backend = "deterministic"
        miner.dual_flywheel_r2_prefix = "miners/dual_flywheel"
        miner.random_hpo_draw = False
        miner.hpo_seed = 0

    engine = AnnotationTrainingEngine(config=_Cfg())
    engine._fetch_image = lambda _url: png

    base_hash = "f" * 64
    synapse = AnnotationAndTrainingTask(
        task_id="task-x",
        challenge_nonce="n1",
        training_images=[
            LabeledTrainingImage(
                image_url="http://example.invalid/lab.png",
                image_id="tr1",
                labels=[
                    TrainingAnnotationLabel(
                        hazard_class="missing_hardhat",
                        bounding_box=[2, 2, 40, 40],
                        severity="high",
                    )
                ],
            )
        ],
        annotation_images=[
            UnlabeledAnnotationImage(
                image_url="http://example.invalid/unl.png",
                image_id="cs_000142",
            )
        ],
        base_model_hash=base_hash,
        baseline_checkpoint=ModelCheckpoint(uri="yolov8s.pt", sha256=base_hash),
        max_training_seconds=30,
    )
    out = engine.run(synapse, miner_hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty")
    assert out.error_message is None
    assert out.annotations_uri.startswith("r2://")
    assert "model_checkpoint" in out.model_checkpoint_uri
    assert out.submitted_training_manifest is not None
    assert out.claim_improvement == pytest.approx(0.42)
    assert out.training_config.get("annotation_backend") == "deterministic"
    assert out.duration_ms is not None


def test_annotation_task_rejects_hash_mismatch():
    engine = AnnotationTrainingEngine(config=None)
    synapse = AnnotationAndTrainingTask(
        task_id="t",
        training_images=[
            LabeledTrainingImage(
                image_url="x",
                labels=[
                    TrainingAnnotationLabel(
                        hazard_class="h",
                        bounding_box=[1, 1, 10, 10],
                        severity="low",
                    )
                ],
            )
        ],
        annotation_images=[
            UnlabeledAnnotationImage(
                image_url="y",
                image_id="id1",
            )
        ],
        base_model_hash="a" * 64,
        baseline_checkpoint=ModelCheckpoint(uri="yolov8s.pt", sha256="b" * 64),
        max_training_seconds=10,
    )
    out = engine.run(synapse, miner_hotkey="hk")
    assert out.error_message is not None
    assert "base_model_hash" in out.error_message


def test_r2_upload_bytes_and_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    put_objects: list[tuple[str, bytes]] = []
    upload_files: list[tuple[str, str]] = []

    class FakeClient:
        def put_object(self, **kwargs):
            put_objects.append((kwargs["Key"], kwargs["Body"]))

        def upload_file(self, filename, bucket, key):
            upload_files.append((bucket, key))

    def fake_client(*a, **k):
        return FakeClient()

    monkeypatch.setattr("boto3.client", fake_client)
    creds = R2AccessCredentials(
        account_id="abcd1234",
        bucket_name="bk",
        s3_endpoint="https://ex.r2.cloudflarestorage.com",
        access_key_id="keyid",
        secret_access_key="longsecretvalue",
    )
    uri = upload_bytes_to_r2(b'{"a":1}', object_key="pref/a.json", creds=creds, content_type="application/json")
    assert uri == "r2://bk/pref/a.json"
    assert put_objects[0][0] == "pref/a.json"

    d = tmp_path / "m"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "w.pt").write_bytes(b"w")
    uri2 = upload_directory_to_r2(d, key_prefix="mprefix/", creds=creds)
    assert uri2.startswith("r2://bk/mprefix/")
    assert any("sub/w.pt" in k for _, k in upload_files)
