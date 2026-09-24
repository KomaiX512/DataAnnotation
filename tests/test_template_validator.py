from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from template.hazard.dataset_assembler import AdoptionLedger
from template.hazard.annotation_eval import PerMinerAnnotationScore
from template.hazard.annotation_eval import _ReliabilityAccumulator
from template.hazard.dual_reward import DualFlywheelBreakdown
from template.hazard.incentives import broad_softmax_scores
from template.protocol import (
    AnnotationTask,
    AnnotationsFilePayload,
    ImageAnnotationDocument,
    PerImageAnnotationItem,
    R2AccessCredentials,
)
from template.validator.dual_forward import (
    _decay_empty_validator_round,
    _download_miner_artifact_bytes,
    _parse_annotations_payload,
    _validate_response_shape,
)
from template.hazard.r2_storage import download_bytes_from_r2
from neurons.validator import Validator


def test_broad_softmax_pays_multiple_value_adding_miners():
    shaped = broad_softmax_scores(
        np.array([0.0, 0.2, 0.4, 0.8], dtype=float),
        temperature=0.35,
        floor=0.01,
        min_score=0.05,
    )
    assert shaped[0] == 0.0
    assert (shaped[1:] > 0.0).all()
    assert abs(float(shaped.sum()) - 1.0) < 1e-6


def test_empty_invalid_round_decays_incentive_and_reliability_state():
    class Reliability:
        def __init__(self):
            self.decays = 0

        def decay_all(self):
            self.decays += 1

    state = SimpleNamespace(
        scores=np.array([0.5, 0.2], dtype=np.float32),
        annotation_scores=np.array([0.4, 0.1], dtype=np.float32),
        adoption_bonus_scores=np.array([0.2, 0.05], dtype=np.float32),
        config=SimpleNamespace(neuron=SimpleNamespace(moving_average_alpha=0.1)),
        reliability=Reliability(),
        dataset_assembler=SimpleNamespace(
            ledger=AdoptionLedger(
                last_round_counts={0: 1}, last_round_contributions={0: 1.0}
            )
        ),
    )

    def update_scores(rewards, uids):
        scattered = np.zeros_like(state.scores)
        scattered[np.asarray(uids)] = rewards
        state.scores = 0.1 * scattered + 0.9 * state.scores
        state.scores[state.scores < 1e-4] = 0.0

    state.update_scores = update_scores
    _decay_empty_validator_round(state)

    assert state.scores == pytest.approx([0.45, 0.18])
    assert state.annotation_scores == pytest.approx([0.36, 0.09])
    assert state.adoption_bonus_scores == pytest.approx([0.18, 0.045])
    assert state.reliability.decays == 1
    assert state.dataset_assembler.ledger.last_round_contributions == {}


def test_response_shape_rejects_nonce_mismatch():
    response = AnnotationTask(task_id="t-1", challenge_nonce="bad-nonce")
    with pytest.raises(ValueError, match="Challenge nonce mismatch"):
        _validate_response_shape(
            response,
            expected_task_id="t-1",
            expected_nonce="good-nonce",
        )


def test_response_shape_requires_annotations_uri():
    response = AnnotationTask(task_id="t-2", challenge_nonce="nonce")
    with pytest.raises(ValueError, match="annotations_uri"):
        _validate_response_shape(
            response,
            expected_task_id="t-2",
            expected_nonce="nonce",
        )


def test_response_shape_rejects_boxes_outside_validator_image_dimensions():
    response = AnnotationTask(
        task_id="t-3", challenge_nonce="nonce", annotations_uri="file:///tmp/a.json"
    )
    payload = AnnotationsFilePayload(
        records=[
            ImageAnnotationDocument(
                image_id="token-1",
                image_url="https://example.test/image.jpg",
                miner_uid="uid-1",
                timestamp="2026-09-23T00:00:00Z",
                model_version="model-v1",
                annotations=[
                    PerImageAnnotationItem(
                        hazard_class="tree", bounding_box=[0, 0, 101, 50]
                    )
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="Out-of-bounds"):
        _validate_response_shape(
            response,
            expected_task_id="t-3",
            expected_nonce="nonce",
            annotations_payload=payload,
            image_dimensions={"image-1": (100, 100)},
            token_to_real_id={"token-1": "image-1"},
        )


def test_response_shape_rejects_duplicate_image_records():
    response = AnnotationTask(
        task_id="t-duplicate", challenge_nonce="nonce", annotations_uri="r2://b/k"
    )
    record = ImageAnnotationDocument(
        image_id="token-1",
        image_url="https://example.test/image.jpg",
        miner_uid="uid-1",
        timestamp="2026-09-23T00:00:00Z",
        model_version="model-v1",
        annotations=[],
    )
    payload = AnnotationsFilePayload(records=[record, record.model_copy(deep=True)])
    with pytest.raises(ValueError, match="Duplicate image_id"):
        _validate_response_shape(
            response,
            expected_task_id="t-duplicate",
            expected_nonce="nonce",
            annotations_payload=payload,
            image_dimensions={"image-1": (100, 100)},
            token_to_real_id={"token-1": "image-1"},
        )


def test_artifact_download_rejects_local_files_and_non_r2_https_hosts(tmp_path: Path):
    artifact = tmp_path / "annotations.json"
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="local/mock"):
        _download_miner_artifact_bytes(artifact.as_uri())
    with pytest.raises(ValueError, match="Cloudflare R2"):
        _download_miner_artifact_bytes("https://127.0.0.1/private")


def test_artifact_download_bounds_local_file_size(tmp_path: Path):
    from template.validator.dual_forward import MAX_ANNOTATION_ARTIFACT_BYTES

    artifact = tmp_path / "oversized.json"
    with artifact.open("wb") as stream:
        stream.truncate(MAX_ANNOTATION_ARTIFACT_BYTES + 1)
    with pytest.raises(ValueError, match="maximum allowed size"):
        _download_miner_artifact_bytes(artifact.as_uri(), allow_file=True)


def test_https_artifact_stream_is_bounded_and_redirects_are_rejected(monkeypatch):
    from template.utils.http_fetch import fetch_url_bytes

    class Response:
        headers = {}
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"12345"

    request_args = {}

    def fake_get(url, **kwargs):
        request_args.update(kwargs)
        return Response()

    monkeypatch.setattr("requests.get", fake_get)
    with pytest.raises(ValueError, match="maximum allowed size"):
        fetch_url_bytes("https://pub-test.r2.dev/annotations.json", max_bytes=4)
    assert request_args["allow_redirects"] is False

    class Redirect(Response):
        status_code = 302
        headers = {"Location": "http://127.0.0.1/admin"}

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Redirect())
    with pytest.raises(ValueError, match="redirects are not permitted"):
        fetch_url_bytes("https://pub-test.r2.dev/annotations.json", max_bytes=100)


def test_r2_artifact_stream_is_bounded_and_closed(monkeypatch):
    import io

    class Body(io.BytesIO):
        closed_after_download = False

        def close(self):
            self.closed_after_download = True
            super().close()

    body = Body(b"12345")
    creds = R2AccessCredentials(
        account_id="abcd1234",
        bucket_name="bucket",
        s3_endpoint="https://abcd1234.r2.cloudflarestorage.com",
        access_key_id="accesskey",
        secret_access_key="secretaccesssecret",
    )
    client = type(
        "Client",
        (),
        {"get_object": lambda self, **kwargs: {"Body": body}},
    )()
    monkeypatch.setattr("template.hazard.r2_storage._s3_client", lambda _creds: client)
    with pytest.raises(ValueError, match="maximum allowed size"):
        download_bytes_from_r2(
            "r2://bucket/annotations.json", creds=creds, max_bytes=4
        )
    assert body.closed_after_download


def test_payload_parser_bounds_polygon_validation_work():
    polygon = [[float(i), float(i % 7)] for i in range(256)]
    payload = {
        "records": [
            {
                "annotations": [
                    {"polygon": polygon} for _ in range(33)
                ]
            }
        ]
    }
    with pytest.raises(ValueError, match="polygon complexity"):
        _parse_annotations_payload(json.dumps(payload).encode("utf-8"))


def test_r2_artifact_download_rejects_attacker_controlled_endpoint():
    creds = R2AccessCredentials(
        account_id="abcd1234",
        bucket_name="bucket",
        s3_endpoint="http://127.0.0.1:9000",
        access_key_id="accesskey",
        secret_access_key="secretaccesssecret",
    )
    with pytest.raises(ValueError, match="Cloudflare R2 endpoint"):
        download_bytes_from_r2("r2://bucket/annotations.json", creds=creds)


def test_r2_artifact_cannot_fall_back_to_validator_credentials():
    with pytest.raises(ValueError, match="Miner R2 credentials are required"):
        _download_miner_artifact_bytes("r2://bucket/annotations.json")


def test_payload_parser_rejects_too_many_annotations_before_model_validation():
    from template.validator.dual_forward import MAX_ANNOTATIONS_PER_ARTIFACT

    payload = {
        "records": [
            {"annotations": [{} for _ in range(MAX_ANNOTATIONS_PER_ARTIFACT + 1)]}
        ]
    }
    with pytest.raises(ValueError, match="too many annotations"):
        _parse_annotations_payload(json.dumps(payload).encode("utf-8"))


def test_payload_parser_rejects_excessive_image_record_count():
    from template.validator.dual_forward import MAX_RECORDS_PER_ARTIFACT

    payload = {"records": [{} for _ in range(MAX_RECORDS_PER_ARTIFACT + 1)]}
    with pytest.raises(ValueError, match="too many image records"):
        _parse_annotations_payload(json.dumps(payload).encode("utf-8"))


def test_adoption_ledger_state_round_trip(tmp_path: Path):
    ledger = AdoptionLedger(
        adoption_counts={1: 4},
        last_round_counts={1: 2},
        adoption_contributions={1: 3.5},
        last_round_contributions={1: 1.5},
        rounds_observed=3,
    )
    path = tmp_path / "adoption_ledger.json"
    path.write_text(json.dumps(ledger.to_jsonable()), encoding="utf-8")
    restored = AdoptionLedger.from_jsonable(json.loads(path.read_text(encoding="utf-8")))
    assert restored.adoption_counts == {1: 4}
    assert restored.last_round_contributions == {1: 1.5}


def test_annotation_score_ema_update():
    state = SimpleNamespace(
        config=SimpleNamespace(neuron=SimpleNamespace(moving_average_alpha=0.25)),
        annotation_scores=np.array([0.0, 0.4], dtype=np.float32),
        adoption_bonus_scores=np.array([0.0, 0.2], dtype=np.float32),
    )

    def update_score_ledgers(breakdowns, uids):
        alpha = float(state.config.neuron.moving_average_alpha)
        for uid, item in zip(uids, breakdowns):
            state.annotation_scores[uid] = (
                alpha * item.annotation_score + (1.0 - alpha) * state.annotation_scores[uid]
            )
            state.adoption_bonus_scores[uid] = (
                alpha * item.adoption_bonus
                + (1.0 - alpha) * state.adoption_bonus_scores[uid]
            )

    update_score_ledgers(
        [
            DualFlywheelBreakdown(
                uid=1,
                annotation_score=1.0,
                adoption_bonus=0.8,
                hallucination_multiplier=1.0,
                final_score=0.94,
                fidelity_image_ids=2,
                consensus_image_ids=3,
                adopted_image_ids_round=2,
                adopted_image_ids_total=5,
            )
        ],
        [1],
    )
    assert state.annotation_scores[1] == pytest.approx(0.55)
    assert state.adoption_bonus_scores[1] == pytest.approx(0.35)


def test_hotkey_replacement_clears_all_uid_owned_history():
    ledger = AdoptionLedger(
        adoption_counts={1: 8},
        last_round_counts={1: 2},
        adoption_contributions={1: 4.5},
        last_round_contributions={1: 1.5},
    )
    reliability = _ReliabilityAccumulator()
    reliability.tp[1]["tree"] = 5.0
    reliability.fp[1]["tree"] = 1.0
    reliability.iou_sum[1] = 4.0
    reliability.iou_count[1] = 5.0
    state = SimpleNamespace(
        scores=np.array([0.2, 0.8], dtype=np.float32),
        annotation_scores=np.array([0.1, 0.7], dtype=np.float32),
        adoption_bonus_scores=np.array([0.05, 0.3], dtype=np.float32),
        dataset_assembler=SimpleNamespace(ledger=ledger),
        reliability=reliability,
    )
    Validator._on_hotkey_changed(state, 1, "old-hotkey", "new-hotkey")
    assert state.scores[1] == 0.0
    assert state.annotation_scores[1] == 0.0
    assert state.adoption_bonus_scores[1] == 0.0
    assert 1 not in ledger.adoption_counts
    assert 1 not in ledger.last_round_contributions
    assert 1 not in reliability.tp
    assert 1 not in reliability.iou_sum


def test_per_miner_average_score_uses_golden_only():
    score = PerMinerAnnotationScore(
        uid=7,
        fidelity_scores_by_image_id={"g1": 0.9, "g2": 0.7},
        consensus_scores_by_image_id={"pool1": 0.1, "pool2": 0.2},
    )
    assert score.average_score() == pytest.approx(0.8)


def test_annotation_item_accepts_float_boxes():
    item = PerImageAnnotationItem(
        hazard_class="trip_hazard",
        bounding_box=[1.5, 2.5, 10.0, 12.0],
        severity="low",
    )
    assert item.bounding_box[0] == pytest.approx(1.5)


@pytest.mark.parametrize(
    "box",
    [[float("nan")] * 4, [0, 0, float("inf"), 1], [4, 0, 3, 1]],
)
def test_annotation_item_rejects_invalid_boxes(box):
    with pytest.raises(ValueError):
        PerImageAnnotationItem(hazard_class="tree", bounding_box=box)


def test_annotation_item_rejects_non_finite_polygon_points():
    with pytest.raises(ValueError, match="polygon"):
        PerImageAnnotationItem(
            hazard_class="tree",
            bounding_box=[0, 0, 10, 10],
            polygon=[[0, 0], [10, 0], [float("nan"), 10]],
        )


def test_annotation_item_rejects_self_intersecting_polygon():
    with pytest.raises(ValueError, match="non-self-intersecting"):
        PerImageAnnotationItem(
            hazard_class="tree",
            bounding_box=[0, 0, 10, 10],
            polygon=[[5, 0], [8, 10], [0, 4], [10, 4], [2, 10]],
        )


def test_broad_softmax_scaling_floor_edge_case():
    # N=20 eligible miners with floor=0.1, which would make floor * N = 2.0 > 1.0
    scores = np.ones(20, dtype=float)
    # Give the first miner a higher score
    scores[0] = 5.0
    
    shaped = broad_softmax_scores(
        scores,
        temperature=0.35,
        floor=0.1,
        min_score=0.05,
    )
    
    # Assert that the sum is exactly 1.0 (or extremely close)
    assert abs(float(shaped.sum()) - 1.0) < 1e-6
    # The first miner must have the highest incentive and not be zeroed out
    assert shaped[0] > shaped[1]
    assert shaped[1] == pytest.approx(shaped[-1])
    assert shaped[0] > 0.0
    assert (shaped >= 0.0).all()


def test_broad_softmax_enforces_minimum_eligibility_score():
    # Tiny stale scores must not remain eligible indefinitely.
    scores = np.array([0.0, 0.0, 0.0, 0.415, 0.0025, 0.0065, 0.0025, 0.0025], dtype=float)
    shaped = broad_softmax_scores(
        scores,
        temperature=0.20,
        floor=0.08,
        min_score=0.0,
    )
    # Inactive miners receive exact 0
    assert (shaped[:3] == 0.0).all()
    assert (shaped[4:] == 0.0).all()
    assert shaped[3] == pytest.approx(1.0)
    # Top miner receives the highest incentive
    assert shaped[3] == shaped[3:].max()
    assert abs(float(shaped.sum()) - 1.0) < 1e-6


def test_broad_softmax_caps_cumulative_floor_at_twenty_percent():
    shaped = broad_softmax_scores(
        np.array([1.0] + [0.05] * 19),
        temperature=0.01,
        floor=0.1,
        min_score=0.05,
    )
    assert float(shaped.sum()) == pytest.approx(1.0)
    assert shaped[1] == pytest.approx(0.01, abs=1e-6)
    assert shaped[0] == pytest.approx(0.81, abs=1e-6)
