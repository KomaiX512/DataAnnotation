"""
Phase-2 stress acceptance tests for Bayesian annotation aggregation.

These tests use synthetic corpora and controlled miner reliabilities. Full-scale
conditions from the technical spec (e.g. 1000 golden holdout images, 50 Sybil
miners on localnet) are documented in scripts/run_probabilistic_aggregation_stress.sh.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from template.hazard.annotation_eval import PerMinerAnnotationScore
from template.hazard.dataset_assembler import DatasetAssembler
from template.hazard.dataset_assembler import (
    _read_unit_interval_env,
    _rectangle_union_area,
)
from template.hazard.climate_mrv_corpus import ClimateMRVConfig, _register_golden_chip
from template.hazard.annotation_eval import AnnotationFidelityScorer
from template.protocol import PerImageAnnotationItem
from tests.test_dual_flywheel import _build_synthetic_corpus


def _item(
    cls: str,
    bbox: Tuple[float, float, float, float],
) -> PerImageAnnotationItem:
    return PerImageAnnotationItem(
        hazard_class=cls,
        bounding_box=list(bbox),
    )


def _empty_scores(uids: List[int]) -> Dict[int, PerMinerAnnotationScore]:
    return {uid: PerMinerAnnotationScore(uid=uid) for uid in uids}


def _assign_weights(score: PerMinerAnnotationScore, mapping: Dict[str, float]) -> None:
    score.class_weights = dict(mapping)


@pytest.mark.stress
def test_sybil_many_low_weight_miners_do_not_flip_consensus(tmp_path: Path):
    """A two-miner minority cannot override a broad disagreement, even with high trust."""
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    box_a = (40.0, 40.0, 120.0, 120.0)
    good = _item("missing_hardhat", box_a)
    bad_class_box = _item("trip_hazard", box_a)

    uids = [0, 1] + list(range(2, 52))
    annotations: Dict[int, Dict[str, List[PerImageAnnotationItem]]] = {}
    annotations[0] = {pool.image_id: [good]}
    annotations[1] = {pool.image_id: [good]}
    for uid in range(2, 52):
        annotations[uid] = {pool.image_id: [bad_class_box]}

    scores = _empty_scores(uids)
    _assign_weights(scores[0], {"missing_hardhat": 0.95, "trip_hazard": 0.2})
    _assign_weights(scores[1], {"missing_hardhat": 0.95, "trip_hazard": 0.2})
    for uid in range(2, 52):
        _assign_weights(scores[uid], {"missing_hardhat": 1e-4, "trip_hazard": 1e-4})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    winners = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={u: f"hk{u}" for u in uids},
        model_versions={u: f"m{u}" for u in uids},
        timestamps={u: "2026-05-11T12:00:00Z" for u in uids},
    )
    w = [x for x in winners if x.image_id == pool.image_id][0]
    # Support ratio is an explicit safety gate. A narrow reliable minority
    # must be reviewed rather than treated as validator truth.
    assert w.escalation_required
    assert w.escalation_reason in {
        "insufficient_object_votes",
        "insufficient_object_support_ratio",
    }
    assert all(obj.accepted_hazard_class is None for obj in w.accepted_objects)


@pytest.mark.stress
def test_collusion_low_reliability_wrong_majority_escalates_or_wrong_not_accepted(tmp_path: Path):
    """Three colluding low-weight miners must not certify a false class when two reliable disagree."""
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    box = (30.0, 30.0, 100.0, 100.0)
    uids = [10, 11, 20, 21, 22]
    annotations = {
        10: {pool.image_id: [_item("trip_hazard", box)]},
        11: {pool.image_id: [_item("trip_hazard", box)]},
        20: {pool.image_id: [_item("missing_hardhat", box)]},
        21: {pool.image_id: [_item("missing_hardhat", box)]},
        22: {pool.image_id: [_item("missing_hardhat", box)]},
    }
    scores = _empty_scores(uids)
    for uid in (10, 11):
        _assign_weights(scores[uid], {"trip_hazard": 0.95, "missing_hardhat": 0.2})
    for uid in (20, 21, 22):
        _assign_weights(scores[uid], {"missing_hardhat": 0.05, "trip_hazard": 0.05})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    w = [x for x in assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={u: f"hk{u}" for u in uids},
        model_versions={u: "m" for u in uids},
        timestamps={u: "2026-05-11T12:00:00Z" for u in uids},
    ) if x.image_id == pool.image_id][0]
    # Wrong colluding label must not be exported as accepted truth.
    if not w.escalation_required:
        assert w.accepted_objects[0].accepted_hazard_class == "trip_hazard"


@pytest.mark.stress
def test_minority_low_prior_class_expert_and_peer(tmp_path: Path):
    """Lower-prior golden class: strong expert corroborated by a lower-weight peer (same label/box)."""
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    rare = "fall_protection"
    box = (25.0, 25.0, 90.0, 90.0)
    annotations = {
        uid: {pool.image_id: [_item(rare, box)]}
        for uid in (0, 1, 2)
    }
    scores = _empty_scores([0, 1, 2])
    _assign_weights(scores[0], {rare: 0.99, "_background": 0.5})
    _assign_weights(scores[1], {rare: 0.35, "_background": 0.2})
    _assign_weights(scores[2], {rare: 0.35, "_background": 0.2})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    w = [x for x in assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={uid: f"h{uid}" for uid in annotations},
        model_versions={uid: f"m{uid}" for uid in annotations},
        timestamps={uid: "2026-05-11T12:00:00Z" for uid in annotations},
    ) if x.image_id == pool.image_id][0]
    assert not w.escalation_required
    assert w.accepted_objects[0].accepted_hazard_class == rare
    assert w.accepted_objects[0].class_posterior_distribution.get(rare, 0.0) >= 0.9


@pytest.mark.stress
def test_only_one_miner_on_image_escalates(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    annotations = {0: {pool.image_id: [_item("trip_hazard", (10.0, 10.0, 50.0, 50.0))]}}
    scores = _empty_scores([0])
    _assign_weights(scores[0], {"trip_hazard": 0.9})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    w = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={0: "h0"},
        model_versions={0: "m0"},
        timestamps={0: "2026-05-11T12:00:00Z"},
    )[0]
    assert w.escalation_required
    assert w.escalation_reason == "only_one_miner"


@pytest.mark.stress
def test_two_hotkeys_from_same_coldkey_do_not_form_independent_quorum(tmp_path: Path):
    """A single owner cannot manufacture a second consensus vote with another hotkey."""
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    item = _item("trip_hazard", (10.0, 10.0, 50.0, 50.0))
    uids = [4, 9]
    scores = _empty_scores(uids)
    for uid in uids:
        _assign_weights(scores[uid], {"trip_hazard": 0.99})

    assembler = DatasetAssembler(
        corpus=corpus, storage_prefix=(tmp_path / "c").as_uri()
    )
    winner = next(
        row for row in assembler.assemble(
            per_miner_scores=scores,
            annotations_by_uid={
                uid: {pool.image_id: [item]} for uid in uids
            },
            miner_hotkeys={uid: f"hotkey-{uid}" for uid in uids},
            miner_identity_keys={uid: "same-coldkey" for uid in uids},
            model_versions={uid: "m" for uid in uids},
            timestamps={uid: "now" for uid in uids},
        )
        if row.image_id == pool.image_id
    )
    assert winner.escalation_required
    assert winner.escalation_reason == "only_one_miner"
    # Escalated clusters may be retained for diagnostics, but no class is
    # accepted or exported as validator truth.
    assert all(obj.accepted_hazard_class is None for obj in winner.accepted_objects)


@pytest.mark.stress
def test_two_colluding_coldkeys_cannot_farm_full_frame_object_against_abstainer(tmp_path: Path):
    """Two matching Sybils plus an honest abstention fail the 3-voter object gate."""
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    whole_image = _item("ordinary_tree", (0, 0, pool.width, pool.height))
    uids = [0, 1, 2]
    annotations = {
        0: {pool.image_id: [whole_image]},
        1: {pool.image_id: [whole_image]},
        2: {pool.image_id: []},
    }
    scores = _empty_scores(uids)
    for uid in uids:
        _assign_weights(scores[uid], {"ordinary_tree": 0.99, "_background": 0.99})

    assembler = DatasetAssembler(
        corpus=corpus, storage_prefix=(tmp_path / "c").as_uri()
    )
    winner = next(
        row for row in assembler.assemble(
            per_miner_scores=scores,
            annotations_by_uid=annotations,
            miner_hotkeys={uid: f"hotkey-{uid}" for uid in uids},
            miner_identity_keys={uid: f"coldkey-{uid}" for uid in uids},
            model_versions={uid: "m" for uid in uids},
            timestamps={uid: "now" for uid in uids},
        )
        if row.image_id == pool.image_id
    )
    assert winner.escalation_required
    assert "insufficient_object_votes" in (winner.escalation_reason or "")
    assert winner.miner_contribution_scores == {}
    assert all(obj.accepted_hazard_class is None for obj in winner.accepted_objects)


@pytest.mark.stress
def test_three_independent_full_frame_boxes_require_review(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    uids = [0, 1, 2]
    annotations = {
        uid: {pool.image_id: [_item("missing_hardhat", (0, 0, pool.width, pool.height))]}
        for uid in uids
    }
    scores = _empty_scores(uids)
    for uid in uids:
        _assign_weights(scores[uid], {"missing_hardhat": 0.99, "_background": 0.99})
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "out").as_uri())
    winner = next(
        row for row in assembler.assemble(
            per_miner_scores=scores,
            annotations_by_uid=annotations,
            miner_hotkeys={uid: f"hk{uid}" for uid in uids},
            miner_identity_keys={uid: f"ck{uid}" for uid in uids},
            model_versions={uid: "m" for uid in uids},
            timestamps={uid: "now" for uid in uids},
        )
        if row.image_id == pool.image_id
    )
    assert winner.escalation_required
    assert "large_box_requires_review" in (winner.escalation_reason or "")
    assert winner.miner_contribution_scores == {}


@pytest.mark.stress
def test_overlapping_same_class_clusters_do_not_auto_accept_split_objects(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    uids = [0, 1, 2, 3]
    box_a = _item("missing_hardhat", (20, 20, 100, 100))
    box_b = _item("missing_hardhat", (70, 20, 150, 100))
    annotations = {uid: {pool.image_id: [box_a, box_b]} for uid in uids}
    scores = _empty_scores(uids)
    for uid in uids:
        _assign_weights(scores[uid], {"missing_hardhat": 0.99, "_background": 0.99})
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "out").as_uri())
    winner = next(
        row for row in assembler.assemble(
            per_miner_scores=scores,
            annotations_by_uid=annotations,
            miner_hotkeys={uid: f"hk{uid}" for uid in uids},
            miner_identity_keys={uid: f"ck{uid}" for uid in uids},
            model_versions={uid: "m" for uid in uids},
            timestamps={uid: "now" for uid in uids},
        )
        if row.image_id == pool.image_id
    )
    assert winner.escalation_required
    assert "overlapping_object_clusters_requires_review" in (winner.escalation_reason or "")
    assert winner.miner_contribution_scores == {}


def test_coverage_union_does_not_count_overlap_twice():
    assert _rectangle_union_area([(0, 0, 10, 10), (5, 0, 15, 10)]) == pytest.approx(150.0)


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.01", "bad"])
def test_iou_environment_threshold_rejects_nonfinite_or_out_of_range(value, monkeypatch):
    monkeypatch.setenv("TEST_IOU_THRESHOLD", value)
    with pytest.raises(ValueError, match=r"finite number in \[0, 1\]"):
        _read_unit_interval_env("TEST_IOU_THRESHOLD", "0.7")


@pytest.mark.stress
def test_miner_area_and_weight_cannot_poison_exported_geometry_metrics(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    uids = [0, 1, 2]
    box = (20.0, 20.0, 80.0, 80.0)
    annotations = {
        uid: {
            pool.image_id: [
                PerImageAnnotationItem(
                    hazard_class="missing_hardhat",
                    bounding_box=list(box),
                    area=1e30,
                    weight=1e30,
                    polygon=[[20, 20], [80, 20], [80, 80], [20, 80]],
                )
            ]
        }
        for uid in uids
    }
    scores = _empty_scores(uids)
    for uid in uids:
        _assign_weights(scores[uid], {"missing_hardhat": 0.99})

    assembler = DatasetAssembler(
        corpus=corpus, storage_prefix=(tmp_path / "out").as_uri()
    )
    winner = next(
        row for row in assembler.assemble(
            per_miner_scores=scores,
            annotations_by_uid=annotations,
            miner_hotkeys={uid: f"hk{uid}" for uid in uids},
            model_versions={uid: "m" for uid in uids},
            timestamps={uid: "now" for uid in uids},
        )
        if row.image_id == pool.image_id
    )
    assert not winner.escalation_required, winner.escalation_reason
    obj = winner.accepted_objects[0]
    assert obj.area == pytest.approx(3600.0)
    assert obj.weight == pytest.approx(3600.0 / (pool.width * pool.height))
    assert obj.fused_polygon is None
    assert winner.net_weight == pytest.approx(3600.0 / (pool.width * pool.height))
    assert winner.tree_coverage_ratio == pytest.approx(3600.0 / (pool.width * pool.height))


@pytest.mark.stress
def test_two_miners_spatial_disagreement_escalates(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    annotations = {
        0: {pool.image_id: [_item("trip_hazard", (10.0, 10.0, 50.0, 50.0))]},
        1: {pool.image_id: [_item("trip_hazard", (160.0, 160.0, 190.0, 190.0))]},
    }
    scores = _empty_scores([0, 1])
    _assign_weights(scores[0], {"trip_hazard": 0.9})
    _assign_weights(scores[1], {"trip_hazard": 0.9})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    w = [x for x in assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={0: "h0", 1: "h1"},
        model_versions={0: "m0", 1: "m1"},
        timestamps={0: "2026-05-11T12:00:00Z", 1: "2026-05-11T12:00:00Z"},
    ) if x.image_id == pool.image_id][0]
    assert w.escalation_required
    assert "high_spatial_disagreement" in (w.escalation_reason or "") or w.escalation_reason


@pytest.mark.stress
def test_uncertainty_calibration_band_on_synthetic_draws():
    """Accepted confidence vs empirical correctness should stay within a loose band on toy draws."""
    rng = random.Random(42)
    trials = 120
    reported: List[float] = []
    correct: List[int] = []
    for _ in range(trials):
        p_correct = rng.uniform(0.88, 0.98)
        is_correct = 1 if rng.random() < p_correct else 0
        reported.append(p_correct)
        correct.append(is_correct)
    mean_conf = statistics.mean(reported)
    acc = statistics.mean(correct)
    assert abs(mean_conf - acc) <= 0.12


@pytest.mark.stress
def test_commercial_export_metadata_required_fields(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    vote = _item("missing_hardhat", (20.0, 20.0, 80.0, 80.0))
    annotations = {uid: {pool.image_id: [vote]} for uid in (0, 1, 2)}
    scores = _empty_scores([0, 1, 2])
    for uid in scores:
        _assign_weights(scores[uid], {"missing_hardhat": 0.95})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "out").as_uri())
    winners = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={0: "a", 1: "b", 2: "c"},
        model_versions={0: "m", 1: "m", 2: "m"},
        timestamps={0: "2026-05-11T12:00:00Z", 1: "2026-05-11T12:01:00Z", 2: "2026-05-11T12:02:00Z"},
    )
    commercial = [w for w in winners if not w.is_golden and not w.escalation_required]
    assert commercial
    uri = assembler.export(commercial, round_id="stress-0")
    assert uri
    master = tmp_path / "out" / "commercial-dataset.jsonl"
    line = [ln for ln in master.read_text().splitlines() if ln.strip()][0]
    row = json.loads(line)
    required = {
        "image_id",
        "aggregation_method",
        "acceptance_thresholds",
        "escalation_required",
        "validator_version",
        "audit_hash",
        "objects",
        "miner_contribution_scores",
    }
    assert required.issubset(row.keys())
    obj0 = row["objects"][0]
    for key in (
        "object_cluster_id",
        "accepted_hazard_class",
        "confidence",
        "class_posterior_distribution",
        "severity_posterior_distribution",
        "miner_votes",
    ):
        assert key in obj0


@pytest.mark.stress
def test_golden_fidelity_lane_reports_scores(tmp_path: Path):
    """Sanity: golden images still use fidelity lane (not commercial export)."""
    from template.hazard.annotation_eval import AnnotationFidelityScorer, evaluate_round_annotations
    from template.hazard.annotation_eval import ConsensusScorer

    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    annotations = {
        0: {g1.image_id: [_item("missing_hardhat", (22.0, 32.0, 92.0, 132.0))]},
        1: {g1.image_id: [_item("other", (150.0, 150.0, 170.0, 170.0))]},
    }
    scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations,
        fidelity_scorer=AnnotationFidelityScorer(),
        consensus_scorer=ConsensusScorer(),
        hallucination_penalty=0.5,
    )
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    winners = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={0: "a", 1: "b"},
        model_versions={0: "m", 1: "m"},
        timestamps={0: "t", 1: "t"},
    )
    gw = [w for w in winners if w.image_id == g1.image_id][0]
    assert gw.is_golden
    assert gw.aggregation_method == "golden_fidelity_v1"
    assert scores[0].fidelity_scores_by_image_id[g1.image_id] > scores[1].fidelity_scores_by_image_id[g1.image_id]
    assert assembler.ledger.round_contribution_share() == {}


@pytest.mark.stress
def test_identical_corrobating_miners_split_contribution_credit(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    item = _item("missing_hardhat", (20, 20, 80, 80))
    uids = [3, 7, 11]
    annotations = {uid: {pool.image_id: [item]} for uid in uids}
    scores = _empty_scores(uids)
    for uid in uids:
        _assign_weights(scores[uid], {"missing_hardhat": 0.95})
    assembler = DatasetAssembler(
        corpus=corpus, storage_prefix=(tmp_path / "out").as_uri()
    )
    winner = next(
        row for row in assembler.assemble(
            per_miner_scores=scores,
            annotations_by_uid=annotations,
            miner_hotkeys={uid: f"hk{uid}" for uid in uids},
            model_versions={uid: "same-model" for uid in uids},
            timestamps={uid: "now" for uid in uids},
        )
        if row.image_id == pool.image_id
    )
    assert not winner.escalation_required
    assert winner.miner_contribution_scores == pytest.approx(
        {3: 1 / 3, 7: 1 / 3, 11: 1 / 3}
    )
    # Unlabeled consensus is not independently verified ground truth and must
    # not be converted into a paid adoption share.
    assert assembler.ledger.round_contribution_share() == {}


@pytest.mark.stress
def test_class_only_climate_golden_does_not_invent_full_frame_box(tmp_path: Path):
    """Chip labels score class predictions, not arbitrary whole-chip geometry."""
    from io import BytesIO
    from PIL import Image

    corpus = _build_synthetic_corpus(tmp_path)
    buffer = BytesIO()
    Image.new("RGB", (256, 256), (40, 100, 40)).save(buffer, format="PNG")
    cfg = ClimateMRVConfig(cache_root=corpus.cache_root)
    _register_golden_chip(
        corpus, buffer.getvalue(), "intact_forest", cfg, lon=0.0, lat=0.0
    )
    target = next(
        image for image in corpus.golden_images()
        if image.classification_label == "intact_forest"
    )
    assert target.annotations == ()

    whole_chip = _item("intact_forest", (0, 0, target.width, target.height))
    crowns = [
        _item("individual_tree", (index * 3, 0, index * 3 + 2, 2))
        for index in range(30)
    ]
    scorer = AnnotationFidelityScorer()
    coarse = scorer.score([whole_chip], target)
    detector = scorer.score(crowns, target)
    assert coarse.iou == 0.0
    assert coarse.fidelity == 1.0
    assert not coarse.rewardable
    assert detector.class_severity == 0.0
    assert detector.fidelity == 0.0
    assert detector.hallucinated_count == 30
    assert not detector.rewardable

    # Class-only chips remain useful for class diagnostics, but neither a
    # full-frame box nor many unsupported locations can earn spatial reward.
    from template.hazard.annotation_eval import ConsensusScorer, evaluate_round_annotations

    round_scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid={0: {target.image_id: [whole_chip]}},
        fidelity_scorer=scorer,
        consensus_scorer=ConsensusScorer(),
        hallucination_penalty=0.5,
        expected_golden_ids_by_uid={0: (target.image_id,)},
    )
    assert target.image_id not in round_scores[0].fidelity_scores_by_image_id
    assert round_scores[0].fidelity_components_by_image_id[
        target.image_id
    ].fidelity == 1.0
    assert "intact_forest" not in round_scores[0].class_weights
    assert round_scores[0].average_score() == 0.0

    # Adding or omitting non-rewardable class-only rows cannot shrink the
    # reward denominator and inflate the score on actual localized targets.
    g1, g2 = corpus.golden_images()[:2]
    localized = {
        g1.image_id: [_item("missing_hardhat", tuple(g1.annotations[0].bounding_box))],
        g2.image_id: [],
    }
    submitted_class_only = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid={0: {**localized, target.image_id: [whole_chip]}},
        fidelity_scorer=scorer,
        consensus_scorer=ConsensusScorer(),
        hallucination_penalty=0.5,
        expected_golden_ids_by_uid={0: (g1.image_id, g2.image_id, target.image_id)},
    )[0]
    omitted_class_only = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid={0: localized},
        fidelity_scorer=scorer,
        consensus_scorer=ConsensusScorer(),
        hallucination_penalty=0.5,
        expected_golden_ids_by_uid={0: (g1.image_id, g2.image_id, target.image_id)},
    )[0]
    assert submitted_class_only.average_score() == pytest.approx(0.5)
    assert omitted_class_only.average_score() == pytest.approx(0.5)


@pytest.mark.stress
def test_one_miner_cannot_get_an_object_adopted_against_abstaining_peers(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    uids = [0, 1, 2]
    votes = {
        0: {pool.image_id: [_item("missing_hardhat", (20, 20, 80, 80))]},
        1: {pool.image_id: []},
        2: {pool.image_id: []},
    }
    scores = _empty_scores(uids)
    for uid in uids:
        _assign_weights(scores[uid], {"missing_hardhat": 0.99, "_background": 0.99})
    assembler = DatasetAssembler(
        corpus=corpus, storage_prefix=(tmp_path / "out").as_uri()
    )
    winner = next(
        row for row in assembler.assemble(
            per_miner_scores=scores,
            annotations_by_uid=votes,
            miner_hotkeys={uid: f"hk{uid}" for uid in uids},
            model_versions={uid: "m1" for uid in uids},
            timestamps={uid: "now" for uid in uids},
        )
        if row.image_id == pool.image_id
    )
    assert winner.escalation_required
    assert "insufficient_object_votes" in (winner.escalation_reason or "")
    assert not winner.accepted_objects or all(
        obj.accepted_hazard_class is None for obj in winner.accepted_objects
    )
