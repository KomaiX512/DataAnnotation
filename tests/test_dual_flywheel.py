"""End-to-end tests for the dual-flywheel validator pipeline.

The tests bypass HuggingFace dataset downloads by constructing an
``ImageCorpus`` with synthetic images plus hand-built Golden / Training /
Annotation / Benchmark records. This exercises every scoring lane,
the injection planner, the dataset assembler, the commercial export,
and the final reward composer using only in-memory state.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from template.hazard.annotation_eval import (
    AnnotationFidelityScorer,
    ConsensusScorer,
    cosine_similarity,
    evaluate_round_annotations,
    iou_xyxy,
)
from template.hazard.dataset_assembler import (
    AdoptionLedger,
    DatasetAssembler,
    WinningAnnotation,
)
from template.hazard.dual_reward import DualFlywheelRewardComposer
from template.hazard.golden_injection import GoldenInjector
from template.hazard.image_corpus import (
    BenchmarkImage,
    GoldenAnnotation,
    GoldenImage,
    ImageCorpus,
    ImageCorpusConfig,
    TrainingImage,
    UnlabeledImage,
    _severity_for_label,
)


def test_image_corpus_normalized_annotation_entries_parses_at_split(tmp_path: Path):
    cfg = ImageCorpusConfig(
        cache_root=tmp_path,
        annotation_dataset_ids="foo/bar@test, baz/qux@validation, plain/id",
        annotation_split="train",
    )
    assert cfg.normalized_annotation_entries() == [
        ("foo/bar", "test"),
        ("baz/qux", "validation"),
        ("plain/id", "train"),
    ]
from template.protocol import PerImageAnnotationItem


def _png_bytes(width: int, height: int, color=(120, 140, 160)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _build_synthetic_corpus(tmp_path: Path) -> ImageCorpus:
    """Build a fully-populated ImageCorpus without touching HuggingFace."""
    cache = tmp_path / "image_cache"
    cache.mkdir(parents=True, exist_ok=True)
    config = ImageCorpusConfig(cache_root=cache)
    corpus = ImageCorpus(config)
    # Deliberately bypass HF loading; populate the in-memory indexes directly.
    corpus._loaded = True

    def _materialize(image_id: str, payload: bytes, ext: str = "png") -> Path:
        path = cache / f"{image_id}.{ext}"
        path.write_bytes(payload)
        corpus._all_image_index[image_id] = path
        return path

    # Two Golden images with known boxes / classes / severities.
    g1_bytes = _png_bytes(200, 200, (100, 110, 120))
    g1_id = hashlib.sha256(g1_bytes).hexdigest()
    g1_path = _materialize(g1_id, g1_bytes)
    g1 = GoldenImage(
        image_id=g1_id,
        image_path=g1_path,
        image_url=g1_path.as_uri(),
        width=200,
        height=200,
        annotations=(
            GoldenAnnotation(
                hazard_class="missing_hardhat",
                bounding_box=(20, 30, 90, 130),
                severity="high",
                reasoning="Worker without hardhat near scaffold.",
            ),
        ),
    )
    corpus._golden.append(g1)
    corpus._golden_index[g1_id] = g1

    g2_bytes = _png_bytes(160, 160, (130, 90, 70))
    g2_id = hashlib.sha256(g2_bytes).hexdigest()
    g2_path = _materialize(g2_id, g2_bytes)
    g2 = GoldenImage(
        image_id=g2_id,
        image_path=g2_path,
        image_url=g2_path.as_uri(),
        width=160,
        height=160,
        annotations=(
            GoldenAnnotation(
                hazard_class="fall_protection",
                bounding_box=(10, 20, 60, 90),
                severity="high",
                reasoning="Unprotected edge with fall hazard.",
            ),
        ),
    )
    corpus._golden.append(g2)
    corpus._golden_index[g2_id] = g2

    # Three unlabeled annotation pool images.
    for idx, color in enumerate([(40, 60, 80), (90, 40, 50), (50, 90, 40)]):
        payload = _png_bytes(180, 180, color)
        img_id = hashlib.sha256(payload).hexdigest()
        path = _materialize(img_id, payload)
        corpus._annotation.append(
            UnlabeledImage(
                image_id=img_id,
                image_path=path,
                image_url=path.as_uri(),
                width=180,
                height=180,
                source_dataset=f"synthetic-{idx}",
            )
        )

    # Two training pool images (labeled).
    for idx, color in enumerate([(10, 20, 30), (200, 200, 200)]):
        payload = _png_bytes(192, 192, color)
        img_id = hashlib.sha256(payload).hexdigest()
        path = _materialize(img_id, payload)
        corpus._training.append(
            TrainingImage(
                image_id=img_id,
                image_path=path,
                image_url=path.as_uri(),
                width=192,
                height=192,
                annotations=(
                    GoldenAnnotation(
                        hazard_class="missing_vest",
                        bounding_box=(30, 30, 90, 100),
                        severity="high",
                        reasoning="Missing safety vest in active work area.",
                    ),
                ),
            )
        )

    # One benchmark image.
    bench_bytes = _png_bytes(220, 220, (10, 10, 10))
    bench_id = hashlib.sha256(bench_bytes).hexdigest()
    bench_path = _materialize(bench_id, bench_bytes)
    corpus._benchmark.append(
        BenchmarkImage(
            image_id=bench_id,
            image_path=bench_path,
            width=220,
            height=220,
            annotations=(
                GoldenAnnotation(
                    hazard_class="fall_protection",
                    bounding_box=(30, 40, 100, 150),
                    severity="high",
                    reasoning="Fall hazard near edge.",
                ),
            ),
            source_dataset="synthetic-benchmark",
        )
    )
    return corpus


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def test_severity_label_mapping_matches_spec():
    assert _severity_for_label("missing-hardhat") == "high"
    assert _severity_for_label("missing_vest") == "high"
    assert _severity_for_label("fall_protection") == "high"
    assert _severity_for_label("trip_hazard") == "low"
    assert _severity_for_label("hardhat") == "medium"


def test_iou_basic():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou_xyxy([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175, rel=1e-4)


def test_cosine_similarity_for_overlapping_text():
    from template.hazard.annotation_eval import _embed_text

    a = _embed_text("Worker without hardhat near scaffold.")
    b = _embed_text("Construction worker missing hardhat next to scaffolding.")
    c = _embed_text("Spilled coffee near a desk in an office.")
    assert cosine_similarity(a, b) > cosine_similarity(a, c)


# ---------------------------------------------------------------------------
# Image corpus + golden injection
# ---------------------------------------------------------------------------

def test_synthetic_corpus_builds(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    assert len(corpus.golden_images()) == 2
    assert len(corpus.training_images()) == 2
    assert len(corpus.annotation_images()) == 3
    assert len(corpus.benchmark_images()) == 1
    g1 = corpus.golden_images()[0]
    assert corpus.is_golden(g1.image_id)
    assert corpus.golden_lookup(g1.image_id) is g1


def test_golden_injector_distributes_correctly(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    injector = GoldenInjector(corpus=corpus, request_size=4, golden_per_request=2)
    rng = random.Random(42)
    plan = injector.build_plan(rng)
    assert len(plan.ordered_images) == 4
    assert len(plan.golden_image_ids) == 2
    assert len(plan.annotation_image_ids) == 2
    all_ids = {iid for iid, _ in plan.ordered_images}
    assert all_ids == set(plan.golden_image_ids) | set(plan.annotation_image_ids)


def test_golden_injector_rejects_oversize_request(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    with pytest.raises(RuntimeError):
        injector = GoldenInjector(corpus=corpus, request_size=10, golden_per_request=2)
        injector.build_plan(random.Random(1))


# ---------------------------------------------------------------------------
# Annotation fidelity + consensus scoring
# ---------------------------------------------------------------------------

def _miner_item(
    *,
    cls: str = "missing_hardhat",
    bbox=(22, 32, 92, 132),
    severity: str = "high",
    confidence: float = 0.9,
    reasoning: str = "Worker missing hardhat near scaffold.",
) -> PerImageAnnotationItem:
    return PerImageAnnotationItem(
        hazard_class=cls,
        bounding_box=list(bbox),
        severity=severity,
        confidence=confidence,
        reasoning_chain=reasoning,
        osha_reference="29CFR1926.95",
    )


def test_fidelity_scorer_high_quality_match(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    scorer = AnnotationFidelityScorer()
    g1 = corpus.golden_images()[0]
    components = scorer.score([_miner_item()], g1)
    assert components.iou > 0.7
    assert components.class_severity == pytest.approx(1.0)
    assert components.reasoning > 0.3
    assert components.hallucination_penalty == 1.0
    assert components.fidelity > 0.6
    assert components.matched_count == 1


def test_fidelity_scorer_penalizes_hallucinations(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    scorer = AnnotationFidelityScorer(hallucination_penalty=0.5)
    g1 = corpus.golden_images()[0]
    halluc = _miner_item(
        cls="random_object",
        bbox=(140, 140, 180, 180),
        severity="low",
        reasoning="Saw something unrelated.",
    )
    components = scorer.score([_miner_item(), halluc], g1)
    assert components.hallucinated_count == 1
    assert components.hallucination_penalty == pytest.approx(0.5)
    assert components.fidelity < 0.5


def test_fidelity_scorer_zeros_for_total_miss(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    scorer = AnnotationFidelityScorer()
    g1 = corpus.golden_images()[0]
    items = [
        _miner_item(
            cls="random_class",
            bbox=(150, 150, 175, 175),
            severity="low",
            reasoning="unrelated",
        )
    ]
    components = scorer.score(items, g1)
    assert components.iou == 0.0
    assert components.fidelity == 0.0
    assert components.matched_count == 0
    assert components.hallucinated_count == 1


def test_consensus_scorer_majority_class(tmp_path):
    scorer = ConsensusScorer()
    miner_items = [_miner_item(cls="missing_hardhat", bbox=(10, 10, 50, 50))]
    peers = {
        2: [_miner_item(cls="missing_hardhat", bbox=(12, 12, 52, 52))],
        3: [_miner_item(cls="missing_hardhat", bbox=(14, 14, 54, 54))],
        4: [_miner_item(cls="trip_hazard", bbox=(80, 80, 100, 100))],
    }
    comp = scorer.score(miner_items, peers)
    assert comp.peer_count == 3
    assert comp.majority_class_match == 1.0
    assert comp.consensus > 0.5


def test_consensus_scorer_zero_peers():
    scorer = ConsensusScorer()
    miner_items = [_miner_item()]
    comp = scorer.score(miner_items, {})
    assert comp.consensus == 0.0
    assert comp.peer_count == 0


def test_evaluate_round_annotations_aggregates_correctly(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    g2 = corpus.golden_images()[1]
    pool = corpus.annotation_images()[0]
    annotations_by_uid = {
        1: {
            g1.image_id: [_miner_item()],
            pool.image_id: [_miner_item(cls="trip_hazard", bbox=(20, 20, 80, 80))],
        },
        2: {
            g1.image_id: [_miner_item(cls="other", bbox=(140, 140, 170, 170))],
            pool.image_id: [_miner_item(cls="trip_hazard", bbox=(22, 22, 82, 82))],
        },
    }
    fidelity = AnnotationFidelityScorer()
    consensus = ConsensusScorer()
    scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations_by_uid,
        fidelity_scorer=fidelity,
        consensus_scorer=consensus,
        hallucination_penalty=0.5,
    )
    assert scores[1].fidelity_scores_by_image_id[g1.image_id] > scores[2].fidelity_scores_by_image_id[g1.image_id]
    assert scores[1].consensus_scores_by_image_id[pool.image_id] > 0.0


# ---------------------------------------------------------------------------
# Dataset assembler
# ---------------------------------------------------------------------------

def test_dataset_assembler_picks_best_per_image_id(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    pool = corpus.annotation_images()[0]
    annotations = {
        1: {g1.image_id: [_miner_item()], pool.image_id: [_miner_item(cls="trip_hazard")]},
        2: {g1.image_id: [_miner_item(cls="other", bbox=(150, 150, 170, 170))],
            pool.image_id: [_miner_item(cls="trip_hazard")]},
    }
    fidelity = AnnotationFidelityScorer()
    consensus = ConsensusScorer()
    scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations,
        fidelity_scorer=fidelity,
        consensus_scorer=consensus,
        hallucination_penalty=0.5,
    )
    storage = (tmp_path / "commercial").as_uri()
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=storage)
    winners = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={1: "hk1", 2: "hk2"},
        model_versions={1: "v1", 2: "v2"},
        timestamps={1: "ts1", 2: "ts2"},
    )
    assert any(w.image_id == g1.image_id and w.chosen_uid == 1 for w in winners)
    assert assembler.ledger.adoption_counts.get(1, 0) >= 1


def test_dataset_assembler_export_local_jsonl(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    items = [_miner_item()]
    winners = [
        WinningAnnotation(
            image_id=g1.image_id,
            chosen_uid=7,
            score=0.85,
            is_golden=True,
            image_url=g1.image_url,
            width=g1.width,
            height=g1.height,
            items=items,
            miner_hotkey="hk7",
            model_version="m" * 32,
            timestamp="2026-05-09T16:00:00Z",
        )
    ]
    target_dir = tmp_path / "commercial"
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=target_dir.as_uri())
    uri = assembler.export(winners, round_id="step-1")
    assert uri.endswith("commercial-dataset-step-1.jsonl")
    master = target_dir / "commercial-dataset.jsonl"
    assert master.exists()
    parsed = [json.loads(line) for line in master.read_text().splitlines() if line.strip()]
    assert parsed[0]["image_id"] == g1.image_id
    assert parsed[0]["chosen_uid"] == 7


# ---------------------------------------------------------------------------
# Reward composer
# ---------------------------------------------------------------------------

def test_dual_reward_composer_combines_three_signals(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    pool = corpus.annotation_images()[0]
    annotations = {
        1: {g1.image_id: [_miner_item()], pool.image_id: [_miner_item(cls="trip_hazard")]},
        2: {g1.image_id: [_miner_item(cls="other", bbox=(150, 150, 170, 170))],
            pool.image_id: [_miner_item(cls="trip_hazard")]},
    }
    scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations,
        fidelity_scorer=AnnotationFidelityScorer(),
        consensus_scorer=ConsensusScorer(),
        hallucination_penalty=0.5,
    )
    storage = (tmp_path / "commercial").as_uri()
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=storage)
    winners = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={1: "hk1", 2: "hk2"},
        model_versions={1: "v1", 2: "v2"},
        timestamps={1: "ts1", 2: "ts2"},
    )

    from template.hazard.model_eval import ModelAccuracyComponents
    model_accuracy = {
        1: ModelAccuracyComponents(
            golden_iou=0.9,
            golden_class_severity=1.0,
            golden_reasoning=0.7,
            golden_confidence=0.8,
            golden_score=0.85,
            benchmark_iou=0.6,
            benchmark_score=0.6,
            overall_score=0.78,
            images_scored=2,
        ),
        2: ModelAccuracyComponents(
            golden_iou=0.2,
            golden_class_severity=0.0,
            golden_reasoning=0.1,
            golden_confidence=0.2,
            golden_score=0.15,
            benchmark_iou=0.2,
            benchmark_score=0.2,
            overall_score=0.17,
            images_scored=2,
        ),
    }

    composer = DualFlywheelRewardComposer(alpha=0.4, beta=0.4, gamma=0.2)
    rewards, breakdowns = composer.compose(
        uids=[1, 2],
        annotation_scores=scores,
        model_accuracy=model_accuracy,
        ledger=assembler.ledger,
        round_winners=winners,
    )
    assert rewards[0] > rewards[1]  # uid 1 should clearly win
    assert breakdowns[0].annotation_score > 0.0
    assert breakdowns[0].model_accuracy_score > 0.0
    assert breakdowns[0].adoption_bonus > 0.0
    assert breakdowns[0].final_score == pytest.approx(rewards[0], abs=1e-6)


def test_dual_reward_composer_validates_weights():
    composer = DualFlywheelRewardComposer(alpha=0.5, beta=0.4, gamma=0.2)  # sums to 1.1
    with pytest.raises(ValueError):
        composer.compose(
            uids=[1],
            annotation_scores={},
            model_accuracy={},
            ledger=AdoptionLedger(),
            round_winners=[],
        )


# ---------------------------------------------------------------------------
# Adoption ledger persistence
# ---------------------------------------------------------------------------

def test_adoption_ledger_round_trips():
    ledger = AdoptionLedger()
    ledger.adoption_counts = {1: 5, 2: 3}
    ledger.last_round_counts = {1: 2}
    ledger.rounds_observed = 7
    serialized = ledger.to_jsonable()
    restored = AdoptionLedger.from_jsonable(serialized)
    assert restored.adoption_counts == {1: 5, 2: 3}
    assert restored.last_round_counts == {1: 2}
    assert restored.rounds_observed == 7
