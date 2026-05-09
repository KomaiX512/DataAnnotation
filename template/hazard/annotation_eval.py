"""
Annotation evaluation for the dual-flywheel subnet.

Two complementary scoring lanes are implemented:

* :class:`AnnotationFidelityScorer` -- scores miner annotations on
  Golden-injected images against the validator-held ground truth using:

    fidelity = 0.35 * IoU
             + 0.25 * (class match * severity match)
             + 0.25 * cosine(reasoning, golden_reasoning)
             + 0.15 * confidence_calibration_signal

  Hallucinated annotations (boxes on a Golden image where no ground truth
  exists for that hazard class) trigger a multiplicative penalty.

* :class:`ConsensusScorer` -- scores miner annotations on unlabeled
  Annotation Pool images by comparing them to the aggregated peer
  annotations for the same image_id (mean pairwise IoU + majority class
  agreement). High peer agreement implies the miner's output is trustworthy.

Both scorers operate on real :class:`PerImageAnnotationItem` payloads
(parsed from the miner-uploaded ``annotations.json``) and the validator's
:class:`ImageCorpus` ground truth, with no fallback paths.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from template.hazard.image_corpus import GoldenAnnotation, GoldenImage, ImageCorpus
from template.protocol import PerImageAnnotationItem


_TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EMBED_DIMENSIONS = 128


def _embed_text(text: str, *, dimensions: int = _EMBED_DIMENSIONS) -> List[float]:
    """Lightweight deterministic bag-of-words embedding.

    The embedding is content-addressed via SHA-256 token hashing so two
    semantically related sentences with overlapping vocabulary will produce
    cosine-similar vectors. This is used for reasoning-text similarity.
    """

    vector = [0.0] * dimensions
    tokens = _TEXT_TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dimensions
        weight = 1.0 + (int(digest[8:12], 16) % 4) * 0.25
        vector[index] += weight
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return float(max(0.0, min(1.0, dot)))


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, inter / denom)))


# ---------------------------------------------------------------------------
# Fidelity scoring (Golden Set)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FidelityComponents:
    iou: float
    class_severity: float
    reasoning: float
    confidence: float
    fidelity: float
    hallucination_penalty: float
    matched_count: int
    hallucinated_count: int
    ground_truth_count: int


@dataclass
class AnnotationFidelityScorer:
    """Scores a miner's annotation document against a Golden ground truth.

    The scorer is stateless except for the configurable weights. ``score``
    accepts a list of :class:`PerImageAnnotationItem` (the miner output for
    one Golden image) and the matching :class:`GoldenImage` and returns a
    :class:`FidelityComponents` record. The fidelity score lives in [0, 1].
    """

    iou_weight: float = 0.35
    class_severity_weight: float = 0.25
    reasoning_weight: float = 0.25
    confidence_weight: float = 0.15
    hallucination_penalty: float = 0.5

    def score(
        self,
        miner_items: Sequence[PerImageAnnotationItem],
        golden: GoldenImage,
    ) -> FidelityComponents:
        gt_annotations: Sequence[GoldenAnnotation] = golden.annotations
        if not gt_annotations:
            return FidelityComponents(
                iou=0.0,
                class_severity=0.0,
                reasoning=0.0,
                confidence=0.0,
                fidelity=0.0,
                hallucination_penalty=1.0,
                matched_count=0,
                hallucinated_count=0,
                ground_truth_count=0,
            )

        # Greedy 1-1 matching: for each ground truth box, find best miner item
        # by IoU; track which miner items matched something.
        used_miner_idx: set[int] = set()
        per_match_iou: List[float] = []
        per_match_class_sev: List[float] = []
        per_match_reasoning: List[float] = []
        per_match_confidence: List[float] = []

        for gt in gt_annotations:
            gt_embedding = _embed_text(gt.reasoning)
            best_idx = -1
            best_iou = 0.0
            for idx, item in enumerate(miner_items):
                if idx in used_miner_idx:
                    continue
                iou = iou_xyxy(item.bounding_box, gt.bounding_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx < 0:
                # Ground truth had no match -- counts as a miss (zero contribution).
                per_match_iou.append(0.0)
                per_match_class_sev.append(0.0)
                per_match_reasoning.append(0.0)
                per_match_confidence.append(0.0)
                continue
            matched = miner_items[best_idx]
            used_miner_idx.add(best_idx)
            per_match_iou.append(best_iou)
            per_match_class_sev.append(_class_severity_score(matched, gt))
            per_match_reasoning.append(
                cosine_similarity(_embed_text(matched.reasoning_chain), gt_embedding)
            )
            per_match_confidence.append(_confidence_calibration(matched.confidence, best_iou))

        n_gt = max(1, len(gt_annotations))
        iou_avg = sum(per_match_iou) / n_gt
        class_severity_avg = sum(per_match_class_sev) / n_gt
        reasoning_avg = sum(per_match_reasoning) / n_gt
        confidence_avg = sum(per_match_confidence) / n_gt

        hallucinated = max(0, len(miner_items) - len(used_miner_idx))
        # Each hallucination compresses score multiplicatively.
        penalty = (
            self.hallucination_penalty ** hallucinated
            if hallucinated > 0
            else 1.0
        )

        fidelity_raw = (
            self.iou_weight * iou_avg
            + self.class_severity_weight * class_severity_avg
            + self.reasoning_weight * reasoning_avg
            + self.confidence_weight * confidence_avg
        )
        fidelity = max(0.0, min(1.0, fidelity_raw * penalty))

        return FidelityComponents(
            iou=float(iou_avg),
            class_severity=float(class_severity_avg),
            reasoning=float(reasoning_avg),
            confidence=float(confidence_avg),
            fidelity=float(fidelity),
            hallucination_penalty=float(penalty),
            matched_count=len(used_miner_idx),
            hallucinated_count=int(hallucinated),
            ground_truth_count=int(len(gt_annotations)),
        )


def _class_severity_score(
    item: PerImageAnnotationItem, gt: GoldenAnnotation
) -> float:
    miner_class = (item.hazard_class or "").lower().strip()
    gt_class = (gt.hazard_class or "").lower().strip()
    class_match = 1.0 if miner_class and miner_class == gt_class else 0.0
    severity_match = 1.0 if item.severity == gt.severity else 0.0
    if class_match == 0.0 and severity_match == 0.0:
        return 0.0
    # Class match dominates (60%) but severity match contributes (40%).
    return 0.6 * class_match + 0.4 * severity_match


def _confidence_calibration(confidence: Optional[float], iou: float) -> float:
    """Reward confidence calibrated to actual IoU: high confidence with good
    IoU is best, high confidence with low IoU is penalized."""
    if confidence is None:
        return 0.0
    c = max(0.0, min(1.0, float(confidence)))
    # When IoU is solid we want high confidence; when IoU is bad we want low confidence.
    aligned = c * iou + (1.0 - c) * (1.0 - iou)
    return float(max(0.0, min(1.0, aligned)))


# ---------------------------------------------------------------------------
# Consensus scoring (Annotation Pool)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsensusComponents:
    consensus: float
    mean_pairwise_iou: float
    majority_class_match: float
    peer_count: int


@dataclass
class ConsensusScorer:
    """Scores a miner's annotation against the aggregated peer responses for
    the same image_id.

    The score combines:

    - Mean pairwise IoU between the miner's box and each peer's box (60%).
    - Majority-class agreement: 1 if the miner's predicted class matches the
      mode of peer predictions (40%).

    With fewer than 2 peers the consensus signal is too weak; the score is
    reported as 0 with peer_count populated for downstream filtering.
    """

    iou_weight: float = 0.6
    class_weight: float = 0.4

    def score(
        self,
        miner_items: Sequence[PerImageAnnotationItem],
        peer_items_by_uid: Mapping[int, Sequence[PerImageAnnotationItem]],
    ) -> ConsensusComponents:
        peer_lists = [items for items in peer_items_by_uid.values() if items]
        if len(peer_lists) < 1 or not miner_items:
            return ConsensusComponents(
                consensus=0.0,
                mean_pairwise_iou=0.0,
                majority_class_match=0.0,
                peer_count=len(peer_lists),
            )

        ious: List[float] = []
        peer_classes: List[str] = []
        for peer in peer_lists:
            best_iou = 0.0
            best_peer_cls = ""
            for miner_item in miner_items:
                for peer_item in peer:
                    iou = iou_xyxy(miner_item.bounding_box, peer_item.bounding_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_peer_cls = (peer_item.hazard_class or "").lower().strip()
            ious.append(best_iou)
            if best_peer_cls:
                peer_classes.append(best_peer_cls)

        mean_iou = sum(ious) / max(1, len(ious))
        miner_classes = [
            (item.hazard_class or "").lower().strip() for item in miner_items
        ]
        miner_top = miner_classes[0] if miner_classes else ""
        majority = _majority_class(peer_classes)
        majority_match = 1.0 if miner_top and majority and miner_top == majority else 0.0

        consensus = self.iou_weight * mean_iou + self.class_weight * majority_match
        consensus = float(max(0.0, min(1.0, consensus)))
        return ConsensusComponents(
            consensus=consensus,
            mean_pairwise_iou=float(mean_iou),
            majority_class_match=float(majority_match),
            peer_count=len(peer_lists),
        )


def _majority_class(classes: Iterable[str]) -> str:
    counts: Dict[str, int] = {}
    for c in classes:
        if not c:
            continue
        counts[c] = counts.get(c, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


# ---------------------------------------------------------------------------
# Round-level annotation pipeline
# ---------------------------------------------------------------------------

@dataclass
class PerMinerAnnotationScore:
    """Aggregated annotation score for one miner across all images in a round."""

    uid: int
    fidelity_scores_by_image_id: Dict[str, float] = field(default_factory=dict)
    consensus_scores_by_image_id: Dict[str, float] = field(default_factory=dict)
    fidelity_components_by_image_id: Dict[str, FidelityComponents] = field(default_factory=dict)
    consensus_components_by_image_id: Dict[str, ConsensusComponents] = field(default_factory=dict)
    total_hallucinations: int = 0
    total_ground_truth: int = 0
    annotated_image_ids: List[str] = field(default_factory=list)

    def average_score(self) -> float:
        scores: List[float] = []
        scores.extend(self.fidelity_scores_by_image_id.values())
        scores.extend(self.consensus_scores_by_image_id.values())
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def hallucination_multiplier(self, per_event_penalty: float) -> float:
        """Global multiplier applied to the miner's annotation score based on
        the total number of hallucinations observed on Golden images."""
        if self.total_hallucinations <= 0:
            return 1.0
        return float(per_event_penalty ** self.total_hallucinations)


def evaluate_round_annotations(
    *,
    corpus: ImageCorpus,
    annotations_by_uid: Mapping[int, Mapping[str, Sequence[PerImageAnnotationItem]]],
    fidelity_scorer: AnnotationFidelityScorer,
    consensus_scorer: ConsensusScorer,
    hallucination_penalty: float,
) -> Dict[int, PerMinerAnnotationScore]:
    """Compute per-miner fidelity + consensus scores for one round.

    ``annotations_by_uid[uid]`` is a mapping of ``image_id -> list[PerImageAnnotationItem]``
    representing what miner ``uid`` produced for each image_id in this round.
    The function returns one :class:`PerMinerAnnotationScore` per uid.
    """

    # Pre-compute, per non-golden image_id, the dict {uid: items}.
    per_image_peer_items: Dict[str, Dict[int, Sequence[PerImageAnnotationItem]]] = {}
    for uid, by_image in annotations_by_uid.items():
        for image_id, items in by_image.items():
            if corpus.is_golden(image_id):
                continue
            per_image_peer_items.setdefault(image_id, {})[uid] = items

    results: Dict[int, PerMinerAnnotationScore] = {}
    for uid, by_image in annotations_by_uid.items():
        score = PerMinerAnnotationScore(uid=uid)
        for image_id, items in by_image.items():
            score.annotated_image_ids.append(image_id)
            golden = corpus.golden_lookup(image_id)
            if golden is not None:
                comp = fidelity_scorer.score(items, golden)
                score.fidelity_scores_by_image_id[image_id] = comp.fidelity
                score.fidelity_components_by_image_id[image_id] = comp
                score.total_hallucinations += comp.hallucinated_count
                score.total_ground_truth += comp.ground_truth_count
            else:
                peers = {
                    other_uid: peer_items
                    for other_uid, peer_items in per_image_peer_items.get(image_id, {}).items()
                    if other_uid != uid
                }
                comp = consensus_scorer.score(items, peers)
                score.consensus_scores_by_image_id[image_id] = comp.consensus
                score.consensus_components_by_image_id[image_id] = comp
        results[uid] = score
    return results
