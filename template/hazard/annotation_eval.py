"""
Annotation evaluation for the annotation-only subnet.

Validators score miner quality only on the secret Golden Set using geometry
and class/severity agreement. These Golden-only scores become per-miner
reliability weights for fusion on the remaining non-Golden images.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import logging
import math

from template.hazard.image_corpus import GoldenAnnotation, GoldenImage, ImageCorpus

_log = logging.getLogger(__name__)
from template.protocol import PerImageAnnotationItem

def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
        bx1, by1, bx2, by2 = (float(v) for v in box_b)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not all(math.isfinite(v) for v in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)):
        return 0.0
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
    fidelity: float
    hallucination_penalty: float
    matched_count: int
    hallucinated_count: int
    ground_truth_count: int
    net_weight_agreement: float = 1.0
    gt_net_weight: float = 0.0
    miner_net_weight: float = 0.0
    rewardable: bool = True


@dataclass
class AnnotationFidelityScorer:
    """Scores a miner's annotation document against a Golden ground truth.

    The scorer is stateless except for the configurable weights. ``score``
    accepts a list of :class:`PerImageAnnotationItem` (the miner output for
    one Golden image) and the matching :class:`GoldenImage` and returns a
    :class:`FidelityComponents` record. The fidelity score lives in [0, 1].
    """

    iou_weight: float = 0.60
    class_weight: float = 0.40
    hallucination_penalty: float = 0.5
    minimum_match_iou: float = 0.50

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_match_iou)
            or not 0.0 <= self.minimum_match_iou <= 1.0
        ):
            raise ValueError("minimum_match_iou must be finite and in [0, 1]")

    def score(
        self,
        miner_items: Sequence[PerImageAnnotationItem],
        golden: GoldenImage,
    ) -> FidelityComponents:
        gt_annotations: Sequence[GoldenAnnotation] = golden.annotations
        img_area = float(max(1, golden.width * golden.height))
        from template.miner.geometry import (
            CARBON_WEIGHT_MULTIPLIERS,
            canonical_carbon_class,
        )

        miner_net_weight = 0.0
        for item in miner_items:
            # The current Golden schema has validator boxes, not masks. Score
            # box area only; a miner-supplied polygon cannot stand in for mask
            # ground truth or change the score. Polygon fidelity requires a
            # separate mask-labeled Golden set and mask-IoU matching.
            box = item.bounding_box
            box_area = max(0.0, float(box[2] - box[0])) * max(
                0.0, float(box[3] - box[1])
            )
            c_cls = canonical_carbon_class(item.hazard_class)
            mult = CARBON_WEIGHT_MULTIPLIERS.get(c_cls, 1.0)
            miner_net_weight += (box_area / img_area) * mult
        miner_net_weight = round(miner_net_weight, 6)

        if not gt_annotations:
            if golden.classification_label:
                return self._score_classification_items(
                    miner_items, golden.classification_label, miner_net_weight
                )
            # Golden image has zero ground-truth hazards.
            if not miner_items:
                # Miner correctly reports "nothing here" — perfect fidelity.
                return FidelityComponents(
                    iou=1.0,
                    class_severity=1.0,
                    fidelity=1.0,
                    hallucination_penalty=1.0,
                    matched_count=0,
                    hallucinated_count=0,
                    ground_truth_count=0,
                    net_weight_agreement=1.0,
                    gt_net_weight=0.0,
                    miner_net_weight=0.0,
                )
            # Miner hallucinated detections on a clean image — penalise.
            penalty = self.hallucination_penalty ** len(miner_items)
            return FidelityComponents(
                iou=0.0,
                class_severity=0.0,
                fidelity=0.0,
                hallucination_penalty=float(penalty),
                matched_count=0,
                hallucinated_count=len(miner_items),
                ground_truth_count=0,
                net_weight_agreement=max(0.0, 1.0 - miner_net_weight),
                gt_net_weight=0.0,
                miner_net_weight=miner_net_weight,
            )

        gt_net_weight = 0.0
        for gt in gt_annotations:
            area = max(0.0, float(gt.bounding_box[2] - gt.bounding_box[0])) * max(0.0, float(gt.bounding_box[3] - gt.bounding_box[1]))
            c_cls = canonical_carbon_class(gt.hazard_class)
            mult = CARBON_WEIGHT_MULTIPLIERS.get(c_cls, 1.0)
            gt_net_weight += (area / img_area) * mult
        gt_net_weight = round(gt_net_weight, 6)

        weight_diff = abs(miner_net_weight - gt_net_weight)
        net_weight_agreement = round(max(0.0, 1.0 - (weight_diff / max(gt_net_weight, 0.05))), 6)

        # Greedy 1-1 matching: for each ground truth box, find best miner item
        # by IoU; track which miner items matched something.
        used_miner_idx: set[int] = set()
        per_match_iou: List[float] = []
        per_match_class: List[float] = []

        for gt in gt_annotations:
            best_idx = -1
            best_iou = 0.0
            for idx, item in enumerate(miner_items):
                if idx in used_miner_idx:
                    continue
                iou = iou_xyxy(item.bounding_box, gt.bounding_box)
                if iou >= self.minimum_match_iou and iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx < 0:
                # Ground truth had no match -- counts as a miss (zero contribution).
                per_match_iou.append(0.0)
                per_match_class.append(0.0)
                continue
            matched = miner_items[best_idx]
            used_miner_idx.add(best_idx)
            per_match_iou.append(best_iou)
            per_match_class.append(_class_match_score(matched, gt))

        n_gt = max(1, len(gt_annotations))
        iou_avg = sum(per_match_iou) / n_gt
        class_avg = sum(per_match_class) / n_gt

        hallucinated = max(0, len(miner_items) - len(used_miner_idx))
        # Bounded precision penalty: penalize over-detection proportionately without annihilating
        # high-recall models on dense high-resolution forestry chips.
        if hallucinated > 0 and len(used_miner_idx) > 0:
            precision_factor = len(used_miner_idx) / (len(used_miner_idx) + 0.25 * hallucinated)
            penalty = max(0.20, precision_factor)
        elif len(used_miner_idx) == 0:
            penalty = 0.0
        else:
            penalty = 1.0

        fidelity_raw = (
            self.iou_weight * iou_avg
            + self.class_weight * class_avg
        )
        # Factor in net carbon weight agreement (weight cross-verification)
        coverage_factor = 0.70 + 0.30 * net_weight_agreement
        fidelity = max(0.0, min(1.0, fidelity_raw * penalty * coverage_factor))

        return FidelityComponents(
            iou=float(iou_avg),
            class_severity=float(class_avg),
            fidelity=float(fidelity),
            hallucination_penalty=float(penalty),
            matched_count=len(used_miner_idx),
            hallucinated_count=int(hallucinated),
            ground_truth_count=int(len(gt_annotations)),
            net_weight_agreement=float(net_weight_agreement),
            gt_net_weight=float(gt_net_weight),
            miner_net_weight=float(miner_net_weight),
        )

    def _score_classification_items(
        self,
        miner_items: Sequence[PerImageAnnotationItem],
        classification_label: str,
        miner_net_weight: float,
    ) -> FidelityComponents:
        """Score class-only Golden chips without inventing spatial boxes."""
        class_scores = [
            _class_label_match_score(item.hazard_class, classification_label)
            for item in miner_items
        ]
        matched_count = sum(score > 0.0 for score in class_scores)
        hallucinated_count = len(miner_items) - matched_count
        class_avg = sum(class_scores) / max(1, len(class_scores))
        if hallucinated_count and matched_count:
            penalty = max(
                0.20,
                matched_count / (matched_count + 0.25 * hallucinated_count),
            )
        elif not matched_count:
            penalty = 0.0
        else:
            penalty = 1.0
        fidelity = max(0.0, min(1.0, class_avg * penalty))
        return FidelityComponents(
            iou=0.0,
            class_severity=float(class_avg),
            fidelity=float(fidelity),
            hallucination_penalty=float(penalty),
            matched_count=int(matched_count),
            hallucinated_count=int(hallucinated_count),
            ground_truth_count=1,
            net_weight_agreement=1.0,
            gt_net_weight=0.0,
            miner_net_weight=float(miner_net_weight),
            # A class label cannot validate the miner's box placement. Keep its
            # class metric for reliability diagnostics, but do not turn it into
            # an annotation reward without localized Golden geometry.
            rewardable=False,
        )



def _class_match_score(
    item: PerImageAnnotationItem, gt: GoldenAnnotation
) -> float:
    return _class_label_match_score(item.hazard_class, gt.hazard_class)


def _class_label_match_score(miner_label: str, gt_label: str) -> float:
    from template.miner.geometry import canonical_annotation_class

    miner_c = canonical_annotation_class(miner_label)
    gt_c = canonical_annotation_class(gt_label)
    # Aliases are canonicalized above. Distinct ontology classes get no class
    # credit; semantic proximity is not a substitute for a reviewed label.
    return 1.0 if miner_c == gt_c else 0.0


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

        from template.miner.geometry import canonical_annotation_class

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
                        best_peer_cls = canonical_annotation_class(peer_item.hazard_class or "")
            ious.append(best_iou)
            if best_peer_cls:
                peer_classes.append(best_peer_cls)

        mean_iou = sum(ious) / max(1, len(ious))
        miner_classes = [
            canonical_annotation_class(item.hazard_class or "") for item in miner_items
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
    golden_missing_count: int = 0
    annotated_image_ids: List[str] = field(default_factory=list)
    class_weights: Dict[str, float] = field(default_factory=dict)
    class_f1: Dict[str, float] = field(default_factory=dict)
    class_severity_accuracy: Dict[str, float] = field(default_factory=dict)
    class_ece: Dict[str, float] = field(default_factory=dict)
    localization_iou_mean: float = 0.0

    def average_score(self, *, golden_missing_penalty: float | None = None) -> float:
        """Mean fidelity; missing Golden rows are already explicit zero scores.

        ``golden_missing_penalty`` remains accepted for state/config
        compatibility, but a second multiplicative penalty over-counts those
        zero rows and compounds with the per-image precision penalty.
        """
        scores = list(self.fidelity_scores_by_image_id.values())
        if not scores:
            base = 0.0
        else:
            base = float(sum(scores) / len(scores))
        return float(max(0.0, min(1.0, base)))

    def hallucination_multiplier(self, per_event_penalty: float) -> float:
        """Global multiplier applied to the miner's annotation score based on
        the total number of hallucinations observed on Golden images."""
        if self.total_hallucinations <= 0:
            return 1.0
        n_gt = max(1, self.total_ground_truth)
        rel_hallucinated = min(5.0, self.total_hallucinations / n_gt)
        return float(per_event_penalty ** rel_hallucinated)

    def weight_for_class(self, hazard_class: str, *, epsilon: float = 1e-4) -> float:
        from template.miner.geometry import canonical_annotation_class

        key = canonical_annotation_class(hazard_class)
        if not key:
            return float(epsilon)
        return float(max(epsilon, self.class_weights.get(key, epsilon)))


@dataclass
class _ReliabilityAccumulator:
    decay: float = 0.95
    epsilon: float = 1e-4
    minimum_match_iou: float = 0.50
    tp: Dict[int, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    fp: Dict[int, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    fn: Dict[int, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    severity_ok: Dict[int, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    severity_total: Dict[int, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    class_iou_sum: Dict[int, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    class_iou_count: Dict[int, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    iou_sum: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    iou_count: Dict[int, float] = field(default_factory=lambda: defaultdict(float))

    def _decay_uid(self, uid: int) -> None:
        for bucket in (
            self.tp, self.fp, self.fn, self.severity_ok, self.severity_total,
            self.class_iou_sum, self.class_iou_count,
        ):
            for key in list(bucket[uid].keys()):
                bucket[uid][key] *= self.decay
        self.iou_sum[uid] *= self.decay
        self.iou_count[uid] *= self.decay

    def decay_all(self) -> None:
        """Apply exponential decay once per round to all tracked UIDs."""
        uids = set(self.tp.keys()) | set(self.fp.keys()) | set(self.fn.keys())
        for uid in uids:
            self._decay_uid(uid)

    def reset_uid(self, uid: int) -> None:
        """Discard all validation history associated with a replaced hotkey."""
        for bucket in (
            self.tp, self.fp, self.fn, self.severity_ok, self.severity_total,
            self.class_iou_sum, self.class_iou_count, self.iou_sum, self.iou_count,
        ):
            bucket.pop(uid, None)

    def to_jsonable(self) -> dict:
        return {
            "tp": {str(uid): dict(classes) for uid, classes in self.tp.items()},
            "fp": {str(uid): dict(classes) for uid, classes in self.fp.items()},
            "fn": {str(uid): dict(classes) for uid, classes in self.fn.items()},
            "severity_ok": {str(uid): dict(classes) for uid, classes in self.severity_ok.items()},
            "severity_total": {str(uid): dict(classes) for uid, classes in self.severity_total.items()},
            "class_iou_sum": {str(uid): dict(classes) for uid, classes in self.class_iou_sum.items()},
            "class_iou_count": {str(uid): dict(classes) for uid, classes in self.class_iou_count.items()},
            "iou_sum": {str(uid): float(v) for uid, v in self.iou_sum.items()},
            "iou_count": {str(uid): float(v) for uid, v in self.iou_count.items()},
        }

    @classmethod
    def from_jsonable(cls, payload: dict) -> _ReliabilityAccumulator:
        acc = cls()
        for field_name in (
            "tp", "fp", "fn", "severity_ok", "severity_total",
            "class_iou_sum", "class_iou_count",
        ):
            if field_name in payload:
                for uid_str, classes in payload[field_name].items():
                    for cls_name, val in classes.items():
                        getattr(acc, field_name)[int(uid_str)][cls_name] = float(val)
        for field_name in ("iou_sum", "iou_count"):
            if field_name in payload:
                for uid_str, val in payload[field_name].items():
                    getattr(acc, field_name)[int(uid_str)] = float(val)
        return acc

    def update(
        self,
        uid: int,
        miner_items: Sequence[PerImageAnnotationItem],
        golden: GoldenImage,
    ) -> None:
        from template.miner.geometry import canonical_annotation_class

        if golden.classification_label:
            target = canonical_annotation_class(golden.classification_label)
            best_item = None
            best_score = 0.0
            for item in miner_items:
                match_score = _class_label_match_score(
                    item.hazard_class, golden.classification_label
                )
                if match_score > best_score:
                    best_score = match_score
                    best_item = item
            if best_item is None:
                self.fn[uid][target] += 1.0
            else:
                predicted = canonical_annotation_class(best_item.hazard_class)
                self.tp[uid][predicted] += best_score
                self.fp[uid][predicted] += 1.0 - best_score
                self.fn[uid][target] += 1.0 - best_score
                for item in miner_items:
                    item_class = canonical_annotation_class(item.hazard_class)
                    if item_class != predicted:
                        self.fp[uid][item_class] += 1.0
            return

        used_miner_idx: set[int] = set()
        gt_annotations: Sequence[GoldenAnnotation] = golden.annotations
        if not gt_annotations and not miner_items:
            self.tp[uid]["_background"] += 1.0
            return
        if not gt_annotations:
            self.fn[uid]["_background"] += 1.0
        for gt in gt_annotations:
            gt_class = canonical_annotation_class(gt.hazard_class)
            best_idx = -1
            best_iou = 0.0
            for idx, item in enumerate(miner_items):
                if idx in used_miner_idx:
                    continue
                iou = iou_xyxy(item.bounding_box, gt.bounding_box)
                if iou >= self.minimum_match_iou and iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx < 0:
                self.fn[uid][gt_class] += 1.0
                self.severity_total[uid][gt_class] += 1.0
                continue

            used_miner_idx.add(best_idx)
            item = miner_items[best_idx]
            pred_class = canonical_annotation_class(item.hazard_class)
            self.iou_sum[uid] += best_iou
            self.iou_count[uid] += 1.0
            if pred_class == gt_class:
                self.tp[uid][gt_class] += 1.0
                self.severity_ok[uid][gt_class] += 1.0
                self.class_iou_sum[uid][gt_class] += best_iou
                self.class_iou_count[uid][gt_class] += 1.0
            else:
                self.fp[uid][pred_class] += 1.0
                self.fn[uid][gt_class] += 1.0
            self.severity_total[uid][gt_class] += 1.0

        for idx, item in enumerate(miner_items):
            if idx in used_miner_idx:
                continue
            pred_class = canonical_annotation_class(item.hazard_class)
            self.fp[uid][pred_class] += 1.0

    def finalize_uid(self, uid: int) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], float]:
        all_classes = set(self.tp[uid].keys()) | set(self.fp[uid].keys()) | set(self.fn[uid].keys())
        class_weights: Dict[str, float] = {}
        class_f1: Dict[str, float] = {}
        class_sev: Dict[str, float] = {}
        class_ece: Dict[str, float] = {}
        for cls in all_classes:
            tp = float(self.tp[uid].get(cls, 0.0))
            fp = float(self.fp[uid].get(cls, 0.0))
            fn = float(self.fn[uid].get(cls, 0.0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            sev_total = float(self.severity_total[uid].get(cls, 0.0))
            sev_acc = float(self.severity_ok[uid].get(cls, 0.0) / sev_total) if sev_total > 0 else 0.0
            evidence = tp + fp + fn
            evidence_factor = evidence / (evidence + 5.0) if evidence > 0.0 else 0.0
            class_iou_n = float(self.class_iou_count[uid].get(cls, 0.0))
            class_iou = (
                float(self.class_iou_sum[uid].get(cls, 0.0) / class_iou_n)
                if class_iou_n > 0.0 else 0.0
            )
            localization_factor = 0.5 + 0.5 * class_iou
            # A single lucky Golden match cannot manufacture full voting
            # reliability. Shrink F1 until the class has repeated evidence and
            # discount weakly localized positives as well.
            weight = max(self.epsilon, f1 * evidence_factor * localization_factor)
            class_weights[cls] = float(weight)
            class_f1[cls] = float(f1)
            class_sev[cls] = float(sev_acc)
            class_ece[cls] = 0.0
        loc_mean = float(self.iou_sum[uid] / self.iou_count[uid]) if self.iou_count[uid] > 0 else 0.0
        return class_weights, class_f1, class_sev, class_ece, loc_mean


def evaluate_round_annotations(
    *,
    corpus: ImageCorpus,
    annotations_by_uid: Mapping[int, Mapping[str, Sequence[PerImageAnnotationItem]]],
    fidelity_scorer: AnnotationFidelityScorer,
    consensus_scorer: ConsensusScorer,
    hallucination_penalty: float,
    golden_missing_penalty: float = 0.0,
    reliability: _ReliabilityAccumulator | None = None,
    expected_golden_ids_by_uid: Mapping[int, Sequence[str]] | None = None,
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

    if reliability is None:
        reliability = _ReliabilityAccumulator()
    else:
        reliability.decay_all()

    results: Dict[int, PerMinerAnnotationScore] = {}
    for uid, by_image in annotations_by_uid.items():
        score = PerMinerAnnotationScore(uid=uid)
        for image_id, items in by_image.items():
            score.annotated_image_ids.append(image_id)
            golden = corpus.golden_lookup(image_id)
            if golden is not None:
                comp = fidelity_scorer.score(items, golden)
                if comp.rewardable:
                    score.fidelity_scores_by_image_id[image_id] = comp.fidelity
                score.fidelity_components_by_image_id[image_id] = comp
                score.total_hallucinations += comp.hallucinated_count
                score.total_ground_truth += comp.ground_truth_count
                if comp.rewardable:
                    reliability.update(uid, items, golden)
            else:
                peers = {
                    other_uid: peer_items
                    for other_uid, peer_items in per_image_peer_items.get(image_id, {}).items()
                    if other_uid != uid
                }
                comp = consensus_scorer.score(items, peers)
                score.consensus_scores_by_image_id[image_id] = comp.consensus
                score.consensus_components_by_image_id[image_id] = comp

        if expected_golden_ids_by_uid is None:
            expected_golden = {g.image_id for g in corpus.golden_images()}
        else:
            expected_golden = {
                image_id for image_id in expected_golden_ids_by_uid.get(uid, ())
            }
        submitted_rewardable = set(score.fidelity_scores_by_image_id)
        for image_id in expected_golden:
            golden = corpus.golden_lookup(image_id)
            # Class-only rows have no spatial target and are excluded from both
            # numerator and denominator whether the miner returns a row or not.
            if golden is not None and golden.classification_label:
                continue
            if image_id in submitted_rewardable:
                continue
            score.golden_missing_count += 1
            score.fidelity_scores_by_image_id[image_id] = 0.0
            _log.warning(
                "event=golden_annotation_missing uid=%s image_id=%s...",
                uid,
                image_id[:16],
            )
        results[uid] = score
    for uid, score in results.items():
        w, f1, sev, ece, loc = reliability.finalize_uid(uid)
        score.class_weights = w
        score.class_f1 = f1
        score.class_severity_accuracy = sev
        score.class_ece = ece
        score.localization_iou_mean = loc
    return results
