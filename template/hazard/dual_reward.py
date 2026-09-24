"""
Final on-chain weight formula for the annotation-only subnet.

For every miner uid in the round:

  weight = alpha * annotation_score + (1 - alpha) * adoption_bonus

Hallucinations are penalized in per-image fidelity. Missing Golden rows are
included as zero scores, so neither signal is multiplied a second time here.

``adoption_bonus`` is the share of image_ids in the round whose winning
annotation came from this miner (normalized to [0, 1]), but it is payable only
after a separate validator or human audit records ground-truth verification.
Peer agreement on an unlabeled image cannot establish that the object exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from template.hazard.annotation_eval import PerMinerAnnotationScore
from template.hazard.dataset_assembler import AdoptionLedger, WinningAnnotation

_MIN_REWARDABLE_GOLDEN_IMAGES_FOR_ADOPTION = 3
_MIN_REWARDED_POSITIVE_GOLDEN_IMAGES = 3
_MIN_ADOPTION_ANNOTATION_SCORE = 0.75
_MIN_ADOPTION_LOCALIZATION_IOU = 0.75
_MIN_REWARDED_GOLDEN_IOU = 0.75


@dataclass(frozen=True)
class DualFlywheelBreakdown:
    """Per-miner breakdown returned alongside the final weight."""

    uid: int
    annotation_score: float
    adoption_bonus: float
    hallucination_multiplier: float
    final_score: float
    fidelity_image_ids: int
    consensus_image_ids: int
    adopted_image_ids_round: int
    adopted_image_ids_total: int


@dataclass
class DualFlywheelRewardComposer:
    alpha: float = 0.7
    # Retained for saved configuration compatibility; per-image precision is
    # now the single hallucination penalty.
    hallucination_penalty_per_event: float = 0.5
    # Retained for saved configuration compatibility; missing rows are zeros.
    golden_missing_penalty: float = 0.5
    min_rewarded_positive_goldens: int = _MIN_REWARDED_POSITIVE_GOLDEN_IMAGES
    min_rewarded_golden_iou: float = _MIN_REWARDED_GOLDEN_IOU
    min_rewarded_class_severity: float = 1.0
    min_rewarded_golden_fidelity: float = _MIN_ADOPTION_ANNOTATION_SCORE

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1] (got {self.alpha})")

    def compose(
        self,
        *,
        uids: Sequence[int],
        annotation_scores: Mapping[int, PerMinerAnnotationScore],
        ledger: AdoptionLedger,
        round_winners: Sequence[WinningAnnotation],
    ) -> tuple[np.ndarray, list[DualFlywheelBreakdown]]:
        round_share = ledger.round_contribution_share()
        has_ground_truth_verified_adoption = any(
            not winner.escalation_required
            and not winner.is_golden
            and winner.ground_truth_verified
            for winner in round_winners
        )
        winners_by_uid: Dict[int, int] = {}
        for w in round_winners:
            if w.escalation_required or w.is_golden:
                continue
            winners_by_uid[w.chosen_uid] = winners_by_uid.get(w.chosen_uid, 0) + 1

        rewards: list[float] = []
        breakdowns: list[DualFlywheelBreakdown] = []
        for uid in uids:
            score = annotation_scores.get(uid)
            evidence_count = len(score.fidelity_scores_by_image_id) if score else 0
            base_annotation = score.average_score() if score is not None else 0.0
            positive_components = []
            qualifying_positive_count = 0
            has_evaluation_components = bool(
                score is not None and score.fidelity_components_by_image_id
            )
            if has_evaluation_components:
                rewardable_components = [
                    component
                    for component in score.fidelity_components_by_image_id.values()
                    if component.rewardable
                ]
                positive_components = [
                    component
                    for component in rewardable_components
                    if component.ground_truth_count > 0
                ]
                qualifying_positive_count = sum(
                    component.matched_count > 0
                    and component.iou >= self.min_rewarded_golden_iou
                    and component.class_severity >= self.min_rewarded_class_severity
                    and component.fidelity >= self.min_rewarded_golden_fidelity
                    for component in positive_components
                )
                # True-negative examples can detect hallucinations, but an
                # empty response cannot earn positive annotation rewards from
                # them. Require independently localized, verified positive
                # Golden images in this round to qualify.
                if positive_components:
                    base_annotation = float(
                        sum(component.fidelity for component in positive_components)
                        / len(positive_components)
                    )
                else:
                    base_annotation = 0.0
                clean_false_positives = sum(
                    component.hallucinated_count
                    for component in rewardable_components
                    if component.ground_truth_count == 0
                )
                if clean_false_positives and positive_components:
                    base_annotation *= len(positive_components) / (
                        len(positive_components) + clean_false_positives
                    )
                if qualifying_positive_count < self.min_rewarded_positive_goldens:
                    base_annotation = 0.0
            elif score is not None:
                # Compatibility for trusted in-process callers that construct
                # score summaries directly. The validator scoring path always
                # supplies components and therefore uses the stricter gate.
                qualifying_positive_count = (
                    evidence_count
                    if base_annotation >= _MIN_ADOPTION_ANNOTATION_SCORE
                    and score.localization_iou_mean >= _MIN_ADOPTION_LOCALIZATION_IOU
                    else 0
                )
            # Hallucinations are already reflected in per-image precision.
            # Missing Golden records are already explicit zeros in average_score.
            # Applying either signal again here would double-penalize the same event.
            annotation_score = float(max(0.0, min(1.0, base_annotation)))
            hallucination_mult = 1.0
            adoption_qualified = bool(
                score is not None
                and (
                    len(positive_components)
                    if has_evaluation_components else evidence_count
                ) >= _MIN_REWARDABLE_GOLDEN_IMAGES_FOR_ADOPTION
                and qualifying_positive_count
                >= _MIN_REWARDED_POSITIVE_GOLDEN_IMAGES
                and base_annotation >= _MIN_ADOPTION_ANNOTATION_SCORE
                and score.localization_iou_mean >= _MIN_ADOPTION_LOCALIZATION_IOU
            )
            # Peer agreement on an unlabeled image is not independent evidence
            # that the object exists. The current pipeline has no post-hoc
            # ground-truth audit, so adoption credit stays zero until a separate
            # audit path sets ground_truth_verified on an accepted winner.
            adoption_bonus = (
                float(round_share.get(uid, 0.0))
                if adoption_qualified and has_ground_truth_verified_adoption
                else 0.0
            )

            final = self.alpha * annotation_score + (1.0 - self.alpha) * adoption_bonus
            final = float(max(0.0, min(1.0, final)))

            rewards.append(final)
            breakdowns.append(
                DualFlywheelBreakdown(
                    uid=int(uid),
                    annotation_score=annotation_score,
                    adoption_bonus=adoption_bonus,
                    hallucination_multiplier=float(hallucination_mult),
                    final_score=final,
                    fidelity_image_ids=evidence_count,
                    consensus_image_ids=len(score.consensus_scores_by_image_id) if score else 0,
                    adopted_image_ids_round=int(winners_by_uid.get(uid, 0)),
                    adopted_image_ids_total=int(ledger.adoption_counts.get(uid, 0)),
                )
            )
        return np.asarray(rewards, dtype=np.float32), breakdowns
