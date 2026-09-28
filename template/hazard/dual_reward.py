"""
Final on-chain weight formula for the annotation-only subnet.

For every miner uid in the round:

  weight = alpha * annotation_score + (1 - alpha) * selection_or_adoption

Hallucinations are penalized in per-image fidelity. Missing Golden rows are
included as zero scores, so neither signal is multiplied a second time here.

``adoption_bonus`` is the share of image_ids in the round whose winning
annotation came from this miner (normalized to [0, 1]), but it is payable only
after a separate validator or human audit records ground-truth verification.
Peer agreement on an unlabeled image cannot establish that the object exists.
For vision-selected rows, the same secondary component uses the selected
public-image fraction. The two signals are reported separately and the larger
one supplies the secondary component; model selection never sets the
ground-truth verification flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from template.hazard.annotation_eval import PerMinerAnnotationScore
from template.hazard.dataset_assembler import AdoptionLedger, WinningAnnotation
from template.hazard.incentives import SELECTION_ELIGIBILITY_MIN_FIDELITY

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
    selection_contribution: float = 0.0


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
        selection_public_image_count: int = 0,
        selection_fidelity_by_uid: Mapping[int, float] | None = None,
        selection_caps_by_uid: Mapping[int, int] | None = None,
        selection_source_uid_by_image: Mapping[str, int] | None = None,
    ) -> tuple[np.ndarray, list[DualFlywheelBreakdown]]:
        round_share = ledger.round_contribution_share()
        has_ground_truth_verified_adoption = any(
            not winner.escalation_required
            and not winner.is_golden
            and winner.aggregation_method != "vision_model_selection_v1"
            and winner.ground_truth_verified
            for winner in round_winners
        )
        winners_by_uid: Dict[int, int] = {}
        selected_rows_by_uid: dict[int, list[WinningAnnotation]] = {}
        selection_fidelity_by_uid = selection_fidelity_by_uid or {}
        selection_caps_by_uid = selection_caps_by_uid or {}
        selection_source_uid_by_image = selection_source_uid_by_image or {}
        for w in round_winners:
            if w.escalation_required or w.is_golden:
                continue
            if w.ground_truth_verified and w.aggregation_method != "vision_model_selection_v1":
                winners_by_uid[w.chosen_uid] = winners_by_uid.get(w.chosen_uid, 0) + 1
            if w.aggregation_method == "vision_model_selection_v1" and w.selection_accepted:
                selected_rows_by_uid.setdefault(w.chosen_uid, []).append(w)

        # Selection rewards fail closed unless the validator supplies the
        # exact task caps. This second boundary rejects forged, duplicate, and
        # over-cap winner rows even if they bypassed the typed adapter.
        caps_valid = bool(selection_caps_by_uid) or not selected_rows_by_uid
        normalized_caps: dict[int, int] = {}
        valid_uids = {
            uid for uid in uids
            if isinstance(uid, int) and not isinstance(uid, bool) and uid >= 0
        }
        valid_public_count = (
            isinstance(selection_public_image_count, int)
            and not isinstance(selection_public_image_count, bool)
            and selection_public_image_count >= 0
        )
        public_image_count = selection_public_image_count if valid_public_count else 0
        try:
            if not valid_public_count:
                caps_valid = False
            for uid, cap in selection_caps_by_uid.items():
                if (
                    isinstance(uid, bool)
                    or not isinstance(uid, int)
                    or uid < 0
                    or isinstance(cap, bool)
                    or not isinstance(cap, int)
                    or cap < 0
                ):
                    caps_valid = False
                    break
                if uid not in valid_uids:
                    caps_valid = False
                    break
                normalized_caps[uid] = cap
            if selection_caps_by_uid and valid_public_count:
                cap_total = sum(normalized_caps.values())
                if cap_total not in (0, public_image_count):
                    caps_valid = False
        except (TypeError, ValueError):
            caps_valid = False

        selected_by_uid: Dict[int, int] = {}
        all_selected_image_ids = [
            row.image_id
            for rows in selected_rows_by_uid.values()
            for row in rows
        ]
        if len(all_selected_image_ids) != len(set(all_selected_image_ids)):
            caps_valid = False
        expected_sources: dict[str, int] = {}
        for uid, rows in selected_rows_by_uid.items():
            for row in rows:
                if not _valid_selected_reward_record(
                    row,
                    uid=uid,
                    task_fidelity=selection_fidelity_by_uid.get(uid),
                ):
                    caps_valid = False
                if not isinstance(row.image_id, str) or not row.image_id:
                    caps_valid = False
                elif row.image_id in expected_sources:
                    caps_valid = False
                else:
                    expected_sources[row.image_id] = uid

        supplied_sources: dict[str, int] = {}
        try:
            for image_id, uid in selection_source_uid_by_image.items():
                if (
                    not isinstance(image_id, str)
                    or not image_id
                    or isinstance(uid, bool)
                    or not isinstance(uid, int)
                    or uid not in valid_uids
                ):
                    caps_valid = False
                    break
                supplied_sources[image_id] = uid
        except (AttributeError, TypeError, ValueError):
            caps_valid = False
        if supplied_sources != expected_sources:
            caps_valid = False
        if len(expected_sources) > public_image_count:
            caps_valid = False
        if caps_valid:
            for uid, rows in selected_rows_by_uid.items():
                try:
                    fidelity = float(selection_fidelity_by_uid.get(uid, 0.0))
                except (TypeError, ValueError, OverflowError):
                    fidelity = 0.0
                cap = normalized_caps.get(uid, 0)
                image_ids = [row.image_id for row in rows]
                if (
                    math.isfinite(fidelity)
                    and fidelity >= SELECTION_ELIGIBILITY_MIN_FIDELITY
                    and len(rows) <= cap
                    and len(image_ids) == len(set(image_ids))
                ):
                    selected_by_uid[uid] = len(rows)

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

            selection_contribution = (
                float(selected_by_uid.get(uid, 0) / public_image_count)
                if public_image_count > 0 else 0.0
            )

            secondary_component = max(adoption_bonus, selection_contribution)
            final = self.alpha * annotation_score + (1.0 - self.alpha) * secondary_component
            final = float(max(0.0, min(1.0, final)))

            rewards.append(final)
            breakdowns.append(
                DualFlywheelBreakdown(
                    uid=int(uid),
                    annotation_score=annotation_score,
                    adoption_bonus=adoption_bonus,
                    selection_contribution=selection_contribution,
                    hallucination_multiplier=float(hallucination_mult),
                    final_score=final,
                    fidelity_image_ids=evidence_count,
                    consensus_image_ids=len(score.consensus_scores_by_image_id) if score else 0,
                    adopted_image_ids_round=int(winners_by_uid.get(uid, 0)),
                    adopted_image_ids_total=int(ledger.adoption_counts.get(uid, 0)),
                )
            )
        return np.asarray(rewards, dtype=np.float32), breakdowns


def _valid_selected_reward_record(
    row: WinningAnnotation, *, uid: int, task_fidelity: object
) -> bool:
    """Reject malformed or forged selected rows at the reward boundary."""
    if (
        row.aggregation_method != "vision_model_selection_v1"
        or row.selection_accepted is not True
        or row.is_golden
        or row.ground_truth_verified
        or row.escalation_required
        or isinstance(row.chosen_uid, bool)
        or not isinstance(row.chosen_uid, int)
        or row.chosen_uid != uid
        or not isinstance(row.image_id, str)
        or not row.image_id
        or isinstance(row.width, bool)
        or not isinstance(row.width, int)
        or row.width <= 0
        or isinstance(row.height, bool)
        or not isinstance(row.height, int)
        or row.height <= 0
    ):
        return False
    try:
        fidelity = float(task_fidelity)
        row_score = float(row.score)
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        not math.isfinite(fidelity)
        or not 0.0 <= fidelity <= 1.0
        or fidelity < SELECTION_ELIGIBILITY_MIN_FIDELITY
        or not math.isfinite(row_score)
        or not math.isclose(row_score, fidelity, rel_tol=1e-6, abs_tol=1e-7)
    ):
        return False
    if not isinstance(row.miner_contribution_scores, Mapping):
        return False
    if set(row.miner_contribution_scores) != {uid}:
        return False
    try:
        contribution = float(row.miner_contribution_scores[uid])
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(contribution) or contribution != 1.0:
        return False

    if not isinstance(row.accepted_objects, Sequence):
        return False
    for obj in row.accepted_objects:
        if not isinstance(obj.miner_votes, Sequence):
            return False
        if not isinstance(obj.accepted_hazard_class, str) or not obj.accepted_hazard_class.strip():
            return False
        box = obj.fused_bounding_box
        if box is None or len(box) != 4:
            return False
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            or x1 < 0.0
            or y1 < 0.0
            or x2 <= x1
            or y2 <= y1
            or x2 > row.width
            or y2 > row.height
        ):
            return False
        votes = obj.miner_votes
        if len(votes) != 1 or votes[0].miner_uid != uid:
            return False
    return True
