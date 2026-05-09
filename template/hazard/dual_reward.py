"""
Final on-chain weight formula for the dual-flywheel subnet.

For every miner uid in the round, combine three signals:

  weight = alpha * annotation_score
         + beta  * model_accuracy_score
         + gamma * adoption_bonus

with a global hallucination multiplier that compresses the annotation
score by ``hallucination_penalty`` per hallucinated Golden annotation.

``adoption_bonus`` is the share of image_ids in the round whose winning
annotation came from this miner (normalized to [0, 1]). Ties and missing
data degrade gracefully but never silently: if the miner returned nothing
in this round, all three components are zero and the weight is zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from template.hazard.annotation_eval import PerMinerAnnotationScore
from template.hazard.dataset_assembler import AdoptionLedger, WinningAnnotation
from template.hazard.model_eval import ModelAccuracyComponents


@dataclass(frozen=True)
class DualFlywheelBreakdown:
    """Per-miner breakdown returned alongside the final weight."""

    uid: int
    annotation_score: float
    model_accuracy_score: float
    adoption_bonus: float
    hallucination_multiplier: float
    final_score: float
    fidelity_image_ids: int
    consensus_image_ids: int
    adopted_image_ids_round: int
    adopted_image_ids_total: int


@dataclass
class DualFlywheelRewardComposer:
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2
    hallucination_penalty_per_event: float = 0.5

    def compose(
        self,
        *,
        uids: Sequence[int],
        annotation_scores: Mapping[int, PerMinerAnnotationScore],
        model_accuracy: Mapping[int, ModelAccuracyComponents],
        ledger: AdoptionLedger,
        round_winners: Sequence[WinningAnnotation],
    ) -> tuple[np.ndarray, list[DualFlywheelBreakdown]]:
        if abs(self.alpha + self.beta + self.gamma - 1.0) > 1e-6:
            raise ValueError(
                f"alpha+beta+gamma must equal 1.0 (got {self.alpha + self.beta + self.gamma:.6f})"
            )

        round_share = ledger.round_share()
        winners_by_uid: Dict[int, int] = {}
        for w in round_winners:
            winners_by_uid[w.chosen_uid] = winners_by_uid.get(w.chosen_uid, 0) + 1

        rewards: list[float] = []
        breakdowns: list[DualFlywheelBreakdown] = []
        for uid in uids:
            score = annotation_scores.get(uid)
            model = model_accuracy.get(uid)
            base_annotation = score.average_score() if score is not None else 0.0
            hallucination_mult = (
                score.hallucination_multiplier(self.hallucination_penalty_per_event)
                if score is not None
                else 1.0
            )
            annotation_score = float(max(0.0, min(1.0, base_annotation * hallucination_mult)))
            model_score = float(model.overall_score) if model is not None else 0.0
            adoption_bonus = float(round_share.get(uid, 0.0))

            final = (
                self.alpha * annotation_score
                + self.beta * model_score
                + self.gamma * adoption_bonus
            )
            final = float(max(0.0, min(1.0, final)))

            rewards.append(final)
            breakdowns.append(
                DualFlywheelBreakdown(
                    uid=int(uid),
                    annotation_score=annotation_score,
                    model_accuracy_score=model_score,
                    adoption_bonus=adoption_bonus,
                    hallucination_multiplier=float(hallucination_mult),
                    final_score=final,
                    fidelity_image_ids=len(score.fidelity_scores_by_image_id) if score else 0,
                    consensus_image_ids=len(score.consensus_scores_by_image_id) if score else 0,
                    adopted_image_ids_round=int(winners_by_uid.get(uid, 0)),
                    adopted_image_ids_total=int(ledger.adoption_counts.get(uid, 0)),
                )
            )
        return np.asarray(rewards, dtype=np.float32), breakdowns
