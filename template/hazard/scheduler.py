from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from template.protocol import DatasetPartition
from template.utils.uids import get_random_uids

Cohort = Literal["training", "exploration", "verification", "promotion", "stability"]


@dataclass(frozen=True)
class CohortSelection:
    cohort: Cohort
    partition: DatasetPartition
    uids: np.ndarray
    task_type: Literal["training", "inference", "verification"]


class CohortScheduler:
    """
    Selects miner cohorts and challenge partitions for scalable evaluation.
    """

    def __init__(self, seed: int = 13):
        self.random = random.Random(seed)

    def select(self, validator) -> CohortSelection:
        cohort = self._choose_cohort(step=validator.step)
        self_uid = getattr(validator, "uid", None)
        exclude = [int(self_uid)] if self_uid is not None and int(self_uid) >= 0 else None
        uids = get_random_uids(validator, k=validator.config.neuron.sample_size, exclude=exclude)
        partition, task_type = self._cohort_target(cohort)
        return CohortSelection(
            cohort=cohort,
            partition=partition,
            uids=uids,
            task_type=task_type,
        )

    def _choose_cohort(self, step: int) -> Cohort:
        # Bias toward training rounds so training ledgers are populated during active validation windows.
        phase = step % 10
        if phase in (0, 1, 2, 3, 4, 5):
            return "training"
        if phase == 6:
            return "exploration"
        if phase == 7:
            return "verification"
        if phase == 8:
            return "promotion"
        return "stability"

    @staticmethod
    def _cohort_target(cohort: Cohort) -> tuple[DatasetPartition, Literal["training", "inference", "verification"]]:
        mapping: Dict[Cohort, tuple[DatasetPartition, Literal["training", "inference", "verification"]]] = {
            "training": ("training_pool", "training"),
            "exploration": ("hidden_eval", "inference"),
            "verification": ("golden", "verification"),
            "promotion": ("promotion", "inference"),
            "stability": ("replay", "verification"),
        }
        return mapping[cohort]

