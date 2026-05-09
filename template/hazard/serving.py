from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelPromotionRecord:
    uid: int
    model_hash: str
    score: float
    step: int


class PromotionRegistry:
    """Tracks validator-promoted models for commercial serving."""

    def __init__(
        self,
        min_promotion_score: float = 0.75,
        recency_decay: float = 0.003,
        min_live_multiplier: float = 0.35,
    ):
        self.min_promotion_score = min_promotion_score
        self.recency_decay = max(0.0, recency_decay)
        self.min_live_multiplier = max(0.0, min(1.0, min_live_multiplier))
        self._records: Dict[int, ModelPromotionRecord] = {}

    def maybe_promote(self, uid: int, model_hash: Optional[str], score: float, step: int) -> bool:
        if model_hash is None or score < self.min_promotion_score:
            return False
        self._records[uid] = ModelPromotionRecord(
            uid=uid,
            model_hash=model_hash,
            score=score,
            step=step,
        )
        return True

    def top_models(self, *, current_step: int, limit: int = 5) -> List[ModelPromotionRecord]:
        ordered = sorted(
            self._records.values(),
            key=lambda item: self.live_score(item, current_step=current_step),
            reverse=True,
        )
        return ordered[: max(0, limit)]

    def live_score(self, record: ModelPromotionRecord, *, current_step: int) -> float:
        age = max(0, int(current_step) - int(record.step))
        recency_multiplier = max(
            self.min_live_multiplier,
            1.0 / (1.0 + self.recency_decay * float(age)),
        )
        return float(record.score) * recency_multiplier


class CommercialServingGateway:
    """
    Represents the serving selection logic for external enterprise requests.
    """

    def __init__(self, promotion_registry: PromotionRegistry):
        self.promotion_registry = promotion_registry

    def select_model_hash(self, *, current_step: int) -> Optional[str]:
        top = self.promotion_registry.top_models(current_step=current_step, limit=1)
        return top[0].model_hash if top else None

