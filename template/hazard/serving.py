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

    def __init__(self, min_promotion_score: float = 0.75):
        self.min_promotion_score = min_promotion_score
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

    def top_models(self, limit: int = 5) -> List[ModelPromotionRecord]:
        ordered = sorted(self._records.values(), key=lambda item: item.score, reverse=True)
        return ordered[: max(0, limit)]


class CommercialServingGateway:
    """
    Represents the serving selection logic for external enterprise requests.
    """

    def __init__(self, promotion_registry: PromotionRegistry):
        self.promotion_registry = promotion_registry

    def select_model_hash(self) -> Optional[str]:
        top = self.promotion_registry.top_models(limit=1)
        return top[0].model_hash if top else None

