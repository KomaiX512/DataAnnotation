"""
Golden Injection: deterministic strategy that mixes validator-only Golden
images into every miner annotation request without revealing labels.

Per the subnet specification, every annotation request contains
``annotation_request_size`` images of which exactly
``golden_injection_per_request`` come from the Golden Set (presented to the
miner as plain unlabeled images). The remaining slots are drawn from the
Annotation Pool. Image IDs (and only IDs) are recorded so the validator
later knows which miner annotations to score against ground truth versus
peer consensus.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Sequence, Set, Tuple

import bittensor as bt

from template.hazard.image_corpus import GoldenImage, ImageCorpus, UnlabeledImage


@dataclass(frozen=True)
class InjectionPlan:
    """Per-request plan describing which images go to the miner.

    ``ordered_images`` is a list of ``(image_id, image_url)`` pairs in the
    order they should appear in the synapse. ``golden_image_ids`` is the
    set of image IDs that are Golden-injected (validator-only knowledge).
    """

    ordered_images: Tuple[Tuple[str, str], ...]
    golden_image_ids: Tuple[str, ...]
    annotation_image_ids: Tuple[str, ...]


@dataclass
class GoldenInjector:
    """Stateless planner that picks request images while mixing in Golden ones.

    The constructor accepts the full ``ImageCorpus`` and a request size /
    golden injection size. ``build_plan`` returns a fresh :class:`InjectionPlan`
    with deterministic randomness driven by the supplied ``random.Random``.
    """

    corpus: ImageCorpus
    request_size: int
    golden_per_request: int
    _all_image_ids_for_round: Set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.request_size < 1:
            raise ValueError("request_size must be >= 1")
        if self.golden_per_request < 0:
            raise ValueError("golden_per_request must be >= 0")
        if self.golden_per_request > self.request_size:
            raise ValueError("golden_per_request cannot exceed request_size")

    def build_plan(self, rng: random.Random) -> InjectionPlan:
        """Construct an :class:`InjectionPlan` for one miner request."""

        golden = self.corpus.golden_images()
        unlabeled = self.corpus.annotation_images()
        if len(golden) < self.golden_per_request:
            raise RuntimeError(
                f"Golden Set has only {len(golden)} images; "
                f"cannot inject {self.golden_per_request} per request."
            )
        non_golden_needed = self.request_size - self.golden_per_request
        if len(unlabeled) < non_golden_needed:
            raise RuntimeError(
                f"Annotation pool has only {len(unlabeled)} images; "
                f"cannot fill {non_golden_needed} non-golden slots."
            )

        chosen_golden: Sequence[GoldenImage] = rng.sample(golden, self.golden_per_request)
        chosen_non_golden: Sequence[UnlabeledImage] = rng.sample(unlabeled, non_golden_needed)

        ordered: List[Tuple[str, str]] = []
        ordered.extend((g.image_id, g.image_url) for g in chosen_golden)
        ordered.extend((u.image_id, u.image_url) for u in chosen_non_golden)
        rng.shuffle(ordered)

        plan = InjectionPlan(
            ordered_images=tuple(ordered),
            golden_image_ids=tuple(g.image_id for g in chosen_golden),
            annotation_image_ids=tuple(u.image_id for u in chosen_non_golden),
        )
        bt.logging.debug(
            f"event=golden_injection_plan request_size={self.request_size} "
            f"golden={len(plan.golden_image_ids)} non_golden={len(plan.annotation_image_ids)}"
        )
        return plan
