"""Anti-plagiarism helpers for annotation-only miner submissions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

from template.protocol import PerImageAnnotationItem


def fingerprint_annotation_items(items: Sequence[PerImageAnnotationItem]) -> str:
    """Stable hash over sorted annotation rows."""

    rows = []
    for it in sorted(
        items,
        key=lambda x: (x.hazard_class.lower(), tuple(x.bounding_box)),
    ):
        rows.append(
            {
                "hazard_class": it.hazard_class.strip().lower(),
                "bounding_box": [float(b) for b in it.bounding_box],
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def full_submission_fingerprint(
    records: Mapping[str, Sequence[PerImageAnnotationItem]],
) -> str:
    """Hash of per-image fingerprints for the whole round payload."""

    parts = {iid: fingerprint_annotation_items(items) for iid, items in sorted(records.items())}
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class AnnotationDuplicateTracker:
    """Within-round similarity telemetry; annotation similarity never rejects a miner.

    Identical outputs can come from the same public model and do not prove
    copying. This tracker reports exact or near-identical nonempty outputs for
    audit, but leaves quality and corroboration decisions to independent
    Golden scoring and the object-vote quorum.
    """

    similarity_threshold: float = 0.95
    _image_submissions: Dict[
        str, List[Tuple[int, Counter]]
    ] = field(default_factory=dict)
    _full_fp_to_uid: Dict[str, int] = field(default_factory=dict)

    def check_and_register(
        self,
        uid: int,
        records: Mapping[str, Sequence[PerImageAnnotationItem]],
    ) -> Tuple[bool, str]:
        """Register a response and report similarity without disqualifying it."""

        has_nonempty = any(bool(items) for items in records.values())
        full_fp = full_submission_fingerprint(records) if has_nonempty else ""
        prior_full = self._full_fp_to_uid.get(full_fp) if has_nonempty else None
        reasons: List[str] = []
        if prior_full is not None and prior_full != uid:
            reasons.append(
                f"exact full response similarity with uid {prior_full}"
            )

        for image_id, items in records.items():
            # Empty outputs are common correct answers and carry no similarity
            # evidence. Do not let them suppress a later miner.
            if not items:
                continue
            signature = _coarse_annotation_signature(items)
            for prior_uid, prior_items in self._image_submissions.get(image_id, []):
                if prior_uid != uid and _signatures_are_near_identical(
                    signature, prior_items, self.similarity_threshold
                ):
                    reasons.append(
                        f"near-identical nonempty annotations on image_id={image_id} "
                        f"with uid {prior_uid}"
                    )
                    break

        if has_nonempty and prior_full is None:
            self._full_fp_to_uid[full_fp] = uid
        for image_id, items in records.items():
            if items:
                self._image_submissions.setdefault(image_id, []).append(
                    (uid, _coarse_annotation_signature(items))
                )
        return True, "; ".join(dict.fromkeys(reasons))


def _coarse_annotation_signature(items: Sequence[PerImageAnnotationItem]) -> Counter:
    from template.miner.geometry import canonical_annotation_class

    # Four-pixel bins catch small coordinate nudges with linear work per
    # annotation. Similarity is telemetry only, so quantization cannot reject.
    return Counter(
        (
            canonical_annotation_class(item.hazard_class),
            *(int(round(float(coord) / 4.0)) for coord in item.bounding_box),
        )
        for item in items
    )


def _signatures_are_near_identical(
    left: Counter, right: Counter, threshold: float
) -> bool:
    left_count = sum(left.values())
    right_count = sum(right.values())
    if not left_count or not right_count:
        return False
    shared_count = sum((left & right).values())
    return shared_count / max(left_count, right_count) >= threshold
