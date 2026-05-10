"""
Anti-plagiarism helpers: structure hashes for miner annotations and a registry
of first-claim subnet UIDs per model checkpoint hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import bittensor as bt

from template.protocol import PerImageAnnotationItem


def fingerprint_annotation_items(items: Sequence[PerImageAnnotationItem]) -> str:
    """Stable hash over sorted annotation rows (structure + reasoning text)."""

    rows = []
    for it in sorted(
        items,
        key=lambda x: (x.hazard_class.lower(), tuple(x.bounding_box), str(x.severity)),
    ):
        rows.append(
            {
                "hazard_class": it.hazard_class.strip().lower(),
                "bounding_box": [int(b) for b in it.bounding_box],
                "severity": str(it.severity),
                "confidence": round(float(it.confidence), 4),
                "reasoning_chain": it.reasoning_chain.strip(),
                "osha_reference": (it.osha_reference or "").strip(),
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
    """Within-round index: first UID wins for identical annotation structure per image."""

    _image_fingerprint_to_uid: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _full_fp_to_uid: Dict[str, int] = field(default_factory=dict)

    def check_and_register(
        self,
        uid: int,
        records: Mapping[str, Sequence[PerImageAnnotationItem]],
    ) -> Tuple[bool, str]:
        """Return (ok, reason). ``ok`` is False if this UID duplicates an earlier one."""

        full_fp = full_submission_fingerprint(records)
        prior_full = self._full_fp_to_uid.get(full_fp)
        if prior_full is not None and prior_full != uid:
            return False, (
                f"annotations payload matches uid {prior_full} (full-submission fingerprint)"
            )

        for image_id, items in records.items():
            fp = fingerprint_annotation_items(items)
            bucket = self._image_fingerprint_to_uid.get(image_id, {})
            owner = bucket.get(fp)
            if owner is not None and owner != uid:
                return False, (
                    f"duplicate annotation structure on image_id={image_id} "
                    f"(first uid={owner})"
                )

        if prior_full is None:
            self._full_fp_to_uid[full_fp] = uid
        for image_id, items in records.items():
            fp = fingerprint_annotation_items(items)
            bucket = self._image_fingerprint_to_uid.setdefault(image_id, {})
            if fp not in bucket:
                bucket[fp] = uid
        return True, ""


@dataclass
class ModelHashClaimRegistry:
    """Maps checkpoint content hash to the first subnet UID that produced it."""

    hash_to_uid: MutableMapping[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ModelHashClaimRegistry":
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            bt.logging.warning(f"event=model_hash_registry_load_failed path={path} err={exc}")
            return cls()
        raw = data.get("model_hash_first_uid") or {}
        mapping: Dict[str, int] = {}
        for k, v in raw.items():
            try:
                mapping[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return cls(hash_to_uid=mapping)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_hash_first_uid": dict(self.hash_to_uid)}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def uid_may_use_model_hash(self, uid: int, model_hash: str) -> Tuple[bool, str]:
        h = (model_hash or "").strip().lower()
        if not h:
            return False, "empty model hash"
        prior = self.hash_to_uid.get(h)
        if prior is None:
            self.hash_to_uid[h] = int(uid)
            return True, ""
        if int(prior) == int(uid):
            return True, ""
        return False, f"checkpoint hash already attributed to uid {prior}"
