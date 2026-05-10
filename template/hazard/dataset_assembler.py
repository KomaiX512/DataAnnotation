"""
Dataset assembler for the dual-flywheel subnet.

For each unique ``image_id`` annotated in a round, the assembler picks the
*best* miner annotation according to:

  - if the image is Golden-injected -> highest annotation fidelity score
  - otherwise (Annotation Pool)     -> highest consensus score

The selection per image_id becomes the canonical record in the subnet's
proprietary commercial dataset. We append every winning record to a JSONL
file under a configurable storage prefix (``file://``, ``r2://``, or
``s3://``). Per-uid adoption counts are tracked so the validator can pay
miners proportionally to how often their annotations were selected.
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

import bittensor as bt

from template.hazard.annotation_eval import PerMinerAnnotationScore
from template.hazard.image_corpus import ImageCorpus
from template.protocol import (
    ImageAnnotationDocument,
    PerImageAnnotationItem,
    R2AccessCredentials,
)


@dataclass(frozen=True)
class WinningAnnotation:
    """One image_id's chosen annotation, ready for the commercial dataset."""

    image_id: str
    chosen_uid: int
    score: float
    is_golden: bool
    image_url: str
    width: int
    height: int
    items: Sequence[PerImageAnnotationItem]
    miner_hotkey: str
    model_version: str
    timestamp: str

    def to_jsonable(self) -> dict:
        return {
            "image_id": self.image_id,
            "chosen_uid": int(self.chosen_uid),
            "score": float(self.score),
            "selection_lane": "golden_fidelity" if self.is_golden else "consensus",
            "image_url": self.image_url,
            "width": int(self.width),
            "height": int(self.height),
            "miner_hotkey": self.miner_hotkey,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "annotations": [
                {
                    "hazard_class": item.hazard_class,
                    "bounding_box": list(item.bounding_box),
                    "severity": item.severity,
                    "confidence": float(item.confidence),
                    "reasoning_chain": item.reasoning_chain,
                    "osha_reference": item.osha_reference,
                }
                for item in self.items
            ],
        }


@dataclass
class AdoptionLedger:
    """Tracks per-uid adoption counts (and recent history) across rounds."""

    adoption_counts: Dict[int, int] = field(default_factory=dict)
    last_round_counts: Dict[int, int] = field(default_factory=dict)
    rounds_observed: int = 0

    def record_round(self, winners: Sequence[WinningAnnotation]) -> None:
        last: Dict[int, int] = {}
        for w in winners:
            self.adoption_counts[w.chosen_uid] = self.adoption_counts.get(w.chosen_uid, 0) + 1
            last[w.chosen_uid] = last.get(w.chosen_uid, 0) + 1
        self.last_round_counts = last
        self.rounds_observed += 1

    def adoption_share(self) -> Dict[int, float]:
        total = float(sum(self.adoption_counts.values())) or 1.0
        return {uid: count / total for uid, count in self.adoption_counts.items()}

    def round_share(self) -> Dict[int, float]:
        total = float(sum(self.last_round_counts.values())) or 1.0
        return {uid: count / total for uid, count in self.last_round_counts.items()}

    def to_jsonable(self) -> dict:
        return {
            "adoption_counts": {str(k): int(v) for k, v in self.adoption_counts.items()},
            "last_round_counts": {str(k): int(v) for k, v in self.last_round_counts.items()},
            "rounds_observed": int(self.rounds_observed),
        }

    @classmethod
    def from_jsonable(cls, payload: dict) -> "AdoptionLedger":
        ledger = cls()
        ledger.adoption_counts = {int(k): int(v) for k, v in payload.get("adoption_counts", {}).items()}
        ledger.last_round_counts = {
            int(k): int(v) for k, v in payload.get("last_round_counts", {}).items()
        }
        ledger.rounds_observed = int(payload.get("rounds_observed", 0))
        return ledger


@dataclass
class DatasetAssembler:
    """Selects winners per image_id and exports the commercial dataset."""

    corpus: ImageCorpus
    storage_prefix: str  # file://..., r2://bucket/prefix/, s3://bucket/prefix/
    ledger: AdoptionLedger = field(default_factory=AdoptionLedger)

    def assemble(
        self,
        *,
        per_miner_scores: Mapping[int, PerMinerAnnotationScore],
        annotations_by_uid: Mapping[int, Mapping[str, Sequence[PerImageAnnotationItem]]],
        miner_hotkeys: Mapping[int, str],
        model_versions: Mapping[int, str],
        timestamps: Mapping[int, str],
    ) -> List[WinningAnnotation]:
        """Pick the best annotation per image_id across all miners."""

        all_image_ids: set[str] = set()
        for by_image in annotations_by_uid.values():
            all_image_ids.update(by_image.keys())

        winners: List[WinningAnnotation] = []
        for image_id in sorted(all_image_ids):
            best: Optional[WinningAnnotation] = None
            best_score = -1.0
            is_golden = self.corpus.is_golden(image_id)
            for uid, by_image in annotations_by_uid.items():
                items = by_image.get(image_id)
                if not items:
                    continue
                miner_score = per_miner_scores.get(uid)
                if miner_score is None:
                    continue
                if is_golden:
                    score = miner_score.fidelity_scores_by_image_id.get(image_id, 0.0)
                else:
                    score = miner_score.consensus_scores_by_image_id.get(image_id, 0.0)
                if score <= best_score:
                    continue
                width, height = self._image_dims(image_id)
                image_url = self._image_url(image_id)
                best = WinningAnnotation(
                    image_id=image_id,
                    chosen_uid=int(uid),
                    score=float(score),
                    is_golden=bool(is_golden),
                    image_url=image_url,
                    width=int(width),
                    height=int(height),
                    items=list(items),
                    miner_hotkey=str(miner_hotkeys.get(uid, "")),
                    model_version=str(model_versions.get(uid, "")),
                    timestamp=str(timestamps.get(uid, "")),
                )
                best_score = score
            if best is not None and best_score > 0.0:
                winners.append(best)

        self.ledger.record_round(winners)
        bt.logging.info(
            f"event=dataset_assembled images={len(winners)} "
            f"unique_winners={len({w.chosen_uid for w in winners})}"
        )
        return winners

    def export(
        self,
        winners: Sequence[WinningAnnotation],
        *,
        round_id: str,
        commercial_r2_credentials: Optional[R2AccessCredentials] = None,
    ) -> str:
        """Append the round's winners to the commercial dataset and return its URI.

        Golden-track rows (``is_golden``) are used for scoring and adoption only;
        they are never written to the commercial JSONL so the hidden Golden Set
        cannot leak to customers.
        """
        if not winners:
            bt.logging.info("event=dataset_export_skip reason=no_winners")
            return ""

        commercial = [w for w in winners if not w.is_golden]
        skipped = len(winners) - len(commercial)
        if skipped:
            bt.logging.info(
                "event=dataset_export_golden_filtered count=%d commercial=%d"
                % (skipped, len(commercial))
            )
        if not commercial:
            bt.logging.info("event=dataset_export_skip reason=no_commercial_rows")
            return ""

        payload_lines = "\n".join(
            json.dumps(w.to_jsonable(), sort_keys=True) for w in commercial
        )
        body = (payload_lines + "\n").encode("utf-8")

        parsed = urlparse(self.storage_prefix or "")
        if parsed.scheme == "file":
            return self._export_local(body, parsed, round_id)
        if parsed.scheme in ("r2", "s3"):
            if commercial_r2_credentials is None:
                raise ValueError(
                    "commercial_r2_credentials are required to export to "
                    f"{parsed.scheme}:// storage."
                )
            return self._export_object_storage(
                body, parsed, round_id, commercial_r2_credentials
            )
        raise ValueError(
            f"Unsupported commercial dataset storage scheme: {self.storage_prefix!r}"
        )

    # ------------------------------------------------------------------ helpers
    def _image_dims(self, image_id: str) -> tuple[int, int]:
        record = self.corpus.golden_lookup(image_id)
        if record is not None:
            return record.width, record.height
        for unl in self.corpus.annotation_images():
            if unl.image_id == image_id:
                return unl.width, unl.height
        return 0, 0

    def _image_url(self, image_id: str) -> str:
        record = self.corpus.golden_lookup(image_id)
        if record is not None:
            return record.image_url
        for unl in self.corpus.annotation_images():
            if unl.image_id == image_id:
                return unl.image_url
        local = self.corpus.known_image_path(image_id)
        return local.as_uri() if local is not None else ""

    def _export_local(self, body: bytes, parsed, round_id: str) -> str:
        directory = Path(parsed.path)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"commercial-dataset-{round_id}.jsonl"
        target.write_bytes(body)
        master = directory / "commercial-dataset.jsonl"
        with master.open("ab") as handle:
            handle.write(body)
        bt.logging.info(
            f"event=dataset_export_local round={round_id} target={target} "
            f"master={master} bytes={len(body)}"
        )
        return target.as_uri()

    def _export_object_storage(
        self,
        body: bytes,
        parsed,
        round_id: str,
        creds: R2AccessCredentials,
    ) -> str:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise ImportError("boto3 is required for commercial dataset export.") from exc
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        if not bucket:
            raise ValueError(f"Storage prefix missing bucket: {self.storage_prefix}")
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        object_key = f"{prefix}commercial-dataset-{round_id}.jsonl"
        client = boto3.client(
            "s3",
            endpoint_url=creds.s3_endpoint,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            region_name="auto",
        )
        client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=body,
            ContentType="application/x-ndjson",
        )
        uri = f"{parsed.scheme}://{bucket}/{object_key}"
        bt.logging.info(
            f"event=dataset_export_remote round={round_id} uri={uri} bytes={len(body)}"
        )
        return uri
