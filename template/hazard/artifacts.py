from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

from template.protocol import TrainingManifest


@dataclass(frozen=True)
class ArtifactVerificationResult:
    passed: bool
    score: float
    reason: str


class ArtifactRegistry:
    """
    Validator-owned registry for miner training commits and verification scores.
    """

    def __init__(self):
        self._latest_by_uid: Dict[int, TrainingManifest] = {}
        self._verification_by_uid: Dict[int, ArtifactVerificationResult] = {}

    def submit(self, uid: int, manifest: Optional[TrainingManifest]) -> None:
        if manifest is None:
            return
        self._latest_by_uid[uid] = manifest

    def verify(
        self,
        uid: int,
        observed_model_hash: Optional[str],
        *,
        golden_score: float = 0.0,
        expected_parent_hash: Optional[str] = None,
    ) -> ArtifactVerificationResult:
        manifest = self._latest_by_uid.get(uid)
        if manifest is None:
            result = ArtifactVerificationResult(
                passed=False,
                score=0.0,
                reason="No training manifest submitted.",
            )
            self._verification_by_uid[uid] = result
            return result

        required_fields = [
            manifest.parent_model_hash,
            manifest.candidate_model_hash,
            manifest.config_hash,
            manifest.dataset_lineage_hash,
            manifest.recipe_uri,
        ]
        if any(not field for field in required_fields):
            result = ArtifactVerificationResult(
                passed=False,
                score=0.0,
                reason="Incomplete training manifest fields.",
            )
            self._verification_by_uid[uid] = result
            return result

        if observed_model_hash is None:
            result = ArtifactVerificationResult(
                passed=False,
                score=0.0,
                reason="Missing model hash in miner response.",
            )
            self._verification_by_uid[uid] = result
            return result

        if observed_model_hash != manifest.candidate_model_hash:
            result = ArtifactVerificationResult(
                passed=False,
                score=0.0,
                reason="Response model hash does not match submitted candidate hash.",
            )
            self._verification_by_uid[uid] = result
            return result

        if expected_parent_hash is not None and manifest.parent_model_hash != expected_parent_hash:
            result = ArtifactVerificationResult(
                passed=False,
                score=0.0,
                reason="Training manifest parent hash does not match current baseline.",
            )
            self._verification_by_uid[uid] = result
            return result

        reproducibility = manifest.metrics.get("reproducibility_score", 0.0)
        uplift = manifest.metrics.get("uplift", 0.0)
        efficiency = manifest.metrics.get("efficiency", 0.0)
        composite = max(
            0.0,
            min(
                1.0,
                0.55 * golden_score
                + 0.2 * reproducibility
                + 0.15 * uplift
                + 0.1 * efficiency,
            ),
        )
        result = ArtifactVerificationResult(
            passed=composite >= 0.45,
            score=composite,
            reason="Verified" if composite >= 0.45 else "Verification score below threshold.",
        )
        self._verification_by_uid[uid] = result
        return result

    def get_latest_result(self, uid: int) -> ArtifactVerificationResult:
        return self._verification_by_uid.get(
            uid,
            ArtifactVerificationResult(
                passed=False,
                score=0.0,
                reason="Not verified yet.",
            ),
        )

    def latest_manifest(self, uid: int) -> Optional[TrainingManifest]:
        return self._latest_by_uid.get(uid)

    @staticmethod
    def _manifest_signature(manifest: TrainingManifest) -> str:
        payload = "|".join(
            [
                manifest.parent_model_hash,
                manifest.candidate_model_hash,
                manifest.config_hash,
                manifest.dataset_lineage_hash,
                manifest.recipe_uri,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

