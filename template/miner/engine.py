from __future__ import annotations

import hashlib
import time
import base64
import copy
from pathlib import Path
from typing import List

from dotenv import load_dotenv
import bittensor as bt
from template.hazard.r2_storage import load_r2_credentials_from_env
from template.hazard.vector_db import OshaVectorDatabase
from template.protocol import BoundingBox, HazardDetection, SeverityTier, TrainingManifest
from template.miner.training import TrainingPipeline, TrainingSettings


class HazardMinerEngine:
    """
    Miner-side deterministic hazard inference and training manifest responder.
    """

    def __init__(self, config=None):
        load_dotenv()
        self.osha_db = OshaVectorDatabase.default()
        workspace = Path(
            getattr(getattr(config, "miner", object()), "training_workspace", "artifacts/miner_training")
        )
        private_root = getattr(getattr(config, "miner", object()), "private_dataset_root", "")
        self.training_pipeline = TrainingPipeline(
            TrainingSettings(
                workspace=workspace,
                private_dataset_root=Path(private_root) if private_root else None,
                enable_auto_hpo=bool(
                    getattr(getattr(config, "miner", object()), "enable_auto_hpo", False)
                    or getattr(getattr(config, "miner", object()), "autoresearch", False)
                ),
                autoresearch_max_iters=int(
                    getattr(getattr(config, "miner", object()), "autoresearch_max_iters", 4)
                ),
                autoresearch_experiment_minutes=int(
                    getattr(getattr(config, "miner", object()), "autoresearch_experiment_minutes", 5)
                ),
                autoresearch_log_level=str(
                    getattr(getattr(config, "miner", object()), "autoresearch_log_level", "INFO")
                ),
                random_hpo_draw=bool(
                    getattr(getattr(config, "miner", object()), "random_hpo_draw", False)
                ),
                hpo_seed=int(getattr(getattr(config, "miner", object()), "hpo_seed", 0)),
            )
        )
        self.current_manifest: TrainingManifest | None = None
        self.response_mode = str(
            getattr(getattr(config, "miner", object()), "response_mode", "standard")
        ).strip()

    def run(self, synapse: HazardDetection) -> HazardDetection:
        started_at = time.time()
        if synapse.task_type in ("inference", "verification"):
            self._solve_inference(synapse)
        elif synapse.task_type == "training":
            self._solve_training(synapse)
        else:
            raise ValueError(f"Unsupported task type: {synapse.task_type}")
        synapse.duration_ms = int((time.time() - started_at) * 1000)
        if synapse.task_type == "training":
            bt.logging.info(
                f"event=miner_training_response_pre_return task_id={synapse.task_id} "
                f"manifest_present={synapse.submitted_training_manifest is not None} "
                f"storage_signal={synapse.miner_storage_signal} "
                f"r2_uri={getattr(synapse.submitted_training_manifest, 'candidate_model_uri', None)} "
                f"duration_ms={synapse.duration_ms}"
            )
        self._apply_response_mode(synapse)
        return synapse

    def _solve_inference(self, synapse: HazardDetection) -> None:
        if not synapse.image_b64:
            raise ValueError("Inference task received without image payload.")
        image_bytes = base64.b64decode(synapse.image_b64)
        if synapse.requested_model_hash is not None:
            if self.current_manifest is None:
                # Allow baseline-hash inference before the miner has completed any training task.
                pass
            elif synapse.requested_model_hash != self.current_manifest.candidate_model_hash:
                raise ValueError("Requested model hash does not match miner's active candidate.")
        digest = hashlib.sha256(image_bytes + synapse.challenge_nonce.encode("utf-8")).hexdigest()
        energy = int(digest[:8], 16) / 0xFFFFFFFF
        hazard_detected = energy > 0.35
        confidence = min(0.99, max(0.05, 0.5 + (energy - 0.5) * 0.8))

        synapse.hazard_detected = hazard_detected
        synapse.confidence = float(confidence)
        synapse.severity = self._severity_from_energy(energy, hazard_detected)
        synapse.bounding_boxes = self._boxes_from_digest(digest, hazard_detected)

        query = f"{synapse.site_id} {synapse.severity} {synapse.task_type}"
        refs = self.osha_db.search(query, top_k=2)
        synapse.osha_refs = [ref.citation_id for ref in refs]
        ref_titles = ", ".join(ref.title for ref in refs)
        synapse.rationale = (
            f"Hazard={'yes' if hazard_detected else 'no'} severity={synapse.severity}. "
            f"Grounded with OSHA context: {ref_titles}."
        )
        synapse.model_hash = (
            self.current_manifest.candidate_model_hash
            if self.current_manifest is not None
            else hashlib.sha256(
                f"{synapse.site_id}:{synapse.task_id}:{synapse.challenge_nonce}:{synapse.severity}".encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        synapse.submitted_training_manifest = self.current_manifest

    def _solve_training(self, synapse: HazardDetection) -> None:
        if synapse.baseline_checkpoint is None:
            raise ValueError("Training task requires baseline_checkpoint.")
        if synapse.training_dataset is None:
            raise ValueError("Training task requires training_dataset.")
        if synapse.max_training_seconds is None:
            raise ValueError("Training task requires max_training_seconds.")
        manifest = self.training_pipeline.run(
            task_id=synapse.task_id,
            baseline=synapse.baseline_checkpoint,
            training_dataset=synapse.training_dataset,
            max_training_seconds=synapse.max_training_seconds,
        )
        self.current_manifest = manifest
        synapse.submitted_training_manifest = manifest
        synapse.miner_r2_credentials = load_r2_credentials_from_env()
        account = synapse.miner_r2_credentials.account_id
        masked_account = f"{account[:3]}***{account[-3:]}" if len(account) >= 6 else "***"
        bt.logging.info(
            f"event=handshake_1_credentials_exchange task_id={synapse.task_id} "
            f"account_id={masked_account} "
            f"bucket={synapse.miner_r2_credentials.bucket_name} "
            f"endpoint={synapse.miner_r2_credentials.s3_endpoint}"
        )
        synapse.miner_storage_signal = "checkpoint_uploaded"
        bt.logging.info(
            f"event=handshake_2_checkpoint_uploaded task_id={synapse.task_id} "
            f"signal={synapse.miner_storage_signal} r2_uri={manifest.candidate_model_uri} "
            f"candidate_hash={manifest.candidate_model_hash}"
        )
        synapse.training_metrics = dict(manifest.metrics)
        synapse.hazard_detected = None
        synapse.severity = "none"
        synapse.confidence = manifest.metrics.get("reproducibility_score", 0.0)
        synapse.bounding_boxes = []
        synapse.osha_refs = []
        synapse.rationale = "Training completed and candidate checkpoint manifest returned."
        synapse.model_hash = manifest.candidate_model_hash

    @staticmethod
    def _severity_from_energy(energy: float, hazard_detected: bool) -> SeverityTier:
        if not hazard_detected:
            return "none"
        if energy < 0.5:
            return "low"
        if energy < 0.7:
            return "medium"
        if energy < 0.85:
            return "high"
        return "critical"

    @staticmethod
    def _boxes_from_digest(digest: str, hazard_detected: bool) -> List[BoundingBox]:
        if not hazard_detected:
            return []
        x_seed = int(digest[8:12], 16) / 0xFFFF
        y_seed = int(digest[12:16], 16) / 0xFFFF
        width_seed = int(digest[16:20], 16) / 0xFFFF
        height_seed = int(digest[20:24], 16) / 0xFFFF
        x_min = min(0.85, x_seed * 0.7)
        y_min = min(0.85, y_seed * 0.7)
        width = 0.1 + width_seed * 0.2
        height = 0.1 + height_seed * 0.2
        return [
            BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=min(0.99, x_min + width),
                y_max=min(0.99, y_min + height),
                label="hazard",
                confidence=0.75,
            )
        ]

    def _apply_response_mode(self, synapse: HazardDetection) -> None:
        if self.response_mode == "standard":
            return
        if self.response_mode == "replay_nonce":
            synapse.challenge_nonce = "0000000000000000"
            bt.logging.warning("event=miner_response_mode_applied mode=replay_nonce")
            return
        if self.response_mode == "wrong_model_hash":
            synapse.model_hash = hashlib.sha256(
                f"wrong:{synapse.task_id}".encode("utf-8")
            ).hexdigest()
            bt.logging.warning("event=miner_response_mode_applied mode=wrong_model_hash")
            return
        if (
            self.response_mode == "malformed_manifest"
            and synapse.submitted_training_manifest is not None
        ):
            bad_manifest = copy.deepcopy(synapse.submitted_training_manifest)
            bad_manifest.dataset_lineage_hash = "bad"
            synapse.submitted_training_manifest = bad_manifest
            bt.logging.warning("event=miner_response_mode_applied mode=malformed_manifest")

