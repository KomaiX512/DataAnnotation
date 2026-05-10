# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer):TECHNOLOGY NUCLEUS
# Copyright © 2023 TECHNOLOGY NUCLEUS

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import json
import os
import time
import random
import hashlib
from pathlib import Path
from typing import List, Mapping, Optional, Dict
from urllib.parse import urlparse

# Bittensor
import bittensor as bt
import numpy as np

from template.base.validator import BaseValidatorNeuron
from template.hazard import (
    ArtifactRegistry,
    CohortScheduler,
    CommercialServingGateway,
    GoldenSetEvaluator,
    HazardDatasetManager,
    PromotionRegistry,
)
from template.hazard.annotation_eval import (
    AnnotationFidelityScorer,
    ConsensusScorer,
)
from template.hazard.dataset_assembler import AdoptionLedger, DatasetAssembler
from template.hazard.dual_reward import (
    DualFlywheelBreakdown,
    DualFlywheelRewardComposer,
)
from template.hazard.golden_injection import GoldenInjector
from template.hazard.image_corpus import ImageCorpus, ImageCorpusConfig
from template.hazard.incentives import broad_softmax_scores
from template.hazard.model_eval import ModelAccuracyComponents, ModelAccuracyEvaluator
from template.hazard.submission_dedup import ModelHashClaimRegistry
from template.hazard.r2_storage import load_r2_credentials_from_env
from template.protocol import R2AccessCredentials
from template.validator.dual_forward import dual_flywheel_forward
from template.validator.reward import RewardBreakdown
from template.validator import forward as legacy_forward


class Validator(BaseValidatorNeuron):
    """
    Dual-flywheel validator neuron.

    Two operating modes are supported:

    * ``dual_flywheel`` (default) -- dispatches AnnotationAndTrainingTask
      synapses, scores annotation fidelity vs the validator-only Golden Set,
      scores annotation consensus across miners on the unlabeled Annotation
      Pool, evaluates miner checkpoints on Golden + cross-domain benchmark,
      assembles the per-image_id commercial dataset, and computes the final
      on-chain weight as ``alpha * annotation + beta * model + gamma * adoption``.

    * ``legacy_hazard_detection`` -- preserves the original cohort-driven
      HazardDetection synapse pipeline for backwards compatibility.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.random = random.Random(self.config.neuron.scheduler_seed)
        self.task_mode = str(getattr(self.config.neuron, "task_mode", "dual_flywheel")).strip()

        # --- Legacy components (still used by the legacy mode and by tests). ---
        self.dataset_manager = HazardDatasetManager(
            dataset_root=Path(self.config.neuron.dataset_root)
        )
        self.scheduler = CohortScheduler(seed=self.config.neuron.scheduler_seed)
        self.artifact_registry = ArtifactRegistry()
        self.golden_evaluator = GoldenSetEvaluator(self.dataset_manager)
        self.promotion_registry = PromotionRegistry(
            min_promotion_score=self.config.neuron.promotion_threshold,
            recency_decay=float(self.config.neuron.serving_recency_decay),
            min_live_multiplier=float(self.config.neuron.serving_min_live_multiplier),
        )
        self.serving_gateway = CommercialServingGateway(self.promotion_registry)
        self.inference_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.training_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.last_serving_model_hash: Optional[str] = None
        self.baseline_checkpoint_hash = self._resolve_baseline_hash()

        # --- Dual-flywheel components (lazy-loaded the first time we run a
        # round; we instantiate now but defer HF dataset materialization). ---
        self.image_corpus = ImageCorpus(self._build_corpus_config())
        self.golden_injector = GoldenInjector(
            corpus=self.image_corpus,
            request_size=int(self.config.neuron.flywheel_annotation_request_size),
            golden_per_request=int(self.config.neuron.flywheel_golden_injection_per_request),
        )
        self.fidelity_scorer = AnnotationFidelityScorer(
            hallucination_penalty=float(self.config.neuron.flywheel_hallucination_penalty),
        )
        self.consensus_scorer = ConsensusScorer()
        self.model_evaluator = ModelAccuracyEvaluator(
            download_root=Path(self.config.neuron.full_path) / "miner_checkpoints",
            docker_sandbox_image=str(
                getattr(self.config.neuron, "flywheel_model_eval_docker_image", "") or ""
            ).strip(),
        )
        self.dataset_assembler = DatasetAssembler(
            corpus=self.image_corpus,
            storage_prefix=str(self.config.neuron.flywheel_commercial_dataset_prefix),
        )
        self.model_hash_registry = ModelHashClaimRegistry.load(
            Path(self.config.neuron.full_path) / "flywheel_model_hash_registry.json"
        )
        self.reward_composer = DualFlywheelRewardComposer(
            alpha=float(self.config.neuron.flywheel_alpha_annotation),
            beta=float(self.config.neuron.flywheel_beta_model),
            gamma=float(self.config.neuron.flywheel_gamma_adoption),
            hallucination_penalty_per_event=float(
                self.config.neuron.flywheel_hallucination_penalty
            ),
        )

        # Per-uid dual-flywheel ledgers (persisted across restarts).
        self.annotation_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.model_accuracy_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.adoption_bonus_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.last_commercial_dataset_uri: Optional[str] = None

        bt.logging.info(f"event=validator_init task_mode={self.task_mode}")
        bt.logging.info("load_state()")
        self.load_state()

    # ------------------------------------------------------------------ Hooks
    async def forward(self):
        """Validator forward pass; route to the configured mode."""
        if self.task_mode == "dual_flywheel":
            return await dual_flywheel_forward(self)
        return await legacy_forward(self)

    def update_score_ledgers(
        self,
        breakdowns: List,
        uids: List[int],
        *,
        model_accuracy: Optional[Mapping[int, ModelAccuracyComponents]] = None,
        promotion_model_hashes: Optional[Mapping[int, str]] = None,
    ) -> None:
        alpha: float = self.config.neuron.moving_average_alpha
        # Dual-flywheel breakdowns expose annotation/model/adoption components.
        if breakdowns and isinstance(breakdowns[0], DualFlywheelBreakdown):
            hashes: Dict[int, str] = (
                dict(promotion_model_hashes) if promotion_model_hashes else {}
            )
            for uid, item in zip(uids, breakdowns):
                self.annotation_scores[uid] = (
                    alpha * item.annotation_score + (1.0 - alpha) * self.annotation_scores[uid]
                )
                self.model_accuracy_scores[uid] = (
                    alpha * item.model_accuracy_score
                    + (1.0 - alpha) * self.model_accuracy_scores[uid]
                )
                self.adoption_bonus_scores[uid] = (
                    alpha * item.adoption_bonus
                    + (1.0 - alpha) * self.adoption_bonus_scores[uid]
                )
                # Mirror to legacy ledgers so existing dashboards still work:
                self.inference_scores[uid] = self.annotation_scores[uid]
                self.training_scores[uid] = self.model_accuracy_scores[uid]
                # Promotion registry uses the final score for serving selection.
                candidate_hash = (hashes.get(uid) or "").strip() or None
                self.promotion_registry.maybe_promote(
                    uid=uid,
                    model_hash=candidate_hash if item.model_accuracy_score > 0 else None,
                    score=item.final_score,
                    step=self.step,
                )
            self.last_serving_model_hash = self.serving_gateway.select_model_hash(
                current_step=self.step
            )
            return

        # Legacy reward breakdowns.
        for uid, item in zip(uids, breakdowns):
            self.inference_scores[uid] = (
                alpha * item.inference_score + (1.0 - alpha) * self.inference_scores[uid]
            )
            self.training_scores[uid] = (
                alpha * item.training_score + (1.0 - alpha) * self.training_scores[uid]
            )
        self.last_serving_model_hash = self.serving_gateway.select_model_hash(
            current_step=self.step
        )

    def set_weights(self):
        raw_scores = self.scores.copy()
        self.scores = broad_softmax_scores(
            raw_scores,
            temperature=self.config.neuron.incentive_temperature,
            floor=self.config.neuron.incentive_floor,
            min_score=self.config.neuron.incentive_min_score,
        )
        try:
            endpoint = str(getattr(self.config.subtensor, "chain_endpoint", ""))
            force_local = os.environ.get("FORCE_LOCAL_SET_WEIGHTS", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if (
                not force_local
                and (
                    endpoint.startswith("ws://127.0.0.1")
                    or endpoint.startswith("ws://localhost")
                )
            ):
                bt.logging.warning(
                    f"Skipping on-chain set_weights in local dev endpoint {endpoint} due commit API mismatch. "
                    "Set FORCE_LOCAL_SET_WEIGHTS=1 to attempt commits anyway."
                )
                return
            super().set_weights()
        finally:
            self.scores = raw_scores

    def resync_metagraph(self):
        previous_size = len(self.scores)
        super().resync_metagraph()
        if len(self.scores) == previous_size:
            return
        new_size = len(self.scores)

        def _resize(array: np.ndarray) -> np.ndarray:
            new_arr = np.zeros(new_size, dtype=np.float32)
            copy_len = min(len(array), new_size)
            new_arr[:copy_len] = array[:copy_len]
            return new_arr

        self.inference_scores = _resize(self.inference_scores)
        self.training_scores = _resize(self.training_scores)
        self.annotation_scores = _resize(self.annotation_scores)
        self.model_accuracy_scores = _resize(self.model_accuracy_scores)
        self.adoption_bonus_scores = _resize(self.adoption_bonus_scores)

    def save_state(self):
        bt.logging.info("Saving validator state.")
        if not hasattr(self, "inference_scores") or not hasattr(self, "training_scores"):
            return
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
            inference_scores=self.inference_scores,
            training_scores=self.training_scores,
            annotation_scores=self.annotation_scores,
            model_accuracy_scores=self.model_accuracy_scores,
            adoption_bonus_scores=self.adoption_bonus_scores,
            last_serving_model_hash=self.last_serving_model_hash or "",
            last_commercial_dataset_uri=self.last_commercial_dataset_uri or "",
        )
        ledger_path = Path(self.config.neuron.full_path) / "adoption_ledger.json"
        ledger_path.write_text(
            json.dumps(self.dataset_assembler.ledger.to_jsonable(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.model_hash_registry.save(
            Path(self.config.neuron.full_path) / "flywheel_model_hash_registry.json"
        )

    def load_state(self):
        bt.logging.info("Loading validator state.")
        state_path = Path(self.config.neuron.full_path) / "state.npz"
        if not state_path.exists():
            bt.logging.warning("No prior validator state found; starting fresh.")
            return
        state = np.load(state_path, allow_pickle=False)
        self.step = int(state["step"])
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
        if "inference_scores" in state:
            self.inference_scores = state["inference_scores"]
        if "training_scores" in state:
            self.training_scores = state["training_scores"]
        if "annotation_scores" in state:
            self.annotation_scores = state["annotation_scores"]
        if "model_accuracy_scores" in state:
            self.model_accuracy_scores = state["model_accuracy_scores"]
        if "adoption_bonus_scores" in state:
            self.adoption_bonus_scores = state["adoption_bonus_scores"]
        if "last_serving_model_hash" in state:
            value = state["last_serving_model_hash"].item()
            self.last_serving_model_hash = value if value else None
        if "last_commercial_dataset_uri" in state:
            value = state["last_commercial_dataset_uri"].item()
            self.last_commercial_dataset_uri = value if value else None
        ledger_path = Path(self.config.neuron.full_path) / "adoption_ledger.json"
        if ledger_path.exists():
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.dataset_assembler.ledger = AdoptionLedger.from_jsonable(payload)

    # ------------------------------------------------------------ helpers
    def _build_corpus_config(self) -> ImageCorpusConfig:
        return ImageCorpusConfig(
            cache_root=Path(self.config.neuron.flywheel_image_cache_root),
            serving_base_url=str(self.config.neuron.flywheel_image_serving_base_url),
            golden_dataset_id=str(self.config.neuron.flywheel_golden_dataset_id),
            golden_split=str(self.config.neuron.flywheel_golden_split),
            golden_ratio=float(self.config.neuron.flywheel_golden_ratio),
            golden_split_seed=int(self.config.neuron.flywheel_golden_split_seed),
            annotation_dataset_ids=str(self.config.neuron.flywheel_annotation_dataset_ids),
            annotation_split=str(self.config.neuron.flywheel_annotation_split),
            annotation_max_per_dataset=int(
                self.config.neuron.flywheel_annotation_max_per_dataset
            ),
            benchmark_dataset_id=str(self.config.neuron.flywheel_benchmark_dataset_id),
            benchmark_split=str(self.config.neuron.flywheel_benchmark_split),
            benchmark_max_samples=int(self.config.neuron.flywheel_benchmark_max_samples),
            hf_revision=str(self.config.neuron.flywheel_hf_revision),
        )

    def _resolve_baseline_hash(self) -> str:
        configured_hash = self.config.neuron.baseline_checkpoint_hash
        if configured_hash:
            return configured_hash
        parsed = urlparse(self.config.neuron.baseline_checkpoint_uri)
        if parsed.scheme == "file":
            path = Path(parsed.path)
            if not path.exists():
                raise FileNotFoundError(f"Baseline checkpoint does not exist: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        digest = hashlib.sha256()
        digest.update(self.config.neuron.baseline_checkpoint_uri.encode("utf-8"))
        return digest.hexdigest()

    def _should_export_commercial(self, step: int) -> bool:
        every = max(1, int(self.config.neuron.flywheel_commercial_export_every))
        return step % every == 0

    def _load_commercial_credentials(self) -> Optional[R2AccessCredentials]:
        scheme = urlparse(str(self.config.neuron.flywheel_commercial_dataset_prefix)).scheme
        if scheme == "file":
            return None
        return load_r2_credentials_from_env()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
