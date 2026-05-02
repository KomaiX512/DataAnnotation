# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import time
import random
import hashlib
from pathlib import Path
from urllib.parse import urlparse

# Bittensor
import bittensor as bt
import numpy as np

# import base validator class which takes care of most of the boilerplate
from template.base.validator import BaseValidatorNeuron
from template.hazard import (
    ArtifactRegistry,
    CohortScheduler,
    CommercialServingGateway,
    GoldenSetEvaluator,
    HazardDatasetManager,
    PromotionRegistry,
)
from template.hazard.incentives import broad_softmax_scores
from template.validator.reward import RewardBreakdown

# Bittensor Validator Template:
from template.validator import forward


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.random = random.Random(self.config.neuron.scheduler_seed)
        self.dataset_manager = HazardDatasetManager(
            dataset_root=Path(self.config.neuron.dataset_root)
        )
        self.scheduler = CohortScheduler(seed=self.config.neuron.scheduler_seed)
        self.artifact_registry = ArtifactRegistry()
        self.golden_evaluator = GoldenSetEvaluator(self.dataset_manager)
        self.promotion_registry = PromotionRegistry(
            min_promotion_score=self.config.neuron.promotion_threshold
        )
        self.serving_gateway = CommercialServingGateway(self.promotion_registry)
        self.inference_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.training_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.last_serving_model_hash = None
        self.baseline_checkpoint_hash = self._resolve_baseline_hash()

        bt.logging.info("load_state()")
        self.load_state()

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return await forward(self)

    def update_score_ledgers(self, breakdowns: list[RewardBreakdown], uids: list[int]) -> None:
        alpha: float = self.config.neuron.moving_average_alpha
        for uid, item in zip(uids, breakdowns):
            self.inference_scores[uid] = (
                alpha * item.inference_score + (1.0 - alpha) * self.inference_scores[uid]
            )
            self.training_scores[uid] = (
                alpha * item.training_score + (1.0 - alpha) * self.training_scores[uid]
            )
        self.last_serving_model_hash = self.serving_gateway.select_model_hash()

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
            if endpoint.startswith("ws://127.0.0.1") or endpoint.startswith("ws://localhost"):
                bt.logging.warning(
                    f"Skipping on-chain set_weights in local dev endpoint {endpoint} due commit API mismatch."
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
        resized_inference = np.zeros(new_size, dtype=np.float32)
        resized_training = np.zeros(new_size, dtype=np.float32)
        copy_len = min(len(self.inference_scores), new_size)
        resized_inference[:copy_len] = self.inference_scores[:copy_len]
        resized_training[:copy_len] = self.training_scores[:copy_len]
        self.inference_scores = resized_inference
        self.training_scores = resized_training

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
            last_serving_model_hash=self.last_serving_model_hash or "",
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
        if "last_serving_model_hash" in state:
            value = state["last_serving_model_hash"].item()
            self.last_serving_model_hash = value if value else None

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
        # For model aliases or remote URIs (e.g., yolov8s.pt), use deterministic URI hash.
        digest = hashlib.sha256()
        digest.update(self.config.neuron.baseline_checkpoint_uri.encode("utf-8"))
        return digest.hexdigest()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
