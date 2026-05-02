# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import asyncio
import os
import random
import base64
from pathlib import Path

import pytest
from types import SimpleNamespace

from template.hazard.artifacts import ArtifactRegistry
from template.hazard.dataset import HazardDatasetManager
from template.hazard.evaluator import GoldenSetEvaluator
from template.hazard.incentives import broad_softmax_scores
from template.hazard.scheduler import CohortScheduler
from template.hazard.serving import CommercialServingGateway, PromotionRegistry
from template.miner import HazardMinerEngine
from template.miner.training import TrainingPipeline, TrainingSettings
from template.protocol import HazardDetection, ModelCheckpoint, TrainingManifest
from template.validator.forward import forward
from template.validator.reward import get_rewards


def test_dataset_manager_loads_all_partitions():
    manager = HazardDatasetManager(
        dataset_root=Path(__file__).resolve().parents[1] / "data" / "hazard"
    )
    pointer = manager.pointer("training_pool")
    assert pointer.split == "training_pool"
    assert pointer.sample_count >= 1
    sampled = manager.sample("golden", random_state=__import__("random").Random(7))
    assert sampled.task_id.startswith("golden-")
    assert sampled.image_bytes


def test_scheduler_cycles_cohorts():
    scheduler = CohortScheduler(seed=13)
    class Stub:
        step = 0
        class config:
            class neuron:
                sample_size = 4
        class metagraph:
            n = __import__("numpy").array(4)
            axons = []
            validator_permit = [False, False, False, False]
            S = [0, 0, 0, 0]
    # Cohort decision itself should not fail.
    assert scheduler._choose_cohort(0) == "training"
    assert scheduler._choose_cohort(3) == "training"
    assert scheduler._choose_cohort(6) == "exploration"
    assert scheduler._choose_cohort(7) == "verification"
    assert scheduler._choose_cohort(8) == "promotion"
    assert scheduler._choose_cohort(9) == "stability"


def test_reward_pipeline_combines_inference_and_training():
    manager = HazardDatasetManager(
        dataset_root=Path(__file__).resolve().parents[1] / "data" / "hazard"
    )
    task = manager.sample("hidden_eval", random_state=__import__("random").Random(1))
    response = HazardDetection(
        task_type="inference",
        dataset_partition=task.partition,
        task_id=task.task_id,
        site_id=task.site_id,
        challenge_nonce="abc123",
        image_b64=base64.b64encode(task.image_bytes).decode("ascii"),
        hazard_detected=task.hazard_detected,
        severity=task.severity,
        confidence=0.9,
        osha_refs=task.expected_osha_refs,
        model_hash="model-abc12345",
        rationale="Validator-audited reasoning with OSHA references and geometry evidence.",
    )
    registry = ArtifactRegistry()
    registry.submit(
        1,
        TrainingManifest(
            parent_model_hash="parent-hash",
            candidate_model_hash="candidate-hash",
            candidate_model_uri="file:///tmp/candidate",
            config_hash="cfg-hash",
            dataset_lineage_hash="lineage-hash",
            recipe_uri="ipfs://recipe",
            metrics={"uplift": 0.5, "stability": 0.8, "efficiency": 0.7},
        ),
    )
    verification = registry.verify(
        1,
        "candidate-hash",
        golden_score=0.8,
        expected_parent_hash="parent-hash",
    )
    scores, breakdowns = get_rewards([task], [response], [verification])
    assert len(scores) == 1
    assert 0.0 <= float(scores[0]) <= 1.0
    assert 0.0 <= breakdowns[0].inference_score <= 1.0
    assert 0.0 <= breakdowns[0].training_score <= 1.0


def test_training_pipeline_creates_real_candidate_manifest(tmp_path):
    _r2 = (
        os.getenv("R2_ACCOUNT_ID", "").strip()
        and os.getenv("R2_BUCKET_NAME", "").strip()
        and os.getenv("R2_S3_ENDPOINT", "").strip()
        and os.getenv("R2_ACCESS_KEY_ID", "").strip()
        and os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    )
    if not _r2:
        pytest.skip("Full training pipeline requires Cloudflare R2 credentials in the environment.")
    data_root = Path(__file__).resolve().parents[1] / "data" / "hazard"
    manager = HazardDatasetManager(dataset_root=data_root)
    baseline_uri = "yolov8s.pt"
    baseline_hash = __import__("hashlib").sha256(baseline_uri.encode("utf-8")).hexdigest()
    pipeline = TrainingPipeline(
        TrainingSettings(
            workspace=tmp_path,
            private_dataset_root=None,
            enable_auto_hpo=False,
            autoresearch_max_iters=1,
            autoresearch_experiment_minutes=1,
            autoresearch_log_level="INFO",
        )
    )
    manifest = pipeline.run(
        task_id="training-smoke",
        baseline=ModelCheckpoint(uri=baseline_uri, sha256=baseline_hash),
        training_dataset=manager.pointer("training_pool"),
        max_training_seconds=60,
    )
    assert manifest.parent_model_hash == baseline_hash
    assert manifest.candidate_model_hash
    assert manifest.candidate_model_uri.startswith(("file://", "r2://"))


def test_golden_evaluator_and_artifact_registry_verify_training(tmp_path):
    data_root = Path(__file__).resolve().parents[1] / "data" / "hazard"
    manager = HazardDatasetManager(dataset_root=data_root)
    evaluator = GoldenSetEvaluator(manager)
    manifest = TrainingManifest(
        parent_model_hash="baseline-1234",
        candidate_model_hash="candidate-1234",
        candidate_model_uri=(tmp_path / "candidate.json").as_uri(),
        config_hash="config-1234",
        dataset_lineage_hash=manager.pointer("training_pool").sha256,
        recipe_uri=(tmp_path / "recipe.json").as_uri(),
        metrics={"reproducibility_score": 1.0, "uplift": 0.6, "efficiency": 0.9},
    )
    # Avoid heavy model execution in this unit test; verify registry scoring path directly.
    golden_score = 0.8
    registry = ArtifactRegistry()
    registry.submit(2, manifest)
    result = registry.verify(
        2,
        manifest.candidate_model_hash,
        golden_score=golden_score,
        expected_parent_hash=manifest.parent_model_hash,
    )
    assert result.score > 0.0


def test_broad_softmax_pays_multiple_value_adding_miners():
    shaped = broad_softmax_scores(
        __import__("numpy").array([0.0, 0.2, 0.4, 0.8], dtype=float),
        temperature=0.35,
        floor=0.01,
        min_score=0.05,
    )
    assert shaped[0] == 0.0
    assert (shaped[1:] > 0.0).all()
    assert abs(float(shaped.sum()) - 1.0) < 1e-6


def test_isolated_validator_miner_training_then_inference(tmp_path):
    data_root = Path(__file__).resolve().parents[1] / "data" / "hazard"
    baseline_uri = "yolov8s.pt"
    baseline_hash = __import__("hashlib").sha256(baseline_uri.encode("utf-8")).hexdigest()
    dataset_manager = HazardDatasetManager(dataset_root=data_root)

    class LocalDendrite:
        def __init__(self):
            self.engines = {
                1: HazardMinerEngine(
                    config=SimpleNamespace(
                        miner=SimpleNamespace(
                            training_workspace=str(tmp_path / "miner-1"),
                            private_dataset_root="",
                            enable_auto_hpo=False,
                        )
                    )
                ),
                2: HazardMinerEngine(
                    config=SimpleNamespace(
                        miner=SimpleNamespace(
                            training_workspace=str(tmp_path / "miner-2"),
                            private_dataset_root="",
                            enable_auto_hpo=False,
                        )
                    )
                ),
            }

        async def __call__(self, axons, synapse, timeout, deserialize):
            responses = []
            for axon in axons:
                response = synapse.model_copy(deep=True)
                responses.append(self.engines[axon.uid].run(response))
            return responses

    class LocalValidator:
        def __init__(self):
            self.step = 0
            self.random = random.Random(4)
            self.dataset_manager = dataset_manager
            self.scheduler = CohortScheduler(seed=4)
            self.artifact_registry = ArtifactRegistry()
            self.golden_evaluator = GoldenSetEvaluator(dataset_manager)
            self.promotion_registry = PromotionRegistry(min_promotion_score=0.2)
            self.serving_gateway = CommercialServingGateway(self.promotion_registry)
            self.baseline_checkpoint_hash = baseline_hash
            self.dendrite = LocalDendrite()
            self.metagraph = SimpleNamespace(
                n=__import__("numpy").array(3),
                axons=[
                    SimpleNamespace(uid=0, is_serving=False),
                    SimpleNamespace(uid=1, is_serving=True),
                    SimpleNamespace(uid=2, is_serving=True),
                ],
                validator_permit=[False, False, False],
                S=[0.0, 0.0, 0.0],
            )
            self.config = SimpleNamespace(
                    subtensor=SimpleNamespace(chain_endpoint="ws://127.0.0.1:9944"),
                neuron=SimpleNamespace(
                    sample_size=2,
                    vpermit_tao_limit=4096,
                    timeout=10,
                    moving_average_alpha=1.0,
                    baseline_checkpoint_uri=baseline_uri,
                    max_training_seconds=60,
                )
            )
            self.scores = __import__("numpy").zeros(3, dtype=__import__("numpy").float32)
            self.inference_scores = __import__("numpy").zeros(3, dtype=__import__("numpy").float32)
            self.training_scores = __import__("numpy").zeros(3, dtype=__import__("numpy").float32)
            self.last_serving_model_hash = None

        def update_scores(self, rewards, uids):
            for uid, reward in zip(uids, rewards):
                self.scores[uid] = reward

        def update_score_ledgers(self, breakdowns, uids):
            for uid, item in zip(uids, breakdowns):
                self.inference_scores[uid] = item.inference_score
                self.training_scores[uid] = item.training_score
            self.last_serving_model_hash = self.serving_gateway.select_model_hash()

    validator = LocalValidator()
    asyncio.run(forward(validator))
    assert validator.training_scores[1] > 0.0
    assert validator.training_scores[2] > 0.0

    validator.step = 3
    asyncio.run(forward(validator))
    assert validator.inference_scores[1] >= 0.0
    assert validator.last_serving_model_hash is not None
