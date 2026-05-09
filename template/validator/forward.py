# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer):TECHNOLOGY NUCLEUS
# Copyright © 2023 TECHNOLOGY NUCLEUS

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

import hashlib
import base64
import copy
import os
import re
import bittensor as bt

from template.hazard.artifacts import ArtifactVerificationResult
from template.protocol import HazardDetection, ModelCheckpoint
from template.utils.localnet_axon import localnet_miner_port_override
from template.validator.reward import get_rewards


def _build_challenge_nonce(step: int, uid: int, task_id: str) -> str:
    payload = f"{step}:{uid}:{task_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    selection = self.scheduler.select(self)
    bt.logging.info(
        f"Validator cohort={selection.cohort} partition={selection.partition} "
        f"task_type={selection.task_type} sampled={selection.uids.tolist()}"
    )

    tasks = []
    responses = []
    artifact_results = []
    scored_uids = []
    for uid in selection.uids.tolist():
        try:
            task = self.dataset_manager.sample(
                selection.partition,
                task_type=selection.task_type,
                random_state=self.random,
            )
            challenge_nonce = _build_challenge_nonce(self.step, uid, task.task_id)
            synapse = _build_synapse(self, uid, task, selection, challenge_nonce)
            target_axon = _resolve_target_axon(self, uid)
            bt.logging.info(
                f"Forward cycle uid={uid} hotkey={getattr(target_axon, 'hotkey', 'unknown')} "
                f"ip={getattr(target_axon, 'ip', 'unknown')} port={getattr(target_axon, 'port', 'unknown')} "
                f"task={selection.task_type} partition={selection.partition}"
            )
            request_timeout = _request_timeout(self, selection.task_type)
            if selection.task_type == "training":
                bt.logging.info(
                    f"event=validator_training_request_pre_send uid={uid} task_id={synapse.task_id} "
                    f"timeout={request_timeout} manifest_present={synapse.submitted_training_manifest is not None} "
                    f"storage_signal={synapse.miner_storage_signal}"
                )
            uid_responses = await self.dendrite(
                axons=[target_axon],
                synapse=synapse,
                timeout=request_timeout,
                deserialize=True,
            )
            response = uid_responses[0]
            _validate_response_integrity(
                response=response,
                expected_task_id=synapse.task_id,
                expected_nonce=challenge_nonce,
            )
            bt.logging.info(
                f"Dendrite response uid={uid} status={getattr(response.dendrite, 'status_code', 'unknown')} "
                f"message={getattr(response.dendrite, 'status_message', 'unknown')} "
                f"model_hash={getattr(response, 'model_hash', None)}"
            )
            bt.logging.info(
                f"event=validator_response_post_decode uid={uid} task_id={getattr(response, 'task_id', None)} "
                f"task_type={selection.task_type} status={getattr(response.dendrite, 'status_code', 'unknown')} "
                f"manifest_present={getattr(response, 'submitted_training_manifest', None) is not None} "
                f"storage_signal={getattr(response, 'miner_storage_signal', None)} "
                f"r2_uri={getattr(getattr(response, 'submitted_training_manifest', None), 'candidate_model_uri', None)}"
            )
            golden_score = 0.0
            if response.submitted_training_manifest is not None:
                self.artifact_registry.submit(uid, response.submitted_training_manifest)
                bt.logging.info(
                    f"event=handshake_1_credentials_exchange uid={uid} "
                    f"task_id={response.task_id} has_credentials={response.miner_r2_credentials is not None}"
                )
                if selection.task_type == "training":
                    if response.miner_storage_signal != "checkpoint_uploaded":
                        raise ValueError("Miner did not send storage download signal after training.")
                    bt.logging.info(
                        f"event=handshake_2_checkpoint_uploaded uid={uid} task_id={response.task_id} "
                        f"signal={response.miner_storage_signal} "
                        f"r2_uri={response.submitted_training_manifest.candidate_model_uri}"
                    )
                    golden_eval = self.golden_evaluator.evaluate(
                        response.submitted_training_manifest,
                        response.miner_r2_credentials,
                    )
                    golden_score = golden_eval.golden_score
                    response.submitted_training_manifest.metrics.update(
                        {
                            "golden_score": golden_eval.golden_score,
                            "golden_severity_score": golden_eval.severity_score,
                            "golden_localization_score": golden_eval.localization_score,
                            "golden_reasoning_score": golden_eval.reasoning_score,
                        }
                    )
            if selection.task_type == "training" and response.submitted_training_manifest is None:
                artifact_result = ArtifactVerificationResult(
                    passed=False,
                    score=0.0,
                    reason="Training task response did not include a submitted manifest.",
                )
            else:
                artifact_result = self.artifact_registry.verify(
                    uid,
                    response.model_hash,
                    golden_score=golden_score,
                    expected_parent_hash=self.baseline_checkpoint_hash,
                )
            bt.logging.info(
                f"event=artifact_verification uid={uid} task_type={selection.task_type} "
                f"passed={artifact_result.passed} training_score={artifact_result.score:.6f} "
                f"reason={artifact_result.reason}"
            )
            tasks.append(task)
            responses.append(response)
            artifact_results.append(artifact_result)
            scored_uids.append(uid)
        except Exception as exc:
            bt.logging.error(
                f"event=uid_forward_failure uid={uid} task_type={selection.task_type} error={exc}"
            )

    if not scored_uids:
        bt.logging.warning(
            f"event=forward_no_scored_uids cohort={selection.cohort} task_type={selection.task_type}"
        )
        return
    rewards, breakdowns = get_rewards(tasks, responses, artifact_results)
    self.update_scores(rewards, scored_uids)
    self.update_score_ledgers(breakdowns, scored_uids)

    for uid, response, breakdown in zip(scored_uids, responses, breakdowns):
        promoted = self.promotion_registry.maybe_promote(
            uid=uid,
            model_hash=response.model_hash,
            score=breakdown.final_score,
            step=self.step,
        )
        if promoted:
            bt.logging.info(f"Promoted uid={uid} model={response.model_hash}")

    bt.logging.info(f"Step rewards: {rewards.tolist()}")


def _validate_response_integrity(
    *,
    response: HazardDetection,
    expected_task_id: str,
    expected_nonce: str,
) -> None:
    if getattr(response, "task_id", "") != expected_task_id:
        raise ValueError(
            f"Mismatched task_id in miner response: expected={expected_task_id} got={getattr(response, 'task_id', None)}"
        )
    if getattr(response, "challenge_nonce", "") != expected_nonce:
        raise ValueError(
            f"Challenge nonce mismatch in miner response for task_id={expected_task_id}"
        )
    manifest = getattr(response, "submitted_training_manifest", None)
    if manifest is None:
        return
    if not _looks_like_hex_digest(manifest.parent_model_hash):
        raise ValueError("Invalid parent_model_hash format in submitted training manifest.")
    if not _looks_like_hex_digest(manifest.candidate_model_hash):
        raise ValueError("Invalid candidate_model_hash format in submitted training manifest.")
    if not _looks_like_hex_digest(manifest.config_hash):
        raise ValueError("Invalid config_hash format in submitted training manifest.")
    if not _looks_like_hex_digest(manifest.dataset_lineage_hash):
        raise ValueError("Invalid dataset_lineage_hash format in submitted training manifest.")
    if not manifest.candidate_model_uri.startswith("r2://"):
        raise ValueError("Submitted candidate_model_uri must use r2:// scheme.")
    if "/miners/current/" not in manifest.candidate_model_uri:
        raise ValueError(
            "Submitted candidate_model_uri must be namespaced under miners/current."
        )
    if not manifest.recipe_uri:
        raise ValueError("Submitted recipe_uri must be non-empty.")


def _looks_like_hex_digest(value: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{16,128}", value or ""))


def _request_timeout(self, task_type: str) -> float:
    if task_type != "training":
        return float(self.config.neuron.timeout)
    configured = float(getattr(self.config.neuron, "training_timeout", 0.0) or 0.0)
    if configured > 0.0:
        return configured
    max_training_seconds = float(getattr(self.config.neuron, "max_training_seconds", 0) or 0)
    return max(float(self.config.neuron.timeout), max_training_seconds + 300.0)


def _resolve_target_axon(self, uid: int):
    axon = self.metagraph.axons[uid]
    subtensor_cfg = getattr(self.config, "subtensor", None)
    endpoint = str(getattr(subtensor_cfg, "chain_endpoint", ""))
    if not endpoint.startswith("ws://127.0.0.1"):
        return axon
    # Localnet override: in single-machine tests, route miner queries directly to localhost.
    self_uid = getattr(self, "uid", -1)
    if uid == int(self_uid):
        return axon
    patched = copy.deepcopy(axon)
    patched.ip = "127.0.0.1"
    hk = self.metagraph.hotkeys[uid]
    port_override = localnet_miner_port_override(hk)
    if port_override is not None:
        patched.port = int(port_override)
    elif int(getattr(patched, "port", 0) or 0) == 0:
        patched.port = int(os.getenv("LOCALNET_MINER_PORT", "8091"))
    bt.logging.debug(
        f"Localnet axon override uid={uid} hotkey={hk[:16]}... "
        f"original={getattr(axon, 'ip', 'unknown')}:{getattr(axon, 'port', 'unknown')} "
        f"patched={patched.ip}:{patched.port}"
    )
    return patched


def _build_synapse(self, uid, task, selection, challenge_nonce: str) -> HazardDetection:
    if selection.task_type == "training":
        return HazardDetection(
            task_type="training",
            dataset_partition="training_pool",
            task_id=f"training-{self.step}-{uid}",
            site_id="training-pool",
            challenge_nonce=challenge_nonce,
            training_dataset=self.dataset_manager.pointer("training_pool"),
            golden_dataset=self.dataset_manager.pointer("golden"),
            baseline_checkpoint=ModelCheckpoint(
                uri=self.config.neuron.baseline_checkpoint_uri,
                sha256=self.baseline_checkpoint_hash,
            ),
            max_training_seconds=self.config.neuron.max_training_seconds,
            requested_model_hash=self.baseline_checkpoint_hash,
        )
    latest_manifest = self.artifact_registry.latest_manifest(uid)
    requested_hash = (
        latest_manifest.candidate_model_hash
        if latest_manifest is not None
        else self.baseline_checkpoint_hash
    )
    return HazardDetection(
        task_type=selection.task_type,
        dataset_partition=selection.partition,
        task_id=task.task_id,
        site_id=task.site_id,
        challenge_nonce=challenge_nonce,
        image_b64=base64.b64encode(task.image_bytes).decode("ascii"),
        image_format=task.image_format,
        requested_model_hash=requested_hash,
    )
