"""
Dual-flywheel validator forward pass.

Each step the validator:

  1. Builds an AnnotationAndTrainingTask synapse for every queried miner that
     mixes ``golden_per_request`` Golden Set images with ``request_size -
     golden_per_request`` Annotation Pool images. Training images come from the
     labeled Training Pool. Image IDs use SHA-256(image_bytes) -- per-image
     traceability across miners.

  2. Dispatches synapses to all selected miners in parallel via the dendrite
     and waits for their responses.

  3. For each response:
       * Downloads ``annotations.json`` from the miner's R2 bucket.
       * Validates per-image_id schema, signed model_version, hallucination caps.
       * Downloads the miner's fine-tuned checkpoint and scores it against the
         Golden Set + Cross-Domain Benchmark.

  4. Computes per-miner annotation fidelity (Golden) and consensus (non-Golden)
     scores, then assembles the highest-fidelity annotation per image_id into
     the commercial dataset.

  5. Combines annotation + model accuracy + adoption bonus into the final
     on-chain weight and persists the round artefacts.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import bittensor as bt

from template.hazard.annotation_eval import (
    AnnotationFidelityScorer,
    ConsensusScorer,
    PerMinerAnnotationScore,
    evaluate_round_annotations,
)
from template.hazard.dataset_assembler import DatasetAssembler, WinningAnnotation
from template.hazard.dual_reward import (
    DualFlywheelBreakdown,
    DualFlywheelRewardComposer,
)
from template.hazard.golden_injection import GoldenInjector, InjectionPlan
from template.hazard.image_corpus import ImageCorpus, TrainingImage
from template.hazard.model_eval import ModelAccuracyComponents, ModelAccuracyEvaluator
from template.protocol import (
    AnnotationAndTrainingTask,
    AnnotationsFilePayload,
    ImageAnnotationDocument,
    LabeledTrainingImage,
    ModelCheckpoint,
    PerImageAnnotationItem,
    R2AccessCredentials,
    TrainingAnnotationLabel,
    UnlabeledAnnotationImage,
)
from template.utils.localnet_axon import localnet_miner_port_override
from template.utils.uids import get_random_uids


# Annotations.json URI must be downloaded by the validator. We support file://
# (single-host development), r2:// and s3:// schemes.
def _download_uri_bytes(uri: str, *, creds: Optional[R2AccessCredentials] = None) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path).read_bytes()
    if parsed.scheme in ("r2", "s3"):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise ImportError("boto3 is required to read object-storage annotations.") from exc
        if creds is None:
            raise ValueError(f"Cannot download {uri} without R2/S3 credentials.")
        client = boto3.client(
            "s3",
            endpoint_url=creds.s3_endpoint,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            region_name="auto",
        )
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        result = client.get_object(Bucket=bucket, Key=key)
        return result["Body"].read()
    if parsed.scheme in ("http", "https"):
        from urllib.request import Request, urlopen

        req = Request(uri, headers={"User-Agent": "hazard-validator/1.0"})
        with urlopen(req, timeout=120) as resp:
            return resp.read()
    raise ValueError(f"Unsupported URI scheme for download: {uri}")


def _build_challenge_nonce(step: int, uid: int, task_id: str) -> str:
    return hashlib.sha256(f"{step}:{uid}:{task_id}".encode("utf-8")).hexdigest()[:16]


def _isoformat_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_target_axon(self, uid: int):
    """Localnet-aware axon resolver mirroring legacy forward."""
    axon = self.metagraph.axons[uid]
    subtensor_cfg = getattr(self.config, "subtensor", None)
    endpoint = str(getattr(subtensor_cfg, "chain_endpoint", ""))
    if not endpoint.startswith("ws://127.0.0.1"):
        return axon
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
        f"Localnet dual_flywheel axon uid={uid} hotkey={hk[:16]}... "
        f"chain_port={getattr(axon, 'port', None)} patched_port={patched.port}"
    )
    return patched


def _build_training_images(images: Sequence[TrainingImage]) -> List[LabeledTrainingImage]:
    payload: List[LabeledTrainingImage] = []
    for image in images:
        labels = [
            TrainingAnnotationLabel(
                hazard_class=ann.hazard_class,
                bounding_box=list(ann.bounding_box),
                severity=ann.severity,
                reasoning=ann.reasoning,
            )
            for ann in image.annotations
        ]
        if not labels:
            continue
        payload.append(
            LabeledTrainingImage(
                image_url=image.image_url,
                image_id=image.image_id,
                labels=labels,
            )
        )
    return payload


def _build_annotation_images(plan: InjectionPlan) -> List[UnlabeledAnnotationImage]:
    return [
        UnlabeledAnnotationImage(image_url=url, image_id=image_id)
        for image_id, url in plan.ordered_images
    ]


def _request_timeout(self) -> float:
    configured = float(getattr(self.config.neuron, "training_timeout", 0.0) or 0.0)
    if configured > 0.0:
        return configured
    max_training_seconds = float(
        getattr(self.config.neuron, "max_training_seconds", 60) or 60
    )
    base = float(self.config.neuron.timeout)
    return max(base, max_training_seconds + 300.0)


def _parse_annotations_payload(raw: bytes) -> AnnotationsFilePayload:
    text = raw.decode("utf-8")
    data = json.loads(text)
    return AnnotationsFilePayload.model_validate(data)


def _validate_response_shape(
    response: AnnotationAndTrainingTask,
    *,
    expected_task_id: str,
    expected_nonce: str,
    base_model_hash: str,
) -> None:
    if (response.task_id or "") != expected_task_id:
        raise ValueError(
            f"Mismatched task_id in miner response: expected={expected_task_id} got={response.task_id}"
        )
    if (response.challenge_nonce or "") != expected_nonce:
        raise ValueError("Challenge nonce mismatch in miner response.")
    if response.error_message:
        raise ValueError(f"Miner reported error: {response.error_message}")
    if not response.annotations_uri:
        raise ValueError("Miner response missing annotations_uri.")
    if not response.model_checkpoint_uri:
        raise ValueError("Miner response missing model_checkpoint_uri.")
    manifest = response.submitted_training_manifest
    if manifest is None:
        raise ValueError("Miner response missing submitted_training_manifest.")
    if manifest.parent_model_hash != base_model_hash:
        raise ValueError(
            f"Manifest parent_model_hash {manifest.parent_model_hash} does not match "
            f"baseline {base_model_hash}."
        )
    if not _looks_like_hex_digest(manifest.candidate_model_hash):
        raise ValueError("Invalid candidate_model_hash in submitted manifest.")
    if not _looks_like_hex_digest(manifest.config_hash):
        raise ValueError("Invalid config_hash in submitted manifest.")
    if not _looks_like_hex_digest(manifest.dataset_lineage_hash):
        raise ValueError("Invalid dataset_lineage_hash in submitted manifest.")


def _looks_like_hex_digest(value: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{16,128}", value or ""))


async def dual_flywheel_forward(self) -> None:
    """Validator entrypoint for the dual-flywheel mode."""

    corpus: ImageCorpus = self.image_corpus
    corpus.ensure_loaded()
    fidelity_scorer: AnnotationFidelityScorer = self.fidelity_scorer
    consensus_scorer: ConsensusScorer = self.consensus_scorer
    model_evaluator: ModelAccuracyEvaluator = self.model_evaluator
    assembler: DatasetAssembler = self.dataset_assembler
    composer: DualFlywheelRewardComposer = self.reward_composer
    injector: GoldenInjector = self.golden_injector

    self_uid = getattr(self, "uid", None)
    exclude = [int(self_uid)] if self_uid is not None and int(self_uid) >= 0 else None
    uids = get_random_uids(self, k=self.config.neuron.sample_size, exclude=exclude).tolist()
    if not uids:
        bt.logging.warning("event=dual_flywheel_no_uids")
        return

    bt.logging.info(
        f"event=dual_flywheel_round_start step={self.step} sampled_uids={uids} "
        f"request_size={injector.request_size} golden_per_req={injector.golden_per_request}"
    )

    # Pre-build the labeled training image bundle (same for all miners this step).
    training_count = max(1, int(self.config.neuron.flywheel_training_images_per_request))
    training_pool = corpus.training_images()
    if not training_pool:
        raise RuntimeError("Training pool is empty; dual-flywheel cannot proceed.")
    rng = self.random
    chosen_training = rng.sample(
        training_pool, min(training_count, len(training_pool))
    )
    training_payload = _build_training_images(chosen_training)
    if not training_payload:
        raise RuntimeError(
            "All sampled training images had zero usable labels; check golden_ratio/seed."
        )

    base_model_hash = self.baseline_checkpoint_hash
    timeout = _request_timeout(self)

    plans_by_uid: Dict[int, InjectionPlan] = {}
    synapses_by_uid: Dict[int, AnnotationAndTrainingTask] = {}
    nonces_by_uid: Dict[int, str] = {}
    for uid in uids:
        plan = injector.build_plan(rng)
        plans_by_uid[uid] = plan
        task_id = f"flywheel-{self.step}-{uid}"
        nonce = _build_challenge_nonce(self.step, uid, task_id)
        synapses_by_uid[uid] = AnnotationAndTrainingTask(
            task_id=task_id,
            challenge_nonce=nonce,
            training_images=training_payload,
            annotation_images=_build_annotation_images(plan),
            base_model_hash=base_model_hash,
            baseline_checkpoint=ModelCheckpoint(
                uri=self.config.neuron.baseline_checkpoint_uri,
                sha256=base_model_hash,
            ),
            max_training_seconds=int(self.config.neuron.max_training_seconds),
        )
        nonces_by_uid[uid] = nonce

    # Dispatch all miners in parallel.
    async def _dispatch(uid: int) -> Tuple[int, Optional[AnnotationAndTrainingTask]]:
        synapse = synapses_by_uid[uid]
        target = _resolve_target_axon(self, uid)
        try:
            responses = await self.dendrite(
                axons=[target],
                synapse=synapse,
                timeout=timeout,
                deserialize=True,
            )
            response = responses[0] if responses else None
            if response is None:
                raise RuntimeError("Empty miner response.")
            return uid, response
        except Exception as exc:  # pragma: no cover - network-driven
            bt.logging.error(f"event=dual_flywheel_dispatch_failure uid={uid} error={exc}")
            return uid, None

    coros = [_dispatch(uid) for uid in uids]
    raw_results = await asyncio.gather(*coros)

    annotations_by_uid: Dict[int, Dict[str, List[PerImageAnnotationItem]]] = {}
    miner_hotkeys: Dict[int, str] = {}
    model_versions: Dict[int, str] = {}
    timestamps: Dict[int, str] = {}
    model_accuracy: Dict[int, ModelAccuracyComponents] = {}
    valid_uids: List[int] = []

    for uid, response in raw_results:
        if response is None:
            continue
        synapse = synapses_by_uid[uid]
        try:
            _validate_response_shape(
                response,
                expected_task_id=synapse.task_id,
                expected_nonce=nonces_by_uid[uid],
                base_model_hash=base_model_hash,
            )
        except Exception as exc:
            bt.logging.error(f"event=dual_flywheel_invalid_response uid={uid} error={exc}")
            continue

        try:
            raw = _download_uri_bytes(
                response.annotations_uri, creds=response.miner_r2_credentials
            )
            payload = _parse_annotations_payload(raw)
        except Exception as exc:
            bt.logging.error(f"event=dual_flywheel_annotations_download_failure uid={uid} error={exc}")
            continue

        valid_records: Dict[str, List[PerImageAnnotationItem]] = {}
        expected_ids = {image_id for image_id, _ in plans_by_uid[uid].ordered_images}
        for record in payload.records:
            if record.image_id not in expected_ids:
                bt.logging.warning(
                    f"event=dual_flywheel_unexpected_image_id uid={uid} image_id={record.image_id}"
                )
                continue
            valid_records[record.image_id] = list(record.annotations)
        if not valid_records:
            bt.logging.warning(f"event=dual_flywheel_no_valid_records uid={uid}")
            continue

        try:
            accuracy = model_evaluator.evaluate(
                corpus=corpus,
                candidate_model_uri=response.submitted_training_manifest.candidate_model_uri,
                candidate_model_hash=response.submitted_training_manifest.candidate_model_hash,
                miner_r2_credentials=response.miner_r2_credentials,
            )
        except Exception as exc:
            bt.logging.error(f"event=dual_flywheel_model_eval_failure uid={uid} error={exc}")
            continue

        annotations_by_uid[uid] = valid_records
        miner_hotkeys[uid] = self.metagraph.hotkeys[uid] if uid < len(self.metagraph.hotkeys) else ""
        model_versions[uid] = response.submitted_training_manifest.candidate_model_hash
        timestamps[uid] = _isoformat_utc()
        model_accuracy[uid] = accuracy
        valid_uids.append(uid)

    if not valid_uids:
        bt.logging.warning("event=dual_flywheel_no_valid_uids step=%d" % self.step)
        return

    per_miner_scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations_by_uid,
        fidelity_scorer=fidelity_scorer,
        consensus_scorer=consensus_scorer,
        hallucination_penalty=composer.hallucination_penalty_per_event,
    )

    winners = assembler.assemble(
        per_miner_scores=per_miner_scores,
        annotations_by_uid=annotations_by_uid,
        miner_hotkeys=miner_hotkeys,
        model_versions=model_versions,
        timestamps=timestamps,
    )

    rewards, breakdowns = composer.compose(
        uids=valid_uids,
        annotation_scores=per_miner_scores,
        model_accuracy=model_accuracy,
        ledger=assembler.ledger,
        round_winners=winners,
    )

    self.update_scores(rewards, valid_uids)
    self.update_score_ledgers(
        breakdowns,
        valid_uids,
        model_accuracy=model_accuracy,
        promotion_model_hashes=model_versions,
    )

    if winners and self._should_export_commercial(self.step):
        try:
            commercial_creds = self._load_commercial_credentials()
            uri = assembler.export(
                winners,
                round_id=f"step-{self.step}",
                commercial_r2_credentials=commercial_creds,
            )
            self.last_commercial_dataset_uri = uri
            bt.logging.info(f"event=dual_flywheel_commercial_export uri={uri}")
        except Exception as exc:
            bt.logging.error(f"event=dual_flywheel_commercial_export_failure error={exc}")

    bt.logging.info(
        "event=dual_flywheel_round_done step=%d uids=%d winners=%d rewards=%s"
        % (self.step, len(valid_uids), len(winners), rewards.tolist())
    )
