"""
Dual-flywheel validator forward pass.

Each step the validator:

  1. Builds an AnnotationAndTrainingTask synapse for every queried miner that
     mixes ``golden_per_request`` Golden Set images with ``request_size -
     golden_per_request`` Annotation Pool images. Training images come from the
     labeled Training Pool. Canonical ``image_id`` values remain SHA-256(original
     corpus bytes) for scoring, while miners receive metadata-stripped JPEGs at
     opaque URLs plus per-image timing jitter so Golden rows are not trivially
     fingerprinted.

  2. Dispatches synapses to all selected miners in parallel via the dendrite
     and waits for their responses.

  3. For each response:
       * Downloads ``annotations.json`` from the miner's R2 bucket.
       * Validates per-image_id schema, signed model_version, hallucination caps.
       * Rejects duplicate annotation structures (per image and full payload)
         and checkpoint hashes already claimed by another UID this round / in
         the persisted registry.
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
    evaluate_round_annotations,
)
from template.hazard.dataset_assembler import DatasetAssembler
from template.hazard.dual_reward import DualFlywheelRewardComposer
from template.hazard.annotation_image_serve import (
    build_camouflaged_annotation_images,
    cleanup_ephemeral_annotation_files,
)
from template.hazard.golden_injection import GoldenInjector, InjectionPlan
from template.hazard.image_corpus import ImageCorpus, TrainingImage
from template.hazard.model_eval import ModelAccuracyComponents, ModelAccuracyEvaluator
from template.hazard.submission_dedup import AnnotationDuplicateTracker
from template.protocol import (
    AnnotationAndTrainingTask,
    AnnotationsFilePayload,
    LabeledTrainingImage,
    ModelCheckpoint,
    PerImageAnnotationItem,
    TrainingAnnotationLabel,
)
from template.utils.localnet_axon import localnet_miner_port_override
from template.utils.uids import get_random_uids


def _download_miner_artifact_bytes(uri: str) -> bytes:
    """Fetch ``annotations.json`` using ``file://`` or short-lived ``https://`` URLs only."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path).read_bytes()
    if parsed.scheme in ("http", "https"):
        from urllib.request import Request, urlopen

        req = Request(uri, headers={"User-Agent": "hazard-validator/1.0"})
        with urlopen(req, timeout=120) as resp:
            return resp.read()
    raise ValueError(
        f"Unsupported annotations_uri scheme {parsed.scheme!r}; "
        "miners must return file:// (local tests) or https:// presigned GET URLs."
    )


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

    ephemeral_annotation_files: List[Path] = []
    try:
        await _dual_flywheel_forward_impl(self, ephemeral_annotation_files)
    finally:
        cleanup_ephemeral_annotation_files(ephemeral_annotation_files)


async def _dual_flywheel_forward_impl(
    self,
    ephemeral_annotation_files: List[Path],
) -> None:
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
    jitter_ms_max = int(
        getattr(self.config.neuron, "flywheel_annotation_image_jitter_ms", 40) or 0
    )
    serving_base = str(getattr(self.config.neuron, "flywheel_image_serving_base_url", "") or "")

    plans_by_uid: Dict[int, InjectionPlan] = {}
    synapses_by_uid: Dict[int, AnnotationAndTrainingTask] = {}
    nonces_by_uid: Dict[int, str] = {}
    for uid in uids:
        plan = injector.build_plan(rng)
        plans_by_uid[uid] = plan
        task_id = f"flywheel-{self.step}-{uid}"
        nonce = _build_challenge_nonce(self.step, uid, task_id)
        ann_images = await build_camouflaged_annotation_images(
            corpus=corpus,
            plan=plan,
            cache_root=corpus.cache_root,
            step=int(self.step),
            uid=int(uid),
            rng=rng,
            serving_base_url=serving_base,
            jitter_ms_max=jitter_ms_max,
            ephemeral_paths=ephemeral_annotation_files,
        )
        synapses_by_uid[uid] = AnnotationAndTrainingTask(
            task_id=task_id,
            challenge_nonce=nonce,
            training_images=training_payload,
            annotation_images=ann_images,
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
    duplicate_tracker = AnnotationDuplicateTracker()

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
            raw = _download_miner_artifact_bytes(response.annotations_uri)
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

        ok_dedup, dedup_reason = duplicate_tracker.check_and_register(uid, valid_records)
        if not ok_dedup:
            bt.logging.error(
                f"event=dual_flywheel_duplicate_annotation_rejected uid={uid} detail={dedup_reason}"
            )
            continue

        manifest = response.submitted_training_manifest
        assert manifest is not None
        ok_model, model_dedup_reason = self.model_hash_registry.uid_may_use_model_hash(
            uid, manifest.candidate_model_hash
        )
        if not ok_model:
            bt.logging.error(
                f"event=dual_flywheel_duplicate_model_hash_rejected uid={uid} "
                f"detail={model_dedup_reason}"
            )
            continue

        try:
            accuracy = model_evaluator.evaluate(
                corpus=corpus,
                candidate_model_uri=manifest.candidate_model_uri,
                candidate_model_hash=manifest.candidate_model_hash,
            )
        except Exception as exc:
            bt.logging.error(f"event=dual_flywheel_model_eval_failure uid={uid} error={exc}")
            continue

        annotations_by_uid[uid] = valid_records
        miner_hotkeys[uid] = self.metagraph.hotkeys[uid] if uid < len(self.metagraph.hotkeys) else ""
        model_versions[uid] = manifest.candidate_model_hash
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
