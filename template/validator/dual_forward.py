"""
Annotation-only validator forward pass.

Each step the validator:

  1. Builds one full-dataset ``AnnotationTask`` plan containing every image in
     the round corpus. A secret subset of those exact same images is Golden.

  2. Dispatches synapses to all selected miners in parallel.

  3. For each response, downloads and validates ``annotations.json``, rejects
     duplicate annotation structures.

  4. Computes per-miner annotation fidelity (Golden) and consensus (non-Golden),
     assembles the highest-fidelity annotation per image_id.

  5. Sets on-chain weights from annotation quality and adoption bonus only.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
from template.hazard.golden_injection import InjectionPlan
from template.hazard.image_corpus import ImageCorpus
from template.hazard.r2_storage import download_bytes_from_r2, load_r2_credentials_from_env
from template.hazard.submission_dedup import AnnotationDuplicateTracker
from template.protocol import (
    AnnotationTask,
    AnnotationsFilePayload,
    PerImageAnnotationItem,
    R2AccessCredentials,
)
from template.utils.localnet_axon import localnet_miner_port_override
from template.utils.uids import get_random_uids


def _download_miner_artifact_bytes(
    uri: str,
    *,
    miner_r2_credentials: R2AccessCredentials | None = None,
) -> bytes:
    """Fetch ``annotations.json`` from ``r2://``, ``https://``, or ``file://``."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path).read_bytes()
    if parsed.scheme in ("http", "https"):
        from template.utils.http_fetch import fetch_url_bytes

        return fetch_url_bytes(uri, timeout=120.0)
    if parsed.scheme == "r2":
        creds = miner_r2_credentials or load_r2_credentials_from_env()
        return download_bytes_from_r2(uri, creds=creds)
    raise ValueError(
        f"Unsupported annotations_uri scheme {parsed.scheme!r}; "
        "expected r2://, file:// (tests), or https://."
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
        f"Localnet annotation_flywheel axon uid={uid} hotkey={hk[:16]}... "
        f"chain_port={getattr(axon, 'port', None)} patched_port={patched.port}"
    )
    return patched


def _request_timeout(self) -> float:
    configured = float(getattr(self.config.neuron, "annotation_timeout", 0.0) or 0.0)
    if configured > 0.0:
        return configured
    return float(self.config.neuron.timeout)


def _parse_annotations_payload(raw: bytes) -> AnnotationsFilePayload:
    text = raw.decode("utf-8")
    data = json.loads(text)
    return AnnotationsFilePayload.model_validate(data)


def _validate_response_shape(
    response: AnnotationTask,
    *,
    expected_task_id: str,
    expected_nonce: str,
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


async def dual_flywheel_forward(self) -> None:
    """Validator entrypoint for annotation-only flywheel mode."""

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
    assembler: DatasetAssembler = self.dataset_assembler
    composer: DualFlywheelRewardComposer = self.reward_composer

    self_uid = getattr(self, "uid", None)
    exclude = [int(self_uid)] if self_uid is not None and int(self_uid) >= 0 else None
    uids = get_random_uids(self, k=self.config.neuron.sample_size, exclude=exclude).tolist()
    if not uids:
        bt.logging.warning("event=annotation_flywheel_no_uids")
        return

    bt.logging.info(
        f"event=annotation_flywheel_round_start step={self.step} sampled_uids={uids} "
        f"dataset_images={len(corpus.golden_images()) + len(corpus.annotation_images())} "
        f"golden_images={len(corpus.golden_images())}"
    )

    timeout = _request_timeout(self)
    jitter_ms_max = int(
        getattr(self.config.neuron, "flywheel_annotation_image_jitter_ms", 40) or 0
    )
    serving_base = str(getattr(self.config.neuron, "flywheel_image_serving_base_url", "") or "")
    rng = self.random

    full_plan = _build_full_dataset_plan(corpus)
    synapses_by_uid: Dict[int, AnnotationTask] = {}
    nonces_by_uid: Dict[int, str] = {}
    for uid in uids:
        task_id = f"flywheel-{self.step}-{uid}"
        nonce = _build_challenge_nonce(self.step, uid, task_id)
        ann_images = await build_camouflaged_annotation_images(
            corpus=corpus,
            plan=full_plan,
            cache_root=corpus.cache_root,
            step=int(self.step),
            uid=int(uid),
            rng=rng,
            serving_base_url=serving_base,
            jitter_ms_max=jitter_ms_max,
            ephemeral_paths=ephemeral_annotation_files,
        )
        synapses_by_uid[uid] = AnnotationTask(
            task_id=task_id,
            challenge_nonce=nonce,
            annotation_images=ann_images,
        )
        nonces_by_uid[uid] = nonce

    async def _dispatch(uid: int) -> Tuple[int, Optional[AnnotationTask]]:
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
            bt.logging.error(f"event=annotation_flywheel_dispatch_failure uid={uid} error={exc}")
            return uid, None

    raw_results = await asyncio.gather(*[_dispatch(uid) for uid in uids])

    annotations_by_uid: Dict[int, Dict[str, List[PerImageAnnotationItem]]] = {}
    miner_hotkeys: Dict[int, str] = {}
    model_versions: Dict[int, str] = {}
    timestamps: Dict[int, str] = {}
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
            )
        except Exception as exc:
            bt.logging.error(f"event=annotation_flywheel_invalid_response uid={uid} error={exc}")
            continue

        try:
            raw = _download_miner_artifact_bytes(
                response.annotations_uri,
                miner_r2_credentials=response.miner_r2_credentials,
            )
            payload = _parse_annotations_payload(raw)
        except Exception as exc:
            bt.logging.error(
                f"event=annotation_flywheel_annotations_download_failure uid={uid} error={exc}"
            )
            continue

        valid_records: Dict[str, List[PerImageAnnotationItem]] = {}
        expected_ids = {image_id for image_id, _ in full_plan.ordered_images}
        version_samples: list[str] = []
        for record in payload.records:
            if record.image_id not in expected_ids:
                bt.logging.warning(
                    f"event=annotation_flywheel_unexpected_image_id uid={uid} image_id={record.image_id}"
                )
                continue
            valid_records[record.image_id] = list(record.annotations)
            version_samples.append(record.model_version)
        if not valid_records:
            bt.logging.warning(f"event=annotation_flywheel_no_valid_records uid={uid}")
            continue

        ok_dedup, dedup_reason = duplicate_tracker.check_and_register(uid, valid_records)
        if not ok_dedup:
            bt.logging.error(
                f"event=annotation_flywheel_duplicate_annotation_rejected uid={uid} detail={dedup_reason}"
            )
            continue

        annotations_by_uid[uid] = valid_records
        miner_hotkeys[uid] = self.metagraph.hotkeys[uid] if uid < len(self.metagraph.hotkeys) else ""
        model_versions[uid] = version_samples[0] if version_samples else ""
        timestamps[uid] = _isoformat_utc()
        valid_uids.append(uid)

    if not valid_uids:
        bt.logging.warning("event=annotation_flywheel_no_valid_uids step=%d" % self.step)
        return

    per_miner_scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations_by_uid,
        fidelity_scorer=fidelity_scorer,
        consensus_scorer=consensus_scorer,
        hallucination_penalty=composer.hallucination_penalty_per_event,
        golden_missing_penalty=composer.golden_missing_penalty,
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
        ledger=assembler.ledger,
        round_winners=winners,
    )

    self.update_scores(rewards, valid_uids)
    self.update_score_ledgers(breakdowns, valid_uids)

    if winners and self._should_export_commercial(self.step):
        try:
            commercial_creds = self._load_commercial_credentials()
            uri = assembler.export(
                winners,
                round_id=f"step-{self.step}",
                commercial_r2_credentials=commercial_creds,
            )
            self.last_commercial_dataset_uri = uri
            bt.logging.info(f"event=annotation_flywheel_commercial_export uri={uri}")
        except Exception as exc:
            bt.logging.error(f"event=annotation_flywheel_commercial_export_failure error={exc}")

    bt.logging.info(
        "event=annotation_flywheel_round_done step=%d uids=%d winners=%d rewards=%s"
        % (self.step, len(valid_uids), len(winners), rewards.tolist())
    )


def _build_full_dataset_plan(corpus: ImageCorpus) -> InjectionPlan:
    ordered = [
        (image.image_id, image.image_url) for image in corpus.golden_images()
    ] + [
        (image.image_id, image.image_url) for image in corpus.annotation_images()
    ]
    ordered = sorted(ordered, key=lambda item: item[0])
    return InjectionPlan(
        ordered_images=tuple(ordered),
        golden_image_ids=tuple(image.image_id for image in corpus.golden_images()),
        annotation_image_ids=tuple(image.image_id for image in corpus.annotation_images()),
    )
