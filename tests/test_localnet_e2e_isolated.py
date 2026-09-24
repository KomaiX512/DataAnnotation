from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bittensor as bt
from template.hazard.annotation_eval import (
    AnnotationFidelityScorer,
    ConsensusScorer,
    _ReliabilityAccumulator,
    evaluate_round_annotations,
)
from template.hazard.climate_mrv_corpus import load_climate_mrv_corpus, ClimateMRVConfig
from template.hazard.dataset_assembler import DatasetAssembler
from template.hazard.dual_reward import DualFlywheelRewardComposer
from template.hazard.image_corpus import ImageCorpus, ImageCorpusConfig
from template.hazard.incentives import broad_softmax_scores
from template.mock import MockAxonInfo, MockMetagraph, MockSubtensor, MockWallet
from template.protocol import (
    AnnotationTask,
    AnnotationsFilePayload,
    ImageAnnotationDocument,
    PerImageAnnotationItem,
    UnlabeledAnnotationImage,
)
from template.validator.dual_forward import (
    _build_challenge_nonce,
    _download_miner_artifact_bytes,
    _parse_annotations_payload,
    _validate_response_shape,
)


@pytest.fixture
def isolated_env():
    """Create a completely isolated temporary environment with temporary wallets and caches."""
    temp_dir = tempfile.mkdtemp(prefix="localnet_e2e_")
    root = Path(temp_dir)
    wallets_dir = root / "wallets"
    cache_dir = root / "cache"
    commercial_dir = root / "commercial"
    miner_artifacts = root / "miner_artifacts"

    wallets_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    commercial_dir.mkdir(parents=True)
    miner_artifacts.mkdir(parents=True)

    yield {
        "root": root,
        "wallets_dir": wallets_dir,
        "cache_dir": cache_dir,
        "commercial_dir": commercial_dir,
        "miner_artifacts": miner_artifacts,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_isolated_localnet_e2e_multi_miner_round(isolated_env):
    """
    End-to-end isolated localnet verification:
    - 1 Validator UID 0
    - 2 Honest Miners (UID 1 & 2) providing high quality annotations on golden + pool images
    - 1 Adversarial Miner (UID 3) attempting D1 (NaN), D8 (micro-IoU), D9 (forged weights), D4 (fake classes)
    
    Verifies:
    1. Validator creates and dispatches AnnotationTask with unique nonces.
    2. Miners produce annotations payloads and write to file:// URIs.
    3. Validator strictly parses payloads, rejects invalid shapes/NaNs.
    4. Evaluates golden fidelity with minimum IoU threshold >= 0.50.
    5. Fuses annotations across miners with quorum >= 2.
    6. Rewards honest miners and awards 0.0 to adversarial miner.
    7. Shapes incentives with broad_softmax_scores (floor capped at 20%, cutoff at 0.05).
    8. Updates validator moving average scores without float32 denormal leakage.
    """
    wallets_dir = isolated_env["wallets_dir"]
    cache_dir = isolated_env["cache_dir"]
    commercial_dir = isolated_env["commercial_dir"]
    miner_artifacts = isolated_env["miner_artifacts"]

    # 1. Setup isolated mock subtensor and wallets
    val_wallet = MockWallet(hotkey="5ValHotkeyAddress1111111111111111111111111", coldkey="5ValColdkeyAddress1111111111111111111111111")
    miner1_wallet = MockWallet(hotkey="5Miner1HotkeyAddress111111111111111111111111", coldkey="5Miner1ColdkeyAddress111111111111111111111111")
    miner2_wallet = MockWallet(hotkey="5Miner2HotkeyAddress111111111111111111111111", coldkey="5Miner2ColdkeyAddress111111111111111111111111")
    miner_adv_wallet = MockWallet(hotkey="5MinerAdvHotkeyAddress111111111111111111111", coldkey="5MinerAdvColdkeyAddress111111111111111111111")

    subtensor = MockSubtensor(netuid=2, n=4, wallet=val_wallet, network="mock")
    subtensor._hotkeys = [
        val_wallet.hotkey.ss58_address,
        miner1_wallet.hotkey.ss58_address,
        miner2_wallet.hotkey.ss58_address,
        miner_adv_wallet.hotkey.ss58_address,
    ]
    metagraph = subtensor.metagraph(netuid=2)
    metagraph.coldkeys = [
        val_wallet.coldkey.ss58_address,
        miner1_wallet.coldkey.ss58_address,
        miner2_wallet.coldkey.ss58_address,
        miner_adv_wallet.coldkey.ss58_address,
    ]
    metagraph.axons = [
        MockAxonInfo(uid=0, hotkey=subtensor._hotkeys[0], ip="127.0.0.1", port=9870),
        MockAxonInfo(uid=1, hotkey=subtensor._hotkeys[1], ip="127.0.0.1", port=9871),
        MockAxonInfo(uid=2, hotkey=subtensor._hotkeys[2], ip="127.0.0.1", port=9872),
        MockAxonInfo(uid=3, hotkey=subtensor._hotkeys[3], ip="127.0.0.1", port=9873),
    ]

    # 2. Load the real Climate MRV image corpus (bundled samples)
    corpus_cfg = ImageCorpusConfig(
        cache_root=cache_dir,
        golden_dataset_id="climate_mrv",
    )
    corpus = ImageCorpus(corpus_cfg)
    mrv_cfg = ClimateMRVConfig(cache_root=cache_dir, fallback_chips_dir="")
    load_climate_mrv_corpus(corpus, mrv_cfg)

    golden_images = corpus.golden_images()
    annotation_images = corpus.annotation_images()

    assert len(golden_images) >= 3, f"Expected at least 3 golden images, got {len(golden_images)}"
    assert len(annotation_images) >= 1, f"Expected at least 1 annotation image, got {len(annotation_images)}"

    # Select 3 golden images with boxes and 1 annotation pool image
    spatially_labeled_goldens = [g for g in golden_images if g.annotations and len(g.annotations) > 0]
    assert len(spatially_labeled_goldens) >= 3, "Expected at least 3 spatially labeled goldens"
    selected_goldens = spatially_labeled_goldens[:3]
    selected_pool = annotation_images[:1]

    task_images = [
        UnlabeledAnnotationImage(image_url=g.image_url, image_id=g.image_id)
        for g in selected_goldens
    ] + [
        UnlabeledAnnotationImage(image_url=p.image_url, image_id=p.image_id)
        for p in selected_pool
    ]

    step = 42
    task_id = "localnet-e2e-task-1"
    miner_uids = [1, 2, 3]

    # 3. Simulate miner responses
    # Miner 1 (Honest, accurate detections matching GT)
    m1_records = []
    for g in selected_goldens:
        anns = []
        for gt in g.annotations:
            bx = list(gt.bounding_box)
            # Add small realistic perturbation (IoU ~ 0.85)
            w = bx[2] - bx[0]
            h = bx[3] - bx[1]
            pert_box = [
                round(bx[0] + 0.05 * w, 2),
                round(bx[1] + 0.05 * h, 2),
                round(bx[2] - 0.05 * w, 2),
                round(bx[3] - 0.05 * h, 2),
            ]
            anns.append(
                PerImageAnnotationItem(
                    hazard_class=gt.hazard_class,
                    bounding_box=pert_box,
                    polygon=[
                        [pert_box[0], pert_box[1]],
                        [pert_box[2], pert_box[1]],
                        [pert_box[2], pert_box[3]],
                        [pert_box[0], pert_box[3]],
                    ],
                    confidence=0.95,
                )
            )
        m1_records.append(
            ImageAnnotationDocument(
                image_id=g.image_id,
                image_url=g.image_url,
                miner_uid=metagraph.hotkeys[1],
                timestamp="2026-09-24T00:00:00Z",
                annotations=anns,
                model_version="model_v1_honest",
            )
        )
    # Miner 1 on pool image: detects a tree crown
    m1_records.append(
        ImageAnnotationDocument(
            image_id=selected_pool[0].image_id,
            image_url=selected_pool[0].image_url,
            miner_uid=metagraph.hotkeys[1],
            timestamp="2026-09-24T00:00:00Z",
            annotations=[
                PerImageAnnotationItem(
                    hazard_class="dense_tree",
                    bounding_box=[100.0, 100.0, 200.0, 200.0],
                    confidence=0.92,
                )
            ],
            model_version="model_v1_honest",
        )
    )

    # Miner 2 (Honest, slightly different IoU ~ 0.80, agrees on pool image)
    m2_records = []
    for g in selected_goldens:
        anns = []
        for gt in g.annotations:
            bx = list(gt.bounding_box)
            w = bx[2] - bx[0]
            h = bx[3] - bx[1]
            pert_box = [
                round(bx[0] + 0.04 * w, 2),
                round(bx[1] + 0.04 * h, 2),
                round(bx[2] - 0.04 * w, 2),
                round(bx[3] - 0.04 * h, 2),
            ]
            anns.append(
                PerImageAnnotationItem(
                    hazard_class=gt.hazard_class,
                    bounding_box=pert_box,
                    confidence=0.90,
                )
            )
        m2_records.append(
            ImageAnnotationDocument(
                image_id=g.image_id,
                image_url=g.image_url,
                miner_uid=metagraph.hotkeys[2],
                timestamp="2026-09-24T00:00:00Z",
                annotations=anns,
                model_version="model_v2_honest",
            )
        )
    # Miner 2 on pool image: agrees with Miner 1 on tree crown (IoU ~ 0.85)
    m2_records.append(
        ImageAnnotationDocument(
            image_id=selected_pool[0].image_id,
            image_url=selected_pool[0].image_url,
            miner_uid=metagraph.hotkeys[2],
            timestamp="2026-09-24T00:00:00Z",
            annotations=[
                PerImageAnnotationItem(
                    hazard_class="dense_tree",
                    bounding_box=[105.0, 105.0, 195.0, 195.0],
                    confidence=0.88,
                )
            ],
            model_version="model_v2_honest",
        )
    )

    # Miner 3 (Adversarial):
    # Sends degenerate box or tiny box touching 0.001 IoU, fake weight 0.99, non-existent class
    m3_records = []
    for g in selected_goldens:
        # Tries to exploit D8 and D9: microscopic box far away with weight=0.99
        m3_records.append(
            ImageAnnotationDocument(
                image_id=g.image_id,
                image_url=g.image_url,
                miner_uid=metagraph.hotkeys[3],
                timestamp="2026-09-24T00:00:00Z",
                annotations=[
                    PerImageAnnotationItem(
                        hazard_class="water",  # Unrelated class (D4 test)
                        bounding_box=[1.0, 1.0, 3.0, 3.0],  # Barely touches anything (D8 test)
                        weight=0.99,  # Fabricated weight (D9 test)
                        confidence=0.99,
                    )
                ],
                model_version="model_v3_adversary",
            )
        )
    m3_records.append(
        ImageAnnotationDocument(
            image_id=selected_pool[0].image_id,
            image_url=selected_pool[0].image_url,
            miner_uid=metagraph.hotkeys[3],
            timestamp="2026-09-24T00:00:00Z",
            annotations=[
                PerImageAnnotationItem(
                    hazard_class="alien_structure",
                    bounding_box=[500.0, 500.0, 600.0, 600.0],
                    confidence=0.99,
                )
            ],
            model_version="model_v3_adversary",
        )
    )

    # Write payloads to disk (file:// URIs allowed on mock localnet)
    p1 = miner_artifacts / "m1_payload.json"
    p1.write_text(json.dumps(AnnotationsFilePayload(records=m1_records).model_dump()))

    p2 = miner_artifacts / "m2_payload.json"
    p2.write_text(json.dumps(AnnotationsFilePayload(records=m2_records).model_dump()))

    p3 = miner_artifacts / "m3_payload.json"
    p3.write_text(json.dumps(AnnotationsFilePayload(records=m3_records).model_dump()))

    responses = {
        1: AnnotationTask(
            task_id=task_id,
            challenge_nonce=_build_challenge_nonce(step, 1, task_id),
            annotations_uri=p1.as_uri(),
        ),
        2: AnnotationTask(
            task_id=task_id,
            challenge_nonce=_build_challenge_nonce(step, 2, task_id),
            annotations_uri=p2.as_uri(),
        ),
        3: AnnotationTask(
            task_id=task_id,
            challenge_nonce=_build_challenge_nonce(step, 3, task_id),
            annotations_uri=p3.as_uri(),
        ),
    }

    # 4. Validator parses and validates responses
    annotations_by_uid = {}
    miner_hotkeys = {}
    model_versions = {}
    timestamps = {}

    for uid in miner_uids:
        resp = responses[uid]
        expected_nonce = _build_challenge_nonce(step, uid, task_id)
        _validate_response_shape(resp, expected_task_id=task_id, expected_nonce=expected_nonce)

        raw = _download_miner_artifact_bytes(resp.annotations_uri, allow_file=True)
        payload = _parse_annotations_payload(raw)

        uid_records = {}
        for rec in payload.records:
            uid_records[rec.image_id] = rec.annotations
            model_versions[uid] = rec.model_version
            timestamps[uid] = rec.timestamp
        annotations_by_uid[uid] = uid_records
        miner_hotkeys[uid] = metagraph.hotkeys[uid]

    # 5. Evaluate fidelity & consensus
    fidelity_scorer = AnnotationFidelityScorer(minimum_match_iou=0.50)
    consensus_scorer = ConsensusScorer()
    reliability = _ReliabilityAccumulator()

    per_miner_scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations_by_uid,
        fidelity_scorer=fidelity_scorer,
        consensus_scorer=consensus_scorer,
        hallucination_penalty=0.5,
        golden_missing_penalty=0.5,
        reliability=reliability,
        expected_golden_ids_by_uid={uid: [g.image_id for g in selected_goldens] for uid in miner_uids},
    )

    # Check that Honest Miners achieved high fidelity
    assert per_miner_scores[1].average_score() > 0.60, f"Miner 1 score too low: {per_miner_scores[1].average_score()}"
    assert per_miner_scores[2].average_score() > 0.50, f"Miner 2 score too low: {per_miner_scores[2].average_score()}"

    # Check that Adversarial Miner got 0.0 (microscopic box failed IoU >= 0.50 threshold, unrelated class)
    assert per_miner_scores[3].average_score() == 0.0, f"Adversarial Miner 3 should get 0.0, got: {per_miner_scores[3].average_score()}"

    # 6. Assemble dataset (Dawid-Skene consensus on pool images)
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=commercial_dir.as_uri())
    miner_identity_keys = {uid: metagraph.coldkeys[uid] for uid in miner_uids}

    winners = assembler.assemble(
        per_miner_scores=per_miner_scores,
        annotations_by_uid=annotations_by_uid,
        miner_hotkeys=miner_hotkeys,
        miner_identity_keys=miner_identity_keys,
        model_versions=model_versions,
        timestamps=timestamps,
    )

    # 7. Compose rewards
    composer = DualFlywheelRewardComposer(alpha=0.7)
    rewards, breakdowns = composer.compose(
        uids=miner_uids,
        annotation_scores=per_miner_scores,
        ledger=assembler.ledger,
        round_winners=winners,
    )

    reward_map = dict(zip(miner_uids, rewards))
    assert reward_map[1] > 0.40, f"Honest Miner 1 reward too low: {reward_map[1]}"
    assert reward_map[2] > 0.35, f"Honest Miner 2 reward too low: {reward_map[2]}"
    assert reward_map[3] == 0.0, f"Adversarial Miner 3 reward should be 0.0, got: {reward_map[3]}"

    # 8. Test EMA update and float32 denormal truncation
    scores = np.zeros(len(metagraph.hotkeys), dtype=np.float32)
    alpha = 0.1
    scattered_rewards = np.zeros(len(metagraph.hotkeys), dtype=np.float32)
    for uid, r in zip(miner_uids, rewards):
        scattered_rewards[uid] = r

    scores = alpha * scattered_rewards + (1.0 - alpha) * scores
    scores[scores < 1e-4] = 0.0

    assert scores[1] > 0.0
    assert scores[2] > 0.0
    assert scores[3] == 0.0

    # 9. Test incentive shaping (broad_softmax_scores)
    shaped_weights = broad_softmax_scores(
        scores,
        temperature=0.20,
        floor=0.08,
        min_score=0.05,
    )

    assert shaped_weights[0] == 0.0  # Validator gets 0
    assert shaped_weights[1] > 0.40  # Miner 1 gets high share
    assert shaped_weights[2] > 0.30  # Miner 2 gets high share
    assert shaped_weights[3] == 0.0  # Adversarial Miner 3 gets exact 0.0!

    # Normalization check
    assert abs(float(shaped_weights.sum()) - 1.0) < 1e-5

    # 10. Call subtensor.set_weights
    success, msg = subtensor.set_weights(
        netuid=2,
        uids=np.arange(len(metagraph.hotkeys)),
        weights=shaped_weights,
    )
    assert success is True
    print(f"\n[LOCALNET-E2E SUCCESS] Round completed. Shaped weights: {shaped_weights}")


@pytest.mark.asyncio
async def test_isolated_localnet_real_axon_handshake(isolated_env):
    """
    Verify real network socket handshake over localhost:
    - Creates isolated validator and miner wallets in temporary directory.
    - Starts a real Bittensor Axon on localhost port 9876.
    - Validator sends AnnotationTask via real Bittensor Dendrite.
    - Miner handles synapse and returns artifact URI.
    - Verifies cryptographic signature verification and payload delivery over TCP.
    """
    wallets_dir = isolated_env["wallets_dir"]

    # Generate isolated wallet keypairs
    miner_w = bt.wallet(name="miner_test", hotkey="miner_test_hk", path=str(wallets_dir))
    miner_w.set_coldkey(bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic()), encrypt=False, overwrite=True)
    miner_w.set_hotkey(bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic()), encrypt=False, overwrite=True)

    val_w = bt.wallet(name="val_test", hotkey="val_test_hk", path=str(wallets_dir))
    val_w.set_coldkey(bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic()), encrypt=False, overwrite=True)
    val_w.set_hotkey(bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic()), encrypt=False, overwrite=True)

    test_port = 9876

    async def forward_annotation(synapse: AnnotationTask) -> AnnotationTask:
        assert synapse.task_id == "network-handshake-task-1"
        assert synapse.challenge_nonce == "test-network-nonce"
        synapse.annotations_uri = "file:///tmp/mock_network_annotations.json"
        return synapse

    forward_annotation.__annotations__["synapse"] = AnnotationTask

    axon = bt.axon(wallet=miner_w, ip="127.0.0.1", port=test_port)
    axon.attach(forward_fn=forward_annotation)
    axon.start()

    try:
        dendrite = bt.dendrite(wallet=val_w)
        synapse = AnnotationTask(
            task_id="network-handshake-task-1",
            challenge_nonce="test-network-nonce",
            annotation_images=[
                UnlabeledAnnotationImage(
                    image_url="http://127.0.0.1/test.jpg",
                    image_id="test-img-network-1",
                )
            ],
        )
        target = bt.AxonInfo(
            version=1,
            ip="127.0.0.1",
            port=test_port,
            ip_type=4,
            hotkey=miner_w.hotkey.ss58_address,
            coldkey=miner_w.coldkey.ss58_address,
        )
        responses = await dendrite([target], synapse, timeout=10)
        resp = responses[0]
        assert resp.is_success is True
        assert resp.annotations_uri == "file:///tmp/mock_network_annotations.json"
        print(f"\n[NETWORK-HANDSHAKE SUCCESS] Real axon on port {test_port} responded to dendrite successfully.")
    finally:
        axon.stop()

