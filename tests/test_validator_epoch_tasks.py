from __future__ import annotations

import asyncio
import hashlib
import io
import json
import random
from pathlib import Path

import pytest

from template.hazard.annotation_image_serve import build_camouflaged_annotation_images
from template.hazard.golden_injection import InjectionPlan
from template.hazard.image_corpus import (
    GoldenAnnotation,
    GoldenImage,
    ImageCorpus,
    ImageCorpusConfig,
    UnlabeledImage,
)
from template.protocol import AnnotationTask
from template.validator.dual_forward import (
    _build_training_pool,
    _download_miner_artifact_bytes,
    _parse_annotations_payload,
    _validate_response_shape,
)
from template.validator.epoch_tasks import (
    EpochTaskScheduler,
    TaskDeadlineExceeded,
    TaskResponseWindow,
)


class FakeClock:
    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}-{index:04d}" for index in range(count)]


@pytest.mark.parametrize(
    ("total", "golden_count", "expected_tasks", "final_size", "final_goldens"),
    [
        (600, 60, 20, 30, 3),
        (500, 50, 17, 20, 2),
    ],
)
def test_epoch_coverage_and_golden_apportionment(
    tmp_path, total, golden_count, expected_tasks, final_size, final_goldens
):
    golden = _ids("gold", golden_count)
    public = _ids("public", total - golden_count)
    scheduler = EpochTaskScheduler(
        state_path=tmp_path / "epoch.json",
        golden_image_ids=golden,
        public_image_ids=public,
        golden_ratio=0.1,
        seed_factory=lambda: 41,
        epoch_id_factory=lambda: "epoch-a",
    )

    tasks = scheduler.task_plan()
    assert len(tasks) == expected_tasks
    assert [len(task.ordered_image_ids) for task in tasks[:-1]] == [30] * (expected_tasks - 1)
    assert len(tasks[-1].ordered_image_ids) == final_size
    assert len(tasks[-1].golden_image_ids) == final_goldens
    assert sum(len(task.golden_image_ids) for task in tasks) == golden_count
    assert sum(len(task.public_image_ids) for task in tasks) == total - golden_count
    assert all(len(task.ordered_image_ids) <= 30 for task in tasks)
    assert all(len(task.ordered_image_ids) == len(set(task.ordered_image_ids)) for task in tasks)
    all_ids = [image_id for task in tasks for image_id in task.ordered_image_ids]
    assert len(all_ids) == total
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == set(golden) | set(public)
    assert set(image_id for task in tasks for image_id in task.golden_image_ids) == set(golden)
    assert set(image_id for task in tasks for image_id in task.public_image_ids) == set(public)


def test_resume_keeps_seed_cursor_and_does_not_repeat_claimed_images(tmp_path):
    state = tmp_path / "validator" / "epoch.json"
    golden, public = _ids("gold", 9), _ids("public", 81)
    original = EpochTaskScheduler(
        state_path=state,
        golden_image_ids=golden,
        public_image_ids=public,
        golden_ratio=0.1,
        seed_factory=lambda: 1234,
        epoch_id_factory=lambda: "stable-epoch",
    )
    expected = original.task_plan()
    first = original.claim_next()
    original.close_task(first.task_id)
    interrupted = original.claim_next()
    original.mark_dispatched(interrupted.task_id)
    assert interrupted.index == 1
    # Simulate process restart with the second task persisted as in-flight.
    resumed = EpochTaskScheduler(
        state_path=state,
        golden_image_ids=golden,
        public_image_ids=public,
        golden_ratio=0.1,
        seed_factory=lambda: 9999,
        epoch_id_factory=lambda: "should-not-replace-loaded-epoch",
    )
    assert resumed.seed == 1234
    assert resumed.epoch_id == "stable-epoch"
    assert resumed.cursor == 2
    assert resumed.active_task_id is None
    assert resumed._state["abandoned_after_restart"] == 1
    assert resumed._state["abandoned_task_ids"] == [interrupted.task_id]
    third = resumed.claim_next()
    assert third == expected[2]
    assert set(first.ordered_image_ids).isdisjoint(third.ordered_image_ids)
    assert set(interrupted.ordered_image_ids).isdisjoint(third.ordered_image_ids)

    consumed = set(first.ordered_image_ids) | set(interrupted.ordered_image_ids)
    consumed.update(third.ordered_image_ids)
    resumed.close_task(third.task_id)
    while resumed.cursor < resumed.task_count:
        task = resumed.claim_next()
        assert consumed.isdisjoint(task.ordered_image_ids)
        consumed.update(task.ordered_image_ids)
        resumed.close_task(task.task_id)
    assert consumed == set(golden) | set(public)
    assert resumed.close_task(third.task_id) is False


def test_restart_rewinds_a_claim_that_was_never_dispatched(tmp_path):
    state = tmp_path / "epoch.json"
    scheduler = EpochTaskScheduler(
        state_path=state,
        golden_image_ids=["g1"],
        public_image_ids=["p1", "p2"],
        golden_ratio=1 / 3,
        seed_factory=lambda: 5,
        epoch_id_factory=lambda: "epoch-pre-dispatch",
    )
    expected = scheduler.task_plan()[0]
    claimed = scheduler.claim_next()
    assert claimed == expected

    resumed = EpochTaskScheduler(
        state_path=state,
        golden_image_ids=["g1"],
        public_image_ids=["p1", "p2"],
        golden_ratio=1 / 3,
        seed_factory=lambda: 99,
        epoch_id_factory=lambda: "unused",
    )
    assert resumed.cursor == 0
    assert resumed._state["released_before_dispatch"] == 1
    assert resumed.claim_next() == expected
    assert resumed.release_unstarted_task(expected.task_id) is True
    assert resumed.cursor == 0


def test_changed_dataset_fails_closed_while_epoch_is_in_progress(tmp_path):
    path = tmp_path / "epoch.json"
    EpochTaskScheduler(
        state_path=path,
        golden_image_ids=["g1"],
        public_image_ids=["p1"],
        golden_ratio=0.5,
        seed_factory=lambda: 1,
        epoch_id_factory=lambda: "epoch",
    )
    with pytest.raises(ValueError, match="loaded dataset differs"):
        EpochTaskScheduler(
            state_path=path,
            golden_image_ids=["g1"],
            public_image_ids=["p1", "p2"],
            golden_ratio=0.5,
            seed_factory=lambda: 2,
            epoch_id_factory=lambda: "other",
        )


def _build_local_corpus(tmp_path: Path) -> tuple[ImageCorpus, list[str], list[str]]:
    from PIL import Image

    cache = tmp_path / "cache"
    corpus = ImageCorpus(ImageCorpusConfig(cache_root=cache))
    corpus._loaded = True
    goldens: list[str] = []
    public: list[str] = []
    for index, is_golden in enumerate((True, False, True, False)):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (index * 20, 10, 30)).save(buffer, format="PNG")
        raw = buffer.getvalue()
        image_id = hashlib.sha256(raw).hexdigest()
        image_path = cache / f"{image_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(raw)
        corpus._all_image_index[image_id] = image_path
        if is_golden:
            image = GoldenImage(
                image_id=image_id,
                image_path=image_path,
                image_url=image_path.as_uri(),
                width=8,
                height=8,
                annotations=(
                    GoldenAnnotation(
                        hazard_class="golden_private_label",
                        bounding_box=(1, 1, 5, 5),
                        severity="low",
                    ),
                ),
            )
            corpus._golden.append(image)
            corpus._golden_index[image_id] = image
            goldens.append(image_id)
        else:
            corpus._annotation.append(
                UnlabeledImage(
                    image_id=image_id,
                    image_path=image_path,
                    image_url=image_path.as_uri(),
                    width=8,
                    height=8,
                    source_dataset="test",
                )
            )
            public.append(image_id)
    return corpus, goldens, public


def test_sampled_miners_get_same_canonical_task_images_with_opaque_ids(
    tmp_path, monkeypatch
):
    import template.hazard.r2_storage as r2_storage

    monkeypatch.setattr(
        r2_storage,
        "load_r2_credentials_from_env",
        lambda: (_ for _ in ()).throw(RuntimeError("disabled in unit test")),
    )
    corpus, goldens, public = _build_local_corpus(tmp_path)
    scheduler = EpochTaskScheduler(
        state_path=tmp_path / "epoch.json",
        golden_image_ids=goldens,
        public_image_ids=public,
        golden_ratio=0.5,
        seed_factory=lambda: 7,
        epoch_id_factory=lambda: "epoch-safe",
    )
    task = scheduler.claim_next()
    plan = InjectionPlan(
        ordered_images=tuple(
            (image_id, corpus.known_image_path(image_id).as_uri())
            for image_id in task.ordered_image_ids
        ),
        golden_image_ids=task.golden_image_ids,
        annotation_image_ids=task.public_image_ids,
    )
    public_training_image = corpus._annotation[0]
    corpus._training_pool.append(
        GoldenImage(
            image_id=public_training_image.image_id,
            image_path=public_training_image.image_path,
            image_url=public_training_image.image_url,
            width=public_training_image.width,
            height=public_training_image.height,
            annotations=(
                GoldenAnnotation(
                    hazard_class="public_training_label",
                    bounding_box=(1, 1, 5, 5),
                    severity="low",
                ),
            ),
        )
    )
    training_pool = _build_training_pool(corpus, "")
    assert all(item.image_id not in goldens for item in training_pool)

    exposed = []
    for miner_uid in (8, 21, 34):
        ephemeral: list[Path] = []
        reverse_map: dict[str, str] = {}
        images = asyncio.run(
            build_camouflaged_annotation_images(
                corpus=corpus,
                plan=plan,
                cache_root=tmp_path / f"miner-{miner_uid}",
                step=0,
                uid=miner_uid,
                rng=random.Random(miner_uid),
                serving_base_url="",
                jitter_ms_max=0,
                ephemeral_paths=ephemeral,
                mask_image_ids=True,
                token_to_real_id=reverse_map,
            )
        )
        synapse = AnnotationTask(
            task_id=task.task_id,
            challenge_nonce=f"nonce-{miner_uid}",
            annotation_images=images,
            training_pool=training_pool,
        )
        public_dump = json.dumps(synapse.model_dump(), sort_keys=True)
        assert "golden" not in public_dump.lower()
        assert "golden_private_label" not in public_dump
        assert "public_training_label" in public_dump
        assert "ground_truth_verified" not in public_dump
        assert "is_golden" not in public_dump
        assert all(image_id not in public_dump for image_id in goldens)
        annotation_dump = json.dumps(
            [image.model_dump() for image in synapse.annotation_images], sort_keys=True
        )
        assert all(image_id not in annotation_dump for image_id in task.ordered_image_ids)
        assert len({image.image_id for image in images}) == len(images)
        assert set(reverse_map.values()) == set(task.ordered_image_ids)
        canonical_order = tuple(reverse_map[image.image_id] for image in images)
        assert canonical_order == task.ordered_image_ids
        exposed.append(
            (
                synapse.task_id,
                canonical_order,
                {image.image_id for image in images},
            )
        )

    assert {task_id for task_id, _, _ in exposed} == {task.task_id}
    assert all(canonical_order == task.ordered_image_ids for _, canonical_order, _ in exposed)
    assert all(canonical_order == exposed[0][1] for _, canonical_order, _ in exposed[1:])
    assert len({tuple(sorted(opaque_ids)) for _, _, opaque_ids in exposed}) == len(exposed)
    scheduler.close_task(task.task_id)


def test_response_at_exact_deadline_is_rejected():
    clock = FakeClock(50.0)
    window = TaskResponseWindow(timeout_seconds=600, monotonic=clock)

    async def response():
        clock.advance(600.0)
        return "too-late"

    async def run():
        with pytest.raises(TaskDeadlineExceeded):
            await window.run(response(), stage="miner response")

    asyncio.run(run())
    assert window.close() is True
    assert window.close() is False


def test_artifact_retrieval_uses_remaining_task_window(tmp_path):
    clock = FakeClock()
    window = TaskResponseWindow(timeout_seconds=600, monotonic=clock)
    accepted = []
    artifact_path = tmp_path / "annotations.json"
    artifact_path.write_bytes(b'{"records": []}')

    async def run():
        async def dispatch():
            clock.advance(599.0)
            return "response"

        response = await window.run(dispatch(), stage="miner response")
        assert response == "response"

        async def delayed_download():
            clock.advance(1.0)
            return _download_miner_artifact_bytes(
                artifact_path.as_uri(), allow_file=True
            )

        artifact = await window.run(delayed_download(), stage="artifact retrieval")
        accepted.append(artifact)

    with pytest.raises(TaskDeadlineExceeded):
        asyncio.run(run())
    assert accepted == []
    window.close()


def test_real_timeout_cancels_pending_operation():
    cancelled = asyncio.Event()
    started = asyncio.Event()
    window = TaskResponseWindow(timeout_seconds=0.02, monotonic=FakeClock())

    async def never_finishes():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def run():
        with pytest.raises(TaskDeadlineExceeded):
            await window.run(never_finishes(), stage="miner response")
        await asyncio.sleep(0)

    asyncio.run(run())
    assert started.is_set()
    assert cancelled.is_set()


def test_close_quarantines_active_and_rejects_late_response():
    clock = FakeClock()
    window = TaskResponseWindow(timeout_seconds=600, monotonic=clock)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_response():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def run():
        task = asyncio.create_task(window.run(blocked_response(), stage="miner response"))
        await started.wait()
        assert window.close() is True
        with pytest.raises(TaskDeadlineExceeded):
            await task
        with pytest.raises(TaskDeadlineExceeded):
            await window.run(asyncio.sleep(0), stage="late response")
        await asyncio.sleep(0)

    asyncio.run(run())
    assert cancelled.is_set()


def test_invalid_payload_is_rejected_and_old_task_response_is_not_reused():
    with pytest.raises(ValueError, match="records must be a list"):
        _parse_annotations_payload(b'{"records":"invalid"}')

    from template.protocol import AnnotationTask

    old_response = AnnotationTask(
        task_id="old-epoch-task",
        challenge_nonce="old-nonce",
        annotations_uri="file:///tmp/old.json",
    )
    with pytest.raises(ValueError, match="Mismatched task_id"):
        _validate_response_shape(
            old_response,
            expected_task_id="new-epoch-task",
            expected_nonce="new-nonce",
        )
