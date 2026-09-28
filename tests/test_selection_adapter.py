from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from template.protocol import PerImageAnnotationItem
from template.validator.selection_adapter import (
    SelectedAnnotationRecord,
    SelectionBatchRequest,
    SelectionContractError,
    SelectionImageInput,
    SelectionPlan,
    SelectionEvaluatorUnavailable,
    SubmittedAnnotationRecord,
    apply_selection,
    expected_selection_allocation,
    hamilton_caps,
    validate_selection_plan,
)


class FakeEvaluator:
    def __init__(self, plan: SelectionPlan | None = None, error: Exception | None = None):
        self.plan = plan
        self.error = error
        self.requests = []

    async def select_batch(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        assert self.plan is not None
        return self.plan


def _record(uid: int, image_id: str, hazard_class: str, version: str):
    return SubmittedAnnotationRecord(
        uid=uid,
        image_id=image_id,
        annotations=(
            PerImageAnnotationItem(
                hazard_class=hazard_class,
                bounding_box=[1, 2, 7, 9],
                polygon=[[1, 2], [7, 2], [7, 9], [1, 9]],
                confidence=0.73,
            ),
        ),
        model_version=version,
    )


def _request():
    image_id = "public-image-a"
    return SelectionBatchRequest(
        task_id="epoch-task-0",
        images=(
            SelectionImageInput(
                image_id=image_id,
                image_path=Path("/tmp/public-image-a.png"),
                width=20,
                height=20,
                candidates_by_uid={
                    4: _record(4, image_id, "tree", "miner-a-v1"),
                    9: _record(9, image_id, "building", "miner-b-v2"),
                },
            ),
        ),
        batch_fidelity_by_uid={4: 0.8, 9: 0.7},
        temperature=0.2,
        floor=0.08,
        min_score=0.05,
    )


def _plan(request, selected_uid_by_image, quality_accepted_by_image=None):
    shares, caps = expected_selection_allocation(
        request.batch_fidelity_by_uid,
        public_image_count=len(request.images),
        temperature=request.temperature,
        floor=request.floor,
        min_score=request.min_score,
    )
    if quality_accepted_by_image is None:
        quality_accepted_by_image = {image_id: False for image_id in selected_uid_by_image}
    return SelectionPlan(shares, caps, selected_uid_by_image, quality_accepted_by_image)


def test_fake_evaluator_selects_exact_submitted_record_without_model_edits():
    request = _request()
    plan = _plan(request, {"public-image-a": 4}, {"public-image-a": True})
    evaluator = FakeEvaluator(plan)

    result = asyncio.run(apply_selection(evaluator, request))

    selected = result.selected_by_image["public-image-a"]
    submitted = request.images[0].candidates_by_uid[4]
    assert selected == SelectedAnnotationRecord(
        image_id="public-image-a",
        source_uid=4,
        annotations=submitted.annotations,
        model_version=submitted.model_version,
        policy_accepted=True,
    )
    assert result.unassigned_image_ids == ()
    assert evaluator.requests == [request]


def test_fake_evaluator_abstention_leaves_image_unassigned():
    request = _request()
    result = asyncio.run(
        apply_selection(
            FakeEvaluator(
                _plan(request, {"public-image-a": None})
            ),
            request,
        )
    )
    assert result.selected_by_image == {}
    assert result.unassigned_image_ids == ("public-image-a",)


@pytest.mark.parametrize(
    "selected_uid,quality_decision,match",
    [
        (4, {}, "separate quality decision"),
        (4, {"public-image-a": 1}, "must be booleans"),
        (None, {"public-image-a": True}, "cannot be quality-accepted"),
    ],
)
def test_quality_acceptance_is_explicit_and_strict(selected_uid, quality_decision, match):
    request = _request()
    with pytest.raises(SelectionContractError, match=match):
        validate_selection_plan(
            request,
            _plan(request, {"public-image-a": selected_uid}, quality_decision),
        )


def test_unknown_uid_is_rejected():
    request = _request()
    with pytest.raises(SelectionContractError, match="unknown UID"):
        asyncio.run(
            apply_selection(
                FakeEvaluator(
                    _plan(request, {"public-image-a": 17})
                ),
                request,
            )
        )


def test_adapter_rejects_cap_overflow():
    first = _request().images[0]
    second_id = "public-image-b"
    second = SelectionImageInput(
        image_id=second_id,
        image_path=Path("/tmp/public-image-b.png"),
        width=20,
        height=20,
        candidates_by_uid={
            4: _record(4, second_id, "tree", "miner-a-v1"),
            9: _record(9, second_id, "building", "miner-b-v2"),
        },
    )
    request = SelectionBatchRequest(
        task_id="epoch-task-1",
        images=(first, second),
        batch_fidelity_by_uid={4: 0.8, 9: 0.8},
        temperature=0.2,
        floor=0.08,
        min_score=0.05,
    )
    plan = _plan(request, {"public-image-a": 4, "public-image-b": 4})
    with pytest.raises(SelectionContractError, match="exceeded its task selection cap"):
        asyncio.run(apply_selection(FakeEvaluator(plan), request))


def test_adapter_rejects_invalid_selected_candidate_geometry():
    request = _request()
    image = request.images[0]
    original = image.candidates_by_uid[4]
    invalid_item = original.annotations[0].model_copy(
        update={"bounding_box": [-1, 2, 7, 9]}
    )
    invalid_record = SubmittedAnnotationRecord(
        uid=4,
        image_id=image.image_id,
        annotations=(invalid_item,),
        model_version=original.model_version,
    )
    invalid_image = SelectionImageInput(
        image_id=image.image_id,
        image_path=image.image_path,
        width=image.width,
        height=image.height,
        candidates_by_uid={4: invalid_record, 9: image.candidates_by_uid[9]},
    )
    invalid_request = SelectionBatchRequest(
        task_id=request.task_id,
        images=(invalid_image,),
        batch_fidelity_by_uid=request.batch_fidelity_by_uid,
        temperature=request.temperature,
        floor=request.floor,
        min_score=request.min_score,
    )
    with pytest.raises(SelectionContractError, match="invalid box geometry"):
        asyncio.run(
            apply_selection(
                FakeEvaluator(
                    _plan(invalid_request, {image.image_id: 4})
                ),
                invalid_request,
            )
        )


def test_absent_evaluator_and_evaluator_failure_do_not_fall_back():
    request = _request()
    with pytest.raises(SelectionEvaluatorUnavailable, match="legacy consensus fallback"):
        asyncio.run(apply_selection(None, request))
    with pytest.raises(RuntimeError, match="fake inference failure"):
        asyncio.run(apply_selection(FakeEvaluator(error=RuntimeError("fake inference failure")), request))


def test_validator_rejects_zero_fidelity_selection_and_forged_caps():
    request = _request()
    request = SelectionBatchRequest(
        task_id=request.task_id,
        images=request.images,
        batch_fidelity_by_uid={4: 0.8, 9: 0.0},
        temperature=request.temperature,
        floor=request.floor,
        min_score=request.min_score,
    )
    forged = SelectionPlan(
        shares_by_uid={4: 0.0, 9: 1.0},
        caps_by_uid={4: 0, 9: 1},
        selected_uid_by_image={"public-image-a": 9},
        quality_accepted_by_image={"public-image-a": False},
    )
    with pytest.raises(SelectionContractError, match="recomputation"):
        asyncio.run(apply_selection(FakeEvaluator(forged), request))


@pytest.mark.parametrize("score,eligible", [(0.199999, False), (0.20, True), (0.200001, True)])
def test_selection_eligibility_boundary_and_all_ineligible(score, eligible):
    shares, caps = expected_selection_allocation(
        {4: score}, public_image_count=1, temperature=0.2, floor=0.08, min_score=0.05
    )
    assert (shares[4] > 0.0) is eligible
    assert caps[4] == (1 if eligible else 0)


def test_hamilton_apportionment_examples_ties_and_variable_counts():
    assert hamilton_caps({11: 0.60, 22: 0.20, 33: 0.15, 44: 0.05}, 27) == {
        11: 16, 22: 6, 33: 4, 44: 1
    }
    assert hamilton_caps({9: 0.5, 3: 0.5}, 1) == {9: 0, 3: 1}
    for count in (0, 1, 3, 27, 30, 31):
        caps = hamilton_caps({1: 0.6, 2: 0.2, 3: 0.15, 4: 0.05}, count)
        assert sum(caps.values()) == count


def test_source_choice_is_not_implicitly_quality_accepted():
    request = _request()
    result = validate_selection_plan(
        request,
        _plan(request, {"public-image-a": 4}),
    )
    selected = result.selected_by_image["public-image-a"]
    assert selected.source_uid == 4
    assert selected.policy_accepted is False
    assert result.rejected_image_ids == ("public-image-a",)


def test_empty_record_remains_valid_when_policy_separately_accepts_it():
    request = _request()
    image = request.images[0]
    empty = SubmittedAnnotationRecord(4, image.image_id, (), "negative-image-model")
    empty_image = SelectionImageInput(
        image_id=image.image_id,
        image_path=image.image_path,
        width=image.width,
        height=image.height,
        candidates_by_uid={4: empty},
    )
    empty_request = SelectionBatchRequest(
        task_id=request.task_id,
        images=(empty_image,),
        batch_fidelity_by_uid={4: 0.8},
        temperature=request.temperature,
        floor=request.floor,
        min_score=request.min_score,
    )
    result = asyncio.run(
        apply_selection(FakeEvaluator(_plan(
            empty_request, {image.image_id: 4}, {image.image_id: True}
        )), empty_request)
    )
    selected = result.selected_by_image[image.image_id]
    assert selected.annotations == ()
    assert selected.policy_accepted is True
    assert selected.requires_escalation is False


def test_full_frame_and_duplicate_boxes_are_selected_only_for_escalation():
    request = _request()
    image_id = request.images[0].image_id
    full_frame = SubmittedAnnotationRecord(
        4, image_id,
        (PerImageAnnotationItem(hazard_class="tree", bounding_box=[0, 0, 20, 20]),),
        "full-frame",
    )
    duplicate_item = PerImageAnnotationItem(hazard_class="tree", bounding_box=[1, 1, 10, 10])
    duplicate = SubmittedAnnotationRecord(4, image_id, (duplicate_item, duplicate_item), "duplicate")
    for record, expected_reason in (
        (full_frame, "full_frame_box_requires_review"),
        (duplicate, "duplicate_annotation_requires_review"),
    ):
        image = SelectionImageInput(
            image_id, request.images[0].image_path, 20, 20, {4: record}
        )
        one_uid_request = SelectionBatchRequest(
            request.task_id, (image,), {4: 0.8}, request.temperature, request.floor, request.min_score
        )
        result = asyncio.run(
            apply_selection(
                FakeEvaluator(_plan(
                    one_uid_request, {image_id: 4}, {image_id: True}
                )), one_uid_request
            )
        )
        selected = result.selected_by_image[image_id]
        assert selected.policy_accepted is True
        assert selected.requires_escalation is True
        assert expected_reason in selected.escalation_reason
        assert result.escalated_image_ids == (image_id,)


def test_duplicate_image_ids_and_candidate_identity_mismatches_fail_closed():
    request = _request()
    image = request.images[0]
    duplicate_request = SelectionBatchRequest(
        request.task_id,
        (image, image),
        request.batch_fidelity_by_uid,
        request.temperature,
        request.floor,
        request.min_score,
    )
    with pytest.raises(SelectionContractError, match="duplicate public images"):
        asyncio.run(apply_selection(FakeEvaluator(_plan(duplicate_request, {image.image_id: 4})), duplicate_request))

    bad_record = SubmittedAnnotationRecord(9, image.image_id, image.candidates_by_uid[4].annotations, "bad-id")
    bad_image = SelectionImageInput(
        image.image_id, image.image_path, image.width, image.height, {4: bad_record}
    )
    bad_request = SelectionBatchRequest(
        request.task_id, (bad_image,), request.batch_fidelity_by_uid,
        request.temperature, request.floor, request.min_score,
    )
    with pytest.raises(SelectionContractError, match="identity"):
        asyncio.run(apply_selection(FakeEvaluator(_plan(bad_request, {image.image_id: 4})), bad_request))


@pytest.mark.parametrize(
    "annotation",
    [
        PerImageAnnotationItem.model_construct(
            hazard_class="", bounding_box=[1, 1, 10, 10], polygon=None,
            area=None, weight=None, confidence=None,
        ),
        PerImageAnnotationItem.model_construct(
            hazard_class="tree", bounding_box=[1, 1, 10, 10], polygon=[],
            area=None, weight=None, confidence=None,
        ),
        PerImageAnnotationItem.model_construct(
            hazard_class="tree", bounding_box=[1, 1, 10, 10],
            polygon=[[0, 0], [10, 1], [10, 10]],
            area=None, weight=None, confidence=None,
        ),
        PerImageAnnotationItem.model_construct(
            hazard_class="tree", bounding_box=[1, 1, 10, 10], polygon=None,
            area=None, weight=None, confidence=float("nan"),
        ),
    ],
)
def test_selection_adapter_rejects_schema_bypassed_candidate_geometry(annotation):
    request = _request()
    image = request.images[0]
    record = SubmittedAnnotationRecord(
        uid=4,
        image_id=image.image_id,
        annotations=(annotation,),
        model_version="malformed",
    )
    bad_image = SelectionImageInput(
        image_id=image.image_id,
        image_path=image.image_path,
        width=image.width,
        height=image.height,
        candidates_by_uid={4: record},
    )
    bad_request = SelectionBatchRequest(
        task_id=request.task_id,
        images=(bad_image,),
        batch_fidelity_by_uid=request.batch_fidelity_by_uid,
        temperature=request.temperature,
        floor=request.floor,
        min_score=request.min_score,
    )
    with pytest.raises(SelectionContractError):
        asyncio.run(
            apply_selection(
                FakeEvaluator(_plan(bad_request, {image.image_id: 4})),
                bad_request,
            )
        )
