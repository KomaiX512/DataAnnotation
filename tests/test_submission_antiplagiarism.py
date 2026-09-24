from __future__ import annotations

from template.hazard.submission_dedup import (
    AnnotationDuplicateTracker,
    fingerprint_annotation_items,
    full_submission_fingerprint,
)
from template.protocol import PerImageAnnotationItem


def _item(**kwargs):
    base = dict(
        hazard_class="missing_hardhat",
        bounding_box=[1, 2, 30, 40],
    )
    base.update(kwargs)
    return PerImageAnnotationItem(**base)


def test_fingerprint_stable_under_box_order():
    a = [_item(bounding_box=[0, 0, 10, 10]), _item(bounding_box=[5, 5, 20, 20], hazard_class="a")]
    b = list(reversed(a))
    assert fingerprint_annotation_items(a) == fingerprint_annotation_items(b)


def test_identical_nonempty_output_is_observed_but_not_rejected():
    tr = AnnotationDuplicateTracker()
    one = {"img1": [_item()]}
    assert tr.check_and_register(1, one)[0] is True
    accepted, reason = tr.check_and_register(2, one)
    assert accepted is True
    assert "similarity" in reason


def test_annotation_duplicate_tracker_allows_same_uid_distinct_payloads():
    tr = AnnotationDuplicateTracker()
    assert tr.check_and_register(1, {"img1": [_item()]})[0] is True
    assert tr.check_and_register(1, {"img1": [_item(bounding_box=[2, 3, 31, 41])]})[0] is True


def test_full_submission_duplicate():
    tr = AnnotationDuplicateTracker()
    payload = {"img1": [_item()], "img2": [_item(hazard_class="z")]}
    fp = full_submission_fingerprint(payload)
    assert fp == full_submission_fingerprint(payload)
    assert tr.check_and_register(3, payload)[0] is True
    accepted, reason = tr.check_and_register(4, payload)
    assert accepted is True
    assert "similarity" in reason


def test_empty_annotations_are_not_duplicate_evidence():
    tr = AnnotationDuplicateTracker()
    assert tr.check_and_register(1, {"img1": []}) == (True, "")
    assert tr.check_and_register(2, {"img1": []}) == (True, "")


def test_small_coordinate_jitter_is_flagged_without_penalty():
    tr = AnnotationDuplicateTracker()
    assert tr.check_and_register(
        1,
        {"img1": [_item(bounding_box=[100, 100, 300, 300])]},
    )[0]
    accepted, reason = tr.check_and_register(
        2,
        {"img1": [_item(bounding_box=[101, 101, 301, 301])]},
    )
    assert accepted is True
    assert "near-identical" in reason
