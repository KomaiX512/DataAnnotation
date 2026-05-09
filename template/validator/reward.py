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
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from template.hazard.dataset import DatasetTask
from template.hazard.artifacts import ArtifactVerificationResult
from template.protocol import BoundingBox, HazardDetection


@dataclass(frozen=True)
class RewardBreakdown:
    inference_score: float
    training_score: float
    final_score: float


def score_inference(task: DatasetTask, response: HazardDetection) -> float:
    if task.task_type == "training":
        return 0.0
    if response.error_message:
        return 0.0
    detection = 1.0 if bool(response.hazard_detected) == task.hazard_detected else 0.0
    severity = 1.0 if response.severity == task.severity else 0.0
    localization = localization_score(task.expected_boxes, response.bounding_boxes)
    confidence = float(np.clip(response.confidence or 0.0, 0.0, 1.0))
    osha = osha_score(task.expected_osha_refs, response.osha_refs)
    rationale = rationale_score(response.rationale)
    score = (
        0.35 * detection
        + 0.2 * severity
        + 0.2 * localization
        + 0.1 * confidence
        + 0.1 * osha
        + 0.05 * rationale
    )
    return float(np.clip(score, 0.0, 1.0))


def score_training(artifact_result: ArtifactVerificationResult) -> float:
    return float(np.clip(artifact_result.score, 0.0, 1.0))


def merge_scores(inference_score: float, training_score: float) -> float:
    if inference_score == 0.0 and training_score > 0.0:
        return float(np.clip(training_score, 0.0, 1.0))
    return float(np.clip(0.6 * inference_score + 0.4 * training_score, 0.0, 1.0))


def get_rewards(
    tasks: Sequence[DatasetTask],
    responses: Sequence[HazardDetection],
    artifact_results: Sequence[ArtifactVerificationResult],
) -> tuple[np.ndarray, List[RewardBreakdown]]:
    if len(tasks) != len(responses) or len(tasks) != len(artifact_results):
        raise ValueError("tasks, responses, and artifact_results must have the same length")
    final_scores: List[float] = []
    breakdowns: List[RewardBreakdown] = []
    for task, response, artifact_result in zip(tasks, responses, artifact_results):
        inference_score = score_inference(task, response)
        training_score = score_training(artifact_result)
        final_score = merge_scores(inference_score, training_score)
        final_scores.append(final_score)
        breakdowns.append(
            RewardBreakdown(
                inference_score=inference_score,
                training_score=training_score,
                final_score=final_score,
            )
        )
    return np.asarray(final_scores, dtype=np.float32), breakdowns


def localization_score(
    expected: Sequence[dict],
    predicted: Sequence[BoundingBox],
) -> float:
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    expected_box = expected[0]
    predicted_box = predicted[0]
    x_left = max(float(expected_box["x_min"]), predicted_box.x_min)
    y_top = max(float(expected_box["y_min"]), predicted_box.y_min)
    x_right = min(float(expected_box["x_max"]), predicted_box.x_max)
    y_bottom = min(float(expected_box["y_max"]), predicted_box.y_max)
    intersection = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    expected_area = (
        max(0.0, float(expected_box["x_max"]) - float(expected_box["x_min"]))
        * max(0.0, float(expected_box["y_max"]) - float(expected_box["y_min"]))
    )
    predicted_area = max(0.0, predicted_box.x_max - predicted_box.x_min) * max(
        0.0, predicted_box.y_max - predicted_box.y_min
    )
    denominator = expected_area + predicted_area - intersection
    if denominator == 0:
        return 0.0
    return float(np.clip(intersection / denominator, 0.0, 1.0))


def osha_score(expected_refs: Sequence[str], predicted_refs: Sequence[str]) -> float:
    if not expected_refs:
        return 1.0
    expected = set(expected_refs)
    predicted = set(predicted_refs)
    if not predicted:
        return 0.0
    overlap = len(expected.intersection(predicted))
    return float(np.clip(overlap / len(expected), 0.0, 1.0))


def rationale_score(rationale: Optional[str]) -> float:
    if not rationale:
        return 0.0
    token_count = len(rationale.split())
    if token_count < 8:
        return 0.25
    if token_count < 16:
        return 0.6
    return 1.0
