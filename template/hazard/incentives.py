from __future__ import annotations

import math

import numpy as np

MIN_INCENTIVE_SCORE = 0.05
MAX_CUMULATIVE_FLOOR = 0.20


def broad_softmax_scores(
    scores: np.ndarray,
    *,
    temperature: float,
    floor: float,
    min_score: float,
) -> np.ndarray:
    """
    Convert raw miner value scores into a broad nonzero incentive surface.

    Miners below min_score receive zero. Every miner above it receives the
    configured floor plus a softmax-shaped share, so useful contributors are
    paid without flattening competition at the top.
    """

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("floor must be finite and non-negative")
    if not math.isfinite(min_score):
        raise ValueError("min_score must be finite")

    raw = np.asarray(scores, dtype=np.float64)
    cutoff = max(float(min_score), MIN_INCENTIVE_SCORE)
    eligible = np.isfinite(raw) & (raw >= cutoff) & (raw > 0.0)
    shaped = np.zeros_like(raw, dtype=np.float64)
    if not eligible.any():
        return shaped.astype(np.float32)

    eligible_scores = raw[eligible]
    centered = eligible_scores - np.max(eligible_scores)
    exp_scores = np.exp(centered / temperature)
    exp_scores = exp_scores / np.sum(exp_scores)

    # Dynamically scale floor to prevent negative multipliers when many miners are eligible
    n_eligible = len(exp_scores)
    effective_floor = floor
    if floor * n_eligible > MAX_CUMULATIVE_FLOOR:
        effective_floor = MAX_CUMULATIVE_FLOOR / n_eligible

    shaped_values = effective_floor + (1.0 - effective_floor * n_eligible) * exp_scores
    shaped_values = np.clip(shaped_values, 0.0, None)
    shaped[eligible] = shaped_values
    total = shaped.sum()
    if total > 0:
        shaped = shaped / total
    return shaped.astype(np.float32)
