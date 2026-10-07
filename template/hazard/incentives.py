from __future__ import annotations

import math

import numpy as np

MIN_INCENTIVE_SCORE = 0.05
MAX_CUMULATIVE_FLOOR = 0.20
SELECTION_ELIGIBILITY_MIN_FIDELITY = 0.20
SELECTION_ELIGIBILITY_RAMP_FLOOR = 0.05


def selection_eligibility_multiplier(fidelity: float) -> float:
    """
    Continuous linear ramp for selection eligibility.
    fidelity >= SELECTION_ELIGIBILITY_MIN_FIDELITY (0.20) -> 1.0
    fidelity <= SELECTION_ELIGIBILITY_RAMP_FLOOR (0.05) -> 0.0
    Linear in-between: (fidelity - 0.05) / (0.20 - 0.05)
    """
    if not math.isfinite(fidelity) or fidelity <= SELECTION_ELIGIBILITY_RAMP_FLOOR:
        return 0.0
    if fidelity >= SELECTION_ELIGIBILITY_MIN_FIDELITY:
        return 1.0
    return float(
        (fidelity - SELECTION_ELIGIBILITY_RAMP_FLOOR)
        / (SELECTION_ELIGIBILITY_MIN_FIDELITY - SELECTION_ELIGIBILITY_RAMP_FLOOR)
    )


def broad_softmax_scores(
    scores: np.ndarray,
    *,
    temperature: float = 1.0,
    floor: float = 0.08,
    min_score: float = 0.05,
    proportional: bool = True,
) -> np.ndarray:
    """
    Convert raw miner value scores into a broad, fair incentive surface.

    Miners below min_score receive zero. Every miner above it receives the
    configured floor plus a proportionally shaped share, ensuring contributors
    earn rewards aligned with their fidelity ratio without artificial exponential cliffs.
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
    n_eligible = len(eligible_scores)
    effective_floor = floor
    if floor * n_eligible > MAX_CUMULATIVE_FLOOR:
        effective_floor = MAX_CUMULATIVE_FLOOR / n_eligible

    variable_mass = 1.0 - effective_floor * n_eligible

    if not proportional or temperature < 0.10:
        centered = eligible_scores - np.max(eligible_scores)
        exp_scores = np.exp(centered / temperature)
        ratio_shares = exp_scores / np.sum(exp_scores)
    else:
        # Fair Prosperity Mechanism: Proportional Fidelity Ratio
        eff_temp = max(0.5, float(temperature))
        if 0.95 <= eff_temp <= 1.05:
            ratio_shares = eligible_scores / np.sum(eligible_scores)
        else:
            power = max(0.5, min(2.0, 1.0 / eff_temp))
            powered = np.power(eligible_scores / np.max(eligible_scores), power)
            ratio_shares = powered / np.sum(powered)

    shaped_values = effective_floor + variable_mass * ratio_shares
    shaped_values = np.clip(shaped_values, 0.0, None)
    shaped[eligible] = shaped_values
    total = shaped.sum()
    if total > 0:
        shaped = shaped / total
    return shaped.astype(np.float32)
