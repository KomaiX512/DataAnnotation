"""Unit tests for scripts/validator_gate_report.py log analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validator_gate_report",
    _ROOT / "scripts" / "validator_gate_report.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["validator_gate_report"] = _mod
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
analyze_log = _mod.analyze_log
run_gates = _mod.run_gates


SAMPLE_LOG = r"""
2025-01-01 12:00:00.000 | INFO | event=dual_flywheel_round_start step=1 sampled_uids=[1, 2, 3]
2025-01-01 12:00:10.000 | INFO | event=dataset_assembled images=6 unique_winners=3
2025-01-01 12:00:11.000 | INFO | event=dual_flywheel_round_done step=1 uids=3 winners=3 rewards=[0.2, 0.3, 0.5]
2025-01-01 12:05:00.000 | INFO | event=dual_flywheel_round_start step=2 sampled_uids=[1, 2, 3]
2025-01-01 12:05:10.000 | INFO | event=dataset_assembled images=4 unique_winners=2
2025-01-01 12:05:11.000 | INFO | event=dual_flywheel_round_done step=2 uids=3 winners=2 rewards=[0.25, 0.35, 0.4]
2025-01-01 12:05:12.000 | INFO | set_weights on chain successfully!
"""


def test_analyze_log_round_metrics() -> None:
    s = analyze_log(SAMPLE_LOG)
    assert s["round_starts"] == 2
    assert s["round_dones"] == 2
    assert s["round_success_pct"] == 100.0
    assert s["invalid_responses"] == 0
    assert s["max_assembly_images"] == 6
    assert s["avg_round_duration_sec"] is not None
    assert s["p95_round_duration_sec"] is not None
    assert pytest.approx(s["reward_cv"], rel=1e-3) == 0.0


def test_gate_passes_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GATE_MIN_ROUND_SUCCESS_PCT", raising=False)
    stats = analyze_log(SAMPLE_LOG)
    rep = run_gates(stats)
    assert rep.all_passed()


def test_gate_fails_low_success(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = SAMPLE_LOG.replace("event=dual_flywheel_round_done", "event=dual_flywheel_round_done_XXX")
    stats = analyze_log(bad)
    monkeypatch.setenv("GATE_MIN_ROUND_SUCCESS_PCT", "70")
    rep = run_gates(stats)
    names = {g.name: g.passed for g in rep.results}
    assert names["round_success_pct"] is False


def test_allow_no_dual_flywheel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATE_ALLOW_NO_DUAL_FLYWHEEL", "1")
    stats = analyze_log("no dual flywheel here\n")
    rep = run_gates(stats)
    assert rep.all_passed()


def test_strict_no_local_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATE_ALLOW_LOCAL_SET_WEIGHTS_SKIP", "0")
    log = SAMPLE_LOG + "\nSkipping on-chain set_weights in local dev endpoint ws://127.0.0.1:9944\n"
    stats = analyze_log(log)
    rep = run_gates(stats)
    assert any(g.name == "no_local_set_weights_skip" and not g.passed for g in rep.results)
