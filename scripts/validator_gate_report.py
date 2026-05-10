#!/usr/bin/env python3
"""
Formal release gate for validator logs (dual-flywheel + optional legacy cues).

Parses a validator log file and emits a pass/fail report with thresholds from
environment variables (defaults are conservative for localnet soak).

Exit codes:
  0 — all gates pass (or N/A where not applicable)
  1 — one or more gates failed
  2 — could not parse log / missing file

Usage:
  python scripts/validator_gate_report.py artifacts/operator_matrix/validator_STAMP.log

Environment (all optional):
  GATE_MIN_ROUND_SUCCESS_PCT       default 70.0   (100 * round_done / round_start)
  GATE_MAX_INVALID_RESPONSE_PCT    default 15.0   (100 * invalid / miner_invocations)
  GATE_MIN_MINER_INVOCATIONS       default 1      (round_starts * avg sample size)
  GATE_MAX_AVG_ROUND_DURATION_SEC  default 1200.0 (mean wall time start->done per step)
  GATE_P95_ROUND_DURATION_SEC      default 1800.0
  GATE_MIN_ASSEMBLY_IMAGES         default 1      (min images in any dataset_assembled when a round completes)
  GATE_MAX_REWARD_CV               default 0.85   (coefficient of variation of per-round reward totals; N/A if <2 rounds)
  GATE_REQUIRE_SET_WEIGHTS_SUCCESS default 0      (if 1, require set_weights success at least once)
  GATE_ALLOW_LOCAL_SET_WEIGHTS_SKIP default 1     (if 1, do not add strict no-skip gate)
  GATE_ZERO_TRANSPORT_ERRORS default 1            (if 1, fail on ClientConnectorError / TimeoutError in log)
  GATE_ALLOW_NO_DUAL_FLYWHEEL default 0           (if 1, skip round/assembly/reward gates when log has no dual_flywheel events)
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
ROUND_START_RE = re.compile(
    r"event=dual_flywheel_round_start step=(\d+) sampled_uids=\[([^\]]*)\]"
)
ROUND_DONE_RE = re.compile(
    r"event=dual_flywheel_round_done step=(\d+) uids=(\d+) winners=(\d+) rewards=\[(.*?)\]"
)
INVALID_RE = re.compile(r"event=dual_flywheel_invalid_response uid=(\d+) error=(.*)")
NO_VALID_RE = re.compile(r"event=dual_flywheel_no_valid_uids step=(\d+)")
ASSEMBLY_RE = re.compile(r"event=dataset_assembled images=(\d+) unique_winners=(\d+)")
SET_WEIGHTS_OK_RE = re.compile(r"set_weights on chain successfully!?", re.IGNORECASE)
SET_WEIGHTS_SKIP_RE = re.compile(r"Skipping on-chain set_weights in local dev endpoint")
CONNECTOR_RE = re.compile(r"ClientConnectorError")
TIMEOUT_RE = re.compile(r"TimeoutError#")


def _strip_line(line: str) -> str:
    return ANSI_RE.sub("", line)


def _parse_ts(line: str) -> Optional[datetime]:
    m = TS_RE.search(_strip_line(line))
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")


def _parse_reward_list(blob: str) -> List[float]:
    blob = blob.strip()
    if not blob:
        return []
    parts = [p.strip() for p in blob.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def _cv(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 1e-12:
        return float("inf")
    var = sum((x - mean) ** 2 for x in values) / max(len(values) - 1, 1)
    return math.sqrt(var) / mean


@dataclass
class GateResult:
    name: str
    passed: bool
    value: object
    threshold: object
    detail: str = ""


@dataclass
class GateReport:
    results: List[GateResult] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


def analyze_log(text: str) -> Dict[str, object]:
    lines = text.splitlines()
    round_starts: List[Tuple[int, datetime, int]] = []  # step, ts, k miners
    round_dones: List[Tuple[int, datetime, int, int, List[float]]] = []
    invalid = 0
    no_valid = 0
    assembly_images: List[int] = []
    set_weights_ok = 0
    set_weights_skip = 0
    connector = 0
    timeouts = 0

    for line in lines:
        raw = _strip_line(line)
        ts = _parse_ts(line)
        if SET_WEIGHTS_OK_RE.search(raw):
            set_weights_ok += 1
        if SET_WEIGHTS_SKIP_RE.search(raw):
            set_weights_skip += 1
        if CONNECTOR_RE.search(raw):
            connector += 1
        if TIMEOUT_RE.search(raw):
            timeouts += 1

        m = ROUND_START_RE.search(raw)
        if m and ts:
            step = int(m.group(1))
            uids_blob = m.group(2).strip()
            k = len([x for x in uids_blob.split(",") if x.strip()]) if uids_blob else 0
            round_starts.append((step, ts, k))

        m = ROUND_DONE_RE.search(raw)
        if m and ts:
            step = int(m.group(1))
            uids = int(m.group(2))
            winners = int(m.group(3))
            rewards = _parse_reward_list(m.group(4))
            round_dones.append((step, ts, uids, winners, rewards))

        if INVALID_RE.search(raw):
            invalid += 1
        if NO_VALID_RE.search(raw):
            no_valid += 1

        m = ASSEMBLY_RE.search(raw)
        if m:
            assembly_images.append(int(m.group(1)))

    # Pair latencies: same step
    start_by_step: Dict[int, datetime] = {}
    for step, ts, _k in round_starts:
        start_by_step[step] = ts
    durations: List[float] = []
    for step, ts_done, _uids, _winners, _rewards in round_dones:
        if step in start_by_step:
            dt = (ts_done - start_by_step[step]).total_seconds()
            if dt >= 0:
                durations.append(dt)

    miner_invocations = sum(k for _s, _t, k in round_starts)
    round_start_n = len(round_starts)
    round_done_n = len(round_dones)

    success_pct = (100.0 * round_done_n / round_start_n) if round_start_n else 0.0
    invalid_pct = (100.0 * invalid / miner_invocations) if miner_invocations else 0.0

    reward_totals = [sum(r) for _s, _t, _u, _w, r in round_dones if r]
    reward_cv = _cv(reward_totals)

    max_assembly = max(assembly_images) if assembly_images else 0

    return {
        "round_starts": round_start_n,
        "round_dones": round_done_n,
        "round_success_pct": success_pct,
        "invalid_responses": invalid,
        "no_valid_uids_events": no_valid,
        "miner_invocations": miner_invocations,
        "invalid_response_pct": invalid_pct,
        "avg_sample_size": (miner_invocations / round_start_n) if round_start_n else 0.0,
        "round_durations_sec": durations,
        "avg_round_duration_sec": (sum(durations) / len(durations)) if durations else None,
        "p95_round_duration_sec": _percentile(durations, 95) if durations else None,
        "assembly_image_counts": assembly_images,
        "max_assembly_images": max_assembly,
        "reward_totals": reward_totals,
        "reward_cv": reward_cv,
        "set_weights_success": set_weights_ok,
        "set_weights_local_skip": set_weights_skip,
        "connector_errors": connector,
        "timeout_errors": timeouts,
    }


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def run_gates(stats: Dict[str, object]) -> GateReport:
    report = GateReport(meta=stats)

    allow_no_dual = os.getenv("GATE_ALLOW_NO_DUAL_FLYWHEEL", "0").strip() in ("1", "true", "yes")
    zero_transport = os.getenv("GATE_ZERO_TRANSPORT_ERRORS", "1").strip() in ("1", "true", "yes")

    min_succ = float(os.getenv("GATE_MIN_ROUND_SUCCESS_PCT", "70"))
    max_inv_pct = float(os.getenv("GATE_MAX_INVALID_RESPONSE_PCT", "15"))
    min_inv = int(os.getenv("GATE_MIN_MINER_INVOCATIONS", "1"))
    max_avg_dur = float(os.getenv("GATE_MAX_AVG_ROUND_DURATION_SEC", "1200"))
    max_p95_dur = float(os.getenv("GATE_P95_ROUND_DURATION_SEC", "1800"))
    min_asm = int(os.getenv("GATE_MIN_ASSEMBLY_IMAGES", "1"))
    max_cv = float(os.getenv("GATE_MAX_REWARD_CV", "0.85"))
    require_sw = os.getenv("GATE_REQUIRE_SET_WEIGHTS_SUCCESS", "0").strip() in ("1", "true", "yes")
    allow_skip = os.getenv("GATE_ALLOW_LOCAL_SET_WEIGHTS_SKIP", "1").strip() in ("1", "true", "yes")

    rs = int(stats["round_starts"])
    rd = int(stats["round_dones"])
    succ_pct = float(stats["round_success_pct"])
    min_invoc = int(stats["miner_invocations"])
    inv_pct = float(stats["invalid_response_pct"])

    dual_active = rs > 0 or rd > 0
    report.results.append(
        GateResult(
            name="dual_flywheel_log_coverage",
            passed=dual_active or allow_no_dual,
            value={"round_starts": rs, "round_dones": rd},
            threshold=">=1 start or done unless GATE_ALLOW_NO_DUAL_FLYWHEEL=1",
            detail="" if dual_active else "legacy/empty log",
        )
    )

    skip_core = allow_no_dual and not dual_active

    report.results.append(
        GateResult(
            name="round_success_pct",
            passed=(succ_pct >= min_succ) if rs > 0 else (True if skip_core else False),
            value=succ_pct,
            threshold=f">= {min_succ}",
            detail="skipped (no dual events)" if skip_core else ("no round_start events" if rs == 0 else ""),
        )
    )

    report.results.append(
        GateResult(
            name="invalid_response_pct",
            passed=(inv_pct <= max_inv_pct) if min_invoc >= min_inv else True,
            value=inv_pct,
            threshold=f"<= {max_inv_pct} (needs miner_invocations>={min_inv})",
            detail=f"miner_invocations={min_invoc}" + ("; skipped (no dual)" if skip_core else ""),
        )
    )

    avg_dur = stats["avg_round_duration_sec"]
    p95_dur = stats["p95_round_duration_sec"]
    dur_ok = True
    if avg_dur is not None and avg_dur > max_avg_dur:
        dur_ok = False
    if p95_dur is not None and p95_dur > max_p95_dur:
        dur_ok = False
    has_dur = bool(stats["round_durations_sec"])
    report.results.append(
        GateResult(
            name="round_latency",
            passed=(dur_ok if has_dur else True) if not skip_core else True,
            value={"avg_sec": avg_dur, "p95_sec": p95_dur},
            threshold=f"avg<={max_avg_dur}s p95<={max_p95_dur}s",
            detail="N/A (no paired start/done)" if not has_dur and not skip_core else "",
        )
    )

    max_asm = int(stats["max_assembly_images"])
    asm_pass = (max_asm >= min_asm) if rd > 0 else (True if skip_core else False)
    report.results.append(
        GateResult(
            name="assembly_yield",
            passed=asm_pass,
            value=max_asm,
            threshold=f">= {min_asm} images in dataset_assembled when rounds complete",
            detail="no round_done" if rd == 0 and not skip_core else ("skipped (no dual)" if skip_core else ""),
        )
    )

    rtot: List[float] = stats["reward_totals"]  # type: ignore[assignment]
    cv = float(stats["reward_cv"])
    cv_pass = True
    if len(rtot) >= 2:
        cv_pass = cv <= max_cv
    report.results.append(
        GateResult(
            name="reward_stability_cv",
            passed=(cv_pass if len(rtot) >= 2 else True) if not skip_core else True,
            value=cv,
            threshold=f"<= {max_cv} (N/A if <2 completed rounds with rewards)",
            detail=f"rounds_with_totals={len(rtot)}",
        )
    )

    sw = int(stats["set_weights_success"])
    sw_pass = (sw > 0) if require_sw else True
    report.results.append(
        GateResult(
            name="set_weights_on_chain",
            passed=sw_pass,
            value=sw,
            threshold=">= 1 success" if require_sw else "not required",
            detail="GATE_REQUIRE_SET_WEIGHTS_SUCCESS=0" if not require_sw else "",
        )
    )

    if not allow_skip and int(stats["set_weights_local_skip"]) > 0:
        report.results.append(
            GateResult(
                name="no_local_set_weights_skip",
                passed=False,
                value=int(stats["set_weights_local_skip"]),
                threshold="0 skip events",
                detail="use testnet gate or FORCE_LOCAL_SET_WEIGHTS=1 on local",
            )
        )

    transport_ok = int(stats["connector_errors"]) == 0 and int(stats["timeout_errors"]) == 0
    report.results.append(
        GateResult(
            name="transport_health",
            passed=transport_ok if zero_transport else True,
            value={"connector": int(stats["connector_errors"]), "timeout": int(stats["timeout_errors"])},
            threshold="0 / 0" if zero_transport else "not enforced",
            detail="" if zero_transport else "GATE_ZERO_TRANSPORT_ERRORS=0",
        )
    )

    return report


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"GATE_FAIL: log not found: {path}", file=sys.stderr)
        return 2
    text = path.read_text(encoding="utf-8", errors="ignore")
    stats = analyze_log(text)
    report = run_gates(stats)

    out = {
        "log": str(path),
        "pass_all": report.all_passed(),
        "metrics": stats,
        "gates": [
            {
                "name": g.name,
                "passed": g.passed,
                "value": g.value,
                "threshold": g.threshold,
                "detail": g.detail,
            }
            for g in report.results
        ],
    }
    json_out = os.getenv("GATE_JSON_OUT", "").strip()
    if json_out:
        Path(json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))

    if not report.all_passed():
        failed = [g.name for g in report.results if not g.passed]
        print(f"\nRELEASE: NO-GO (failed: {', '.join(failed)})", file=sys.stderr)
        return 1
    print("\nRELEASE: GO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
