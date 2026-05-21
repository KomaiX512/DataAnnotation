# Subnet Deployment Readiness Gates

This project uses a strict one-path flow (annotation upload + Golden Set scoring + commercial dataset assembly).  
Use this checklist to promote from localnet to testnet.

## Phase 1: Core loop sanity (localnet)

```bash
export GOLDEN_MAX_SAMPLES=8 RUN_SECONDS=1800
./scripts/run_localnet_e2e_real.sh
```

Pass criteria:
- `event=annotation_flywheel_round_start` appears.
- `event=annotation_flywheel_round_done ...` appears with non-zero rewards for responsive miners.
- No `Error during validation step`.

## Phase 2: Multi-miner + adversarial stress (localnet)

```bash
export RUN_SECONDS=1800
./scripts/run_localnet_stress_matrix.sh
```

Pass criteria:
- Validator remains alive (`step(...)` continues).
- Duplicate or malformed annotation payloads are rejected by integrity checks.
- Honest miners still receive non-zero annotation quality scores.

## Phase 2b: Probabilistic aggregation production readiness (≥10 miners)

See **`docs/PRODUCTION_READINESS_LOCALNET.md`** for the seven acceptance tests, golden-holdout and calibration evaluators, and commercial JSONL schema validation.

```bash
export RUN_SECONDS=3600 MIN_MINERS=10
./scripts/run_production_readiness_localnet.sh
```

Optional: set `GOLDEN_MANIFEST`, `COMMERCIAL_JSONL`, and `EXTRA_LABELS` (pool ground truth) for holdout and calibration gates.

## Phase 3: Testnet weight-setting validation

```bash
export VALIDATOR_WALLET_NAME=... VALIDATOR_WALLET_HOTKEY=...
export CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443 NETUID=<your_netuid>
export RUN_SECONDS=3600 MIN_SET_WEIGHTS_SUCCESSES=3
./scripts/check_testnet_set_weights.sh
```

Pass criteria:
- At least `MIN_SET_WEIGHTS_SUCCESSES` successful `set_weights on chain successfully!`.
- Zero `set_weights failed`.

## Phase 4: Soak watchdog (hours/days)

```bash
./scripts/soak_watchdog.sh artifacts/stress/validator_<timestamp>.log
```

Pass criteria:
- Step log is fresh (no stall beyond threshold).
- Error budget not exceeded.

## CI expectations

- `subnet-readiness` workflow must pass unit tests.
- R2 integration test is required in protected branches where R2 secrets are available.
