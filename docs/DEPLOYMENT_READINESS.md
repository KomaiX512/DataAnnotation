# Subnet Deployment Readiness Gates

This project uses a strict one-path flow (real training + R2 + golden evaluation).  
Use this checklist to promote from localnet to testnet.

## Phase 1: Core loop sanity (localnet)

```bash
export MINER_MAX_TRAIN_SAMPLES=16 MINER_MAX_VAL_SAMPLES=8 MINER_MAX_EPOCHS=1 GOLDEN_MAX_SAMPLES=8
export ENABLE_AUTORESEARCH=0 MAX_TRAINING_SECONDS=300 TRAINING_TIMEOUT=900 RUN_SECONDS=1800
./scripts/run_localnet_e2e_real.sh
```

Pass criteria:
- `event=evaluator_golden_score_payload` appears.
- `event=artifact_verification ... task_type=training ... passed=True` appears.
- No `Error during validation step`.

## Phase 2: Multi-miner + adversarial stress (localnet)

```bash
export RUN_SECONDS=1800 ADVERSARIAL_MINER_INDEX=2 ADVERSARIAL_MODE=malformed_manifest
./scripts/run_localnet_stress_matrix.sh
```

Pass criteria:
- Validator remains alive (`step(...)` continues).
- Adversarial responses are rejected by integrity checks.
- Honest miners still receive non-zero training verification.

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
