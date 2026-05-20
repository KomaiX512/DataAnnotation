# Production readiness (localnet ≥10 miners)

This checklist maps the **seven production acceptance tests** to concrete commands. Some require **controlled miner behavior** (Sybil, collusion, minority expert-only); those are covered by the **offline synthetic suite** (`tests/stress/test_probabilistic_aggregation_acceptance.py`) because live YOLO miners cannot be scripted to emit arbitrary labels. **Localnet** still proves wiring, scale (≥10 miners), artifact export, and schema.

## One-shot orchestrator

```bash
chmod +x scripts/run_production_readiness_localnet.sh
export RUN_SECONDS=1800          # longer for more commercial rows
export MIN_MINERS=10             # default; increase to 60 for Sybil scale-out
# Optional: after dumping labels (see below)
export GOLDEN_MANIFEST=/path/to/golden_manifest.json
export COMMERCIAL_JSONL=/path/to/commercial-dataset.jsonl
export EXTRA_LABELS=/path/to/pool_labels.json   # optional
./scripts/run_production_readiness_localnet.sh
```

The script:

1. Ensures **≥ `MIN_MINERS`** entries in `MINER_WALLET_NAMES` (defaults to `miner` + `miner2`…`miner10` with ports 8091…8100).
2. Runs `scripts/run_localnet_stress_matrix.sh`.
3. Runs `python scripts/production_readiness_eval.py simulate` (offline Sybil / collusion / minority / low-miner / calibration band / metadata / golden lane).
4. Runs **schema** validation on `artifacts/commercial_dataset/commercial-dataset*.jsonl` when present.
5. If `GOLDEN_MANIFEST` and `COMMERCIAL_JSONL` are set, runs **golden-holdout** and **calibration** against exported lines.

## 1) Golden holdout (1000 images, never used for reliability)

**Validator note:** Reliability updates today use all Golden-injected rounds; a strict “holdout never touches reliability” split requires a follow-up code change (exclude `holdout_ids` in `evaluate_round_annotations`).  

**Offline evaluation today:** `production_readiness_eval.py golden-holdout` takes a deterministic shuffle of `image_id`s present in **both** the manifest and the commercial file, selects up to `--holdout-count` (default 1000), and scores **accepted** objects vs ground truth (IoU match ≥ 0.5, then exact class).

- **Golden manifest** (`golden_manifest.v1`): dump from a validator-side corpus, e.g.:

  `PYTHONPATH=. python -c "from pathlib import Path; from template.hazard.image_corpus import ImageCorpus, ImageCorpusConfig, dump_golden_manifest; c=ImageCorpus(ImageCorpusConfig(Path('artifacts/image_cache'))); c.ensure_loaded(); dump_golden_manifest(c, Path('artifacts/eval/golden_manifest.json'))"`

- **Pool labels:** Commercial rows use **annotation pool** `image_id`s. Merge audited labels with:

  `{"schema_version":"pool_labels.v1","labels_by_image_id":{"<image_id>":[{"hazard_class":"missing_hardhat","bounding_box":[x1,y1,x2,y2]}]}}`

  Pass as `EXTRA_LABELS` to the eval script (see `run_production_readiness_localnet.sh`).

**Pass:** `--min-overall-acc 0.90` and `--min-class-f1 0.85` (defaults).

## 2) Sybil attack (+50 random miners, ≤2% accuracy drop)

Requires **two** full localnet campaigns (baseline vs +50) and the same **ground-truth** source as (1). The repo does not auto-spawn 50 adversarial labelers on chain; use **offline** `test_sybil_many_low_weight_miners_do_not_flip_consensus` as the certified gate, and extend localnet miner count when you add scripted adversarial axons.

## 3) Minority hazard preservation

Offline: `test_minority_low_prior_class_expert_and_peer` in the stress file.  
Live: needs curated pool labels + one high-reliability expert uid (see reliability model).

## 4) Collusion resistance

Offline: `test_collusion_low_reliability_wrong_majority_escalates_or_wrong_not_accepted`.

## 5) Low miner count

Offline: `test_only_one_miner_on_image_escalates`, `test_two_miners_spatial_disagreement_escalates`, and high-agreement two-miner path in stress suite.

## 6) Uncertainty calibration (≥100 samples, ±5%)

`production_readiness_eval.py calibration` compares **mean reported object confidence** to **empirical exact-class accuracy** on IoU-matched objects (needs `GOLDEN_MANIFEST` + `COMMERCIAL_JSONL` + optional `EXTRA_LABELS`).

## 7) Metadata completeness

`production_readiness_eval.py schema --glob "$COMM_DIR/commercial-dataset-step-*.jsonl"` validates required keys, nested `objects` / `miner_votes`, and recomputes **`audit_hash`**.  
`run_production_readiness_localnet.sh` uses that glob by default (append-only `commercial-dataset.jsonl` may mix legacy rows). Set `ALLOW_LEGACY_COMMERCIAL_JSONL=1` for warn-only during migration.

## Related

- `docs/PROBABILISTIC_AGGREGATION_REPRODUCIBILITY.md` — formulas and thresholds.
- `scripts/run_probabilistic_aggregation_stress.sh` — fast synthetic only (no chain).
