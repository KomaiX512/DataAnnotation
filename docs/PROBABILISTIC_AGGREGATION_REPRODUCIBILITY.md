# Probabilistic annotation aggregation: reproducibility (v1)

This document pins the mathematics, hyperparameters, and export schema for the dual-flywheel **Bayesian / Dawid–Skene–style** fusion in `template/hazard/dataset_assembler.py` and reliability updates in `template/hazard/annotation_eval.py`.

## Method overview

1. **Per-miner reliability** (Golden images only, exponential decay per round batch, factor **λ = 0.95**):  
   For each class \(c\): precision/recall/F1 from aggregated TP/FP/FN and severity exact-match rate.  
   **Weight:**  
   \[
   w_{m,c} = \max(\varepsilon,\ \text{F1}_c \cdot \text{sev\_acc}_c),\quad \varepsilon = 10^{-4}.
   \]

2. **Class priors** from the validator Golden corpus with symmetric Dirichlet smoothing **α = 1.1** per class (including `_background`).

3. **Box clustering:** greedy IoU **≥ 0.5**.

4. **Likelihood (simplified DS):** for miner \(m\) observing label \(o\) when truth is \(t\), with reliability \(r = w_{m,o}\) clamped to \([10^{-4},1]\):  
   \(p_{\text{match}} = 0.5 + 0.5 r\); if \(o=t\), \(P(o\mid t)=p_{\text{match}}\); else \(P(o\mid t)=(1-p_{\text{match}})/(K-1)\) for \(K\) classes in the active label set.  
   Log-posterior contributions are scaled by \(\max(10^{-4}, w_{m,o})\) per vote.

5. **Severity:** independent weighted vote over tiers `{none, low, medium, high, critical}`; weight \(w_m\); off-vote mass **0.05**.

6. **Localization:** weighted box fusion with weights \(w_m\); spatial score = mean IoU of contributing boxes to the coordinate-wise median box.

7. **Marginal contribution (reward proxy):** for accepted class \(t^\*\), impact\(_m = P(t^\* \mid \text{all}) - P(t^\* \mid \text{all} \setminus m)\) (non-negative clamp).

## Acceptance thresholds (commercial export)

| Gate | Value |
|------|-------|
| Class posterior \( \max_t P(t\mid\cdot) \) | **≥ 0.9** |
| Severity posterior \( \max_s P(s\mid\cdot) \) | **≥ 0.8** |
| Miners on image (non-golden) | **≥ 2** |
| Mean IoU to median box | **≥ 0.7** |

If **any** cluster on a pool image fails a gate, the **whole image** is `escalation_required=true` and is **not** written to the commercial JSONL.

## Golden vs pool lanes

- **Golden images:** scored with `AnnotationFidelityScorer` only; `aggregation_method = golden_fidelity_v1`; never exported commercially.
- **Annotation pool:** `aggregation_method = bayesian_dawid_skene_v1`.

## Miner annotation backend (production)

- **Only `yolo`** is allowed in `AnnotationEngine` (`template/miner/annotation.py`).  
- `deterministic` raises at engine init.  
- Validators reject failed/malformed submissions through `error_message`, nonce checks, and missing `annotations_uri` validation (`template/validator/dual_forward.py`).  
- CLI default: `--miner.annotation_backend yolo` (`template/utils/config.py`).

## Export metadata (mandatory fields)

Each commercial JSONL object includes at least:  
`image_id`, `aggregation_method`, `acceptance_thresholds`, `reliability_window`, `escalation_required`, `escalation_reason`, `validator_version`, `audit_hash`, `miner_contribution_scores`, `objects[]`.  

Each `objects[]` entry includes:  
`object_cluster_id`, `accepted_hazard_class`, `accepted_severity`, `confidence`, `severity_confidence`, `class_posterior_distribution`, `severity_posterior_distribution`, `miner_votes`, `fused_bounding_box`, `spatial_mean_iou_to_median`, `escalation_reason`.

`audit_hash` = SHA-256 of the canonical JSON payload **before** the `audit_hash` key is inserted (see `WinningAnnotation.to_jsonable`).

`VALIDATOR_VERSION` env overrides default **1.2.0**.

## Stress harness

- **Fast synthetic:** `./scripts/run_probabilistic_aggregation_stress.sh`  
- **Full-scale localnet matrix (operator stub):** `./scripts/run_probabilistic_aggregation_stress_localnet_matrix.sh`  
- **Production readiness (≥10 miners + evaluators):** `docs/PRODUCTION_READINESS_LOCALNET.md`, `./scripts/run_production_readiness_localnet.sh`  
- Tests: `tests/stress/test_probabilistic_aggregation_acceptance.py` (marker `stress`).

## Reproduce a single fused image (developer)

1. Build `ImageCorpus` + `DatasetAssembler`.  
2. Populate `annotations_by_uid` and `per_miner_scores` (with `class_weights` from Golden).  
3. Call `assemble(...)`; inspect `WinningAnnotation.accepted_objects` and `to_jsonable()`.

---
*Version: bayesian_dawid_skene_v1 / golden_fidelity_v1 · Last aligned: Phase 2 miner + stress harness.*
