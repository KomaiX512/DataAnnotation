# Validator Guide

## Install

```bash
git clone <repo-url>
cd bittensor-subnet-template-1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with wallet, subnet, R2, corpus, and reward/export settings.

## Prepare Data

Validators need a labeled Golden Set and an annotation pool. You can use Hugging Face datasets or a local COCO-style manifest.

For Hugging Face, set:

```bash
VALIDATOR_GOLDEN_DATASET=keremberke/construction-safety-object-detection
VALIDATOR_GOLDEN_SPLIT=train
VALIDATOR_GOLDEN_RATIO=0.1
VALIDATOR_ANNOTATION_DATASET=
VALIDATOR_ANNOTATION_SPLIT=train
```

For a local manifest, set:

```bash
VALIDATOR_COCO_MANIFEST=/absolute/path/to/manifest.json
```

Manifest rows must include image identity, local path or URL, dimensions, labels for Golden/training rows, and whether the image is Golden. Golden labels stay validator-only and are never sent to miners.

## Configure Rounds

Set the corpus, cache, injection, and export controls in `.env`:

```bash
VALIDATOR_IMAGE_CACHE_ROOT=./data/flywheel/image_cache
VALIDATOR_REQUEST_SIZE=0
VALIDATOR_GOLDEN_INJECTION_PER_REQUEST=0
VALIDATOR_COMMERCIAL_DATASET_PREFIX=
VALIDATOR_COMMERCIAL_EXPORT_EVERY=10
```

Use `VALIDATOR_REQUEST_SIZE=0` to send the full corpus for each round. For sampled rounds, set a positive request size and inject hidden Golden images with `VALIDATOR_GOLDEN_INJECTION_PER_REQUEST`. Leave `VALIDATOR_COMMERCIAL_DATASET_PREFIX` empty for the default local export path, or set an absolute URI such as `file:///home/me/exports` or `r2://bucket/prefix`.

## Run

```bash
source .env
python neurons/validator.py
```

The validator logs round planning, miner responses, Golden fidelity payloads, rewards, set-weights behavior, and commercial export URIs. On localnet, set `FORCE_LOCAL_SET_WEIGHTS=1` only if your local subtensor supports the commit path; otherwise the validator still computes rewards and skips incompatible local commits.

## Rewards And Export

Rewards combine Golden Set fidelity, consensus/reliability, hallucination penalties, and commercial adoption. Golden rows are scoring-only and excluded from the public commercial JSONL export. Exported JSONL records include accepted objects, confidence distributions, miner votes, reliability weights, and an audit hash.
