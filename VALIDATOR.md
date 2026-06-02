# Validator Setup Guide

This guide describes how to run and configure a validator on the Decentralized Data Annotation Subnet. Validators manage the image corpus, inject secret Golden Set images into miner query tasks, score miner responses, run Bayesian consensus fusion, set network weights, and export the resulting dataset.

---

## 1. Prerequisites

Before running a validator, ensure you have:
1. **Bittensor Wallets**: A coldkey and a hotkey registered on the subnet.
2. **TAO**: Sufficient TAO in your wallet to register and set weights.
3. **Cloudflare R2 Bucket**: An active R2 bucket (or S3-compatible bucket) to download and inspect miner annotations.
4. **Hardware**: An NVIDIA GPU with at least 8GB VRAM (recommended for image camouflaging and local validation runs).

---

## 2. Dataset Setup

Validators can load datasets either dynamically from Hugging Face or locally via a COCO-style manifest.

### COCO-200 Local Dataset Splitter
The codebase includes a script that automatically downloads a subset of COCO val2017 and prepares a 200-image dataset split.

Run the following command to download and split the dataset:
```bash
python scripts/localnet/prepare_coco_val2017_subset.py \
  --out-dir artifacts/localnet/coco200 \
  --dataset-size 200 \
  --golden-ratio 0.1 \
  --seed 7
```

This creates the following directory layout under `artifacts/localnet/coco200/`:
* `manifest.json`: Contains image details, annotations, and metadata defining which images are golden.
* `images/`: The 200 extracted JPEG images, renamed to their SHA-256 hashes to prevent miners from guessing their source.

To instruct the validator to use this local dataset, set `VALIDATOR_COCO_MANIFEST` in your `.env` to the absolute path of the generated `manifest.json`.

---

## 3. Golden Set & Training Pool

The prepared 200-image dataset is partitioned into three groups by the validator:
1. **Golden Set (20 images / 10% default)**: Labeled images containing validated ground-truth boxes. The labels are kept strictly validator-side and are **never** shared with miners. These are used to calculate the **Fidelity Score**.
2. **Training Pool (20-30 images)**: Labeled images that are publicly shared with miners (along with their ground-truth annotations) to allow miners to train/fine-tune their detection models.
3. **Annotation Pool (160 images)**: Unlabeled images that miners must annotate. Miner annotations on these images are aggregated via Bayesian consensus fusion.

---

## 4. Configuration (.env)

Configure your validator by setting the appropriate environment variables in `.env`:

```bash
# ===== SUBNET & WALLET =====
NETUID=1
SUBTENSOR_NETWORK=localnet
SUBTENSOR_CHAIN_ENDPOINT=ws://127.0.0.1:9944
WALLET_NAME=default
WALLET_HOTKEY=default

# ===== CLOUDFLARE R2 =====
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
R2_BUCKET_NAME=annotation-subnet

# ===== VALIDATOR DATASET CONFIG =====
# Set this to use a local COCO dataset (Recommended for local tests)
VALIDATOR_COCO_MANIFEST=/absolute/path/to/artifacts/localnet/coco200/manifest.json

# Alternative Hugging Face Dataset (If VALIDATOR_COCO_MANIFEST is empty)
VALIDATOR_GOLDEN_DATASET=keremberke/construction-safety-object-detection
VALIDATOR_GOLDEN_SPLIT=train
VALIDATOR_GOLDEN_RATIO=0.1
VALIDATOR_GOLDEN_SPLIT_SEED=20260509
VALIDATOR_ANNOTATION_DATASET=
VALIDATOR_ANNOTATION_SPLIT=train
VALIDATOR_ANNOTATION_MAX_PER_DATASET=512

# ===== VALIDATOR PARAMETERS =====
VALIDATOR_IMAGE_CACHE_ROOT=./data/flywheel/image_cache
VALIDATOR_IMAGE_SERVING_BASE_URL=
VALIDATOR_REQUEST_SIZE=0                     # 0 = send the full corpus in one task
VALIDATOR_GOLDEN_INJECTION_PER_REQUEST=0     # Used if VALIDATOR_REQUEST_SIZE > 0
VALIDATOR_COMMERCIAL_DATASET_PREFIX=file:///absolute/path/to/exports
VALIDATOR_COMMERCIAL_EXPORT_EVERY=10         # Export every N validation cycles
VALIDATOR_SAMPLE_SIZE=50
VALIDATOR_TIMEOUT=10
VALIDATOR_ANNOTATION_TIMEOUT=0               # 0 = fallback to VALIDATOR_TIMEOUT
VALIDATOR_NUM_CONCURRENT_FORWARDS=1
VALIDATOR_FORWARD_STEP_SLEEP_SECONDS=0
VALIDATOR_DISABLE_SET_WEIGHTS=0
VALIDATOR_AXON_OFF=0
FORCE_LOCAL_SET_WEIGHTS=1                    # Set to 1 on local subtensor nodes
```

---

## 5. Running the Validator

Run the validator using the following commands:
```bash
source .env
python neurons/validator.py
```

---

## 6. Output & Dataset Export

Every `VALIDATOR_COMMERCIAL_EXPORT_EVERY` rounds, the validator merges consensus annotations and exports a commercial dataset in **JSONL** format.

### Export Location
* If `VALIDATOR_COMMERCIAL_DATASET_PREFIX` starts with `file://`, the export will be written to the local filesystem (default: `./artifacts/commercial_dataset/commercial-dataset.jsonl`).
* If `VALIDATOR_COMMERCIAL_DATASET_PREFIX` starts with `r2://` or `s3://`, the file will be uploaded to your Cloudflare R2 bucket.

### Export Record Format
Each exported line is a JSON object matching the schema below:
```json
{
  "image_id": "sha256_hash_of_image",
  "annotations": [
    {
      "hazard_class": "excavator",
      "bounding_box": [100, 150, 450, 600],
      "confidence": 0.88,
      "reliability": 0.95
    }
  ],
  "miner_votes": 3,
  "reliability_score": 0.92,
  "audit_hash": "audit_sha256_hash"
}
```
All secret **Golden Set** images are strictly excluded from this public export to maintain dataset integrity and avoid label leaks.
