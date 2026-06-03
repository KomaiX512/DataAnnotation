# Validator Setup Guide

What you need:
* A wallet registered on the subnet
* A Cloudflare R2 bucket for receiving miner submissions
* A pre-prepared corpus: Golden Set images + unlabeled annotation pool images

## Step 1: Install & configure

```bash
git clone https://github.com/KomaiX512/JHA_subnet.git bittensor-subnet-template-1 && cd bittensor-subnet-template-1
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with validator-side variables: `WALLET_NAME`, `WALLET_HOTKEY`, `NETUID`, `SUBTENSOR_NETWORK`, R2 credentials, and corpus paths.

## Step 2: Prepare your corpus

* Create a folder of Golden Set images with JSON annotations (class + bounding box).
* Create a folder of unlabeled images for the annotation pool.

The validator expects these paths in `.env`:
* `VALIDATOR_GOLDEN_IMAGES_DIR`
* `VALIDATOR_GOLDEN_ANNOTATIONS_DIR`
* `VALIDATOR_ANNOTATION_IMAGES_DIR`

*(Alternatively, specify Hugging Face dataset IDs – see `template/hazard/image_corpus.py` for details.)*

## Step 3: Run the validator

```bash
source .env
PYTHONPATH=. python neurons/validator.py
```

It will start the main loop, send tasks to miners, score them on the Golden Set, and periodically export the commercial dataset to `EXPORT_DIR`.

## Verification

* Check the logs: you should see `event=evaluator_golden_score_payload` and `event=annotation_flywheel_round_done`.
* After a few rounds, open the `commercial-dataset.jsonl` file in your export directory – it must contain only accepted annotations (no Golden images).
* If you see `set_weights on chain successfully!`, your validator is fully operational.
