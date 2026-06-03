# Validator Setup Guide

This guide covers validator setup from Bittensor install to wallet registration, corpus configuration, and running. It is written as step-by-step CLI instructions.

## What a validator does

Validators create tasks for miners, score miner responses against a hidden Golden Set, and publish on-chain weights. Accepted annotations are exported into a commercial dataset.

## Step 0: Install prerequisites

You need Python 3.10+ and Git. Then clone the repo and install dependencies:

```bash
git clone https://github.com/KomaiX512/JHA_subnet.git bittensor-subnet-template-1
cd bittensor-subnet-template-1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs `bittensor` (and `btcli`) from [requirements.txt](requirements.txt).

## Step 1: Create a wallet (coldkey + hotkey)

Run the wallet creation flow and follow the prompts:

```bash
btcli wallet create
```

Typical prompt flow:

- Choose to create a new wallet.
- Enter a wallet name (this becomes `WALLET_NAME`).
- Create or select a hotkey (this becomes `WALLET_HOTKEY`).
- Save your seed words securely.

If you already have a wallet, use `btcli wallet list` to confirm the names.

## Step 2: Register the hotkey on the subnet

Register the hotkey for the target subnet:

```bash
btcli subnet register \
	--netuid 1 \
	--wallet.name <WALLET_NAME> \
	--wallet.hotkey <WALLET_HOTKEY> \
	--subtensor.network localnet
```

If your chain is not `localnet`, replace `--subtensor.network` with the correct network name, or pass a chain endpoint with `--subtensor.chain_endpoint ws://host:port`.

## Step 3: Configure `.env`

Copy the example file and edit it:

```bash
cp .env.example .env
```

These fields are required for a validator:

- `WALLET_NAME` and `WALLET_HOTKEY` (from Step 1)
- `NETUID` and `SUBTENSOR_NETWORK` (from your chain)
- `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT_URL`, `R2_BUCKET_NAME`
- Dataset configuration (see Step 4)
- `VALIDATOR_IMAGE_CACHE_ROOT` (where images are cached)
- `VALIDATOR_COMMERCIAL_DATASET_PREFIX` (where exports are written)

Path guidance:

- `VALIDATOR_IMAGE_CACHE_ROOT` should be a fast local disk path.
- `VALIDATOR_COMMERCIAL_DATASET_PREFIX` can be a local folder using `file:///...`.

## Step 4: Choose your corpus source

You have two options for the Golden Set and annotation pool.

### Option A: Local manifest (recommended for localnet)

Set a local COCO manifest path:

```
VALIDATOR_COCO_MANIFEST=./artifacts/localnet/coco200/manifest.json
```

This uses a local split and requires the manifest to exist at that path.

### Option B: Hugging Face datasets

If you do not set `VALIDATOR_COCO_MANIFEST`, the validator uses Hugging Face datasets:

```
VALIDATOR_GOLDEN_DATASET=keremberke/construction-safety-object-detection
VALIDATOR_GOLDEN_SPLIT=train
VALIDATOR_GOLDEN_RATIO=0.1
VALIDATOR_ANNOTATION_SPLIT=train
VALIDATOR_ANNOTATION_MAX_PER_DATASET=512
```

You can adjust dataset IDs and splits as needed. For implementation details, see `template/hazard/image_corpus.py`.

## Step 5: Run the validator

```bash
source .env
PYTHONPATH=. python neurons/validator.py
```

The validator will start the main loop, dispatch tasks to miners, score Golden Set accuracy, and export the commercial dataset on a schedule.

## Step 6: Verify your validator

- Check the logs for `event=evaluator_golden_score_payload` and `event=annotation_flywheel_round_done`.
- Check the export path for `commercial-dataset.jsonl` and confirm it excludes Golden images.
- When you see `set_weights on chain successfully!`, your validator is operating normally.
