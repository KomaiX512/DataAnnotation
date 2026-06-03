# Miner Setup Guide

This guide walks through the full miner flow from Bittensor install to wallet registration, configuration, and running. It is written as step-by-step CLI instructions so you can follow it end-to-end.

## What a miner does

Miners download unlabeled images, run a vision model to produce bounding boxes and class labels, then upload `annotations.json` to R2. Validators score miners against a hidden Golden Set and reward the best submissions. This subnet is model-agnostic: any vision model is supported as long as it can produce the required labels.

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

You must register the hotkey on the target subnet before mining:

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

These fields are required for a miner:

- `WALLET_NAME` and `WALLET_HOTKEY` (from Step 1)
- `NETUID` and `SUBTENSOR_NETWORK` (from your chain)
- `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT_URL`, `R2_BUCKET_NAME`
- `MINER_MODEL_BACKEND` (choose one backend below)
- `MINER_ANNOTATION_WORKSPACE` (where miner writes temporary files)

Path guidance for `MINER_ANNOTATION_WORKSPACE`:

- Use a local SSD path with plenty of space.
- Avoid network mounts or slow disks.
- Default is `./artifacts/miner_annotation` and is safe for most setups.

## Step 4: Choose a backend and configure it

Set `MINER_MODEL_BACKEND` to one of the options below, then fill the matching variables in `.env`. You can use any vision model; examples below mention YOLO only as a reference implementation.

### Backend A: `self_hosted`

Use your own REST API that implements `/train` and `/infer`. This is the most flexible path for any custom vision model.

In `.env`:

```
MINER_MODEL_BACKEND=self_hosted
SELF_HOSTED_TRAIN_URL=http://localhost:8081/train
SELF_HOSTED_INFER_URL=http://localhost:8081/infer
```

Optional: start the reference server (YOLOv8 training and inference as an example):

```bash
source .env
PYTHONPATH=. python server.py --host 127.0.0.1 --port 8081 --checkpoint yolov8n.pt
```

### Backend B: `yolo_local`

Fine-tune a local vision model on your GPU (YOLO is the built-in reference backend).

In `.env`:

```
MINER_MODEL_BACKEND=yolo_local
YOLO_MODEL_PATH=yolov8n.pt
YOLO_EPOCHS=10
YOLO_IMGSZ=640
YOLO_BATCH=16
```

### Backend C: `openai_vision`

Use OpenAI fine-tuning for a hosted vision model. This can incur API costs.

In `.env`:

```
MINER_MODEL_BACKEND=openai_vision
OPENAI_API_KEY=sk-...
OPENAI_BASE_MODEL=gpt-4o-2024-08-06
```

## Step 5: Run the miner

```bash
source .env
PYTHONPATH=. python neurons/miner.py
```

The miner will receive annotation tasks, train (if needed), run inference, and upload `annotations.json` files to your R2 bucket.

## Step 6: Verify your miner

- Check your R2 bucket for new `annotations.json` files under `MINER_R2_PREFIX`.
- Watch logs for successful uploads and inference runs.
- If nothing uploads, re-check wallet registration, R2 credentials, and backend settings.
