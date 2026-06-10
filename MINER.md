# Miner Setup Guide

This guide walks through the full miner flow from installation to wallet
registration, configuration, and running. Every command is copy-paste ready.

## What a miner does

Miners download unlabeled images, run a vision model to produce bounding boxes, and it could finetune that with golden samples and class labels, then upload `annotations.json` to R2. Validators score miners
against a hidden Golden Set and reward the best submissions. This subnet is
model-agnostic: any vision model is supported as long as it can produce the
required labels.

---

## Step 0: Install prerequisites

You need Python 3.10+ and Git. Then clone the repo and install dependencies:

```bash
git clone https://github.com/KomaiX512/DataAnnotation.git bittensor-subnet-template-1
cd bittensor-subnet-template-1
python -m venv .venv-neurons
source .venv-neurons/bin/activate
pip install -r requirements.txt
```

This installs `bittensor` 9.7.0 (and all dependencies) from
[requirements.txt](requirements.txt).

---

## Step 1: Create a wallet (coldkey + hotkey)

### For localnet

```bash
source .venv-btcli/bin/activate

btcli wallet create \
  --wallet-name miner \
  --wallet-path ~/.bittensor/wallets \
  --hotkey minerhk \
  --n-words 12 \
  --no-use-password \
  --overwrite
```

### For testnet / mainnet

```bash
source .venv-btcli/bin/activate

# Create wallet keys (interactive prompts will ask for password/words)
btcli wallet create \
  --wallet-name <WALLET_NAME> \
  --hotkey <WALLET_HOTKEY> \
  --n-words 12
```

> [!IMPORTANT]
> **btcli flag syntax**: btcli uses `--wallet-name` (dashes) and `--hotkey` (no prefix), **not** `--wallet.name` or `--wallet.hotkey` (dots). The dot-notation is only used by neuron scripts (`miner.py`, `validator.py`).

---

## Step 2: Fund and register the hotkey on the subnet

### Transferring TAO (For Testnet Funding)
If you need to transfer testnet TAO from an existing wallet to fund your registration:

```bash
source .venv-btcli/bin/activate

btcli wallet transfer \
  --wallet-name <SOURCE_WALLET_NAME> \
  --dest <DESTINATION_COLDKEY_ADDRESS> \
  --amount <AMOUNT_IN_TAO> \
  --network test \
  -y
```

#### Option A: Localnet Registration (Simulation)

```bash
source .venv-neurons/bin/activate
python scripts/localnet_setup.py --chain-endpoint ws://127.0.0.1:9944
```

#### Option B: Registration via Python Script (Recommended for Testnet/Mainnet)
In case `btcli` encounters network/compatibility errors:

```bash
source .venv-neurons/bin/activate

python scripts/register_on_testnet.py \
  --wallet.name <WALLET_NAME> \
  --wallet.hotkey <WALLET_HOTKEY> \
  --subtensor.network test \
  --netuid 498
```

#### Option C: Registration via `btcli`

```bash
source .venv-btcli/bin/activate

btcli subnets register \
  --netuid 498 \
  --wallet-name <WALLET_NAME> \
  --hotkey <WALLET_HOTKEY> \
  --network test \
  -y
```

#### For mainnet

```bash
source .venv-btcli/bin/activate

btcli subnets register \
  --netuid <NETUID> \
  --wallet-name <WALLET_NAME> \
  --hotkey <WALLET_HOTKEY> \
  --network finney \
  -y
```

---

## Step 3: Configure `.env`

Copy the example file and edit it:

```bash
cp .env.example .env
```

### Subnet Information
* **Subnet Name**: `DataAnnotation`
* **Assigned Netuid**: `498`
* **Repository URL**: `https://github.com/KomaiX512/DataAnnotation.git`

### Shared Validator Bucket Credentials (Localnet Testing)
For localnet simulation, validators and miners can share the following R2 credentials to ensure every miner has access to the dataset:

```bash
R2_BUCKET_NAME=subnet
R2_ACCOUNT_ID=51abf57b5c6f9b6cf2f91cc87e0b9ffe
R2_S3_ENDPOINT=https://51abf57b5c6f9b6cf2f91cc87e0b9ffe.r2.cloudflarestorage.com
R2_ENDPOINT_URL=https://51abf57b5c6f9b6cf2f91cc87e0b9ffe.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=6db9f1b555e51d83a73b3d6f0c3a5c26
R2_SECRET_ACCESS_KEY=1270b967bbd3cc88c65f6d3216e8cf730ea7954b37cb23f867abd57a7ac2f4ba
R2_PUBLIC_BUCKET_URL=https://pub-3aa7ed152eb9407cb756c8349a5ef02f.r2.dev
```

**Required fields for a miner:**

| Variable | Description |
|---|---|
| `R2_ACCESS_KEY_ID` | Cloudflare R2 access key |
| `R2_SECRET_ACCESS_KEY` | Cloudflare R2 secret key |
| `R2_ENDPOINT_URL` | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_BUCKET_NAME` | Your R2 bucket name |
| `MINER_MODEL_BACKEND` | One of: `yolo_local`, `self_hosted`, `openai_vision` |
| `MINER_ANNOTATION_WORKSPACE` | Where miner writes temporary files (default: `./artifacts/miner_annotation`) |

> [!TIP]
> For `MINER_ANNOTATION_WORKSPACE`, use a local SSD path with plenty of space.
> The default `./artifacts/miner_annotation` is fine for most setups.

---

## Step 4: Choose a backend and configure it

Set `MINER_MODEL_BACKEND` in `.env` to one of the options below, then fill
the matching variables. You can use **any** vision model; examples below
mention YOLO only as a reference implementation.

### Backend A: `self_hosted`

Use your own REST API that implements `/train` and `/infer`. This is the most
flexible path for any custom vision model.

In `.env`:
```
MINER_MODEL_BACKEND=self_hosted
SELF_HOSTED_TRAIN_URL=http://localhost:8081/train
SELF_HOSTED_INFER_URL=http://localhost:8081/infer
```

Optional: start the reference server (YOLOv8 training and inference):
```bash
source .venv-neurons/bin/activate
env PYTHONPATH=. python server.py --host 127.0.0.1 --port 8081 --checkpoint yolov8n.pt
```

### Backend B: `yolo_local`

Fine-tune a local YOLO model on your GPU.

In `.env`:
```
MINER_MODEL_BACKEND=yolo_local
YOLO_MODEL_PATH=yolov8n.pt
YOLO_EPOCHS=10
YOLO_IMGSZ=640
YOLO_BATCH=16
```

### Backend C: `openai_vision`

Use OpenAI fine-tuning for a hosted vision model. **This can incur API costs.**

In `.env`:
```
MINER_MODEL_BACKEND=openai_vision
OPENAI_API_KEY=sk-...
OPENAI_BASE_MODEL=gpt-4o-2024-08-06
```

---

## Step 5: Run the miner

### Localnet

```bash
source .venv-neurons/bin/activate
source .env

env PYTHONPATH=. MINER_ADVERSARIAL=0 \
  python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey minerhk \
  --subtensor.network local \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --netuid 2 \
  --miner.model_backend yolo_local \
  --miner.yolo_pretrained_weights yolov8n.pt \
  --miner.skip_training \
  --axon.port 8091 \
  --logging.debug \
  --miner.dual_flywheel_r2_prefix localnet/miners/miner1
```

### Testnet

```bash
source .venv-neurons/bin/activate
source .env

env PYTHONPATH=. python neurons/miner.py \
  --wallet.name <WALLET_NAME> \
  --wallet.hotkey <WALLET_HOTKEY> \
  --subtensor.network test \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
  --netuid <NETUID> \
  --miner.model_backend yolo_local \
  --miner.yolo_pretrained_weights yolov8n.pt \
  --axon.port 8091 \
  --logging.debug
```

### Mainnet

```bash
source .venv-neurons/bin/activate
source .env

env PYTHONPATH=. python neurons/miner.py \
  --wallet.name <WALLET_NAME> \
  --wallet.hotkey <WALLET_HOTKEY> \
  --subtensor.network finney \
  --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
  --netuid <NETUID> \
  --miner.model_backend yolo_local \
  --miner.yolo_pretrained_weights yolov8n.pt \
  --axon.port 8091 \
  --logging.debug
```

> [!NOTE]
> **Neuron scripts use dot-notation**: `--wallet.name`, `--wallet.hotkey`,
> `--subtensor.network`, `--subtensor.chain_endpoint`. This is different from
> btcli which uses dash-notation (`--wallet-name`, `--network`).

---

## Step 6: Verify your miner

- **Logs**: Look for `Miner running...` and `Serving miner axon` messages.
- **R2 bucket**: Check for new `annotations.json` files under your R2 prefix.
- **Process**: `ps aux | grep miner.py | grep -v grep`

If nothing uploads, check:
1. Wallet registration (is the miner registered on the correct netuid?)
2. R2 credentials (are `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` correct?)
3. Backend settings (is the model backend configured?)
4. Validator status (is a validator running and dispatching tasks?)
