# Validator Setup Guide

This guide covers validator setup from installation to wallet registration,
corpus configuration, and running. Every command is copy-paste ready.

## What a validator does

Validators create annotation tasks for miners, score miner responses against a
hidden Golden Set, and publish on-chain weights. Accepted annotations are
aggregated and exported into a commercial dataset.

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
  --wallet-name validator \
  --wallet-path ~/.bittensor/wallets \
  --hotkey valhk \
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

### Subnet Registration

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

**Required fields for a validator:**

| Variable | Description |
|---|---|
| `R2_ACCESS_KEY_ID` | Cloudflare R2 access key |
| `R2_SECRET_ACCESS_KEY` | Cloudflare R2 secret key |
| `R2_ENDPOINT_URL` | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_BUCKET_NAME` | Your R2 bucket name |
| `VALIDATOR_IMAGE_CACHE_ROOT` | Local path for cached images (must be absolute for `file://` prefix) |
| `VALIDATOR_COMMERCIAL_DATASET_PREFIX` | Where commercial exports are written |

---

## Step 4: Choose your corpus source

You have two options for the Golden Set and annotation pool.

### Option A: Hugging Face datasets (default)

If you do **not** set `VALIDATOR_COCO_MANIFEST`, the validator uses Hugging Face
datasets. In `.env`:

```bash
VALIDATOR_GOLDEN_DATASET=keremberke/construction-safety-object-detection
VALIDATOR_GOLDEN_SPLIT=train
VALIDATOR_GOLDEN_RATIO=0.1
VALIDATOR_ANNOTATION_SPLIT=train
VALIDATOR_ANNOTATION_MAX_PER_DATASET=512
```

You can adjust dataset IDs and splits as needed. For implementation details, see
`template/hazard/image_corpus.py`.

### Option B: Local COCO manifest (recommended for localnet)

Set a local COCO manifest path in `.env`:

```bash
VALIDATOR_COCO_MANIFEST=./artifacts/localnet/coco200/manifest.json
```

This uses a local dataset split and requires the manifest to exist at that path.

---

## Step 5: Run the validator

### Localnet

Open a **dedicated terminal** and get the miner SS58 address first:

```bash
source .venv-neurons/bin/activate
python -c "import bittensor as bt; print(bt.wallet(name='miner', hotkey='minerhk').hotkey.ss58_address)"
```

Replace `<MINER_SS58_ADDRESS>` below with the address from above:

```bash
source .venv-neurons/bin/activate
source .env

PROJECT_ROOT="$(pwd)"

env PYTHONPATH=. \
  FORCE_LOCAL_SET_WEIGHTS=1 \
  DEFAULT_ACCEPT_CONFIDENCE=0.01 \
  DEFAULT_ACCEPT_SEVERITY_CONFIDENCE=0.01 \
  DEFAULT_MIN_VOTERS=1 \
  DEFAULT_MIN_MEAN_IOU_TO_MEDIAN=0.1 \
  LOCALNET_MINER_PORT_BY_SS58="<MINER_SS58_ADDRESS>=8091" \
  FALLBACK_SINGLE_MINER_ENABLED=1 \
  FALLBACK_SINGLE_MINER_MIN_RELIABILITY=0.3 \
  python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey valhk \
  --subtensor.network local \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --netuid 2 \
  --axon.port 8090 \
  --neuron.sample_size 3 \
  --neuron.forward_step_sleep_seconds 15 \
  --neuron.annotation_timeout 300 \
  --neuron.flywheel_commercial_export_every 1 \
  --neuron.flywheel_commercial_dataset_prefix "file://${PROJECT_ROOT}/artifacts/localnet/self_hosted_commercial" \
  --neuron.flywheel_image_cache_root "${PROJECT_ROOT}/artifacts/localnet/self_hosted_image_cache" \
  --logging.debug
```

> [!WARNING]
> **Absolute paths required!** The `--neuron.flywheel_image_cache_root` and
> `--neuron.flywheel_commercial_dataset_prefix` must use **absolute paths**.
> Relative paths cause `ValueError: relative path can't be expressed as a
> file URI`. Use `PROJECT_ROOT="$(pwd)"` to construct them.

**Key localnet environment variables:**

| Variable | Purpose |
|---|---|
| `FORCE_LOCAL_SET_WEIGHTS=1` | Bypasses commit-reveal weight setting |
| `DEFAULT_ACCEPT_CONFIDENCE=0.01` | Relaxed annotation acceptance threshold |
| `DEFAULT_MIN_VOTERS=1` | Accept consensus with just 1 miner |
| `LOCALNET_MINER_PORT_BY_SS58` | Maps miner SS58 addresses to axon ports |
| `FALLBACK_SINGLE_MINER_ENABLED=1` | Allows single-miner annotation acceptance |

### Testnet

```bash
source .venv-neurons/bin/activate
source .env

env PYTHONPATH=. python neurons/validator.py \
  --wallet.name <WALLET_NAME> \
  --wallet.hotkey <WALLET_HOTKEY> \
  --subtensor.network test \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
  --netuid <NETUID> \
  --axon.port 8090 \
  --logging.debug
```

### Mainnet

```bash
source .venv-neurons/bin/activate
source .env

env PYTHONPATH=. python neurons/validator.py \
  --wallet.name <WALLET_NAME> \
  --wallet.hotkey <WALLET_HOTKEY> \
  --subtensor.network finney \
  --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
  --netuid <NETUID> \
  --axon.port 8090 \
  --logging.debug
```

> [!NOTE]
> On testnet and mainnet, **do not set** `FORCE_LOCAL_SET_WEIGHTS`,
> `LOCALNET_MINER_PORT_BY_SS58`, or the relaxed threshold variables.
> These are localnet-only overrides.

---

## Step 6: Verify your validator

**Log events to watch for:**

| Event | Meaning |
|---|---|
| `event=validator_init mode=annotation_only` | Validator initialized |
| `event=image_corpus_load_start` | Loading dataset images |
| `step(N) block(M)` | Main loop is running |
| `event=evaluator_golden_score_payload` | Scoring miner annotations |
| `event=annotation_flywheel_round_done` | Annotation round completed |
| `set_weights on chain successfully!` | Weights published on-chain |

**Check commercial dataset export:**
```bash
ls -la artifacts/localnet/self_hosted_commercial/
cat artifacts/localnet/self_hosted_commercial/commercial-dataset-step-0.jsonl | head -5
```

Key outputs in the exported JSONL:
- `image_url` is replaced with a permanent R2 link
- `objects` array displays aggregated consensus bounding boxes
- No Golden set images are leaked (filtered out)
