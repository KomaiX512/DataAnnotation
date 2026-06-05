# Comprehensive Bittensor Subnet Setup Guide

## Complete Step-by-Step: Subtensor Node → Wallets → Subnet → Mining → Validation

> **Verified on:** btcli 9.7.1, bittensor SDK 9.7.0,
> `ghcr.io/opentensor/subtensor:latest`.
> Every command has been tested end-to-end.

This guide covers the full lifecycle of running the Data Annotation Subnet:
- Subtensor node (Docker)
- Wallet creation (owner, miner, validator)
- Token distribution via Alice devnet account
- Subnet creation and neuron registration
- Miner and validator launch with full configuration

---

## TABLE OF CONTENTS

1. [Prerequisites](#1-prerequisites)
2. [Environment Cleanup](#2-environment-cleanup)
3. [Subtensor Node](#3-subtensor-node)
4. [Wallet Creation](#4-wallet-creation)
5. [Chain Setup (Fund, Subnet, Register)](#5-chain-setup)
6. [Miner Launch](#6-miner-launch)
7. [Validator Launch](#7-validator-launch)
8. [Monitoring & Verification](#8-monitoring--verification)
9. [Testnet & Mainnet Deployment](#9-testnet--mainnet-deployment)
10. [R2 Bucket Management](#10-r2-bucket-management)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

```bash
# Check Python version (3.10+ required)
python3 --version

# Check Docker
docker --version

# Check available disk space (need ~5 GB)
df -h .
```

**Clone the repository** (skip if you already have it):
```bash
git clone https://github.com/KomaiX512/DataAnnotation.git bittensor-subnet-template-1
cd bittensor-subnet-template-1
```

**Install dependencies** (if not already done):
```bash
# Create and activate neuron virtual environment
python -m venv .venv-neurons
source .venv-neurons/bin/activate
pip install -r requirements.txt

# Create btcli virtual environment (separate to avoid conflicts)
python -m venv .venv-btcli
source .venv-btcli/bin/activate
pip install bittensor-cli
```

---

## 2. Environment Cleanup

**Always run this before starting a fresh deployment.**

```bash
# 1. Kill running processes
pkill -f "neurons/miner.py"    || true
pkill -f "neurons/validator.py" || true
pkill -f "server.py"           || true

# 2. Force-remove the old Docker container (stops and removes atomically)
docker rm -f subtensor-devnet-stable 2>/dev/null || true

# 3. Remove wallet data and neuron state
rm -rf ~/.bittensor/wallets/
rm -rf ~/.bittensor/miners/

# 4. Remove cached images and datasets
rm -rf ./artifacts/localnet/self_hosted_image_cache/
rm -rf ./artifacts/localnet/self_hosted_commercial/
rm -rf ./artifacts/miner_annotation/
```

> [!IMPORTANT]
> The `docker rm -f` step is **essential**. Without it, creating a new container
> with the same name will fail with:
> `Conflict. The container name … is already in use`.

---

## 3. Subtensor Node

### Start the devnet node

```bash
docker run -d \
  --name subtensor-devnet-stable \
  -p 9944:9944 \
  -p 30333:30333 \
  ghcr.io/opentensor/subtensor:latest \
  --dev \
  --rpc-external \
  --rpc-methods=unsafe \
  --rpc-cors=all \
  --alice \
  --force-authoring
```

### Wait for the node to become ready

```bash
for i in $(seq 1 30); do
  if curl -sS -m 2 -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' \
    http://127.0.0.1:9944 2>/dev/null | grep -q '"result"'; then
    echo "Subtensor RPC is ready (attempt $i)"
    break
  fi
  echo "Waiting for subtensor... (attempt $i)"
  sleep 2
done
```

### Verify the container is running

```bash
docker ps | grep subtensor-devnet-stable
docker logs --tail 5 subtensor-devnet-stable
```

---

## 4. Wallet Creation

**Activate the btcli environment:**
```bash
source .venv-btcli/bin/activate
```

### Create all wallets

```bash
# Alice wallet — maps to the pre-funded devnet account (1,000,000 τ)
btcli wallet create \
  --wallet-name alice \
  --wallet-path ~/.bittensor/wallets \
  --hotkey default \
  --no-use-password \
  --uri Alice

# Owner wallet — will create the subnet
btcli wallet create \
  --wallet-name owner \
  --wallet-path ~/.bittensor/wallets \
  --hotkey ownerhk \
  --n-words 12 \
  --no-use-password

# Validator wallet — will score miner performance
btcli wallet create \
  --wallet-name validator \
  --wallet-path ~/.bittensor/wallets \
  --hotkey valhk \
  --n-words 12 \
  --no-use-password

# Miner wallet — will process annotation tasks
btcli wallet create \
  --wallet-name miner \
  --wallet-path ~/.bittensor/wallets \
  --hotkey minerhk \
  --n-words 12 \
  --no-use-password
```

### Verify wallets
```bash
btcli wallet list --wallet-path ~/.bittensor/wallets
```

> [!NOTE]
> **btcli flag syntax**: btcli 9.x uses `--wallet-name` (dashes), **not**
> `--wallet.name` (dots). The dot-notation flags are only used by the
> bittensor SDK when launching neuron processes (miner.py, validator.py).

---

## 5. Chain Setup

### Use the helper script (recommended)

The helper script uses the bittensor SDK directly, which is more reliable
than btcli for chain operations:

```bash
source .venv-neurons/bin/activate

python scripts/localnet_setup.py --chain-endpoint ws://127.0.0.1:9944
```

This performs three steps in sequence:
1. **Fund** — transfers TAO from Alice to owner (10,000 τ), validator (1,000 τ), miner (1,000 τ)
2. **Subnet** — creates subnet netuid 2
3. **Register** — registers miner and validator on netuid 2

### Verify on-chain state

```bash
source .venv-neurons/bin/activate

python -c "
import bittensor as bt

st = bt.subtensor(network='ws://127.0.0.1:9944')
print('Subnets:', st.get_subnets())

mg = st.metagraph(2)
print(f'Subnet 2 neurons: {mg.n.item()}')
for uid in range(mg.n.item()):
    balance = st.get_balance(mg.coldkeys[uid])
    print(f'  UID {uid} | hotkey={mg.hotkeys[uid][:20]}... | balance={balance}')

miner = bt.wallet(name='miner', hotkey='minerhk')
print(f'\nMiner hotkey SS58: {miner.hotkey.ss58_address}')
print('(Save this address — you need it for the validator start command)')
"
```

---

## 6. Miner Launch

### Prepare workspace

```bash
mkdir -p ./artifacts/miner_annotation
mkdir -p ./artifacts/localnet/self_hosted_image_cache
mkdir -p ./artifacts/localnet/self_hosted_commercial
```

### Configure R2 credentials

Ensure your `.env` file has valid R2 credentials:
```bash
cp .env.example .env
# Edit .env and set:
#   R2_ACCESS_KEY_ID=...
#   R2_SECRET_ACCESS_KEY=...
#   R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
#   R2_BUCKET_NAME=...
#   R2_PUBLIC_BUCKET_URL=https://pub-<hash>.r2.dev
```

### Start the miner

Open a **dedicated terminal**:

```bash
cd /path/to/bittensor-subnet-template-1
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

**Expected logs:**
```
Miner using ModelTrainingAnnotationEngine with backend=yolo_local
Serving miner axon … on network: ws://127.0.0.1:9944 with netuid: 2
Miner running...
```

> [!NOTE]
> The `--miner.skip_training` flag tells the miner to use the pre-trained
> YOLOv8n weights for inference without fine-tuning. Remove this flag if you
> want the miner to train on incoming data.

---

## 7. Validator Launch

Open a **second dedicated terminal**.

### Get the miner SS58 address

```bash
source .venv-neurons/bin/activate
python -c "import bittensor as bt; print(bt.wallet(name='miner', hotkey='minerhk').hotkey.ss58_address)"
```

### Start the validator

Replace `<MINER_SS58_ADDRESS>` with the address from above:

```bash
cd /path/to/bittensor-subnet-template-1
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
> `--neuron.flywheel_commercial_dataset_prefix` must be **absolute paths**.
> Using relative paths causes `ValueError: relative path can't be expressed
> as a file URI`. The `PROJECT_ROOT="$(pwd)"` variable handles this.

**Key environment variables explained:**

| Variable | Purpose |
|---|---|
| `FORCE_LOCAL_SET_WEIGHTS=1` | Bypasses commit-reveal weight setting (localnet only) |
| `DEFAULT_ACCEPT_CONFIDENCE=0.01` | Relaxed annotation acceptance (localnet only) |
| `DEFAULT_MIN_VOTERS=1` | Accept consensus with just 1 miner (localnet) |
| `LOCALNET_MINER_PORT_BY_SS58` | Maps miner SS58 address to axon port |
| `FALLBACK_SINGLE_MINER_ENABLED=1` | Allows single-miner annotation acceptance |

---

## 8. Monitoring & Verification

### Success criteria

| # | Check | How to verify |
|---|---|---|
| 1 | Subtensor running | `docker ps \| grep subtensor` |
| 2 | Wallets created | `btcli wallet list --wallet-path ~/.bittensor/wallets` |
| 3 | Subnet netuid 2 exists | Helper script prints subnets |
| 4 | Miner + validator registered | Metagraph check shows 3 UIDs |
| 5 | Miner listening on port 8091 | Miner logs show "Miner running..." |
| 6 | Validator listening on port 8090 | Validator logs show "step(N) block(M)" |
| 7 | Miner receives tasks | Miner logs show annotation processing |
| 8 | Annotations uploaded to R2 | R2 bucket check shows objects |
| 9 | Commercial dataset exported | `ls artifacts/localnet/self_hosted_commercial/` |

### Quick status check

```bash
# Check processes
ps aux | grep -E "(miner|validator)" | grep neurons | grep -v grep

# Check Docker
docker ps | grep subtensor

# Check metagraph
source .venv-neurons/bin/activate
python -c "
import bittensor as bt
st = bt.subtensor(network='ws://127.0.0.1:9944')
mg = st.metagraph(2)
print(f'Neurons: {mg.n.item()}')
for uid in range(mg.n.item()):
    print(f'  UID {uid} | port={mg.axons[uid].port} | hotkey={mg.hotkeys[uid][:20]}...')
"
```

---

## 9. Testnet & Mainnet Deployment

The same miner and validator commands work on any network.
Only change the network-related flags:

### Testnet

```bash
# Miner on testnet
env PYTHONPATH=. python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey minerhk \
  --subtensor.network test \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
  --netuid <YOUR_TESTNET_NETUID> \
  --miner.model_backend yolo_local \
  --miner.yolo_pretrained_weights yolov8n.pt \
  --axon.port 8091 \
  --logging.debug

# Validator on testnet (no FORCE_LOCAL_SET_WEIGHTS, stricter thresholds)
env PYTHONPATH=. python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey valhk \
  --subtensor.network test \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
  --netuid <YOUR_TESTNET_NETUID> \
  --axon.port 8090 \
  --logging.debug
```

### Settings comparison

| Setting | Localnet | Testnet | Mainnet |
|---|---|---|---|
| `--subtensor.network` | `local` | `test` | `finney` |
| `--subtensor.chain_endpoint` | `ws://127.0.0.1:9944` | `wss://test.finney.opentensor.ai:443` | `wss://entrypoint-finney.opentensor.ai:443` |
| `FORCE_LOCAL_SET_WEIGHTS` | `1` | `0` | `0` |
| `LOCALNET_MINER_PORT_BY_SS58` | Required | Not used | Not used |
| `DEFAULT_ACCEPT_CONFIDENCE` | `0.01` | `0.3` | `0.5` |
| `DEFAULT_MIN_VOTERS` | `1` | `2` | `3` |
| `--neuron.sample_size` | `3` | `50` | `50` |
| Funding | Alice → transfer | Faucet | Purchase TAO |

---

## 10. R2 Bucket Management

### List bucket contents
```bash
source .venv-neurons/bin/activate && source .env

python -c "
import boto3, os
s3 = boto3.client('s3',
    endpoint_url=os.environ.get('R2_ENDPOINT_URL') or os.environ.get('R2_S3_ENDPOINT'),
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto',
)
resp = s3.list_objects_v2(Bucket=os.environ['R2_BUCKET_NAME'], MaxKeys=50)
for obj in resp.get('Contents', []):
    print(f'  {obj[\"Key\"]} ({obj[\"Size\"]} bytes)')
print(f'Total: {resp.get(\"KeyCount\", 0)} objects')
"
```

### Clear all bucket contents (fresh start)
```bash
source .venv-neurons/bin/activate && source .env

python -c "
import boto3, os
s3 = boto3.client('s3',
    endpoint_url=os.environ.get('R2_ENDPOINT_URL') or os.environ.get('R2_S3_ENDPOINT'),
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto',
)
bucket = os.environ['R2_BUCKET_NAME']
paginator = s3.get_paginator('list_objects_v2')
deleted = 0
for page in paginator.paginate(Bucket=bucket):
    objects = [{'Key': obj['Key']} for obj in page.get('Contents', [])]
    if objects:
        s3.delete_objects(Bucket=bucket, Delete={'Objects': objects})
        deleted += len(objects)
print(f'Deleted {deleted} objects from bucket \"{bucket}\"')
"
```

---

## 11. Troubleshooting

### Docker: "name is already in use"
```bash
docker stop subtensor-devnet-stable 2>/dev/null
docker rm   subtensor-devnet-stable 2>/dev/null
# Then re-run the docker run command
```

### btcli: "'bool' object has no attribute 'metadata'"
This is a known btcli 9.7.x compatibility issue with certain subtensor
Docker image versions. Use the Python helper script (`scripts/localnet_setup.py`)
or the bittensor SDK directly.

### Validator: "relative path can't be expressed as a file URI"
The `--neuron.flywheel_image_cache_root` and
`--neuron.flywheel_commercial_dataset_prefix` must be **absolute paths**.
Use `PROJECT_ROOT="$(pwd)"` and reference `"${PROJECT_ROOT}/..."`.

### Miner not receiving tasks
1. Verify miner is registered on subnet (check metagraph)
2. Verify `LOCALNET_MINER_PORT_BY_SS58` uses the correct miner SS58 address
3. Verify miner is serving: `curl http://127.0.0.1:8091`
4. Check validator logs for "dendrite" errors

### YOLO model not found
Download the model:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Subnet creation fails with insufficient funds
The owner wallet needs TAO. In devnet mode, transfer from Alice first:
```bash
python scripts/localnet_setup.py --step fund
```

### Wallet creation prompts for path interactively
Always pass `--wallet-path ~/.bittensor/wallets` to suppress the prompt.

---

## Quick Reference: Full Clean-Slate Run

```bash
# === CLEANUP ===
pkill -f "neurons/miner.py" || true; pkill -f "neurons/validator.py" || true
docker stop subtensor-devnet-stable 2>/dev/null; docker rm subtensor-devnet-stable 2>/dev/null
rm -rf ~/.bittensor/wallets/ ~/.bittensor/miners/ ./artifacts/localnet/ ./artifacts/miner_annotation/

# === SUBTENSOR ===
docker run -d --name subtensor-devnet-stable -p 9944:9944 -p 30333:30333 \
  ghcr.io/opentensor/subtensor:latest \
  --dev --rpc-external --rpc-methods=unsafe --rpc-cors=all --alice --force-authoring

# Wait for RPC
for i in $(seq 1 30); do curl -sS -m 2 -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' \
  http://127.0.0.1:9944 2>/dev/null | grep -q '"result"' && echo "RPC up" && break; sleep 2; done

# === WALLETS ===
source .venv-btcli/bin/activate
btcli wallet create --wallet-name alice --wallet-path ~/.bittensor/wallets --hotkey default --no-use-password --overwrite --uri Alice
btcli wallet create --wallet-name owner --wallet-path ~/.bittensor/wallets --hotkey ownerhk --n-words 12 --no-use-password --overwrite
btcli wallet create --wallet-name validator --wallet-path ~/.bittensor/wallets --hotkey valhk --n-words 12 --no-use-password --overwrite
btcli wallet create --wallet-name miner --wallet-path ~/.bittensor/wallets --hotkey minerhk --n-words 12 --no-use-password --overwrite

# === CHAIN SETUP (fund + subnet + register) ===
source .venv-neurons/bin/activate
python scripts/localnet_setup.py --chain-endpoint ws://127.0.0.1:9944

# === WORKSPACE ===
mkdir -p ./artifacts/{miner_annotation,localnet/self_hosted_image_cache,localnet/self_hosted_commercial}

# === MINER (Terminal 1) ===
source .env
MINER_SS58=$(python -c "import bittensor as bt; print(bt.wallet(name='miner',hotkey='minerhk').hotkey.ss58_address)")
echo "Miner SS58: $MINER_SS58"
env PYTHONPATH=. MINER_ADVERSARIAL=0 python neurons/miner.py \
  --wallet.name miner --wallet.hotkey minerhk \
  --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --netuid 2 --miner.model_backend yolo_local --miner.yolo_pretrained_weights yolov8n.pt \
  --miner.skip_training --axon.port 8091 --logging.debug \
  --miner.dual_flywheel_r2_prefix localnet/miners/miner1

# === VALIDATOR (Terminal 2) ===
PROJECT_ROOT="$(pwd)"
env PYTHONPATH=. FORCE_LOCAL_SET_WEIGHTS=1 \
  DEFAULT_ACCEPT_CONFIDENCE=0.01 DEFAULT_ACCEPT_SEVERITY_CONFIDENCE=0.01 \
  DEFAULT_MIN_VOTERS=1 DEFAULT_MIN_MEAN_IOU_TO_MEDIAN=0.1 \
  LOCALNET_MINER_PORT_BY_SS58="${MINER_SS58}=8091" \
  FALLBACK_SINGLE_MINER_ENABLED=1 FALLBACK_SINGLE_MINER_MIN_RELIABILITY=0.3 \
  python neurons/validator.py \
  --wallet.name validator --wallet.hotkey valhk \
  --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --netuid 2 --axon.port 8090 --neuron.sample_size 3 \
  --neuron.forward_step_sleep_seconds 15 --neuron.annotation_timeout 300 \
  --neuron.flywheel_commercial_export_every 1 \
  --neuron.flywheel_commercial_dataset_prefix "file://${PROJECT_ROOT}/artifacts/localnet/self_hosted_commercial" \
  --neuron.flywheel_image_cache_root "${PROJECT_ROOT}/artifacts/localnet/self_hosted_image_cache" \
  --logging.debug
```

---

## Additional Resources

- [Localnet Guide](LOCALNET_GUIDE.md) — Focused localnet quick-start
- [Miner Guide](MINER.md) — Detailed miner configuration
- [Validator Guide](VALIDATOR.md) — Detailed validator configuration
- [Official Bittensor Docs](https://docs.bittensor.com/)
- [Bittensor Learn](https://docs.learnbittensor.org/)