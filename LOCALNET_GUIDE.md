# Subnet E2E Localnet Testing Guide

This guide describes how to run a ground-up, end-to-end localnet verification of the Hazard Subnet with 3 miners (YOLO Local, Self-Hosted API, and Adversarial Local) and a Validator.

---

## 1. Environment Clean Slate

Kill any running processes and clean local state:

```bash
# Kill subtensor, miner, validator, or python server processes
pkill -f "neurons/miner.py"
pkill -f "neurons/validator.py"
pkill -f "server.py"
docker stop subtensor-devnet-stable || true

# Remove wallet data, chain state, and cached models/data
rm -rf ~/.bittensor/wallets/
rm -rf ~/.bittensor/miners/
rm -rf ./artifacts/localnet/self_hosted_image_cache/
rm -rf ./artifacts/localnet/self_hosted_commercial/
```

---

## 2. Infrastructure & Subnet Setup

### Step A: Start the Subtensor Devnet Container
Run a local subtensor in devnet mode:
```bash
docker run --rm --name subtensor-devnet-stable -d -p 9944:9944 -p 30333:30333 opentensor/subtensor:latest subtensor-node --dev --ws-external --rpc-external
```

### Step B: Create Wallets
Create the necessary wallets (Owner, Validator, and three Miners):
```bash
# Owner Wallet
btcli wallet create --wallet.name owner --wallet.hotkey ownerhk --no_use_password

# Validator Wallet
btcli wallet create --wallet.name validator --wallet.hotkey valhk --no_use_password

# Miner 1 Wallet (Honest YOLO Local)
btcli wallet create --wallet.name miner --wallet.hotkey minerhk --no_use_password

# Miner 2 Wallet (Honest Self-Hosted API)
btcli wallet create --wallet.name miner2 --wallet.hotkey minerhk2 --no_use_password

# Miner 3 Wallet (Adversarial YOLO Local)
btcli wallet create --wallet.name miner3 --wallet.hotkey minerhk3 --no_use_password
```

### Step C: Mint Tokens and Transfer
Fund the keys from the default devnet Alice key (which is prefunded in `--dev` mode):
```bash
# Transfer to Owner (default Alice address is active)
btcli wallet transfer --wallet.name owner --dest 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY --amount 1000 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944

# Transfer to Validator
btcli wallet transfer --wallet.name owner --dest <VAL_ADDRESS> --amount 100 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944

# Transfer to Miners
btcli wallet transfer --wallet.name owner --dest <MINER1_ADDRESS> --amount 50 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944
btcli wallet transfer --wallet.name owner --dest <MINER2_ADDRESS> --amount 50 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944
btcli wallet transfer --wallet.name owner --dest <MINER3_ADDRESS> --amount 50 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944
```

### Step D: Register Subnet (netuid 2)
Register the subnet on the local subtensor node:
```bash
btcli subnet create --wallet.name owner --wallet.hotkey ownerhk --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944
```

### Step E: Register Validators and Miners
```bash
# Register Validator
btcli subnet register --netuid 2 --wallet.name validator --wallet.hotkey valhk --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944

# Register Miner 1
btcli subnet register --netuid 2 --wallet.name miner --wallet.hotkey minerhk --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944

# Register Miner 2
btcli subnet register --netuid 2 --wallet.name miner2 --wallet.hotkey minerhk2 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944

# Register Miner 3
btcli subnet register --netuid 2 --wallet.name miner3 --wallet.hotkey minerhk3 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944
```

---

## 3. Running Subnet Components

### Step A: Start the Miner 2 Backend API
Start the REST reference inference API on port `8081`:
```bash
env PYTHONPATH=. python server.py --host 127.0.0.1 --port 8081 --checkpoint yolov8n.pt
```

### Step B: Start Miner 1 (YOLO Local, Honest)
Run the YOLO local miner on port `8091`:
```bash
env PYTHONPATH=. MINER_ADVERSARIAL=0 .venv-neurons/bin/python neurons/miner.py \
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

### Step C: Start Miner 2 (Self-Hosted Backend, Honest)
Run the self-hosted REST backend miner on port `8092`:
```bash
env PYTHONPATH=. MINER_ADVERSARIAL=0 .venv-neurons/bin/python neurons/miner.py \
  --wallet.name miner2 \
  --wallet.hotkey minerhk2 \
  --subtensor.network local \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --netuid 2 \
  --miner.model_backend self_hosted \
  --miner.self_hosted_infer_url http://127.0.0.1:8081/infer \
  --miner.self_hosted_train_url http://127.0.0.1:8081/train \
  --miner.skip_training \
  --axon.port 8092 \
  --logging.debug \
  --miner.dual_flywheel_r2_prefix localnet/miners/miner2
```

### Step D: Start Miner 3 (YOLO Local, Adversarial)
Run the YOLO local miner in adversarial mode (`MINER_ADVERSARIAL=1` environment variable forces it to return garbage synthetic boxes) on port `8093`:
```bash
env PYTHONPATH=. MINER_ADVERSARIAL=1 .venv-neurons/bin/python neurons/miner.py \
  --wallet.name miner3 \
  --wallet.hotkey minerhk3 \
  --subtensor.network local \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --netuid 2 \
  --miner.model_backend yolo_local \
  --miner.yolo_pretrained_weights yolov8n.pt \
  --miner.skip_training \
  --axon.port 8093 \
  --logging.debug \
  --miner.dual_flywheel_r2_prefix localnet/miners/miner3
```

### Step E: Start Validator
Run the validator on port `8090` with weight setting enabled and relaxed thresholds to verify export features immediately:
```bash
env PYTHONPATH=. FORCE_LOCAL_SET_WEIGHTS=1 \
  DEFAULT_ACCEPT_CONFIDENCE=0.01 \
  DEFAULT_ACCEPT_SEVERITY_CONFIDENCE=0.01 \
  DEFAULT_MIN_VOTERS=1 \
  DEFAULT_MIN_MEAN_IOU_TO_MEDIAN=0.1 \
  LOCALNET_MINER_PORT_BY_SS58="5G3gGcsnQ7pCQaPxDzw2tUJT8CqVYWNRKyU9CGyvgurqV8wR=8091,5DydbWcSSVrsNSrePzbVkuBqJmotvXqsbEaBWjocGu7QuTdJ=8092,5EJ7gjxraHvGUte62c3zarrm3GE2RMj6kobCPYQKkUHJ6Von=8093" \
  .venv-neurons/bin/python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey valhk \
  --subtensor.network local \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --netuid 2 \
  --axon.port 8090 \
  --neuron.sample_size 3 \
  --neuron.forward_step_sleep_seconds 15 \
  --neuron.annotation_timeout 300 \
  --neuron.flywheel_coco_manifest artifacts/localnet/coco200/manifest.json \
  --neuron.flywheel_annotation_request_size 15 \
  --neuron.flywheel_golden_injection_per_request 5 \
  --neuron.flywheel_commercial_export_every 1 \
  --neuron.flywheel_commercial_dataset_prefix file:///home/komail/bittensor-subnet-template-1/artifacts/localnet/self_hosted_commercial \
  --neuron.flywheel_image_cache_root artifacts/localnet/self_hosted_image_cache \
  --logging.debug
```

---

## 4. Verification and Output Checking

### Verify Miner Scoring and Incentives
Once a step finishes, check the validator console output/logs for the updated moving average weights:
```
Updated moving avg scores: [0.  0.  0.0322753  0.02556132  0.00236959]
```
* **Miner 1 (UID 2)**: High Score (`~0.032`)
* **Miner 2 (UID 3)**: High Score (`~0.025`)
* **Miner 3 (UID 4)**: Extremely Low/Zero Score (`~0.002`) due to submitting garbage adversarial bounding boxes.

### Verify Commercial Dataset Export
Confirm the commercial dataset JSONL contains processed annotations with replaced public R2 images:
```bash
cat artifacts/localnet/self_hosted_commercial/commercial-dataset-step-0.jsonl
```
Key outputs to verify in the exported JSONL:
1. `image_url` is replaced with a permanent HTTP(S) R2 link (e.g. `https://pub-3aa7ed152eb9407cb756c8349a5ef02f.r2.dev/commercial-images/<hash>.jpg`).
2. Objects array displays aggregated consensus bounding boxes fused from Miner 1 and Miner 2.
3. No secret Golden images are leaked (filtered out of the export).

---
