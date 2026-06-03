# Miner Setup Guide

What you need:
* A wallet registered on the subnet
* A Cloudflare R2 bucket (or MinIO) with access keys
* A model: either a self-hosted API, a local YOLO model, or an OpenAI API key

## Step 1: Install & configure

```bash
git clone https://github.com/KomaiX512/JHA_subnet.git bittensor-subnet-template-1 && cd bittensor-subnet-template-1
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your `WALLET_NAME`, `WALLET_HOTKEY`, `NETUID`, `SUBTENSOR_NETWORK`, R2 credentials, and choose one backend.

## Step 2: Choose your backend

The miner supports three backends. Set `MINER_MODEL_BACKEND` to one of:

* **`self_hosted`** – Use your own REST API that implements `/train` and `/infer`.
  Set `SELF_HOSTED_TRAIN_URL`, `SELF_HOSTED_INFER_URL`.
  You can use the provided reference server:
  ```bash
  source .env
  PYTHONPATH=. python server.py --host 127.0.0.1 --port 8081 --checkpoint yolov8n.pt
  ```
  This starts a real YOLOv8 training & inference server. The miner will call it automatically.

* **`yolo_local`** – Fine-tune YOLO directly on your GPU.
  Set `YOLO_MODEL_PATH`, `YOLO_EPOCHS`, `YOLO_IMGSZ`, `YOLO_BATCH`.
  The miner handles training, caching, and inference locally.

* **`openai_vision`** – Use OpenAI's vision fine-tuning API.
  Set `OPENAI_API_KEY`, `OPENAI_BASE_MODEL`.
  *Warning: this path incurs API costs. Test with a dry-run first.*

## Step 3: Run the miner

```bash
source .env
PYTHONPATH=. python neurons/miner.py
```

The miner will receive annotation tasks, train (if needed), run inference, upload `annotations.json` to your R2 bucket, and print its progress.

## Verification

Check your R2 bucket for `annotations.json` files. If you see them, the miner is working. The validator (when running) will pick them up.
