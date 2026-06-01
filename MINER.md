# Miner Guide

## Install

```bash
git clone <repo-url>
cd bittensor-subnet-template-1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with `NETUID`, `SUBTENSOR_NETWORK`, `WALLET_NAME`, `WALLET_HOTKEY`, Cloudflare R2 credentials, and one miner backend.

## Run

```bash
source .env
python neurons/miner.py
```

No command-line flags are required for the basic path when `.env` is populated. CLI flags still override `.env` values.

## Backends

`self_hosted` sends training and inference requests to your own REST service:

```bash
MINER_MODEL_BACKEND=self_hosted
SELF_HOSTED_TRAIN_URL=http://localhost:8081/train
SELF_HOSTED_INFER_URL=http://localhost:8081/infer
SELF_HOSTED_API_KEY=
```

Start the reference server:

```bash
source .env
python server.py --host 127.0.0.1 --port 8081 --checkpoint "$YOLO_MODEL_PATH"
```

The server exposes `/train` and `/infer`. Use it as a reference for replacing the backend with your own model service.

`yolo_local` trains and runs Ultralytics YOLO directly inside the miner process:

```bash
MINER_MODEL_BACKEND=yolo_local
YOLO_MODEL_PATH=yolov8n.pt
YOLO_EPOCHS=10
YOLO_IMGSZ=640
YOLO_BATCH=16
```

`openai_vision` loads the OpenAI fine-tuning backend:

```bash
MINER_MODEL_BACKEND=openai_vision
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=
OPENAI_BASE_MODEL=gpt-4o-2024-08-06
```

Actual OpenAI fine-tuning can incur API costs and may require account/project setup. Do a dry run in a development environment before using funded credentials.

## R2 Output

Miners upload one `annotations.json` object per task under `MINER_R2_PREFIX`. The validator reads these objects with real boto3/R2 calls. There is no mock S3 path in production code.
