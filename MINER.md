# Miner Setup Guide

This guide will walk you through setting up and running a miner on the Decentralized Data Annotation Subnet. Miners receive image annotation tasks, train or fine-tune their detectors, run inference, upload their annotations to Cloudflare R2, and return references to the validator.

---

## 1. Install & Run

First, clone the repository, set up a virtual environment, and install dependencies:

```bash
git clone https://github.com/KomaiX512/JHA_subnet.git
cd JHA_subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

To run your miner:

```bash
source .env
python neurons/miner.py
```

No command-line flags are required if your `.env` is fully populated. Any command-line flags passed will override the `.env` settings.

---

## 2. Backend Choice

Choose the backend that matches your infrastructure and resources:

| Backend | Description | When to Use |
| :--- | :--- | :--- |
| **`self_hosted`** | Connects to your own REST API | You have a custom model served via an external API. |
| **`yolo_local`** | Trains YOLO directly on your GPU | You want a simple, fully local setup using Ultralytics. |
| **`openai_vision`** | Fine-tunes and queries OpenAI | You want to use GPT-4o for high-fidelity annotations. |

---

## 3. Setup Guides for Each Backend

### Option A: Self-Hosted Setup (`self_hosted`)

The self-hosted backend delegates training and inference requests to an external REST server. A reference FastAPI server wrapping a YOLOv8 training/inference lifecycle is provided in the codebase.

1. **Start the reference server** on port `8081`:
   ```bash
   python server.py --host 127.0.0.1 --port 8081 --checkpoint yolov8n.pt
   ```

2. **Configure your `.env` file**:
   ```bash
   # ===== SUBNET & WALLET =====
   NETUID=1
   SUBTENSOR_NETWORK=localnet
   WALLET_NAME=default
   WALLET_HOTKEY=default

   # ===== CLOUDFLARE R2 =====
   R2_ACCESS_KEY_ID=your_access_key
   R2_SECRET_ACCESS_KEY=your_secret_key
   R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
   R2_BUCKET_NAME=annotation-subnet

   # ===== MINER CONFIG =====
   MINER_MODEL_BACKEND=self_hosted
   MINER_ANNOTATION_WORKSPACE=./artifacts/miner_annotation
   MINER_R2_PREFIX=miners/annotations

   # ===== SELF-HOSTED SETTINGS =====
   SELF_HOSTED_TRAIN_URL=http://localhost:8081/train
   SELF_HOSTED_INFER_URL=http://localhost:8081/infer
   SELF_HOSTED_API_KEY=your_api_key_if_configured
   SELF_HOSTED_POLL_INTERVAL_SECONDS=30
   ```

3. **Run the miner**:
   ```bash
   python neurons/miner.py
   ```

---

### Option B: YOLO Local Setup (`yolo_local`)

The `yolo_local` backend runs training and inference using Ultralytics YOLO directly inside the miner process. This requires an NVIDIA GPU with at least 8GB of VRAM.

1. **Configure your `.env` file**:
   ```bash
   # ===== SUBNET & WALLET =====
   NETUID=1
   SUBTENSOR_NETWORK=localnet
   WALLET_NAME=default
   WALLET_HOTKEY=default

   # ===== CLOUDFLARE R2 =====
   R2_ACCESS_KEY_ID=your_access_key
   R2_SECRET_ACCESS_KEY=your_secret_key
   R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
   R2_BUCKET_NAME=annotation-subnet

   # ===== MINER CONFIG =====
   MINER_MODEL_BACKEND=yolo_local
   MINER_ANNOTATION_WORKSPACE=./artifacts/miner_annotation
   MINER_R2_PREFIX=miners/annotations

   # ===== YOLO LOCAL SETTINGS =====
   YOLO_MODEL_PATH=yolov8n.pt
   YOLO_EPOCHS=10
   YOLO_IMGSZ=640
   YOLO_BATCH=16
   ```

2. **Run the miner**:
   ```bash
   python neurons/miner.py
   ```

---

### Option C: OpenAI Setup (`openai_vision`)

The `openai_vision` backend uploads training datasets, initiates OpenAI fine-tuning jobs, and queries the fine-tuned GPT-4o model for inference.

> [!WARNING]
> Fine-tuning GPT-4o can incur substantial API costs. We strongly recommend testing first with a local mock or a custom endpoint before using funded credentials.

1. **Configure your `.env` file**:
   ```bash
   # ===== SUBNET & WALLET =====
   NETUID=1
   SUBTENSOR_NETWORK=localnet
   WALLET_NAME=default
   WALLET_HOTKEY=default

   # ===== CLOUDFLARE R2 =====
   R2_ACCESS_KEY_ID=your_access_key
   R2_SECRET_ACCESS_KEY=your_secret_key
   R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
   R2_BUCKET_NAME=annotation-subnet

   # ===== MINER CONFIG =====
   MINER_MODEL_BACKEND=openai_vision
   MINER_ANNOTATION_WORKSPACE=./artifacts/miner_annotation
   MINER_R2_PREFIX=miners/annotations

   # ===== OPENAI SETTINGS =====
   OPENAI_API_KEY=sk-proj-...your_api_key...
   OPENAI_BASE_URL=
   OPENAI_BASE_MODEL=gpt-4o-2024-08-06
   OPENAI_N_EPOCHS=3
   OPENAI_BATCH_SIZE=1
   OPENAI_LEARNING_RATE_MULTIPLIER=1.8
   ```

   *If you are using a custom gateway or a mock service, set `OPENAI_BASE_URL` to point to it (e.g. `http://localhost:8000/v1`).*

2. **Run the miner**:
   ```bash
   python neurons/miner.py
   ```

---

## 4. Output Storage

Miners must write their bounding boxes and labels to a JSON document formatted as a `PerImageAnnotationItem` list and upload it to their Cloudflare R2 bucket. The uploaded object must match the key pattern:
`<MINER_R2_PREFIX>/<task_id>/annotations.json`

The validator will download this file using real S3 API calls. Ensure your S3 bucket permissions are configured to allow public read access or that presigned URL options are fully compatible.
