# Decentralized Data Annotation Subnet

This subnet turns Bittensor miners into competing image annotation workers. Validators send miners batches of images, miners label objects and hazards, and validators score the results against a hidden Golden Set plus peer consensus. The best accepted annotations are fused into an auditable commercial JSONL dataset.

## How It Works

1. Validators prepare a corpus with hidden Golden Set images and unlabeled annotation-pool images.
2. Miners receive image URLs, train or adapt their selected backend, run inference, and upload `annotations.json` to Cloudflare R2.
3. Validators download miner submissions, score Golden Set fidelity without leaking labels, compute rewards, set weights, and export accepted non-Golden annotations.
4. The dataset assembler keeps provenance, confidence, reliability, and per-miner vote audit fields for every exported object.

## Quick Start

```bash
git clone <repo-url>
cd bittensor-subnet-template-1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your wallet, subnet, Cloudflare R2 credentials, validator corpus, and miner backend. For a local miner using the reference REST backend, set:

```bash
MINER_MODEL_BACKEND=self_hosted
SELF_HOSTED_TRAIN_URL=http://localhost:8081/train
SELF_HOSTED_INFER_URL=http://localhost:8081/infer
```

Run a validator:

```bash
source .env
python neurons/validator.py
```

Run a miner:

```bash
source .env
python neurons/miner.py
```

Miner setup details live in `MINER.md`; validator corpus, Golden Set, reward, and export details live in `VALIDATOR.md`.
