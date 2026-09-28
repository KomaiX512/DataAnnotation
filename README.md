# Decentralized Data Annotation Subnet (v1.5.0)

The DataAnnotation subnet produces verified, high-precision carbon and forestry ecological annotations for Climate MRV (Monitoring, Reporting, and Verification).

Miners run high-accuracy computer vision segmentation models to delineate individual tree canopies, dense vegetation clusters, and mangrove forests from high-resolution aerial and satellite tiles. Validators score submissions against hidden labeled golden samples and adjudicate public imagery using multimodal visual reasoning models.

---

## What's New in Version 1.5.0

- **High-Precision Multi-Vertex Polygons**: Transitioned from crude bounding boxes to organic polygon crown contours (`polygon: [[x, y], ...]`). High-precision polygon delineations receive priority scoring from multimodal vision judges.
- **Round-by-Round Dataset Partitioning**: Datasets are segmented into deterministic batches (`batch_1`, `batch_2`, etc.) matching network rounds.
- **Mid-Round Arrival Synchronization**: Dynamic synchronization protocol ensuring miners registering mid-round cleanly await the next round before competing.
- **Strict Batch Verification & Anti-Cheat**: Out-of-sync batch injection protection, opaque per-hotkey token mapping, and quadratic polygon complexity bounds.
- **Cloudflare R2 Direct Artifact Pipeline**: Scalable, high-throughput payload retrieval pipeline via Cloudflare R2 object storage.
- **Hardened Validator Security**: Proprietary validator adjudicator and selection pipelines protected via authenticated in-memory runtime execution.

---

## Documentation Links

- **[Miner Documentation (v1.5.0)](MINER.md)**: Setup guide, high-precision polygon requirements, R2 payload format, and batch sync specifications.
- **[Validator Architecture](VALIDATOR.md)**: Epoch task scheduling, golden sample injection, and consensus adjudication.
- **[System Architecture](docs/ARCHITECTURE.md)**: Deep dive into the dual flywheel, incentive mechanisms, and security bounds.

---

## Quickstart

### Clone & Install
```bash
git clone https://github.com/KomaiX512/DataAnnotation.git
cd DataAnnotation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running a Miner
Consult [MINER.md](MINER.md) for full miner setup instructions.

```bash
python neurons/miner.py \
    --netuid 498 \
    --subtensor.network test \
    --wallet.name miner \
    --wallet.hotkey default \
    --axon.port 8091
```

---

## License
MIT License. Protected evaluation runtime components are proprietary to the subnet operator.
