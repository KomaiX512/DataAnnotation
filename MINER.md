# Miner Documentation — Version 1.5.0

This guide details the miner requirements, payload schemas, batch synchronization rules, and high-precision polygon specifications for **DataAnnotation Subnet v1.5.0**.

---

## 1. Subnet Overview & v1.5.0 Changes

Miners in the DataAnnotation subnet detect, classify, and delineate ecological features (tree canopies, forestry crowns, biomass regions) on high-resolution aerial and satellite imagery.

### Key Highlights in v1.5.0:
- **Round-by-Round Batch Synchronization**: Datasets are dispatched in discrete batches (`batch_1`, `batch_2`, etc.). Miners must strictly submit annotations matching the active `batch_id` of the current round.
- **Mid-Round Arrival Protocol**: Newly registered or restarted miners that join mid-round must await the completion of the in-flight round and synchronize cleanly with the next round.
- **High-Precision Polygons Requirement**: Crude rectangular bounding boxes are penalized. The multimodal evaluator explicitly rewards tight, multi-vertex segmentation polygons (`polygon: [[x, y], ...]`) hugging tree canopies.
- **Cloudflare R2 Artifact Transport**: Large annotation payloads are uploaded to R2 and referenced via presigned or authenticated URIs in the synapse response.

---

## 2. Protocol & Task Lifecycle

### Task Synapse (`AnnotationTask`)
Validators query miners with an `AnnotationTask` containing:
- `task_id` (str): Unique cryptographic UUID for this evaluation round.
- `batch_id` (str): Identifier of the active dataset batch (e.g. `"batch_1"`).
- `round_num` (int): Monotonically increasing round number.
- `annotation_images` (list): Up to 30 images. Each entry contains:
  - `image_id`: Opaque token unique to your hotkey (e.g. `"img_a8f9c1..."`).
  - `image_url`: Download URL or accessible image endpoint.
- `response_window`: Maximum time window (600 seconds) from dispatch to artifact upload.

### Mid-Round Arrival Behavior
If your miner initializes while a round is actively being processed on the network:
1. The miner queries the validator status or catches the current round's dispatch.
2. If the miner missed the start of the round or cannot meet the response window, it enters a `WAIT_FOR_NEXT_ROUND` state.
3. Upon arrival of the subsequent `round_num` (e.g. Round 4 after Round 3 finishes), the miner immediately participates in the full round.

### Strict Batch Synchronization
- Always inspect `synapse.batch_id`.
- The annotations uploaded to R2 **must** correspond only to the images received in the active batch.
- **Warning**: Submitting annotations from an earlier batch (e.g. `batch_1` during a `batch_2` round) triggers a **SECURITY VIOLATION**, resulting in immediate rejection and 0.0 reward.

---

## 3. High-Precision Polygon Specification

Validators employ multimodal visual reasoning models (such as Qwen2.5-VL) to adjudicate miner submissions. The evaluator directly inspects visual fidelity, crown completeness, and boundary precision.

### Polygon Data Format
Each annotation item in `annotations` must include a high-precision polygon:
```json
{
  "hazard_class": "individual_tree",
  "confidence": 0.94,
  "bounding_box": [1667.22, 1739.97, 1889.21, 1968.41],
  "polygon": [
    [1724.8, 1740.8],
    [1721.6, 1744.0],
    [1718.4, 1744.0],
    [1692.0, 1756.2],
    [1675.4, 1780.0],
    [1667.2, 1820.5],
    [1678.1, 1890.3],
    [1740.2, 1955.0],
    [1820.0, 1968.4],
    [1889.2, 1930.1],
    [1875.0, 1820.0],
    [1820.5, 1750.2]
  ],
  "area": 38450.0,
  "weight": 0.009167
}
```

### Quality Guidelines:
1. **Contour Detail**: Polygons should contain sufficient vertices (typically 12 to 64 points) to trace the natural crown boundary.
2. **Exclusion of Non-Vegetation**: Ensure bare ground, asphalt, roofs, and man-made structures are not encompassed inside the polygon.
3. **Recall**: Detect all visible trees across the tile; severe under-detection or missing prominent trees leads to record rejection.

---

## 4. R2 Upload & Artifact Schema

Miners upload their serialized annotation results as a JSON file to Cloudflare R2 before returning the synapse.

### Payload Schema (`AnnotationsFilePayload`)
```json
{
  "records": [
    {
      "image_id": "img_a8f9c1...",
      "model_version": "tree_detection.pt",
      "timestamp": "2026-09-28T14:30:00Z",
      "annotations": [
        {
          "hazard_class": "individual_tree",
          "confidence": 0.95,
          "bounding_box": [100.0, 120.0, 250.0, 280.0],
          "polygon": [[120.0, 120.0], [240.0, 125.0], [250.0, 270.0], [110.0, 260.0]],
          "area": 19500.0,
          "weight": 0.00465
        }
      ]
    }
  ]
}
```

### Synapse Return Value
The miner populates the synapse with:
- `annotations_uri`: The URI pointing to the uploaded JSON file (e.g. `s3://subnet/miners/annotations/task-uuid/miner-uid.json` or `https://...`).
- `miner_r2_credentials`: Optional read-only credentials if the bucket requires authentication.

---

## 5. Security Bounds & Anti-Cheat Rules

To protect network integrity and prevent denial-of-service, validators enforce strict payload validation:

1. **Size Limits**:
   - Maximum artifact size: 16 MB.
   - Maximum records per artifact: 8,192.
   - Maximum annotations per image: 512.
   - Maximum total annotations per artifact: 100,000.
2. **Polygon Complexity Limit**:
   - Total quadratic polygon complexity $\sum (\text{vertices})^2 \le 2,000,000$. Payloads exceeding this threshold are rejected.
3. **Opaque Token Integrity**:
   - Use only the opaque `image_id` strings provided in the synapse. Responses referencing unknown image IDs are discarded.
4. **Duplicate & Coalition Detection**:
   - Submissions identical across multiple UIDs are flagged. Sybil coalitions copying weights or sharing identical coordinate sets are penalized.
5. **Annotation Density Cap**:
   - The protocol accepts up to 512 annotations per image. If a detector outputs more than 512 items, the validator automatically retains the top 512 predictions sorted by confidence descending.
6. **Canvas-Covering Box Filter**:
   - Oversized false-positive bounding boxes covering $\ge 70\%$ of the image canvas are automatically filtered out per-box without invalidating the rest of the batch. Miners should pre-filter any predictions exceeding $60\%$ area ratio.
7. **Decentralized Ensemble Consensus**:
   - In Architecture v1.5, winners are determined per-image rather than winner-takes-all for the entire task. Miners whose predictions excel on specific tiles have their data selected, rewarded, and incorporated into the commercial dataset.

---

## 6. Running a Miner

```bash
python neurons/miner.py \
    --netuid <NETUID> \
    --subtensor.network <NETWORK> \
    --wallet.name <WALLET> \
    --wallet.hotkey <HOTKEY> \
    --axon.port <PORT> \
    --model_path models/tree_detection.pt
```

Ensure your environment variables for R2 credentials (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`) are set in your `.env` or system environment.
