#!/usr/bin/env python3
"""Dataset Batch Partitioning and Golden Sample Vault Isolation Engine.

Partitions raw aerial raster imagery into strict 30-chip rounds/batches:
- Exactly 3 Golden Audit Samples held in the Validator's Private Vault (NEVER uploaded to public R2).
- Exactly 27 Public Commercial Chips uploaded to Cloudflare R2 under {dataset_name}/batch_{N}/images/.
- Neutralizes all chip identifiers (removes 'golden', 'raw', or revealing metadata).
- Generates public batch manifests and private validator ground-truth annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import dotenv_values
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from template.protocol import R2AccessCredentials
from template.hazard.r2_storage import (
    load_r2_credentials_from_env,
    upload_bytes_to_r2,
    upload_image_to_r2,
    _s3_client,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("batch_partitioner")


def get_neutral_chip_id(image_path: Path, prefix: str = "chip") -> str:
    """Derive a clean, neutral, unrevealing identifier based on content hash."""
    h = hashlib.sha256(image_path.read_bytes()).hexdigest()[:16]
    suffix = ".jpg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else ".png"
    return f"{prefix}_{h}{suffix}"


def _unique_images(paths: List[Path]) -> Tuple[List[Path], int]:
    """Keep the first deterministic path for each distinct content hash."""
    unique: List[Path] = []
    seen: set[str] = set()
    duplicates = 0
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        unique.append(path)
    return unique, duplicates


def partition_dataset(
    raw_images_dir: Path,
    golden_labels_path: Path,
    golden_source_dir: Optional[Path] = None,
    dataset_name: str = "climate_mrv",
    batch_size: int = 30,
    golden_per_batch: int = 3,
    public_per_batch: int = 27,
    max_batches: Optional[int] = None,
    upload_r2: bool = False,
    r2_creds: Optional[R2AccessCredentials] = None,
    public_base_dir: Optional[Path] = None,
    private_vault_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Partition raw dataset into batches with strict golden isolation."""
    if golden_per_batch + public_per_batch != batch_size:
        raise ValueError(f"Golden ({golden_per_batch}) + Public ({public_per_batch}) must equal Batch Size ({batch_size})")

    public_base = public_base_dir or (REPO_ROOT / "artifacts" / "commercial_dataset" / dataset_name)
    private_base = private_vault_dir or (REPO_ROOT / "artifacts" / "validator_private_golden" / dataset_name)
    public_base.mkdir(parents=True, exist_ok=True)
    private_base.mkdir(parents=True, exist_ok=True)
    if any(public_base.glob("batch_*")) or any(private_base.glob("batch_*")):
        raise FileExistsError("Output directories already contain batches; use fresh isolated output paths to avoid mixing runs.")

    # 1. Load ground truth labels
    golden_labels: Dict[str, Any] = {}
    if golden_labels_path.is_file():
        golden_labels = json.loads(golden_labels_path.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(golden_labels)} ground-truth annotations from {golden_labels_path}")

    # 2. Gather distinct candidate images. Content hashes, rather than
    # filenames, prevent duplicate copies from being sampled more than once.
    all_raw_files, raw_duplicates = _unique_images(
        sorted(list(raw_images_dir.glob("*.jpg")) + list(raw_images_dir.glob("*.jpeg")) + list(raw_images_dir.glob("*.png")))
    )
    if not all_raw_files:
        raise FileNotFoundError(f"No images found in {raw_images_dir}")
    logger.info(f"Discovered {len(all_raw_files)} distinct raw images in {raw_images_dir}")

    golden_pool_files: List[Path] = []
    golden_duplicates = 0
    if golden_source_dir and Path(golden_source_dir).is_dir():
        golden_pool_files, golden_duplicates = _unique_images(
            sorted(list(Path(golden_source_dir).glob("*.jpg")) + list(Path(golden_source_dir).glob("*.jpeg")) + list(Path(golden_source_dir).glob("*.png")))
        )
        logger.info(f"Discovered {len(golden_pool_files)} distinct golden images in {golden_source_dir}")

    # A chip may have only one role in the run. Exclude cross-pool duplicates.
    golden_hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in golden_pool_files}
    overlapping_raw = [p for p in all_raw_files if hashlib.sha256(p.read_bytes()).hexdigest() in golden_hashes]
    if overlapping_raw:
        overlap_paths = set(overlapping_raw)
        all_raw_files = [p for p in all_raw_files if p not in overlap_paths]
        logger.warning("Excluded %d raw chips duplicated in the private golden pool.", len(overlapping_raw))

    # If there is no separate golden pool, only raw images with an explicit
    # ground-truth record may be held back as golden samples.
    if not golden_pool_files:
        golden_pool_files = [p for p in all_raw_files if p.name in golden_labels]
        if not golden_pool_files:
            raise ValueError("No golden pool was supplied and no raw images have ground-truth entries.")
        golden_hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in golden_pool_files}
        all_raw_files = [p for p in all_raw_files if hashlib.sha256(p.read_bytes()).hexdigest() not in golden_hashes]

    missing_golden_labels = [p.name for p in golden_pool_files if p.name not in golden_labels]
    if missing_golden_labels:
        raise ValueError(f"Golden samples lack ground-truth entries: {missing_golden_labels[:5]}")

    capacity = min(len(all_raw_files) // public_per_batch, len(golden_pool_files) // golden_per_batch)
    requested = max_batches if max_batches is not None else capacity
    if requested < 1:
        raise ValueError("max_batches must be a positive integer.")
    total_possible_batches = min(capacity, requested)
    if total_possible_batches < 1:
        raise ValueError(
            "Not enough distinct, labeled images for one complete batch: "
            f"public={len(all_raw_files)} (need {public_per_batch}), "
            f"golden={len(golden_pool_files)} (need {golden_per_batch})."
        )
    if requested > capacity:
        logger.warning(
            "Requested up to %d complete batches, but only %d fit without reuse. "
            "%d public and %d golden images will remain unused.",
            requested,
            capacity,
            len(all_raw_files) - total_possible_batches * public_per_batch,
            len(golden_pool_files) - total_possible_batches * golden_per_batch,
        )
    if upload_r2 and r2_creds is None:
        raise RuntimeError("R2 upload was requested but no valid R2 credentials were loaded.")

    batches_summary = []

    for batch_num in range(1, total_possible_batches + 1):
        batch_id = f"batch_{batch_num}"
        logger.info(f"\n=======================================================")
        logger.info(f"Formulating {dataset_name} / {batch_id} (Target: {batch_size} chips)")
        logger.info(f"=======================================================")

        # Directories
        batch_pub_dir = public_base / batch_id
        batch_pub_images_dir = batch_pub_dir / "images"
        batch_pub_images_dir.mkdir(parents=True, exist_ok=True)

        batch_priv_dir = private_base / batch_id
        batch_priv_images_dir = batch_priv_dir / "images"
        batch_priv_images_dir.mkdir(parents=True, exist_ok=True)

        # Select disjoint, fully populated portions of each pool. Never wrap
        # or repeat examples to pad a short final batch.
        golden_start = (batch_num - 1) * golden_per_batch
        batch_golden_srcs = golden_pool_files[golden_start : golden_start + golden_per_batch]
        public_start = (batch_num - 1) * public_per_batch
        batch_public_srcs = all_raw_files[public_start : public_start + public_per_batch]
        if len(batch_golden_srcs) != golden_per_batch or len(batch_public_srcs) != public_per_batch:
            raise RuntimeError(f"Refusing incomplete batch {batch_id}; sample reuse or padding is prohibited.")

        logger.info(f"Selected {len(batch_golden_srcs)} golden audit chips and {len(batch_public_srcs)} public commercial chips.")

        # C. Process Private Golden Samples (NEVER UPLOADED TO R2)
        golden_vault_records = []
        for idx, g_src in enumerate(batch_golden_srcs, 1):
            neutral_id = get_neutral_chip_id(g_src, prefix="chip")
            dest_path = batch_priv_images_dir / neutral_id
            shutil.copy2(g_src, dest_path)

            gt_info = golden_labels.get(g_src.name, {})
            with Image.open(dest_path) as im:
                width, height = im.size

            golden_vault_records.append({
                "chip_id": neutral_id,
                "original_filename": g_src.name,
                "vault_path": str(dest_path),
                "width": width,
                "height": height,
                "annotations": gt_info.get("annotations", []),
                "biome": gt_info.get("biome", "Boreal Forest SOTA"),
                "is_golden": True,
            })

        # Save private golden labels to validator vault
        priv_labels_file = batch_priv_dir / "golden_labels.json"
        priv_labels_file.write_text(json.dumps(golden_vault_records, indent=2), encoding="utf-8")
        logger.info(f"  [PRIVATE VAULT] Saved {len(golden_vault_records)} golden ground-truth records to {priv_labels_file} (Zero public exposure)")

        # D. Process Public Commercial Images (Uploaded to R2 & Local Mirror)
        public_manifest_records = []
        upload_tasks = []

        def _process_and_upload_chip(item):
            idx, p_src = item
            neutral_id = get_neutral_chip_id(p_src, prefix="chip")
            dest_path = batch_pub_images_dir / neutral_id
            shutil.copy2(p_src, dest_path)

            with Image.open(dest_path) as im:
                width, height = im.size

            r2_url = f"/chips/{neutral_id}"
            if upload_r2 and r2_creds is not None:
                r2_key = f"{dataset_name}/{batch_id}/images/{neutral_id}"
                try:
                    r2_url = upload_image_to_r2(dest_path, object_key=r2_key, creds=r2_creds)
                except Exception as e:
                    raise RuntimeError(f"R2 image upload failed for {r2_key}: {e}") from e

            return {
                "index": idx,
                "image_id": Path(neutral_id).stem,
                "image_name": neutral_id,
                "image_url": r2_url,
                "r2_object_key": f"{dataset_name}/{batch_id}/images/{neutral_id}",
                "local_path": f"{dataset_name}/{batch_id}/images/{neutral_id}",
                "width": width,
                "height": height,
                "is_golden": False,
            }

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            public_manifest_records = list(executor.map(_process_and_upload_chip, enumerate(batch_public_srcs, 1)))

        # Sort back to preserve index order
        public_manifest_records.sort(key=lambda x: x["index"])

        # Save public manifest
        pub_manifest_file = batch_pub_dir / "manifest.json"
        pub_manifest_file.write_text(json.dumps(public_manifest_records, indent=2), encoding="utf-8")

        receipt = {
            "batch_id": batch_id,
            "expected_images": len(public_manifest_records),
            "verified_images": 0,
            "manifest_verified": False,
            "annotations_verified": False,
            "verified": False,
            "status": "not_uploaded",
        }
        if upload_r2 and r2_creds is not None:
            r2_manifest_key = f"{dataset_name}/{batch_id}/manifest.json"
            try:
                upload_bytes_to_r2(
                    pub_manifest_file.read_bytes(),
                    object_key=r2_manifest_key,
                    creds=r2_creds,
                    content_type="application/json",
                )
                s3 = _s3_client(r2_creds)
                for record in public_manifest_records:
                    s3.head_object(Bucket=r2_creds.bucket_name, Key=record["r2_object_key"])
                    receipt["verified_images"] += 1
                s3.head_object(Bucket=r2_creds.bucket_name, Key=r2_manifest_key)
                receipt["manifest_verified"] = True
                receipt["status"] = "source_uploaded"
                logger.info(f"  [PUBLIC R2] Uploaded and verified {batch_id} source images and manifest")
            except Exception as e:
                raise RuntimeError(f"R2 upload verification failed for {batch_id}: {e}") from e
        (batch_pub_dir / "r2_upload_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

        logger.info(f"  [PUBLIC BATCH] Exported {len(public_manifest_records)} public commercial chips to {batch_pub_images_dir}")

        batches_summary.append({
            "batch_id": batch_id,
            "round_num": batch_num,
            "total_images": len(golden_vault_records) + len(public_manifest_records),
            "golden_samples_private": len(golden_vault_records),
            "public_commercial_images": len(public_manifest_records),
            "public_manifest_path": str(pub_manifest_file),
            "private_golden_vault_path": str(priv_labels_file),
        })

    summary_file = public_base / "batch_summary.json"
    summary_payload = {
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "golden_per_batch": golden_per_batch,
        "public_per_batch": public_per_batch,
        "complete_batches": len(batches_summary),
        "source_inventory": {
            "unique_public_images": len(all_raw_files),
            "duplicate_public_files_excluded": raw_duplicates,
            "unique_labeled_golden_images": len(golden_pool_files),
            "duplicate_golden_files_excluded": golden_duplicates,
            "public_golden_overlap_excluded": len(overlapping_raw),
            "unused_public_images": len(all_raw_files) - len(batches_summary) * public_per_batch,
            "unused_golden_images": len(golden_pool_files) - len(batches_summary) * golden_per_batch,
        },
        "batches": batches_summary,
    }
    summary_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    logger.info(f"\nSuccessfully partitioned {len(batches_summary)} batches! Summary written to {summary_file}")
    return batches_summary


def main():
    parser = argparse.ArgumentParser(description="Partition Raw Dataset into 30-Image Batches (27 Public + 3 Golden Private)")
    parser.add_argument("--raw-dir", type=str, default="data/climate_mrv/samples/raw", help="Path to raw source images")
    parser.add_argument("--golden-dir", type=str, default="data/climate_mrv/samples/golden", help="Path to golden source images")
    parser.add_argument("--golden-labels", type=str, default="data/climate_mrv/samples/golden_labels.json", help="Path to golden ground-truth labels")
    parser.add_argument("--dataset-name", type=str, default="climate_mrv", help="Dataset namespace (default: climate_mrv)")
    parser.add_argument("--max-batches", type=int, default=None, help="Maximum complete batches to formulate")
    parser.add_argument("--public-base-dir", type=str, default=None, help="Optional isolated output root for public batches")
    parser.add_argument("--private-vault-dir", type=str, default=None, help="Optional isolated output root for golden batches")
    parser.add_argument("--upload-r2", action="store_true", help="Upload public commercial chips to Cloudflare R2")
    args = parser.parse_args()

    r2_creds = None
    if args.upload_r2:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            r2_creds = load_r2_credentials_from_env()
            logger.info(f"Loaded Cloudflare R2 credentials for bucket: {r2_creds.bucket_name}")
        except Exception as e:
            logger.error(f"Could not load R2 credentials from env: {e}")
            raise SystemExit(1)

    resolve_input = lambda value: Path(value) if Path(value).is_absolute() else REPO_ROOT / value
    partition_dataset(
        raw_images_dir=resolve_input(args.raw_dir),
        golden_labels_path=resolve_input(args.golden_labels),
        golden_source_dir=resolve_input(args.golden_dir),
        dataset_name=args.dataset_name,
        batch_size=30,
        golden_per_batch=3,
        public_per_batch=27,
        max_batches=args.max_batches,
        upload_r2=args.upload_r2,
        r2_creds=r2_creds,
        public_base_dir=Path(args.public_base_dir) if args.public_base_dir else None,
        private_vault_dir=Path(args.private_vault_dir) if args.private_vault_dir else None,
    )


if __name__ == "__main__":
    main()
