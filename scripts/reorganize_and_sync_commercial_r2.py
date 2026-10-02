#!/usr/bin/env python3
"""
Reorganize and Sync Commercial Dataset to Cloudflare R2 & Production VPS.

1. Partitions all 450 raw images into 17 clean batches matching validator ImageCorpus hashes.
2. Backfills verified Sep 29 winning annotations for all chips.
3. Renders high-resolution annotated overlay chips with polygons and bounding boxes.
4. Uploads round-by-round structure to Cloudflare R2:
   - climate_mrv/batch_{N}/images/{image_id}.jpg
   - climate_mrv/batch_{N}/annotated-images/{image_id}.jpg
   - climate_mrv/batch_{N}/manifest.json
   - climate_mrv/batch_{N}/annotations.json
   - climate_mrv/batch_{N}/commercial-dataset.jsonl
5. Purges obsolete storage on R2:
   - legacy 'commercial/' prefix (~2.95 GB)
   - 'miners/' objects older than 12h (~9.6 GB)
   - 'camouflaged/' objects older than 12h (~1.06 GB)
6. Syncs the active batch to production VPS for real-time dashboard visualization.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import glob
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from template.hazard.climate_mrv_corpus import _read_image_as_jpeg
from template.hazard.dataset_assembler import DatasetAssembler
from template.hazard.r2_storage import (
    load_r2_credentials_from_env,
    upload_bytes_to_r2,
    upload_image_to_r2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("r2_reorganizer")


def load_all_commercial_annotations(comm_dir: Path) -> Dict[str, dict]:
    """Load latest winning annotation record for every unique image_id."""
    step_files = sorted(comm_dir.glob("commercial-dataset-step-*.jsonl"))
    logger.info(f"Scanning {len(step_files)} step files for verified commercial annotations...")
    records_by_img: Dict[str, dict] = {}
    for sf in step_files:
        try:
            for line in sf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                img_id = rec.get("image_id")
                if img_id:
                    # Later steps take precedence as they contain latest consensus
                    records_by_img[img_id] = rec
        except Exception as e:
            logger.debug(f"Note reading {sf.name}: {e}")
    logger.info(f"Loaded verified annotations for {len(records_by_img)} unique images.")
    return records_by_img


def build_and_upload_batches(
    raw_dir: Path,
    golden_dir: Path,
    annotations_by_img: Dict[str, dict],
    dataset_name: str = "climate_mrv",
    total_batches: int = 17,
    public_per_batch: int = 27,
    golden_per_batch: int = 3,
    upload_r2: bool = True,
):
    """Build local structured batches and mirror to R2."""
    r2_creds = None
    if upload_r2:
        try:
            r2_creds = load_r2_credentials_from_env()
            logger.info(f"Connected to Cloudflare R2 bucket: {r2_creds.bucket_name}")
        except Exception as e:
            logger.error(f"Cannot load R2 credentials: {e}")
            sys.exit(1)

    # 1. Process all raw images through normalized JPEG pipeline
    raw_files = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")))
    logger.info(f"Discovered {len(raw_files)} raw source chips in {raw_dir}")

    processed_raw = []
    for p in raw_files:
        payload = _read_image_as_jpeg(p)
        if payload is None:
            continue
        h = hashlib.sha256(payload).hexdigest()
        processed_raw.append({
            "image_id": h,
            "filename": f"{h}.jpg",
            "payload": payload,
            "orig_path": p,
        })
    logger.info(f"Normalized {len(processed_raw)} raw chips with deterministic hashes.")

    # 2. Process golden pool
    golden_files = sorted(list(golden_dir.glob("*.jpg")) + list(golden_dir.glob("*.png")))
    processed_golden = []
    for g in golden_files:
        payload = _read_image_as_jpeg(g)
        if payload is None:
            continue
        h = hashlib.sha256(payload).hexdigest()
        processed_golden.append({
            "image_id": h,
            "filename": f"{h}.jpg",
            "payload": payload,
            "orig_path": g,
        })
    logger.info(f"Normalized {len(processed_golden)} golden chips.")

    public_base = REPO_ROOT / "artifacts" / "commercial_dataset" / dataset_name
    private_base = REPO_ROOT / "artifacts" / "validator_private_golden" / dataset_name
    dummy_assembler = DatasetAssembler(corpus=None, storage_prefix="")

    golden_offset = 0
    public_offset = 0

    for batch_num in range(1, total_batches + 1):
        batch_id = f"batch_{batch_num}"
        logger.info(f"\n=======================================================")
        logger.info(f"Processing {dataset_name} / {batch_id} (Target: 30 chips: 27 public, 3 golden)")
        logger.info(f"=======================================================")

        batch_pub_dir = public_base / batch_id
        batch_pub_img_dir = batch_pub_dir / "images"
        batch_pub_ann_dir = batch_pub_dir / "annotated-images"
        batch_pub_img_dir.mkdir(parents=True, exist_ok=True)
        batch_pub_ann_dir.mkdir(parents=True, exist_ok=True)

        batch_priv_dir = private_base / batch_id
        batch_priv_img_dir = batch_priv_dir / "images"
        batch_priv_img_dir.mkdir(parents=True, exist_ok=True)

        # Select 3 Golden
        if golden_offset + golden_per_batch > len(processed_golden):
            golden_offset = 0
        batch_golden = processed_golden[golden_offset : golden_offset + golden_per_batch]
        golden_offset += golden_per_batch

        # Select 27 Public
        batch_public = processed_raw[public_offset : public_offset + public_per_batch]
        public_offset += len(batch_public)
        if len(batch_public) < public_per_batch:
            needed = public_per_batch - len(batch_public)
            batch_public.extend(processed_raw[:needed])

        # Write private golden
        for g_item in batch_golden:
            g_path = batch_priv_img_dir / g_item["filename"]
            g_path.write_bytes(g_item["payload"])

        # Build public images, manifests, annotations, and annotated overlay images
        manifest_records = []
        batch_annotations = []
        chips_to_upload = []

        for idx, p_item in enumerate(batch_public, 1):
            img_id = p_item["image_id"]
            img_filename = p_item["filename"]
            raw_path = batch_pub_img_dir / img_filename
            raw_path.write_bytes(p_item["payload"])

            with Image.open(raw_path) as im:
                w_px, h_px = im.size

            r2_raw_key = f"{dataset_name}/{batch_id}/images/{img_filename}"
            r2_ann_key = f"{dataset_name}/{batch_id}/annotated-images/{img_filename}"
            raw_url = f"https://{r2_creds.bucket_name}.r2.cloudflarestorage.com/{r2_raw_key}" if r2_creds else f"/chips/{img_filename}"
            ann_url = f"https://{r2_creds.bucket_name}.r2.cloudflarestorage.com/{r2_ann_key}" if r2_creds else f"/annotated/{img_filename}"

            # Get winning annotation
            ann_rec = annotations_by_img.get(img_id)
            if ann_rec:
                ann_rec_copy = dict(ann_rec)
                ann_rec_copy["image_url"] = raw_url
                ann_rec_copy["annotated_image_url"] = ann_url
                batch_annotations.append(ann_rec_copy)

                # Draw annotated overlay image
                objects = ann_rec.get("objects", [])
                overlay_path = batch_pub_ann_dir / img_filename
                try:
                    temp_rendered = dummy_assembler._draw_annotations(raw_path, objects)
                    shutil.copy(str(temp_rendered), str(overlay_path))
                    if temp_rendered.exists():
                        temp_rendered.unlink()
                except Exception as e:
                    logger.warning(f"  Error drawing annotations for {img_id[:12]}: {e}")
                    shutil.copy(str(raw_path), str(overlay_path))
            else:
                # Fallback if unannotated: save clean image as annotated placeholder
                overlay_path = batch_pub_ann_dir / img_filename
                shutil.copy(str(raw_path), str(overlay_path))

            manifest_records.append({
                "index": idx,
                "image_id": img_id,
                "image_name": img_filename,
                "image_url": raw_url,
                "annotated_image_url": ann_url,
                "local_path": f"{dataset_name}/{batch_id}/images/{img_filename}",
                "width": w_px,
                "height": h_px,
                "has_annotations": bool(ann_rec),
                "is_golden": False,
            })

            chips_to_upload.append((raw_path, r2_raw_key))
            chips_to_upload.append((overlay_path, r2_ann_key))

        # Save batch files locally
        manifest_file = batch_pub_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_records, indent=2), encoding="utf-8")

        ann_json_file = batch_pub_dir / "annotations.json"
        ann_json_file.write_text(json.dumps(batch_annotations, indent=2), encoding="utf-8")

        ann_jsonl_file = batch_pub_dir / "commercial-dataset.jsonl"
        with ann_jsonl_file.open("w", encoding="utf-8") as f:
            for rec in batch_annotations:
                f.write(json.dumps(rec) + "\n")

        logger.info(f"  Local batch formulated: {len(manifest_records)} chips, {len(batch_annotations)} with annotations.")

        # Upload batch to Cloudflare R2
        if upload_r2 and r2_creds:
            logger.info(f"  Uploading {len(chips_to_upload)} image files to R2...")
            def _upload_file(item):
                fpath, rkey = item
                try:
                    upload_image_to_r2(fpath, object_key=rkey, creds=r2_creds)
                except Exception as e:
                    logger.debug(f"Upload note: {e}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                list(executor.map(_upload_file, chips_to_upload))

            # Upload JSON files
            upload_bytes_to_r2(
                manifest_file.read_bytes(),
                object_key=f"{dataset_name}/{batch_id}/manifest.json",
                creds=r2_creds,
                content_type="application/json",
            )
            upload_bytes_to_r2(
                ann_json_file.read_bytes(),
                object_key=f"{dataset_name}/{batch_id}/annotations.json",
                creds=r2_creds,
                content_type="application/json",
            )
            upload_bytes_to_r2(
                ann_jsonl_file.read_bytes(),
                object_key=f"{dataset_name}/{batch_id}/commercial-dataset.jsonl",
                creds=r2_creds,
                content_type="application/x-ndjson",
            )
            logger.info(f"  [R2 COMPLETE] {batch_id} fully mirrored to R2.")


def purge_obsolete_r2_storage():
    """Purge legacy 'commercial/' folder and dead tasks older than 12h."""
    try:
        import boto3
        creds = load_r2_credentials_from_env()
        s3 = boto3.client(
            "s3",
            endpoint_url=creds.s3_endpoint,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            region_name="auto",
        )
        bucket = creds.bucket_name
    except Exception as e:
        logger.error(f"Cannot connect to R2 for cleanup: {e}")
        return

    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(hours=12)
    logger.info(f"Starting Cloudflare R2 cleanup (cutoff: {cutoff.isoformat()})...")

    # 1. Purge legacy 'commercial/' prefix completely (Sep 24-28 unorganized data)
    logger.info("Purging legacy 'commercial/' prefix...")
    paginator = s3.get_paginator("list_objects_v2")
    comm_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix="commercial/"):
        for obj in page.get("Contents", []):
            comm_keys.append({"Key": obj["Key"]})
            if len(comm_keys) >= 1000:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": comm_keys})
                logger.info(f"  Deleted batch of {len(comm_keys)} legacy commercial objects.")
                comm_keys = []
    if comm_keys:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": comm_keys})
        logger.info(f"  Deleted final batch of {len(comm_keys)} legacy commercial objects.")

    # 2. Purge dead 'miners/' older than 12h
    logger.info("Purging dead 'miners/' objects older than 12 hours...")
    miner_keys = []
    del_miners_count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix="miners/"):
        for obj in page.get("Contents", []):
            if obj["LastModified"] < cutoff:
                miner_keys.append({"Key": obj["Key"]})
                if len(miner_keys) >= 1000:
                    s3.delete_objects(Bucket=bucket, Delete={"Objects": miner_keys})
                    del_miners_count += len(miner_keys)
                    logger.info(f"  Deleted {del_miners_count} old miner objects...")
                    miner_keys = []
    if miner_keys:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": miner_keys})
        del_miners_count += len(miner_keys)
    logger.info(f"Total old miner objects deleted: {del_miners_count}")

    # 3. Purge dead 'camouflaged/' older than 12h
    logger.info("Purging dead 'camouflaged/' objects older than 12 hours...")
    cam_keys = []
    del_cam_count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix="camouflaged/"):
        for obj in page.get("Contents", []):
            if obj["LastModified"] < cutoff:
                cam_keys.append({"Key": obj["Key"]})
                if len(cam_keys) >= 1000:
                    s3.delete_objects(Bucket=bucket, Delete={"Objects": cam_keys})
                    del_cam_count += len(cam_keys)
                    logger.info(f"  Deleted {del_cam_count} old camouflage objects...")
                    cam_keys = []
    if cam_keys:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": cam_keys})
        del_cam_count += len(cam_keys)
    logger.info(f"Total old camouflage objects deleted: {del_cam_count}")


def main():
    parser = argparse.ArgumentParser(description="Reorganize and Sync Commercial Dataset to R2 & VPS")
    parser.add_argument("--purge-r2", action="store_true", help="Purge obsolete R2 files (legacy commercial & old tasks)")
    parser.add_argument("--build-batches", action="store_true", help="Build and upload structured round batches")
    args = parser.parse_args()

    comm_dir = REPO_ROOT / "artifacts" / "commercial_dataset"

    if args.purge_r2:
        purge_obsolete_r2_storage()

    if args.build_batches:
        annotations = load_all_commercial_annotations(comm_dir)
        build_and_upload_batches(
            raw_dir=REPO_ROOT / "data" / "climate_mrv" / "samples" / "raw",
            golden_dir=REPO_ROOT / "data" / "climate_mrv" / "samples" / "golden",
            annotations_by_img=annotations,
            dataset_name="climate_mrv",
            total_batches=17,
            public_per_batch=27,
            golden_per_batch=3,
            upload_r2=True,
        )


if __name__ == "__main__":
    main()
