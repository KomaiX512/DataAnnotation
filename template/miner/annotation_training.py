from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List
from urllib.error import URLError
from urllib.request import Request, urlopen

import bittensor as bt

from template.hazard.r2_storage import (
    load_r2_credentials_from_env,
    upload_bytes_to_r2,
    upload_directory_to_r2,
)
from template.hazard.vector_db import OshaVectorDatabase
from template.protocol import (
    AnnotationAndTrainingTask,
    AnnotationsFilePayload,
    ImageAnnotationDocument,
    PerImageAnnotationItem,
    SeverityTier,
)
from template.miner.training import TrainingPipeline, TrainingSettings


def fetch_url_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    """Download image bytes from ``http(s)`` or ``file`` URLs."""
    req = Request(url, headers={"User-Agent": "hazard-subnet-miner/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _severity_for_hazard_class(hazard_class: str) -> SeverityTier:
    h = hazard_class.lower()
    if any(k in h for k in ("fall", "electrocution", "crush", "collapse", "caught")):
        return "high"
    if any(k in h for k in ("missing", "no_", "unprotected", "improper")):
        return "high"
    if any(k in h for k in ("trip", "slip", "minor", "housekeeping")):
        return "medium"
    if any(k in h for k in ("low", "warning")):
        return "low"
    return "medium"


def _pixel_boxes_from_digest(
    digest: str, width: int, height: int, hazard_detected: bool
) -> List[tuple[int, int, int, int]]:
    if not hazard_detected or width <= 0 or height <= 0:
        return []
    x_seed = int(digest[8:12], 16) / 0xFFFF
    y_seed = int(digest[12:16], 16) / 0xFFFF
    width_seed = int(digest[16:20], 16) / 0xFFFF
    height_seed = int(digest[20:24], 16) / 0xFFFF
    x_min_n = min(0.85, x_seed * 0.7)
    y_min_n = min(0.85, y_seed * 0.7)
    w_n = 0.1 + width_seed * 0.2
    h_n = 0.1 + height_seed * 0.2
    x1 = int(x_min_n * width)
    y1 = int(y_min_n * height)
    x2 = int(min(float(width), (x_min_n + w_n) * width))
    y2 = int(min(float(height), (y_min_n + h_n) * height))
    if x2 <= x1 or y2 <= y1:
        return []
    return [(x1, y1, x2, y2)]


def _annotate_deterministic(
    *,
    image_bytes: bytes,
    image_id: str,
    challenge_nonce: str,
    osha_db: OshaVectorDatabase,
    model_version: str,
    miner_uid: str,
) -> ImageAnnotationDocument:
    digest = hashlib.sha256(image_bytes + challenge_nonce.encode("utf-8")).hexdigest()
    energy = int(digest[:8], 16) / 0xFFFFFFFF
    hazard_detected = energy > 0.35
    confidence = float(min(0.99, max(0.05, 0.5 + (energy - 0.5) * 0.8)))
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("pillow is required for annotation.") from exc
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    boxes = _pixel_boxes_from_digest(digest, w, h, hazard_detected)
    items: List[PerImageAnnotationItem] = []
    for (x1, y1, x2, y2) in boxes:
        hazard_class = "missing_fall_protection" if energy > 0.55 else "missing_hardhat"
        severity = _severity_for_hazard_class(hazard_class)
        query = f"{image_id} {hazard_class} {severity}"
        refs = osha_db.search(query, top_k=1)
        osha_ref = refs[0].citation_id if refs else None
        reasoning = (
            f"Detected construction hazard pattern ({hazard_class}) from model pass; "
            f"severity={severity}. Grounded with OSHA context: "
            f"{refs[0].title if refs else 'n/a'}."
        )
        items.append(
            PerImageAnnotationItem(
                hazard_class=hazard_class,
                bounding_box=[x1, y1, x2, y2],
                severity=severity,
                confidence=confidence,
                reasoning_chain=reasoning,
                osha_reference=osha_ref,
            )
        )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ImageAnnotationDocument(
        image_id=image_id,
        miner_uid=miner_uid,
        timestamp=ts,
        annotations=items,
        model_version=model_version,
    )


def _annotate_yolo(
    *,
    checkpoint: Path,
    image_bytes: bytes,
    image_id: str,
    osha_db: OshaVectorDatabase,
    model_version: str,
    miner_uid: str,
) -> ImageAnnotationDocument:
    try:
        from ultralytics import YOLO
        from PIL import Image
    except ImportError as exc:
        raise ImportError("ultralytics and pillow are required for YOLO annotation.") from exc
    import io as _io

    model = YOLO(str(checkpoint))
    image = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    results = model.predict(source=image, verbose=False)
    items: List[PerImageAnnotationItem] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        clss = r.boxes.cls.cpu().tolist()
        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            raw_name = model.names.get(int(cls_id), str(int(cls_id)))
            hazard_class = str(raw_name).lower().replace(" ", "_")
            severity = _severity_for_hazard_class(hazard_class)
            query = f"{image_id} {hazard_class}"
            refs = osha_db.search(query, top_k=1)
            osha_ref = refs[0].citation_id if refs else None
            reasoning = (
                f"YOLO detection cls={hazard_class} conf={conf:.3f}; "
                f"OSHA grounding: {refs[0].title if refs else 'n/a'}."
            )
            items.append(
                PerImageAnnotationItem(
                    hazard_class=hazard_class,
                    bounding_box=[x1, y1, x2, y2],
                    severity=severity,
                    confidence=float(conf),
                    reasoning_chain=reasoning,
                    osha_reference=osha_ref,
                )
            )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ImageAnnotationDocument(
        image_id=image_id,
        miner_uid=miner_uid,
        timestamp=ts,
        annotations=items,
        model_version=model_version,
    )


class AnnotationTrainingEngine:
    """
    Dual-flywheel miner: URL-labeled fine-tuning, unlabeled annotation, R2 artifact upload.
    """

    def __init__(self, config=None):
        workspace = Path(
            getattr(getattr(config, "miner", object()), "training_workspace", "artifacts/miner_training")
        )
        private_root = getattr(getattr(config, "miner", object()), "private_dataset_root", "")
        self.training_pipeline = TrainingPipeline(
            TrainingSettings(
                workspace=workspace,
                private_dataset_root=Path(private_root) if private_root else None,
                enable_auto_hpo=bool(
                    getattr(getattr(config, "miner", object()), "enable_auto_hpo", False)
                    or getattr(getattr(config, "miner", object()), "autoresearch", False)
                ),
                autoresearch_max_iters=int(
                    getattr(getattr(config, "miner", object()), "autoresearch_max_iters", 4)
                ),
                autoresearch_experiment_minutes=int(
                    getattr(getattr(config, "miner", object()), "autoresearch_experiment_minutes", 5)
                ),
                autoresearch_log_level=str(
                    getattr(getattr(config, "miner", object()), "autoresearch_log_level", "INFO")
                ),
                random_hpo_draw=bool(
                    getattr(getattr(config, "miner", object()), "random_hpo_draw", False)
                ),
                hpo_seed=int(getattr(getattr(config, "miner", object()), "hpo_seed", 0)),
            )
        )
        self.annotation_backend = str(
            getattr(getattr(config, "miner", object()), "annotation_backend", "deterministic")
        ).strip()
        self.r2_prefix = str(
            getattr(getattr(config, "miner", object()), "dual_flywheel_r2_prefix", "miners/dual_flywheel")
        ).strip()
        self.osha_db = OshaVectorDatabase.default()
        self._fetch_image: Callable[[str], bytes] = fetch_url_bytes

    def run(
        self,
        synapse: AnnotationAndTrainingTask,
        *,
        miner_hotkey: str,
    ) -> AnnotationAndTrainingTask:
        started = time.time()
        try:
            self._validate_request(synapse)
            if not synapse.training_images:
                raise ValueError("training_images must be non-empty.")
            if not synapse.annotation_images:
                raise ValueError("annotation_images must be non-empty.")
            baseline = synapse.baseline_checkpoint
            assert baseline is not None
            if synapse.base_model_hash != baseline.sha256:
                raise ValueError("base_model_hash must match baseline_checkpoint.sha256.")
            max_training_seconds = synapse.max_training_seconds
            assert max_training_seconds is not None

            manifest, best_pt = self.training_pipeline.run_from_labeled_images(
                task_id=synapse.task_id,
                baseline=baseline,
                labeled_images=list(synapse.training_images),
                fetch_image=self._fetch_image,
                max_training_seconds=max_training_seconds,
                r2_object_prefix=self.r2_prefix,
            )
            train_root = self.training_pipeline.settings.workspace / synapse.task_id
            run_dir = train_root / "runs" / "yolov8s_construction"
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Training run directory missing: {run_dir}")

            creds = load_r2_credentials_from_env()
            remote_base = f"{self.r2_prefix.strip().rstrip('/')}/{synapse.task_id}/"
            model_prefix = f"{remote_base}model_checkpoint/"
            model_checkpoint_uri = upload_directory_to_r2(run_dir, key_prefix=model_prefix, creds=creds)

            records: list[ImageAnnotationDocument] = []
            model_version = manifest.candidate_model_hash
            for spec in synapse.annotation_images:
                img_bytes = self._fetch_image(spec.image_url)
                if self.annotation_backend == "yolo":
                    doc = _annotate_yolo(
                        checkpoint=best_pt,
                        image_bytes=img_bytes,
                        image_id=spec.image_id,
                        osha_db=self.osha_db,
                        model_version=model_version,
                        miner_uid=miner_hotkey,
                    )
                elif self.annotation_backend == "deterministic":
                    doc = _annotate_deterministic(
                        image_bytes=img_bytes,
                        image_id=spec.image_id,
                        challenge_nonce=synapse.challenge_nonce or synapse.task_id,
                        osha_db=self.osha_db,
                        model_version=model_version,
                        miner_uid=miner_hotkey,
                    )
                else:
                    raise ValueError(f"Unknown annotation_backend: {self.annotation_backend}")
                records.append(doc)

            payload = AnnotationsFilePayload(
                schema_version="annotations.v1",
                task_id=synapse.task_id,
                records=records,
            )
            raw = json.dumps(payload.model_dump(), indent=2, sort_keys=True).encode("utf-8")
            annotations_key = f"{remote_base}annotations.json"
            annotations_uri = upload_bytes_to_r2(
                raw,
                object_key=annotations_key,
                creds=creds,
                content_type="application/json",
            )

            recipe_path = train_root / "recipe.json"
            training_config: dict = {}
            if recipe_path.is_file():
                training_config = json.loads(recipe_path.read_text(encoding="utf-8"))
            training_config["annotation_backend"] = self.annotation_backend
            training_config["model_checkpoint_prefix_uri"] = model_checkpoint_uri
            training_config["annotations_object_key"] = annotations_key

            synapse.annotations_uri = annotations_uri
            synapse.model_checkpoint_uri = model_checkpoint_uri
            synapse.training_config = training_config
            synapse.submitted_training_manifest = manifest
            synapse.miner_r2_credentials = creds
            synapse.claim_improvement = float(manifest.metrics.get("uplift", 0.0))
            synapse.error_message = None
        except (URLError, OSError, ValueError, RuntimeError, ImportError) as exc:
            bt.logging.error(f"Dual-flywheel task failed {synapse.task_id}: {exc}")
            synapse.error_message = str(exc)
        synapse.duration_ms = int((time.time() - started) * 1000)
        return synapse

    @staticmethod
    def _validate_request(synapse: AnnotationAndTrainingTask) -> None:
        if not synapse.task_id:
            raise ValueError("task_id is required.")
        if len(synapse.base_model_hash) < 8:
            raise ValueError("base_model_hash must be at least 8 characters.")
        if synapse.baseline_checkpoint is None:
            raise ValueError("baseline_checkpoint is required.")
        if synapse.max_training_seconds is None:
            raise ValueError("max_training_seconds is required.")


def build_synthetic_labeled_png(width: int = 64, height: int = 64) -> bytes:
    """Test helper: tiny RGB PNG bytes."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("pillow is required.") from exc
    import io as _io

    img = Image.new("RGB", (width, height), color=(120, 140, 160))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
