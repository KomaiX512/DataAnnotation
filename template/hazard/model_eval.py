"""
Model accuracy evaluation for the dual-flywheel subnet.

The validator downloads each miner's fine-tuned checkpoint from R2,
runs YOLO inference against the validator-only Golden Set and a
cross-domain Benchmark (Roboflow 100 / cppe-5 family), and produces a
single ``model_accuracy_score`` in [0, 1] used as ``model_accuracy_score``
in the final on-chain weight formula.

Same metric breakdown as annotation fidelity:

  golden_score = 0.35 * IoU + 0.25 * class+severity match + 0.25 * reasoning
                 + 0.15 * confidence calibration

Plus a benchmark IoU on the held-out cross-domain split. The combined
score weights Golden 70% and Benchmark 30% (Golden is the operating
distribution; benchmark detects overfitting).
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import bittensor as bt

from template.hazard.annotation_eval import (
    cosine_similarity,
    iou_xyxy,
    _embed_text,
)
from template.hazard.image_corpus import (
    BenchmarkImage,
    GoldenAnnotation,
    GoldenImage,
    ImageCorpus,
    _reasoning_for_label,
    _severity_for_label,
)
from template.hazard.r2_storage import download_checkpoint_from_r2
from template.protocol import R2AccessCredentials


@dataclass(frozen=True)
class ModelAccuracyComponents:
    golden_iou: float
    golden_class_severity: float
    golden_reasoning: float
    golden_confidence: float
    golden_score: float
    benchmark_iou: float
    benchmark_score: float
    overall_score: float
    images_scored: int


@dataclass
class ModelAccuracyEvaluator:
    """Real validator-side checkpoint evaluator for the dual-flywheel."""

    iou_weight: float = 0.35
    class_severity_weight: float = 0.25
    reasoning_weight: float = 0.25
    confidence_weight: float = 0.15
    golden_blend: float = 0.7
    benchmark_blend: float = 0.3
    download_root: Path = Path("/tmp/flywheel_models")

    def evaluate(
        self,
        *,
        corpus: ImageCorpus,
        candidate_model_uri: str,
        candidate_model_hash: str,
        miner_r2_credentials: Optional[R2AccessCredentials],
    ) -> ModelAccuracyComponents:
        local_path = self._resolve_checkpoint(
            candidate_model_uri, candidate_model_hash, miner_r2_credentials
        )
        golden = corpus.golden_images()
        benchmark = corpus.benchmark_images()
        if not golden:
            raise RuntimeError("Golden Set is empty; model accuracy cannot be evaluated.")
        if not benchmark:
            raise RuntimeError("Benchmark set is empty; model accuracy cannot be evaluated.")

        model = _load_yolo(local_path)

        gi, gcs, gr, gc, gn = self._score_against_golden(model, golden)
        bi, bn = self._score_against_benchmark(model, benchmark)
        if gn == 0:
            raise RuntimeError(
                f"No Golden samples produced predictions for checkpoint {candidate_model_hash}."
            )
        if bn == 0:
            raise RuntimeError(
                f"No Benchmark samples produced predictions for checkpoint {candidate_model_hash}."
            )

        golden_score = (
            self.iou_weight * gi
            + self.class_severity_weight * gcs
            + self.reasoning_weight * gr
            + self.confidence_weight * gc
        )
        benchmark_score = bi
        overall = (
            self.golden_blend * golden_score + self.benchmark_blend * benchmark_score
        )
        overall = float(max(0.0, min(1.0, overall)))

        bt.logging.info(
            f"event=model_accuracy_evaluated candidate_hash={candidate_model_hash} "
            f"golden_iou={gi:.4f} class_sev={gcs:.4f} reasoning={gr:.4f} confidence={gc:.4f} "
            f"benchmark_iou={bi:.4f} overall={overall:.4f}"
        )
        return ModelAccuracyComponents(
            golden_iou=float(gi),
            golden_class_severity=float(gcs),
            golden_reasoning=float(gr),
            golden_confidence=float(gc),
            golden_score=float(max(0.0, min(1.0, golden_score))),
            benchmark_iou=float(bi),
            benchmark_score=float(max(0.0, min(1.0, benchmark_score))),
            overall_score=overall,
            images_scored=int(gn + bn),
        )

    # ------------------------------------------------------------------ Internals
    def _resolve_checkpoint(
        self,
        uri: str,
        candidate_hash: str,
        creds: Optional[R2AccessCredentials],
    ) -> Path:
        parsed = urlparse(uri or "")
        if parsed.scheme == "file":
            path = Path(parsed.path)
            if not path.exists():
                raise FileNotFoundError(f"Candidate checkpoint missing: {path}")
            if path.is_dir():
                resolved = _find_best_checkpoint(path)
                if resolved is None:
                    raise FileNotFoundError(f"No .pt found under {path}")
                return resolved
            return path
        if parsed.scheme == "r2":
            if creds is None:
                raise ValueError(
                    "Candidate model URI uses r2:// but no miner_r2_credentials handshake was supplied."
                )
            self.download_root.mkdir(parents=True, exist_ok=True)
            if uri.endswith("/"):
                # Prefix-style: the validator must enumerate files. Pull best.pt or first .pt.
                key_prefix = parsed.path.lstrip("/")
                bucket = parsed.netloc
                target = self.download_root / f"{candidate_hash}_best.pt"
                _download_best_pt_from_r2_prefix(
                    bucket=bucket,
                    prefix=key_prefix,
                    creds=creds,
                    target=target,
                )
                return target
            target = self.download_root / f"candidate-{candidate_hash}.pt"
            return download_checkpoint_from_r2(uri, creds=creds, target_path=target)
        raise ValueError(f"Unsupported candidate model URI scheme: {uri}")

    def _score_against_golden(
        self, model, golden: Sequence[GoldenImage]
    ) -> Tuple[float, float, float, float, int]:
        from PIL import Image  # type: ignore

        total_iou = 0.0
        total_class_sev = 0.0
        total_reasoning = 0.0
        total_confidence = 0.0
        n = 0
        for image in golden:
            with Image.open(image.image_path) as pil_img:
                pil = pil_img.convert("RGB")
            preds = model.predict(source=pil, verbose=False)
            if not preds:
                continue
            pred_boxes, pred_classes, pred_confs = _extract_yolo_predictions(
                preds[0], model.names
            )
            iou_avg, class_sev_avg, reasoning_avg, confidence_avg = _score_one_labeled_image(
                gt_annotations=image.annotations,
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                pred_confs=pred_confs,
                iou_weight=self.iou_weight,
                class_severity_weight=self.class_severity_weight,
                reasoning_weight=self.reasoning_weight,
                confidence_weight=self.confidence_weight,
            )
            total_iou += iou_avg
            total_class_sev += class_sev_avg
            total_reasoning += reasoning_avg
            total_confidence += confidence_avg
            n += 1
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0, 0
        return (
            total_iou / n,
            total_class_sev / n,
            total_reasoning / n,
            total_confidence / n,
            n,
        )

    def _score_against_benchmark(
        self, model, benchmark: Sequence[BenchmarkImage]
    ) -> Tuple[float, int]:
        from PIL import Image  # type: ignore

        total_iou = 0.0
        n = 0
        for image in benchmark:
            with Image.open(image.image_path) as pil_img:
                pil = pil_img.convert("RGB")
            preds = model.predict(source=pil, verbose=False)
            if not preds:
                continue
            pred_boxes, _, _ = _extract_yolo_predictions(preds[0], model.names)
            best_iou_per_gt = []
            for gt in image.annotations:
                best = 0.0
                for box in pred_boxes:
                    iou = iou_xyxy(gt.bounding_box, box)
                    if iou > best:
                        best = iou
                best_iou_per_gt.append(best)
            if best_iou_per_gt:
                total_iou += sum(best_iou_per_gt) / len(best_iou_per_gt)
                n += 1
        if n == 0:
            return 0.0, 0
        return total_iou / n, n


def _load_yolo(path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ultralytics is required for dual-flywheel model accuracy evaluation."
        ) from exc
    return YOLO(str(path))


def _extract_yolo_predictions(result, names) -> Tuple[List[List[float]], List[str], List[float]]:
    if result.boxes is None or len(result.boxes) == 0:
        return [], [], []
    xyxy = result.boxes.xyxy.cpu().tolist()
    clss = result.boxes.cls.cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()
    classes: List[str] = []
    for cls_idx in clss:
        try:
            cls_int = int(cls_idx)
        except Exception:
            cls_int = 0
        raw_name = names.get(cls_int, str(cls_int)) if isinstance(names, dict) else (
            names[cls_int] if 0 <= cls_int < len(names) else str(cls_int)
        )
        classes.append(str(raw_name).lower().replace(" ", "_"))
    boxes = [[float(v) for v in box] for box in xyxy]
    confs = [float(c) for c in confs]
    return boxes, classes, confs


def _score_one_labeled_image(
    *,
    gt_annotations: Sequence[GoldenAnnotation],
    pred_boxes: Sequence[Sequence[float]],
    pred_classes: Sequence[str],
    pred_confs: Sequence[float],
    iou_weight: float,
    class_severity_weight: float,
    reasoning_weight: float,
    confidence_weight: float,
) -> Tuple[float, float, float, float]:
    if not gt_annotations:
        return 0.0, 0.0, 0.0, 0.0

    used_pred_idx: set[int] = set()
    iou_sum = 0.0
    class_sev_sum = 0.0
    reasoning_sum = 0.0
    confidence_sum = 0.0
    for gt in gt_annotations:
        best_idx = -1
        best_iou = 0.0
        for idx, box in enumerate(pred_boxes):
            if idx in used_pred_idx:
                continue
            iou = iou_xyxy(box, gt.bounding_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx < 0:
            continue
        used_pred_idx.add(best_idx)
        iou_sum += best_iou

        pred_class = pred_classes[best_idx]
        pred_severity = _severity_for_label(pred_class)
        class_match = 1.0 if pred_class == gt.hazard_class else 0.0
        sev_match = 1.0 if pred_severity == gt.severity else 0.0
        class_sev_sum += 0.6 * class_match + 0.4 * sev_match

        pred_reasoning = _reasoning_for_label(pred_class, pred_severity)
        reasoning_sum += cosine_similarity(_embed_text(pred_reasoning), _embed_text(gt.reasoning))

        c = max(0.0, min(1.0, float(pred_confs[best_idx])))
        confidence_sum += c * best_iou + (1.0 - c) * (1.0 - best_iou)

    n = max(1, len(gt_annotations))
    return iou_sum / n, class_sev_sum / n, reasoning_sum / n, confidence_sum / n


def _find_best_checkpoint(directory: Path) -> Optional[Path]:
    """Locate the best YOLO weight file under a directory."""
    pt_files = sorted(directory.rglob("*.pt"))
    if not pt_files:
        return None
    for candidate in pt_files:
        if candidate.name.lower() == "best.pt":
            return candidate
    return pt_files[0]


def _download_best_pt_from_r2_prefix(
    *,
    bucket: str,
    prefix: str,
    creds: R2AccessCredentials,
    target: Path,
) -> Path:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise ImportError("boto3 is required for R2 prefix downloads.") from exc

    client = boto3.client(
        "s3",
        endpoint_url=creds.s3_endpoint,
        aws_access_key_id=creds.access_key_id,
        aws_secret_access_key=creds.secret_access_key,
        region_name="auto",
    )
    continuation_token: Optional[str] = None
    candidates: List[str] = []
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token is not None:
            kwargs["ContinuationToken"] = continuation_token
        result = client.list_objects_v2(**kwargs)
        for item in result.get("Contents", []):
            key = item["Key"]
            if key.lower().endswith(".pt"):
                candidates.append(key)
        if not result.get("IsTruncated"):
            break
        continuation_token = result.get("NextContinuationToken")

    if not candidates:
        raise FileNotFoundError(
            f"No .pt files found under r2://{bucket}/{prefix}; cannot evaluate model."
        )
    chosen = next((k for k in candidates if k.lower().endswith("/best.pt")), candidates[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, chosen, str(target))
    return target
