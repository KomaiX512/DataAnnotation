from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse
import os
import bittensor as bt

from template.hazard.dataset import HazardDatasetManager
from template.hazard.r2_storage import download_checkpoint_from_r2
from template.protocol import R2AccessCredentials, TrainingManifest

DATASET_REPO_ID = "cppe-5"


@dataclass(frozen=True)
class GoldenEvaluation:
    golden_score: float
    severity_score: float
    localization_score: float
    reasoning_score: float


class GoldenSetEvaluator:
    """
    Validator-side real checkpoint evaluator.

    One-path behavior:
    - Runs true YOLO inference from the miner checkpoint.
    - Evaluates hidden golden 20 percent split from the shared dataset.
    - Produces mAP/severity/reasoning style scores used by the reward pipeline.
    """

    def __init__(self, dataset_manager: HazardDatasetManager):
        self.dataset_manager = dataset_manager

    def evaluate(
        self,
        manifest: TrainingManifest,
        miner_r2_credentials: R2AccessCredentials | None,
    ) -> GoldenEvaluation:
        model_path = self._resolve_model_path(
            manifest.candidate_model_uri,
            manifest.candidate_model_hash,
            miner_r2_credentials,
        )
        samples = self._load_golden_samples()
        map_score, severity_score, reasoning_score = self._run_model_scoring(model_path, samples)
        golden_score = (
            0.6 * map_score
            + 0.25 * severity_score
            + 0.15 * reasoning_score
        )
        bt.logging.info(
            f"event=evaluator_golden_score_payload model_uri={manifest.candidate_model_uri} "
            f"candidate_hash={manifest.candidate_model_hash} golden_score={golden_score:.6f} "
            f"localization={map_score:.6f} severity={severity_score:.6f} reasoning={reasoning_score:.6f}"
        )
        return GoldenEvaluation(
            golden_score=golden_score,
            severity_score=severity_score,
            localization_score=map_score,
            reasoning_score=reasoning_score,
        )

    @staticmethod
    def _resolve_model_path(
        model_uri: str,
        candidate_hash: str,
        miner_r2_credentials: R2AccessCredentials | None,
    ) -> Path:
        parsed = urlparse(model_uri)
        if parsed.scheme == "file":
            path = Path(parsed.path)
            if not path.exists():
                raise FileNotFoundError(f"Candidate checkpoint does not exist: {path}")
            return path
        if parsed.scheme == "r2":
            if miner_r2_credentials is None:
                raise ValueError("R2 candidate model requires miner_r2_credentials handshake.")
            target = Path("/tmp") / f"candidate-{candidate_hash}.pt"
            return download_checkpoint_from_r2(
                model_uri,
                creds=miner_r2_credentials,
                target_path=target,
            )
        raise ValueError(f"Unsupported model URI scheme for evaluation: {model_uri}")

    @staticmethod
    def _load_golden_samples():
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets is required for golden set evaluation.") from exc
        ds = load_dataset(DATASET_REPO_ID)
        if "train" not in ds:
            raise ValueError(f"Dataset {DATASET_REPO_ID} does not contain a train split.")
        split = ds["train"].train_test_split(test_size=0.2, seed=42)["test"]
        if len(split) == 0:
            raise ValueError("Golden split is empty.")
        max_samples = int(os.getenv("GOLDEN_MAX_SAMPLES", "64"))
        if max_samples > 0:
            split = split.select(range(min(len(split), max_samples)))
        return split

    @staticmethod
    def _run_model_scoring(model_path: Path, samples):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for golden set evaluation.") from exc
        model = YOLO(str(model_path))
        total_iou = 0.0
        total_severity = 0.0
        total_reasoning = 0.0
        count = 0
        for sample in samples:
            image = sample.get("image")
            objects = sample.get("objects")
            if image is None or objects is None:
                raise ValueError("Golden sample missing 'image' or 'objects'.")
            preds = model.predict(image, verbose=False)
            if not preds:
                continue
            pred = preds[0]
            gt_bbox = objects.get("bbox", [])
            gt_labels = [str(x) for x in objects.get("category", [])]
            if not gt_bbox:
                continue
            pred_boxes = pred.boxes.xyxy.cpu().tolist() if pred.boxes is not None else []
            pred_cls = pred.boxes.cls.cpu().tolist() if pred.boxes is not None else []
            pred_conf = pred.boxes.conf.cpu().tolist() if pred.boxes is not None else []
            map_iou = GoldenSetEvaluator._max_iou_score(gt_bbox[0], pred_boxes, image.size)
            severity_score = GoldenSetEvaluator._severity_match_score(gt_labels, pred_cls, pred.names)
            reasoning_score = GoldenSetEvaluator._reasoning_consistency(map_iou, pred_conf)
            total_iou += map_iou
            total_severity += severity_score
            total_reasoning += reasoning_score
            count += 1
        if count == 0:
            raise ValueError("No valid golden samples were scored.")
        return total_iou / count, total_severity / count, total_reasoning / count

    @staticmethod
    def _max_iou_score(gt_box_xywh, pred_xyxy, image_size) -> float:
        gx, gy, gw, gh = [float(v) for v in gt_box_xywh]
        gt = [gx, gy, gx + gw, gy + gh]
        best = 0.0
        for p in pred_xyxy:
            iou = GoldenSetEvaluator._iou(gt, p)
            if iou > best:
                best = iou
        return max(0.0, min(1.0, best))

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    @staticmethod
    def _severity_match_score(gt_labels, pred_cls, names) -> float:
        if not gt_labels:
            return 0.0
        pred_label = ""
        if pred_cls:
            idx = int(pred_cls[0])
            pred_label = str(names[idx]) if idx in names else ""
        gt_severity = GoldenSetEvaluator._severity_from_label(gt_labels[0])
        pred_severity = GoldenSetEvaluator._severity_from_label(pred_label)
        return 1.0 if gt_severity == pred_severity else 0.0

    @staticmethod
    def _severity_from_label(label: str) -> str:
        token = label.lower()
        if any(k in token for k in ("fire", "fall", "electrical", "explosion")):
            return "high"
        if any(k in token for k in ("helmet", "vest", "mask")):
            return "medium"
        return "low"

    @staticmethod
    def _reasoning_consistency(map_iou: float, pred_conf: list[float]) -> float:
        conf = max(pred_conf) if pred_conf else 0.0
        return max(0.0, min(1.0, 0.7 * map_iou + 0.3 * conf))

