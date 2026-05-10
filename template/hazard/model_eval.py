"""
Model accuracy evaluation for the dual-flywheel subnet.

The validator downloads each miner's fine-tuned checkpoint via a short-lived
HTTPS URL (R2 presigned GET), then runs YOLO inference against the
validator-only Golden Set and a cross-domain Benchmark.

When ``docker_sandbox_image`` is set, inference runs inside ``docker run
--network none`` so poisoned weights are not executed in the validator process.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
    """Validator-side checkpoint evaluator for the dual-flywheel."""

    iou_weight: float = 0.35
    class_severity_weight: float = 0.25
    reasoning_weight: float = 0.25
    confidence_weight: float = 0.15
    golden_blend: float = 0.7
    benchmark_blend: float = 0.3
    download_root: Path = Path("/tmp/flywheel_models")
    docker_sandbox_image: str = ""

    def evaluate(
        self,
        *,
        corpus: ImageCorpus,
        candidate_model_uri: str,
        candidate_model_hash: str,
    ) -> ModelAccuracyComponents:
        local_path = self._resolve_checkpoint(candidate_model_uri, candidate_model_hash)
        golden = corpus.golden_images()
        benchmark = corpus.benchmark_images()
        if not golden:
            raise RuntimeError("Golden Set is empty; model accuracy cannot be evaluated.")
        if not benchmark:
            raise RuntimeError("Benchmark set is empty; model accuracy cannot be evaluated.")

        if (self.docker_sandbox_image or "").strip():
            gi, gcs, gr, gc, gn, bi, bn = self._evaluate_in_docker(
                local_path=local_path,
                golden=golden,
                benchmark=benchmark,
                candidate_model_hash=candidate_model_hash,
            )
        else:
            model = _load_yolo(local_path)
            g_preds = _predictions_for_golden_inplace(model, golden)
            b_preds = _predictions_for_benchmark_inplace(model, benchmark)
            gi, gcs, gr, gc, gn = self._score_against_golden_from_preds(golden, g_preds)
            bi, bn = self._score_against_benchmark_from_preds(benchmark, b_preds)

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
            f"benchmark_iou={bi:.4f} overall={overall:.4f} sandbox={'docker' if self.docker_sandbox_image else 'inproc'}"
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

    def _evaluate_in_docker(
        self,
        *,
        local_path: Path,
        golden: Sequence[GoldenImage],
        benchmark: Sequence[BenchmarkImage],
        candidate_model_hash: str,
    ) -> Tuple[float, float, float, float, int, float, int]:
        docker_bin = shutil.which("docker")
        if not docker_bin:
            raise RuntimeError(
                "flywheel_model_eval_docker_image is set but ``docker`` was not found on PATH."
            )
        job = self.download_root / f"sandbox-{candidate_model_hash}"
        if job.is_dir():
            shutil.rmtree(job)
        job.mkdir(parents=True)
        images_dir = job / "images"
        images_dir.mkdir()
        shutil.copy2(local_path, job / "model.pt")
        spec_images: list[dict] = []
        for i, image in enumerate(golden):
            ext = image.image_path.suffix or ".png"
            rel = f"images/g{i}{ext}"
            shutil.copy2(image.image_path, job / rel)
            spec_images.append({"key": f"g{i}", "relpath": rel})
        for i, image in enumerate(benchmark):
            ext = image.image_path.suffix or ".png"
            rel = f"images/b{i}{ext}"
            shutil.copy2(image.image_path, job / rel)
            spec_images.append({"key": f"b{i}", "relpath": rel})
        spec = {"model_relpath": "model.pt", "images": spec_images}
        (job / "spec.json").write_text(json.dumps(spec), encoding="utf-8")

        image = self.docker_sandbox_image.strip()
        cmd = [
            docker_bin,
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{job}:/job:rw",
            image,
            "/job",
        ]
        bt.logging.info("event=model_eval_docker_start image=%s job=%s" % (image, job))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "Docker YOLO worker failed (code=%s): stdout=%r stderr=%r"
                % (proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:])
            )
        pred_path = job / "predictions.json"
        if not pred_path.is_file():
            raise RuntimeError("Docker YOLO worker did not write predictions.json")
        raw = json.loads(pred_path.read_text(encoding="utf-8"))
        by_key = raw.get("images", {})

        def _pack(key: str) -> Tuple[List[List[float]], List[str], List[float]]:
            block = by_key.get(key) or {}
            return (
                [list(map(float, row)) for row in block.get("xyxy", [])],
                list(block.get("classes", [])),
                [float(x) for x in block.get("confs", [])],
            )

        g_preds = [_pack(f"g{i}") for i in range(len(golden))]
        b_preds = [_pack(f"b{i}") for i in range(len(benchmark))]
        gi, gcs, gr, gc, gn = self._score_against_golden_from_preds(golden, g_preds)
        bi, bn = self._score_against_benchmark_from_preds(benchmark, b_preds)
        return gi, gcs, gr, gc, gn, bi, bn

    # ------------------------------------------------------------------ Internals
    def _resolve_checkpoint(self, uri: str, candidate_hash: str) -> Path:
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
        if parsed.scheme in ("http", "https"):
            self.download_root.mkdir(parents=True, exist_ok=True)
            target = self.download_root / f"candidate-{candidate_hash}.pt"
            _download_url_to_file(uri, target)
            return target
        raise ValueError(
            f"Unsupported candidate model URI scheme {parsed.scheme!r}; "
            "miners must supply a short-lived https:// presigned GET URL (or file:// for local tests)."
        )

    def _score_against_golden_from_preds(
        self,
        golden: Sequence[GoldenImage],
        golden_preds: Sequence[Tuple[List[List[float]], List[str], List[float]]],
    ) -> Tuple[float, float, float, float, int]:
        total_iou = 0.0
        total_class_sev = 0.0
        total_reasoning = 0.0
        total_confidence = 0.0
        n = 0
        for image, (pred_boxes, pred_classes, pred_confs) in zip(golden, golden_preds):
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

    def _score_against_benchmark_from_preds(
        self,
        benchmark: Sequence[BenchmarkImage],
        bench_preds: Sequence[Tuple[List[List[float]], List[str], List[float]]],
    ) -> Tuple[float, int]:
        total_iou = 0.0
        n = 0
        for image, (pred_boxes, _, _) in zip(benchmark, bench_preds):
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


def _download_url_to_file(url: str, target: Path) -> None:
    req = Request(url, headers={"User-Agent": "hazard-validator-model-eval/1.0"})
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(req, timeout=600) as resp:
        data = resp.read()
    target.write_bytes(data)


def _predictions_for_golden_inplace(model, golden: Sequence[GoldenImage]):
    from PIL import Image  # type: ignore

    out: list[Tuple[List[List[float]], List[str], List[float]]] = []
    for image in golden:
        with Image.open(image.image_path) as pil_img:
            pil = pil_img.convert("RGB")
        preds = model.predict(source=pil, verbose=False)
        if not preds:
            out.append(([], [], []))
            continue
        out.append(_extract_yolo_predictions(preds[0], model.names))
    return out


def _predictions_for_benchmark_inplace(model, benchmark: Sequence[BenchmarkImage]):
    from PIL import Image  # type: ignore

    out: list[Tuple[List[List[float]], List[str], List[float]]] = []
    for image in benchmark:
        with Image.open(image.image_path) as pil_img:
            pil = pil_img.convert("RGB")
        preds = model.predict(source=pil, verbose=False)
        if not preds:
            out.append(([], [], []))
            continue
        out.append(_extract_yolo_predictions(preds[0], model.names))
    return out


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
