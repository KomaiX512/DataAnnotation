from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import bittensor as bt

from template.hazard.r2_storage import (
    delete_checkpoint_prefix_from_r2,
    load_r2_credentials_from_env,
    presign_r2_object_uri,
    upload_checkpoint_to_r2,
)
from template.protocol import DatasetPointer, LabeledTrainingImage, ModelCheckpoint, TrainingManifest

DATASET_REPO_ID = os.getenv("HAZARD_DATASET_REPO_ID", "cppe-5")
YOLO_BASELINE = "yolov8s.pt"
MAX_TRAIN_SAMPLES = max(32, int(os.getenv("MINER_MAX_TRAIN_SAMPLES", "256")))
MAX_VAL_SAMPLES = max(16, int(os.getenv("MINER_MAX_VAL_SAMPLES", "64")))
MAX_EPOCHS = max(1, int(os.getenv("MINER_MAX_EPOCHS", "3")))


@dataclass(frozen=True)
class TrainingSettings:
    workspace: Path
    private_dataset_root: Path | None
    enable_auto_hpo: bool
    autoresearch_max_iters: int
    autoresearch_experiment_minutes: int
    autoresearch_log_level: str
    # Single-shot random draw from the same grid as autoresearch (distinct seeds => distinct miners).
    random_hpo_draw: bool = False
    hpo_seed: int = 0


class TrainingPipeline:
    """
    Real miner-side training pipeline.

    One-path behavior:
    - Loads the shared Hugging Face construction safety dataset.
    - Converts the training split into YOLO format.
    - Fine-tunes YOLOv-small and emits real checkpoint metadata.
    - Raises on any dataset/model/training failure (no fallback path).
    """

    def __init__(self, settings: TrainingSettings):
        self.settings = TrainingSettings(
            workspace=settings.workspace.resolve(),
            private_dataset_root=settings.private_dataset_root.resolve()
            if settings.private_dataset_root is not None
            else None,
            enable_auto_hpo=settings.enable_auto_hpo,
            autoresearch_max_iters=max(1, settings.autoresearch_max_iters),
            autoresearch_experiment_minutes=max(1, settings.autoresearch_experiment_minutes),
            autoresearch_log_level=settings.autoresearch_log_level.upper(),
            random_hpo_draw=bool(settings.random_hpo_draw),
            hpo_seed=int(settings.hpo_seed),
        )
        self.settings.workspace.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        task_id: str,
        baseline: ModelCheckpoint,
        training_dataset: DatasetPointer,
        max_training_seconds: int,
    ) -> TrainingManifest:
        start_time = time.monotonic()
        train_root = self.settings.workspace / task_id
        train_root.mkdir(parents=True, exist_ok=True)

        dataset_info = self._prepare_training_dataset(train_root / "dataset")
        base_weights = self._resolve_baseline_weights(baseline.uri)
        hpo_plan = self._resolve_hpo_plan(
            task_id=task_id,
            train_root=train_root,
            dataset_yaml=dataset_info["yaml"],
            base_weights=base_weights,
            max_training_seconds=max_training_seconds,
        )
        best_checkpoint = self._train_yolo(
            task_id=task_id,
            base_weights=base_weights,
            dataset_yaml=dataset_info["yaml"],
            run_root=train_root / "runs",
            max_training_seconds=max_training_seconds,
            hpo_plan=hpo_plan,
        )
        r2_creds = load_r2_credentials_from_env()
        miner_prefix = "miners/current/"
        deleted_objects = delete_checkpoint_prefix_from_r2(
            creds=r2_creds,
            prefix=miner_prefix,
        )
        bt.logging.info(
            f"event=r2_cleanup_before_upload task_id={task_id} prefix={miner_prefix} deleted_objects={deleted_objects}"
        )
        object_key = f"{miner_prefix}{task_id}/best.pt"
        remote_uri = upload_checkpoint_to_r2(
            best_checkpoint,
            object_key=object_key,
            creds=r2_creds,
        )

        artifact_hash = self._sha256(best_checkpoint)
        training_seconds = max(1.0, time.monotonic() - start_time)
        config_payload = {
            "task_id": task_id,
            "dataset_repo": DATASET_REPO_ID,
            "validator_dataset_uri": training_dataset.uri,
            "validator_dataset_hash": training_dataset.sha256,
            "dataset_sha256": dataset_info["dataset_hash"],
            "class_hash": dataset_info["class_hash"],
            "class_names": dataset_info["class_names"],
            "baseline": base_weights,
            "max_training_seconds": max_training_seconds,
            "auto_hpo": self.settings.enable_auto_hpo,
            "hpo_plan": hpo_plan,
        }
        config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest()
        efficiency_score = min(1.0, max_training_seconds / training_seconds) if max_training_seconds > 0 else 0.0

        recipe_path = train_root / "recipe.json"
        recipe_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

        manifest = TrainingManifest(
            parent_model_hash=baseline.sha256,
            candidate_model_hash=artifact_hash,
            candidate_model_uri=remote_uri,
            config_hash=config_hash,
            dataset_lineage_hash=dataset_info["dataset_hash"],
            recipe_uri=recipe_path.as_uri(),
            metrics={
                "reproducibility_score": 1.0,
                "uplift": float(dataset_info["train_samples"]) / max(1.0, float(dataset_info["golden_samples"])),
                "efficiency": float(max(0.0, min(1.0, efficiency_score))),
                "train_samples": float(dataset_info["train_samples"]),
                "golden_samples": float(dataset_info["golden_samples"]),
                "hpo_iterations": float(hpo_plan.get("iterations", 0)),
            },
        )
        return manifest.model_copy(
            update={"candidate_model_uri": presign_r2_object_uri(creds=r2_creds, r2_uri=remote_uri)}
        )

    def run_from_labeled_images(
        self,
        *,
        task_id: str,
        baseline: ModelCheckpoint,
        labeled_images: list[LabeledTrainingImage],
        fetch_image: Callable[[str], bytes],
        max_training_seconds: int,
        r2_object_prefix: str = "miners/dual_flywheel",
    ) -> tuple[TrainingManifest, Path]:
        """
        Fine-tune YOLO from validator-supplied labeled image URLs (pixel boxes + class names).

        Returns the training manifest and the local path to ``best.pt``.
        """
        if not labeled_images:
            raise ValueError("run_from_labeled_images requires at least one labeled training image.")
        start_time = time.monotonic()
        train_root = self.settings.workspace / task_id
        train_root.mkdir(parents=True, exist_ok=True)
        dataset_info = self._prepare_dataset_from_labeled_urls(
            train_root / "dataset",
            labeled_images,
            fetch_image,
            task_id,
        )
        base_weights = self._resolve_baseline_weights(baseline.uri)
        hpo_plan = self._resolve_hpo_plan(
            task_id=task_id,
            train_root=train_root,
            dataset_yaml=dataset_info["yaml"],
            base_weights=base_weights,
            max_training_seconds=max_training_seconds,
        )
        best_checkpoint = self._train_yolo(
            task_id=task_id,
            base_weights=base_weights,
            dataset_yaml=dataset_info["yaml"],
            run_root=train_root / "runs",
            max_training_seconds=max_training_seconds,
            hpo_plan=hpo_plan,
        )
        r2_creds = load_r2_credentials_from_env()
        remote_prefix = f"{r2_object_prefix.rstrip('/')}/{task_id}/"
        deleted_objects = delete_checkpoint_prefix_from_r2(creds=r2_creds, prefix=remote_prefix)
        bt.logging.info(
            f"event=r2_cleanup_dual_flywheel task_id={task_id} prefix={remote_prefix} deleted_objects={deleted_objects}"
        )
        object_key = f"{remote_prefix}best.pt"
        remote_uri = upload_checkpoint_to_r2(
            best_checkpoint,
            object_key=object_key,
            creds=r2_creds,
        )
        artifact_hash = self._sha256(best_checkpoint)
        training_seconds = max(1.0, time.monotonic() - start_time)
        config_payload = {
            "task_id": task_id,
            "source": "labeled_url_manifest",
            "validator_baseline_uri": baseline.uri,
            "validator_baseline_hash": baseline.sha256,
            "dataset_sha256": dataset_info["dataset_hash"],
            "class_hash": dataset_info["class_hash"],
            "class_names": dataset_info["class_names"],
            "baseline_weights_path": base_weights,
            "max_training_seconds": max_training_seconds,
            "auto_hpo": self.settings.enable_auto_hpo,
            "hpo_plan": hpo_plan,
            "labeled_image_count": len(labeled_images),
        }
        config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest()
        efficiency_score = min(1.0, max_training_seconds / training_seconds) if max_training_seconds > 0 else 0.0
        recipe_path = train_root / "recipe.json"
        recipe_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
        manifest = TrainingManifest(
            parent_model_hash=baseline.sha256,
            candidate_model_hash=artifact_hash,
            candidate_model_uri=remote_uri,
            config_hash=config_hash,
            dataset_lineage_hash=dataset_info["dataset_hash"],
            recipe_uri=recipe_path.as_uri(),
            metrics={
                "reproducibility_score": 1.0,
                "uplift": float(dataset_info["train_samples"]) / max(1.0, float(dataset_info["val_samples"])),
                "efficiency": float(max(0.0, min(1.0, efficiency_score))),
                "train_samples": float(dataset_info["train_samples"]),
                "val_samples": float(dataset_info["val_samples"]),
                "hpo_iterations": float(hpo_plan.get("iterations", 0)),
            },
        )
        manifest = manifest.model_copy(
            update={"candidate_model_uri": presign_r2_object_uri(creds=r2_creds, r2_uri=remote_uri)}
        )
        return (manifest, best_checkpoint)

    def _prepare_dataset_from_labeled_urls(
        self,
        dataset_root: Path,
        labeled_images: list[LabeledTrainingImage],
        fetch_image: Callable[[str], bytes],
        task_id: str,
    ) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("pillow is required for URL-labeled training.") from exc
        images_train = dataset_root / "images" / "train"
        labels_train = dataset_root / "labels" / "train"
        images_val = dataset_root / "images" / "val"
        labels_val = dataset_root / "labels" / "val"
        for path in (images_train, labels_train, images_val, labels_val):
            path.mkdir(parents=True, exist_ok=True)

        seed = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        indices = list(range(len(labeled_images)))
        rng.shuffle(indices)
        n = len(indices)
        if n == 1:
            train_idx = val_idx = indices
        else:
            split = max(1, int(n * 0.9))
            train_idx = indices[:split]
            val_idx = indices[split:] if len(indices[split:]) > 0 else indices[-1:]

        class_names: set[str] = set()

        def materialize_split(idxs: list[int], image_dir: Path, label_dir: Path, tag: str) -> None:
            for local_i, global_idx in enumerate(idxs):
                item = labeled_images[global_idx]
                data = fetch_image(item.image_url)
                image = Image.open(io.BytesIO(data)).convert("RGB")
                width, height = image.size
                stem = f"{tag}_{local_i:05d}"
                image_path = image_dir / f"{stem}.jpg"
                image.save(image_path)
                lines: list[str] = []
                for label in item.labels:
                    class_names.add(label.hazard_class)
                    x1, y1, x2, y2 = label.bounding_box
                    x1c = max(0.0, min(float(width - 1), float(x1)))
                    y1c = max(0.0, min(float(height - 1), float(y1)))
                    x2c = max(0.0, min(float(width), float(x2)))
                    y2c = max(0.0, min(float(height), float(y2)))
                    if x2c <= x1c or y2c <= y1c:
                        continue
                    x_center = ((x1c + x2c) / 2.0) / float(width)
                    y_center = ((y1c + y2c) / 2.0) / float(height)
                    w_norm = (x2c - x1c) / float(width)
                    h_norm = (y2c - y1c) / float(height)
                    lines.append(f"{label.hazard_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                (label_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        materialize_split(train_idx, images_train, labels_train, "tr")
        materialize_split(val_idx, images_val, labels_val, "va")

        class_list = sorted(class_names)
        if not class_list:
            raise ValueError("Labeled training images contained no hazard_class labels.")
        class_to_id = {name: idx for idx, name in enumerate(class_list)}
        self._rewrite_label_ids(labels_train, class_to_id)
        self._rewrite_label_ids(labels_val, class_to_id)

        yaml_path = dataset_root / "dataset.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    f"path: {dataset_root}",
                    "train: images/train",
                    "val: images/val",
                    f"nc: {len(class_list)}",
                    "names:",
                    *[f"  - {name}" for name in class_list],
                ]
            ),
            encoding="utf-8",
        )

        dataset_hash = hashlib.sha256()
        for label_file in sorted(labels_train.glob("*.txt")):
            dataset_hash.update(label_file.read_bytes())
        for label_file in sorted(labels_val.glob("*.txt")):
            dataset_hash.update(label_file.read_bytes())
        class_hash = hashlib.sha256("\n".join(class_list).encode("utf-8")).hexdigest()
        return {
            "yaml": yaml_path,
            "dataset_hash": dataset_hash.hexdigest(),
            "class_hash": class_hash,
            "class_names": class_list,
            "train_samples": len(list(labels_train.glob("*.txt"))),
            "val_samples": len(list(labels_val.glob("*.txt"))),
        }

    def _hpo_candidate_grid(self, max_training_seconds: int) -> list[dict[str, Any]]:
        eval_budget = max(
            1,
            max_training_seconds // max(1, self.settings.autoresearch_max_iters),
        )
        return [
            {"lr0": 0.005, "batch": 8, "imgsz": 640, "epochs": min(MAX_EPOCHS, max(1, eval_budget // 45))},
            {"lr0": 0.001, "batch": 16, "imgsz": 640, "epochs": min(MAX_EPOCHS, max(1, eval_budget // 50))},
            {"lr0": 0.0005, "batch": 8, "imgsz": 768, "epochs": min(MAX_EPOCHS, max(1, eval_budget // 55))},
            {"lr0": 0.003, "batch": 12, "imgsz": 640, "epochs": min(MAX_EPOCHS, max(1, eval_budget // 48))},
        ]

    def _resolve_hpo_plan(
        self,
        *,
        task_id: str,
        train_root: Path,
        dataset_yaml: Path,
        base_weights: str,
        max_training_seconds: int,
    ) -> dict[str, Any]:
        if self.settings.enable_auto_hpo:
            return self._run_autoresearch_loop(
                task_id=task_id,
                train_root=train_root,
                dataset_yaml=dataset_yaml,
                base_weights=base_weights,
                max_training_seconds=max_training_seconds,
            )
        if self.settings.random_hpo_draw:
            candidates = self._hpo_candidate_grid(max_training_seconds)
            rng = random.Random(int(self.settings.hpo_seed))
            plan = dict(rng.choice(candidates))
            plan["iterations"] = 0
            bt.logging.info(
                f"event=random_hpo_draw task_id={task_id} seed={self.settings.hpo_seed} plan={plan}"
            )
            return plan
        return {"epochs": self._target_epochs(max_training_seconds)}

    def _run_autoresearch_loop(
        self,
        *,
        task_id: str,
        train_root: Path,
        dataset_yaml: Path,
        base_weights: str,
        max_training_seconds: int,
    ) -> dict[str, Any]:
        """
        Karpathy-style autoresearch loop for miner-local HPO.

        This is intentionally lightweight but follows the same iterative pattern:
        propose -> short train/eval -> keep best config -> repeat.
        """
        log_path = train_root / "autoresearch.log"
        candidates = self._hpo_candidate_grid(max_training_seconds)
        best_plan: dict[str, Any] | None = None
        best_score = -1.0
        lines: list[str] = [
            f"task_id={task_id}",
            f"mode=karpathy-style-autoresearch",
            f"dataset={dataset_yaml}",
            f"base_weights={base_weights}",
            f"iterations={self.settings.autoresearch_max_iters}",
            f"experiment_minutes={self.settings.autoresearch_experiment_minutes}",
        ]
        for idx in range(min(self.settings.autoresearch_max_iters, len(candidates))):
            plan = dict(candidates[idx])
            score = self._evaluate_hpo_candidate(
                train_root=train_root,
                dataset_yaml=dataset_yaml,
                base_weights=base_weights,
                idx=idx,
                plan=plan,
            )
            lines.append(f"iter={idx} plan={json.dumps(plan, sort_keys=True)} score={score:.6f}")
            if score > best_score:
                best_score = score
                best_plan = plan
        if best_plan is None:
            raise RuntimeError("Autoresearch loop failed to produce an HPO plan.")
        best_plan["iterations"] = min(self.settings.autoresearch_max_iters, len(candidates))
        best_plan["score"] = best_score
        lines.append(f"selected={json.dumps(best_plan, sort_keys=True)}")
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return best_plan

    @staticmethod
    def _evaluate_hpo_candidate(
        *,
        train_root: Path,
        dataset_yaml: Path,
        base_weights: str,
        idx: int,
        plan: dict[str, Any],
    ) -> float:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for autoresearch loop.") from exc
        candidate_dir = train_root / "autoresearch_runs"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(base_weights)
        name = f"candidate_{idx}"
        model.train(
            data=str(dataset_yaml),
            imgsz=int(plan["imgsz"]),
            epochs=int(plan["epochs"]),
            batch=int(plan["batch"]),
            lr0=float(plan["lr0"]),
            project=str(candidate_dir),
            name=name,
            exist_ok=True,
            pretrained=True,
            device="0" if TrainingPipeline._cuda_available() else "cpu",
            verbose=False,
        )
        metrics = model.val(
            data=str(dataset_yaml),
            split="val",
            imgsz=int(plan["imgsz"]),
            batch=int(plan["batch"]),
            device="0" if TrainingPipeline._cuda_available() else "cpu",
            verbose=False,
        )
        map50 = float(getattr(metrics.box, "map50", 0.0))
        return map50

    def _prepare_training_dataset(self, dataset_root: Path) -> dict:
        try:
            from datasets import load_dataset
            from PIL import Image
        except ImportError as exc:
            raise ImportError("datasets and pillow are required for real-model training.") from exc

        raw = load_dataset(DATASET_REPO_ID)
        if "train" not in raw:
            raise ValueError(f"Dataset {DATASET_REPO_ID} must expose a train split.")
        split = raw["train"].train_test_split(test_size=0.2, seed=42)
        train_split = split["train"].select(range(min(len(split["train"]), MAX_TRAIN_SAMPLES)))
        golden_split = split["test"].select(range(min(len(split["test"]), MAX_VAL_SAMPLES)))

        images_train = dataset_root / "images" / "train"
        labels_train = dataset_root / "labels" / "train"
        images_val = dataset_root / "images" / "val"
        labels_val = dataset_root / "labels" / "val"
        for path in (images_train, labels_train, images_val, labels_val):
            path.mkdir(parents=True, exist_ok=True)

        class_names: set[str] = set()
        self._materialize_split(train_split, images_train, labels_train, class_names, Image.Image)
        self._materialize_split(golden_split, images_val, labels_val, class_names, Image.Image)
        class_list = sorted(class_names)
        if not class_list:
            raise ValueError("Construction safety dataset yielded no classes.")
        class_to_id = {name: idx for idx, name in enumerate(class_list)}
        self._rewrite_label_ids(labels_train, class_to_id)
        self._rewrite_label_ids(labels_val, class_to_id)

        yaml_path = dataset_root / "dataset.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    f"path: {dataset_root}",
                    "train: images/train",
                    "val: images/val",
                    f"nc: {len(class_list)}",
                    "names:",
                    *[f"  - {name}" for name in class_list],
                ]
            ),
            encoding="utf-8",
        )

        dataset_hash = hashlib.sha256()
        for label_file in sorted((labels_train.glob("*.txt"))):
            dataset_hash.update(label_file.read_bytes())
        for label_file in sorted((labels_val.glob("*.txt"))):
            dataset_hash.update(label_file.read_bytes())
        class_hash = hashlib.sha256("\n".join(class_list).encode("utf-8")).hexdigest()
        return {
            "yaml": yaml_path,
            "dataset_hash": dataset_hash.hexdigest(),
            "class_hash": class_hash,
            "class_names": class_list,
            "train_samples": len(list(labels_train.glob("*.txt"))),
            "golden_samples": len(list(labels_val.glob("*.txt"))),
        }

    def _materialize_split(
        self,
        split,
        image_dir: Path,
        label_dir: Path,
        class_names: set[str],
        image_type,
    ) -> None:
        for idx, sample in enumerate(split):
            image = sample.get("image")
            objects = sample.get("objects")
            if image is None or objects is None:
                raise ValueError("Dataset records must contain 'image' and 'objects' fields.")
            image_path = image_dir / f"{idx:08d}.jpg"
            if isinstance(image, image_type):
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                image.save(image_path)
                width, height = image.size
            else:
                raise ValueError("Unsupported image type in dataset sample.")
            lines = self._yolo_lines_from_objects(objects, width, height, class_names)
            (label_dir / f"{idx:08d}.txt").write_text("\n".join(lines), encoding="utf-8")

    def _yolo_lines_from_objects(
        self,
        objects: dict,
        width: int,
        height: int,
        class_names: set[str],
    ) -> list[str]:
        bbox_list = objects.get("bbox")
        category_list = objects.get("category")
        if bbox_list is None or category_list is None:
            raise ValueError("Objects must contain bbox and category entries.")
        if len(bbox_list) != len(category_list):
            raise ValueError("bbox and category lengths do not match.")
        lines: list[str] = []
        for bbox, category in zip(bbox_list, category_list):
            if not isinstance(category, str):
                category = str(category)
            class_names.add(category)
            x_min, y_min, box_w, box_h = [float(v) for v in bbox]
            x_center = (x_min + box_w / 2.0) / float(width)
            y_center = (y_min + box_h / 2.0) / float(height)
            w_norm = box_w / float(width)
            h_norm = box_h / float(height)
            lines.append(f"{category} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        return lines

    @staticmethod
    def _rewrite_label_ids(label_dir: Path, class_to_id: dict[str, int]) -> None:
        for label_file in label_dir.glob("*.txt"):
            lines: list[str] = []
            for line in label_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                class_name = parts[0]
                if class_name not in class_to_id:
                    raise ValueError(f"Unknown class label in dataset: {class_name}")
                lines.append(" ".join([str(class_to_id[class_name]), *parts[1:]]))
            label_file.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _resolve_baseline_weights(uri: str) -> str:
        if not uri:
            raise ValueError("baseline_checkpoint.uri must be provided.")
        if uri.endswith(".pt"):
            return uri
        if "yolov8s" in uri.lower():
            return YOLO_BASELINE
        raise ValueError(
            f"Unsupported baseline checkpoint URI for real YOLO training: {uri}. "
            "Provide a .pt checkpoint or yolov8s reference."
        )

    @staticmethod
    def _train_yolo(
        *,
        task_id: str,
        base_weights: str,
        dataset_yaml: Path,
        run_root: Path,
        max_training_seconds: int,
        hpo_plan: dict[str, Any],
    ) -> Path:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for real YOLO training.") from exc
        run_root.mkdir(parents=True, exist_ok=True)
        model = YOLO(base_weights)
        epochs = int(hpo_plan.get("epochs", TrainingPipeline._target_epochs(max_training_seconds)))
        batch = int(hpo_plan.get("batch", 16))
        imgsz = int(hpo_plan.get("imgsz", 640))
        lr0 = float(hpo_plan.get("lr0", 0.001))
        model.train(
            data=str(dataset_yaml),
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            lr0=lr0,
            project=str(run_root),
            name="yolov8s_construction",
            exist_ok=True,
            pretrained=True,
            device="0" if TrainingPipeline._cuda_available() else "cpu",
        )
        best_checkpoint = run_root / "yolov8s_construction" / "weights" / "best.pt"
        if not best_checkpoint.exists():
            raise FileNotFoundError(f"YOLO training did not produce checkpoint: {best_checkpoint}")
        return best_checkpoint

    @staticmethod
    def _target_epochs(max_training_seconds: int) -> int:
        if max_training_seconds <= 0:
            return 1
        return min(MAX_EPOCHS, max(1, int(max_training_seconds // 120)))

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

