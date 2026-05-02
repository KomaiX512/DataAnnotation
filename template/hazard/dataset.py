from __future__ import annotations

import base64
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from template.protocol import DatasetPartition, DatasetPointer, SeverityTier

DatasetTaskType = Literal["inference", "verification", "training"]
HF_CONSTRUCTION_SAFETY_DATASET = "cppe-5"


@dataclass(frozen=True)
class DatasetTask:
    task_id: str
    site_id: str
    image_bytes: bytes
    image_format: str
    hazard_detected: bool
    severity: SeverityTier
    expected_boxes: List[Dict[str, float]]
    expected_osha_refs: List[str]
    partition: DatasetPartition
    expected_model_hash: Optional[str]
    task_type: DatasetTaskType


class HazardDatasetManager:
    """
    Validator-owned hidden dataset partitions.
    """

    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self._partitions: Dict[DatasetPartition, List[DatasetTask]] = {
            "training_pool": [],
            "golden": [],
            "hidden_eval": [],
            "replay": [],
            "promotion": [],
        }
        self._load()

    def sample(
        self,
        partition: DatasetPartition,
        *,
        task_type: DatasetTaskType = "inference",
        random_state: random.Random,
    ) -> DatasetTask:
        items = self._partitions[partition]
        if not items:
            raise ValueError(f"Dataset partition {partition} is empty.")
        candidate = random_state.choice(items)
        return DatasetTask(
            task_id=candidate.task_id,
            site_id=candidate.site_id,
            image_bytes=candidate.image_bytes,
            image_format=candidate.image_format,
            hazard_detected=candidate.hazard_detected,
            severity=candidate.severity,
            expected_boxes=list(candidate.expected_boxes),
            expected_osha_refs=list(candidate.expected_osha_refs),
            partition=candidate.partition,
            expected_model_hash=candidate.expected_model_hash,
            task_type=task_type,
        )

    def _load(self) -> None:
        for partition in self._partitions.keys():
            path = self.dataset_root / f"{partition}.jsonl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing dataset partition file: {path}. "
                    "All partitions are required for the subnet runtime."
                )
            self._partitions[partition] = self._read_partition(path, partition)

    def pointer(self, partition: DatasetPartition) -> DatasetPointer:
        if partition == "training_pool":
            synthetic_hash = hashlib.sha256(
                f"{HF_CONSTRUCTION_SAFETY_DATASET}:train[:80%]".encode("utf-8")
            ).hexdigest()
            return DatasetPointer(
                uri=f"hf-dataset://{HF_CONSTRUCTION_SAFETY_DATASET}/train[:80%]",
                sha256=synthetic_hash,
                split=partition,
                sample_count=max(1, len(self._partitions[partition])),
            )
        path = self.dataset_root / f"{partition}.jsonl"
        items = self._partitions[partition]
        if not items:
            raise ValueError(f"Dataset partition {partition} is empty.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return DatasetPointer(
            uri=path.as_uri(),
            sha256=digest,
            split=partition,
            sample_count=len(items),
        )

    def golden_hash(self) -> str:
        return self.pointer("golden").sha256

    @staticmethod
    def _read_partition(path: Path, partition: DatasetPartition) -> List[DatasetTask]:
        tasks: List[DatasetTask] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                tasks.append(
                    DatasetTask(
                        task_id=str(payload["task_id"]),
                        site_id=str(payload["site_id"]),
                        image_bytes=base64.b64decode(payload["image_b64"]),
                        image_format=str(payload.get("image_format", "jpg")),
                        hazard_detected=bool(payload["hazard_detected"]),
                        severity=payload["severity"],
                        expected_boxes=list(payload.get("expected_boxes", [])),
                        expected_osha_refs=list(payload.get("expected_osha_refs", [])),
                        partition=partition,
                        expected_model_hash=payload.get("expected_model_hash"),
                        task_type="inference",
                    )
                )
        return tasks

