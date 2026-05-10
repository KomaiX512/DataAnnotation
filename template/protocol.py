# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer):TECHNOLOGY NUCLEUS
# Copyright © 2023 TECHNOLOGY NUCLEUS

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import bittensor as bt
from pydantic import BaseModel, Field, field_validator

TaskType = Literal["inference", "verification", "training"]
DatasetPartition = Literal["training_pool", "golden", "hidden_eval", "replay", "promotion"]
SeverityTier = Literal["none", "low", "medium", "high", "critical"]


class BoundingBox(BaseModel):
    """Axis-aligned detection box in normalized coordinates."""

    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)
    label: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class TrainingManifest(BaseModel):
    """Miner-submitted metadata describing a training contribution."""

    parent_model_hash: str = Field(..., min_length=8)
    candidate_model_hash: str = Field(..., min_length=8)
    candidate_model_uri: str = Field(..., min_length=1)
    config_hash: str = Field(..., min_length=8)
    dataset_lineage_hash: str = Field(..., min_length=8)
    recipe_uri: str = Field(..., min_length=1)
    metrics: Dict[str, float] = Field(default_factory=dict)


class DatasetPointer(BaseModel):
    """Configurable dataset pointer shared by validators and miners."""

    uri: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=8)
    split: DatasetPartition
    sample_count: int = Field(..., ge=1)


class ModelCheckpoint(BaseModel):
    """Current global baseline checkpoint issued by validators."""

    uri: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=8)


class TrainingAnnotationLabel(BaseModel):
    """Single labeled instance for miner fine-tuning (pixel boxes, aligned with eval schema)."""

    hazard_class: str = Field(..., min_length=1)
    bounding_box: List[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Axis-aligned box [x_min, y_min, x_max, y_max] in pixel coordinates.",
    )
    severity: SeverityTier
    reasoning: str = Field("", max_length=4096)

    @field_validator("bounding_box")
    @classmethod
    def _box_non_negative(cls, v: List[int]) -> List[int]:
        if any(x < 0 for x in v):
            raise ValueError("bounding_box coordinates must be non-negative")
        if v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError("bounding_box must have x_max > x_min and y_max > y_min")
        return v


class LabeledTrainingImage(BaseModel):
    """Labeled image from the miner training pool (URL + structured labels)."""

    image_url: str = Field(..., min_length=1)
    image_id: str = Field("", description="Optional lineage id (e.g. content hash prefix).")
    labels: List[TrainingAnnotationLabel] = Field(default_factory=list)


class UnlabeledAnnotationImage(BaseModel):
    """Unlabeled construction image the miner must annotate."""

    image_url: str = Field(..., min_length=1)
    image_id: str = Field(..., min_length=1)


class PerImageAnnotationItem(BaseModel):
    """One predicted hazard instance for annotations.json."""

    hazard_class: str = Field(..., min_length=1)
    bounding_box: List[int] = Field(..., min_length=4, max_length=4)
    severity: SeverityTier
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_chain: str = Field(..., min_length=1, max_length=8192)
    osha_reference: Optional[str] = Field(
        None,
        description="OSHA citation id when applicable (e.g. 1926.501(b)(1)).",
    )


class ImageAnnotationDocument(BaseModel):
    """Per-image record written into annotations.json."""

    image_id: str
    miner_uid: str
    timestamp: str
    annotations: List[PerImageAnnotationItem]
    model_version: str = Field(..., min_length=8)


class AnnotationsFilePayload(BaseModel):
    """Wrapper for uploaded annotations.json."""

    schema_version: str = Field("annotations.v1", min_length=1)
    task_id: str = Field("", min_length=0)
    records: List[ImageAnnotationDocument]


class R2AccessCredentials(BaseModel):
    """Miner-advertised Cloudflare R2 access credentials for validator download."""

    account_id: str = Field(..., min_length=4)
    bucket_name: str = Field(..., min_length=1)
    s3_endpoint: str = Field(..., min_length=8)
    access_key_id: str = Field(..., min_length=4)
    secret_access_key: str = Field(..., min_length=8)
    token: Optional[str] = Field(None)
    public_bucket_url: Optional[str] = Field(None)


class HazardDetection(bt.Synapse):
    """
    Production protocol for construction hazard subnet tasks.

    Validators submit one explicit task type and miners must answer through the
    same schema. There is no alternate protocol path.
    """

    schema_version: str = Field(
        "hazard.v2",
        title="Schema Version",
        description="Protocol schema version enforced by validators.",
    )
    task_type: str = Field(
        "",
        title="Task Type",
        description="Task category issued by the validator.",
    )
    dataset_partition: str = Field(
        "",
        title="Dataset Partition",
        description="Validator-owned partition used to generate this task.",
    )
    task_id: str = Field(
        "",
        title="Task ID",
        description="Stable identifier for reproducibility and auditing.",
    )
    site_id: str = Field(
        "",
        title="Site ID",
        description="Construction site identifier attached by validator.",
    )
    challenge_nonce: str = Field(
        "",
        title="Challenge Nonce",
        description="Verifier nonce to prevent stale response replay.",
    )
    image_b64: str = Field(
        "",
        title="Image Base64",
        description="Base64-encoded construction image payload for inference tasks.",
    )
    image_format: str = Field(
        "jpg",
        title="Image Format",
        description="Image encoding format for image_b64 payload.",
    )
    training_dataset: Optional[DatasetPointer] = Field(
        None,
        title="Training Dataset Pointer",
        description="70 percent miner-visible CSDataset split or compatible replacement.",
    )
    golden_dataset: Optional[DatasetPointer] = Field(
        None,
        title="Golden Dataset Pointer",
        description="Validator-only held-out split used for checkpoint scoring.",
    )
    baseline_checkpoint: Optional[ModelCheckpoint] = Field(
        None,
        title="Baseline Checkpoint",
        description="Current global baseline that miners must fine-tune.",
    )
    max_training_seconds: Optional[int] = Field(
        None,
        ge=1,
        title="Max Training Seconds",
        description="Validator-enforced wall-clock training budget for smoke or production tasks.",
    )
    requested_model_hash: Optional[str] = Field(
        None,
        title="Requested Model Hash",
        description="Optional model hash to enforce during verification tasks.",
    )
    training_manifest: Optional[TrainingManifest] = Field(
        None,
        title="Training Manifest",
        description="Training claim under audit for training_commit tasks.",
    )
    submitted_training_manifest: Optional[TrainingManifest] = Field(
        None,
        title="Submitted Training Manifest",
        description="Miner-provided training claim captured during response.",
    )
    miner_r2_credentials: Optional[R2AccessCredentials] = Field(
        None,
        title="Miner R2 Credentials Handshake",
        description=(
            "Deprecated. Validators must not require long-lived object-storage keys in-band; "
            "miners should expose checkpoints via short-lived https:// presigned URLs on the manifest."
        ),
    )
    miner_storage_signal: Optional[str] = Field(
        None,
        title="Miner Storage Download Signal",
        description="Signal sent by miner after upload to trigger validator checkpoint download and scoring.",
    )

    hazard_detected: Optional[bool] = Field(None, title="Hazard Detected")
    severity: Optional[SeverityTier] = Field(None, title="Severity")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)
    rationale: Optional[str] = Field(None, max_length=1024)
    osha_refs: List[str] = Field(default_factory=list)
    model_hash: Optional[str] = Field(None, min_length=8)
    training_metrics: Dict[str, float] = Field(default_factory=dict)
    duration_ms: Optional[int] = Field(None, ge=0)
    error_message: Optional[str] = Field(None, max_length=512)

    def deserialize(self) -> "HazardDetection":
        """Return the complete response payload to validator callers."""
        return self


class AnnotationAndTrainingTask(bt.Synapse):
    """
    Dual-flywheel round: fine-tune on labeled training URLs and produce structured
    annotations for unlabeled images, with artifacts uploaded to object storage.
    """

    schema_version: str = Field(
        "hazard.dual_flywheel.v1",
        title="Schema Version",
        description="Dual-flywheel annotation + training protocol version.",
    )
    task_id: str = Field("", title="Task ID", description="Stable round identifier.")
    challenge_nonce: str = Field(
        "",
        title="Challenge Nonce",
        description="Nonce mixed into deterministic test backends and auditing.",
    )
    training_images: List[LabeledTrainingImage] = Field(
        default_factory=list,
        title="Training Images",
        description="Labeled images (URLs) for fine-tuning.",
    )
    annotation_images: List[UnlabeledAnnotationImage] = Field(
        default_factory=list,
        title="Annotation Images",
        description="Unlabeled images (URLs + image_id) to annotate.",
    )
    base_model_hash: str = Field(
        "",
        title="Base Model Hash",
        description="SHA-256 of the global baseline checkpoint miners must fine-tune.",
    )
    baseline_checkpoint: Optional[ModelCheckpoint] = Field(
        None,
        title="Baseline Checkpoint",
        description="URI and hash for the baseline weights (must match base_model_hash).",
    )
    max_training_seconds: Optional[int] = Field(
        None,
        ge=1,
        title="Max Training Seconds",
        description="Wall-clock budget for fine-tuning in this round.",
    )

    annotations_uri: str = Field(
        "",
        title="Annotations Download URL",
        description="file:// path for local tests or https:// presigned GET for annotations.json.",
    )
    model_checkpoint_uri: str = Field("", title="Checkpoint Prefix or Object URI")
    training_config: Dict[str, Any] = Field(default_factory=dict, title="Training Config")
    claim_improvement: Optional[float] = Field(
        None,
        title="Claimed Improvement",
        description="Optional miner self-report (e.g. expected mAP delta).",
    )
    submitted_training_manifest: Optional[TrainingManifest] = Field(
        None,
        title="Training Manifest",
        description="Checkpoint lineage returned for validator verification.",
    )
    miner_r2_credentials: Optional[R2AccessCredentials] = Field(
        None,
        title="Miner R2 Credentials Handshake",
        description=(
            "Deprecated. Use short-lived https:// presigned GET URLs in annotations_uri and "
            "submitted_training_manifest.candidate_model_uri instead of passing API keys."
        ),
    )
    duration_ms: Optional[int] = Field(None, ge=0)
    error_message: Optional[str] = Field(None, max_length=1024)

    def deserialize(self) -> "AnnotationAndTrainingTask":
        return self
