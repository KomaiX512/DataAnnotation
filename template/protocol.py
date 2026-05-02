# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

from typing import Dict, List, Literal, Optional

import bittensor as bt
from pydantic import BaseModel, Field

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
        description="Handshake payload giving validator access to miner-hosted checkpoint storage.",
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
