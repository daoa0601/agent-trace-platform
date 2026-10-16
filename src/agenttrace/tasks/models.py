"""Task and environment contracts independent of any particular agent harness."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from benchmark_audit_core import AuditPolicy, SessionPolicy


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvironmentKind(StrEnum):
    NONE = "none"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    REMOTE = "remote"


class ResourceLimits(StrictModel):
    cpu: float = Field(default=1.0, gt=0)
    memory_mb: int = Field(default=1024, ge=64)
    disk_mb: int = Field(default=2048, ge=64)
    gpu_count: int = Field(default=0, ge=0)
    timeout_seconds: int = Field(default=300, ge=1)


class EnvironmentSpec(StrictModel):
    kind: EnvironmentKind = EnvironmentKind.NONE
    image: str | None = None
    image_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    entrypoint: list[str] = Field(default_factory=list)
    setup: list[list[str]] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    network: Literal["none", "restricted", "full"] = "none"
    network_name: str | None = None
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    state_capture: Literal["none", "metadata", "full"] = "metadata"

    @model_validator(mode="after")
    def validate_runtime(self) -> EnvironmentSpec:
        if self.kind in {EnvironmentKind.DOCKER, EnvironmentKind.KUBERNETES} and not self.image:
            raise ValueError(f"{self.kind.value} environments require an image")
        if self.network == "restricted" and not self.network_name:
            raise ValueError("restricted network mode requires network_name")
        return self


class EvaluatorSpec(StrictModel):
    evaluator_id: str
    kind: Literal[
        "exact_match",
        "contains",
        "regex",
        "json_schema",
        "tool_success",
        "latency_budget",
        "custom",
    ]
    version: str
    weight: float = Field(default=1.0, ge=0)
    required: bool = True
    config: dict[str, JsonValue] = Field(default_factory=dict)


class DataPolicy(StrictModel):
    privacy_level: Literal["public", "internal", "confidential", "restricted"] = "internal"
    retain_raw_days: int = Field(default=30, ge=1)
    retain_canonical_days: int = Field(default=365, ge=1)
    allow_training: bool = False
    allow_private_reasoning: bool = False
    redaction_mode: Literal["balanced", "strict"] = "balanced"


class TaskDefinition(StrictModel):
    """Portable task definition consumed by runners and evaluators."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    task_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")
    title: str
    description: str
    domain: str
    tags: list[str] = Field(default_factory=list)
    split: str | None = None
    input_schema: dict[str, JsonValue]
    expected_output_schema: dict[str, JsonValue] | None = None
    reference: JsonValue | None = None
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    evaluators: list[EvaluatorSpec] = Field(min_length=1)
    data_policy: DataPolicy = Field(default_factory=DataPolicy)
    session_policy: SessionPolicy | None = None
    audit_policy: AuditPolicy | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evaluators(self) -> TaskDefinition:
        ids = [evaluator.evaluator_id for evaluator in self.evaluators]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluator_id values must be unique within a task")
        return self
