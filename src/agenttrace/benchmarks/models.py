"""Strict schemas for direct-model benchmark evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from agenttrace.models import TokenUsage


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkEvaluation(StrictModel):
    evaluator_id: str
    evaluator_version: str
    kind: str
    required: bool
    weight: float = Field(ge=0)
    status: Literal["passed", "failed", "skipped"]
    passed: bool | None
    score: float | None
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> BenchmarkEvaluation:
        if self.status == "skipped" and (self.passed is not None or self.score is not None):
            raise ValueError("skipped evaluations cannot have pass or score values")
        if self.status != "skipped" and (self.passed is None or self.score is None):
            raise ValueError("completed evaluations require pass and score values")
        return self


class BenchmarkError(StrictModel):
    error_type: str
    status_code: int | None = None
    request_id: str | None = None


class BenchmarkRecord(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    experiment_id: str
    run_id: str
    episode_id: str
    trace_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    task_id: str
    task_version: str
    task_split: str | None = None
    sample_id: str
    status: Literal["completed", "provider_error"]
    provider: Literal["zai"] = "zai"
    endpoint_plan: Literal["general"] = "general"
    requested_model: str
    response_model: str | None = None
    thinking_mode: Literal["enabled", "disabled"]
    reasoning_effort: Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"] | None = (
        None
    )
    started_at: datetime
    ended_at: datetime
    latency_ms: float = Field(ge=0)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_bytes: int = Field(ge=0)
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_bytes: int | None = Field(default=None, ge=0)
    output_ref: str | None = None
    finish_reason: str | None = None
    provider_request_id: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    evaluations: list[BenchmarkEvaluation] = Field(default_factory=list)
    aggregate_score: float | None = Field(default=None, ge=0)
    required_evaluators_passed: bool | None = None
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    error: BenchmarkError | None = None

    @model_validator(mode="after")
    def validate_completion(self) -> BenchmarkRecord:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        if self.status == "completed":
            if self.output_sha256 is None or self.output_bytes is None or self.error is not None:
                raise ValueError("completed records require output metadata and no error")
        elif self.error is None:
            raise ValueError("provider_error records require an error")
        return self


class BenchmarkRecordRef(StrictModel):
    sample_id: str
    status: Literal["completed", "provider_error"]
    required_evaluators_passed: bool | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    key: str


class BenchmarkSummary(StrictModel):
    sample_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    provider_error_count: int = Field(ge=0)
    required_pass_count: int = Field(ge=0)
    required_fail_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)


class BenchmarkManifest(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    created_at: datetime
    experiment_id: str
    task_id: str
    task_version: str
    provider: Literal["zai"] = "zai"
    endpoint_plan: Literal["general"] = "general"
    model: str
    thinking_mode: Literal["enabled", "disabled"]
    reasoning_effort: Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"] | None = (
        None
    )
    task_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    samples_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    store_outputs: bool
    records: list[BenchmarkRecordRef]
    summary: BenchmarkSummary


@dataclass(frozen=True)
class BenchmarkExecution:
    manifest: BenchmarkManifest
    manifest_key: str
