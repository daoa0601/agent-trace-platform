"""Strict evidence models for OpenCode Coding Plan benchmark sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agenttrace.benchmarks.models import BenchmarkEvaluation
from agenttrace.harnesses.opencode.proxy import BrokerRequest
from agenttrace.harnesses.opencode.workspace import WorkspaceChange
from agenttrace.models import TokenUsage


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BrokerRequestEvidence(StrictModel):
    sequence: int = Field(ge=0)
    status_code: int
    latency_ms: float = Field(ge=0)
    request_bytes: int = Field(ge=0)
    response_bytes: int = Field(ge=0)
    thinking_mode: Literal["enabled"] | None = None
    reasoning_effort: Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"] | None = (
        None
    )

    @classmethod
    def from_request(cls, request: BrokerRequest) -> BrokerRequestEvidence:
        return cls(**request.__dict__)


class OpenCodeBenchmarkRecord(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    experiment_id: str
    run_id: str
    episode_id: str
    trace_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    task_id: str
    task_version: str
    task_split: str | None = None
    sample_id: str
    status: Literal["completed", "process_error", "timed_out"]
    provider: Literal["zai"] = "zai"
    transport: Literal["opencode"] = "opencode"
    endpoint_plan: Literal["coding"] = "coding"
    model: str
    thinking_mode: Literal["enabled"] | None = None
    reasoning_effort: Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"] | None = (
        None
    )
    started_at: datetime
    ended_at: datetime
    latency_ms: float = Field(ge=0)
    exit_code: int
    agent_decided_stop: bool
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_bytes: int = Field(ge=0)
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stdout_bytes: int = Field(ge=0)
    stderr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stderr_bytes: int = Field(ge=0)
    raw_events_ref: str
    event_count: int = Field(ge=0)
    invalid_event_count: int = Field(ge=0)
    redaction_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_bytes: int | None = Field(default=None, ge=0)
    output_ref: str | None = None
    broker_requests: list[BrokerRequestEvidence]
    workspace_changes: list[WorkspaceChange]
    evaluations: list[BenchmarkEvaluation] = Field(default_factory=list)
    aggregate_score: float | None = Field(default=None, ge=0)
    required_evaluators_passed: bool | None = None


class OpenCodeRecordRef(StrictModel):
    sample_id: str
    status: Literal["completed", "process_error", "timed_out"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    key: str
    trace_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class OpenCodeBenchmarkSummary(StrictModel):
    sample_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    process_error_count: int = Field(ge=0)
    timed_out_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)


class OpenCodeBenchmarkManifest(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    created_at: datetime
    experiment_id: str
    task_id: str
    task_version: str
    provider: Literal["zai"] = "zai"
    transport: Literal["opencode"] = "opencode"
    endpoint_plan: Literal["coding"] = "coding"
    model: str
    thinking_mode: Literal["enabled"] | None = None
    reasoning_effort: Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"] | None = (
        None
    )
    task_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    samples_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    records: list[OpenCodeRecordRef]
    summary: OpenCodeBenchmarkSummary


@dataclass(frozen=True)
class OpenCodeBenchmarkExecution:
    manifest: OpenCodeBenchmarkManifest
    manifest_key: str
