"""Versioned canonical models for agent trace events and episodes."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from agenttrace.ids import new_id, new_span_id, new_trace_id

TRACE_SCHEMA_VERSION = "1.0.0"
# Bounds per-event attribute work at ingestion (OpenTelemetry SDKs default to 128 per span).
MAX_EVENT_ATTRIBUTES = 512


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Base model that rejects accidental schema drift."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StepType(StrEnum):
    """Portable step categories across agent and environment implementations."""

    EPISODE_START = "episode_start"
    USER_INPUT = "user_input"
    SYSTEM_MESSAGE = "system_message"
    MODEL_GENERATION = "model_generation"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ENVIRONMENT_OBSERVATION = "environment_observation"
    ENVIRONMENT_ACTION = "environment_action"
    COMPILE = "compile"
    CHECK = "check"
    BENCHMARK = "benchmark"
    AUDIT_RETEST = "audit_retest"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    AGENT_STATE = "agent_state"
    EVALUATOR_RESULT = "evaluator_result"
    REWARD = "reward"
    FINAL_OUTPUT = "final_output"
    EPISODE_END = "episode_end"
    ERROR = "error"


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    TRUNCATED = "truncated"


class PrivacyLevel(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class Content(StrictModel):
    """Inline content or a reference to a content-addressed object."""

    inline: JsonValue | None = None
    ref: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    media_type: str = "application/json"
    byte_size: int | None = Field(default=None, ge=0)
    encoding: str | None = None
    redacted: bool = False

    @model_validator(mode="after")
    def validate_location(self) -> Content:
        if self.inline is not None and self.ref is not None:
            raise ValueError("content must be inline or referenced, not both")
        if self.ref is not None and self.sha256 is None:
            raise ValueError("referenced content must include a sha256 digest")
        return self


class SamplingConfig(StrictModel):
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    seed: int | None = None
    max_tokens: int | None = Field(default=None, ge=1)


class ModelInvocation(StrictModel):
    provider: str
    model_id: str
    checkpoint_id: str | None = None
    tokenizer_id: str | None = None
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    input_token_ids: Content | None = None
    output_token_ids: Content | None = None
    behavior_logprobs: Content | None = None
    reference_logprobs: Content | None = None
    value_estimates: Content | None = None


class ToolInteraction(StrictModel):
    call_id: str | None = None
    name: str
    arguments: Content | None = None
    result: Content | None = None
    exit_code: int | None = None
    success: bool | None = None
    error_type: str | None = None


class EnvironmentState(StrictModel):
    environment_id: str
    environment_version: str
    state_before: Content | None = None
    state_after: Content | None = None
    observation: Content | None = None
    action: Content | None = None
    terminated: bool | None = None
    truncated: bool | None = None


class RewardData(StrictModel):
    total: float
    components: dict[str, float] = Field(default_factory=dict)
    reward_function_version: str
    reward_model_checkpoint: str | None = None
    judge_prompt_version: str | None = None
    normalization_method: str | None = None
    aggregation_method: str = "sum"
    calculated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_finite(self) -> RewardData:
        values = [self.total, *self.components.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("reward values must be finite")
        return self


class TokenUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)


class StepMetrics(StrictModel):
    latency_ms: float | None = Field(default=None, ge=0)
    queue_ms: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    tokens: TokenUsage = Field(default_factory=TokenUsage)


class Provenance(StrictModel):
    code_commit: str | None = None
    config_digest: str | None = None
    prompt_version: str | None = None
    prompt_template_version: str | None = None
    dataset_version: str | None = None
    policy_checkpoint_id: str | None = None
    tokenizer_version: str | None = None
    generation_engine_version: str | None = None
    tool_version: str | None = None
    environment_version: str | None = None
    behavior_policy_id: str | None = None
    benchmark_protocol_version: str | None = None
    final_artifact_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class TaskRef(StrictModel):
    task_id: str
    task_version: str
    split: str | None = None
    task_instance_id: str | None = None


class Privacy(StrictModel):
    level: PrivacyLevel = PrivacyLevel.INTERNAL
    contains_pii: bool = False
    contains_secrets: bool = False
    contains_private_reasoning: bool = False
    retention_days: int | None = Field(default=None, ge=1)


class TraceEvent(StrictModel):
    """One immutable, ordered event from an agent episode."""

    schema_version: Literal["1.0.0"] = TRACE_SCHEMA_VERSION
    event_id: str = Field(default_factory=lambda: new_id("evt"), min_length=8, max_length=128)
    experiment_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    episode_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(default_factory=new_trace_id, pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(default_factory=new_span_id, pattern=r"^[0-9a-f]{16}$")
    parent_span_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    step_id: str = Field(default_factory=lambda: new_id("step"), min_length=8, max_length=128)
    sequence_number: int = Field(ge=0)
    step_type: StepType
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    task: TaskRef | None = None
    input: Content | None = None
    output: Content | None = None
    model: ModelInvocation | None = None
    tool: ToolInteraction | None = None
    environment: EnvironmentState | None = None
    reward: RewardData | None = None
    metrics: StepMetrics = Field(default_factory=StepMetrics)
    provenance: Provenance = Field(default_factory=Provenance)
    privacy: Privacy = Field(default_factory=Privacy)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    # Attribute values too large to keep inline, moved to the blob store by the raw writer.
    offloaded_attributes: dict[str, Content] | None = None

    @model_validator(mode="after")
    def validate_timestamps_and_shape(self) -> TraceEvent:
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if self.ended_at is not None:
            if self.ended_at.tzinfo is None:
                raise ValueError("ended_at must be timezone-aware")
            if self.ended_at < self.started_at:
                raise ValueError("ended_at cannot precede started_at")
        if self.step_type == StepType.REWARD and self.reward is None:
            raise ValueError("reward steps must include reward data")
        if self.step_type in {StepType.TOOL_CALL, StepType.TOOL_RESULT} and self.tool is None:
            raise ValueError("tool steps must include tool data")
        if self.offloaded_attributes:
            if any(content.ref is None for content in self.offloaded_attributes.values()):
                raise ValueError("offloaded attributes must be object references")
            overlap = sorted(self.attributes.keys() & self.offloaded_attributes.keys())
            if overlap:
                raise ValueError(f"attributes cannot be both inline and offloaded: {overlap}")
        return self


class EventBatch(StrictModel):
    """Bounded native ingestion request."""

    worker_id: str = Field(default="unknown", min_length=1, max_length=128)
    events: Annotated[list[TraceEvent], Field(min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def validate_partition(self) -> EventBatch:
        experiments = {event.experiment_id for event in self.events}
        dates = {event.started_at.date() for event in self.events}
        event_ids = [event.event_id for event in self.events]
        if len(experiments) != 1 or len(dates) != 1:
            raise ValueError("one batch must contain one experiment and collection date")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique within a batch")
        for event in self.events:
            count = len(event.attributes) + len(event.offloaded_attributes or {})
            if count > MAX_EVENT_ATTRIBUTES:
                raise ValueError(
                    f"event {event.event_id} has {count} attributes; "
                    f"the limit is {MAX_EVENT_ATTRIBUTES}"
                )
        return self


class IngestResponse(StrictModel):
    accepted: int
    duplicate_count: int = 0
    batch_ref: str
    event_ids: list[str]


class QualityIssue(StrictModel):
    code: str
    severity: Literal["warning", "error"]
    message: str
    event_id: str | None = None


class Episode(StrictModel):
    """Ordered episode reconstructed from canonical steps."""

    schema_version: Literal["1.0.0"] = TRACE_SCHEMA_VERSION
    experiment_id: str
    run_id: str
    episode_id: str
    task: TaskRef | None = None
    status: RunStatus
    started_at: datetime
    ended_at: datetime | None = None
    steps: list[TraceEvent]
    quality_issues: list[QualityIssue] = Field(default_factory=list)
    total_reward: float | None = None
    terminal_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.quality_issues)


class EvaluationResult(StrictModel):
    evaluator_id: str
    evaluator_version: str
    passed: bool
    score: float
    weight: float = 1.0
    reward_components: dict[str, float] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    details: dict[str, JsonValue] = Field(default_factory=dict)
    calculated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_score(self) -> EvaluationResult:
        if not math.isfinite(self.score):
            raise ValueError("evaluation score must be finite")
        return self
