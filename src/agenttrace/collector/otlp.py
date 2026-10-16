"""OTLP protobuf to canonical AgentTrace event conversion."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, cast

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from pydantic import JsonValue

from agenttrace.models import (
    Content,
    EventBatch,
    ModelInvocation,
    Provenance,
    SamplingConfig,
    StepMetrics,
    StepType,
    TaskRef,
    TokenUsage,
    ToolInteraction,
    TraceEvent,
)

OPENINFERENCE_KIND_MAP = {
    "AGENT": StepType.AGENT_STATE,
    "CHAIN": StepType.AGENT_STATE,
    "EMBEDDING": StepType.EMBEDDING,
    "EVALUATOR": StepType.EVALUATOR_RESULT,
    "GUARDRAIL": StepType.EVALUATOR_RESULT,
    "LLM": StepType.MODEL_GENERATION,
    "RERANKER": StepType.RETRIEVAL,
    "RETRIEVER": StepType.RETRIEVAL,
    "TOOL": StepType.TOOL_CALL,
}


def _any_value(value: AnyValue) -> JsonValue:
    kind = value.WhichOneof("value")
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "bytes_value":
        return value.bytes_value.hex()
    if kind == "array_value":
        return [_any_value(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return {item.key: _any_value(item.value) for item in value.kvlist_value.values}
    return None


def _attributes(items: Any) -> dict[str, JsonValue]:
    return {item.key: _any_value(item.value) for item in items}


def _string(attributes: dict[str, JsonValue], *keys: str) -> str | None:
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _integer(attributes: dict[str, JsonValue], *keys: str) -> int:
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _number(attributes: dict[str, JsonValue], *keys: str) -> float | None:
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
            return float(value)
    return None


def _content(value: JsonValue | None) -> Content | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return Content(inline=cast(JsonValue, parsed))
        except json.JSONDecodeError:
            return Content(inline=value, media_type="text/plain")
    return Content(inline=value)


def _step_type(span: Span, attributes: dict[str, JsonValue]) -> StepType:
    explicit = _string(attributes, "agenttrace.step.type")
    if explicit is not None:
        try:
            return StepType(explicit)
        except ValueError:
            pass
    inference_kind = _string(attributes, "openinference.span.kind")
    if inference_kind is not None and inference_kind.upper() in OPENINFERENCE_KIND_MAP:
        return OPENINFERENCE_KIND_MAP[inference_kind.upper()]
    operation = _string(attributes, "gen_ai.operation.name")
    if operation in {"chat", "text_completion", "generate_content"}:
        return StepType.MODEL_GENERATION
    if span.status.code == 2:
        return StepType.ERROR
    return StepType.AGENT_STATE


MAX_STATUS_DESCRIPTION_BYTES = 256


def _record_status_description(attributes: dict[str, JsonValue], message: str) -> str | None:
    """Keep a short status description inline; reference a long one by digest only.

    Status descriptions are meant to be error codes. Anything longer is free text that would
    bypass payload offloading, so only its size and SHA-256 are kept.
    """
    if not message:
        return None
    encoded = message.encode("utf-8")
    if len(encoded) <= MAX_STATUS_DESCRIPTION_BYTES:
        attributes["otel.status_description"] = message
        return message
    attributes["otel.status_description_sha256"] = hashlib.sha256(encoded).hexdigest()
    attributes["otel.status_description_bytes"] = len(encoded)
    return None


def _event_id(trace_id: str, span_id: str, end_nanos: int) -> str:
    digest = hashlib.sha256(f"{trace_id}:{span_id}:{end_nanos}".encode()).hexdigest()
    return f"evt_{digest[:32]}"


def _timestamp(nanoseconds: int) -> datetime:
    return datetime.fromtimestamp(nanoseconds / 1_000_000_000, tz=UTC)


def convert_otlp_request(request: ExportTraceServiceRequest) -> EventBatch:
    """Convert one OTLP export request, preserving all span/resource attributes."""
    span_rows: list[tuple[Span, dict[str, JsonValue]]] = []
    worker_id = "otel-collector"
    for resource_spans in request.resource_spans:
        resource_attributes = _attributes(resource_spans.resource.attributes)
        worker_id = _string(resource_attributes, "service.instance.id") or worker_id
        for scope_spans in resource_spans.scope_spans:
            scope_attributes: dict[str, JsonValue] = {
                "otel.scope.name": scope_spans.scope.name,
                "otel.scope.version": scope_spans.scope.version,
            }
            for span in scope_spans.spans:
                merged = {**resource_attributes, **scope_attributes, **_attributes(span.attributes)}
                span_rows.append((span, merged))

    if not span_rows:
        raise ValueError("OTLP request contains no spans")
    span_rows.sort(key=lambda row: (row[0].start_time_unix_nano, row[0].span_id))

    inferred_sequence_by_trace: dict[str, int] = {}
    events: list[TraceEvent] = []
    for span, attributes in span_rows:
        trace_id = bytes(span.trace_id).hex()
        span_id = bytes(span.span_id).hex()
        if len(trace_id) != 32 or len(span_id) != 16:
            raise ValueError("OTLP spans require non-empty trace_id and span_id")
        explicit_sequence = attributes.get("agenttrace.sequence_number")
        if isinstance(explicit_sequence, int) and explicit_sequence >= 0:
            sequence_number = explicit_sequence
        else:
            sequence_number = inferred_sequence_by_trace.get(trace_id, 0)
            inferred_sequence_by_trace[trace_id] = sequence_number + 1
            attributes["agenttrace.sequence.inferred"] = True

        failed = span.status.code == 2
        status_description: str | None = None
        if failed:
            # Keep the failure on every step type, not only tool interactions.
            attributes["otel.status_code"] = "ERROR"
            status_description = _record_status_description(attributes, span.status.message)
        step_type = _step_type(span, attributes)
        input_value = attributes.get("input.value", attributes.get("openinference.input.value"))
        output_value = attributes.get("output.value", attributes.get("openinference.output.value"))
        model_id = _string(
            attributes,
            "gen_ai.request.model",
            "llm.model_name",
            "model.name",
        )
        provider = _string(attributes, "gen_ai.provider.name", "llm.provider") or "unknown"
        model = (
            ModelInvocation(
                provider=provider,
                model_id=model_id,
                sampling=SamplingConfig(
                    temperature=_number(
                        attributes, "gen_ai.request.temperature", "llm.temperature"
                    ),
                    max_tokens=_integer(
                        attributes,
                        "gen_ai.request.max_tokens",
                        "llm.invocation_parameters.max_tokens",
                    )
                    or None,
                ),
            )
            if model_id
            else None
        )
        tool = None
        if step_type in {StepType.TOOL_CALL, StepType.TOOL_RESULT}:
            tool = ToolInteraction(
                call_id=_string(attributes, "tool.call.id", "tool_call.id"),
                name=_string(attributes, "tool.name") or span.name,
                arguments=_content(input_value),
                result=_content(output_value),
                success=not failed,
                error_type=(status_description or "error") if failed else None,
            )

        experiment_id = _string(attributes, "agenttrace.experiment.id") or "otel-default"
        run_id = _string(attributes, "agenttrace.run.id") or f"run_{trace_id}"
        episode_id = _string(attributes, "agenttrace.episode.id") or f"episode_{trace_id}"
        task_id = _string(attributes, "agenttrace.task.id")
        task_version = _string(attributes, "agenttrace.task.version")
        task = (
            TaskRef(
                task_id=task_id,
                task_version=task_version,
                split=_string(attributes, "agenttrace.task.split"),
                task_instance_id=_string(attributes, "agenttrace.task.instance_id"),
            )
            if task_id and task_version
            else None
        )
        metrics = StepMetrics(
            latency_ms=(span.end_time_unix_nano - span.start_time_unix_nano) / 1_000_000,
            cost_usd=_number(attributes, "llm.cost.total", "gen_ai.usage.cost"),
            tokens=TokenUsage(
                input_tokens=_integer(
                    attributes,
                    "llm.token_count.prompt",
                    "gen_ai.usage.input_tokens",
                ),
                output_tokens=_integer(
                    attributes,
                    "llm.token_count.completion",
                    "gen_ai.usage.output_tokens",
                ),
                cached_input_tokens=_integer(
                    attributes,
                    "llm.token_count.prompt_details.cache_read",
                    "gen_ai.usage.cached_input_tokens",
                ),
                reasoning_tokens=_integer(
                    attributes,
                    "llm.token_count.completion_details.reasoning",
                    "gen_ai.usage.reasoning_tokens",
                ),
            ),
        )
        provenance = Provenance(
            code_commit=_string(attributes, "agenttrace.provenance.code_commit"),
            config_digest=_string(attributes, "agenttrace.provenance.config_digest"),
            prompt_version=_string(attributes, "agenttrace.provenance.prompt_version"),
            dataset_version=_string(attributes, "agenttrace.provenance.dataset_version"),
            generation_engine_version=_string(
                attributes,
                "agenttrace.provenance.generation_engine_version",
            ),
            environment_version=_string(attributes, "agenttrace.provenance.environment_version"),
            benchmark_protocol_version=_string(
                attributes,
                "agenttrace.provenance.benchmark_protocol_version",
            ),
            final_artifact_digest=_string(
                attributes,
                "agenttrace.provenance.final_artifact_digest",
            ),
        )
        events.append(
            TraceEvent(
                event_id=_event_id(trace_id, span_id, span.end_time_unix_nano),
                experiment_id=experiment_id,
                run_id=run_id,
                episode_id=episode_id,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=bytes(span.parent_span_id).hex() or None,
                step_id=_string(attributes, "agenttrace.step.id") or f"step_{span_id}",
                sequence_number=sequence_number,
                step_type=step_type,
                started_at=_timestamp(span.start_time_unix_nano),
                ended_at=_timestamp(span.end_time_unix_nano),
                task=task,
                input=_content(input_value),
                output=_content(output_value),
                model=model,
                tool=tool,
                metrics=metrics,
                provenance=provenance,
                attributes=attributes,
            )
        )
    return EventBatch(worker_id=worker_id, events=events)
