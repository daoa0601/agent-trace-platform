"""Manual OpenInference spans for arbitrary agent and evaluation runtimes.

The bridge intentionally uses only OpenTelemetry primitives. It therefore works with
framework-specific instrumentors as well as with agents that have no supported SDK.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from agenttrace.ids import new_id
from agenttrace.redaction import Redactor


class CaptureMode(StrEnum):
    """How much request/response content is placed on exported spans."""

    NONE = "none"
    METADATA = "metadata"
    FULL = "full"


class TelemetrySettings(BaseSettings):
    """Runtime telemetry configuration loaded from ``AGENTTRACE_*`` variables."""

    model_config = SettingsConfigDict(env_prefix="AGENTTRACE_", env_file=".env", extra="ignore")

    telemetry_enabled: bool = True
    otlp_endpoint: str = "http://127.0.0.1:4318/v1/traces"
    otlp_headers: SecretStr | None = None
    service_name: str = "agent-runtime"
    service_version: str | None = None
    deployment_environment: str = "development"
    capture_content: CaptureMode = CaptureMode.METADATA
    redaction_mode: str = "balanced"

    @property
    def trace_endpoint(self) -> str:
        endpoint = self.otlp_endpoint.rstrip("/")
        return endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"

    def parsed_headers(self) -> dict[str, str]:
        """Parse OTel's comma-separated ``key=value`` header syntax without exposing it."""
        if self.otlp_headers is None:
            return {}
        headers: dict[str, str] = {}
        for item in self.otlp_headers.get_secret_value().split(","):
            key, separator, value = item.partition("=")
            if not separator or not key.strip():
                raise ValueError("AGENTTRACE_OTLP_HEADERS entries must use key=value syntax")
            headers[key.strip()] = value.strip()
        return headers


@dataclass
class SpanResult:
    """Mutable result information populated while an operation span is active."""

    output: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None


_CURRENT_RUN: ContextVar[AgentRun | None] = ContextVar("agenttrace_current_run", default=None)


def current_run() -> AgentRun | None:
    """Return the current run for logging and compatibility emitters."""
    return _CURRENT_RUN.get()


def current_trace_fields() -> dict[str, str]:
    """Return correlation identifiers suitable for structured log records."""
    run = current_run()
    fields: dict[str, str] = {}
    if run is not None:
        fields.update(
            experiment_id=run.experiment_id,
            run_id=run.run_id,
            episode_id=run.episode_id,
        )
    context = trace.get_current_span().get_span_context()
    if context.is_valid:
        fields["trace_id"] = trace.format_trace_id(context.trace_id)
        fields["span_id"] = trace.format_span_id(context.span_id)
    return fields


def _json_value(value: Any) -> Any:
    """Convert an arbitrary value to JSON-compatible data deterministically."""
    return json.loads(json.dumps(value, default=str, sort_keys=True))


def _serialized(value: Any) -> str:
    return json.dumps(_json_value(value), separators=(",", ":"), sort_keys=True)


def _content_metadata(prefix: str, value: Any) -> dict[str, str | int]:
    encoded = _serialized(value).encode("utf-8")
    return {
        f"agenttrace.{prefix}.sha256": hashlib.sha256(encoded).hexdigest(),
        f"agenttrace.{prefix}.bytes": len(encoded),
    }


def _otel_value(value: Any) -> str | bool | int | float | list[str]:
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return cast(list[str], value)
    return _serialized(value)


def _context_value(value: str | None, *, name: str, ascii_only: bool = False) -> str | None:
    if value is None:
        return None
    if value != value.strip() or not value or len(value) > 200:
        raise ValueError(f"{name} must contain 1 to 200 characters without outer whitespace")
    if ascii_only and not value.isascii():
        raise ValueError(f"{name} must contain printable ASCII characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must contain printable ASCII characters")
    return value


class TelemetryRuntime:
    """Own an isolated tracer provider and its exporter lifecycle."""

    def __init__(
        self,
        *,
        settings: TelemetrySettings,
        tracer: Tracer,
        provider: TracerProvider | None,
    ) -> None:
        self.settings = settings
        self.tracer = tracer
        self._provider = provider
        self._closed = False

    def run(
        self,
        *,
        experiment_id: str,
        session_id: str | None = None,
        trace_name: str | None = None,
        run_id: str | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        task_version: str | None = None,
        task_split: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AgentRun:
        return AgentRun(
            runtime=self,
            experiment_id=experiment_id,
            session_id=_context_value(session_id, name="session_id", ascii_only=True),
            trace_name=_context_value(trace_name, name="trace_name"),
            run_id=run_id or new_id("run"),
            episode_id=episode_id or new_id("episode"),
            task_id=task_id,
            task_version=task_version,
            task_split=task_split,
            attributes=dict(attributes or {}),
        )

    def force_flush(self, timeout_millis: int = 10_000) -> bool:
        if self._provider is None or self._closed:
            return True
        return bool(self._provider.force_flush(timeout_millis=timeout_millis))

    def close(self) -> None:
        if self._provider is not None and not self._closed:
            self._provider.shutdown()
        self._closed = True

    def __enter__(self) -> TelemetryRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class AgentRun:
    """One correlated agent/evaluation episode represented as OpenInference spans."""

    def __init__(
        self,
        *,
        runtime: TelemetryRuntime,
        experiment_id: str,
        session_id: str | None,
        trace_name: str | None,
        run_id: str,
        episode_id: str,
        task_id: str | None,
        task_version: str | None,
        task_split: str | None,
        attributes: dict[str, Any],
    ) -> None:
        self.runtime = runtime
        self.experiment_id = experiment_id
        self.session_id = session_id
        self.trace_name = trace_name
        self.run_id = run_id
        self.episode_id = episode_id
        self.task_id = task_id
        self.task_version = task_version
        self.task_split = task_split
        self.attributes = attributes
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._root: Span | None = None
        self._root_scope: AbstractContextManager[Span] | None = None
        self._context_token: Token[AgentRun | None] | None = None
        self._redactor = Redactor(cast(Any, runtime.settings.redaction_mode))

    @property
    def trace_id(self) -> str | None:
        if self._root is None:
            return None
        context = self._root.get_span_context()
        return trace.format_trace_id(context.trace_id) if context.is_valid else None

    def _next_sequence(self) -> int:
        with self._sequence_lock:
            sequence = self._sequence
            self._sequence += 1
            return sequence

    def _common_attributes(self, step_type: str, kind: str) -> dict[str, Any]:
        values: dict[str, Any] = {
            "openinference.span.kind": kind,
            "agenttrace.experiment.id": self.experiment_id,
            "agenttrace.run.id": self.run_id,
            "agenttrace.episode.id": self.episode_id,
            "agenttrace.sequence_number": self._next_sequence(),
            "agenttrace.step.type": step_type,
        }
        if self.task_id:
            values["agenttrace.task.id"] = self.task_id
        if self.task_version:
            values["agenttrace.task.version"] = self.task_version
        if self.task_split:
            values["agenttrace.task.split"] = self.task_split
        if self.session_id:
            values["session.id"] = self.session_id
        return values

    def __enter__(self) -> AgentRun:
        if self._root is not None:
            raise RuntimeError("AgentRun cannot be entered twice")
        attributes = self._common_attributes("episode_start", "AGENT")
        attributes.update({key: _otel_value(value) for key, value in self.attributes.items()})
        self._root = self.runtime.tracer.start_span(
            self.trace_name or "agent.episode",
            attributes=attributes,
        )
        self._root_scope = trace.use_span(self._root, end_on_exit=False)
        self._root_scope.__enter__()
        self._context_token = _CURRENT_RUN.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del traceback
        if self._root is None or self._root_scope is None or self._context_token is None:
            return
        status = "succeeded" if exc_type is None else "failed"
        with self.runtime.tracer.start_as_current_span(
            "agent.episode.end",
            attributes={
                **self._common_attributes("episode_end", "AGENT"),
                "agenttrace.run.status": status,
                "output.value": _serialized({"status": status}),
            },
        ) as terminal:
            terminal.set_status(Status(StatusCode.OK if exc_type is None else StatusCode.ERROR))
            if exc is not None:
                terminal.record_exception(exc)
        if exc is not None:
            self._root.record_exception(exc)
            self._root.set_status(Status(StatusCode.ERROR, type(exc).__name__))
        else:
            self._root.set_status(Status(StatusCode.OK))
        _CURRENT_RUN.reset(self._context_token)
        self._root_scope.__exit__(exc_type, exc, None)
        self._root.end()
        self.runtime.force_flush()

    def _capture(self, span: Span, prefix: str, value: Any) -> None:
        if value is None or self.runtime.settings.capture_content == CaptureMode.NONE:
            return
        cleaned, _ = self._redactor.redact(cast(Any, _json_value(value)))
        if self.runtime.settings.capture_content == CaptureMode.FULL:
            span.set_attribute(f"{prefix}.value", _serialized(cleaned))
            span.set_attribute(f"{prefix}.mime_type", "application/json")
        else:
            for key, metadata_value in _content_metadata(prefix, cleaned).items():
                span.set_attribute(key, metadata_value)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        step_type: str = "agent_state",
        kind: str = "CHAIN",
        input: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[SpanResult]:
        """Trace a duration and attach structured output/usage on successful exit."""
        initial = self._common_attributes(step_type, kind)
        initial.update({key: _otel_value(value) for key, value in (attributes or {}).items()})
        with self.runtime.tracer.start_as_current_span(name, attributes=initial) as span:
            self._capture(span, "input", input)
            result = SpanResult()
            try:
                yield result
            except BaseException as error:
                span.set_attribute("agenttrace.step.type", "error")
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                raise
            self._capture(span, "output", result.output)
            for key, value in result.attributes.items():
                span.set_attribute(key, _otel_value(value))
            usage = {
                "llm.token_count.prompt": result.input_tokens,
                "llm.token_count.completion": result.output_tokens,
                "llm.token_count.prompt_details.cache_read": result.cached_input_tokens,
                "llm.token_count.completion_details.reasoning": result.reasoning_tokens,
                "llm.cost.total": result.cost_usd,
            }
            for key, value in usage.items():
                if value is not None:
                    span.set_attribute(key, value)
            span.set_status(Status(StatusCode.OK))

    def benchmark_stage(
        self,
        stage: str,
        *,
        input: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[SpanResult]:
        """Trace a compile, check, benchmark, or independent audit re-test stage."""
        allowed = {"compile", "check", "benchmark", "audit_retest"}
        if stage not in allowed:
            raise ValueError(f"unsupported benchmark stage: {stage}")
        return self.span(
            f"benchmark.{stage}",
            step_type=stage,
            kind="TOOL" if stage in {"compile", "check"} else "CHAIN",
            input=input,
            attributes=attributes,
        )

    def emit_legacy(self, *, event: str, path: str, data: Mapping[str, Any]) -> None:
        """Convert an instantaneous legacy eval event into a correlated span."""
        lowered = event.casefold()
        if "tool" in lowered:
            kind, step_type = "TOOL", "tool_call"
        elif "score" in lowered or "eval" in lowered and "end" in lowered:
            kind, step_type = "EVALUATOR", "evaluator_result"
        elif "model" in lowered or "inference" in lowered:
            kind, step_type = "LLM", "model_generation"
        else:
            kind, step_type = "CHAIN", "agent_state"
        with self.span(
            f"legacy.{event}",
            kind=kind,
            step_type=step_type,
            input=dict(data),
            attributes={"agenttrace.legacy.event": event, "code.namespace": path},
        ):
            pass


def create_telemetry(
    settings: TelemetrySettings | None = None,
    *,
    span_exporter: SpanExporter | None = None,
) -> TelemetryRuntime:
    """Create an isolated telemetry runtime; callers own and must close it."""
    settings = settings or TelemetrySettings()
    if not settings.telemetry_enabled:
        tracer = trace.NoOpTracerProvider().get_tracer("agenttrace.integrations")
        return TelemetryRuntime(settings=settings, tracer=tracer, provider=None)

    if span_exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        span_exporter = OTLPSpanExporter(
            endpoint=settings.trace_endpoint,
            headers=settings.parsed_headers(),
        )
    resource_values: dict[str, str] = {
        "service.name": settings.service_name,
        "deployment.environment.name": settings.deployment_environment,
    }
    if settings.service_version:
        resource_values["service.version"] = settings.service_version
    provider = TracerProvider(resource=Resource.create(resource_values))
    processor = (
        SimpleSpanProcessor(span_exporter)
        if span_exporter.__class__.__module__.startswith("opentelemetry.sdk.trace.export.in_memory")
        else BatchSpanProcessor(span_exporter)
    )
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("agenttrace.integrations", "0.1.0")
    return TelemetryRuntime(settings=settings, tracer=tracer, provider=provider)
