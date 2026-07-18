"""Episode-aware runtime instrumentation with explicit sequence numbers."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType

from pydantic import JsonValue

from agenttrace.ids import new_id, new_span_id, new_trace_id
from agenttrace.models import (
    Content,
    EnvironmentState,
    ModelInvocation,
    Privacy,
    Provenance,
    RewardData,
    RunStatus,
    StepMetrics,
    StepType,
    TaskRef,
    ToolInteraction,
    TraceEvent,
    utc_now,
)
from agenttrace.sdk.client import TraceClient


def _content(value: JsonValue | Content | None) -> Content | None:
    if value is None or isinstance(value, Content):
        return value
    return Content(inline=value)


@dataclass
class StepHandle:
    """Mutable result holder yielded by TraceSession.step()."""

    output: JsonValue | Content | None = None
    model: ModelInvocation | None = None
    tool: ToolInteraction | None = None
    environment: EnvironmentState | None = None
    reward: RewardData | None = None
    metrics: StepMetrics | None = None


class TraceSession:
    """Build correctly linked events for one agent episode."""

    def __init__(
        self,
        *,
        client: TraceClient,
        experiment_id: str,
        run_id: str | None = None,
        episode_id: str | None = None,
        task: TaskRef | None = None,
        provenance: Provenance | None = None,
        privacy: Privacy | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.client = client
        self.experiment_id = experiment_id
        self.run_id = run_id or new_id("run")
        self.episode_id = episode_id or new_id("episode")
        self.task = task
        self.provenance = provenance or Provenance()
        self.privacy = privacy or Privacy()
        self.trace_id = trace_id or new_trace_id()
        self.root_span_id = new_span_id()
        self._sequence = 0
        self._lock = threading.Lock()
        self._entered = False

    def _next_sequence(self) -> int:
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
            return sequence

    def emit(
        self,
        step_type: StepType,
        *,
        input: JsonValue | Content | None = None,
        output: JsonValue | Content | None = None,
        model: ModelInvocation | None = None,
        tool: ToolInteraction | None = None,
        environment: EnvironmentState | None = None,
        reward: RewardData | None = None,
        metrics: StepMetrics | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, JsonValue] | None = None,
    ) -> TraceEvent:
        """Create and record one event with monotonic episode ordering."""
        started_at = started_at or utc_now()
        event = TraceEvent(
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            episode_id=self.episode_id,
            trace_id=self.trace_id,
            span_id=span_id or new_span_id(),
            parent_span_id=parent_span_id if parent_span_id is not None else self.root_span_id,
            sequence_number=self._next_sequence(),
            step_type=step_type,
            started_at=started_at,
            ended_at=ended_at or started_at,
            task=self.task,
            input=_content(input),
            output=_content(output),
            model=model,
            tool=tool,
            environment=environment,
            reward=reward,
            metrics=metrics or StepMetrics(),
            provenance=self.provenance,
            privacy=self.privacy,
            attributes=attributes or {},
        )
        self.client.record(event)
        return event

    @contextmanager
    def step(
        self,
        step_type: StepType,
        *,
        input: JsonValue | Content | None = None,
        attributes: dict[str, JsonValue] | None = None,
    ) -> Iterator[StepHandle]:
        """Measure a step and emit either its result or a structured error."""
        started_at = utc_now()
        span_id = new_span_id()
        handle = StepHandle()
        try:
            yield handle
        except Exception as error:
            ended_at = utc_now()
            self.emit(
                StepType.ERROR,
                input=input,
                output={"error_type": type(error).__name__, "message": str(error)},
                started_at=started_at,
                ended_at=ended_at,
                span_id=span_id,
                metrics=StepMetrics(latency_ms=(ended_at - started_at).total_seconds() * 1000),
                attributes=attributes,
            )
            raise
        ended_at = utc_now()
        self.emit(
            step_type,
            input=input,
            output=handle.output,
            model=handle.model,
            tool=handle.tool,
            environment=handle.environment,
            reward=handle.reward,
            metrics=handle.metrics
            or StepMetrics(latency_ms=(ended_at - started_at).total_seconds() * 1000),
            started_at=started_at,
            ended_at=ended_at,
            span_id=span_id,
            attributes=attributes,
        )

    def __enter__(self) -> TraceSession:
        if self._entered:
            raise RuntimeError("TraceSession cannot be entered twice")
        self._entered = True
        self.emit(
            StepType.EPISODE_START,
            span_id=self.root_span_id,
            parent_span_id=None,
            attributes={"status": RunStatus.RUNNING.value},
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del traceback
        status = RunStatus.SUCCEEDED if exc_type is None else RunStatus.FAILED
        output: JsonValue = {"status": status.value}
        if exc is not None:
            output = {"status": status.value, "error_type": type(exc).__name__, "message": str(exc)}
        self.emit(StepType.EPISODE_END, output=output, attributes={"status": status.value})
        self.client.flush()
