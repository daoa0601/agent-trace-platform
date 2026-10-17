"""Shared deterministic fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from agenttrace.models import (
    Content,
    EventBatch,
    ModelInvocation,
    StepMetrics,
    StepType,
    TaskRef,
    TraceEvent,
)
from agenttrace.storage import FileObjectStore, RawBatchWriter
from agenttrace.tasks import TaskDefinition, load_task

START = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
TRACE_ID = "1" * 32
ROOT_SPAN_ID = "2" * 16


@pytest.fixture
def support_task() -> TaskDefinition:
    root = Path(__file__).resolve().parents[1]
    return load_task(root / "taskpacks/examples/support-routing.yaml")


@pytest.fixture
def object_store(tmp_path: Path) -> FileObjectStore:
    return FileObjectStore(tmp_path / "objects")


def make_event(
    sequence: int,
    step_type: StepType,
    *,
    task: TaskDefinition,
    output: JsonValue | None = None,
    input: JsonValue | None = None,
    **updates: Any,
) -> TraceEvent:
    task_ref = TaskRef(task_id=task.task_id, task_version=task.version, split="test")
    fields: dict[str, Any] = {
        "event_id": f"evt_{sequence:04d}_fixture",
        "experiment_id": "exp_test",
        "run_id": "run_test",
        "episode_id": "episode_test",
        "trace_id": TRACE_ID,
        "span_id": ROOT_SPAN_ID if sequence == 0 else f"{sequence + 2:016x}",
        "parent_span_id": None if sequence == 0 else ROOT_SPAN_ID,
        "step_id": f"step_{sequence:04d}_fixture",
        "sequence_number": sequence,
        "step_type": step_type,
        "started_at": START + timedelta(seconds=sequence),
        "ended_at": START + timedelta(seconds=sequence, milliseconds=10),
        "task": task_ref,
        "input": Content(inline=input) if input is not None else None,
        "output": Content(inline=output) if output is not None else None,
        "metrics": StepMetrics(latency_ms=10),
    }
    fields.update(updates)
    return TraceEvent.model_validate(fields)


def make_valid_events(
    task: TaskDefinition,
    *,
    final_output: JsonValue | None = None,
) -> list[TraceEvent]:
    output = final_output or {
        "intent": "billing_refund",
        "urgency": "normal",
        "route": "billing",
    }
    return [
        make_event(
            0,
            StepType.EPISODE_START,
            task=task,
            attributes={"status": "running"},
        ),
        make_event(
            1,
            StepType.USER_INPUT,
            task=task,
            input={"message": "I was charged twice."},
        ),
        make_event(
            2,
            StepType.MODEL_GENERATION,
            task=task,
            output=output,
            model=ModelInvocation(provider="test", model_id="policy"),
        ),
        make_event(3, StepType.FINAL_OUTPUT, task=task, output=output),
        make_event(
            4,
            StepType.EPISODE_END,
            task=task,
            output={"status": "succeeded"},
            attributes={"status": "succeeded"},
        ),
    ]


def write_valid_batch(
    objects: FileObjectStore,
    task: TaskDefinition,
    *,
    final_output: JsonValue | None = None,
) -> list[TraceEvent]:
    events = make_valid_events(task, final_output=final_output)
    RawBatchWriter(objects, inline_payload_bytes=256).write(
        EventBatch(worker_id="pytest", events=events)
    )
    return events
