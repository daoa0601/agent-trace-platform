"""Runtime SDK batching, episode ordering, and durable delivery."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agenttrace.models import MAX_EVENT_ATTRIBUTES, EventBatch, StepType, TaskRef
from agenttrace.sdk import TraceClient

from .conftest import make_event


def _success_response(request: httpx.Request) -> httpx.Response:
    batch = EventBatch.model_validate_json(request.content)
    return httpx.Response(
        202,
        json={
            "accepted": len(batch.events),
            "duplicate_count": 0,
            "batch_ref": "object://raw/test",
            "event_ids": [event.event_id for event in batch.events],
        },
    )


def test_client_batches_and_session_assigns_monotonic_sequence(support_task) -> None:
    captured: list[EventBatch] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(EventBatch.model_validate_json(request.content))
        return _success_response(request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = TraceClient(
        "http://collector",
        api_key="secret",
        batch_size=100,
        http_client=http_client,
    )
    with client.session(
        experiment_id="exp_sdk",
        task=TaskRef(task_id=support_task.task_id, task_version=support_task.version),
    ) as trace:
        trace.emit(StepType.USER_INPUT, input={"message": "hello"})
        with trace.step(StepType.MODEL_GENERATION) as step:
            step.output = "answer"
        trace.emit(StepType.FINAL_OUTPUT, output="answer")
    assert len(captured) == 1
    events = captured[0].events
    assert [event.sequence_number for event in events] == list(range(5))
    assert events[0].step_type == StepType.EPISODE_START
    assert events[-1].step_type == StepType.EPISODE_END
    assert all(event.trace_id == events[0].trace_id for event in events)
    assert all(event.parent_span_id == events[0].span_id for event in events[1:])


def test_client_spools_then_replays_failed_batch(tmp_path: Path, support_task) -> None:
    state = {"fail": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            return httpx.Response(503, text="unavailable")
        return _success_response(request)

    client = TraceClient(
        "http://collector",
        max_retries=0,
        spool_dir=tmp_path / "spool",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.record(make_event(1, StepType.USER_INPUT, task=support_task))
    assert client.flush() is None
    assert len(list((tmp_path / "spool").glob("batch-*.json"))) == 1
    state["fail"] = False
    assert client.replay_spool() == (1, 0)


def test_session_emits_error_and_reraises(support_task) -> None:
    captured: list[EventBatch] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(EventBatch.model_validate_json(request.content))
        return _success_response(request)

    client = TraceClient(
        "http://collector",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RuntimeError, match="boom"):
        with client.session(experiment_id="exp_error") as trace:
            with trace.step(StepType.AGENT_STATE):
                raise RuntimeError("boom")
    steps = [event.step_type for event in captured[0].events]
    assert steps == [StepType.EPISODE_START, StepType.ERROR, StepType.EPISODE_END]
    error = captured[0].events[1]
    assert error.output is not None
    assert json.dumps(error.output.inline).find("RuntimeError") >= 0


def test_client_flushes_when_partition_changes(support_task) -> None:
    captured: list[EventBatch] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(EventBatch.model_validate_json(request.content))
        return _success_response(request)

    client = TraceClient(
        "http://collector",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    first = make_event(1, StepType.USER_INPUT, task=support_task)
    second = first.model_copy(update={"event_id": "evt_second", "experiment_id": "exp_second"})
    client.record(first)
    client.record(second)
    client.flush()
    assert [batch.events[0].experiment_id for batch in captured] == ["exp_test", "exp_second"]


def test_client_rejects_oversized_event_without_blocking_later_flushes(support_task) -> None:
    captured: list[EventBatch] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(EventBatch.model_validate_json(request.content))
        return _success_response(request)

    client = TraceClient(
        "http://collector",
        batch_size=100,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    oversized = make_event(
        1,
        StepType.USER_INPUT,
        task=support_task,
        attributes={f"tag.{index}": index for index in range(MAX_EVENT_ATTRIBUTES + 1)},
    )
    with pytest.raises(ValueError, match="attributes"):
        client.record(oversized)
    client.record(make_event(2, StepType.USER_INPUT, task=support_task))
    client.flush()
    assert [event.sequence_number for batch in captured for event in batch.events] == [2]
