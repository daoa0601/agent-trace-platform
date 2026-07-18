"""Trace and task contract validation."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from agenttrace.models import EventBatch, RewardData, StepType, TraceEvent
from agenttrace.settings import CollectorSettings
from agenttrace.tasks import TaskRegistry
from agenttrace.tasks.models import EnvironmentKind, EnvironmentSpec

from .conftest import make_event


def test_trace_event_rejects_naive_timestamp(support_task) -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_event(
            1,
            StepType.USER_INPUT,
            task=support_task,
            started_at=datetime(2026, 1, 1),
        )


def test_trace_event_requires_tool_and_reward_shapes(support_task) -> None:
    with pytest.raises(ValidationError, match="tool steps must include"):
        make_event(1, StepType.TOOL_CALL, task=support_task)
    with pytest.raises(ValidationError, match="reward steps must include"):
        make_event(1, StepType.REWARD, task=support_task)
    with pytest.raises(ValidationError, match="finite"):
        RewardData(total=float("nan"), reward_function_version="bad")


def test_event_batch_rejects_duplicate_and_cross_partition_events(support_task) -> None:
    event = make_event(1, StepType.USER_INPUT, task=support_task)
    with pytest.raises(ValidationError, match="unique within a batch"):
        EventBatch(events=[event, event])
    other = event.model_copy(update={"event_id": "evt_other", "experiment_id": "exp_other"})
    with pytest.raises(ValidationError, match="one experiment"):
        EventBatch(events=[event, other])


def test_task_registry_rejects_duplicates_and_unknown_tasks(support_task) -> None:
    with pytest.raises(ValueError, match="duplicate task"):
        TaskRegistry([support_task, support_task])
    registry = TaskRegistry([support_task])
    assert registry.get(support_task.task_id, support_task.version) == support_task
    with pytest.raises(KeyError, match="unknown task"):
        registry.get("missing", "1.0.0")


def test_environment_and_production_settings_fail_closed() -> None:
    with pytest.raises(ValidationError, match="require an image"):
        EnvironmentSpec(kind=EnvironmentKind.DOCKER)
    with pytest.raises(ValidationError, match="network_name"):
        EnvironmentSpec(kind=EnvironmentKind.NONE, network="restricted")
    with pytest.raises(ValidationError, match="requires at least one"):
        CollectorSettings(environment="production", api_keys="")
    settings = CollectorSettings(environment="production", api_keys="one, two")
    assert settings.parsed_api_keys == ("one", "two")


def test_model_rejects_unknown_fields(support_task) -> None:
    event = make_event(1, StepType.USER_INPUT, task=support_task)
    payload = event.model_dump()
    payload["surprise"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        TraceEvent.model_validate(payload)
