"""Domain-neutral built-in evaluator behavior."""

from __future__ import annotations

from agenttrace.evaluators import evaluate_episode
from agenttrace.models import Episode, RunStatus, StepType, ToolInteraction
from agenttrace.storage import BlobStore, FileObjectStore
from agenttrace.tasks.models import EvaluatorSpec

from .conftest import START, make_event, make_valid_events


def _episode(task, steps) -> Episode:
    return Episode(
        experiment_id="exp_test",
        run_id="run_test",
        episode_id="episode_test",
        task=steps[0].task,
        status=RunStatus.SUCCEEDED,
        started_at=START,
        ended_at=steps[-1].ended_at,
        steps=steps,
    )


def test_contains_regex_and_tool_success_evaluators(tmp_path, support_task) -> None:
    specs = [
        EvaluatorSpec(
            evaluator_id="contains-billing",
            kind="contains",
            version="1.0.0",
            config={"values": ["billing", "refund"]},
        ),
        EvaluatorSpec(
            evaluator_id="regex-route",
            kind="regex",
            version="1.0.0",
            config={"pattern": "billing"},
        ),
        EvaluatorSpec(
            evaluator_id="tools",
            kind="tool_success",
            version="1.0.0",
            config={"minimum_success_rate": 0.5},
        ),
    ]
    task = support_task.model_copy(update={"evaluators": specs})
    steps = make_valid_events(task)
    steps.insert(
        -1,
        make_event(
            4,
            StepType.TOOL_RESULT,
            task=task,
            tool=ToolInteraction(call_id="call", name="lookup", success=True),
        ),
    )
    steps[-1] = steps[-1].model_copy(update={"sequence_number": 5})
    results = evaluate_episode(
        _episode(task, steps),
        task,
        BlobStore(FileObjectStore(tmp_path / "objects")),
    )
    assert [result.passed for result in results] == [True, True, True]
    assert results[2].score == 1.0


def test_json_schema_reports_invalid_json(tmp_path, support_task) -> None:
    spec = EvaluatorSpec(
        evaluator_id="schema",
        kind="json_schema",
        version="1.0.0",
    )
    task = support_task.model_copy(update={"evaluators": [spec]})
    steps = make_valid_events(task, final_output="not-json")
    result = evaluate_episode(
        _episode(task, steps),
        task,
        BlobStore(FileObjectStore(tmp_path / "objects")),
    )[0]
    assert result.passed is False
    assert "invalid JSON" in str(result.details["validation_error"])
