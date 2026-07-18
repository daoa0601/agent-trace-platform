"""Canonical, episode, evaluation, and frozen dataset pipelines."""

from __future__ import annotations

import json

import zstandard
from pydantic import JsonValue

from agenttrace.models import (
    Content,
    EnvironmentState,
    EventBatch,
    ModelInvocation,
    Provenance,
    RewardData,
    StepType,
)
from agenttrace.pipelines import (
    DatasetKind,
    build_dataset_snapshot,
    build_episode_snapshot,
    build_evaluation_snapshot,
    normalize_raw_traces,
)
from agenttrace.pipelines.episodes import validate_episode_steps
from agenttrace.pipelines.snapshots import load_manifest, read_parquet
from agenttrace.storage import FileObjectStore, RawBatchWriter
from agenttrace.tasks import TaskRegistry

from .conftest import make_event, make_valid_events, write_valid_batch


def _text(row: dict[str, object], key: str) -> str:
    value = row[key]
    assert isinstance(value, str)
    return value


def _number(row: dict[str, object], key: str) -> float:
    value = row[key]
    assert isinstance(value, int | float) and not isinstance(value, bool)
    return float(value)


def _build_through_evaluation(objects: FileObjectStore, task, *, final_output=None):
    write_valid_batch(objects, task, final_output=final_output)
    canonical = normalize_raw_traces(objects)
    episodes = build_episode_snapshot(objects, canonical_manifest_key=canonical.manifest_key)
    evaluations = build_evaluation_snapshot(
        objects,
        episode_manifest_key=episodes.manifest_key,
        tasks=TaskRegistry([task]),
    )
    return canonical, episodes, evaluations


def test_end_to_end_snapshots_are_valid_reproducible_and_queryable(
    object_store,
    support_task,
) -> None:
    canonical, episodes, evaluations = _build_through_evaluation(object_store, support_task)
    assert canonical.manifest.metrics["valid_event_count"] == 5
    assert canonical.manifest.metrics["episode_count"] == 1
    assert episodes.manifest.metrics["valid_episode_count"] == 1
    assert evaluations.manifest.metrics["evaluation_result_count"] == 3
    assert evaluations.manifest.metrics["episode_pass_rate"] == 1.0
    repeated = normalize_raw_traces(object_store)
    assert repeated.manifest.snapshot_id == canonical.manifest.snapshot_id
    persisted = load_manifest(object_store, canonical.manifest_key)
    assert persisted.snapshot_id == canonical.manifest.snapshot_id
    reward_rows = read_parquet(object_store, evaluations.manifest.tables["episode_rewards"])
    assert _number(reward_rows[0], "reward_total") > 3
    components = json.loads(_text(reward_rows[0], "reward_components_json"))
    assert set(components) == {"latency", "output-schema", "reference-answer"}


def test_dataset_builder_exports_eval_and_sft_rows(object_store, support_task) -> None:
    _, episodes, evaluations = _build_through_evaluation(object_store, support_task)
    registry = TaskRegistry([support_task])
    eval_dataset = build_dataset_snapshot(
        object_store,
        name="support-eval",
        kind=DatasetKind.EVAL,
        episode_manifest_key=episodes.manifest_key,
        evaluation_manifest_key=evaluations.manifest_key,
        tasks=registry,
    )
    sft_dataset = build_dataset_snapshot(
        object_store,
        name="support-sft",
        kind=DatasetKind.SFT,
        episode_manifest_key=episodes.manifest_key,
        evaluation_manifest_key=evaluations.manifest_key,
        tasks=registry,
    )
    eval_rows = read_parquet(object_store, eval_dataset.manifest.tables["data"])
    sft_rows = read_parquet(object_store, sft_dataset.manifest.tables["data"])
    assert len(eval_rows) == len(sft_rows) == 1
    assert json.loads(_text(eval_rows[0], "output_json"))["route"] == "billing"
    assert json.loads(_text(sft_rows[0], "target_response_json"))["intent"] == "billing_refund"
    assert json.loads(_text(sft_rows[0], "messages_json"))[0]["role"] == "user"
    assert eval_dataset.manifest.tables["data"].sha256


def test_training_dataset_honors_task_data_policy(object_store, support_task) -> None:
    blocked_policy = support_task.data_policy.model_copy(update={"allow_training": False})
    blocked_task = support_task.model_copy(update={"data_policy": blocked_policy})
    _, episodes, evaluations = _build_through_evaluation(object_store, blocked_task)
    dataset = build_dataset_snapshot(
        object_store,
        name="blocked-sft",
        kind=DatasetKind.SFT,
        episode_manifest_key=episodes.manifest_key,
        evaluation_manifest_key=evaluations.manifest_key,
        tasks=TaskRegistry([blocked_task]),
    )
    assert dataset.manifest.metrics["row_count"] == 0
    excluded = read_parquet(object_store, dataset.manifest.tables["excluded"])
    assert excluded == [
        {"episode_id": "episode_test", "reason": "task data policy forbids training"}
    ]


def test_preference_dataset_requires_chosen_and_rejected(object_store, support_task) -> None:
    output = {
        "chosen": "clear and grounded answer",
        "rejected": "unsupported answer",
        "preference_source": "human",
        "annotator_confidence": 0.9,
    }
    _, episodes, evaluations = _build_through_evaluation(
        object_store,
        support_task,
        final_output=output,
    )
    dataset = build_dataset_snapshot(
        object_store,
        name="support-preferences",
        kind=DatasetKind.PREFERENCE,
        episode_manifest_key=episodes.manifest_key,
        evaluation_manifest_key=evaluations.manifest_key,
        tasks=TaskRegistry([support_task]),
        include_failed=True,
    )
    rows = read_parquet(object_store, dataset.manifest.tables["data"])
    assert json.loads(_text(rows[0], "chosen_json")) == output["chosen"]
    assert rows[0]["preference_source"] == "human"


def test_strict_rl_dataset_retains_policy_and_transition_fields(object_store, support_task) -> None:
    output: dict[str, JsonValue] = {
        "intent": "billing_refund",
        "urgency": "normal",
        "route": "billing",
    }
    events = [
        make_event(0, StepType.EPISODE_START, task=support_task, attributes={"status": "running"}),
        make_event(1, StepType.USER_INPUT, task=support_task, input={"message": "refund"}),
        make_event(
            2,
            StepType.ENVIRONMENT_OBSERVATION,
            task=support_task,
            environment=EnvironmentState(
                environment_id="support-env",
                environment_version="1.0.0",
                observation=Content(inline={"queue": "unassigned"}),
            ),
        ),
        make_event(
            3,
            StepType.MODEL_GENERATION,
            task=support_task,
            output=output,
            model=ModelInvocation(
                provider="test",
                model_id="policy",
                checkpoint_id="ckpt-10",
                tokenizer_id="tok-v1",
                output_token_ids=Content(inline=[1, 2, 3]),
                behavior_logprobs=Content(inline=[-0.1, -0.2, -0.3]),
            ),
            provenance=Provenance(
                policy_checkpoint_id="ckpt-10",
                tokenizer_version="tok-v1",
                generation_engine_version="engine-v2",
                behavior_policy_id="policy-a",
            ),
        ),
        make_event(
            4,
            StepType.ENVIRONMENT_ACTION,
            task=support_task,
            environment=EnvironmentState(
                environment_id="support-env",
                environment_version="1.0.0",
                action=Content(inline={"route": "billing"}),
                terminated=True,
            ),
        ),
        make_event(
            5,
            StepType.REWARD,
            task=support_task,
            reward=RewardData(
                total=1.0,
                components={"task_success": 1.0},
                reward_function_version="support-rm-v1",
            ),
        ),
        make_event(
            6,
            StepType.ENVIRONMENT_OBSERVATION,
            task=support_task,
            environment=EnvironmentState(
                environment_id="support-env",
                environment_version="1.0.0",
                observation=Content(inline={"queue": "billing"}),
                terminated=True,
            ),
        ),
        make_event(7, StepType.FINAL_OUTPUT, task=support_task, output=output),
        make_event(
            8,
            StepType.EPISODE_END,
            task=support_task,
            output={"status": "succeeded"},
            attributes={"status": "succeeded"},
        ),
    ]
    RawBatchWriter(object_store).write(EventBatch(worker_id="rl", events=events))
    canonical = normalize_raw_traces(object_store)
    episodes = build_episode_snapshot(object_store, canonical_manifest_key=canonical.manifest_key)
    evaluations = build_evaluation_snapshot(
        object_store,
        episode_manifest_key=episodes.manifest_key,
        tasks=TaskRegistry([support_task]),
    )
    dataset = build_dataset_snapshot(
        object_store,
        name="support-rl",
        kind=DatasetKind.RL,
        episode_manifest_key=episodes.manifest_key,
        evaluation_manifest_key=evaluations.manifest_key,
        tasks=TaskRegistry([support_task]),
    )
    rows = read_parquet(object_store, dataset.manifest.tables["data"])
    assert rows[0]["policy_checkpoint_id"] == "ckpt-10"
    assert rows[0]["behavior_policy_id"] == "policy-a"
    transitions = json.loads(_text(rows[0], "transitions_json"))
    assert transitions[0]["reward"] == 1.0
    assert transitions[0]["terminated"] is True


def test_episode_quality_detects_pairing_parent_and_terminal_errors(support_task) -> None:
    from agenttrace.models import ToolInteraction

    steps = [
        make_event(0, StepType.EPISODE_START, task=support_task),
        make_event(
            1,
            StepType.TOOL_CALL,
            task=support_task,
            tool=ToolInteraction(call_id="call-1", name="lookup"),
            parent_span_id="f" * 16,
        ),
    ]
    codes = {issue.code for issue in validate_episode_steps(steps)}
    assert {"missing_parent_span", "missing_tool_result", "missing_terminal_state"} <= codes


def test_normalizer_quarantines_conflicts_and_broken_references(object_store, support_task) -> None:
    events = make_valid_events(support_task)
    RawBatchWriter(object_store).write(EventBatch(worker_id="base", events=events))
    conflict = events[1].model_copy(update={"output": Content(inline="different")})
    compressed = zstandard.ZstdCompressor().compress(
        (conflict.model_dump_json(exclude_none=True) + "\n").encode()
    )
    object_store.put_bytes(
        "raw/collection_date=2026-07-17/experiment_id=exp_test/worker_id=conflict/conflict.jsonl.zst",
        compressed,
    )
    broken = events[2].model_copy(
        update={
            "event_id": "evt_broken_ref",
            "step_id": "step_broken_ref",
            "sequence_number": 20,
            "output": Content(ref="object://blobs/missing.json.zst", sha256="a" * 64),
        }
    )
    RawBatchWriter(object_store).write(EventBatch(worker_id="broken", events=[broken]))
    normalized = normalize_raw_traces(object_store)
    assert normalized.manifest.metrics["conflicting_event_count"] == 1
    assert normalized.manifest.metrics["broken_reference_count"] == 1
    assert normalized.manifest.metrics["quarantined_record_count"] == 3
    quarantine = read_parquet(object_store, normalized.manifest.tables["quarantine"])
    assert any("conflicting duplicate" in str(row["error"]) for row in quarantine)
    assert any("broken object references" in str(row["error"]) for row in quarantine)
