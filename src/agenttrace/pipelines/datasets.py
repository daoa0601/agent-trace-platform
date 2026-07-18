"""Frozen eval, SFT, preference, and RL dataset exports."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from pydantic import JsonValue

from agenttrace.models import Content, Episode, StepType, TraceEvent
from agenttrace.pipelines.snapshots import (
    SnapshotManifest,
    digest_bytes,
    load_manifest,
    make_snapshot_id,
    now_utc,
    put_manifest,
    put_parquet,
    read_parquet,
)
from agenttrace.storage.blobs import BlobStore
from agenttrace.storage.objects import ObjectStore
from agenttrace.tasks.loader import TaskRegistry

DATASET_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


class DatasetKind(StrEnum):
    EVAL = "eval"
    SFT = "sft"
    PREFERENCE = "preference"
    RL = "rl"


@dataclass(frozen=True)
class DatasetBuildResult:
    manifest_key: str
    manifest: SnapshotManifest


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _resolve(blobs: BlobStore, content: Content | None) -> JsonValue:
    return blobs.get_json(content) if content is not None else None


def _last_event(episode: Episode, kind: StepType) -> TraceEvent | None:
    return next((event for event in reversed(episode.steps) if event.step_type == kind), None)


def _episode_input(episode: Episode, blobs: BlobStore) -> JsonValue:
    event = next((event for event in episode.steps if event.step_type == StepType.USER_INPUT), None)
    if event is None:
        return None
    return _resolve(blobs, event.input or event.output)


def _episode_output(episode: Episode, blobs: BlobStore) -> JsonValue:
    event = _last_event(episode, StepType.FINAL_OUTPUT)
    if event is None:
        event = next(
            (
                candidate
                for candidate in reversed(episode.steps)
                if candidate.step_type == StepType.MODEL_GENERATION
            ),
            None,
        )
    return _resolve(blobs, event.output) if event else None


def _messages(episode: Episode, blobs: BlobStore) -> list[dict[str, JsonValue]]:
    messages: list[dict[str, JsonValue]] = []
    for event in episode.steps:
        if event.step_type == StepType.SYSTEM_MESSAGE:
            messages.append(
                {"role": "system", "content": _resolve(blobs, event.output or event.input)}
            )
        elif event.step_type == StepType.USER_INPUT:
            messages.append(
                {"role": "user", "content": _resolve(blobs, event.input or event.output)}
            )
        elif event.step_type == StepType.MODEL_GENERATION:
            messages.append({"role": "assistant", "content": _resolve(blobs, event.output)})
        elif event.step_type == StepType.TOOL_CALL and event.tool:
            messages.append(
                {
                    "role": "assistant",
                    "tool_call": {
                        "id": event.tool.call_id,
                        "name": event.tool.name,
                        "arguments": _resolve(blobs, event.tool.arguments),
                    },
                }
            )
        elif event.step_type == StepType.TOOL_RESULT and event.tool:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event.tool.call_id,
                    "content": _resolve(blobs, event.tool.result),
                }
            )
    return messages


def _rl_payload(
    episode: Episode, blobs: BlobStore, *, strict: bool
) -> tuple[dict[str, object] | None, str | None]:
    generations = [event for event in episode.steps if event.step_type == StepType.MODEL_GENERATION]
    required_provenance = {
        "policy_checkpoint_id": next(
            (
                event.provenance.policy_checkpoint_id
                for event in generations
                if event.provenance.policy_checkpoint_id
            ),
            None,
        ),
        "tokenizer_version": next(
            (
                event.provenance.tokenizer_version
                for event in generations
                if event.provenance.tokenizer_version
            ),
            None,
        ),
        "generation_engine_version": next(
            (
                event.provenance.generation_engine_version
                for event in generations
                if event.provenance.generation_engine_version
            ),
            None,
        ),
    }
    missing = [name for name, value in required_provenance.items() if value is None]
    if strict and missing:
        return None, f"missing RL provenance: {missing}"
    if strict and any(
        event.model is None or event.model.behavior_logprobs is None for event in generations
    ):
        return None, "missing behavior log probabilities"

    transitions: list[dict[str, object]] = []
    action_indexes = [
        index
        for index, event in enumerate(episode.steps)
        if event.step_type == StepType.ENVIRONMENT_ACTION
    ]
    for action_position, index in enumerate(action_indexes):
        action_event = episode.steps[index]
        next_index = (
            action_indexes[action_position + 1]
            if action_position + 1 < len(action_indexes)
            else len(episode.steps)
        )
        before = next(
            (
                event
                for event in reversed(episode.steps[:index])
                if event.step_type == StepType.ENVIRONMENT_OBSERVATION
            ),
            None,
        )
        after = next(
            (
                event
                for event in episode.steps[index + 1 : next_index + 1]
                if event.step_type == StepType.ENVIRONMENT_OBSERVATION
            ),
            None,
        )
        reward = sum(
            event.reward.total
            for event in episode.steps[index:next_index]
            if event.reward is not None
        )
        environment = action_event.environment
        transitions.append(
            {
                "observation": _resolve(
                    blobs,
                    before.environment.observation
                    if before and before.environment
                    else before.output
                    if before
                    else None,
                ),
                "action": _resolve(
                    blobs,
                    environment.action if environment else action_event.output,
                ),
                "reward": reward,
                "next_observation": _resolve(
                    blobs,
                    after.environment.observation
                    if after and after.environment
                    else after.output
                    if after
                    else None,
                ),
                "terminated": bool(environment.terminated) if environment else False,
                "truncated": bool(environment.truncated) if environment else False,
            }
        )

    model_actions = []
    for event in generations:
        model = event.model
        if model is None:
            if strict:
                return None, "model generation is missing model metadata"
            model_actions.append({"output": _resolve(blobs, event.output)})
            continue
        model_actions.append(
            {
                "output": _resolve(blobs, event.output),
                "token_ids": _resolve(blobs, model.output_token_ids),
                "behavior_logprobs": _resolve(blobs, model.behavior_logprobs),
                "reference_logprobs": _resolve(blobs, model.reference_logprobs),
                "value_estimates": _resolve(blobs, model.value_estimates),
                "sampling": model.sampling.model_dump(mode="json"),
            }
        )
    return {
        "episode_id": episode.episode_id,
        "transitions_json": json.dumps(transitions, sort_keys=True),
        "model_actions_json": json.dumps(model_actions, sort_keys=True),
        "policy_checkpoint_id": required_provenance["policy_checkpoint_id"],
        "tokenizer_version": required_provenance["tokenizer_version"],
        "generation_engine_version": required_provenance["generation_engine_version"],
        "behavior_policy_id": next(
            (
                event.provenance.behavior_policy_id
                for event in generations
                if event.provenance.behavior_policy_id
            ),
            None,
        ),
    }, None


def build_dataset_snapshot(
    objects: ObjectStore,
    *,
    name: str,
    kind: DatasetKind,
    episode_manifest_key: str,
    evaluation_manifest_key: str,
    tasks: TaskRegistry,
    minimum_reward: float | None = None,
    include_failed: bool = False,
    strict_rl: bool = True,
) -> DatasetBuildResult:
    """Create an immutable, provenance-complete consumer dataset."""
    if not DATASET_NAME.fullmatch(name):
        raise ValueError("dataset name must be a lowercase slug")
    episode_bytes = objects.get_bytes(episode_manifest_key)
    evaluation_bytes = objects.get_bytes(evaluation_manifest_key)
    episode_manifest = load_manifest(objects, episode_manifest_key)
    evaluation_manifest = load_manifest(objects, evaluation_manifest_key)
    if episode_manifest.kind != "episodes" or evaluation_manifest.kind != "evaluations":
        raise ValueError("dataset builder requires episode and evaluation snapshots")
    if evaluation_manifest.config.get("episode_snapshot_id") != episode_manifest.snapshot_id:
        raise ValueError("evaluation snapshot was built from a different episode snapshot")

    episode_rows = read_parquet(objects, episode_manifest.tables["episodes"])
    reward_rows = read_parquet(objects, evaluation_manifest.tables["episode_rewards"])
    result_rows = read_parquet(objects, evaluation_manifest.tables["evaluation_results"])
    rewards = {str(row["episode_id"]): row for row in reward_rows}
    evaluations: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in result_rows:
        evaluations[str(row["episode_id"])].append(row)

    blobs = BlobStore(objects)
    output_rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    task_digests: dict[str, str] = {}
    for row in episode_rows:
        episode_json = row.get("episode_json")
        if not isinstance(episode_json, str):
            continue
        episode = Episode.model_validate_json(episode_json)
        reward = rewards.get(episode.episode_id)
        reason: str | None = None
        task = None
        if episode.task is None:
            reason = "missing task reference"
        else:
            try:
                task = tasks.get(episode.task.task_id, episode.task.task_version)
            except KeyError as error:
                reason = str(error)
        if reward is None:
            reason = reason or "missing evaluation reward"
        elif not include_failed and not bool(reward["passed"]):
            reason = "required evaluator failed"
        elif (
            minimum_reward is not None
            and _as_float(reward["reward_total"], "reward_total") < minimum_reward
        ):
            reason = "reward below threshold"
        if kind != DatasetKind.EVAL and task is not None and not task.data_policy.allow_training:
            reason = "task data policy forbids training"
        if (
            kind != DatasetKind.EVAL
            and task is not None
            and not task.data_policy.allow_private_reasoning
            and any(event.privacy.contains_private_reasoning for event in episode.steps)
        ):
            reason = "episode is marked as containing private reasoning"
        if reason:
            excluded.append({"episode_id": episode.episode_id, "reason": reason})
            continue
        assert task is not None and reward is not None
        task_key = f"{task.task_id}@{task.version}"
        task_digests[task_key] = hashlib.sha256(task.model_dump_json().encode()).hexdigest()
        common: dict[str, object] = {
            "episode_id": episode.episode_id,
            "experiment_id": episode.experiment_id,
            "run_id": episode.run_id,
            "task_id": task.task_id,
            "task_version": task.version,
            "reward_total": reward["reward_total"],
            "reward_components_json": reward["reward_components_json"],
            "reward_function_version": reward["reward_function_version"],
            "source_episode_snapshot": episode_manifest.snapshot_id,
            "source_evaluation_snapshot": evaluation_manifest.snapshot_id,
        }
        if kind == DatasetKind.EVAL:
            common.update(
                {
                    "input_json": json.dumps(_episode_input(episode, blobs), sort_keys=True),
                    "output_json": json.dumps(_episode_output(episode, blobs), sort_keys=True),
                    "evaluations_json": json.dumps(
                        evaluations.get(episode.episode_id, []),
                        sort_keys=True,
                        default=str,
                    ),
                }
            )
        elif kind == DatasetKind.SFT:
            common.update(
                {
                    "messages_json": json.dumps(_messages(episode, blobs), sort_keys=True),
                    "target_response_json": json.dumps(
                        _episode_output(episode, blobs), sort_keys=True
                    ),
                    "weight": max(0.0, _as_float(reward["reward_total"], "reward_total")),
                }
            )
        elif kind == DatasetKind.PREFERENCE:
            payload = _episode_output(episode, blobs)
            if (
                not isinstance(payload, dict)
                or "chosen" not in payload
                or "rejected" not in payload
            ):
                excluded.append(
                    {"episode_id": episode.episode_id, "reason": "missing chosen/rejected output"}
                )
                continue
            common.update(
                {
                    "prompt_json": json.dumps(_episode_input(episode, blobs), sort_keys=True),
                    "chosen_json": json.dumps(payload["chosen"], sort_keys=True),
                    "rejected_json": json.dumps(payload["rejected"], sort_keys=True),
                    "preference_source": payload.get("preference_source", "unknown"),
                    "annotator_confidence": payload.get("annotator_confidence"),
                }
            )
        elif kind == DatasetKind.RL:
            rl_payload, rl_error = _rl_payload(episode, blobs, strict=strict_rl)
            if rl_payload is None:
                excluded.append(
                    {"episode_id": episode.episode_id, "reason": rl_error or "invalid RL row"}
                )
                continue
            common.update(rl_payload)
        output_rows.append(common)

    source_digests = {
        episode_manifest_key: digest_bytes(episode_bytes),
        evaluation_manifest_key: digest_bytes(evaluation_bytes),
        **{f"task://{key}": value for key, value in task_digests.items()},
    }
    config: dict[str, JsonValue] = {
        "name": name,
        "kind": kind.value,
        "episode_snapshot_id": episode_manifest.snapshot_id,
        "evaluation_snapshot_id": evaluation_manifest.snapshot_id,
        "minimum_reward": minimum_reward,
        "include_failed": include_failed,
        "strict_rl": strict_rl,
    }
    snapshot_id = make_snapshot_id("dataset", source_digests, config)
    prefix = f"datasets/{name}/versions/{snapshot_id}"
    tables = {
        "data": put_parquet(objects, f"{prefix}/data.parquet", output_rows),
        "excluded": put_parquet(objects, f"{prefix}/excluded.parquet", excluded),
    }
    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        kind="dataset",
        created_at=now_utc(),
        source_refs=[
            f"object://{episode_manifest_key}",
            f"object://{evaluation_manifest_key}",
            *[f"task://{key}" for key in sorted(task_digests)],
        ],
        source_digests=source_digests,
        tables=tables,
        config=config,
        metrics={
            "row_count": len(output_rows),
            "excluded_count": len(excluded),
            "input_episode_count": len(episode_rows),
        },
    )
    manifest_key = put_manifest(objects, prefix, manifest)
    return DatasetBuildResult(manifest_key=manifest_key, manifest=manifest)
