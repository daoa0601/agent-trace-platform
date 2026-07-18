"""Batch evaluation and decomposed reward snapshot construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from agenttrace.evaluators import EvaluatorRegistry, evaluate_episode
from agenttrace.models import Episode
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


@dataclass(frozen=True)
class EvaluationBuildResult:
    manifest_key: str
    manifest: SnapshotManifest


def _reward_version(episode: Episode, evaluator_payload: list[dict[str, object]]) -> str:
    payload = json.dumps(evaluator_payload, sort_keys=True, separators=(",", ":"))
    task = episode.task
    prefix = f"{task.task_id}@{task.task_version}" if task else "unknown"
    return f"{prefix}:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def build_evaluation_snapshot(
    objects: ObjectStore,
    *,
    episode_manifest_key: str,
    tasks: TaskRegistry,
    evaluator_registry: EvaluatorRegistry | None = None,
) -> EvaluationBuildResult:
    """Evaluate every valid episode against its versioned task definition."""
    episode_manifest_bytes = objects.get_bytes(episode_manifest_key)
    episode_manifest = load_manifest(objects, episode_manifest_key)
    if episode_manifest.kind != "episodes":
        raise ValueError("evaluation requires an episode snapshot")
    rows = read_parquet(objects, episode_manifest.tables["episodes"])
    blobs = BlobStore(objects)
    evaluation_rows: list[dict[str, object]] = []
    reward_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    passed_episodes = 0

    task_fingerprint = {
        f"{task.task_id}@{task.version}": hashlib.sha256(
            task.model_dump_json(exclude_none=True).encode()
        ).hexdigest()
        for task in tasks.list()
    }
    for row in rows:
        episode_json = row.get("episode_json")
        if not isinstance(episode_json, str):
            raise ValueError("episode table is missing episode_json")
        episode = Episode.model_validate_json(episode_json)
        if episode.task is None:
            error_rows.append(
                {"episode_id": episode.episode_id, "error": "episode has no task reference"}
            )
            continue
        try:
            task = tasks.get(episode.task.task_id, episode.task.task_version)
            results = evaluate_episode(
                episode,
                task,
                blobs,
                registry=evaluator_registry,
            )
        except (KeyError, ValueError) as error:
            error_rows.append({"episode_id": episode.episode_id, "error": str(error)})
            continue

        required_by_id = {spec.evaluator_id: spec.required for spec in task.evaluators}
        episode_passed = all(
            result.passed or not required_by_id[result.evaluator_id] for result in results
        )
        passed_episodes += int(episode_passed)
        result_payload: list[dict[str, object]] = []
        components: dict[str, float] = {}
        for result in results:
            payload = result.model_dump(mode="json")
            result_payload.append(payload)
            components[result.evaluator_id] = result.score * result.weight
            evaluation_rows.append(
                {
                    "experiment_id": episode.experiment_id,
                    "run_id": episode.run_id,
                    "episode_id": episode.episode_id,
                    "task_id": task.task_id,
                    "task_version": task.version,
                    "evaluator_id": result.evaluator_id,
                    "evaluator_version": result.evaluator_version,
                    "passed": result.passed,
                    "score": result.score,
                    "weight": result.weight,
                    "weighted_score": result.score * result.weight,
                    "details_json": json.dumps(result.details, sort_keys=True),
                    "calculated_at": result.calculated_at,
                }
            )
        total = sum(components.values())
        reward_rows.append(
            {
                "experiment_id": episode.experiment_id,
                "run_id": episode.run_id,
                "episode_id": episode.episode_id,
                "task_id": task.task_id,
                "task_version": task.version,
                "passed": episode_passed,
                "reward_total": total,
                "reward_components_json": json.dumps(components, sort_keys=True),
                "reward_function_version": _reward_version(episode, result_payload),
                "aggregation_method": "weighted_sum",
            }
        )

    source_digests = {
        episode_manifest_key: digest_bytes(episode_manifest_bytes),
        **{f"task://{key}": value for key, value in task_fingerprint.items()},
    }
    config = {
        "episode_snapshot_id": episode_manifest.snapshot_id,
        "task_versions": sorted(task_fingerprint),
    }
    snapshot_id = make_snapshot_id("evaluations", source_digests, config)
    prefix = f"evaluations/snapshots/{snapshot_id}"
    tables = {
        "evaluation_results": put_parquet(
            objects,
            f"{prefix}/tables/evaluation_results.parquet",
            evaluation_rows,
        ),
        "episode_rewards": put_parquet(
            objects,
            f"{prefix}/tables/episode_rewards.parquet",
            reward_rows,
        ),
        "evaluation_errors": put_parquet(
            objects,
            f"{prefix}/tables/evaluation_errors.parquet",
            error_rows,
        ),
    }
    evaluated = len(reward_rows)
    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        kind="evaluations",
        created_at=now_utc(),
        source_refs=[f"object://{episode_manifest_key}", *sorted(task_fingerprint)],
        source_digests=source_digests,
        tables=tables,
        config=config,
        metrics={
            "evaluated_episode_count": evaluated,
            "evaluation_result_count": len(evaluation_rows),
            "evaluation_error_count": len(error_rows),
            "episode_pass_rate": passed_episodes / evaluated if evaluated else 0.0,
        },
    )
    manifest_key = put_manifest(objects, prefix, manifest)
    return EvaluationBuildResult(manifest_key=manifest_key, manifest=manifest)
