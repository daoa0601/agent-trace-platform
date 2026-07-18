"""Ordered episode reconstruction and cross-event quality checks."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass

from pydantic import JsonValue

from agenttrace.models import Episode, QualityIssue, RunStatus, StepType, TraceEvent
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
from agenttrace.storage.objects import ObjectStore


@dataclass(frozen=True)
class EpisodeBuildResult:
    manifest_key: str
    manifest: SnapshotManifest


def _issue(
    code: str,
    severity: str,
    message: str,
    event_id: str | None = None,
) -> QualityIssue:
    return QualityIssue(
        code=code,
        severity="error" if severity == "error" else "warning",
        message=message,
        event_id=event_id,
    )


def validate_episode_steps(steps: list[TraceEvent]) -> list[QualityIssue]:
    """Apply checks that cannot be enforced on one event in isolation."""
    issues: list[QualityIssue] = []
    if not steps:
        return [_issue("empty_episode", "error", "episode contains no steps")]

    step_counts = Counter(event.step_id for event in steps)
    for step_id, count in step_counts.items():
        if count > 1:
            issues.append(_issue("duplicate_step_id", "error", f"{step_id} appears {count} times"))

    sequences = [event.sequence_number for event in steps]
    duplicate_sequences = [number for number, count in Counter(sequences).items() if count > 1]
    if duplicate_sequences:
        issues.append(
            _issue(
                "duplicate_sequence_number",
                "error",
                f"duplicate sequence numbers: {sorted(duplicate_sequences)}",
            )
        )
    if sequences and sequences[0] != 0:
        issues.append(
            _issue(
                "sequence_does_not_start_at_zero",
                "warning",
                f"first sequence number is {sequences[0]}",
            )
        )
    if sequences:
        expected = set(range(min(sequences), max(sequences) + 1))
        missing = sorted(expected - set(sequences))
        if missing:
            issues.append(
                _issue("sequence_gap", "warning", f"missing sequence numbers: {missing[:20]}")
            )

    span_ids = {event.span_id for event in steps}
    for event in steps:
        if event.parent_span_id is not None and event.parent_span_id not in span_ids:
            issues.append(
                _issue(
                    "missing_parent_span",
                    "error",
                    f"parent span {event.parent_span_id} is not present in the episode",
                    event.event_id,
                )
            )

    starts = [event for event in steps if event.step_type == StepType.EPISODE_START]
    ends = [event for event in steps if event.step_type == StepType.EPISODE_END]
    if not starts:
        issues.append(_issue("missing_episode_start", "warning", "episode has no start event"))
    elif len(starts) > 1:
        issues.append(
            _issue("multiple_episode_starts", "error", "episode has multiple start events")
        )
    if not ends:
        issues.append(_issue("missing_terminal_state", "error", "episode has no end event"))
    elif len(ends) > 1:
        issues.append(
            _issue("multiple_terminal_states", "error", "episode has multiple end events")
        )

    calls_with_result = {
        event.tool.call_id
        for event in steps
        if event.step_type == StepType.TOOL_CALL
        and event.tool is not None
        and event.tool.call_id
        and event.tool.result is not None
    }
    result_ids = {
        event.tool.call_id
        for event in steps
        if event.step_type == StepType.TOOL_RESULT and event.tool is not None and event.tool.call_id
    }
    for event in steps:
        if event.step_type != StepType.TOOL_CALL or event.tool is None:
            continue
        if event.tool.call_id is None:
            issues.append(
                _issue(
                    "missing_tool_call_id",
                    "warning",
                    f"tool call {event.tool.name} cannot be paired",
                    event.event_id,
                )
            )
        elif event.tool.call_id not in result_ids | calls_with_result:
            issues.append(
                _issue(
                    "missing_tool_result",
                    "error",
                    f"tool call {event.tool.call_id} has no result",
                    event.event_id,
                )
            )

    task_refs = {
        (event.task.task_id, event.task.task_version) for event in steps if event.task is not None
    }
    if len(task_refs) > 1:
        issues.append(_issue("inconsistent_task_version", "error", f"task refs: {task_refs}"))

    tokenizer_versions = {
        version
        for event in steps
        for version in (
            event.model.tokenizer_id if event.model else None,
            event.provenance.tokenizer_version,
        )
        if version is not None
    }
    if len(tokenizer_versions) > 1:
        issues.append(
            _issue(
                "inconsistent_tokenizer_version",
                "error",
                f"tokenizer versions: {sorted(tokenizer_versions)}",
            )
        )
    return issues


def _status(steps: list[TraceEvent]) -> tuple[RunStatus, str | None]:
    end = next(
        (event for event in reversed(steps) if event.step_type == StepType.EPISODE_END), None
    )
    if end is None:
        return RunStatus.TRUNCATED, "missing_terminal_state"
    raw_status = end.attributes.get("status")
    if isinstance(raw_status, str):
        try:
            return RunStatus(raw_status), raw_status
        except ValueError:
            pass
    if any(event.step_type == StepType.ERROR for event in steps):
        return RunStatus.FAILED, str(raw_status) if raw_status else "error_event"
    return RunStatus.SUCCEEDED, str(raw_status) if raw_status else "completed"


def build_episode_snapshot(
    objects: ObjectStore,
    *,
    canonical_manifest_key: str,
) -> EpisodeBuildResult:
    """Build validated episodes from a canonical steps snapshot."""
    canonical_bytes = objects.get_bytes(canonical_manifest_key)
    canonical = load_manifest(objects, canonical_manifest_key)
    if canonical.kind != "canonical":
        raise ValueError("episode builder requires a canonical snapshot")
    step_rows = read_parquet(objects, canonical.tables["steps"])
    groups: dict[tuple[str, str, str], list[TraceEvent]] = defaultdict(list)
    for row in step_rows:
        event_json = row.get("event_json")
        if not isinstance(event_json, str):
            raise ValueError("canonical steps table is missing event_json")
        event = TraceEvent.model_validate_json(event_json)
        groups[(event.experiment_id, event.run_id, event.episode_id)].append(event)

    config: dict[str, JsonValue] = {
        "canonical_snapshot_id": canonical.snapshot_id,
        "quality_policy": "strict-v1",
    }
    source_digests = {canonical_manifest_key: digest_bytes(canonical_bytes)}
    snapshot_id = make_snapshot_id("episodes", source_digests, config)
    prefix = f"episodes/snapshots/{snapshot_id}"
    valid_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    issue_counts: Counter[str] = Counter()
    completed = 0

    for _, unordered_steps in sorted(groups.items()):
        steps = sorted(
            unordered_steps,
            key=lambda event: (event.sequence_number, event.started_at, event.event_id),
        )
        issues = validate_episode_steps(steps)
        issue_counts.update(issue.code for issue in issues)
        status, terminal_reason = _status(steps)
        if status == RunStatus.SUCCEEDED:
            completed += 1
        task = next((event.task for event in steps if event.task is not None), None)
        reward_values = [event.reward.total for event in steps if event.reward is not None]
        episode = Episode(
            experiment_id=steps[0].experiment_id,
            run_id=steps[0].run_id,
            episode_id=steps[0].episode_id,
            task=task,
            status=status,
            started_at=min(event.started_at for event in steps),
            ended_at=max((event.ended_at or event.started_at) for event in steps),
            steps=steps,
            quality_issues=issues,
            total_reward=sum(reward_values) if reward_values else None,
            terminal_reason=terminal_reason,
        )
        row: dict[str, object] = {
            "experiment_id": episode.experiment_id,
            "run_id": episode.run_id,
            "episode_id": episode.episode_id,
            "task_id": task.task_id if task else None,
            "task_version": task.task_version if task else None,
            "status": episode.status.value,
            "started_at": episode.started_at,
            "ended_at": episode.ended_at,
            "step_count": len(episode.steps),
            "total_reward": episode.total_reward,
            "valid": episode.is_valid,
            "quality_issue_codes_json": json.dumps([issue.code for issue in issues]),
            "episode_json": episode.model_dump_json(exclude_none=True),
        }
        (valid_rows if episode.is_valid else invalid_rows).append(row)

    total = len(groups)
    tables = {
        "episodes": put_parquet(objects, f"{prefix}/tables/episodes.parquet", valid_rows),
        "quarantined_episodes": put_parquet(
            objects,
            f"{prefix}/tables/quarantined_episodes.parquet",
            invalid_rows,
        ),
        "quality_summary": put_parquet(
            objects,
            f"{prefix}/tables/quality_summary.parquet",
            [{"issue_code": code, "count": count} for code, count in sorted(issue_counts.items())],
        ),
    }
    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        kind="episodes",
        created_at=now_utc(),
        source_refs=[f"object://{canonical_manifest_key}"],
        source_digests=source_digests,
        tables=tables,
        config=config,
        metrics={
            "episode_count": total,
            "valid_episode_count": len(valid_rows),
            "invalid_episode_count": len(invalid_rows),
            "completion_rate": completed / total if total else 0.0,
            "invalid_trace_rate": len(invalid_rows) / total if total else 0.0,
        },
    )
    manifest_key = put_manifest(objects, prefix, manifest)
    return EpisodeBuildResult(manifest_key=manifest_key, manifest=manifest)
