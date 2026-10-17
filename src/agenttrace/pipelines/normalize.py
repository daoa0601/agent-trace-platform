"""Raw trace validation, deduplication, reference checking, and Parquet normalization."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import zstandard
from pydantic import ValidationError

from agenttrace.models import Content, StepType, TraceEvent
from agenttrace.pipelines.snapshots import (
    SnapshotManifest,
    digest_bytes,
    make_snapshot_id,
    now_utc,
    put_manifest,
    put_parquet,
)
from agenttrace.storage.blobs import canonical_json
from agenttrace.storage.objects import ObjectStore


@dataclass(frozen=True)
class NormalizationResult:
    manifest_key: str
    manifest: SnapshotManifest


def _content_columns(prefix: str, content: Content | None) -> dict[str, object]:
    if content is None:
        return {
            f"{prefix}_ref": None,
            f"{prefix}_sha256": None,
            f"{prefix}_inline_json": None,
        }
    return {
        f"{prefix}_ref": content.ref,
        f"{prefix}_sha256": content.sha256,
        f"{prefix}_inline_json": json.dumps(content.inline, sort_keys=True)
        if content.inline is not None
        else None,
    }


def event_to_row(event: TraceEvent, source_key: str) -> dict[str, object]:
    """Flatten stable analytical columns while retaining a lossless event JSON."""
    row: dict[str, object] = {
        "schema_version": event.schema_version,
        "event_id": event.event_id,
        "experiment_id": event.experiment_id,
        "run_id": event.run_id,
        "episode_id": event.episode_id,
        "trace_id": event.trace_id,
        "span_id": event.span_id,
        "parent_span_id": event.parent_span_id,
        "step_id": event.step_id,
        "sequence_number": event.sequence_number,
        "step_type": event.step_type.value,
        "started_at": event.started_at,
        "ended_at": event.ended_at,
        "task_id": event.task.task_id if event.task else None,
        "task_version": event.task.task_version if event.task else None,
        "task_split": event.task.split if event.task else None,
        "model_provider": event.model.provider if event.model else None,
        "model_id": event.model.model_id if event.model else None,
        "checkpoint_id": event.model.checkpoint_id if event.model else None,
        "tokenizer_id": event.model.tokenizer_id if event.model else None,
        "tool_name": event.tool.name if event.tool else None,
        "tool_call_id": event.tool.call_id if event.tool else None,
        "tool_success": event.tool.success if event.tool else None,
        "tool_exit_code": event.tool.exit_code if event.tool else None,
        "environment_id": event.environment.environment_id if event.environment else None,
        "environment_version": event.environment.environment_version if event.environment else None,
        "reward_total": event.reward.total if event.reward else None,
        "reward_components_json": json.dumps(event.reward.components, sort_keys=True)
        if event.reward
        else None,
        "input_tokens": event.metrics.tokens.input_tokens,
        "output_tokens": event.metrics.tokens.output_tokens,
        "latency_ms": event.metrics.latency_ms,
        "cost_usd": event.metrics.cost_usd,
        "privacy_level": event.privacy.level.value,
        "source_raw_key": source_key,
        "event_json": event.model_dump_json(exclude_none=True),
    }
    row.update(_content_columns("input", event.input))
    row.update(_content_columns("output", event.output))
    return row


def _logical_json(event: TraceEvent) -> str:
    """Event JSON that is equal whether or not the raw writer offloaded its attributes."""
    payload = event.model_dump(
        mode="json", exclude_none=True, exclude={"attributes", "offloaded_attributes"}
    )
    digests: dict[str, str | None] = {
        key: hashlib.sha256(canonical_json(value)).hexdigest()
        for key, value in event.attributes.items()
    }
    digests.update(
        {key: content.sha256 for key, content in (event.offloaded_attributes or {}).items()}
    )
    payload["attribute_sha256"] = digests
    return json.dumps(payload, sort_keys=True)


def _referenced_contents(event: TraceEvent) -> Iterable[Content]:
    values: list[Content | None] = [event.input, event.output]
    if event.model:
        values.extend(
            [
                event.model.input_token_ids,
                event.model.output_token_ids,
                event.model.behavior_logprobs,
                event.model.reference_logprobs,
                event.model.value_estimates,
            ]
        )
    if event.tool:
        values.extend([event.tool.arguments, event.tool.result])
    if event.environment:
        values.extend(
            [
                event.environment.state_before,
                event.environment.state_after,
                event.environment.observation,
                event.environment.action,
            ]
        )
    values.extend((event.offloaded_attributes or {}).values())
    return [content for content in values if content is not None and content.ref is not None]


def normalize_raw_traces(
    objects: ObjectStore,
    *,
    raw_prefix: str = "raw",
    strict_references: bool = True,
) -> NormalizationResult:
    """Build an immutable canonical snapshot from every raw object under a prefix."""
    raw_keys = sorted(key for key in objects.list_keys(raw_prefix) if key.endswith(".jsonl.zst"))
    if not raw_keys:
        raise ValueError(f"no raw trace objects found under {raw_prefix!r}")

    source_digests: dict[str, str] = {}
    candidates: dict[str, tuple[TraceEvent, str, str]] = {}
    conflicts: set[str] = set()
    quarantine: list[dict[str, object]] = []
    duplicate_count = 0
    broken_reference_count = 0

    for key in raw_keys:
        compressed = objects.get_bytes(key)
        source_digests[key] = digest_bytes(compressed)
        try:
            raw = zstandard.ZstdDecompressor().decompress(compressed)
        except zstandard.ZstdError as error:
            quarantine.append(
                {
                    "source_raw_key": key,
                    "line_number": None,
                    "error": str(error),
                    "raw_preview": None,
                }
            )
            continue
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = TraceEvent.model_validate_json(line)
            except ValidationError as error:
                quarantine.append(
                    {
                        "source_raw_key": key,
                        "line_number": line_number,
                        "error": str(error),
                        "raw_preview": line[:1000].decode("utf-8", errors="replace"),
                    }
                )
                continue
            canonical = event.model_dump_json(exclude_none=True)
            existing = candidates.get(event.event_id)
            if existing is not None:
                duplicate_count += 1
                # Raw batches written before attribute offloading keep large values inline.
                if existing[2] != canonical and _logical_json(existing[0]) != _logical_json(event):
                    conflicts.add(event.event_id)
                    quarantine.append(
                        {
                            "source_raw_key": key,
                            "line_number": line_number,
                            "error": f"conflicting duplicate event_id: {event.event_id}",
                            "raw_preview": line[:1000].decode("utf-8", errors="replace"),
                        }
                    )
                continue
            candidates[event.event_id] = (event, key, canonical)

    for event_id in conflicts:
        event, key, canonical = candidates.pop(event_id)
        quarantine.append(
            {
                "source_raw_key": key,
                "line_number": None,
                "error": f"first occurrence of conflicting event_id: {event_id}",
                "raw_preview": canonical[:1000],
            }
        )

    events: list[tuple[TraceEvent, str]] = []
    for event, key, canonical in candidates.values():
        broken = [
            content.ref
            for content in _referenced_contents(event)
            if content.ref and not objects.exists(content.ref.removeprefix("object://"))
        ]
        if broken:
            broken_reference_count += len(broken)
            quarantine.append(
                {
                    "source_raw_key": key,
                    "line_number": None,
                    "error": f"broken object references: {broken}",
                    "raw_preview": canonical[:1000],
                }
            )
            if strict_references:
                continue
        events.append((event, key))

    if not events:
        raise ValueError("normalization produced no valid events")
    events.sort(
        key=lambda item: (
            item[0].experiment_id,
            item[0].run_id,
            item[0].episode_id,
            item[0].sequence_number,
            item[0].started_at,
            item[0].event_id,
        )
    )
    config = {
        "raw_prefix": raw_prefix,
        "strict_references": strict_references,
        # Names the duplicate rule so a rule change yields a new snapshot, not a stale reuse.
        "duplicate_comparison": "attribute-digests",
    }
    snapshot_id = make_snapshot_id("canonical", source_digests, config)
    prefix = f"canonical/snapshots/{snapshot_id}"
    rows = [event_to_row(event, key) for event, key in events]

    run_groups: dict[tuple[str, str, str], list[TraceEvent]] = defaultdict(list)
    for event, _ in events:
        run_groups[(event.experiment_id, event.run_id, event.episode_id)].append(event)
    run_rows: list[dict[str, object]] = []
    for (experiment_id, run_id, episode_id), group in sorted(run_groups.items()):
        end_events = [event for event in group if event.step_type == StepType.EPISODE_END]
        run_rows.append(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "episode_id": episode_id,
                "task_id": next((event.task.task_id for event in group if event.task), None),
                "task_version": next(
                    (event.task.task_version for event in group if event.task), None
                ),
                "started_at": min(event.started_at for event in group),
                "ended_at": max((event.ended_at or event.started_at) for event in group),
                "step_count": len(group),
                "terminal": bool(end_events),
                "status": end_events[-1].attributes.get("status") if end_events else "incomplete",
            }
        )

    tables = {
        "steps": put_parquet(objects, f"{prefix}/tables/steps.parquet", rows),
        "runs": put_parquet(objects, f"{prefix}/tables/runs.parquet", run_rows),
        "model_generations": put_parquet(
            objects,
            f"{prefix}/tables/model_generations.parquet",
            [row for row in rows if row["step_type"] == StepType.MODEL_GENERATION.value],
        ),
        "tool_interactions": put_parquet(
            objects,
            f"{prefix}/tables/tool_interactions.parquet",
            [
                row
                for row in rows
                if row["step_type"] in {StepType.TOOL_CALL.value, StepType.TOOL_RESULT.value}
            ],
        ),
        "rewards": put_parquet(
            objects,
            f"{prefix}/tables/rewards.parquet",
            [row for row in rows if row["reward_total"] is not None],
        ),
        "quarantine": put_parquet(
            objects,
            f"{prefix}/tables/quarantine.parquet",
            quarantine,
        ),
    }
    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        kind="canonical",
        created_at=now_utc(),
        source_refs=[f"object://{key}" for key in raw_keys],
        source_digests=source_digests,
        tables=tables,
        config=config,
        metrics={
            "raw_object_count": len(raw_keys),
            "valid_event_count": len(events),
            "episode_count": len(run_groups),
            "duplicate_event_count": duplicate_count,
            "conflicting_event_count": len(conflicts),
            "quarantined_record_count": len(quarantine),
            "broken_reference_count": broken_reference_count,
        },
    )
    manifest_key = put_manifest(objects, prefix, manifest)
    return NormalizationResult(manifest_key=manifest_key, manifest=manifest)
