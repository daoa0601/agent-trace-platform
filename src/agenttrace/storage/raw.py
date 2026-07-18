"""Immutable compressed raw batch writer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import zstandard

from agenttrace.models import Content, EventBatch, TraceEvent
from agenttrace.redaction import RedactionStats, Redactor
from agenttrace.storage.blobs import BlobStore
from agenttrace.storage.objects import ObjectStore


@dataclass(frozen=True)
class RawWriteResult:
    key: str
    accepted: int
    duplicate: bool
    bytes_written: int
    redactions: RedactionStats
    event_ids: list[str]


class RawBatchWriter:
    """Redact, offload payloads, and append a deterministic raw batch."""

    def __init__(
        self,
        objects: ObjectStore,
        *,
        inline_payload_bytes: int = 16 * 1024,
        redactor: Redactor | None = None,
    ) -> None:
        self.objects = objects
        self.blobs = BlobStore(objects)
        self.inline_payload_bytes = inline_payload_bytes
        self.redactor = redactor or Redactor("balanced")
        self._compressor = zstandard.ZstdCompressor(level=6)

    def write(self, batch: EventBatch) -> RawWriteResult:
        events: list[TraceEvent] = []
        total_stats = RedactionStats()
        for event in batch.events:
            cleaned_payload, stats = self.redactor.redact(event.model_dump(mode="json"))
            total_stats = total_stats.add(stats)
            cleaned = TraceEvent.model_validate(cleaned_payload)
            events.append(self._offload_event(cleaned))

        lines = [event.model_dump_json(exclude_none=True) for event in events]
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        compressed = self._compressor.compress(raw)
        digest = hashlib.sha256(compressed).hexdigest()
        first = events[0]
        date = first.started_at.date().isoformat()
        key = (
            f"raw/collection_date={date}/experiment_id={first.experiment_id}/"
            f"worker_id={batch.worker_id}/part-{digest[:32]}.jsonl.zst"
        )
        info = self.objects.put_bytes(
            key,
            compressed,
            content_type="application/zstd",
            if_absent=True,
        )
        return RawWriteResult(
            key=key,
            accepted=len(events),
            duplicate=not info.created,
            bytes_written=info.size,
            redactions=total_stats,
            event_ids=[event.event_id for event in events],
        )

    def _offload_event(self, event: TraceEvent) -> TraceEvent:
        updates: dict[str, object] = {}
        for field_name in ("input", "output"):
            content = getattr(event, field_name)
            updates[field_name] = self.blobs.maybe_offload(content, self.inline_payload_bytes)

        if event.model is not None:
            model_updates = {
                name: self.blobs.maybe_offload(
                    getattr(event.model, name),
                    self.inline_payload_bytes,
                )
                for name in (
                    "input_token_ids",
                    "output_token_ids",
                    "behavior_logprobs",
                    "reference_logprobs",
                    "value_estimates",
                )
            }
            updates["model"] = event.model.model_copy(update=model_updates)
        if event.tool is not None:
            updates["tool"] = event.tool.model_copy(
                update={
                    "arguments": self.blobs.maybe_offload(
                        event.tool.arguments,
                        self.inline_payload_bytes,
                    ),
                    "result": self.blobs.maybe_offload(
                        event.tool.result,
                        self.inline_payload_bytes,
                    ),
                }
            )
        if event.environment is not None:
            env_updates: dict[str, Content | None] = {
                name: self.blobs.maybe_offload(
                    getattr(event.environment, name),
                    self.inline_payload_bytes,
                )
                for name in ("state_before", "state_after", "observation", "action")
            }
            updates["environment"] = event.environment.model_copy(update=env_updates)
        return event.model_copy(update=updates)


def decode_raw_batch(data: bytes) -> list[TraceEvent]:
    """Decode and validate one raw JSONL-Zstandard object."""
    raw = zstandard.ZstdDecompressor().decompress(data)
    return [TraceEvent.model_validate_json(line) for line in raw.splitlines() if line.strip()]
