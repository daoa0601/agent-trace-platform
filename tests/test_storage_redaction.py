"""Object, blob, raw-batch, and privacy behavior."""

from __future__ import annotations

from typing import cast

import pytest
import zstandard
from pydantic import JsonValue

from agenttrace.models import MAX_EVENT_ATTRIBUTES, Content, EventBatch, StepType
from agenttrace.redaction import Redactor
from agenttrace.storage import BlobStore, FileObjectStore, RawBatchWriter
from agenttrace.storage.objects import ObjectInfo
from agenttrace.storage.raw import decode_raw_batch

from .conftest import make_event


def test_file_store_is_immutable_and_blocks_traversal(tmp_path) -> None:
    store = FileObjectStore(tmp_path / "objects")
    first = store.put_bytes("safe/value.txt", b"first")
    second = store.put_bytes("safe/value.txt", b"second")
    assert first.created is True
    assert second.created is False
    assert store.get_bytes("safe/value.txt") == b"first"
    assert list(store.list_keys("safe")) == ["safe/value.txt"]
    with pytest.raises(ValueError, match="unsafe"):
        store.put_bytes("../escape", b"bad")


def test_blob_store_deduplicates_and_verifies_checksum(tmp_path) -> None:
    store = FileObjectStore(tmp_path / "objects")
    blobs = BlobStore(store)
    one = blobs.put_json({"b": 2, "a": 1})
    two = blobs.put_json({"a": 1, "b": 2})
    assert one.created is True
    assert two.created is False
    assert blobs.get_json(one.content) == {"a": 1, "b": 2}
    assert one.content.ref is not None
    store.put_bytes(one.content.ref.removeprefix("object://"), b"corrupt", if_absent=False)
    with pytest.raises(zstandard.ZstdError):
        blobs.get_json(one.content)


def test_redactor_handles_secret_keys_patterns_pii_and_reasoning() -> None:
    payload: JsonValue = {
        "password": "hunter2",
        "message": "mail me at jane@example.com using sk-abcdefghijklmnop",
        "reasoning_content": "private chain",
    }
    balanced_value, balanced_stats = Redactor("balanced").redact(payload)
    balanced = cast(dict[str, JsonValue], balanced_value)
    assert balanced["password"] == "[REDACTED_SECRET]"
    assert "[REDACTED_SECRET]" in cast(str, balanced["message"])
    assert "jane@example.com" in cast(str, balanced["message"])
    assert balanced["reasoning_content"] == "[PRIVATE_REASONING_REMOVED]"
    strict_value, strict_stats = Redactor("strict").redact(payload)
    strict = cast(dict[str, JsonValue], strict_value)
    assert "[REDACTED_EMAIL]" in cast(str, strict["message"])
    assert strict_stats.total > balanced_stats.secret_keys


def test_raw_writer_redacts_offloads_and_is_retry_safe(tmp_path, support_task) -> None:
    store = FileObjectStore(tmp_path / "objects")
    event = make_event(
        1,
        StepType.USER_INPUT,
        task=support_task,
        input={"password": "do-not-store", "text": "x" * 200},
    )
    batch = EventBatch(worker_id="worker-a", events=[event])
    writer = RawBatchWriter(store, inline_payload_bytes=64, redactor=Redactor("balanced"))
    first = writer.write(batch)
    second = writer.write(batch)
    assert first.key == second.key
    assert second.duplicate is True
    stored = decode_raw_batch(store.get_bytes(first.key))[0]
    assert stored.input is not None and stored.input.ref is not None
    restored = cast(dict[str, JsonValue], BlobStore(store).get_json(stored.input))
    assert restored["password"] == "[REDACTED_SECRET]"
    assert first.redactions.secret_keys == 1


def test_content_requires_digest_for_reference() -> None:
    with pytest.raises(ValueError, match="sha256"):
        Content(ref="object://missing")
    with pytest.raises(ValueError, match="not both"):
        Content(inline="x", ref="object://x", sha256="a" * 64)


def test_raw_writer_offloads_oversized_attributes_after_redaction(tmp_path, support_task) -> None:
    store = FileObjectStore(tmp_path / "objects")
    prompt = "summarize sk-abcdefghijklmnopqrstuv " + "y" * 400
    event = make_event(
        1,
        StepType.USER_INPUT,
        task=support_task,
        attributes={"status": "running", "llm.input_messages.0.message.content": prompt},
    )
    writer = RawBatchWriter(store, inline_payload_bytes=256)
    result = writer.write(EventBatch(worker_id="worker-a", events=[event]))
    raw = zstandard.ZstdDecompressor().decompress(store.get_bytes(result.key))
    assert b"y" * 400 not in raw
    stored = decode_raw_batch(store.get_bytes(result.key))[0]
    assert stored.attributes == {"status": "running"}
    assert stored.offloaded_attributes is not None
    reference = stored.offloaded_attributes["llm.input_messages.0.message.content"]
    restored = cast(str, BlobStore(store).get_json(reference))
    assert restored.startswith("summarize [REDACTED_SECRET] ")
    assert result.redactions.secret_values == 1


def test_raw_writer_bounds_total_inline_attribute_size(tmp_path, support_task) -> None:
    store = FileObjectStore(tmp_path / "objects")
    attributes: dict[str, JsonValue] = {
        f"llm.input_messages.{index}.message.content": str(index) * (100 + index)
        for index in range(5)
    }
    attributes["status"] = "running"
    event = make_event(1, StepType.USER_INPUT, task=support_task, attributes=attributes)
    writer = RawBatchWriter(store, inline_payload_bytes=256)
    result = writer.write(EventBatch(worker_id="worker-a", events=[event]))
    stored = decode_raw_batch(store.get_bytes(result.key))[0]
    assert stored.offloaded_attributes is not None
    # No single value is over the limit, but together they are. Largest values move first, so the
    # smallest message stays inline with the status flag.
    assert set(stored.offloaded_attributes) == {
        f"llm.input_messages.{index}.message.content" for index in (1, 2, 3, 4)
    }
    assert set(stored.attributes) == {"status", "llm.input_messages.0.message.content"}
    blobs = BlobStore(store)
    for key, reference in stored.offloaded_attributes.items():
        assert blobs.get_json(reference) == attributes[key]


def test_raw_writer_keeps_small_attributes_inline(tmp_path, support_task) -> None:
    store = FileObjectStore(tmp_path / "objects")
    event = make_event(1, StepType.USER_INPUT, task=support_task, attributes={"status": "ok"})
    result = RawBatchWriter(store).write(EventBatch(worker_id="worker-a", events=[event]))
    stored = decode_raw_batch(store.get_bytes(result.key))[0]
    assert stored.attributes == {"status": "ok"}
    assert stored.offloaded_attributes is None
    raw = zstandard.ZstdDecompressor().decompress(store.get_bytes(result.key))
    assert b"offloaded_attributes" not in raw


def test_trace_event_rejects_attribute_both_inline_and_offloaded(support_task) -> None:
    reference = Content(ref="object://blobs/x.json.zst", sha256="a" * 64)
    with pytest.raises(ValueError, match="both inline and offloaded"):
        make_event(
            1,
            StepType.USER_INPUT,
            task=support_task,
            attributes={"input.value": "x"},
            offloaded_attributes={"input.value": reference},
        )


def test_raw_writer_does_not_offload_values_below_a_quarter_of_the_limit(
    tmp_path, support_task
) -> None:
    store = FileObjectStore(tmp_path / "objects")
    attributes: dict[str, JsonValue] = {f"tag.{index}": "v" * 8 for index in range(400)}
    event = make_event(1, StepType.USER_INPUT, task=support_task, attributes=attributes)
    writer = RawBatchWriter(store, inline_payload_bytes=256)
    result = writer.write(EventBatch(worker_id="worker-a", events=[event]))
    stored = decode_raw_batch(store.get_bytes(result.key))[0]
    # Offloading tiny values one blob each would turn one request into hundreds of writes.
    assert stored.offloaded_attributes is None
    assert stored.attributes == attributes
    assert list(store.list_keys("blobs")) == []


def test_event_batch_rejects_events_with_too_many_attributes(support_task) -> None:
    attributes: dict[str, JsonValue] = {
        f"tag.{index}": index for index in range(MAX_EVENT_ATTRIBUTES + 1)
    }
    event = make_event(1, StepType.USER_INPUT, task=support_task, attributes=attributes)
    with pytest.raises(ValueError, match="attributes"):
        EventBatch(worker_id="worker-a", events=[event])


class CountingStore(FileObjectStore):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.blob_puts = 0

    def put_bytes(self, key: str, data: bytes, **options) -> ObjectInfo:
        if key.startswith("blobs/"):
            self.blob_puts += 1
        return super().put_bytes(key, data, **options)


def test_raw_writer_writes_a_shared_attribute_once_per_batch(tmp_path, support_task) -> None:
    store = CountingStore(tmp_path / "objects")
    shared = {"service.description": "s" * 300}
    events = [
        make_event(index, StepType.USER_INPUT, task=support_task, attributes=dict(shared))
        for index in range(1, 6)
    ]
    RawBatchWriter(store, inline_payload_bytes=256).write(
        EventBatch(worker_id="worker-a", events=events)
    )
    assert store.blob_puts == 1
