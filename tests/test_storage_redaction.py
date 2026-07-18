"""Object, blob, raw-batch, and privacy behavior."""

from __future__ import annotations

from typing import cast

import pytest
import zstandard
from pydantic import JsonValue

from agenttrace.models import Content, EventBatch, StepType
from agenttrace.redaction import Redactor
from agenttrace.storage import BlobStore, FileObjectStore, RawBatchWriter
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
