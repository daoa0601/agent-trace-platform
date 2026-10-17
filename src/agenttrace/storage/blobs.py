"""Content-addressed compressed JSON payload storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import zstandard
from pydantic import JsonValue

from agenttrace.models import Content, blob_ref
from agenttrace.storage.objects import ObjectStore


def canonical_json(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True)
class BlobWriteResult:
    content: Content
    created: bool


class BlobStore:
    """Deduplicated JSON blob layer over an object store."""

    def __init__(self, objects: ObjectStore, *, compression_level: int = 6) -> None:
        self.objects = objects
        self._compressor = zstandard.ZstdCompressor(level=compression_level)
        self._decompressor = zstandard.ZstdDecompressor()

    def put_json(self, value: JsonValue, *, redacted: bool = False) -> BlobWriteResult:
        raw = canonical_json(value)
        digest = hashlib.sha256(raw).hexdigest()
        ref = blob_ref(digest)
        key = ref.removeprefix("object://")
        compressed = self._compressor.compress(raw)
        info = self.objects.put_bytes(
            key,
            compressed,
            content_type="application/zstd",
            if_absent=True,
        )
        return BlobWriteResult(
            content=Content(
                ref=ref,
                sha256=digest,
                media_type="application/json",
                byte_size=len(raw),
                encoding="zstd",
                redacted=redacted,
            ),
            created=info.created,
        )

    def get_json(self, content: Content) -> JsonValue:
        if content.inline is not None:
            return content.inline
        if content.ref is None:
            return None
        key = content.ref.removeprefix("object://")
        compressed = self.objects.get_bytes(key)
        raw = self._decompressor.decompress(compressed)
        digest = hashlib.sha256(raw).hexdigest()
        if content.sha256 is not None and digest != content.sha256:
            raise ValueError(f"blob checksum mismatch for {content.ref}")
        return json.loads(raw)

    def maybe_offload(self, content: Content | None, threshold: int) -> Content | None:
        if content is None or content.inline is None or content.ref is not None:
            return content
        if len(canonical_json(content.inline)) <= threshold:
            return content
        result = self.put_json(content.inline, redacted=content.redacted)
        return result.content.model_copy(
            update={
                "media_type": content.media_type,
                "redacted": content.redacted,
            }
        )
