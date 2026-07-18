"""Content-addressed artifact writes for OpenCode benchmark evidence."""

from __future__ import annotations

import hashlib
import json

from agenttrace.storage.objects import ObjectStore


def persist_json(objects: ObjectStore, prefix: str, value: object) -> tuple[str, str]:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    key = f"{prefix}/sha256/{digest[:2]}/{digest}.json"
    objects.put_bytes(key, encoded, content_type="application/json", if_absent=True)
    return key, digest


def persist_bytes(
    objects: ObjectStore,
    prefix: str,
    content: bytes,
    *,
    suffix: str,
    content_type: str,
) -> tuple[str, str]:
    digest = hashlib.sha256(content).hexdigest()
    key = f"{prefix}/sha256/{digest[:2]}/{digest}.{suffix}"
    objects.put_bytes(key, content, content_type=content_type, if_absent=True)
    return f"object://{key}", digest
