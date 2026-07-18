"""Immutable snapshot manifests and Parquet IO."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agenttrace.storage.objects import ObjectStore

PIPELINE_VERSION = "0.1.0"


class TableManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    sha256: str
    row_count: int = Field(ge=0)
    byte_size: int = Field(ge=0)


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    snapshot_id: str
    kind: Literal["canonical", "episodes", "evaluations", "dataset"]
    pipeline_version: str = PIPELINE_VERSION
    created_at: datetime
    source_refs: list[str]
    source_digests: dict[str, str]
    tables: dict[str, TableManifest]
    config: dict[str, JsonValue] = Field(default_factory=dict)
    metrics: dict[str, int | float | str] = Field(default_factory=dict)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_snapshot_id(kind: str, inputs: dict[str, str], config: dict[str, JsonValue]) -> str:
    payload = json.dumps(
        {
            "kind": kind,
            "inputs": dict(sorted(inputs.items())),
            "config": config,
            "pipeline_version": PIPELINE_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"{kind}_{digest_bytes(payload)[:20]}"


def parquet_bytes(rows: list[dict[str, object]]) -> bytes:
    table = pa.Table.from_pylist(rows) if rows else pa.table({"_empty": pa.array([], pa.bool_())})
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    return sink.getvalue().to_pybytes()


def put_parquet(
    objects: ObjectStore,
    key: str,
    rows: list[dict[str, object]],
) -> TableManifest:
    data = parquet_bytes(rows)
    info = objects.put_bytes(
        key, data, content_type="application/vnd.apache.parquet", if_absent=True
    )
    return TableManifest(
        key=key,
        sha256=digest_bytes(data),
        row_count=len(rows),
        byte_size=info.size,
    )


def read_parquet(objects: ObjectStore, table: TableManifest) -> list[dict[str, object]]:
    data = objects.get_bytes(table.key)
    if digest_bytes(data) != table.sha256:
        raise ValueError(f"Parquet checksum mismatch: {table.key}")
    parsed = pq.read_table(pa.BufferReader(data)).to_pylist()
    if parsed and set(parsed[0]) == {"_empty"}:
        return []
    return parsed


def put_manifest(objects: ObjectStore, prefix: str, manifest: SnapshotManifest) -> str:
    key = f"{prefix}/manifest.json"
    objects.put_bytes(
        key,
        manifest.canonical_bytes(),
        content_type="application/json",
        if_absent=True,
    )
    return key


def load_manifest(objects: ObjectStore, key: str) -> SnapshotManifest:
    return SnapshotManifest.model_validate_json(objects.get_bytes(key))


def now_utc() -> datetime:
    return datetime.now(UTC)
