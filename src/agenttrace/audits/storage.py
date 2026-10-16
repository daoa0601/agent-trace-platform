"""Immutable persistence for benchmark audit bundles."""

from __future__ import annotations

from agenttrace.storage.objects import ObjectInfo, ObjectStore
from benchmark_audit_core import AuditBundle


def persist_audit_bundle(objects: ObjectStore, bundle: AuditBundle) -> ObjectInfo:
    """Write a content-addressed audit bundle without overwriting prior evidence."""
    return objects.put_bytes(
        f"audits/bundles/{bundle.digest()}.json",
        bundle.canonical_bytes(),
        content_type="application/json",
        if_absent=True,
    )
