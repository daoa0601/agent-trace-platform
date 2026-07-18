"""Compatibility re-exports for the standalone deterministic audit gate."""

from benchmark_audit_core import (
    build_audit_bundle,
    evaluate_headline_gate,
    render_audit_report,
)

__all__ = [
    "build_audit_bundle",
    "evaluate_headline_gate",
    "render_audit_report",
]
