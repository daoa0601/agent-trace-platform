"""Identifier helpers compatible with OpenTelemetry trace identifiers."""

from __future__ import annotations

import secrets
import uuid


def new_id(prefix: str) -> str:
    """Return a sortable-enough, collision-resistant application identifier."""
    return f"{prefix}_{uuid.uuid4().hex}"


def new_trace_id() -> str:
    """Return a non-zero 128-bit lowercase hexadecimal trace identifier."""
    while True:
        value = secrets.token_hex(16)
        if int(value, 16) != 0:
            return value


def new_span_id() -> str:
    """Return a non-zero 64-bit lowercase hexadecimal span identifier."""
    while True:
        value = secrets.token_hex(8)
        if int(value, 16) != 0:
            return value
