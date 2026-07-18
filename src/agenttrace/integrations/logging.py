"""Correlated JSON logging with collection-compatible redaction."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, cast

from agenttrace.integrations.telemetry import current_trace_fields
from agenttrace.redaction import Redactor

_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__)


class TraceContextFilter(logging.Filter):
    """Attach current OpenTelemetry and AgentTrace identifiers to a log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in current_trace_fields().items():
            setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    """Emit one redacted JSON object per line for ingestion by any log backend."""

    def __init__(self, *, redaction_mode: str = "balanced") -> None:
        super().__init__()
        self._redactor = Redactor(cast(Any, redaction_mode))

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("experiment_id", "run_id", "episode_id", "trace_id", "span_id"):
            value = getattr(record, key, None)
            if value:
                payload[key] = value
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_FIELDS
            and key not in payload
            and key not in {"message", "asctime"}
            and not key.startswith("_")
        }
        if extras:
            payload["attributes"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        serializable = json.loads(json.dumps(payload, default=str))
        cleaned, _ = self._redactor.redact(cast(Any, serializable))
        return json.dumps(cleaned, separators=(",", ":"), sort_keys=True)


def configure_structured_logging(
    *,
    level: str = "INFO",
    json_output: bool = True,
    redaction_mode: str = "balanced",
) -> None:
    """Configure the process root logger once at an application entry point."""
    handler = logging.StreamHandler()
    handler.addFilter(TraceContextFilter())
    if json_output:
        handler.setFormatter(JsonFormatter(redaction_mode=redaction_mode))
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
