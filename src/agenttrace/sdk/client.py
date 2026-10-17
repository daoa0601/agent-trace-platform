"""Thread-safe buffered HTTP client with a durable local failure spool."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from agenttrace.ids import new_id
from agenttrace.models import (
    EventBatch,
    IngestResponse,
    Privacy,
    Provenance,
    TaskRef,
    TraceEvent,
    check_ingestible,
)

if TYPE_CHECKING:
    from agenttrace.sdk.session import TraceSession

log = logging.getLogger(__name__)


class TraceDeliveryError(RuntimeError):
    """Raised when a batch cannot be delivered and strict delivery is enabled."""


class TraceClient:
    """Buffered native collector client suitable for embedding in agent runtimes."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        worker_id: str = "agent-worker",
        batch_size: int = 100,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        spool_dir: Path | None = None,
        raise_on_failure: bool = False,
        http_client: httpx.Client | None = None,
    ) -> None:
        if batch_size < 1 or batch_size > 1000:
            raise ValueError("batch_size must be between 1 and 1000")
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.spool_dir = spool_dir
        self.raise_on_failure = raise_on_failure
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = http_client is None
        self._buffer: list[TraceEvent] = []
        self._lock = threading.RLock()

    def record(self, event: TraceEvent) -> IngestResponse | None:
        """Buffer an event and flush when the configured batch size is reached."""
        # Reject here: a buffered event the collector refuses would fail every later flush.
        check_ingestible(event)
        with self._lock:
            if self._buffer:
                first = self._buffer[0]
                if (
                    first.experiment_id != event.experiment_id
                    or first.started_at.date() != event.started_at.date()
                ):
                    self.flush()
            self._buffer.append(event)
            if len(self._buffer) >= self.batch_size:
                return self.flush()
        return None

    def flush(self) -> IngestResponse | None:
        """Deliver the current buffer or persist it to the local spool."""
        with self._lock:
            if not self._buffer:
                return None
            batch = EventBatch(worker_id=self.worker_id, events=list(self._buffer))
            self._buffer.clear()
        try:
            return self._send(batch)
        except Exception as error:
            spool_path = self._spool(batch) if self.spool_dir is not None else None
            message = f"failed to deliver {len(batch.events)} trace events"
            if spool_path is not None:
                message += f"; spooled at {spool_path}"
            if self.raise_on_failure:
                raise TraceDeliveryError(message) from error
            log.warning(message, exc_info=True)
            return None

    def replay_spool(self) -> tuple[int, int]:
        """Replay durable batches; return (delivered files, remaining files)."""
        if self.spool_dir is None or not self.spool_dir.exists():
            return 0, 0
        delivered = 0
        for path in sorted(self.spool_dir.glob("batch-*.json")):
            try:
                batch = EventBatch.model_validate_json(path.read_bytes())
                self._send(batch)
                path.unlink()
                delivered += 1
            except Exception:
                log.warning("spool replay failed for %s", path, exc_info=True)
        remaining = len(list(self.spool_dir.glob("batch-*.json")))
        return delivered, remaining

    def _send(self, batch: EventBatch) -> IngestResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-AgentTrace-Key"] = self.api_key
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(
                    f"{self.endpoint}/v1/events",
                    content=batch.model_dump_json(exclude_none=True),
                    headers=headers,
                )
                response.raise_for_status()
                return IngestResponse.model_validate(response.json())
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2**attempt), 2.0))
        assert last_error is not None
        raise last_error

    def _spool(self, batch: EventBatch) -> Path:
        assert self.spool_dir is not None
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        path = self.spool_dir / f"batch-{new_id('spool')}.json"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(batch.model_dump_json(exclude_none=True))
            handle.write("\n")
        return path

    def session(
        self,
        *,
        experiment_id: str,
        run_id: str | None = None,
        episode_id: str | None = None,
        task: TaskRef | None = None,
        provenance: Provenance | None = None,
        privacy: Privacy | None = None,
        trace_id: str | None = None,
    ) -> TraceSession:
        """Create a TraceSession without introducing an import cycle at module load."""
        from agenttrace.sdk.session import TraceSession

        return TraceSession(
            client=self,
            experiment_id=experiment_id,
            run_id=run_id,
            episode_id=episode_id,
            task=task,
            provenance=provenance,
            privacy=privacy,
            trace_id=trace_id,
        )

    def close(self) -> None:
        self.flush()
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TraceClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
