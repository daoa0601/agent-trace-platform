"""Native and OTLP HTTP ingestion service."""

from __future__ import annotations

import hmac
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from agenttrace.collector.metrics import CollectorMetrics
from agenttrace.collector.otlp import convert_otlp_request
from agenttrace.models import EventBatch, IngestResponse
from agenttrace.redaction import Redactor
from agenttrace.settings import CollectorSettings, get_settings
from agenttrace.storage import FileObjectStore, ObjectStore, RawBatchWriter, S3ObjectStore

log = logging.getLogger(__name__)


def build_object_store(settings: CollectorSettings) -> ObjectStore:
    if settings.storage_backend == "s3":
        return S3ObjectStore(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
        )
    return FileObjectStore(settings.object_root)


class RequestSizeLimitMiddleware:
    """Reject declared or streamed request bodies over the configured limit."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Awaitable[Any]],
        send: Callable[..., Awaitable[Any]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length and int(content_length) > self.max_bytes:
            await self._reject(send)
            return
        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestTooLargeError:
            await self._reject(send)

    @staticmethod
    async def _reject(send: Callable[..., Awaitable[Any]]) -> None:
        body = b'{"detail":"request body too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class RequestTooLargeError(Exception):
    pass


def create_app(
    settings: CollectorSettings | None = None,
    object_store: ObjectStore | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    objects = object_store or build_object_store(settings)
    writer = RawBatchWriter(
        objects,
        inline_payload_bytes=settings.inline_payload_bytes,
        redactor=Redactor(settings.redaction_mode),
    )
    metrics = CollectorMetrics()
    app = FastAPI(
        title="AgentTrace Collector",
        version="0.1.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url=None,
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.state.settings = settings
    app.state.object_store = objects
    app.state.writer = writer
    app.state.metrics = metrics

    def authorize(x_agenttrace_key: str | None = Header(default=None)) -> None:
        keys = settings.parsed_api_keys
        if not keys and settings.environment in {"development", "test"}:
            return
        if x_agenttrace_key is None or not any(
            hmac.compare_digest(x_agenttrace_key, expected) for expected in keys
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")

    def persist(batch: EventBatch, protocol: str) -> IngestResponse:
        started = time.perf_counter()
        try:
            result = writer.write(batch)
        except Exception:
            metrics.requests.labels(protocol=protocol, outcome="error").inc()
            log.exception("raw batch write failed", extra={"protocol": protocol})
            raise HTTPException(status_code=503, detail="raw storage unavailable") from None
        metrics.requests.labels(protocol=protocol, outcome="accepted").inc()
        metrics.events.labels(protocol=protocol).inc(result.accepted)
        if result.duplicate:
            metrics.duplicates.labels(protocol=protocol).inc(result.accepted)
        metrics.raw_bytes.inc(result.bytes_written)
        metrics.observe_redactions(result.redactions)
        metrics.write_seconds.labels(protocol=protocol).observe(time.perf_counter() - started)
        return IngestResponse(
            accepted=result.accepted,
            duplicate_count=result.accepted if result.duplicate else 0,
            batch_ref=f"object://{result.key}",
            event_ids=result.event_ids,
        )

    @app.get("/health/live", include_in_schema=False)
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", include_in_schema=False)
    def ready() -> dict[str, str]:
        try:
            next(objects.list_keys("raw"), None)
        except Exception as error:
            metrics.ready.set(0)
            raise HTTPException(status_code=503, detail="object store unavailable") from error
        metrics.ready.set(1)
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/events",
        response_model=IngestResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authorize)],
    )
    def ingest_events(batch: EventBatch) -> IngestResponse:
        return persist(batch, "native")

    @app.post(
        "/v1/otlp/v1/traces",
        dependencies=[Depends(authorize)],
        include_in_schema=False,
    )
    async def ingest_otlp(request: Request) -> Response:
        body = await request.body()
        try:
            export_request = ExportTraceServiceRequest.FromString(body)
            batch = convert_otlp_request(export_request)
        except (DecodeError, ValueError, TypeError) as error:
            metrics.requests.labels(protocol="otlp", outcome="invalid").inc()
            log.warning("invalid OTLP request: %s", error)
            raise HTTPException(status_code=400, detail=f"invalid OTLP request: {error}") from error
        persist(batch, "otlp")
        response = ExportTraceServiceResponse().SerializeToString()
        return Response(response, media_type="application/x-protobuf")

    return app


app = create_app()
