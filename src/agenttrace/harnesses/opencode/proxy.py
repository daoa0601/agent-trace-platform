"""Loopback credential broker for a supported OpenCode Coding Plan session."""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
from pydantic import SecretStr

from agenttrace.providers.zai import (
    CODING_PLAN_API_BASE,
    ZAIReasoningEffort,
    ZAIThinkingMode,
)


@dataclass(frozen=True)
class BrokerRequest:
    sequence: int
    status_code: int
    latency_ms: float
    request_bytes: int
    response_bytes: int
    thinking_mode: str
    reasoning_effort: str


class CodingPlanCredentialBroker:
    """Inject the real key upstream without placing it in the agent process.

    This is intentionally a narrow forwarder: one model, one endpoint, bounded calls and payloads,
    loopback only. It is not a general-purpose HTTP proxy.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        reasoning_effort: ZAIReasoningEffort = ZAIReasoningEffort.MAX,
        max_requests: int = 256,
        max_request_bytes: int = 4 * 1024 * 1024,
        timeout_seconds: float = 600.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        self._api_key = api_key.get_secret_value()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_requests = max_requests
        self.max_request_bytes = max_request_bytes
        self.timeout_seconds = timeout_seconds
        self.client_token = secrets.token_urlsafe(32)
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = http_client is None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._requests: list[BrokerRequest] = []
        self._claimed_requests = 0

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("credential broker is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def requests(self) -> tuple[BrokerRequest, ...]:
        with self._lock:
            return tuple(sorted(self._requests, key=lambda request: request.sequence))

    def _claim_request(self) -> int | None:
        """Atomically reserve one upstream request slot and its stable sequence number."""
        with self._lock:
            if self._claimed_requests >= self.max_requests:
                return None
            sequence = self._claimed_requests
            self._claimed_requests += 1
            return sequence

    def _append_request(
        self,
        *,
        sequence: int,
        status_code: int,
        latency_ms: float,
        request_bytes: int,
        response_bytes: int,
        thinking_mode: str,
        reasoning_effort: str,
    ) -> None:
        with self._lock:
            self._requests.append(
                BrokerRequest(
                    sequence=sequence,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                    thinking_mode=thinking_mode,
                    reasoning_effort=reasoning_effort,
                )
            )

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        broker = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AgentTraceCredentialBroker/1.0"

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

            def _json_error(self, status: int, message: str) -> None:
                body = json.dumps({"error": {"message": message}}).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                started = time.perf_counter()
                authorization = self.headers.get("Authorization", "")
                if authorization != f"Bearer {broker.client_token}":
                    self._json_error(401, "invalid local broker capability")
                    return
                if not self.path.split("?", 1)[0].endswith("/chat/completions"):
                    self._json_error(404, "unsupported broker endpoint")
                    return
                raw_length = self.headers.get("Content-Length", "0")
                try:
                    content_length = int(raw_length)
                except ValueError:
                    self._json_error(400, "invalid content length")
                    return
                if content_length < 1 or content_length > broker.max_request_bytes:
                    self._json_error(413, "request payload is outside the allowed size")
                    return
                body = self.rfile.read(content_length)
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    self._json_error(400, "request body is not JSON")
                    return
                if not isinstance(payload, dict) or payload.get("model") != broker.model:
                    self._json_error(400, "request model is not allowed for this session")
                    return
                payload["thinking"] = {"type": ZAIThinkingMode.ENABLED.value}
                payload["reasoning_effort"] = broker.reasoning_effort.value
                upstream_body = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                sequence = broker._claim_request()
                if sequence is None:
                    self._json_error(429, "session model-call limit reached")
                    return
                try:
                    upstream = broker._client.post(
                        f"{CODING_PLAN_API_BASE}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {broker._api_key}",
                            "Content-Type": "application/json",
                            "Accept": self.headers.get("Accept", "application/json"),
                            "User-Agent": self.headers.get("User-Agent", "opencode"),
                        },
                        content=upstream_body,
                        timeout=broker.timeout_seconds,
                    )
                    response_body = upstream.content
                    status_code = upstream.status_code
                    content_type = upstream.headers.get("Content-Type", "application/json")
                except httpx.HTTPError:
                    self._json_error(502, "upstream model request failed")
                    broker._append_request(
                        sequence=sequence,
                        status_code=502,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        request_bytes=len(body),
                        response_bytes=0,
                        thinking_mode=ZAIThinkingMode.ENABLED.value,
                        reasoning_effort=broker.reasoning_effort.value,
                    )
                    return
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
                broker._append_request(
                    sequence=sequence,
                    status_code=status_code,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    request_bytes=len(body),
                    response_bytes=len(response_body),
                    thinking_mode=ZAIThinkingMode.ENABLED.value,
                    reasoning_effort=broker.reasoning_effort.value,
                )

        return Handler

    def __enter__(self) -> CodingPlanCredentialBroker:
        if self._server is not None:
            raise RuntimeError("credential broker cannot be entered twice")
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._owns_client:
            self._client.close()
        self._api_key = ""
        self.client_token = ""
        self._server = None
        self._thread = None
