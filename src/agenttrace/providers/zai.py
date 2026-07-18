"""Secure OpenAI-compatible Z.AI client with explicit endpoint-plan selection."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from pydantic import Field, JsonValue, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agenttrace.integrations.telemetry import AgentRun, SpanResult
from agenttrace.models import TokenUsage

GENERAL_API_BASE = "https://api.z.ai/api/paas/v4"
CODING_PLAN_API_BASE = "https://api.z.ai/api/coding/paas/v4"


class ZAIEndpointPlan(StrEnum):
    """Billing/entitlement boundary selected by the caller."""

    GENERAL = "general"
    CODING = "coding"


class ZAIThinkingMode(StrEnum):
    """Provider-supported turn-level reasoning mode."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class ZAIReasoningEffort(StrEnum):
    """GLM-5.2 reasoning effort accepted by the Z.AI API."""

    MAX = "max"
    XHIGH = "xhigh"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"
    NONE = "none"


class ZAISettings(BaseSettings):
    """Z.AI settings; credentials are accepted only through env or a secret file."""

    model_config = SettingsConfigDict(env_prefix="ZAI_", env_file=None, extra="ignore")

    api_key: SecretStr | None = None
    api_key_file: Path | None = None
    endpoint_plan: ZAIEndpointPlan = ZAIEndpointPlan.GENERAL
    base_url: str | None = None
    model: str = "glm-5.2"
    thinking_mode: ZAIThinkingMode = ZAIThinkingMode.DISABLED
    reasoning_effort: ZAIReasoningEffort = ZAIReasoningEffort.MAX
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlparse(value)
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise ValueError(
                "ZAI_BASE_URL must use HTTPS (HTTP is allowed only for localhost tests)"
            )
        if not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("ZAI_BASE_URL must be an absolute base URL without query or fragment")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_secret_sources(self) -> ZAISettings:
        if self.api_key is not None and self.api_key_file is not None:
            raise ValueError("set only one of ZAI_API_KEY or ZAI_API_KEY_FILE")
        return self

    @property
    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return (
            CODING_PLAN_API_BASE
            if self.endpoint_plan == ZAIEndpointPlan.CODING
            else GENERAL_API_BASE
        )

    @property
    def chat_completions_url(self) -> str:
        return f"{self.resolved_base_url.rstrip('/')}/chat/completions"

    def resolve_api_key(self) -> str:
        if self.api_key is not None:
            value = self.api_key.get_secret_value().strip()
        elif self.api_key_file is not None:
            path = self.api_key_file.expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"ZAI_API_KEY_FILE is not a readable file: {path}")
            value = path.read_text(encoding="utf-8").strip()
        else:
            raise ValueError("set ZAI_API_KEY or ZAI_API_KEY_FILE")
        if not value:
            raise ValueError("Z.AI API key is empty")
        return value


@dataclass(frozen=True)
class ZAIChatResponse:
    content: str
    model: str
    finish_reason: str | None
    usage: TokenUsage
    request_id: str | None


class ZAIProviderError(RuntimeError):
    """Provider failure whose message deliberately excludes response bodies and credentials."""

    def __init__(self, message: str, *, status_code: int | None, request_id: str | None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


def _int(value: object) -> int:
    return (
        int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
    )


def _usage(payload: dict[str, Any]) -> TokenUsage:
    prompt_details = payload.get("prompt_tokens_details")
    completion_details = payload.get("completion_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    return TokenUsage(
        input_tokens=_int(payload.get("prompt_tokens")),
        output_tokens=_int(payload.get("completion_tokens")),
        cached_input_tokens=_int(prompt_details.get("cached_tokens")),
        reasoning_tokens=_int(completion_details.get("reasoning_tokens")),
    )


class ZAIClient:
    """Small synchronous harness suitable for smoke tests and deterministic eval adapters."""

    def __init__(
        self,
        settings: ZAISettings | None = None,
        *,
        trace_run: AgentRun | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or ZAISettings()
        self.trace_run = trace_run
        self._api_key = self.settings.resolve_api_key()
        self._client = http_client or httpx.Client(timeout=self.settings.timeout_seconds)
        self._owns_client = http_client is None

    def complete(
        self,
        messages: list[dict[str, JsonValue]],
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> ZAIChatResponse:
        if not messages:
            raise ValueError("messages cannot be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        request_payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "thinking": {"type": self.settings.thinking_mode.value},
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self.settings.thinking_mode == ZAIThinkingMode.ENABLED:
            request_payload["reasoning_effort"] = self.settings.reasoning_effort.value
        span_context = (
            self.trace_run.span(
                "zai.chat.completions",
                step_type="model_generation",
                kind="LLM",
                input=messages,
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "zai",
                    "gen_ai.request.model": self.settings.model,
                    "gen_ai.request.max_tokens": max_tokens,
                    "gen_ai.request.temperature": temperature,
                    "agenttrace.provider.endpoint_plan": self.settings.endpoint_plan.value,
                    "agenttrace.provider.thinking_mode": self.settings.thinking_mode.value,
                    "agenttrace.provider.reasoning_effort": self.settings.reasoning_effort.value,
                },
            )
            if self.trace_run is not None
            else nullcontext(SpanResult())
        )
        with span_context as span_result:
            response = self._post_with_retries(request_payload)
            result = self._parse_response(response)
            span_result.output = result.content
            span_result.input_tokens = result.usage.input_tokens
            span_result.output_tokens = result.usage.output_tokens
            span_result.cached_input_tokens = result.usage.cached_input_tokens
            span_result.reasoning_tokens = result.usage.reasoning_tokens
            span_result.attributes.update(
                {
                    "gen_ai.response.model": result.model,
                    "gen_ai.response.finish_reasons": [result.finish_reason]
                    if result.finish_reason
                    else [],
                }
            )
            if result.request_id:
                span_result.attributes["gen_ai.response.id"] = result.request_id
            return result

    def _post_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "agenttrace-platform/0.1",
        }
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self._client.post(
                    self.settings.chat_completions_url,
                    headers=headers,
                    json=payload,
                )
            except httpx.HTTPError as error:
                last_error = error
                if attempt < self.settings.max_retries:
                    time.sleep(min(0.5 * (2**attempt), 4.0))
                    continue
                raise ZAIProviderError(
                    "Z.AI request failed before a response was received",
                    status_code=None,
                    request_id=None,
                ) from error
            request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
            retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
            if response.is_success:
                return response
            if retryable and attempt < self.settings.max_retries:
                retry_after = response.headers.get("retry-after")
                delay = (
                    min(float(retry_after), 10.0)
                    if retry_after and retry_after.isdigit()
                    else min(0.5 * (2**attempt), 4.0)
                )
                time.sleep(delay)
                continue
            raise ZAIProviderError(
                f"Z.AI returned HTTP {response.status_code}",
                status_code=response.status_code,
                request_id=request_id,
            )
        assert last_error is not None
        raise ZAIProviderError(
            "Z.AI request failed",
            status_code=None,
            request_id=None,
        ) from last_error

    @staticmethod
    def _parse_response(response: httpx.Response) -> ZAIChatResponse:
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        try:
            payload = response.json()
            payload_request_id = payload.get("request_id") or payload.get("id")
            if request_id is None and isinstance(payload_request_id, str):
                request_id = payload_request_id
            choices = payload["choices"]
            choice = choices[0]
            content = choice["message"]["content"]
            model = payload.get("model") or "unknown"
            usage_payload = payload.get("usage") or {}
            if (
                not isinstance(content, str)
                or not isinstance(model, str)
                or not isinstance(usage_payload, dict)
            ):
                raise TypeError
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ZAIProviderError(
                "Z.AI returned an invalid chat-completions payload",
                status_code=response.status_code,
                request_id=request_id,
            ) from error
        finish_reason = choice.get("finish_reason")
        return ZAIChatResponse(
            content=content,
            model=model,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=_usage(cast(dict[str, Any], usage_payload)),
            request_id=request_id,
        )

    def close(self) -> None:
        self._api_key = ""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ZAIClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
