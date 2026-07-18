"""Z.AI endpoint isolation, secret handling, response parsing, and CLI safety."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from typer.testing import CliRunner

from agenttrace.cli import app
from agenttrace.models import TokenUsage
from agenttrace.providers.zai import (
    CODING_PLAN_API_BASE,
    GENERAL_API_BASE,
    ZAIChatResponse,
    ZAIClient,
    ZAIEndpointPlan,
    ZAIProviderError,
    ZAIReasoningEffort,
    ZAISettings,
    ZAIThinkingMode,
)


def _settings(**kwargs) -> ZAISettings:
    return ZAISettings(api_key=SecretStr("zai-test-secret"), max_retries=0, **kwargs)


def test_zai_settings_keep_endpoint_plans_and_secret_sources_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    general = _settings(endpoint_plan=ZAIEndpointPlan.GENERAL)
    coding = _settings(endpoint_plan=ZAIEndpointPlan.CODING)
    assert general.resolved_base_url == GENERAL_API_BASE
    assert coding.resolved_base_url == CODING_PLAN_API_BASE
    assert general.chat_completions_url.endswith("/chat/completions")
    assert "zai-test-secret" not in repr(general)

    secret_path = tmp_path / "zai-key"
    secret_path.write_text("file-secret\n")
    from_file = ZAISettings(api_key_file=secret_path)
    assert from_file.resolve_api_key() == "file-secret"
    with pytest.raises(ValueError, match="only one"):
        ZAISettings(api_key=SecretStr("a"), api_key_file=secret_path)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_API_KEY_FILE", raising=False)
    with pytest.raises(ValueError, match="set ZAI_API_KEY"):
        ZAISettings().resolve_api_key()
    with pytest.raises(ValidationError, match="HTTPS"):
        _settings(base_url="http://example.com/v4")
    assert _settings(base_url="http://localhost:8080/v4").resolved_base_url.startswith("http://")


def test_zai_client_parses_usage_and_never_sends_key_in_body() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "request-123"},
            json={
                "id": "chat-1",
                "model": "glm-5.2",
                "choices": [{"message": {"content": "READY"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = ZAIClient(_settings(endpoint_plan=ZAIEndpointPlan.CODING), http_client=http_client)
    response = client.complete([{"role": "user", "content": "hello"}], max_tokens=10)
    client.close()
    assert response.content == "READY"
    assert response.request_id == "request-123"
    assert response.usage == TokenUsage(
        input_tokens=8,
        output_tokens=2,
        cached_input_tokens=3,
        reasoning_tokens=1,
    )
    assert captured["authorization"] == "Bearer zai-test-secret"
    captured_body = captured["body"]
    assert isinstance(captured_body, dict)
    assert "zai-test-secret" not in json.dumps(captured_body)
    assert captured_body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in captured_body

    enabled_body: dict[str, object] = {}

    def enabled_handler(request: httpx.Request) -> httpx.Response:
        enabled_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "request_id": "body-request-id",
                "model": "glm-5.2",
                "choices": [{"message": {"content": "READY"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    enabled_client = ZAIClient(
        _settings(
            thinking_mode=ZAIThinkingMode.ENABLED,
            reasoning_effort=ZAIReasoningEffort.HIGH,
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(enabled_handler)),
    )
    enabled_response = enabled_client.complete([{"role": "user", "content": "hello"}])
    assert enabled_body["thinking"] == {"type": "enabled"}
    assert enabled_body["reasoning_effort"] == "high"
    assert enabled_response.request_id == "body-request-id"


def test_zai_client_errors_exclude_provider_body_and_key() -> None:
    def denied(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="zai-test-secret leaked provider detail")

    client = ZAIClient(
        _settings(),
        http_client=httpx.Client(transport=httpx.MockTransport(denied)),
    )
    with pytest.raises(ZAIProviderError) as captured:
        client.complete([{"role": "user", "content": "hello"}])
    assert captured.value.status_code == 401
    assert "zai-test-secret" not in str(captured.value)
    assert "provider detail" not in str(captured.value)

    def invalid(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    invalid_client = ZAIClient(
        _settings(),
        http_client=httpx.Client(transport=httpx.MockTransport(invalid)),
    )
    with pytest.raises(ZAIProviderError, match="invalid"):
        invalid_client.complete([{"role": "user", "content": "hello"}])
    with pytest.raises(ValueError, match="cannot be empty"):
        invalid_client.complete([])


def test_zai_smoke_cli_rejects_coding_plan_and_never_prints_content(monkeypatch) -> None:
    runner = CliRunner()
    rejected = runner.invoke(app, ["provider", "zai-smoke", "--endpoint-plan", "coding"])
    assert rejected.exit_code != 0
    assert "OpenCode" in rejected.output

    monkeypatch.setenv("ZAI_API_KEY", "runtime-only-secret")

    def fake_complete(self, messages, *, max_tokens=64, temperature=0.0):
        del self, messages, max_tokens, temperature
        return ZAIChatResponse(
            content="READY",
            model="glm-5.2",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            request_id="req-safe",
        )

    monkeypatch.setattr(ZAIClient, "complete", fake_complete)
    result = runner.invoke(
        app,
        [
            "provider",
            "zai-smoke",
            "--endpoint-plan",
            "general",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "READY" not in result.output
    assert "runtime-only-secret" not in result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["content_printed"] is False
