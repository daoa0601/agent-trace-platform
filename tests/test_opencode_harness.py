"""Isolated OpenCode transport, credential broker, and trace evidence tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from agenttrace.harnesses.opencode import ZAICredentialSource, run_opencode_benchmark
from agenttrace.harnesses.opencode.config import build_opencode_config
from agenttrace.harnesses.opencode.models import OpenCodeBenchmarkRecord
from agenttrace.harnesses.opencode.process import (
    OpenCodeProcessResult,
    isolated_child_environment,
    run_opencode,
)
from agenttrace.harnesses.opencode.proxy import CodingPlanCredentialBroker
from agenttrace.harnesses.opencode.workspace import isolated_workspace
from agenttrace.integrations.telemetry import CaptureMode, TelemetrySettings, create_telemetry
from agenttrace.providers.zai import ZAIReasoningEffort
from agenttrace.storage import FileObjectStore
from agenttrace.tasks import TaskSample, load_task


def test_credential_source_and_child_environment_never_mix_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "environment-secret")
    assert ZAICredentialSource().resolve().get_secret_value() == "environment-secret"

    secret_file = tmp_path / "zai-key"
    secret_file.write_text("file-secret\n")
    with pytest.raises(ValueError, match="only one"):
        ZAICredentialSource(api_key_file=secret_file).resolve()

    child = isolated_child_environment(tmp_path / "runtime")
    assert "ZAI_API_KEY" not in child
    assert "OPENAI_API_KEY" not in child
    assert child["HOME"].startswith(str(tmp_path))


def test_opencode_config_and_process_command_use_only_a_local_capability(tmp_path: Path) -> None:
    payload = build_opencode_config(
        prompt="private benchmark prompt",
        model="glm-5.2",
        small_model="glm-4.7-flash",
        broker_base_url="http://127.0.0.1:12345",
        broker_token="short-lived-capability",
        reasoning_effort=ZAIReasoningEffort.MAX,
    )
    serialized = json.dumps(payload)
    assert "private benchmark prompt" in serialized
    assert "real-provider-secret" not in serialized
    assert payload["share"] == "disabled"
    assert payload["enabled_providers"] == ["zai-coding-plan"]
    assert payload["provider"]["zai-coding-plan"]["npm"] == "@ai-sdk/openai-compatible"
    assert set(payload["provider"]["zai-coding-plan"]["models"]) == {
        "glm-5.2",
        "glm-4.7-flash",
    }
    assert payload["permission"]["webfetch"] == "deny"
    assert payload["provider"]["zai-coding-plan"]["models"]["glm-5.2"]["variants"]["max"] == {
        "reasoningEffort": "max"
    }

    config_path = tmp_path / "opencode.json"
    config_path.write_text(serialized)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = run_opencode(
        executable="true",
        workspace=workspace,
        config_path=config_path,
        runtime_root=tmp_path / "runtime",
        model="glm-5.2",
        reasoning_effort=ZAIReasoningEffort.MAX,
        sample_id="sample-safe",
        timeout_seconds=5,
    )
    assert result.exit_code == 0
    assert "private benchmark prompt" not in " ".join(result.argv)
    assert "short-lived-capability" not in " ".join(result.argv)
    assert result.argv[result.argv.index("--variant") + 1] == "max"


def test_loopback_broker_forwards_only_the_allowed_model_without_revealing_key() -> None:
    captured: dict[str, str] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"choices": []})

    client = httpx.Client(transport=httpx.MockTransport(upstream))
    with CodingPlanCredentialBroker(
        api_key=SecretStr("real-provider-secret"),
        model="glm-5.2",
        max_requests=1,
        http_client=client,
    ) as broker:
        response = httpx.post(
            f"{broker.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {broker.client_token}"},
            json={"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]},
            timeout=5,
        )
        denied = httpx.post(
            f"{broker.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {broker.client_token}"},
            json={"model": "other-model", "messages": []},
            timeout=5,
        )
        over_limit = httpx.post(
            f"{broker.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {broker.client_token}"},
            json={"model": "glm-5.2", "messages": []},
            timeout=5,
        )
        evidence = broker.requests
    client.close()

    assert response.status_code == 200
    assert denied.status_code == 400
    assert over_limit.status_code == 429
    assert captured["authorization"] == "Bearer real-provider-secret"
    assert "real-provider-secret" not in captured["body"]
    forwarded = json.loads(captured["body"])
    assert forwarded["thinking"] == {"type": "enabled"}
    assert forwarded["reasoning_effort"] == "max"
    assert len(evidence) == 1
    assert "real-provider-secret" not in repr(evidence)


def test_opencode_benchmark_collects_sanitized_trace_and_workspace_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_key = "real-coding-secret-123456"
    monkeypatch.setenv("ZAI_API_KEY", real_key)
    root = Path(__file__).resolve().parents[1]
    task = load_task(root / "taskpacks/examples/code-review.yaml")
    sample = TaskSample(
        sample_id="unsafe-yaml",
        input={
            "language": "python",
            "diff": "- yaml.safe_load(text)\n+ yaml.load(text, Loader=yaml.Loader)",
        },
        target="unsafe YAML deserialization",
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("value = 1\n")
    (source / ".env").write_text("DO_NOT_COPY=secret\n")
    (source / ".opencode").mkdir()
    (source / ".opencode" / "plugin.js").write_text("malicious")

    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "glm-5.2",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    broker_client = httpx.Client(transport=httpx.MockTransport(upstream))

    def fake_process(**kwargs) -> OpenCodeProcessResult:
        workspace = kwargs["workspace"]
        config = json.loads(kwargs["config_path"].read_text())
        serialized_config = json.dumps(config)
        assert real_key not in serialized_config
        assert not (workspace / ".env").exists()
        assert not (workspace / ".opencode").exists()
        assert kwargs["reasoning_effort"] == ZAIReasoningEffort.MAX
        provider = config["provider"]["zai-coding-plan"]["options"]
        response = httpx.post(
            f"{provider['baseURL']}/chat/completions",
            headers={"Authorization": f"Bearer {provider['apiKey']}"},
            json={"model": "glm-5.2", "messages": []},
            timeout=5,
        )
        assert response.status_code == 200
        (workspace / "main.py").write_text("value = 2\n")
        events = [
            {
                "type": "tool_use",
                "part": {
                    "tool": "edit",
                    "state": {
                        "input": {"path": "main.py"},
                        "output": f"changed without exposing {real_key}",
                        "status": "completed",
                    },
                },
            },
            {
                "type": "step_finish",
                "part": {
                    "tokens": {
                        "input": 12,
                        "output": 5,
                        "reasoning": 2,
                        "cache": {"read": 3},
                    }
                },
            },
            {"type": "text", "part": {"text": "unsafe YAML deserialization"}},
        ]
        return OpenCodeProcessResult(
            argv=("opencode", "run", "--command", "agenttrace-task"),
            exit_code=0,
            stdout="\n".join(json.dumps(event) for event in events),
            stderr=f"debug value {real_key}",
            elapsed_seconds=1.25,
            timed_out=False,
        )

    exporter = InMemorySpanExporter()
    telemetry = create_telemetry(
        TelemetrySettings(capture_content=CaptureMode.METADATA),
        span_exporter=exporter,
    )
    objects = FileObjectStore(tmp_path / "objects")
    execution = run_opencode_benchmark(
        task=task,
        samples=[sample],
        credential_source=ZAICredentialSource(),
        telemetry=telemetry,
        objects=objects,
        experiment_id="opencode-test",
        workspace_source=source,
        process_runner=fake_process,
        broker_http_client=broker_client,
    )
    telemetry.close()
    broker_client.close()

    assert execution.manifest.summary.completed_count == 1
    assert execution.manifest.summary.model_call_count == 1
    assert execution.manifest.summary.tool_call_count == 1
    ref = execution.manifest.records[0]
    record = json.loads(objects.get_bytes(ref.key))
    assert record["usage"] == {
        "cached_input_tokens": 3,
        "input_tokens": 12,
        "output_tokens": 5,
        "reasoning_tokens": 2,
    }
    assert record["required_evaluators_passed"] is True
    assert record["model"] == "glm-5.2"
    assert record["thinking_mode"] == "enabled"
    assert record["reasoning_effort"] == "max"
    assert record["broker_requests"][0]["reasoning_effort"] == "max"
    legacy_record = dict(record)
    legacy_record.pop("thinking_mode")
    legacy_record.pop("reasoning_effort")
    legacy_record["broker_requests"] = [dict(record["broker_requests"][0])]
    legacy_record["broker_requests"][0].pop("thinking_mode")
    legacy_record["broker_requests"][0].pop("reasoning_effort")
    parsed_legacy = OpenCodeBenchmarkRecord.model_validate(legacy_record)
    assert parsed_legacy.reasoning_effort is None
    assert record["redaction_count"] >= 1
    assert record["workspace_changes"][0]["path"] == "main.py"
    raw_key = record["raw_events_ref"].removeprefix("object://")
    assert b"[REDACTED_SECRET]" in objects.get_bytes(raw_key)
    stored = b"\n".join(objects.get_bytes(key) for key in objects.list_keys())
    assert real_key.encode("utf-8") not in stored

    spans = list(exporter.get_finished_spans())
    assert sum(span.name == "agent.episode" for span in spans) == 1
    assert any(span.name == "opencode.tool.edit" for span in spans)
    assert any(span.name == "opencode.model_generation" for span in spans)
    assert any(span.name == "credential_broker.forward" for span in spans)


def test_workspace_rejects_symlinks_that_escape_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (source / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        with isolated_workspace(source):
            pass
