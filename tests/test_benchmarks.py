"""Direct API benchmark manifests, evaluations, and trace isolation."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import JsonValue, SecretStr

from agenttrace.benchmarks import run_zai_api_benchmark
from agenttrace.benchmarks.models import BenchmarkManifest, BenchmarkRecord
from agenttrace.integrations.telemetry import CaptureMode, TelemetrySettings, create_telemetry
from agenttrace.providers.zai import ZAIEndpointPlan, ZAISettings
from agenttrace.storage import FileObjectStore
from agenttrace.tasks import TaskSample


def _samples() -> list[TaskSample]:
    target: JsonValue = {
        "intent": "billing_refund",
        "urgency": "normal",
        "route": "billing",
    }
    return [
        TaskSample(
            sample_id="billing-001",
            input={"message": "I was charged twice."},
            target=target,
        ),
        TaskSample(
            sample_id="billing-002",
            input={"message": "Please refund the duplicate charge."},
            target=target,
        ),
    ]


def _response() -> dict[str, object]:
    return {
        "id": "chat-safe",
        "model": "glm-5.2",
        "choices": [
            {
                "message": {
                    "content": ('{"intent":"billing_refund","route":"billing","urgency":"normal"}')
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8},
    }


def test_zai_api_benchmark_isolates_samples_and_keeps_metadata_only(
    tmp_path: Path,
    support_task,
) -> None:
    seen_authorization: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_authorization.append(request.headers["authorization"])
        return httpx.Response(200, json=_response())

    exporter = InMemorySpanExporter()
    telemetry = create_telemetry(
        TelemetrySettings(capture_content=CaptureMode.METADATA),
        span_exporter=exporter,
    )
    objects = FileObjectStore(tmp_path / "objects")
    execution = run_zai_api_benchmark(
        task=support_task,
        samples=_samples(),
        settings=ZAISettings(
            api_key=SecretStr("runtime-only-secret"),
            model="glm-5.2",
            max_retries=0,
        ),
        telemetry=telemetry,
        objects=objects,
        experiment_id="glm-support-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    telemetry.close()

    assert execution.manifest.summary.sample_count == 2
    assert execution.manifest.summary.completed_count == 2
    assert execution.manifest.summary.required_pass_count == 2
    assert execution.manifest.summary.input_tokens == 20
    assert execution.manifest.model == "glm-5.2"
    assert execution.manifest.reasoning_effort == "max"
    legacy_manifest = execution.manifest.model_dump(mode="json")
    legacy_manifest.pop("reasoning_effort")
    assert BenchmarkManifest.model_validate(legacy_manifest).reasoning_effort is None
    assert seen_authorization == ["Bearer runtime-only-secret"] * 2
    assert objects.exists(execution.manifest_key)

    stored = b"\n".join(objects.get_bytes(key) for key in objects.list_keys())
    assert b"runtime-only-secret" not in stored
    assert b'billing_refund","route"' not in stored
    assert not any("benchmarks/outputs" in key for key in objects.list_keys())

    legacy_record = json.loads(objects.get_bytes(execution.manifest.records[0].key))
    legacy_record.pop("reasoning_effort")
    assert BenchmarkRecord.model_validate(legacy_record).reasoning_effort is None

    spans = list(exporter.get_finished_spans())
    roots = [span for span in spans if span.name == "agent.episode"]
    assert len(roots) == 2
    assert len({span.context.trace_id for span in roots if span.context is not None}) == 2
    assert sum(span.name == "zai.chat.completions" for span in spans) == 2
    assert sum(span.name == "agent.final_output" for span in spans) == 2
    attributes = json.dumps([dict(span.attributes or {}) for span in spans])
    assert "runtime-only-secret" not in attributes
    assert 'billing_refund","route"' not in attributes


def test_zai_api_benchmark_can_retain_content_addressed_outputs(
    tmp_path: Path,
    support_task,
) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_response()))
    )
    telemetry = create_telemetry(TelemetrySettings(telemetry_enabled=False))
    objects = FileObjectStore(tmp_path / "objects")
    execution = run_zai_api_benchmark(
        task=support_task,
        samples=_samples()[:1],
        settings=ZAISettings(api_key=SecretStr("secret"), max_retries=0),
        telemetry=telemetry,
        objects=objects,
        experiment_id="retained-output-test",
        store_outputs=True,
        http_client=client,
    )
    telemetry.close()

    ref = execution.manifest.records[0]
    record = json.loads(objects.get_bytes(ref.key))
    output_key = record["output_ref"].removeprefix("object://")
    output = objects.get_bytes(output_key)
    assert b"billing_refund" in output
    assert record["output_sha256"] in output_key


def test_zai_api_benchmark_records_safe_provider_errors_and_rejects_coding_plan(
    tmp_path: Path,
    support_task,
) -> None:
    telemetry = create_telemetry(TelemetrySettings(telemetry_enabled=False))
    objects = FileObjectStore(tmp_path / "objects")
    denied = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(401, text="secret provider response body")
        )
    )
    execution = run_zai_api_benchmark(
        task=support_task,
        samples=_samples()[:1],
        settings=ZAISettings(api_key=SecretStr("secret"), max_retries=0),
        telemetry=telemetry,
        objects=objects,
        experiment_id="provider-error-test",
        http_client=denied,
    )
    assert execution.manifest.summary.provider_error_count == 1
    record = json.loads(objects.get_bytes(execution.manifest.records[0].key))
    assert record["error"] == {
        "error_type": "ZAIProviderError",
        "request_id": None,
        "status_code": 401,
    }
    assert "provider response body" not in json.dumps(record)

    with pytest.raises(ValueError, match="OpenCode"):
        run_zai_api_benchmark(
            task=support_task,
            samples=_samples()[:1],
            settings=ZAISettings(
                api_key=SecretStr("secret"),
                endpoint_plan=ZAIEndpointPlan.CODING,
            ),
            telemetry=telemetry,
            objects=objects,
            experiment_id="coding-plan-rejected",
        )
    telemetry.close()
