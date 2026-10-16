"""Collector API, metrics, authentication, and OTLP compatibility."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Status

from agenttrace.collector.app import create_app
from agenttrace.collector.otlp import convert_otlp_request
from agenttrace.models import EventBatch, StepType
from agenttrace.settings import CollectorSettings
from agenttrace.storage import FileObjectStore
from agenttrace.storage.raw import decode_raw_batch

from .conftest import START, TRACE_ID, make_event


def _add_string_attribute(target, key: str, value: str) -> None:
    attribute = target.attributes.add()
    attribute.key = key
    attribute.value.string_value = value


def _add_int_attribute(target, key: str, value: int) -> None:
    attribute = target.attributes.add()
    attribute.key = key
    attribute.value.int_value = value


def make_otlp_request() -> ExportTraceServiceRequest:
    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    _add_string_attribute(resource_spans.resource, "service.instance.id", "worker-otel")
    _add_string_attribute(resource_spans.resource, "agenttrace.experiment.id", "exp_otel")
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.scope.name = "openinference.instrumentation"
    span = scope_spans.spans.add()
    span.trace_id = bytes.fromhex(TRACE_ID)
    span.span_id = bytes.fromhex("3" * 16)
    span.name = "chat completion"
    span.start_time_unix_nano = int(START.timestamp() * 1_000_000_000)
    span.end_time_unix_nano = span.start_time_unix_nano + 1_000_000
    _add_string_attribute(span, "openinference.span.kind", "LLM")
    _add_string_attribute(span, "agenttrace.run.id", "run_otel")
    _add_string_attribute(span, "agenttrace.episode.id", "episode_otel")
    _add_int_attribute(span, "agenttrace.sequence_number", 4)
    _add_string_attribute(span, "gen_ai.request.model", "test-model")
    _add_string_attribute(span, "gen_ai.provider.name", "zai")
    _add_string_attribute(span, "agenttrace.task.id", "support.intent-routing")
    _add_string_attribute(span, "agenttrace.task.version", "1.0.0")
    _add_int_attribute(span, "llm.token_count.prompt", 8)
    _add_int_attribute(span, "llm.token_count.completion", 2)
    _add_string_attribute(span, "input.value", '{"question":"hello"}')
    _add_string_attribute(span, "output.value", '{"answer":"world"}')
    return request


def test_convert_otlp_maps_openinference_and_content() -> None:
    batch = convert_otlp_request(make_otlp_request())
    assert batch.worker_id == "worker-otel"
    event = batch.events[0]
    assert event.step_type == StepType.MODEL_GENERATION
    assert event.sequence_number == 4
    assert event.model is not None and event.model.model_id == "test-model"
    assert event.model.provider == "zai"
    assert event.task is not None and event.task.task_id == "support.intent-routing"
    assert event.metrics.tokens.input_tokens == 8
    assert event.metrics.tokens.output_tokens == 2
    assert event.metrics.latency_ms == 1.0
    assert event.input is not None and event.input.inline == {"question": "hello"}
    assert event.output is not None and event.output.inline == {"answer": "world"}


def test_convert_otlp_preserves_benchmark_stage_and_provenance() -> None:
    request = make_otlp_request()
    span = request.resource_spans[0].scope_spans[0].spans[0]
    _add_string_attribute(span, "agenttrace.step.type", "benchmark")
    _add_string_attribute(
        span,
        "agenttrace.provenance.benchmark_protocol_version",
        "kernel-v1",
    )
    _add_string_attribute(
        span,
        "agenttrace.provenance.final_artifact_digest",
        "a" * 64,
    )

    event = convert_otlp_request(request).events[0]
    assert event.step_type == StepType.BENCHMARK
    assert event.provenance.benchmark_protocol_version == "kernel-v1"
    assert event.provenance.final_artifact_digest == "a" * 64


def test_convert_otlp_infers_sequence_and_rejects_empty_request() -> None:
    request = make_otlp_request()
    request.resource_spans[0].scope_spans[0].spans[0].attributes.pop(3)
    batch = convert_otlp_request(request)
    assert batch.events[0].attributes["agenttrace.sequence.inferred"] is True
    try:
        convert_otlp_request(ExportTraceServiceRequest())
    except ValueError as error:
        assert "no spans" in str(error)
    else:
        raise AssertionError("empty OTLP request should fail")


def test_native_collector_auth_ingestion_idempotency_and_metrics(
    tmp_path: Path,
    support_task,
) -> None:
    objects = FileObjectStore(tmp_path / "objects")
    settings = CollectorSettings(
        environment="test",
        api_keys="secret-key",
        object_root=tmp_path / "unused",
        max_request_bytes=100_000,
    )
    client = TestClient(create_app(settings, objects))
    event = make_event(1, StepType.USER_INPUT, task=support_task, input={"message": "hello"})
    payload = EventBatch(worker_id="api-test", events=[event]).model_dump(mode="json")
    assert client.post("/v1/events", json=payload).status_code == 401
    first = client.post("/v1/events", json=payload, headers={"X-AgentTrace-Key": "secret-key"})
    assert first.status_code == 202
    assert first.json()["accepted"] == 1
    second = client.post("/v1/events", json=payload, headers={"X-AgentTrace-Key": "secret-key"})
    assert second.json()["duplicate_count"] == 1
    raw_keys = list(objects.list_keys("raw"))
    assert len(raw_keys) == 1
    assert decode_raw_batch(objects.get_bytes(raw_keys[0]))[0].event_id == event.event_id
    metrics = client.get("/metrics").text
    assert 'agenttrace_ingested_events_total{protocol="native"} 2.0' in metrics
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").status_code == 200


def test_otlp_http_endpoint_writes_native_raw_object(tmp_path: Path) -> None:
    objects = FileObjectStore(tmp_path / "objects")
    settings = CollectorSettings(
        environment="test",
        api_keys="otel-key",
        object_root=tmp_path / "unused",
    )
    client = TestClient(create_app(settings, objects))
    response = client.post(
        "/v1/otlp/v1/traces",
        content=make_otlp_request().SerializeToString(),
        headers={
            "Content-Type": "application/x-protobuf",
            "X-AgentTrace-Key": "otel-key",
        },
    )
    assert response.status_code == 200
    keys = list(objects.list_keys("raw"))
    assert len(keys) == 1
    assert decode_raw_batch(objects.get_bytes(keys[0]))[0].experiment_id == "exp_otel"


def test_collector_rejects_oversized_and_invalid_otlp(tmp_path: Path) -> None:
    settings = CollectorSettings(
        environment="test",
        object_root=tmp_path / "objects",
        max_request_bytes=1024,
    )
    client = TestClient(create_app(settings))
    oversized = client.post(
        "/v1/events",
        content=b"x" * 2048,
        headers={"Content-Type": "application/json"},
    )
    assert oversized.status_code == 413
    invalid = client.post(
        "/v1/otlp/v1/traces",
        content=b"not protobuf",
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert invalid.status_code == 400


def test_convert_otlp_preserves_failure_status_on_non_tool_spans() -> None:
    request = make_otlp_request()
    span = request.resource_spans[0].scope_spans[0].spans[0]
    _add_string_attribute(span, "agenttrace.step.type", "agent_state")
    span.status.code = Status.STATUS_CODE_ERROR
    span.status.message = "forward_incomplete"

    event = convert_otlp_request(request).events[0]
    assert event.step_type == StepType.AGENT_STATE
    assert event.attributes["otel.status_code"] == "ERROR"
    assert event.attributes["otel.status_description"] == "forward_incomplete"

    ok = convert_otlp_request(make_otlp_request()).events[0]
    assert "otel.status_code" not in ok.attributes


def test_convert_otlp_records_tool_error_type() -> None:
    request = make_otlp_request()
    span = request.resource_spans[0].scope_spans[0].spans[0]
    _add_string_attribute(span, "agenttrace.step.type", "tool_result")
    _add_string_attribute(span, "tool.name", "bash")
    span.status.code = Status.STATUS_CODE_ERROR
    span.status.message = "tool_error"

    event = convert_otlp_request(request).events[0]
    assert event.tool is not None
    assert event.tool.success is False
    assert event.tool.error_type == "tool_error"


def test_convert_otlp_references_long_status_descriptions_by_digest() -> None:
    import hashlib

    request = make_otlp_request()
    span = request.resource_spans[0].scope_spans[0].spans[0]
    _add_string_attribute(span, "agenttrace.step.type", "tool_result")
    span.status.code = Status.STATUS_CODE_ERROR
    span.status.message = "x" * 100_000

    event = convert_otlp_request(request).events[0]
    assert "otel.status_description" not in event.attributes
    assert event.attributes["otel.status_description_bytes"] == 100_000
    assert (
        event.attributes["otel.status_description_sha256"]
        == hashlib.sha256(b"x" * 100_000).hexdigest()
    )
    assert event.tool is not None and event.tool.error_type == "error"
