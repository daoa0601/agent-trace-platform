"""OpenInference telemetry, structured logging, and Inspect adapter tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr

from agenttrace.integrations.logging import JsonFormatter
from agenttrace.integrations.telemetry import (
    CaptureMode,
    TelemetrySettings,
    create_telemetry,
    current_trace_fields,
)
from agenttrace.tasks import load_task
from agenttrace.tasks.models import EnvironmentKind, EvaluatorSpec


def test_telemetry_emits_linked_openinference_spans_and_redacts_content() -> None:
    exporter = InMemorySpanExporter()
    runtime = create_telemetry(
        TelemetrySettings(
            capture_content=CaptureMode.FULL,
            service_name="test-agent",
            otlp_headers=SecretStr("authorization=secret"),
        ),
        span_exporter=exporter,
    )
    assert runtime.settings.parsed_headers() == {"authorization": "secret"}
    with runtime.run(
        experiment_id="exp_test",
        session_id="session-test",
        trace_name="support routing test",
        run_id="run_test",
        episode_id="episode_test",
        task_id="support.intent-routing",
        task_version="1.0.0",
    ) as run:
        fields = current_trace_fields()
        assert fields["run_id"] == "run_test"
        assert len(fields["trace_id"]) == 32
        with run.span(
            "model.generate",
            step_type="model_generation",
            kind="LLM",
            input={"api_key": "must-not-escape", "message": "hello"},
        ) as result:
            result.output = {"answer": "world"}
            result.input_tokens = 4
            result.output_tokens = 2
            result.reasoning_tokens = 1
            result.cost_usd = 0.01
            result.attributes["gen_ai.request.model"] = "test-model"
        run.emit_legacy(event="tool_selected", path="tests.runtime", data={"tool": "search"})
    runtime.close()

    spans = list(exporter.get_finished_spans())
    assert {span.name for span in spans} == {
        "support routing test",
        "agent.episode.end",
        "model.generate",
        "legacy.tool_selected",
    }
    sequences: list[int] = []
    for span in spans:
        sequence = dict(span.attributes or {}).get("agenttrace.sequence_number")
        assert isinstance(sequence, int)
        sequences.append(sequence)
    sequences.sort()
    assert sequences == [0, 1, 2, 3]
    assert all(dict(span.attributes or {})["session.id"] == "session-test" for span in spans)
    model = next(span for span in spans if span.name == "model.generate")
    model_attributes = dict(model.attributes or {})
    assert model_attributes["openinference.span.kind"] == "LLM"
    assert model_attributes["llm.token_count.prompt"] == 4
    assert model_attributes["llm.token_count.completion"] == 2
    assert "must-not-escape" not in str(model_attributes["input.value"])
    assert "[REDACTED_SECRET]" in str(model_attributes["input.value"])
    root = next(span for span in spans if span.name == "support routing test")
    assert root.parent is None
    assert root.context is not None
    for span in spans:
        if span is root:
            continue
        assert span.parent is not None
        assert span.parent.span_id == root.context.span_id


def test_telemetry_metadata_disabled_and_error_paths() -> None:
    with pytest.raises(ValueError, match="key=value"):
        TelemetrySettings(otlp_headers=SecretStr("invalid")).parsed_headers()

    disabled = create_telemetry(TelemetrySettings(telemetry_enabled=False))
    with disabled.run(experiment_id="disabled") as run:
        with run.span("noop") as result:
            result.output = "not exported"
    assert disabled.force_flush()
    disabled.close()

    exporter = InMemorySpanExporter()
    runtime = create_telemetry(
        TelemetrySettings(capture_content=CaptureMode.METADATA),
        span_exporter=exporter,
    )
    with pytest.raises(RuntimeError, match="boom"):
        with runtime.run(experiment_id="exp_error") as run:
            with pytest.raises(RuntimeError, match="entered twice"):
                run.__enter__()
            with run.span("failing", input="secret text"):
                raise RuntimeError("boom")
    failing = next(span for span in exporter.get_finished_spans() if span.name == "failing")
    failing_attributes = dict(failing.attributes or {})
    assert failing_attributes["agenttrace.step.type"] == "error"
    assert "input.value" not in failing_attributes
    input_bytes = failing_attributes["agenttrace.input.bytes"]
    assert isinstance(input_bytes, int) and input_bytes > 0
    runtime.close()

    with pytest.raises(ValueError, match="printable ASCII"):
        runtime.run(experiment_id="invalid", session_id="not\nprintable")


def test_span_result_error_type_marks_a_completed_span_as_failed() -> None:
    exporter = InMemorySpanExporter()
    runtime = create_telemetry(TelemetrySettings(), span_exporter=exporter)
    with runtime.run(experiment_id="exp_tool_error") as run:
        with run.span("tool.failed", step_type="tool_call", kind="TOOL") as result:
            result.error_type = "tool_error"
        with run.span("tool.ok", step_type="tool_call", kind="TOOL"):
            pass
    runtime.close()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["tool.failed"].status.status_code == StatusCode.ERROR
    assert spans["tool.failed"].status.description == "tool_error"
    assert spans["tool.ok"].status.status_code == StatusCode.OK


def test_telemetry_emits_explicit_benchmark_stages() -> None:
    exporter = InMemorySpanExporter()
    runtime = create_telemetry(
        TelemetrySettings(capture_content=CaptureMode.METADATA),
        span_exporter=exporter,
    )
    with runtime.run(experiment_id="benchmark-test") as run:
        with run.benchmark_stage("compile", input={"source_sha256": "a" * 64}) as result:
            result.output = {"exit_code": 0}
        with run.benchmark_stage("check") as result:
            result.output = {"passed": True}
        with run.benchmark_stage("benchmark") as result:
            result.output = {"samples_sha256": "b" * 64}
        with run.benchmark_stage("audit_retest") as result:
            result.output = {"passed": True}
        with pytest.raises(ValueError, match="unsupported benchmark stage"):
            run.benchmark_stage("publish")
    runtime.close()

    attributes_by_name = {
        span.name: dict(span.attributes or {}) for span in exporter.get_finished_spans()
    }
    assert attributes_by_name["benchmark.compile"]["agenttrace.step.type"] == "compile"
    assert attributes_by_name["benchmark.check"]["agenttrace.step.type"] == "check"
    assert attributes_by_name["benchmark.benchmark"]["agenttrace.step.type"] == "benchmark"
    assert attributes_by_name["benchmark.audit_retest"]["agenttrace.step.type"] == "audit_retest"


def test_json_formatter_redacts_and_adds_structured_extras() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="Authorization: Bearer abcdefghijklmnop",
        args=(),
        exc_info=None,
    )
    record.password = "plain-secret"
    record.run_id = "run_log"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["severity"] == "WARNING"
    assert payload["run_id"] == "run_log"
    assert "abcdefghijklmnop" not in payload["message"]
    assert payload["attributes"]["password"] == "[REDACTED_SECRET]"


def test_inspect_adapter_builds_coding_and_non_coding_tasks(tmp_path: Path) -> None:
    from inspect_ai.scorer import Target

    from agenttrace.integrations.inspect_ai import (
        InspectSample,
        _evaluate_one,
        build_inspect_task,
        load_inspect_samples,
    )

    root = Path(__file__).resolve().parents[1]
    support = load_task(root / "taskpacks/examples/support-routing.yaml")
    samples = load_inspect_samples(root / "examples/inspect/support-routing.jsonl")
    inspect_task = build_inspect_task(support, samples)
    assert inspect_task.name == support.task_id
    assert len(inspect_task.dataset) == 1
    assert inspect_task.metadata is not None
    assert inspect_task.metadata["post_run_evaluators"] == ["latency"]
    assert "Return only JSON" in inspect_task.dataset[0].input

    target = Target(json.dumps(support.reference))
    exact = next(spec for spec in support.evaluators if spec.kind == "exact_match")
    assert _evaluate_one(
        exact, support, f"```json\n{json.dumps(support.reference)}\n```", target
    ) == (
        True,
        None,
    )
    schema = next(spec for spec in support.evaluators if spec.kind == "json_schema")
    passed, explanation = _evaluate_one(schema, support, "not json", target)
    assert not passed and explanation == "output is not JSON"

    duplicate_path = tmp_path / "duplicates.jsonl"
    duplicate_path.write_text(
        "\n".join(
            [
                '{"sample_id":"same","input":"a"}',
                '{"sample_id":"same","input":"b"}',
            ]
        )
    )
    with pytest.raises(ValueError, match="unique"):
        load_inspect_samples(duplicate_path)

    required_custom = support.model_copy(
        update={
            "evaluators": [
                EvaluatorSpec(
                    evaluator_id="custom-required",
                    kind="custom",
                    version="1.0.0",
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="custom-required"):
        build_inspect_task(required_custom, [InspectSample(sample_id="one", input="hi")])
    docker_task = support.model_copy(
        update={
            "environment": support.environment.model_copy(update={"kind": EnvironmentKind.DOCKER})
        }
    )
    with pytest.raises(ValueError, match="sandbox"):
        build_inspect_task(docker_task, [InspectSample(sample_id="one", input="hi")])


def test_inspect_contains_matches_a_null_target_by_its_json_text() -> None:
    from inspect_ai.scorer import Target

    from agenttrace.integrations.inspect_ai import _evaluate_one

    root = Path(__file__).resolve().parents[1]
    support = load_task(root / "taskpacks/examples/support-routing.yaml")
    contains = EvaluatorSpec(evaluator_id="mentions-null", kind="contains", version="1.0.0")

    assert _evaluate_one(contains, support, "the answer is null", Target("null")) == (True, None)


def test_inspect_task_keeps_an_explicit_null_sample_target(tmp_path: Path) -> None:
    from agenttrace.integrations.inspect_ai import build_inspect_task, load_inspect_samples
    from agenttrace.tasks.samples import json_text

    root = Path(__file__).resolve().parents[1]
    support = load_task(root / "taskpacks/examples/support-routing.yaml")
    path = tmp_path / "samples.jsonl"
    path.write_text(
        '{"sample_id": "explicit", "input": "hi", "target": null}\n'
        '{"sample_id": "omitted", "input": "hi"}\n'
    )

    dataset = build_inspect_task(support, load_inspect_samples(path)).dataset
    targets = {sample.id: sample.target for sample in dataset}

    assert targets["explicit"] == "null"
    assert targets["omitted"] == json_text(support.reference)
