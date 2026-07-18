"""Operator CLI for local development and batch pipeline jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from agenttrace.ids import new_id, new_span_id, new_trace_id
from agenttrace.models import Content, EventBatch, StepMetrics, StepType, TaskRef, TraceEvent
from agenttrace.pipelines import (
    DatasetKind,
    build_dataset_snapshot,
    build_episode_snapshot,
    build_evaluation_snapshot,
    normalize_raw_traces,
)
from agenttrace.storage import FileObjectStore, RawBatchWriter
from agenttrace.tasks import TaskRegistry, load_task

app = typer.Typer(help="Collect, process, evaluate, and export versioned agent traces.")
task_app = typer.Typer(help="Validate and inspect task definitions.")
pipeline_app = typer.Typer(help="Run immutable batch pipelines.")
provider_app = typer.Typer(help="Run securely configured model-provider harnesses.")
benchmark_app = typer.Typer(help="Run provider-neutral tasks with traced model adapters.")
audit_app = typer.Typer(help="Validate and gate independently audited benchmark claims.")
app.add_typer(task_app, name="task")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(provider_app, name="provider")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(audit_app, name="audit")


def _store(root: Path) -> FileObjectStore:
    return FileObjectStore(root)


def _print_manifest(key: str, manifest: object) -> None:
    payload = {"manifest_key": key, "manifest": json.loads(manifest.model_dump_json())}  # type: ignore[attr-defined]
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def collector(
    host: Annotated[str, typer.Option()] = "0.0.0.0",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
    log_level: Annotated[str, typer.Option()] = "INFO",
    log_format: Annotated[str, typer.Option(help="Collector log format: json or text.")] = "json",
) -> None:
    """Start the collection API using AGENTTRACE_* settings."""
    import uvicorn

    from agenttrace.integrations.logging import configure_structured_logging

    if log_format not in {"json", "text"}:
        raise typer.BadParameter("log format must be json or text")
    configure_structured_logging(level=log_level, json_output=log_format == "json")
    uvicorn.run(
        "agenttrace.collector.app:app",
        host=host,
        port=port,
        factory=False,
        access_log=False,
        log_config=None,
        use_colors=False,
    )


@task_app.command("validate")
def validate_task(path: Path) -> None:
    """Validate one YAML task definition."""
    task = load_task(path)
    typer.echo(f"valid: {task.task_id}@{task.version} ({len(task.evaluators)} evaluators)")


@task_app.command("list")
def list_tasks(root: Path = Path("taskpacks")) -> None:
    """List every valid definition below a task-pack directory."""
    registry = TaskRegistry.from_directory(root)
    for task in registry.list():
        typer.echo(f"{task.task_id}@{task.version}\t{task.domain}\t{task.title}")


@audit_app.command("gate")
def gate_audit(
    manifest: Annotated[Path, typer.Argument(help="AuditableSession JSON manifest.")],
    report: Annotated[
        Path | None,
        typer.Option(help="Optional Markdown report path; findings get a dedicated section."),
    ] = None,
    object_root: Annotated[
        Path | None,
        typer.Option(help="Optional immutable object-store root for the audit bundle."),
    ] = None,
) -> None:
    """Apply the deterministic headline-publication gate to an audited session."""
    from pydantic import ValidationError

    from agenttrace.audits import (
        AuditableSession,
        GateStatus,
        build_audit_bundle,
        persist_audit_bundle,
        render_audit_report,
    )

    try:
        session = AuditableSession.model_validate_json(manifest.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as error:
        typer.echo(f"invalid audit manifest: {error}", err=True)
        raise typer.Exit(code=1) from error
    bundle = build_audit_bundle(session)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_audit_report(bundle), encoding="utf-8")
    bundle_key = None
    if object_root is not None:
        bundle_key = persist_audit_bundle(_store(object_root), bundle).key
    typer.echo(
        json.dumps(
            {
                "session_id": session.session_id,
                "session_digest": bundle.headline_gate.session_digest,
                "status": bundle.headline_gate.status.value,
                "eligible_for_headline": bundle.headline_gate.eligible_for_headline,
                "headline_claim_ids": bundle.headline_gate.headline_claim_ids,
                "failed_checks": [
                    check.check_id for check in bundle.headline_gate.checks if not check.passed
                ],
                "report": str(report) if report is not None else None,
                "bundle_key": bundle_key,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if bundle.headline_gate.status == GateStatus.BLOCKED:
        raise typer.Exit(code=2)


@pipeline_app.command("normalize")
def normalize(
    object_root: Path = Path("data/object-store"),
    raw_prefix: str = "raw",
    strict_references: bool = True,
) -> None:
    result = normalize_raw_traces(
        _store(object_root),
        raw_prefix=raw_prefix,
        strict_references=strict_references,
    )
    _print_manifest(result.manifest_key, result.manifest)


@pipeline_app.command("episodes")
def episodes(
    canonical_manifest: str,
    object_root: Path = Path("data/object-store"),
) -> None:
    result = build_episode_snapshot(
        _store(object_root),
        canonical_manifest_key=canonical_manifest,
    )
    _print_manifest(result.manifest_key, result.manifest)


@pipeline_app.command("evaluate")
def evaluate(
    episode_manifest: str,
    object_root: Path = Path("data/object-store"),
    task_root: Path = Path("taskpacks"),
) -> None:
    result = build_evaluation_snapshot(
        _store(object_root),
        episode_manifest_key=episode_manifest,
        tasks=TaskRegistry.from_directory(task_root),
    )
    _print_manifest(result.manifest_key, result.manifest)


@pipeline_app.command("dataset")
def dataset(
    name: str,
    kind: DatasetKind,
    episode_manifest: str,
    evaluation_manifest: str,
    object_root: Path = Path("data/object-store"),
    task_root: Path = Path("taskpacks"),
    minimum_reward: float | None = None,
    include_failed: bool = False,
    strict_rl: bool = True,
) -> None:
    result = build_dataset_snapshot(
        _store(object_root),
        name=name,
        kind=kind,
        episode_manifest_key=episode_manifest,
        evaluation_manifest_key=evaluation_manifest,
        tasks=TaskRegistry.from_directory(task_root),
        minimum_reward=minimum_reward,
        include_failed=include_failed,
        strict_rl=strict_rl,
    )
    _print_manifest(result.manifest_key, result.manifest)


@app.command()
def demo(
    root: Annotated[Path, typer.Option(help="Disposable output directory.")] = Path("outputs/demo"),
) -> None:
    """Run collection through frozen eval-dataset export without external services."""
    objects = FileObjectStore(root / "object-store")
    task_path = Path(__file__).resolve().parents[2] / "taskpacks/examples/support-routing.yaml"
    task = load_task(task_path)
    task_ref = TaskRef(task_id=task.task_id, task_version=task.version, split="smoke")
    trace_id = new_trace_id()
    root_span = new_span_id()
    experiment_id = new_id("exp")
    run_id = new_id("run")
    episode_id = new_id("episode")
    output: dict[str, JsonValue] = {
        "intent": "billing_refund",
        "urgency": "normal",
        "route": "billing",
    }
    common = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "episode_id": episode_id,
        "trace_id": trace_id,
        "task": task_ref,
    }
    events = [
        TraceEvent(
            **common,
            span_id=root_span,
            parent_span_id=None,
            sequence_number=0,
            step_type=StepType.EPISODE_START,
            attributes={"status": "running"},
        ),
        TraceEvent(
            **common,
            parent_span_id=root_span,
            sequence_number=1,
            step_type=StepType.USER_INPUT,
            input=Content(
                inline={"message": "I was charged twice and need a refund."},
            ),
        ),
        TraceEvent(
            **common,
            parent_span_id=root_span,
            sequence_number=2,
            step_type=StepType.MODEL_GENERATION,
            output=Content(inline=output),
            metrics=StepMetrics(latency_ms=42.0),
        ),
        TraceEvent(
            **common,
            parent_span_id=root_span,
            sequence_number=3,
            step_type=StepType.FINAL_OUTPUT,
            output=Content(inline=output),
        ),
        TraceEvent(
            **common,
            parent_span_id=root_span,
            sequence_number=4,
            step_type=StepType.EPISODE_END,
            output=Content(inline={"status": "succeeded"}),
            attributes={"status": "succeeded"},
        ),
    ]
    raw = RawBatchWriter(objects).write(EventBatch(worker_id="demo", events=events))
    canonical = normalize_raw_traces(objects)
    episode_snapshot = build_episode_snapshot(
        objects,
        canonical_manifest_key=canonical.manifest_key,
    )

    registry = TaskRegistry([task])
    evaluation = build_evaluation_snapshot(
        objects,
        episode_manifest_key=episode_snapshot.manifest_key,
        tasks=registry,
    )
    frozen = build_dataset_snapshot(
        objects,
        name="support-routing-smoke",
        kind=DatasetKind.EVAL,
        episode_manifest_key=episode_snapshot.manifest_key,
        evaluation_manifest_key=evaluation.manifest_key,
        tasks=registry,
    )
    typer.echo(
        json.dumps(
            {
                "raw": f"object://{raw.key}",
                "canonical": canonical.manifest_key,
                "episodes": episode_snapshot.manifest_key,
                "evaluations": evaluation.manifest_key,
                "dataset": frozen.manifest_key,
                "rows": frozen.manifest.metrics["row_count"],
            },
            indent=2,
        )
    )


@benchmark_app.command("zai-api")
def zai_api_benchmark(
    task_path: Annotated[Path, typer.Argument(help="Versioned AgentTrace task YAML.")],
    samples_path: Annotated[Path, typer.Argument(help="Strict TaskSample JSONL file.")],
    model: Annotated[str, typer.Option()] = "glm-5.2",
    api_key_file: Annotated[
        Path | None,
        typer.Option(
            "--api-key-file",
            help="Read the general API credential from a file instead of ZAI_API_KEY.",
        ),
    ] = None,
    max_tokens: Annotated[int, typer.Option(min=1)] = 512,
    temperature: Annotated[float, typer.Option(min=0)] = 0.0,
    thinking_mode: Annotated[
        str,
        typer.Option(help="Turn-level reasoning mode: disabled or enabled."),
    ] = "disabled",
    reasoning_effort: Annotated[
        str,
        typer.Option(
            help="GLM-5.2 reasoning effort: max, xhigh, high, medium, low, minimal, or none."
        ),
    ] = "max",
    experiment_id: Annotated[str, typer.Option()] = "zai-api-benchmark",
    output_root: Annotated[
        Path,
        typer.Option(help="Local immutable benchmark object-store root."),
    ] = Path("outputs/benchmarks"),
    store_outputs: Annotated[
        bool,
        typer.Option(
            "--store-outputs/--metadata-only",
            help="Retain full responses as content-addressed objects; default stores hashes only.",
        ),
    ] = False,
    otlp_endpoint: Annotated[
        str | None,
        typer.Option(help="OTLP base URL or /v1/traces URL. Omit to disable trace export."),
    ] = None,
    capture_content: Annotated[
        str,
        typer.Option(help="OTLP payload policy: none, metadata, or full."),
    ] = "metadata",
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast/--continue-on-provider-error"),
    ] = False,
) -> None:
    """Run general-API GLM eval samples without printing prompts, responses, or credentials."""
    from agenttrace.benchmarks import run_zai_api_benchmark
    from agenttrace.integrations.telemetry import CaptureMode, TelemetrySettings, create_telemetry
    from agenttrace.providers.zai import (
        ZAIEndpointPlan,
        ZAIReasoningEffort,
        ZAISettings,
        ZAIThinkingMode,
    )
    from agenttrace.tasks import load_task_samples

    try:
        capture = CaptureMode(capture_content)
        thinking = ZAIThinkingMode(thinking_mode)
        effort = ZAIReasoningEffort(reasoning_effort)
        task = load_task(task_path)
        samples = load_task_samples(samples_path)
        settings_kwargs = {
            "api_key_file": api_key_file,
            "endpoint_plan": ZAIEndpointPlan.GENERAL,
            "model": model,
            "thinking_mode": thinking,
            "reasoning_effort": effort,
        }
        if api_key_file is not None:
            settings_kwargs["api_key"] = None
        provider_settings = ZAISettings(**settings_kwargs)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    telemetry = create_telemetry(
        TelemetrySettings(
            telemetry_enabled=otlp_endpoint is not None,
            otlp_endpoint=otlp_endpoint or "http://127.0.0.1:4318/v1/traces",
            service_name="agenttrace-zai-api-benchmark",
            capture_content=capture,
        )
    )
    try:
        execution = run_zai_api_benchmark(
            task=task,
            samples=samples,
            settings=provider_settings,
            telemetry=telemetry,
            objects=_store(output_root),
            experiment_id=experiment_id,
            max_tokens=max_tokens,
            temperature=temperature,
            store_outputs=store_outputs,
            fail_fast=fail_fast,
        )
    except ValueError as error:
        typer.echo(
            json.dumps(
                {
                    "ok": False,
                    "provider": "zai",
                    "endpoint_plan": "general",
                    "error_type": type(error).__name__,
                    "content_printed": False,
                },
                indent=2,
                sort_keys=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from error
    finally:
        telemetry.close()

    summary = execution.manifest.summary
    typer.echo(
        json.dumps(
            {
                "ok": summary.provider_error_count == 0,
                "provider": "zai",
                "endpoint_plan": "general",
                "model": execution.manifest.model,
                "thinking_mode": execution.manifest.thinking_mode,
                "reasoning_effort": execution.manifest.reasoning_effort,
                "experiment_id": execution.manifest.experiment_id,
                "manifest_key": execution.manifest_key,
                "object_root": str(output_root.expanduser().resolve()),
                "summary": summary.model_dump(mode="json"),
                "content_printed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if summary.provider_error_count:
        raise typer.Exit(code=2)


@benchmark_app.command("zai-opencode")
def zai_opencode_benchmark(
    task_path: Annotated[Path, typer.Argument(help="Versioned AgentTrace task YAML.")],
    samples_path: Annotated[Path, typer.Argument(help="Strict TaskSample JSONL file.")],
    workspace: Annotated[
        Path | None,
        typer.Option(help="Optional source tree copied into a fresh workspace for every sample."),
    ] = None,
    model: Annotated[str, typer.Option()] = "glm-5.2",
    reasoning_effort: Annotated[
        str,
        typer.Option(
            help="GLM-5.2 reasoning effort: max, xhigh, high, medium, low, minimal, or none."
        ),
    ] = "max",
    small_model: Annotated[
        str | None,
        typer.Option(
            help="Optional lightweight model; defaults to --model for a one-model session."
        ),
    ] = None,
    api_key_file: Annotated[
        Path | None,
        typer.Option(
            "--api-key-file",
            help="Parent-only Coding Plan credential file; otherwise use ZAI_API_KEY.",
        ),
    ] = None,
    executable: Annotated[str, typer.Option(help="OpenCode executable name or path.")] = "opencode",
    watchdog_seconds: Annotated[
        int | None,
        typer.Option(min=0, help="0 disables the watchdog; omitted uses the task policy."),
    ] = None,
    max_model_calls: Annotated[int, typer.Option(min=1)] = 256,
    experiment_id: Annotated[str, typer.Option()] = "zai-opencode-benchmark",
    output_root: Annotated[
        Path,
        typer.Option(help="Local immutable benchmark object-store root."),
    ] = Path("outputs/benchmarks"),
    store_outputs: Annotated[
        bool,
        typer.Option("--store-outputs/--metadata-only"),
    ] = False,
    otlp_endpoint: Annotated[
        str | None,
        typer.Option(help="OTLP base URL or /v1/traces URL. Omit to disable trace export."),
    ] = None,
    capture_content: Annotated[
        str,
        typer.Option(help="OTLP payload policy: none, metadata, or full."),
    ] = "metadata",
) -> None:
    """Run isolated OpenCode sessions without exposing the Coding Plan key to agent tools."""
    from agenttrace.harnesses.opencode import ZAICredentialSource, run_opencode_benchmark
    from agenttrace.integrations.telemetry import CaptureMode, TelemetrySettings, create_telemetry
    from agenttrace.providers.zai import ZAIReasoningEffort
    from agenttrace.tasks import load_task_samples

    try:
        capture = CaptureMode(capture_content)
        effort = ZAIReasoningEffort(reasoning_effort)
        task = load_task(task_path)
        samples = load_task_samples(samples_path)
        if watchdog_seconds is None:
            timeout = (
                task.session_policy.watchdog_seconds
                if task.session_policy is not None
                else task.environment.resources.timeout_seconds
            )
        else:
            timeout = watchdog_seconds or None
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    telemetry = create_telemetry(
        TelemetrySettings(
            telemetry_enabled=otlp_endpoint is not None,
            otlp_endpoint=otlp_endpoint or "http://127.0.0.1:4318/v1/traces",
            service_name="agenttrace-zai-opencode-benchmark",
            capture_content=capture,
        )
    )
    try:
        execution = run_opencode_benchmark(
            task=task,
            samples=samples,
            credential_source=ZAICredentialSource(api_key_file=api_key_file),
            telemetry=telemetry,
            objects=_store(output_root),
            experiment_id=experiment_id,
            workspace_source=workspace,
            model=model,
            reasoning_effort=effort,
            small_model=small_model,
            executable=executable,
            timeout_seconds=timeout,
            max_model_calls=max_model_calls,
            store_outputs=store_outputs,
        )
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(
            json.dumps(
                {
                    "ok": False,
                    "provider": "zai",
                    "transport": "opencode",
                    "endpoint_plan": "coding",
                    "error_type": type(error).__name__,
                    "content_printed": False,
                },
                indent=2,
                sort_keys=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from error
    finally:
        telemetry.close()
    summary = execution.manifest.summary
    typer.echo(
        json.dumps(
            {
                "ok": summary.process_error_count == 0 and summary.timed_out_count == 0,
                "provider": "zai",
                "transport": "opencode",
                "endpoint_plan": "coding",
                "model": execution.manifest.model,
                "thinking_mode": execution.manifest.thinking_mode,
                "reasoning_effort": execution.manifest.reasoning_effort,
                "experiment_id": execution.manifest.experiment_id,
                "manifest_key": execution.manifest_key,
                "object_root": str(output_root.expanduser().resolve()),
                "summary": summary.model_dump(mode="json"),
                "content_printed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if summary.process_error_count or summary.timed_out_count:
        raise typer.Exit(code=2)


@provider_app.command("zai-smoke")
def zai_smoke(
    endpoint_plan: Annotated[
        str,
        typer.Option(
            "--endpoint-plan", help="Explicit entitlement/billing endpoint: general or coding."
        ),
    ],
    model: Annotated[str, typer.Option()] = "glm-5.2",
    max_tokens: Annotated[int, typer.Option(min=1, max=256)] = 32,
    otlp_endpoint: Annotated[
        str | None,
        typer.Option(help="OTLP base URL or full /v1/traces URL. Omit to disable export."),
    ] = None,
    experiment_id: Annotated[str, typer.Option()] = "zai-live-smoke",
    capture_content: Annotated[
        str,
        typer.Option(help="Trace payload policy: none, metadata, or full."),
    ] = "metadata",
    thinking_mode: Annotated[
        str,
        typer.Option(help="Turn-level reasoning mode: disabled or enabled."),
    ] = "disabled",
    reasoning_effort: Annotated[
        str,
        typer.Option(
            help="GLM-5.2 reasoning effort: max, xhigh, high, medium, low, minimal, or none."
        ),
    ] = "max",
) -> None:
    """Send a minimal general-API request without printing prompt, response, or credentials."""
    from agenttrace.integrations.telemetry import CaptureMode, TelemetrySettings, create_telemetry
    from agenttrace.providers.zai import (
        ZAIClient,
        ZAIEndpointPlan,
        ZAIProviderError,
        ZAIReasoningEffort,
        ZAISettings,
        ZAIThinkingMode,
    )

    try:
        plan = ZAIEndpointPlan(endpoint_plan)
        capture = CaptureMode(capture_content)
        thinking = ZAIThinkingMode(thinking_mode)
        effort = ZAIReasoningEffort(reasoning_effort)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if plan == ZAIEndpointPlan.CODING:
        raise typer.BadParameter(
            "custom SDK calls cannot use the Coding Plan; use the supported OpenCode harness"
        )
    telemetry = create_telemetry(
        TelemetrySettings(
            telemetry_enabled=otlp_endpoint is not None,
            otlp_endpoint=otlp_endpoint or "http://127.0.0.1:4318/v1/traces",
            service_name="agenttrace-zai-smoke",
            capture_content=capture,
        )
    )
    try:
        with telemetry.run(
            experiment_id=experiment_id,
            task_id="provider.zai-connectivity",
            task_version="1.0.0",
            task_split="smoke",
            attributes={
                "agenttrace.provider.endpoint_plan": plan.value,
                "agenttrace.provider.thinking_mode": thinking.value,
                "agenttrace.provider.reasoning_effort": effort.value,
            },
        ) as run:
            with ZAIClient(
                ZAISettings(
                    endpoint_plan=plan,
                    model=model,
                    thinking_mode=thinking,
                    reasoning_effort=effort,
                ),
                trace_run=run,
            ) as client:
                response = client.complete(
                    [{"role": "user", "content": "Reply with exactly READY."}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
        completion_valid = response.finish_reason == "stop" and response.content.strip() == "READY"
        typer.echo(
            json.dumps(
                {
                    "ok": completion_valid,
                    "provider": "zai",
                    "endpoint_plan": plan.value,
                    "thinking_mode": thinking.value,
                    "reasoning_effort": effort.value,
                    "model": response.model,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                    "usage": response.usage.model_dump(),
                    "response_sha256": hashlib.sha256(response.content.encode("utf-8")).hexdigest(),
                    "trace_id": run.trace_id,
                    "content_printed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not completion_valid:
            raise typer.Exit(code=2)
    except (ZAIProviderError, ValueError) as error:
        payload = {
            "ok": False,
            "provider": "zai",
            "endpoint_plan": plan.value,
            "error_type": type(error).__name__,
            "status_code": getattr(error, "status_code", None),
            "request_id": getattr(error, "request_id", None),
            "content_printed": False,
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True), err=True)
        raise typer.Exit(code=1) from error
    finally:
        telemetry.close()


if __name__ == "__main__":
    app()
