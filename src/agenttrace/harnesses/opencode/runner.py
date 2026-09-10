"""One-task-per-session OpenCode Coding Plan benchmark orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import httpx
from pydantic import JsonValue

from agenttrace.benchmarks.evaluation import evaluate_response
from agenttrace.harnesses.opencode.artifacts import persist_bytes, persist_json
from agenttrace.harnesses.opencode.config import build_opencode_config, write_opencode_config
from agenttrace.harnesses.opencode.credentials import ZAICredentialSource
from agenttrace.harnesses.opencode.events import emit_opencode_spans, sanitize_and_parse_jsonl
from agenttrace.harnesses.opencode.models import (
    BrokerRequestEvidence,
    OpenCodeBenchmarkExecution,
    OpenCodeBenchmarkManifest,
    OpenCodeBenchmarkRecord,
    OpenCodeBenchmarkSummary,
    OpenCodeRecordRef,
)
from agenttrace.harnesses.opencode.process import OpenCodeProcessResult, run_opencode
from agenttrace.harnesses.opencode.proxy import CodingPlanCredentialBroker
from agenttrace.harnesses.opencode.workspace import (
    isolated_workspace,
    snapshot_workspace,
    workspace_changes,
)
from agenttrace.integrations.telemetry import TelemetryRuntime
from agenttrace.models import TokenUsage
from agenttrace.providers.zai import ZAIReasoningEffort
from agenttrace.storage.objects import ObjectStore
from agenttrace.tasks.models import TaskDefinition
from agenttrace.tasks.samples import TaskSample, digest_json, render_task_prompt

ProcessRunner = Callable[..., OpenCodeProcessResult]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _agent_prompt(task: TaskDefinition, sample: TaskSample) -> str:
    task_prompt = render_task_prompt(task, sample.input)
    return (
        "You are running one isolated autonomous benchmark session. Work only inside the current "
        "workspace, use local tools as needed, do not access external networks, and decide when "
        "the task is complete. Return a concise final answer when finished.\n\n"
        f"{task_prompt}"
    )


def _sanitize_text(text: str, secrets: tuple[str, ...]) -> bytes:
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "[REDACTED_SECRET]")
    return cleaned.encode("utf-8")


def run_opencode_benchmark(
    *,
    task: TaskDefinition,
    samples: list[TaskSample],
    credential_source: ZAICredentialSource,
    telemetry: TelemetryRuntime,
    objects: ObjectStore,
    experiment_id: str,
    session_id: str | None = None,
    workspace_source: Path | None = None,
    model: str = "glm-5.2",
    reasoning_effort: ZAIReasoningEffort = ZAIReasoningEffort.MAX,
    small_model: str | None = None,
    executable: str = "opencode",
    timeout_seconds: int | None = None,
    max_model_calls: int = 256,
    store_outputs: bool = False,
    process_runner: ProcessRunner = run_opencode,
    broker_http_client: httpx.Client | None = None,
) -> OpenCodeBenchmarkExecution:
    """Run every sample in a fresh workspace and OpenCode process.

    The real Coding Plan key exists only in this parent process and its loopback credential broker.
    OpenCode receives a short-lived local capability, never the provider key.
    """
    if not samples:
        raise ValueError("samples cannot be empty")
    if max_model_calls < 1:
        raise ValueError("max_model_calls must be positive")
    effective_small_model = small_model or model
    credential = credential_source.resolve()
    real_key = credential.get_secret_value()
    task_payload = cast(JsonValue, task.model_dump(mode="json"))
    samples_payload = cast(JsonValue, [sample.model_dump(mode="json") for sample in samples])
    task_digest = digest_json(task_payload)
    samples_digest = digest_json(samples_payload)
    workspace_digest = None
    if workspace_source is not None:
        source_snapshot = snapshot_workspace(workspace_source.expanduser().resolve())
        workspace_digest = _digest(source_snapshot)

    records: list[OpenCodeBenchmarkRecord] = []
    refs: list[OpenCodeRecordRef] = []
    for sample in samples:
        prompt = _agent_prompt(task, sample)
        prompt_bytes = prompt.encode("utf-8")
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        config_digest = _digest(
            {
                "schema": "agenttrace-opencode-v1",
                "model": model,
                "small_model": effective_small_model,
                "reasoning_effort": reasoning_effort.value,
                "prompt_sha256": prompt_sha256,
                "max_model_calls": max_model_calls,
                "workspace_digest": workspace_digest,
            }
        )
        started_at = _utc_now()
        run = telemetry.run(
            experiment_id=experiment_id,
            session_id=session_id,
            trace_name=f"{task.task_id}/{sample.sample_id}" if session_id else None,
            task_id=task.task_id,
            task_version=task.version,
            task_split=task.split,
            attributes={
                "agenttrace.task.sample_id": sample.sample_id,
                "agenttrace.provider.name": "zai",
                "agenttrace.provider.endpoint_plan": "coding",
                "agenttrace.provider.transport": "opencode",
                "agenttrace.provider.thinking_mode": "enabled",
                "agenttrace.provider.reasoning_effort": reasoning_effort.value,
                "agenttrace.provenance.config_digest": config_digest,
                "agenttrace.session.task_count": 1,
            },
        )
        with isolated_workspace(workspace_source) as workspace:
            before = snapshot_workspace(workspace.path)
            with TemporaryDirectory(prefix="agenttrace-opencode-runtime-") as runtime_dir:
                runtime_root = Path(runtime_dir)
                with CodingPlanCredentialBroker(
                    api_key=credential,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    max_requests=max_model_calls,
                    http_client=broker_http_client,
                ) as broker:
                    config = build_opencode_config(
                        prompt=prompt,
                        model=model,
                        small_model=effective_small_model,
                        broker_base_url=broker.base_url,
                        broker_token=broker.client_token,
                        reasoning_effort=reasoning_effort,
                    )
                    config_path = write_opencode_config(
                        runtime_root / "config" / "opencode.json",
                        config,
                    )
                    with run:
                        with run.span(
                            "task.input",
                            step_type="user_input",
                            kind="AGENT",
                            input=prompt,
                            attributes={"agenttrace.task.sample_id": sample.sample_id},
                        ):
                            pass
                        with run.benchmark_stage(
                            "benchmark",
                            input={
                                "sample_id": sample.sample_id,
                                "prompt_sha256": prompt_sha256,
                                "workspace_digest": workspace_digest or "empty",
                            },
                            attributes={"agenttrace.transport": "opencode"},
                        ) as benchmark_result:
                            process = process_runner(
                                executable=executable,
                                workspace=workspace.path,
                                config_path=config_path,
                                runtime_root=runtime_root,
                                model=model,
                                reasoning_effort=reasoning_effort,
                                sample_id=sample.sample_id,
                                timeout_seconds=timeout_seconds,
                            )
                            sanitized_stdout, parsed = sanitize_and_parse_jsonl(
                                process.stdout,
                                known_secrets=(real_key, broker.client_token),
                                redaction_mode=task.data_policy.redaction_mode,
                            )
                            emit_opencode_spans(run, parsed, model_name=model)
                            benchmark_result.output = {
                                "exit_code": process.exit_code,
                                "timed_out": process.timed_out,
                                "event_count": parsed.event_count,
                                "model_call_count": len(broker.requests),
                                "tool_call_count": len(parsed.tools),
                            }
                        if process.exit_code != 0 or process.timed_out:
                            with run.span(
                                "opencode.process.result",
                                step_type="error",
                                kind="TOOL",
                                input={"sample_id": sample.sample_id},
                                attributes={
                                    "agenttrace.process.exit_code": process.exit_code,
                                    "agenttrace.process.timed_out": process.timed_out,
                                },
                            ) as process_result:
                                process_result.output = {
                                    "stderr_sha256": hashlib.sha256(
                                        _sanitize_text(process.stderr, (real_key,))
                                    ).hexdigest(),
                                    "stderr_bytes": len(process.stderr.encode("utf-8")),
                                }
                        for request in broker.requests:
                            with run.span(
                                "credential_broker.forward",
                                step_type="agent_state",
                                kind="CLIENT",
                                input={"sequence": request.sequence},
                                attributes={
                                    "http.response.status_code": request.status_code,
                                    "agenttrace.broker.request_bytes": request.request_bytes,
                                    "agenttrace.broker.response_bytes": request.response_bytes,
                                    "agenttrace.provider.thinking_mode": request.thinking_mode,
                                    "agenttrace.provider.reasoning_effort": (
                                        request.reasoning_effort
                                    ),
                                },
                            ) as broker_result:
                                broker_result.output = {"latency_ms": request.latency_ms}
                        outcomes, aggregate, required_passed = evaluate_response(
                            task,
                            sample,
                            parsed.final_text,
                            latency_ms=process.elapsed_seconds * 1000,
                        )
                        for outcome in outcomes:
                            with run.span(
                                f"evaluator.{outcome.evaluator_id}",
                                step_type="evaluator_result",
                                kind="EVALUATOR",
                                input={"sample_id": sample.sample_id},
                                attributes={
                                    "agenttrace.evaluator.id": outcome.evaluator_id,
                                    "agenttrace.evaluator.required": outcome.required,
                                },
                            ) as evaluation_result:
                                evaluation_result.output = outcome.model_dump(mode="json")
                        with run.span(
                            "agent.final_output",
                            step_type="final_output",
                            kind="AGENT",
                            attributes={
                                "agenttrace.evaluation.aggregate_score": aggregate,
                                "agenttrace.evaluation.required_passed": required_passed,
                            },
                        ) as final_result:
                            final_result.output = parsed.final_text
                    broker_requests = tuple(broker.requests)
                changes = workspace_changes(
                    workspace=workspace.path,
                    before=before,
                    objects=objects,
                )

        raw_ref, stdout_sha256 = persist_bytes(
            objects,
            "benchmarks/opencode/events",
            sanitized_stdout,
            suffix="jsonl",
            content_type="application/x-ndjson",
        )
        stderr = _sanitize_text(process.stderr, (real_key,))
        stderr_sha256 = hashlib.sha256(stderr).hexdigest()
        final_output = parsed.final_text.encode("utf-8")
        output_sha256 = hashlib.sha256(final_output).hexdigest()
        output_ref = None
        if store_outputs:
            output_ref, stored_digest = persist_bytes(
                objects,
                "benchmarks/opencode/outputs",
                final_output,
                suffix="txt",
                content_type="text/plain; charset=utf-8",
            )
            if stored_digest != output_sha256:
                raise RuntimeError("stored OpenCode output failed its checksum invariant")
        usage = TokenUsage(
            input_tokens=sum(event.input_tokens for event in parsed.models),
            output_tokens=sum(event.output_tokens for event in parsed.models),
            cached_input_tokens=sum(event.cached_input_tokens for event in parsed.models),
            reasoning_tokens=sum(event.reasoning_tokens for event in parsed.models),
        )
        status = (
            "timed_out"
            if process.timed_out
            else "completed"
            if process.exit_code == 0
            else "process_error"
        )
        record = OpenCodeBenchmarkRecord(
            experiment_id=experiment_id,
            run_id=run.run_id,
            episode_id=run.episode_id,
            trace_id=run.trace_id,
            task_id=task.task_id,
            task_version=task.version,
            task_split=task.split,
            sample_id=sample.sample_id,
            status=status,
            model=model,
            thinking_mode="enabled",
            reasoning_effort=reasoning_effort.value,
            started_at=started_at,
            ended_at=_utc_now(),
            latency_ms=process.elapsed_seconds * 1000,
            exit_code=process.exit_code,
            agent_decided_stop=process.exit_code == 0 and not process.timed_out,
            prompt_sha256=prompt_sha256,
            prompt_bytes=len(prompt_bytes),
            config_digest=config_digest,
            stdout_sha256=stdout_sha256,
            stdout_bytes=len(sanitized_stdout),
            stderr_sha256=stderr_sha256,
            stderr_bytes=len(stderr),
            raw_events_ref=raw_ref,
            event_count=parsed.event_count,
            invalid_event_count=parsed.invalid_line_count,
            redaction_count=parsed.redaction_stats.total,
            tool_call_count=len(parsed.tools),
            model_call_count=len(broker_requests),
            usage=usage,
            output_sha256=output_sha256,
            output_bytes=len(final_output),
            output_ref=output_ref,
            broker_requests=[
                BrokerRequestEvidence.from_request(request) for request in broker_requests
            ],
            workspace_changes=changes,
            evaluations=outcomes,
            aggregate_score=aggregate,
            required_evaluators_passed=required_passed,
        )
        records.append(record)
        record_key, record_digest = persist_json(
            objects,
            "benchmarks/opencode/records",
            record.model_dump(mode="json"),
        )
        refs.append(
            OpenCodeRecordRef(
                sample_id=sample.sample_id,
                status=record.status,
                sha256=record_digest,
                key=record_key,
                trace_id=record.trace_id,
            )
        )

    usage = TokenUsage(
        input_tokens=sum(record.usage.input_tokens for record in records),
        output_tokens=sum(record.usage.output_tokens for record in records),
        cached_input_tokens=sum(record.usage.cached_input_tokens for record in records),
        reasoning_tokens=sum(record.usage.reasoning_tokens for record in records),
    )
    manifest = OpenCodeBenchmarkManifest(
        created_at=_utc_now(),
        experiment_id=experiment_id,
        task_id=task.task_id,
        task_version=task.version,
        model=model,
        thinking_mode="enabled",
        reasoning_effort=reasoning_effort.value,
        task_digest=task_digest,
        samples_digest=samples_digest,
        workspace_digest=workspace_digest,
        records=refs,
        summary=OpenCodeBenchmarkSummary(
            sample_count=len(records),
            completed_count=sum(record.status == "completed" for record in records),
            process_error_count=sum(record.status == "process_error" for record in records),
            timed_out_count=sum(record.status == "timed_out" for record in records),
            model_call_count=sum(record.model_call_count for record in records),
            tool_call_count=sum(record.tool_call_count for record in records),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        ),
    )
    manifest_key, _ = persist_json(
        objects,
        "benchmarks/opencode/manifests",
        manifest.model_dump(mode="json"),
    )
    return OpenCodeBenchmarkExecution(manifest=manifest, manifest_key=manifest_key)
