"""Secure direct Z.AI API benchmark runner with OTLP trace fan-out."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import cast

import httpx
from pydantic import JsonValue

from agenttrace.benchmarks.evaluation import evaluate_response, validate_direct_task
from agenttrace.benchmarks.models import (
    BenchmarkError,
    BenchmarkExecution,
    BenchmarkManifest,
    BenchmarkRecord,
    BenchmarkRecordRef,
    BenchmarkSummary,
)
from agenttrace.integrations.telemetry import TelemetryRuntime
from agenttrace.models import TokenUsage
from agenttrace.providers.zai import (
    ZAIClient,
    ZAIEndpointPlan,
    ZAIProviderError,
    ZAISettings,
)
from agenttrace.storage.objects import ObjectStore
from agenttrace.tasks.models import TaskDefinition
from agenttrace.tasks.samples import TaskSample, canonical_json, digest_json, render_task_prompt


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


def _persist_json(objects: ObjectStore, prefix: str, value: object) -> tuple[str, str]:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    key = f"{prefix}/sha256/{digest[:2]}/{digest}.json"
    objects.put_bytes(key, encoded, content_type="application/json", if_absent=True)
    return key, digest


def _store_output(objects: ObjectStore, content: str) -> tuple[str, str, int]:
    encoded = content.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    key = f"benchmarks/outputs/sha256/{digest[:2]}/{digest}.txt"
    objects.put_bytes(key, encoded, content_type="text/plain; charset=utf-8", if_absent=True)
    return f"object://{key}", digest, len(encoded)


def run_zai_api_benchmark(
    *,
    task: TaskDefinition,
    samples: list[TaskSample],
    settings: ZAISettings,
    telemetry: TelemetryRuntime,
    objects: ObjectStore,
    experiment_id: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    store_outputs: bool = False,
    fail_fast: bool = False,
    http_client: httpx.Client | None = None,
) -> BenchmarkExecution:
    """Run one isolated trace session per sample through the general Z.AI API.

    The Coding Plan is deliberately rejected here. Its credentials must be used through an
    officially supported coding tool such as OpenCode, never through this custom SDK path.
    """
    if settings.endpoint_plan != ZAIEndpointPlan.GENERAL:
        raise ValueError(
            "direct API benchmarks require endpoint_plan=general; use the OpenCode harness "
            "for a Z.AI Coding Plan credential"
        )
    if not samples:
        raise ValueError("samples cannot be empty")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    validate_direct_task(task)
    settings.resolve_api_key()

    task_payload = cast(JsonValue, task.model_dump(mode="json"))
    sample_payload = cast(JsonValue, [sample.model_dump(mode="json") for sample in samples])
    task_digest = digest_json(task_payload)
    samples_digest = digest_json(sample_payload)
    safe_config = {
        "provider": "zai",
        "endpoint_plan": "general",
        "base_url": settings.resolved_base_url,
        "model": settings.model,
        "thinking_mode": settings.thinking_mode.value,
        "reasoning_effort": settings.reasoning_effort.value,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "task_digest": task_digest,
        "samples_digest": samples_digest,
        "store_outputs": store_outputs,
    }
    config_digest = _digest(safe_config)
    records: list[BenchmarkRecord] = []
    refs: list[BenchmarkRecordRef] = []

    for sample in samples:
        prompt = render_task_prompt(task, sample.input)
        prompt_bytes = prompt.encode("utf-8")
        input_sha256 = hashlib.sha256(canonical_json(sample.input)).hexdigest()
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        started_at = _utc_now()
        started_clock = time.perf_counter()
        run = telemetry.run(
            experiment_id=experiment_id,
            task_id=task.task_id,
            task_version=task.version,
            task_split=task.split,
            attributes={
                "agenttrace.task.sample_id": sample.sample_id,
                "agenttrace.provenance.config_digest": config_digest,
                "agenttrace.provenance.task_digest": task_digest,
                "agenttrace.provenance.samples_digest": samples_digest,
                "agenttrace.provider.name": "zai",
                "agenttrace.provider.endpoint_plan": "general",
                "agenttrace.provider.model": settings.model,
                "agenttrace.provider.thinking_mode": settings.thinking_mode.value,
                "agenttrace.provider.reasoning_effort": settings.reasoning_effort.value,
            },
        )
        response = None
        outcomes = []
        aggregate_score: float | None = None
        required_passed: bool | None = None
        output_ref: str | None = None
        error: BenchmarkError | None = None
        try:
            with run:
                with run.span(
                    "task.input",
                    step_type="user_input",
                    kind="AGENT",
                    input=sample.input,
                    attributes={"agenttrace.task.sample_id": sample.sample_id},
                ):
                    pass
                with run.benchmark_stage(
                    "benchmark",
                    input={
                        "sample_id": sample.sample_id,
                        "input_sha256": input_sha256,
                        "prompt_sha256": prompt_sha256,
                    },
                    attributes={
                        "agenttrace.provider.name": "zai",
                        "agenttrace.provider.endpoint_plan": "general",
                    },
                ) as benchmark_result:
                    with ZAIClient(settings, trace_run=run, http_client=http_client) as client:
                        response = client.complete(
                            [{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                    latency_ms = (time.perf_counter() - started_clock) * 1000
                    benchmark_result.output = {
                        "finish_reason": response.finish_reason,
                        "latency_ms": latency_ms,
                        "output_sha256": hashlib.sha256(
                            response.content.encode("utf-8")
                        ).hexdigest(),
                    }
                outcomes, aggregate_score, required_passed = evaluate_response(
                    task,
                    sample,
                    response.content,
                    latency_ms=latency_ms,
                )
                for outcome in outcomes:
                    with run.span(
                        f"evaluator.{outcome.evaluator_id}",
                        step_type="evaluator_result",
                        kind="EVALUATOR",
                        input={
                            "sample_id": sample.sample_id,
                            "output_sha256": hashlib.sha256(
                                response.content.encode("utf-8")
                            ).hexdigest(),
                        },
                        attributes={
                            "agenttrace.evaluator.id": outcome.evaluator_id,
                            "agenttrace.evaluator.version": outcome.evaluator_version,
                            "agenttrace.evaluator.required": outcome.required,
                        },
                    ) as evaluation_result:
                        evaluation_result.output = outcome.model_dump(mode="json")
                with run.span(
                    "agent.final_output",
                    step_type="final_output",
                    kind="AGENT",
                    attributes={
                        "agenttrace.evaluation.aggregate_score": aggregate_score,
                        "agenttrace.evaluation.required_passed": required_passed,
                    },
                ) as final_result:
                    final_result.output = response.content
        except ZAIProviderError as provider_error:
            error = BenchmarkError(
                error_type=type(provider_error).__name__,
                status_code=provider_error.status_code,
                request_id=provider_error.request_id,
            )

        ended_at = _utc_now()
        latency_ms = (time.perf_counter() - started_clock) * 1000
        if response is not None:
            encoded_output = response.content.encode("utf-8")
            output_sha256 = hashlib.sha256(encoded_output).hexdigest()
            output_bytes = len(encoded_output)
            if store_outputs:
                output_ref, stored_digest, stored_bytes = _store_output(objects, response.content)
                if stored_digest != output_sha256 or stored_bytes != output_bytes:
                    raise RuntimeError("stored benchmark output failed its checksum invariant")
            record = BenchmarkRecord(
                experiment_id=experiment_id,
                run_id=run.run_id,
                episode_id=run.episode_id,
                trace_id=run.trace_id,
                task_id=task.task_id,
                task_version=task.version,
                task_split=task.split,
                sample_id=sample.sample_id,
                status="completed",
                requested_model=settings.model,
                response_model=response.model,
                thinking_mode=settings.thinking_mode.value,
                reasoning_effort=settings.reasoning_effort.value,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                input_sha256=input_sha256,
                prompt_sha256=prompt_sha256,
                prompt_bytes=len(prompt_bytes),
                output_sha256=output_sha256,
                output_bytes=output_bytes,
                output_ref=output_ref,
                finish_reason=response.finish_reason,
                provider_request_id=response.request_id,
                usage=response.usage,
                evaluations=outcomes,
                aggregate_score=aggregate_score,
                required_evaluators_passed=required_passed,
                config_digest=config_digest,
            )
        else:
            assert error is not None
            record = BenchmarkRecord(
                experiment_id=experiment_id,
                run_id=run.run_id,
                episode_id=run.episode_id,
                trace_id=run.trace_id,
                task_id=task.task_id,
                task_version=task.version,
                task_split=task.split,
                sample_id=sample.sample_id,
                status="provider_error",
                requested_model=settings.model,
                thinking_mode=settings.thinking_mode.value,
                reasoning_effort=settings.reasoning_effort.value,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                input_sha256=input_sha256,
                prompt_sha256=prompt_sha256,
                prompt_bytes=len(prompt_bytes),
                config_digest=config_digest,
                error=error,
            )
        records.append(record)
        record_key, record_digest = _persist_json(
            objects,
            "benchmarks/records",
            record.model_dump(mode="json"),
        )
        refs.append(
            BenchmarkRecordRef(
                sample_id=sample.sample_id,
                status=record.status,
                required_evaluators_passed=record.required_evaluators_passed,
                sha256=record_digest,
                key=record_key,
            )
        )
        if fail_fast and record.status == "provider_error":
            break

    completed = [record for record in records if record.status == "completed"]
    usage = TokenUsage(
        input_tokens=sum(record.usage.input_tokens for record in completed),
        output_tokens=sum(record.usage.output_tokens for record in completed),
        cached_input_tokens=sum(record.usage.cached_input_tokens for record in completed),
        reasoning_tokens=sum(record.usage.reasoning_tokens for record in completed),
    )
    manifest = BenchmarkManifest(
        created_at=_utc_now(),
        experiment_id=experiment_id,
        task_id=task.task_id,
        task_version=task.version,
        model=settings.model,
        thinking_mode=settings.thinking_mode.value,
        reasoning_effort=settings.reasoning_effort.value,
        task_digest=task_digest,
        samples_digest=samples_digest,
        config_digest=config_digest,
        store_outputs=store_outputs,
        records=refs,
        summary=BenchmarkSummary(
            sample_count=len(records),
            completed_count=len(completed),
            provider_error_count=sum(record.status == "provider_error" for record in records),
            required_pass_count=sum(
                record.required_evaluators_passed is True for record in completed
            ),
            required_fail_count=sum(
                record.required_evaluators_passed is False for record in completed
            ),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        ),
    )
    manifest_key, _ = _persist_json(
        objects,
        "benchmarks/manifests",
        manifest.model_dump(mode="json"),
    )
    return BenchmarkExecution(manifest=manifest, manifest_key=manifest_key)
