"""Operator CLI for local development and batch pipeline jobs."""

from __future__ import annotations

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
app.add_typer(task_app, name="task")
app.add_typer(pipeline_app, name="pipeline")


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


if __name__ == "__main__":
    app()
