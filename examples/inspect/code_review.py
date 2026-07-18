"""Coding-domain Inspect example using the same AgentTrace adapter."""

from pathlib import Path

from inspect_ai import Task, task

from agenttrace.integrations.inspect_ai import build_inspect_task, load_inspect_samples
from agenttrace.tasks import load_task

ROOT = Path(__file__).resolve().parents[2]


@task
def code_review() -> Task:
    definition = load_task(ROOT / "taskpacks/examples/code-review.yaml")
    samples = load_inspect_samples(ROOT / "examples/inspect/code-review.jsonl")
    return build_inspect_task(definition, samples)
