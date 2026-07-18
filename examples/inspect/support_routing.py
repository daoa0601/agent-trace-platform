"""Non-coding Inspect example generated from an AgentTrace task pack."""

from pathlib import Path

from inspect_ai import Task, task

from agenttrace.integrations.inspect_ai import build_inspect_task, load_inspect_samples
from agenttrace.tasks import load_task

ROOT = Path(__file__).resolve().parents[2]


@task
def support_routing() -> Task:
    definition = load_task(ROOT / "taskpacks/examples/support-routing.yaml")
    samples = load_inspect_samples(ROOT / "examples/inspect/support-routing.jsonl")
    return build_inspect_task(definition, samples)
