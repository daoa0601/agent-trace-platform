"""Inspect AI adapter backed by AgentTrace task definitions.

Inspect is optional. Import this module only in environments installed with the
``inspect`` project extra.
"""

from __future__ import annotations

import re
from typing import Any

import jsonschema
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Solver, TaskState, generate
from pydantic import JsonValue

from agenttrace.tasks.models import EnvironmentKind, EvaluatorSpec, TaskDefinition
from agenttrace.tasks.samples import (
    ParsedJson,
    TaskSample,
    describe_schema_violation,
    json_text,
    load_task_samples,
    parse_json_output,
    render_task_prompt,
)

SUPPORTED_EVALUATORS = {"exact_match", "contains", "regex", "json_schema"}


InspectSample = TaskSample
load_inspect_samples = load_task_samples


def _json_text(value: JsonValue) -> str:
    return json_text(value)


def _prompt(task: TaskDefinition, value: JsonValue) -> str:
    return render_task_prompt(task, value)


def _parse_json_output(text: str) -> ParsedJson | None:
    return parse_json_output(text)


def _expected(spec: EvaluatorSpec, task: TaskDefinition, target: Target) -> JsonValue:
    configured = spec.config.get("expected")
    if configured is not None:
        return configured
    parsed_target = _parse_json_output(target.text)
    if parsed_target is not None:
        return parsed_target.value
    return task.reference if task.reference is not None else target.text


def _evaluate_one(
    spec: EvaluatorSpec,
    task: TaskDefinition,
    output_text: str,
    target: Target,
) -> tuple[bool, str | None]:
    parsed = _parse_json_output(output_text)
    if spec.kind == "exact_match":
        actual: JsonValue = parsed.value if parsed is not None else output_text
        expected = _expected(spec, task, target)
        if isinstance(actual, str) and isinstance(expected, str):
            if bool(spec.config.get("strip", True)):
                actual, expected = actual.strip(), expected.strip()
            if not bool(spec.config.get("case_sensitive", True)):
                actual, expected = actual.casefold(), expected.casefold()
        return actual == expected, None
    if spec.kind == "contains":
        configured = spec.config.get(
            "values", spec.config.get("value", _expected(spec, task, target))
        )
        values = configured if isinstance(configured, list) else [configured]
        # Non-string targets (including a parsed JSON null) are matched by their JSON text.
        needles = [value if isinstance(value, str) else json_text(value) for value in values]
        case_sensitive = bool(spec.config.get("case_sensitive", False))
        haystack = output_text if case_sensitive else output_text.casefold()
        missing = [
            needle
            for needle in needles
            if (needle if case_sensitive else needle.casefold()) not in haystack
        ]
        return not missing, f"missing values: {missing}" if missing else None
    if spec.kind == "regex":
        pattern = spec.config.get("pattern")
        if not isinstance(pattern, str):
            return False, "config.pattern is required"
        flags = re.IGNORECASE if spec.config.get("ignore_case", False) else 0
        return re.search(pattern, output_text, flags=flags) is not None, None
    if spec.kind == "json_schema":
        schema = spec.config.get("schema", task.expected_output_schema)
        if parsed is None:
            return False, "output is not JSON"
        if not isinstance(schema, dict):
            return False, "output schema is missing"
        try:
            jsonschema.validate(parsed.value, schema)
        except jsonschema.ValidationError as error:
            violation = describe_schema_violation(error)
            return (
                False,
                f"{violation['validator']} failed at schema path {violation['schema_path']}",
            )
        return True, None
    raise ValueError(f"unsupported Inspect evaluator: {spec.kind}")


@scorer(metrics=[mean()], name="agenttrace_weighted_score")
def agenttrace_weighted_scorer(task_json: str) -> Scorer:
    """Apply compatible AgentTrace evaluators and preserve component details."""
    task = TaskDefinition.model_validate_json(task_json)
    specs = [spec for spec in task.evaluators if spec.kind in SUPPORTED_EVALUATORS]

    async def score(state: TaskState, target: Target) -> Score:
        components: dict[str, float] = {}
        details: dict[str, Any] = {}
        weighted = 0.0
        total_weight = 0.0
        required_passed = True
        for spec in specs:
            passed, explanation = _evaluate_one(spec, task, state.output.completion, target)
            value = 1.0 if passed else 0.0
            components[spec.evaluator_id] = value
            details[spec.evaluator_id] = {
                "passed": passed,
                "required": spec.required,
                "weight": spec.weight,
                **({"explanation": explanation} if explanation else {}),
            }
            weighted += value * spec.weight
            total_weight += spec.weight
            required_passed = required_passed and (passed or not spec.required)
        aggregate = weighted / total_weight if total_weight else 0.0
        return Score(
            value=aggregate,
            explanation="required evaluators passed"
            if required_passed
            else "a required evaluator failed",
            metadata={"components": components, "evaluators": details},
        )

    return score


def build_inspect_task(
    task: TaskDefinition,
    samples: list[InspectSample],
    *,
    solver: Solver | None = None,
    sandbox: Any = None,
) -> Task:
    """Convert a task recipe plus samples into an Inspect task.

    Required evaluator kinds that Inspect cannot reproduce are rejected. Optional
    operational evaluators remain in AgentTrace's post-run pipeline and are listed in
    the Inspect task metadata rather than being silently approximated.
    """
    unsupported_required = [
        spec.evaluator_id
        for spec in task.evaluators
        if spec.kind not in SUPPORTED_EVALUATORS and spec.required
    ]
    if unsupported_required:
        raise ValueError(f"Inspect adapter cannot run required evaluators: {unsupported_required}")
    if task.environment.kind != EnvironmentKind.NONE and sandbox is None:
        raise ValueError(
            "environment-backed tasks require an explicit Inspect sandbox configuration"
        )
    inspect_samples = [
        Sample(
            id=sample.sample_id,
            input=_prompt(task, sample.input),
            target=_json_text(
                sample.target
                if sample.target is not None
                else task.reference
                if task.reference is not None
                else ""
            ),
            metadata={
                **sample.metadata,
                "agenttrace.task_id": task.task_id,
                "agenttrace.task_version": task.version,
                "agenttrace.input": sample.input,
            },
            sandbox=sandbox,
        )
        for sample in samples
    ]
    ignored = [
        spec.evaluator_id for spec in task.evaluators if spec.kind not in SUPPORTED_EVALUATORS
    ]
    return Task(
        dataset=MemoryDataset(inspect_samples, name=f"{task.task_id}@{task.version}"),
        solver=solver or generate(),
        scorer=agenttrace_weighted_scorer(task.model_dump_json()),
        sandbox=sandbox,
        time_limit=task.environment.resources.timeout_seconds,
        name=task.task_id,
        display_name=task.title,
        version=task.version,
        metadata={
            "agenttrace_task": task.model_dump(mode="json"),
            "post_run_evaluators": ignored,
            **task.metadata,
        },
        tags=task.tags,
    )
