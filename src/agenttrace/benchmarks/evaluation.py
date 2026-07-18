"""Inline benchmark evaluators shared by direct provider runners."""

from __future__ import annotations

import re
from typing import cast

import jsonschema
from pydantic import JsonValue

from agenttrace.benchmarks.models import BenchmarkEvaluation
from agenttrace.tasks.models import EvaluatorSpec, TaskDefinition
from agenttrace.tasks.samples import TaskSample, parse_json_output

SUPPORTED_DIRECT_EVALUATORS = {
    "exact_match",
    "contains",
    "regex",
    "json_schema",
    "latency_budget",
}


def validate_direct_task(task: TaskDefinition) -> None:
    unsupported = [
        spec.evaluator_id
        for spec in task.evaluators
        if spec.required and spec.kind not in SUPPORTED_DIRECT_EVALUATORS
    ]
    if unsupported:
        raise ValueError(f"direct API runner cannot execute required evaluators: {unsupported}")
    if task.environment.kind.value != "none":
        raise ValueError("direct API runner requires environment.kind=none; use an agent harness")


def _expected(spec: EvaluatorSpec, task: TaskDefinition, sample: TaskSample) -> JsonValue:
    configured = spec.config.get("expected")
    if configured is not None:
        return configured
    if sample.target is not None:
        return sample.target
    return task.reference


def _completed(
    spec: EvaluatorSpec,
    *,
    passed: bool,
    score: float,
    details: dict[str, JsonValue] | None = None,
) -> BenchmarkEvaluation:
    return BenchmarkEvaluation(
        evaluator_id=spec.evaluator_id,
        evaluator_version=spec.version,
        kind=spec.kind,
        required=spec.required,
        weight=spec.weight,
        status="passed" if passed else "failed",
        passed=passed,
        score=score,
        details=details or {},
    )


def evaluate_response(
    task: TaskDefinition,
    sample: TaskSample,
    output_text: str,
    *,
    latency_ms: float,
) -> tuple[list[BenchmarkEvaluation], float, bool]:
    """Evaluate one output without coupling the result to a specific eval framework."""
    parsed = parse_json_output(output_text)
    outcomes: list[BenchmarkEvaluation] = []
    for spec in task.evaluators:
        if spec.kind not in SUPPORTED_DIRECT_EVALUATORS:
            outcomes.append(
                BenchmarkEvaluation(
                    evaluator_id=spec.evaluator_id,
                    evaluator_version=spec.version,
                    kind=spec.kind,
                    required=spec.required,
                    weight=spec.weight,
                    status="skipped",
                    passed=None,
                    score=None,
                    details={"reason": "requires an environment or custom evaluator registry"},
                )
            )
            continue
        if spec.kind == "exact_match":
            actual: JsonValue = parsed if parsed is not None else output_text.strip()
            expected = _expected(spec, task, sample)
            if isinstance(actual, str) and isinstance(expected, str):
                if bool(spec.config.get("strip", True)):
                    actual, expected = actual.strip(), expected.strip()
                if not bool(spec.config.get("case_sensitive", True)):
                    actual, expected = actual.casefold(), expected.casefold()
            passed = actual == expected
            outcomes.append(_completed(spec, passed=passed, score=float(passed)))
            continue
        if spec.kind == "contains":
            configured = spec.config.get(
                "values",
                spec.config.get("value", _expected(spec, task, sample)),
            )
            values = configured if isinstance(configured, list) else [configured]
            expected = [str(value) for value in values if value is not None]
            case_sensitive = bool(spec.config.get("case_sensitive", False))
            haystack = output_text if case_sensitive else output_text.casefold()
            matched = [
                value
                for value in expected
                if (value if case_sensitive else value.casefold()) in haystack
            ]
            score = len(matched) / len(expected) if expected else 0.0
            outcomes.append(
                _completed(
                    spec,
                    passed=score == 1.0,
                    score=score,
                    details={
                        "matched": cast(JsonValue, matched),
                        "expected_count": len(expected),
                    },
                )
            )
            continue
        if spec.kind == "regex":
            pattern = spec.config.get("pattern")
            if not isinstance(pattern, str):
                raise ValueError(f"regex evaluator {spec.evaluator_id} requires config.pattern")
            flags = re.IGNORECASE if spec.config.get("ignore_case", False) else 0
            passed = re.search(pattern, output_text, flags=flags) is not None
            outcomes.append(_completed(spec, passed=passed, score=float(passed)))
            continue
        if spec.kind == "json_schema":
            schema = spec.config.get("schema", task.expected_output_schema)
            if not isinstance(schema, dict):
                raise ValueError(f"json_schema evaluator {spec.evaluator_id} requires a schema")
            if parsed is None:
                outcomes.append(
                    _completed(
                        spec,
                        passed=False,
                        score=0.0,
                        details={"validation_error": "output is not valid JSON"},
                    )
                )
                continue
            try:
                jsonschema.validate(parsed, cast(dict[str, object], schema))
            except jsonschema.ValidationError as error:
                outcomes.append(
                    _completed(
                        spec,
                        passed=False,
                        score=0.0,
                        details={"validation_error": error.message},
                    )
                )
            else:
                outcomes.append(_completed(spec, passed=True, score=1.0))
            continue
        if spec.kind == "latency_budget":
            configured = spec.config.get("max_latency_ms")
            if isinstance(configured, bool) or not isinstance(configured, int | float):
                raise ValueError(
                    f"latency evaluator {spec.evaluator_id} requires numeric config.max_latency_ms"
                )
            maximum = float(configured)
            if maximum <= 0:
                raise ValueError(
                    f"latency evaluator {spec.evaluator_id} requires positive config.max_latency_ms"
                )
            passed = latency_ms <= maximum
            score = max(0.0, 1.0 - latency_ms / maximum)
            outcomes.append(
                _completed(
                    spec,
                    passed=passed,
                    score=score,
                    details={"latency_ms": latency_ms, "budget_ms": maximum},
                )
            )

    scored = [outcome for outcome in outcomes if outcome.score is not None]
    total_weight = sum(outcome.weight for outcome in scored)
    aggregate = (
        sum(cast(float, outcome.score) * outcome.weight for outcome in scored) / total_weight
        if total_weight
        else 0.0
    )
    required_passed = all(outcome.passed is True for outcome in outcomes if outcome.required)
    return outcomes, aggregate, required_passed
