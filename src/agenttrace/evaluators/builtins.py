"""General evaluators that do not assume a coding task."""

from __future__ import annotations

import json
import re
from typing import cast

import jsonschema
from pydantic import JsonValue

from agenttrace.evaluators.base import EvaluationContext, EvaluatorRegistry
from agenttrace.models import Episode, EvaluationResult, StepType
from agenttrace.storage.blobs import BlobStore
from agenttrace.tasks.models import TaskDefinition
from agenttrace.tasks.samples import describe_schema_violation


def _result(
    context: EvaluationContext,
    *,
    passed: bool,
    score: float,
    details: dict[str, JsonValue] | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        evaluator_id=context.spec.evaluator_id,
        evaluator_version=context.spec.version,
        passed=passed,
        score=score,
        weight=context.spec.weight,
        reward_components={context.spec.evaluator_id: score * context.spec.weight},
        details=details or {},
    )


def _expected(context: EvaluationContext) -> JsonValue:
    return context.spec.config.get("expected", context.task.reference)


def _float_config(context: EvaluationContext, key: str, default: float) -> float:
    value = context.spec.config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context.spec.kind} evaluator requires numeric config.{key}")
    return float(value)


class ExactMatchEvaluator:
    kind = "exact_match"

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        actual = context.final_output()
        expected = _expected(context)
        case_sensitive = bool(context.spec.config.get("case_sensitive", True))
        strip = bool(context.spec.config.get("strip", True))
        if isinstance(actual, str) and isinstance(expected, str):
            if strip:
                actual, expected = actual.strip(), expected.strip()
            if not case_sensitive:
                actual, expected = actual.casefold(), expected.casefold()
        passed = actual == expected
        return _result(context, passed=passed, score=1.0 if passed else 0.0)


class ContainsEvaluator:
    kind = "contains"

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        actual = context.final_output()
        text = actual if isinstance(actual, str) else json.dumps(actual, sort_keys=True)
        configured = context.spec.config.get("values", context.spec.config.get("value"))
        values = configured if isinstance(configured, list) else [configured]
        expected = [str(value) for value in values if value is not None]
        case_sensitive = bool(context.spec.config.get("case_sensitive", False))
        haystack = text if case_sensitive else text.casefold()
        found: list[JsonValue] = [
            value
            for value in expected
            if (value if case_sensitive else value.casefold()) in haystack
        ]
        score = len(found) / len(expected) if expected else 0.0
        return _result(
            context,
            passed=score == 1.0,
            score=score,
            details={"matched": found, "expected_count": len(expected)},
        )


class RegexEvaluator:
    kind = "regex"

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        pattern = context.spec.config.get("pattern")
        if not isinstance(pattern, str):
            raise ValueError("regex evaluator requires config.pattern")
        actual = context.final_output()
        text = actual if isinstance(actual, str) else json.dumps(actual, sort_keys=True)
        flags = re.IGNORECASE if context.spec.config.get("ignore_case", False) else 0
        matched = re.search(pattern, text, flags=flags) is not None
        return _result(context, passed=matched, score=1.0 if matched else 0.0)


class JsonSchemaEvaluator:
    kind = "json_schema"

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        schema = context.spec.config.get("schema", context.task.expected_output_schema)
        if not isinstance(schema, dict):
            raise ValueError("json_schema evaluator requires an output schema")
        actual = context.final_output()
        if isinstance(actual, str):
            try:
                actual = cast(JsonValue, json.loads(actual))
            except json.JSONDecodeError as error:
                return _result(
                    context,
                    passed=False,
                    score=0.0,
                    details={"validation_error": f"invalid JSON: {error.msg}"},
                )
        try:
            jsonschema.validate(actual, schema)
        except jsonschema.ValidationError as error:
            return _result(
                context,
                passed=False,
                score=0.0,
                details=describe_schema_violation(error),
            )
        return _result(context, passed=True, score=1.0)


class ToolSuccessEvaluator:
    kind = "tool_success"

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        tools = [
            event.tool
            for event in context.episode.steps
            if event.step_type in {StepType.TOOL_CALL, StepType.TOOL_RESULT}
            and event.tool is not None
            and event.tool.success is not None
        ]
        successful = sum(tool.success is True for tool in tools)
        rate = successful / len(tools) if tools else 0.0
        minimum = _float_config(context, "minimum_success_rate", 1.0)
        return _result(
            context,
            passed=rate >= minimum,
            score=rate,
            details={"successful": successful, "total": len(tools), "minimum": minimum},
        )


class LatencyBudgetEvaluator:
    kind = "latency_budget"

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        maximum = _float_config(context, "max_latency_ms", 0.0)
        if maximum <= 0:
            raise ValueError("latency_budget evaluator requires positive config.max_latency_ms")
        total = sum(event.metrics.latency_ms or 0 for event in context.episode.steps)
        passed = total <= maximum
        score = max(0.0, 1.0 - total / maximum) if maximum else 0.0
        if passed and total == 0:
            score = 1.0
        return _result(
            context,
            passed=passed,
            score=score,
            details={"latency_ms": total, "budget_ms": maximum},
        )


def builtin_registry() -> EvaluatorRegistry:
    registry = EvaluatorRegistry()
    for evaluator in (
        ExactMatchEvaluator(),
        ContainsEvaluator(),
        RegexEvaluator(),
        JsonSchemaEvaluator(),
        ToolSuccessEvaluator(),
        LatencyBudgetEvaluator(),
    ):
        registry.register(evaluator)
    return registry


def evaluate_episode(
    episode: Episode,
    task: TaskDefinition,
    blobs: BlobStore,
    *,
    registry: EvaluatorRegistry | None = None,
) -> list[EvaluationResult]:
    registry = registry or builtin_registry()
    results = []
    for spec in task.evaluators:
        if spec.kind == "custom":
            evaluator = registry.get(spec.evaluator_id)
        else:
            evaluator = registry.get(spec.kind)
        context = EvaluationContext(episode=episode, task=task, spec=spec, blobs=blobs)
        results.append(evaluator.evaluate(context))
    return results
