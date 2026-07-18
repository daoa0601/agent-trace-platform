"""Versioned, task-driven evaluators."""

from agenttrace.evaluators.base import EvaluationContext, Evaluator, EvaluatorRegistry
from agenttrace.evaluators.builtins import builtin_registry, evaluate_episode

__all__ = [
    "EvaluationContext",
    "Evaluator",
    "EvaluatorRegistry",
    "builtin_registry",
    "evaluate_episode",
]
