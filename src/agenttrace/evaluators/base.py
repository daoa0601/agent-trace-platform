"""Evaluator protocols and content resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue

from agenttrace.models import Content, Episode, EvaluationResult, StepType
from agenttrace.storage.blobs import BlobStore
from agenttrace.tasks.models import EvaluatorSpec, TaskDefinition


@dataclass(frozen=True)
class EvaluationContext:
    episode: Episode
    task: TaskDefinition
    spec: EvaluatorSpec
    blobs: BlobStore

    def resolve(self, content: Content | None) -> JsonValue:
        if content is None:
            return None
        return self.blobs.get_json(content)

    def final_output(self) -> JsonValue:
        final = next(
            (
                event
                for event in reversed(self.episode.steps)
                if event.step_type == StepType.FINAL_OUTPUT
            ),
            None,
        )
        if final is None:
            final = next(
                (
                    event
                    for event in reversed(self.episode.steps)
                    if event.step_type == StepType.MODEL_GENERATION and event.output is not None
                ),
                None,
            )
        return self.resolve(final.output) if final else None


class Evaluator(Protocol):
    kind: str

    def evaluate(self, context: EvaluationContext) -> EvaluationResult: ...


class EvaluatorRegistry:
    def __init__(self) -> None:
        self._evaluators: dict[str, Evaluator] = {}

    def register(self, evaluator: Evaluator) -> None:
        if evaluator.kind in self._evaluators:
            raise ValueError(f"evaluator already registered: {evaluator.kind}")
        self._evaluators[evaluator.kind] = evaluator

    def get(self, kind: str) -> Evaluator:
        try:
            return self._evaluators[kind]
        except KeyError as error:
            raise KeyError(f"no evaluator registered for kind {kind!r}") from error
