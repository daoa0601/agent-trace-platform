"""YAML task registry with duplicate and schema checks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from agenttrace.tasks.models import TaskDefinition


def load_task(path: Path) -> TaskDefinition:
    """Load one strict task definition from YAML or JSON-compatible YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"task file must contain an object: {path}")
    return TaskDefinition.model_validate(payload)


class TaskRegistry:
    """In-memory immutable registry keyed by (task_id, version)."""

    def __init__(self, tasks: Iterable[TaskDefinition]) -> None:
        self._tasks: dict[tuple[str, str], TaskDefinition] = {}
        for task in tasks:
            key = (task.task_id, task.version)
            if key in self._tasks:
                raise ValueError(f"duplicate task definition: {task.task_id}@{task.version}")
            self._tasks[key] = task

    @classmethod
    def from_directory(cls, root: Path) -> TaskRegistry:
        paths = sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")])
        return cls(load_task(path) for path in paths)

    def get(self, task_id: str, version: str) -> TaskDefinition:
        try:
            return self._tasks[(task_id, version)]
        except KeyError as error:
            raise KeyError(f"unknown task: {task_id}@{version}") from error

    def list(self) -> list[TaskDefinition]:
        return [self._tasks[key] for key in sorted(self._tasks)]
