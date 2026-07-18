"""Provider-neutral task samples and prompt rendering."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agenttrace.tasks.models import TaskDefinition


class TaskSample(BaseModel):
    """One portable task instance, independent of model and eval harness."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    input: JsonValue
    target: JsonValue | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


def canonical_json(value: JsonValue) -> bytes:
    """Serialize JSON deterministically for prompts, hashes, and provenance."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def json_text(value: JsonValue) -> str:
    return value if isinstance(value, str) else canonical_json(value).decode("utf-8")


def digest_json(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def load_task_samples(path: Path) -> list[TaskSample]:
    """Load a strict JSONL sample set with unique, stable IDs."""
    samples: list[TaskSample] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read sample file {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            samples.append(TaskSample.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid sample at {path}:{line_number}: {error}") from error
    if not samples:
        raise ValueError(f"sample file is empty: {path}")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"sample IDs must be unique: {path}")
    return samples


def render_task_prompt(task: TaskDefinition, value: JsonValue) -> str:
    """Render the common prompt used by direct API and third-party eval adapters."""
    prompt = [task.description, "", "Input:", json_text(value)]
    if task.expected_output_schema is not None:
        prompt.extend(
            [
                "",
                "Return only JSON matching this schema:",
                json_text(task.expected_output_schema),
            ]
        )
    return "\n".join(prompt)


def parse_json_output(text: str) -> JsonValue | None:
    """Parse plain or single-fenced JSON without attempting lossy repairs."""
    candidate = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        candidate,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        candidate = fenced.group(1)
    try:
        return cast(JsonValue, json.loads(candidate))
    except json.JSONDecodeError:
        return None
