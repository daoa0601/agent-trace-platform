"""Provider-neutral task samples and prompt rendering."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Annotated, Union, cast, get_args, get_origin

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from pydantic.fields import FieldInfo

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
        except ValidationError as error:
            summary = describe_validation_error(error, TaskSample)
            raise ValueError(f"invalid sample at {path}:{line_number}: {summary}") from error
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


def describe_validation_error(error: ValidationError, model: type[BaseModel] | None = None) -> str:
    """Summarize a Pydantic error by field location and error code, never by the offending input.

    Pydantic's human ``msg`` can quote rejected values (union tags, custom validators), so only
    the stable error ``type`` code is used. Locations can also carry user-supplied mapping keys
    (string or integer) or extra field names, so the location is followed through ``model``'s
    annotations: declared field names and sequence indices are shown, anything else is masked.
    Without ``model`` every location part is masked.
    """
    summaries: list[str] = []
    for detail in error.errors(include_url=False, include_input=False, include_context=False):
        parts = _safe_location(model, tuple(detail["loc"]))
        if detail["type"] == "extra_forbidden" and parts:
            parts[-1] = "<extra field>"
        summaries.append(f"{'.'.join(parts) or '<root>'}: {detail['type']}")
    return "; ".join(summaries)


_SEQUENCE_ORIGINS = (list, tuple, set, frozenset, Sequence, AbstractSet)
_MAPPING_ORIGINS = (dict, Mapping)


def _safe_location(model: type[BaseModel] | None, loc: tuple[int | str, ...]) -> list[str]:
    """Follow ``loc`` through the annotations it was produced from, masking non-schema parts."""
    current: object = model
    parts: list[str] = []
    for part in loc:
        current = _strip_optional(current)
        members = _union_members(current)
        if members:
            # Pydantic tags a union branch by its member's name; other tags are opaque.
            tagged = next(
                (m for m in members if isinstance(m, type) and m.__name__ == part),
                None,
            )
            parts.append(str(part) if tagged is not None else "<key>")
            current = tagged
            continue
        origin = get_origin(current)
        args = get_args(current)
        if isinstance(current, type) and issubclass(current, BaseModel):
            field = _field_named(current, part)
            parts.append(str(part) if field is not None else "<key>")
            current = field.annotation if field is not None else None
        elif isinstance(origin, type) and issubclass(origin, _MAPPING_ORIGINS):
            parts.append("<key>")
            current = args[1] if len(args) == 2 else None
        elif (
            isinstance(origin, type)
            and issubclass(origin, _SEQUENCE_ORIGINS)
            and isinstance(part, int)
        ):
            parts.append(str(part))
            if origin is tuple and args and args[-1] is not Ellipsis:
                current = args[part] if part < len(args) else None
            else:
                current = args[0] if args else None
        else:
            parts.append("<key>")
            current = None
    return parts


def _strip_optional(annotation: object) -> object:
    while True:
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
            continue
        members = _union_members(annotation)
        if len(members) == 1:
            annotation = members[0]
            continue
        return annotation


def _union_members(annotation: object) -> list[object]:
    if get_origin(annotation) not in (Union, UnionType):
        return []
    return [arg for arg in get_args(annotation) if arg is not type(None)]


def _field_named(model: type[BaseModel], part: int | str) -> FieldInfo | None:
    if not isinstance(part, str):
        return None
    for name, field in model.model_fields.items():
        if part in (name, field.alias, field.validation_alias):
            return field
    return None


def describe_schema_violation(error: jsonschema.ValidationError) -> dict[str, JsonValue]:
    """Describe a JSON Schema failure by schema location only.

    ``error.message`` quotes the offending instance, which would copy model output into
    metadata-only records, so it is deliberately not used.
    """
    return {
        "validation_error": "output does not match the schema",
        "validator": str(error.validator),
        "schema_path": "/".join(str(part) for part in error.absolute_schema_path),
    }


@dataclass(frozen=True)
class ParsedJson:
    """A successfully parsed JSON document; ``value`` may itself be JSON ``null``."""

    value: JsonValue


def parse_json_output(text: str) -> ParsedJson | None:
    """Parse plain or single-fenced JSON without attempting lossy repairs.

    Returns ``None`` only when the text is not JSON, so a literal ``null`` stays distinguishable.
    """
    candidate = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        candidate,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        candidate = fenced.group(1)
    try:
        return ParsedJson(cast(JsonValue, json.loads(candidate)))
    except json.JSONDecodeError:
        return None
