"""Sample loading and model-output parsing contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenttrace.tasks import ParsedJson, load_task_samples, parse_json_output


def test_parse_json_output_distinguishes_null_from_non_json() -> None:
    assert parse_json_output("null") == ParsedJson(None)
    assert parse_json_output("```json\nnull\n```") == ParsedJson(None)
    assert parse_json_output(' {"a": 1} ') == ParsedJson({"a": 1})
    assert parse_json_output("not json") is None


def test_invalid_sample_error_names_the_field_without_echoing_its_value(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text('{"sample_id": "bad id with spaces", "input": "private-sample-text"}\n')

    with pytest.raises(ValueError) as raised:
        load_task_samples(path)

    message = str(raised.value)
    assert f"{path}:1" in message
    assert "sample_id" in message
    assert "bad id with spaces" not in message
    assert "private-sample-text" not in message


def test_validation_summary_uses_error_codes_not_messages_that_quote_values() -> None:
    from pydantic import BaseModel, ValidationError, field_validator

    from agenttrace.tasks import describe_validation_error

    class Strict(BaseModel):
        name: str

        @field_validator("name")
        @classmethod
        def reject(cls, value: str) -> str:
            raise ValueError(f"rejected {value}")

    with pytest.raises(ValidationError) as raised:
        Strict(name="private-value")

    summary = describe_validation_error(raised.value, Strict)
    assert summary == "name: value_error"
    assert "private-value" not in summary


def test_validation_summary_masks_user_keys_and_extra_fields() -> None:
    from pydantic import BaseModel, ConfigDict, ValidationError

    from agenttrace.tasks import describe_validation_error

    class Inner(BaseModel):
        model_config = ConfigDict(extra="forbid")
        count: int

    class Outer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        labels: dict[str, int]
        inner: list[Inner]

    with pytest.raises(ValidationError) as raised:
        Outer.model_validate(
            {
                "labels": {"private-key": "x"},
                "inner": [{"count": 1, "secret-extra": 1}],
                "top-secret": 1,
            }
        )

    summary = describe_validation_error(raised.value, Outer)
    assert "private-key" not in summary
    assert "secret-extra" not in summary
    assert "top-secret" not in summary
    assert "labels.<key>: int_parsing" in summary
    assert "inner.0.<extra field>: extra_forbidden" in summary
    assert "<extra field>: extra_forbidden" in summary

    unknown = describe_validation_error(raised.value)
    assert "labels" not in unknown


def test_invalid_sample_metadata_key_is_not_echoed(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text('{"sample_id": "ok", "input": 1, "private-extra-key": 2}\n')

    with pytest.raises(ValueError) as raised:
        load_task_samples(path)

    assert "private-extra-key" not in str(raised.value)
    assert "<extra field>: extra_forbidden" in str(raised.value)


def test_validation_summary_follows_model_types_for_keys_and_indices() -> None:
    from collections.abc import Callable

    from pydantic import BaseModel, ValidationError

    from agenttrace.tasks import describe_validation_error

    class Leaf(BaseModel):
        count: int

    class Other(BaseModel):
        name: str

    class Root(BaseModel):
        counters: dict[int, int]
        leaves: list[Leaf] | None = None
        pair: tuple[int, Leaf] | None = None
        choice: Leaf | Other | None = None
        hook: Callable[[], None] | None = None

    with pytest.raises(ValidationError) as raised:
        Root.model_validate(
            {
                "counters": {1234567890123456: "x"},
                "leaves": [{"count": "x"}],
                "pair": [1, {"count": "x"}],
                "choice": {"count": "x"},
            }
        )

    summary = describe_validation_error(raised.value, Root)
    assert "1234567890123456" not in summary
    assert "counters.<key>: int_parsing" in summary
    assert "leaves.0.count: int_parsing" in summary
    assert "pair.1.count: int_parsing" in summary
    assert "choice.Leaf.count: int_parsing" in summary
