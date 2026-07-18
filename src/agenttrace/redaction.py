"""Deterministic collection-boundary redaction for secrets and sensitive data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import JsonValue

SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credentials",
    "password",
    "private_key",
    "secret",
    "secret_key",
    "session_cookie",
}
PRIVATE_REASONING_KEYS = {
    "chain_of_thought",
    "private_reasoning",
    "reasoning_content",
}

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[opurs]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"),
]
EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")


@dataclass(frozen=True)
class RedactionStats:
    secret_values: int = 0
    secret_keys: int = 0
    pii_values: int = 0
    private_reasoning: int = 0

    @property
    def total(self) -> int:
        return self.secret_values + self.secret_keys + self.pii_values + self.private_reasoning

    def add(self, other: RedactionStats) -> RedactionStats:
        return RedactionStats(
            secret_values=self.secret_values + other.secret_values,
            secret_keys=self.secret_keys + other.secret_keys,
            pii_values=self.pii_values + other.pii_values,
            private_reasoning=self.private_reasoning + other.private_reasoning,
        )


class Redactor:
    def __init__(self, mode: Literal["off", "balanced", "strict"] = "balanced") -> None:
        self.mode = mode

    def redact(self, value: JsonValue) -> tuple[JsonValue, RedactionStats]:
        if self.mode == "off":
            return value, RedactionStats()
        result, stats = self._walk(value)
        return cast(JsonValue, result), stats

    def _walk(self, value: JsonValue, key: str | None = None) -> tuple[JsonValue, RedactionStats]:
        normalized_key = key.lower() if key else None
        if normalized_key in SECRET_KEYS:
            return "[REDACTED_SECRET]", RedactionStats(secret_keys=1)
        if normalized_key in PRIVATE_REASONING_KEYS:
            return "[PRIVATE_REASONING_REMOVED]", RedactionStats(private_reasoning=1)

        if isinstance(value, dict):
            output: dict[str, JsonValue] = {}
            total = RedactionStats()
            for child_key, child_value in value.items():
                cleaned, stats = self._walk(child_value, child_key)
                output[child_key] = cleaned
                total = total.add(stats)
            return output, total
        if isinstance(value, list):
            output_list: list[JsonValue] = []
            total = RedactionStats()
            for child in value:
                cleaned, stats = self._walk(child)
                output_list.append(cleaned)
                total = total.add(stats)
            return output_list, total
        if isinstance(value, str):
            cleaned = value
            secret_count = 0
            for pattern in SECRET_PATTERNS:
                cleaned, count = pattern.subn("[REDACTED_SECRET]", cleaned)
                secret_count += count
            pii_count = 0
            if self.mode == "strict":
                cleaned, count = EMAIL_PATTERN.subn("[REDACTED_EMAIL]", cleaned)
                pii_count += count
                cleaned, count = PHONE_PATTERN.subn("[REDACTED_PHONE]", cleaned)
                pii_count += count
            return cleaned, RedactionStats(secret_values=secret_count, pii_values=pii_count)
        return value, RedactionStats()
