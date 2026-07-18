"""Parent-process-only Z.AI credential resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr


@dataclass(frozen=True)
class ZAICredentialSource:
    """Resolve exactly one secret source without exposing it to the agent process."""

    api_key_file: Path | None = None
    environment_variable: str = "ZAI_API_KEY"

    def resolve(self, environ: dict[str, str] | None = None) -> SecretStr:
        values = os.environ if environ is None else environ
        environment_value = values.get(self.environment_variable, "").strip()
        if self.api_key_file is not None and environment_value:
            raise ValueError(f"set only one of {self.environment_variable} or --api-key-file")
        if self.api_key_file is not None:
            path = self.api_key_file.expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"Z.AI API key file is not readable: {path}")
            value = path.read_text(encoding="utf-8").strip()
        else:
            value = environment_value
        if not value:
            raise ValueError(f"set {self.environment_variable} or provide --api-key-file")
        return SecretStr(value)
