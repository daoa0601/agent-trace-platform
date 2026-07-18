"""Validated process configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CollectorSettings(BaseSettings):
    """Collector settings loaded from AGENTTRACE_* environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AGENTTRACE_",
        env_file=".env",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    api_keys: str = ""
    storage_backend: Literal["file", "s3"] = "file"
    object_root: Path = Path("data/object-store")
    s3_bucket: str = "agent-traces"
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    max_request_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    inline_payload_bytes: int = Field(default=16 * 1024, ge=256)
    redaction_mode: Literal["off", "balanced", "strict"] = "balanced"
    log_level: str = "INFO"

    @property
    def parsed_api_keys(self) -> tuple[str, ...]:
        return tuple(key.strip() for key in self.api_keys.split(",") if key.strip())

    @model_validator(mode="after")
    def validate_production(self) -> CollectorSettings:
        if self.environment == "production" and not self.parsed_api_keys:
            raise ValueError("production requires at least one AGENTTRACE_API_KEYS value")
        if self.storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("S3 storage requires AGENTTRACE_S3_BUCKET")
        return self


@lru_cache(maxsize=1)
def get_settings() -> CollectorSettings:
    return CollectorSettings()
