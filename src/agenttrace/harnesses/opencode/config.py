"""Minimal OpenCode configuration generated for exactly one benchmark sample."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenttrace.providers.zai import ZAIReasoningEffort

COMMAND_NAME = "agenttrace-task"
PROVIDER_ID = "zai-coding-plan"


def build_opencode_config(
    *,
    prompt: str,
    model: str,
    small_model: str,
    broker_base_url: str,
    broker_token: str,
    reasoning_effort: ZAIReasoningEffort,
) -> dict[str, Any]:
    """Build an allowlisted config with no reusable provider credential."""
    provider_model = f"{PROVIDER_ID}/{model}"
    models: dict[str, Any] = {
        model: {
            "name": model,
            "variants": {
                effort.value: {"reasoningEffort": effort.value} for effort in ZAIReasoningEffort
            },
        }
    }
    models.setdefault(small_model, {"name": small_model})
    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "snapshot": False,
        "share": "disabled",
        "enabled_providers": [PROVIDER_ID],
        "plugin": [],
        "mcp": {},
        "model": provider_model,
        "small_model": f"{PROVIDER_ID}/{small_model}",
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Z.AI Coding Plan (AgentTrace broker)",
                "options": {
                    "apiKey": broker_token,
                    "baseURL": broker_base_url,
                },
                "models": models,
            }
        },
        "permission": {
            "read": "allow",
            "grep": "allow",
            "glob": "allow",
            "list": "allow",
            "edit": "allow",
            "bash": "allow",
            "external_directory": "deny",
            "webfetch": "deny",
            "websearch": "deny",
            "task": "deny",
        },
        "command": {
            COMMAND_NAME: {
                "template": prompt,
                "description": "Run one isolated AgentTrace benchmark sample.",
                "agent": "build",
                "model": provider_model,
                "subtask": False,
            }
        },
    }


def write_opencode_config(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path
