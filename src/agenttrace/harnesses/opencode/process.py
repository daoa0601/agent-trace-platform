"""No-shell OpenCode subprocess lifecycle with process-group timeout handling."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from agenttrace.harnesses.opencode.config import COMMAND_NAME, PROVIDER_ID
from agenttrace.providers.zai import ZAIReasoningEffort


@dataclass(frozen=True)
class OpenCodeProcessResult:
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool


def isolated_child_environment(runtime_root: Path) -> dict[str, str]:
    """Pass only non-secret host settings; provider credentials stay in the parent broker."""
    allowed = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TERM"}
    }
    home = runtime_root / "home"
    config = runtime_root / "config"
    data = runtime_root / "data"
    state = runtime_root / "state"
    custom = runtime_root / "custom"
    for path in (home, config, data, state, custom):
        path.mkdir(parents=True, exist_ok=True)
    allowed.update(
        {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(config),
            "XDG_DATA_HOME": str(data),
            "XDG_STATE_HOME": str(state),
            "OPENCODE_CONFIG_DIR": str(custom),
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "OPENCODE_DISABLE_PRUNE": "true",
            "OPENCODE_DISABLE_TERMINAL_TITLE": "true",
        }
    )
    return allowed


def run_opencode(
    *,
    executable: str,
    workspace: Path,
    config_path: Path,
    runtime_root: Path,
    model: str,
    reasoning_effort: ZAIReasoningEffort,
    sample_id: str,
    timeout_seconds: int | None,
) -> OpenCodeProcessResult:
    resolved = shutil.which(executable)
    if resolved is None:
        raise ValueError(f"OpenCode executable was not found: {executable}")
    argv = (
        resolved,
        "--pure",
        "run",
        "--auto",
        "--format",
        "json",
        "--model",
        f"{PROVIDER_ID}/{model}",
        "--variant",
        reasoning_effort.value,
        "--agent",
        "build",
        "--command",
        COMMAND_NAME,
        "--title",
        sample_id,
    )
    environment = isolated_child_environment(runtime_root)
    environment["OPENCODE_CONFIG"] = str(config_path)
    started = time.perf_counter()
    process = subprocess.Popen(
        argv,
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    return OpenCodeProcessResult(
        argv=argv,
        exit_code=124 if timed_out else process.returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=time.perf_counter() - started,
        timed_out=timed_out,
    )
