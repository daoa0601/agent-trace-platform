"""Docker environment command construction and fail-closed behavior."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from agenttrace.environments import DockerEnvironment
from agenttrace.tasks.models import EnvironmentKind, EnvironmentSpec, ResourceLimits


def test_docker_environment_requires_digest(tmp_path: Path) -> None:
    spec = EnvironmentSpec(kind=EnvironmentKind.DOCKER, image="agenttrace/sandbox:0.1.0")
    with pytest.raises(ValueError, match="image_digest"):
        DockerEnvironment(spec, workspace=tmp_path)


def test_docker_environment_hardening_setup_snapshot_and_cleanup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, timeout: int, check: bool):
        del timeout, check
        calls.append(argv)
        if argv[1] == "inspect":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps(
                    [
                        {
                            "Id": "container-id",
                            "Config": {"Image": "sandbox"},
                            "State": {"Running": True},
                        }
                    ]
                ),
                stderr="",
            )
        if argv[1] == "diff":
            return subprocess.CompletedProcess(argv, 0, stdout="C /workspace/out.txt\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(DockerEnvironment, "_run_host", staticmethod(fake_run))
    spec = EnvironmentSpec(
        kind=EnvironmentKind.DOCKER,
        image="agenttrace/sandbox:0.1.0",
        image_digest="sha256:" + "a" * 64,
        setup=[["python", "--version"]],
        state_capture="full",
        resources=ResourceLimits(cpu=0.5, memory_mb=256, timeout_seconds=10),
    )
    environment = DockerEnvironment(spec, workspace=tmp_path)
    environment.start()
    result = environment.execute(["python", "-c", "print('ok')"])
    snapshot = environment.snapshot()
    snapshot_object = cast(dict[str, JsonValue], snapshot)
    environment.close()
    run_command = calls[0]
    assert "--cap-drop" in run_command and "ALL" in run_command
    assert "no-new-privileges" in run_command
    assert ["--network", "none"] == run_command[run_command.index("--network") :][:2]
    assert result.succeeded
    assert snapshot_object["container_id"] == "container-id"
    assert snapshot_object["filesystem_diff"] == ["C /workspace/out.txt"]
    assert calls[-1][1:3] == ["rm", "--force"]
