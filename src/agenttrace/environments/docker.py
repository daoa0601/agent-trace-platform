"""Hardened Docker task environment driven by EnvironmentSpec."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from agenttrace.ids import new_id
from agenttrace.tasks.models import EnvironmentKind, EnvironmentSpec


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


class DockerEnvironment:
    """One isolated container with a bounded writable workspace."""

    def __init__(
        self,
        spec: EnvironmentSpec,
        *,
        workspace: Path,
        require_image_digest: bool = True,
    ) -> None:
        if spec.kind != EnvironmentKind.DOCKER:
            raise ValueError("DockerEnvironment requires kind=docker")
        if require_image_digest and spec.image_digest is None:
            raise ValueError("production Docker environments require image_digest")
        if not workspace.is_dir():
            raise ValueError(f"workspace does not exist: {workspace}")
        self.spec = spec
        self.workspace = workspace.resolve()
        self.container_name = new_id("agenttrace").replace("_", "-")[:63]
        self._started = False

    @property
    def image_ref(self) -> str:
        assert self.spec.image is not None
        if self.spec.image_digest:
            return f"{self.spec.image}@{self.spec.image_digest}"
        return self.spec.image

    def start(self) -> None:
        if self._started:
            raise RuntimeError("environment already started")
        network = "none" if self.spec.network == "none" else self.spec.network_name or "bridge"
        command = [
            "docker",
            "run",
            "--detach",
            "--name",
            self.container_name,
            "--network",
            network,
            "--cpus",
            str(self.spec.resources.cpu),
            "--memory",
            f"{self.spec.resources.memory_mb}m",
            "--pids-limit",
            "256",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--mount",
            f"type=bind,src={self.workspace},dst=/workspace,rw",
            "--workdir",
            "/workspace",
        ]
        for key, value in sorted(self.spec.environment.items()):
            command.extend(["--env", f"{key}={value}"])
        command.extend([self.image_ref, "sleep", "infinity"])
        self._run_host(command, timeout=60, check=True)
        self._started = True
        try:
            for setup_command in self.spec.setup:
                result = self.execute(setup_command)
                if not result.succeeded:
                    raise RuntimeError(
                        f"environment setup failed ({result.exit_code}): {result.stderr.strip()}"
                    )
        except Exception:
            self.close()
            raise

    def execute(self, argv: list[str], *, timeout_seconds: int | None = None) -> CommandResult:
        if not self._started:
            raise RuntimeError("environment is not started")
        if not argv:
            raise ValueError("command cannot be empty")
        timeout = timeout_seconds or self.spec.resources.timeout_seconds
        result = self._run_host(
            ["docker", "exec", self.container_name, *argv],
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            argv=argv,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def snapshot(self) -> JsonValue:
        if not self._started:
            raise RuntimeError("environment is not started")
        inspect = self._run_host(
            ["docker", "inspect", self.container_name],
            timeout=30,
            check=True,
        )
        payload = json.loads(inspect.stdout)[0]
        snapshot: dict[str, JsonValue] = {
            "container_id": payload["Id"],
            "image": payload["Config"]["Image"],
            "state": payload["State"],
        }
        if self.spec.state_capture == "full":
            diff = self._run_host(
                ["docker", "diff", self.container_name],
                timeout=30,
                check=True,
            )
            snapshot["filesystem_diff"] = list[JsonValue](diff.stdout.splitlines())
        return snapshot

    def close(self) -> None:
        if not self._started:
            return
        self._run_host(
            ["docker", "rm", "--force", self.container_name],
            timeout=30,
            check=False,
        )
        self._started = False

    @staticmethod
    def _run_host(
        argv: list[str],
        *,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )

    def __enter__(self) -> DockerEnvironment:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
