"""The deployment target interface every target implements, plus the shared helpers
(rendered files on disk, readiness polling)."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Callable

from evalbuilder.deploy.runner import CommandRunner
from evalbuilder.deploy.spec import DeploymentRecord, DeploymentSpec


class DeploymentError(RuntimeError):
    """A deployment step failed; the message says which command or wait."""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def default_health(endpoint: str, timeout: float = 5.0) -> dict:
    from evalbuilder.agent_client import RemoteAgent

    return RemoteAgent(endpoint, timeout=timeout).health()


class DeploymentTarget:
    """`up` → record with an endpoint; `status` → live facts; `down` → nothing left."""

    kind = "base"

    def __init__(self, runner: CommandRunner | None = None, log: Callable[[str], None] | None = None,
                 health: Callable[[str], dict] | None = None, sleep: Callable[[float], None] = time.sleep):
        self.log = log or (lambda msg: None)
        self.runner = runner or CommandRunner(log=self.log)
        self.health = health or default_health
        self.sleep = sleep

    # ── interface ──
    def available(self) -> tuple[bool, str]:
        return True, "ok"

    def render(self, spec: DeploymentSpec) -> dict[str, str]:
        """file name → content of every generated file (nothing for the local target)."""
        return {}

    def build(self, spec: DeploymentSpec) -> str:
        """Build the image; returns its reference ('' when the target has none)."""
        return ""

    def up(self, spec: DeploymentSpec) -> DeploymentRecord:
        raise NotImplementedError

    def status(self, spec: DeploymentSpec, record: DeploymentRecord) -> dict:
        raise NotImplementedError

    def logs(self, spec: DeploymentSpec, record: DeploymentRecord, lines: int = 100) -> str:
        return ""

    def down(self, spec: DeploymentSpec, record: DeploymentRecord) -> None:
        raise NotImplementedError

    # ── helpers ──
    def write_files(self, spec: DeploymentSpec) -> dict[str, Path]:
        """Render into `spec.work_dir`; returns file name → path."""
        out: dict[str, Path] = {}
        spec.work_path.mkdir(parents=True, exist_ok=True)
        for name, content in self.render(spec).items():
            path = spec.work_path / name
            path.write_text(content)
            out[name] = path
        return out

    def new_record(self, spec: DeploymentSpec) -> DeploymentRecord:
        return DeploymentRecord(name=spec.name, target=self.kind, image=spec.image, expose=spec.expose, spec=spec.to_dict())

    def wait_healthy(self, endpoint: str, timeout: float, interval: float = 1.0,
                     alive: Callable[[], bool] | None = None, diagnose: Callable[[], str] | None = None) -> dict:
        """Poll `/health` until it answers ok; raises DeploymentError after `timeout`
        seconds — or at once when `alive()` says the process behind the endpoint is gone
        (`diagnose()` adds its last log lines to the message)."""
        deadline = time.time() + timeout
        last: dict = {}
        while True:
            last = self.health(endpoint)
            if last.get("ok"):
                return last
            if alive is not None and not alive():
                tail = diagnose() if diagnose is not None else ""
                raise DeploymentError(f"the agent server behind {endpoint} exited before becoming healthy" + (f": {tail}" if tail else ""))
            if time.time() >= deadline:
                raise DeploymentError(f"{endpoint} did not become healthy within {timeout}s: {last.get('error') or last}")
            self.sleep(interval)

    def finish(self, record: DeploymentRecord, endpoint: str, timeout: float,
               alive: Callable[[], bool] | None = None, diagnose: Callable[[], str] | None = None) -> DeploymentRecord:
        """Wait for the endpoint, store the health facts and the command history."""
        health = self.wait_healthy(endpoint, timeout, alive=alive, diagnose=diagnose)
        record.endpoint = endpoint
        record.status = "up"
        record.details["health"] = {k: v for k, v in health.items() if k in ("module", "factory", "tools", "agent_model", "mock_model", "server")}
        record.commands = list(self.runner.history)
        record.touch()
        return record

    def failed(self, record: DeploymentRecord, error: Exception) -> DeploymentRecord:
        record.status = "failed"
        record.details["error"] = f"{type(error).__name__}: {error}"
        record.commands = list(self.runner.history)
        record.touch()
        return record
