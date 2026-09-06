"""The `local` target: the agent server as a detached subprocess of this interpreter —
no Docker, same code path (HTTP) as every container target. What the offline tests,
the UI browser tests and a config without a `deploy` section use."""

from __future__ import annotations

import sys
from pathlib import Path

from evalbuilder.deploy.base import DeploymentTarget, free_port
from evalbuilder.deploy.render import server_env
from evalbuilder.deploy.spec import DeploymentRecord, DeploymentSpec

SERVE_LOG = "serve.log"


class LocalTarget(DeploymentTarget):
    kind = "local"

    def available(self) -> tuple[bool, str]:
        return True, "runs `evalbuilder serve` as a subprocess of this interpreter"

    def up(self, spec: DeploymentSpec) -> DeploymentRecord:
        record = self.new_record(spec)
        port = free_port()
        argv = [sys.executable, "-m", "evalbuilder.cli", "serve", "--module", spec.module, "--factory", spec.factory,
                "--host", "127.0.0.1", "--port", str(port)]
        log_path = spec.work_path / SERVE_LOG
        try:
            pid = self.runner.spawn(argv, log_path=log_path, env=server_env(spec), cwd=spec.project_root)
            record.resources = {"pid": pid, "port": port, "log": str(log_path), "argv": argv}
            return self.finish(record, f"http://127.0.0.1:{port}", spec.timeout,
                               alive=lambda: self.runner.pid_alive(pid),
                               diagnose=lambda: " | ".join(self.logs(spec, record, lines=6).splitlines()[-6:]))
        except Exception as e:  # noqa: BLE001 - the record keeps the diagnosis; the caller decides
            self.runner.kill(record.resources.get("pid"))
            self.failed(record, e)
            raise

    def status(self, spec: DeploymentSpec, record: DeploymentRecord) -> dict:
        pid = (record.resources or {}).get("pid")
        alive = self.runner.pid_alive(pid)
        health = self.health(record.endpoint) if (alive and record.endpoint) else {"ok": False, "error": "server process is not running"}
        return {"ready": bool(alive and health.get("ok")), "endpoint": record.endpoint, "health": health,
                "replicas": {"ready": 1 if alive else 0, "wanted": 1}, "details": {"pid": pid, "alive": alive}}

    def logs(self, spec: DeploymentSpec, record: DeploymentRecord, lines: int = 100) -> str:
        path = Path((record.resources or {}).get("log") or spec.work_path / SERVE_LOG)
        if not path.exists():
            return ""
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])

    def down(self, spec: DeploymentSpec, record: DeploymentRecord) -> None:
        self.runner.kill((record.resources or {}).get("pid"))
