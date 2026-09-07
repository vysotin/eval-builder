"""The `docker` target: the agent image built and run by Docker Compose on the local
daemon (`docker compose -p <name> up -d --build --wait`), reachable on
`http://127.0.0.1:<host_port>` (`deploy.host_port`, or the container port)."""

from __future__ import annotations

import json
from pathlib import Path

from evalbuilder.deploy.base import DeploymentTarget
from evalbuilder.deploy.render import render_compose, render_dockerfile
from evalbuilder.deploy.spec import DeploymentRecord, DeploymentSpec

COMPOSE_FILE = "compose.yaml"
DOCKERFILE = "Dockerfile"
SERVICE = "agent"


class DockerComposeTarget(DeploymentTarget):
    kind = "docker"

    def available(self) -> tuple[bool, str]:
        if not self.runner.which("docker"):
            return False, "docker is not on PATH"
        info = self.runner.run(["docker", "info", "--format", "{{.ServerVersion}}"], check=False, timeout=30)
        if not info.ok:
            return False, f"docker daemon not reachable: {(info.stderr or info.stdout).strip()[:200]}"
        compose = self.runner.run(["docker", "compose", "version", "--short"], check=False, timeout=30)
        if not compose.ok:
            return False, "docker compose (v2) is not available"
        return True, f"docker {info.stdout.strip()} / compose {compose.stdout.strip()}"

    def dockerfile_path(self, spec: DeploymentSpec) -> Path:
        return Path(spec.dockerfile) if spec.dockerfile else spec.work_path / DOCKERFILE

    def render(self, spec: DeploymentSpec) -> dict[str, str]:
        files: dict[str, str] = {}
        if not spec.dockerfile:
            files[DOCKERFILE] = render_dockerfile(spec)
        files[COMPOSE_FILE] = render_compose(spec, self.dockerfile_path(spec))
        return files

    def _compose(self, spec: DeploymentSpec, *args: str) -> list[str]:
        return ["docker", "compose", "-p", spec.resource, "-f", str(spec.work_path / COMPOSE_FILE), *args]

    def build(self, spec: DeploymentSpec) -> str:
        self.write_files(spec)
        self.runner.run(["docker", "build", "-t", spec.image, "-f", str(self.dockerfile_path(spec)), spec.build_context], timeout=1800)
        return spec.image

    def up(self, spec: DeploymentSpec) -> DeploymentRecord:
        record = self.new_record(spec)
        record.resources = {"project": spec.resource, "compose_file": str(spec.work_path / COMPOSE_FILE), "service": SERVICE}
        try:
            self.write_files(spec)
            self.runner.run(self._compose(spec, "up", "-d", "--build", "--wait", "--wait-timeout", str(spec.timeout)), timeout=spec.timeout + 1800)
            return self.finish(record, f"http://127.0.0.1:{spec.host_port}", spec.timeout)
        except Exception as e:  # noqa: BLE001
            self.failed(record, e)
            raise

    def _ps(self, spec: DeploymentSpec) -> list[dict]:
        result = self.runner.run(self._compose(spec, "ps", "--all", "--format", "json"), check=False, timeout=60)
        rows: list[dict] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            rows.extend(data if isinstance(data, list) else [data])
        return rows

    def status(self, spec: DeploymentSpec, record: DeploymentRecord) -> dict:
        rows = self._ps(spec)
        running = [r for r in rows if str(r.get("State", "")).lower() == "running"]
        health = self.health(record.endpoint) if record.endpoint else {"ok": False, "error": "no endpoint"}
        return {"ready": bool(running and health.get("ok")), "endpoint": record.endpoint, "health": health,
                "replicas": {"ready": len(running), "wanted": 1},
                "details": {"containers": [{"name": r.get("Name"), "state": r.get("State"), "health": r.get("Health"), "status": r.get("Status")} for r in rows]}}

    def logs(self, spec: DeploymentSpec, record: DeploymentRecord, lines: int = 100) -> str:
        return self.runner.run(self._compose(spec, "logs", "--no-color", "--tail", str(lines), SERVICE), check=False, timeout=60).stdout

    def down(self, spec: DeploymentSpec, record: DeploymentRecord) -> None:
        self.runner.run(self._compose(spec, "down", "--remove-orphans"), check=False, timeout=600)
