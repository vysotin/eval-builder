"""Real deployments on the local Docker daemon and the Kubernetes cluster kubectl reaches
(Docker Desktop): build the weather bot's image, deploy it, run cases against it, tear it
down. Marker `docker`; skipped when the daemon or the cluster is unavailable.

    uv run pytest -m docker tests/test_deploy_integration.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from evalbuilder.cli import app
from evalbuilder.pipeline.config import load_config

pytestmark = pytest.mark.docker

EXAMPLE = Path("examples/weather_bot/pipeline.yaml")


def _ready(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


DOCKER = shutil.which("docker") is not None and _ready(["docker", "info"])
KUBE = DOCKER and shutil.which("kubectl") is not None and _ready(["kubectl", "get", "nodes"])


def _config(tmp_path: Path, target: str) -> tuple[Path, Path]:
    data = yaml.safe_load(EXAMPLE.read_text())
    data["deploy"]["target"] = target
    data["output"] = {"dir": str(tmp_path / "out")}
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))
    return cfg, tmp_path / "out"


def _ok(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


@pytest.mark.skipif(not DOCKER, reason="docker daemon not available")
def test_docker_compose_pipeline_end_to_end(tmp_path):
    cfg, out = _config(tmp_path, "docker")
    runner = CliRunner()
    result = runner.invoke(app, ["pipeline", "run", str(cfg), "--quiet"])
    assert result.exit_code == 0, result.output
    report = json.loads((out / "report.json").read_text())
    assert report["verdict"] == "pass"
    assert {k: v["status"] for k, v in report["stages"].items() if k in ("deploy", "infer", "simulate", "teardown")} == {
        "deploy": "ok", "infer": "ok", "simulate": "ok", "teardown": "ok"}
    assert report["deployment"]["target"] == "docker" and report["deployment"]["status"] == "down"
    assert report["stages"]["deploy"]["details"]["endpoint"] == "http://127.0.0.1:8080"
    assert report["stages"]["infer"]["details"]["execution"]["mode"] == "remote"
    run = json.loads(Path(report["runs"][0]["run"]).read_text())
    assert run["execution"]["endpoint"] == "http://127.0.0.1:8080" and all(cr["error"] is None for cr in run["case_runs"])
    assert (out / "work" / "deploy" / "Dockerfile").exists() and (out / "work" / "deploy" / "compose.yaml").exists()
    ps = subprocess.run(["docker", "compose", "-p", "evalbuilder-weather-bot", "ps", "-q"], capture_output=True, text=True)
    assert ps.stdout.strip() == ""  # torn down


@pytest.mark.skipif(not KUBE, reason="kubectl cluster not reachable")
def test_kubernetes_deploy_infer_down(tmp_path):
    cfg, out = _config(tmp_path, "kubernetes")
    runner = CliRunner()
    # a dataset to run: generate it with the local target first (no LLM needed)
    local_cfg, local_out = _config(tmp_path / "local", "local")
    assert runner.invoke(app, ["pipeline", "run", str(local_cfg), "--until", "verify", "--quiet"]).exit_code == 0
    up = _ok(runner.invoke(app, ["deploy", "up", str(cfg), "--quiet"]))
    try:
        assert up["status"] == "up" and up["endpoint"].startswith("http://127.0.0.1:") and up["target"] == "kubernetes"
        assert up["resources"]["loader"].endswith("image-loader") and up["resources"]["port_forward"]["pid"]
        status = _ok(runner.invoke(app, ["deploy", "status", str(out)]))
        assert status["ready"] and status["replicas"] == {"ready": 1, "wanted": 1}
        summary = _ok(runner.invoke(app, ["infer", str(local_out / "dataset.json"), "--deployment", str(out), "--out", str(out / "results"), "--workers", "4"]))
        assert summary["mode"] == "remote" and summary["errors"] == 0 and summary["mocking"]["calls"]["llm"] > 0
    finally:
        down = _ok(runner.invoke(app, ["deploy", "down", str(out)]))
    assert down["status"] == "down"
    left = subprocess.run(["kubectl", "get", "deployment,daemonset", "-o", "name"], capture_output=True, text=True).stdout
    assert "evalbuilder-weather-bot" not in left
    assert load_config(cfg).deploy.target == "kubernetes"
