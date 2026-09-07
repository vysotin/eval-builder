"""Deployment targets: the spec, the rendered files, every target against a fake
command runner (what it runs, in which order, what it records), the local target for
real, and the deploy_up / deploy_status / deploy_down entry points."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import yaml

from evalbuilder.deploy import (
    DeploymentError,
    deploy_down,
    deploy_render,
    deploy_status,
    deploy_up,
    load_record,
    record_path,
    target_for,
)
from evalbuilder.deploy.docker_compose import DockerComposeTarget
from evalbuilder.deploy.kubernetes import KubernetesTarget
from evalbuilder.deploy.local import LocalTarget
from evalbuilder.deploy.openshift import OpenShiftTarget
from evalbuilder.deploy.render import package_dir, render_compose, render_dockerfile, render_loader, render_manifests
from evalbuilder.deploy.runner import CommandError, CommandResult, CommandRunner
from evalbuilder.deploy.spec import DEPLOYMENT_SCHEMA, DeploymentRecord, resolve_env, spec_from_config
from evalbuilder.pipeline.config import PipelineConfig

WEATHER = {"source": "examples/weather_bot/agent.py", "module": "examples.weather_bot.agent"}
SCRIPTED = "scripted:examples.weather_bot.agent:default_scripted_model"


def _cfg(tmp_path, **deploy) -> PipelineConfig:
    return PipelineConfig.model_validate({
        "schema": "evalbuilder/pipeline-config/v1", "name": "weather-small", "target": WEATHER,
        "models": {"agent": SCRIPTED, "mock": "scripted:examples.weather_bot.offline:mock_model"},
        "mocking": {"on_miss": "llm"}, "deploy": deploy, "output": {"dir": str(tmp_path / "out")},
    })


class FakeRunner:
    """Records every command; answers with canned stdout by substring; `FAIL` fails."""

    def __init__(self, outputs: dict | None = None, missing: tuple = ()):
        self.calls: list[list[str]] = []
        self.spawned: list[dict] = []
        self.pipes: list[tuple[list[str], list[str]]] = []
        self.killed: list = []
        self.history: list[dict] = []
        self.outputs = outputs or {}
        self.missing = set(missing)
        self.alive: set[int] = set()

    def which(self, binary):
        return None if binary in self.missing else f"/usr/bin/{binary}"

    def _stdout(self, argv):
        joined = " ".join(argv)
        for key, value in self.outputs.items():
            if key in joined:
                return value
        return ""

    def run(self, argv, *, input=None, check=True, timeout=None, env=None, cwd=None):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        out = self._stdout(argv)
        result = CommandResult(argv, 1 if out == "FAIL" else 0, "" if out == "FAIL" else out, "boom" if out == "FAIL" else "", 0.01)
        self.history.append(result.summary())
        if check and not result.ok:
            raise CommandError(result)
        return result

    def pipe(self, producer, consumer, *, check=True, timeout=None, env=None):
        pair = ([str(a) for a in producer], [str(a) for a in consumer])
        self.pipes.append(pair)
        result = CommandResult([*pair[0], "|", *pair[1]], 0, "imported", "", 0.01)
        self.history.append(result.summary())
        return result

    def spawn(self, argv, *, log_path, env=None, cwd=None):
        self.spawned.append({"argv": [str(a) for a in argv], "log": str(log_path), "env": env, "cwd": cwd})
        pid = 1000 + len(self.spawned)
        self.alive.add(pid)
        self.history.append({"argv": [str(a) for a in argv], "returncode": None, "seconds": 0.0, "output": f"spawned {pid}"})
        return pid

    def pid_alive(self, pid):
        return pid in self.alive

    def kill(self, pid, timeout=5.0):
        self.killed.append(pid)
        self.alive.discard(pid)


HEALTHY = {"ok": True, "module": "examples.weather_bot.agent", "factory": "build_agent", "tools": ["get_weather", "get_alerts"], "server": "evalbuilder/serve/v1"}


def _target(cls, runner, health=None):
    return cls(runner=runner, health=health or (lambda endpoint: dict(HEALTHY)), sleep=lambda s: None)


def _calls(runner, *needles):
    return [c for c in runner.calls if all(n in " ".join(c) for n in needles)]


# ── spec ───────────────────────────────────────────────────────


def test_spec_from_config_and_env_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("ACME_KEY", "secret")
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cfg = _cfg(tmp_path, target="kubernetes", image={"registry": "quay.io/team", "extras": ["llm"]}, env={"ACME_KEY": None, "MODE": "eval", "MISSING_KEY": None},
               namespace="evals", replicas=2, expose="nodeport", keep=True, timeout=99, context="ctx")
    spec, missing = spec_from_config(cfg)
    assert spec.target == "kubernetes" and spec.image == "quay.io/team/evalbuilder-weather-small:latest" and spec.registry == "quay.io/team"
    assert spec.resource == "evalbuilder-weather-small" and spec.module == WEATHER["module"] and spec.factory == "build_agent"
    assert spec.env == {"ACME_KEY": "secret", "MODE": "eval"} and missing == ["MISSING_KEY"]
    assert (spec.namespace, spec.replicas, spec.expose, spec.keep, spec.timeout, spec.context, spec.extras) == ("evals", 2, "nodeport", True, 99, "ctx", ["llm"])
    assert spec.agent_model == SCRIPTED and spec.mock_model == "scripted:examples.weather_bot.offline:mock_model"
    assert spec.work_dir == str(tmp_path / "out" / "work" / "deploy")
    assert resolve_env(None) == ({}, [])
    local, _ = spec_from_config(_cfg(tmp_path), out_dir=tmp_path / "elsewhere", target="docker")
    assert local.target == "docker" and local.work_dir == str(tmp_path / "elsewhere" / "work" / "deploy") and local.image == "evalbuilder-weather-small:latest"
    ocp, _ = spec_from_config(_cfg(tmp_path, target="openshift", image={"registry": "reg.example/ns"}))
    assert ocp.push == "registry" and ocp.expose == "route"
    nomock, _ = spec_from_config(PipelineConfig.model_validate({"schema": "evalbuilder/pipeline-config/v1", "name": "n", "target": WEATHER}))
    assert nomock.mock_model is None and nomock.agent_model is None


def test_record_round_trip(tmp_path):
    record = DeploymentRecord(name="n", target="docker", image="i:1", endpoint="http://x", status="up", resources={"project": "p"})
    path = record.save(tmp_path / "deployment.json")
    data = json.loads(path.read_text())
    assert list(data)[0] == "schema" and data["schema"] == DEPLOYMENT_SCHEMA and data["updated_at"]
    assert DeploymentRecord.load(path) == DeploymentRecord.from_dict(data)


# ── rendered files ─────────────────────────────────────────────


def test_dockerfile_compose_manifests_and_loader(tmp_path):
    cfg = _cfg(tmp_path, target="kubernetes", image={"extras": ["llm", "ui"]}, build={"include": ["data"], "requirements": "reqs.txt"}, env={"MODE": "eval"})
    spec, _ = spec_from_config(cfg)
    dockerfile = render_dockerfile(spec)
    assert "FROM python:3.12-slim" in dockerfile and 'pip install --no-cache-dir ".[llm,ui]"' in dockerfile
    assert "COPY src ./src" in dockerfile and "COPY examples ./examples" in dockerfile and "COPY data ./data" in dockerfile
    assert "COPY reqs.txt ./requirements-target.txt" in dockerfile and "EXPOSE 8080" in dockerfile
    assert '"evalbuilder", "serve", "--module", "examples.weather_bot.agent", "--factory", "build_agent", "--host", "0.0.0.0", "--port", "8080"' in dockerfile
    assert package_dir("examples.weather_bot.agent") == "examples" and package_dir("agent") == "agent.py"

    compose = yaml.safe_load(render_compose(spec, tmp_path / "Dockerfile"))
    svc = compose["services"]["agent"]
    assert compose["name"] == "evalbuilder-weather-small" and svc["ports"] == ["8080:8080"] and svc["image"] == spec.image
    assert svc["environment"] == {"MODE": "eval", "EVALBUILDER_AGENT_MODEL": SCRIPTED, "EVALBUILDER_MOCK_MODEL": "scripted:examples.weather_bot.offline:mock_model"}
    assert svc["build"]["dockerfile"].endswith("Dockerfile") and svc["healthcheck"]["test"][0] == "CMD"

    docs = list(yaml.safe_load_all(render_manifests(spec, "Never")))
    deployment, service = docs
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert deployment["kind"] == "Deployment" and deployment["metadata"]["namespace"] == "default" and deployment["spec"]["replicas"] == 1
    assert container["image"] == spec.image and container["imagePullPolicy"] == "Never" and container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert {e["name"]: e["value"] for e in container["env"]}["EVALBUILDER_AGENT_MODEL"] == SCRIPTED
    assert service["kind"] == "Service" and service["spec"]["type"] == "ClusterIP" and service["spec"]["ports"][0]["port"] == 8080
    spec.expose = "nodeport"
    assert list(yaml.safe_load_all(render_manifests(spec, "Always")))[1]["spec"]["type"] == "NodePort"
    spec.expose = "route"
    route = list(yaml.safe_load_all(render_manifests(spec, "Always")))[2]
    assert route["kind"] == "Route" and route["spec"]["to"]["name"] == "evalbuilder-weather-small"
    loader = yaml.safe_load(render_loader(spec))
    assert loader["kind"] == "DaemonSet" and loader["metadata"]["name"] == "evalbuilder-weather-small-image-loader"
    assert loader["spec"]["template"]["spec"]["hostPID"] is True and loader["spec"]["template"]["spec"]["containers"][0]["securityContext"]["privileged"] is True


def test_custom_dockerfile_is_used_verbatim(tmp_path):
    own = tmp_path / "My.Dockerfile"
    own.write_text("FROM scratch\n")
    spec, _ = spec_from_config(_cfg(tmp_path, target="docker", build={"dockerfile": str(own)}))
    files = DockerComposeTarget(runner=FakeRunner()).render(spec)
    assert set(files) == {"compose.yaml"} and str(own) in files["compose.yaml"]


# ── docker compose target ──────────────────────────────────────


def test_docker_target_availability(tmp_path):
    assert _target(DockerComposeTarget, FakeRunner(missing=("docker",))).available() == (False, "docker is not on PATH")
    assert not _target(DockerComposeTarget, FakeRunner({"docker info": "FAIL"})).available()[0]
    assert not _target(DockerComposeTarget, FakeRunner({"compose version": "FAIL", "docker info": "29.4"})).available()[0]
    ok, reason = _target(DockerComposeTarget, FakeRunner({"docker info": "29.4", "compose version": "5.1"})).available()
    assert ok and "29.4" in reason


def test_docker_target_up_status_logs_down(tmp_path):
    runner = FakeRunner({"ps --all": json.dumps({"Name": "evalbuilder-weather-small-agent-1", "State": "running", "Health": "healthy", "Status": "Up"}),
                         "logs --no-color": "serving"})
    spec, _ = spec_from_config(_cfg(tmp_path, target="docker"))
    tgt = _target(DockerComposeTarget, runner)
    record = tgt.up(spec)
    up = _calls(runner, "docker compose", "up")[0]
    assert up[:6] == ["docker", "compose", "-p", "evalbuilder-weather-small", "-f", str(Path(spec.work_dir) / "compose.yaml")]
    assert up[6:] == ["up", "-d", "--build", "--wait", "--wait-timeout", "240"]
    assert record.status == "up" and record.endpoint == "http://127.0.0.1:8080" and record.target == "docker"
    assert record.resources["project"] == "evalbuilder-weather-small" and record.details["health"]["tools"] == ["get_weather", "get_alerts"]
    assert (Path(spec.work_dir) / "Dockerfile").exists() and (Path(spec.work_dir) / "compose.yaml").exists()
    assert any(c["argv"][:2] == ["docker", "compose"] for c in record.commands)
    status = tgt.status(spec, record)
    assert status["ready"] and status["replicas"] == {"ready": 1, "wanted": 1} and status["details"]["containers"][0]["health"] == "healthy"
    assert tgt.logs(spec, record) == "serving"
    tgt.down(spec, record)
    assert _calls(runner, "compose", "down")[0][-2:] == ["down", "--remove-orphans"]


def test_docker_target_failure_is_recorded(tmp_path):
    runner = FakeRunner({"compose -p evalbuilder-weather-small -f": "FAIL"})
    spec, _ = spec_from_config(_cfg(tmp_path, target="docker"))
    tgt = _target(DockerComposeTarget, runner)
    with pytest.raises(CommandError, match="exited 1"):
        tgt.up(spec)


# ── kubernetes target ──────────────────────────────────────────


def test_kubernetes_load_path_end_to_end(tmp_path):
    runner = FakeRunner({"get pods -l app=evalbuilder-weather-small-image-loader": "loader-a loader-b", "get deployment": json.dumps({"spec": {"replicas": 1}, "status": {"readyReplicas": 1, "availableReplicas": 1}})})
    spec, _ = spec_from_config(_cfg(tmp_path, target="kubernetes", namespace="evals", context="docker-desktop"))
    tgt = _target(KubernetesTarget, runner)
    assert tgt.push_mode(spec) == "load"
    record = tgt.up(spec)
    work = Path(spec.work_dir)
    joined = [" ".join(c) for c in runner.calls]
    assert joined[0] == f"docker build -t evalbuilder-weather-small:latest -f {work / 'Dockerfile'} ."
    assert joined[1] == f"kubectl --context docker-desktop -n evals apply -f {work / 'loader.yaml'}"
    assert joined[2] == "kubectl --context docker-desktop -n evals rollout status daemonset/evalbuilder-weather-small-image-loader --timeout=240s"
    assert [p[0] for p in runner.pipes] == [["docker", "save", "evalbuilder-weather-small:latest"]] * 2
    assert runner.pipes[0][1] == ["kubectl", "--context", "docker-desktop", "-n", "evals", "exec", "-i", "loader-a", "--", "nsenter", "-t", "1", "-m", "-u", "-i", "-n", "--", "ctr", "-n", "k8s.io", "images", "import", "-"]
    assert runner.pipes[1][1][7] == "loader-b"
    assert f"apply -f {work / 'manifests.yaml'}" in joined[4] and "rollout status deployment/evalbuilder-weather-small --timeout=240s" in joined[5]
    pf = runner.spawned[0]
    assert pf["argv"][:6] == ["kubectl", "--context", "docker-desktop", "-n", "evals", "port-forward"] and pf["argv"][6] == "svc/evalbuilder-weather-small"
    local_port = int(pf["argv"][7].split(":")[0])
    assert pf["argv"][7] == f"{local_port}:8080" and record.endpoint == f"http://127.0.0.1:{local_port}"
    assert record.status == "up" and record.resources["port_forward"]["pid"] == 1001 and record.resources["loader"].endswith("image-loader")
    assert record.details["push"] == "load"
    manifests = list(yaml.safe_load_all((work / "manifests.yaml").read_text()))
    assert manifests[0]["spec"]["template"]["spec"]["containers"][0]["imagePullPolicy"] == "Never"
    status = tgt.status(spec, record)
    assert status["ready"] and status["replicas"] == {"ready": 1, "wanted": 1} and status["details"]["port_forward"]["alive"] is True
    # a dead port-forward is restarted by status()
    runner.alive.clear()
    status = tgt.status(spec, record)
    assert status["details"]["restarted_port_forward"] is True and record.resources["port_forward"]["pid"] == 1002 and status["ready"]
    tgt.down(spec, record)
    assert runner.killed[-1] == 1002
    deletes = [" ".join(c) for c in _calls(runner, "delete")]
    assert any(f"delete -f {work / 'manifests.yaml'}" in d for d in deletes) and any("daemonset/evalbuilder-weather-small-image-loader" in d for d in deletes)


def test_kubernetes_registry_push_and_nodeport(tmp_path):
    runner = FakeRunner({"jsonpath={.spec.ports[0].nodePort}": "30123", "jsonpath={.items[0].status.addresses": "172.19.0.3"})
    spec, _ = spec_from_config(_cfg(tmp_path, target="kubernetes", image={"registry": "quay.io/team"}, expose="nodeport"))
    tgt = _target(KubernetesTarget, runner)
    assert tgt.push_mode(spec) == "registry" and tgt.render(spec).keys() == {"Dockerfile", "manifests.yaml"}
    record = tgt.up(spec)
    joined = [" ".join(c) for c in runner.calls]
    assert joined[1] == "docker push quay.io/team/evalbuilder-weather-small:latest" and not runner.pipes and not runner.spawned
    assert record.endpoint == "http://172.19.0.3:30123" and record.details["push"] == "registry"
    manifests = list(yaml.safe_load_all((Path(spec.work_dir) / "manifests.yaml").read_text()))
    assert manifests[0]["spec"]["template"]["spec"]["containers"][0]["imagePullPolicy"] == "Always" and manifests[1]["spec"]["type"] == "NodePort"
    none_spec, _ = spec_from_config(_cfg(tmp_path, target="kubernetes", image={"push": "none"}))
    assert tgt.push_mode(none_spec) == "none"


def test_kubernetes_availability_and_failed_rollout(tmp_path):
    assert _target(KubernetesTarget, FakeRunner(missing=("kubectl",))).available() == (False, "kubectl is not on PATH")
    assert not _target(KubernetesTarget, FakeRunner({"get nodes": "FAIL"})).available()[0]
    ok, reason = _target(KubernetesTarget, FakeRunner({"get nodes -o name": "node/a\nnode/b"})).available()
    assert ok and "2 node(s)" in reason
    runner = FakeRunner({"get pods -l": "loader-a", "rollout status deployment": "FAIL"})
    spec, _ = spec_from_config(_cfg(tmp_path, target="kubernetes"))
    tgt = _target(KubernetesTarget, runner)
    with pytest.raises(CommandError, match="rollout status deployment"):
        tgt.up(spec)
    runner = FakeRunner({"get pods -l": ""})
    with pytest.raises(DeploymentError, match="loader pods"):
        _target(KubernetesTarget, runner).up(spec)


def test_wait_healthy_times_out(tmp_path):
    spec, _ = spec_from_config(_cfg(tmp_path, target="docker"))
    tgt = DockerComposeTarget(runner=FakeRunner(), health=lambda e: {"ok": False, "error": "refused"}, sleep=lambda s: None)
    with pytest.raises(DeploymentError, match="did not become healthy"):
        tgt.wait_healthy("http://127.0.0.1:1", timeout=0)


# ── openshift prototype ────────────────────────────────────────


def test_openshift_uses_oc_registry_push_and_a_route(tmp_path):
    runner = FakeRunner({"oc whoami": "dev", "jsonpath={.spec.host}": "agent-evals.apps.example", "jsonpath={.spec.tls.termination}": "edge",
                         "get deployment": json.dumps({"spec": {"replicas": 1}, "status": {"readyReplicas": 1}})})
    spec, _ = spec_from_config(_cfg(tmp_path, target="openshift", image={"registry": "image-registry.openshift-image-registry.svc:5000/evals"}, namespace="evals"))
    tgt = _target(OpenShiftTarget, runner)
    assert tgt.available() == (True, "oc logged in as dev") and tgt.cli == "oc" and tgt.push_mode(spec) == "registry"
    record = tgt.up(spec)
    joined = [" ".join(c) for c in runner.calls]
    build = next(i for i, c in enumerate(joined) if c.startswith("docker build"))
    assert joined[build + 1] == "docker push image-registry.openshift-image-registry.svc:5000/evals/evalbuilder-weather-small:latest"
    assert joined[build + 2].startswith("oc -n evals apply -f") and "oc -n evals rollout status deployment/evalbuilder-weather-small" in joined[build + 3]
    assert record.endpoint == "https://agent-evals.apps.example" and record.resources["route"] == "evalbuilder-weather-small" and not runner.spawned
    docs = list(yaml.safe_load_all((Path(spec.work_dir) / "manifests.yaml").read_text()))
    assert docs[2]["kind"] == "Route"
    assert tgt.status(spec, record)["ready"]
    tgt.down(spec, record)
    assert any(c[0] == "oc" and "delete" in c for c in runner.calls)
    assert _target(OpenShiftTarget, FakeRunner(missing=("oc",))).available()[0] is False
    assert "oc login" in _target(OpenShiftTarget, FakeRunner({"oc whoami": "FAIL"})).available()[1]
    bare, _ = spec_from_config(_cfg(tmp_path, target="openshift"))
    with pytest.raises(DeploymentError, match="image.registry"):
        _target(OpenShiftTarget, FakeRunner({"oc whoami": "dev"})).up(bare)


# ── local target, for real ─────────────────────────────────────


def test_local_target_serves_the_agent_for_real(tmp_path):
    cfg = _cfg(tmp_path)  # target local
    spec, _ = spec_from_config(cfg)
    tgt = LocalTarget(runner=CommandRunner(log_path=Path(spec.work_dir) / "commands.log"))
    record = tgt.up(spec)
    try:
        assert record.status == "up" and record.endpoint.startswith("http://127.0.0.1:") and record.resources["pid"]
        assert record.details["health"]["module"] == WEATHER["module"] and "get_weather" in record.details["health"]["tools"]
        status = tgt.status(spec, record)
        assert status["ready"] and status["details"]["alive"] and status["health"]["agent_model"] == SCRIPTED
        from evalbuilder.agent_client import RemoteAgent

        result = RemoteAgent(record.endpoint).invoke([{"role": "user", "content": "What is the weather in Paris?"}],
                                                     {"tools": {"get_weather": [{"matchArgs": {}, "response": {"temp": 3, "condition": "mocked", "city": "Paris"}}]}})
        assert result.error is None and "mocked" in result.response
        assert "evalbuilder serve" in tgt.logs(spec, record)
    finally:
        tgt.down(spec, record)
    assert not tgt.runner.pid_alive(record.resources["pid"])
    assert not tgt.status(spec, record)["ready"]


# ── entry points ───────────────────────────────────────────────


def test_deploy_up_status_down_with_the_local_target(tmp_path):
    cfg = _cfg(tmp_path)
    record = deploy_up(cfg)
    try:
        path = record_path(cfg.output_dir)
        assert path == cfg.output_dir / "deployment.json" and path.exists()
        assert load_record(cfg.output_dir).endpoint == record.endpoint and record.details["missing_host_env"] == []
        status = deploy_status(cfg)
        assert status["status"] == "up" and status["ready"] and status["record"]["target"] == "local"
        again = deploy_up(cfg)  # already up and healthy → reused, same endpoint
        assert again.endpoint == record.endpoint and again.details.get("reused") is True
        by_dir = deploy_status(cfg.output_dir)
        assert by_dir["ready"] and by_dir["endpoint"] == record.endpoint
        assert (cfg.output_dir / "work" / "deploy" / "commands.log").exists()
    finally:
        result = deploy_down(cfg.output_dir)
    assert result["status"] == "down" and load_record(cfg.output_dir).status == "down" and load_record(cfg.output_dir).endpoint is None
    assert deploy_status(cfg)["status"] == "down" and deploy_status(cfg)["ready"] is False
    assert deploy_down(cfg)["status"] == "down"  # idempotent
    assert deploy_status(tmp_path / "nowhere")["status"] == "none" and deploy_down(tmp_path / "nowhere")["status"] == "none"


def test_deploy_up_records_failures_and_unavailable_targets(tmp_path):
    cfg = _cfg(tmp_path, target="kubernetes")
    runner = FakeRunner({"get pods -l": "loader-a", "rollout status deployment": "FAIL", "get nodes -o name": "node/a"})
    with pytest.raises(CommandError):
        deploy_up(cfg, runner=runner)
    failed = load_record(cfg.output_dir)
    assert failed.status == "failed" and "rollout status deployment" in failed.details["error"] and failed.target == "kubernetes"
    with pytest.raises(DeploymentError, match="unavailable"):
        deploy_up(cfg, runner=FakeRunner(missing=("kubectl",)))
    with pytest.raises(ValueError, match="unknown deployment target"):
        target_for("heroku")


def test_deploy_render_returns_the_generated_files(tmp_path):
    cfg = _cfg(tmp_path, target="kubernetes")
    files = deploy_render(cfg)
    assert set(files) == {"Dockerfile", "manifests.yaml", "loader.yaml"} and not (cfg.output_dir / "work").exists()
    deploy_render(cfg, write=True, target="docker")
    assert (cfg.output_dir / "work" / "deploy" / "compose.yaml").exists()
    assert deploy_render(_cfg(tmp_path)) == {}


# ── the real command runner ────────────────────────────────────


def _runner(tmp_path, sink: list | None = None) -> CommandRunner:
    return CommandRunner(log_path=tmp_path / "commands.log", log=(sink if sink is not None else []).append)


def _forwarded(sink: list) -> list[str]:
    return [line for _, line in sink if line.startswith("  | ")]


def test_runner_streams_output_while_the_command_runs(tmp_path):
    stamped: list[tuple[float, str]] = []
    runner = CommandRunner(log_path=tmp_path / "commands.log", log=lambda msg: stamped.append((time.monotonic(), msg)))
    result = runner.run([sys.executable, "-c", "import time\nfor i in (1, 2, 3):\n    print(f'line {i}', flush=True)\n    time.sleep(0.15)\n"])
    forwarded = _forwarded(stamped)
    assert result.ok and result.returncode == 0
    assert forwarded == ["  | line 1", "  | line 2", "  | line 3"]
    assert result.stdout.splitlines() == ["line 1", "line 2", "line 3"] and result.stderr == ""
    # the lines arrived as they were printed, not in one dump after the exit
    at = [t for t, line in stamped if line.startswith("  | ")]
    assert at[-1] - at[0] >= 0.2
    assert stamped[-1][1].startswith("$ ") and "→ 0" in stamped[-1][1]  # the summary line comes last
    assert "line 3" in (tmp_path / "commands.log").read_text()


def test_runner_keeps_a_bounded_output_tail(tmp_path):
    sink: list[tuple[float, str]] = []
    runner = CommandRunner(log_path=tmp_path / "commands.log", log=lambda msg: sink.append((0.0, msg)))
    result = runner.run([sys.executable, "-c", "for i in range(1, 501): print(f'line {i}')"])
    lines = result.stdout.splitlines()
    assert result.ok and len(_forwarded(sink)) == 500  # every line is streamed …
    assert len(lines) == 200 and lines[-1] == "line 500" and lines[0] == "line 301"  # … only the tail is kept
    assert "line 1" not in lines and "line 300" not in lines
    assert len(result.summary()["output"]) <= 400 and len(runner.history[-1]["output"]) <= 400


def test_runner_non_zero_exit_is_returned_or_raised(tmp_path):
    runner = _runner(tmp_path)
    argv = [sys.executable, "-c", "print('working'); print('the last line'); raise SystemExit(3)"]
    result = runner.run(argv, check=False)
    assert result.returncode == 3 and not result.ok and result.stdout.endswith("the last line")
    assert runner.history[-1]["returncode"] == 3
    with pytest.raises(CommandError, match="the last line") as excinfo:
        runner.run(argv)
    assert excinfo.value.result.returncode == 3 and "exited 3" in str(excinfo.value)


def test_runner_writes_input_to_the_child_stdin(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"], input=b"hello runner\n")
    assert result.ok and "HELLO RUNNER" in result.stdout
    echo = runner.run([sys.executable, "-c", "import os; print(os.environ['EB_MARK']); print(os.getcwd())"], env={"EB_MARK": "marked"}, cwd=tmp_path)
    assert echo.stdout.splitlines()[0] == "marked" and Path(echo.stdout.splitlines()[1]).resolve() == tmp_path.resolve()


def test_runner_timeout_kills_the_command(tmp_path):
    runner = _runner(tmp_path)
    sleeper = [sys.executable, "-c", "import time; time.sleep(5)"]
    started = time.monotonic()
    result = runner.run(sleeper, check=False, timeout=0.5)
    elapsed = time.monotonic() - started
    assert result.returncode == 124 and "timed out" in result.stderr and "0.5" in result.stderr
    assert elapsed < 3 and result.seconds < 3 and runner.history[-1]["returncode"] == 124
    with pytest.raises(CommandError, match="exited 124"):
        runner.run(sleeper, timeout=0.5)


def test_runner_missing_binary_is_127(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run(["evalbuilder-no-such-binary-xyz", "--version"], check=False)
    assert result.returncode == 127 and "evalbuilder-no-such-binary-xyz" in result.stderr and result.stdout == ""
    assert runner.which("evalbuilder-no-such-binary-xyz") is None
    with pytest.raises(CommandError, match="exited 127"):
        runner.run(["evalbuilder-no-such-binary-xyz"])
