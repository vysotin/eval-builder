"""Deployment targets: run the target agent (with both mock layers) behind an HTTP
endpoint — locally, in Docker Compose, on Kubernetes, or on OpenShift (prototype).

`spec.py` says what to deploy and records what was deployed (`deployment.json`),
`render.py` generates the Dockerfile / compose file / manifests, `runner.py` runs and
logs every external command, `base.py` is the target interface, and one module per
target implements it. `deploy_up`, `deploy_status`, `deploy_down`, `deploy_build`
and `deploy_render` are the entry points the CLI, the pipeline stages and the UI use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from evalbuilder.deploy.base import DeploymentError, DeploymentTarget
from evalbuilder.deploy.docker_compose import DockerComposeTarget
from evalbuilder.deploy.kubernetes import KubernetesTarget
from evalbuilder.deploy.local import LocalTarget
from evalbuilder.deploy.openshift import OpenShiftTarget
from evalbuilder.deploy.runner import CommandError, CommandRunner
from evalbuilder.deploy.spec import DEPLOYMENT_SCHEMA, WORK_SUBDIR, DeploymentRecord, DeploymentSpec, spec_from_config

TARGETS: dict[str, type[DeploymentTarget]] = {
    "local": LocalTarget,
    "docker": DockerComposeTarget,
    "kubernetes": KubernetesTarget,
    "openshift": OpenShiftTarget,
}

COMMANDS_LOG = "commands.log"


def target_for(kind: str, runner: CommandRunner | None = None, log: Callable[[str], None] | None = None, **kwargs) -> DeploymentTarget:
    if kind not in TARGETS:
        raise ValueError(f"unknown deployment target {kind!r}; use one of {', '.join(TARGETS)}")
    return TARGETS[kind](runner=runner, log=log, **kwargs)


def record_path(out_dir: Path) -> Path:
    from evalbuilder.pipeline.layout import path_for

    return path_for(Path(out_dir), "deployment")


def load_record(out_dir: Path) -> DeploymentRecord | None:
    from evalbuilder.pipeline.layout import existing_path

    path = existing_path(Path(out_dir), "deployment")
    if path is None:
        return None
    try:
        return DeploymentRecord.load(path)
    except (ValueError, TypeError):
        return None


def default_runner(spec: DeploymentSpec, log: Callable[[str], None] | None = None) -> CommandRunner:
    return CommandRunner(log_path=spec.work_path / COMMANDS_LOG, log=log or (lambda msg: None))


def _prepare(cfg, out_dir, target, runner, log):
    spec, missing = spec_from_config(cfg, out_dir, target=target)
    runner = runner or default_runner(spec, log)
    return spec, missing, target_for(spec.target, runner=runner, log=log)


def deploy_render(cfg, out_dir: Path | None = None, *, target: str | None = None, write: bool = False) -> dict[str, str]:
    """The generated files for the config's target (written to `work/deploy/` when `write`)."""
    spec, _, tgt = _prepare(cfg, out_dir, target, None, None)
    files = tgt.render(spec)
    if write:
        tgt.write_files(spec)
    return files


def deploy_build(cfg, out_dir: Path | None = None, *, target: str | None = None, runner=None, log=None) -> str:
    spec, _, tgt = _prepare(cfg, out_dir, target, runner, log)
    return tgt.build(spec)


def deploy_up(cfg, out_dir: Path | None = None, *, target: str | None = None, runner=None, log=None) -> DeploymentRecord:
    """Deploy per the config, wait for `/health`, write `deployment.json`; raises
    DeploymentError / CommandError (the failed record is written first)."""
    spec, missing, tgt = _prepare(cfg, out_dir, target, runner, log)
    ok, reason = tgt.available()
    if not ok:
        raise DeploymentError(f"deployment target {spec.target!r} unavailable: {reason}")
    out = Path(out_dir) if out_dir is not None else cfg.output_dir
    path = record_path(out)
    existing = load_record(out)
    if existing is not None and existing.status == "up" and existing.target == spec.target:
        status = tgt.status(spec, existing)
        if status.get("ready"):
            existing.details["reused"] = True
            existing.save(path)
            return existing  # already running and healthy: reuse
        tgt.down(spec, existing)
    try:
        record = tgt.up(spec)
    except Exception as e:  # noqa: BLE001 - the failed record is the diagnosis
        failed = tgt.new_record(spec)
        tgt.failed(failed, e)
        failed.save(path)
        raise
    record.details["missing_host_env"] = missing
    record.save(path)
    return record


def _spec_and_record(cfg_or_dir, out_dir, runner, log):
    from evalbuilder.pipeline.config import PipelineConfig

    if isinstance(cfg_or_dir, PipelineConfig):
        cfg = cfg_or_dir
        out = Path(out_dir) if out_dir is not None else cfg.output_dir
        record = load_record(out)
        if record is None:
            return None, None, None, out
        spec = DeploymentSpec(**record.spec) if record.spec else spec_from_config(cfg, out)[0]
    else:
        out = Path(cfg_or_dir)
        record = load_record(out)
        if record is None or not record.spec:
            return None, record, None, out
        spec = DeploymentSpec(**record.spec)
    runner = runner or default_runner(spec, log)
    return spec, record, target_for(record.target, runner=runner, log=log), out


def deploy_status(cfg_or_dir, out_dir: Path | None = None, *, runner=None, log=None) -> dict:
    """Live status of the recorded deployment (`{status: none}` when there is none)."""
    spec, record, tgt, out = _spec_and_record(cfg_or_dir, out_dir, runner, log)
    if record is None:
        return {"status": "none", "record": None, "ready": False}
    if tgt is None:
        return {"status": record.status, "record": record.to_dict(), "ready": False}
    if record.status != "up":
        return {"status": record.status, "record": record.to_dict(), "ready": False}
    live = tgt.status(spec, record)
    record.details["last_status"] = {k: v for k, v in live.items() if k != "details"}
    record.save(record_path(out))
    return {"status": "up" if live.get("ready") else "unhealthy", "record": record.to_dict(), **live}


def deploy_logs(cfg_or_dir, out_dir: Path | None = None, *, lines: int = 100, runner=None, log=None) -> str:
    spec, record, tgt, _ = _spec_and_record(cfg_or_dir, out_dir, runner, log)
    if record is None or tgt is None:
        return ""
    return tgt.logs(spec, record, lines=lines)


def deploy_down(cfg_or_dir, out_dir: Path | None = None, *, runner=None, log=None) -> dict:
    """Tear the recorded deployment down and mark the record `down`."""
    spec, record, tgt, out = _spec_and_record(cfg_or_dir, out_dir, runner, log)
    if record is None:
        return {"status": "none", "record": None}
    if tgt is not None and record.status in ("up", "unhealthy", "failed"):
        tgt.down(spec, record)
        record.commands = list(tgt.runner.history)
    record.status = "down"
    record.endpoint = None
    record.save(record_path(out))
    return {"status": "down", "record": record.to_dict()}


__all__ = [
    "TARGETS", "DEPLOYMENT_SCHEMA", "WORK_SUBDIR", "DeploymentError", "CommandError", "CommandRunner", "DeploymentRecord",
    "DeploymentSpec", "DeploymentTarget", "spec_from_config", "target_for", "record_path", "load_record", "default_runner",
    "deploy_render", "deploy_build", "deploy_up", "deploy_status", "deploy_logs", "deploy_down",
]
