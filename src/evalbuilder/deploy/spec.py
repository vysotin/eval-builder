"""What to deploy (`DeploymentSpec`) and what was deployed (`DeploymentRecord`,
the `deployment.json` artifact, schema `evalbuilder/deployment/v1`)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEPLOYMENT_SCHEMA = "evalbuilder/deployment/v1"
WORK_SUBDIR = "deploy"  # rendered files, command log, server / port-forward logs under <out_dir>/work/deploy/
PYTHON_VERSION = "3.12"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resource_name(name: str) -> str:
    """A DNS-1123 label for compose projects, k8s objects and image names."""
    cleaned = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in name.lower()).strip("-")
    return (cleaned or "agent")[:50]


def resolve_env(env: dict[str, str | None] | None) -> tuple[dict[str, str], list[str]]:
    """Container environment: a `None` value copies the variable from the host.
    Returns (resolved, names that were requested but unset on the host)."""
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for key, value in (env or {}).items():
        if value is None:
            host = os.environ.get(key)
            if host is None or host == "":
                missing.append(key)
            else:
                resolved[key] = host
        else:
            resolved[key] = str(value)
    return resolved, missing


@dataclass
class DeploymentSpec:
    name: str  # pipeline name → resource names
    target: str  # local | docker | kubernetes | openshift
    module: str  # target module exposing TOOLS + build_agent
    factory: str = "build_agent"
    image: str = ""  # [registry/]name:tag
    push: str = "auto"  # auto | registry | load | none
    registry: str | None = None
    extras: list[str] = field(default_factory=list)
    build_context: str = "."
    dockerfile: str | None = None
    include: list[str] = field(default_factory=list)
    requirements: str | None = None
    port: int = 8080
    namespace: str = "default"
    replicas: int = 1
    expose: str = "port-forward"  # port-forward | nodeport | route
    env: dict[str, str] = field(default_factory=dict)  # resolved container environment
    keep: bool = False
    timeout: int = 240
    context: str | None = None  # kubectl / oc context
    agent_model: str | None = None  # spec injected into the server (EVALBUILDER_AGENT_MODEL)
    mock_model: str | None = None  # spec for the LLM mock engine (EVALBUILDER_MOCK_MODEL)
    work_dir: str = ""  # where rendered files and logs go (<out_dir>/work/deploy)
    python_version: str = PYTHON_VERSION
    project_root: str = "."  # cwd for the local target and the build

    @property
    def resource(self) -> str:
        return f"evalbuilder-{resource_name(self.name)}"

    @property
    def work_path(self) -> Path:
        return Path(self.work_dir)

    def to_dict(self) -> dict:
        return asdict(self)


def spec_from_config(cfg, out_dir: Path | None = None, *, target: str | None = None) -> tuple[DeploymentSpec, list[str]]:
    """Build the spec from a `PipelineConfig`; returns (spec, host variables requested
    but unset). `target` overrides `deploy.target` (CLI `--target`)."""
    from evalbuilder.pipeline.layout import WORK_DIR

    d = cfg.deploy
    env, missing = resolve_env(d.env)
    out = Path(out_dir) if out_dir is not None else cfg.output_dir
    mock_model = cfg.mock_model_spec if cfg.llm_mocking else None
    spec = DeploymentSpec(
        name=cfg.name, target=target or d.target, module=cfg.target.module, factory=cfg.target.factory,
        image=cfg.image_ref, push=d.image.push, registry=d.image.registry, extras=list(d.image.extras),
        build_context=d.build.context, dockerfile=d.build.dockerfile, include=list(d.build.include),
        requirements=d.build.requirements, port=d.port, namespace=d.namespace, replicas=d.replicas,
        expose=d.expose, env=env, keep=d.keep, timeout=d.timeout, context=d.context,
        agent_model=cfg.models.agent, mock_model=mock_model, work_dir=str(out / WORK_DIR / WORK_SUBDIR),
    )
    if spec.target == "openshift" and spec.push == "auto":
        spec.push = "registry"
    if spec.target == "openshift" and spec.expose == "port-forward" and d.expose == "port-forward":
        spec.expose = "route"
    return spec, missing


@dataclass
class DeploymentRecord:
    """The `deployment.json` artifact."""

    name: str
    target: str
    image: str = ""
    endpoint: str | None = None
    expose: str = ""
    status: str = "pending"  # pending | up | failed | down
    resources: dict[str, Any] = field(default_factory=dict)  # per target: project / namespace / names / pids
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    commands: list[dict] = field(default_factory=list)  # {argv, returncode, seconds} of every external command
    details: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    schema: str = DEPLOYMENT_SCHEMA

    def touch(self) -> None:
        self.updated_at = _now()

    def to_dict(self) -> dict:
        data = asdict(self)
        return {"schema": data.pop("schema"), **data}

    @classmethod
    def from_dict(cls, data: dict) -> "DeploymentRecord":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.touch()
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "DeploymentRecord":
        return cls.from_dict(json.loads(Path(path).read_text()))
