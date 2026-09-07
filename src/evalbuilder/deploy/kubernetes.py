"""The `kubernetes` target: build the image with Docker, get it into the cluster
(`push` to a registry, `load` it straight into the nodes, or `none`), apply a
Deployment + Service, wait for the rollout, and reach it through `kubectl port-forward`
(the default — it works on every cluster, Docker Desktop's included) or a NodePort.

Verified on Docker Desktop's kind-based cluster, where locally built images are not
visible to the nodes and NodePort / LoadBalancer services are not reachable from the
host: `load` + `port-forward` is the combination that works there.
"""

from __future__ import annotations

from pathlib import Path

from evalbuilder.deploy.base import DeploymentError, DeploymentTarget, free_port
from evalbuilder.deploy.render import loader_name, render_dockerfile, render_loader, render_manifests
from evalbuilder.deploy.spec import DeploymentRecord, DeploymentSpec

MANIFESTS = "manifests.yaml"
LOADER = "loader.yaml"
DOCKERFILE = "Dockerfile"
PORT_FORWARD_LOG = "port-forward.log"
PULL_POLICY = {"registry": "Always", "load": "Never", "none": "IfNotPresent"}
REPLICAS_JSONPATH = 'jsonpath={.spec.replicas}{"|"}{.status.readyReplicas}{"|"}{.status.availableReplicas}'


def _count(field: str) -> int:
    """A jsonpath replica field: absent (empty) or non-numeric counts as 0."""
    field = field.strip()
    return int(field) if field.isdigit() else 0


class KubernetesTarget(DeploymentTarget):
    kind = "kubernetes"
    cli = "kubectl"

    # ── command helpers ──
    def kube(self, spec: DeploymentSpec, *args: str, namespaced: bool = True) -> list[str]:
        argv = [self.cli]
        if spec.context:
            argv += ["--context", spec.context]
        if namespaced:
            argv += ["-n", spec.namespace]
        return argv + list(args)

    def push_mode(self, spec: DeploymentSpec) -> str:
        if spec.push == "auto":
            return "registry" if spec.registry else "load"
        return spec.push

    def dockerfile_path(self, spec: DeploymentSpec) -> Path:
        return Path(spec.dockerfile) if spec.dockerfile else spec.work_path / DOCKERFILE

    # ── interface ──
    def available(self) -> tuple[bool, str]:
        if not self.runner.which(self.cli):
            return False, f"{self.cli} is not on PATH"
        if not self.runner.which("docker"):
            return False, "docker is not on PATH (needed to build the image)"
        probe = self.runner.run([self.cli, "get", "nodes", "-o", "name"], check=False, timeout=60)
        if not probe.ok:
            return False, f"cluster not reachable: {(probe.stderr or probe.stdout).strip()[:200]}"
        return True, f"{self.cli} reaches {len(probe.stdout.split())} node(s)"

    def render(self, spec: DeploymentSpec) -> dict[str, str]:
        files: dict[str, str] = {}
        if not spec.dockerfile:
            files[DOCKERFILE] = render_dockerfile(spec)
        files[MANIFESTS] = render_manifests(spec, PULL_POLICY[self.push_mode(spec)])
        if self.push_mode(spec) == "load":
            files[LOADER] = render_loader(spec)
        return files

    def build(self, spec: DeploymentSpec) -> str:
        self.write_files(spec)
        self.runner.run(["docker", "build", "-t", spec.image, "-f", str(self.dockerfile_path(spec)), spec.build_context], timeout=1800)
        return spec.image

    def distribute(self, spec: DeploymentSpec, record: DeploymentRecord) -> None:
        """Make `spec.image` pullable / present in the cluster according to `push`."""
        mode = self.push_mode(spec)
        record.details["push"] = mode
        if mode == "registry":
            self.runner.run(["docker", "push", spec.image], timeout=1800)
        elif mode == "load":
            self.runner.run(self.kube(spec, "apply", "-f", str(spec.work_path / LOADER)), timeout=120)
            self.runner.run(self.kube(spec, "rollout", "status", f"daemonset/{loader_name(spec)}", f"--timeout={spec.timeout}s"), timeout=spec.timeout + 30)
            pods = self.runner.run(self.kube(spec, "get", "pods", "-l", f"app={loader_name(spec)}", "-o", "jsonpath={.items[*].metadata.name}"), timeout=60).stdout.split()
            if not pods:
                raise DeploymentError("image loader pods did not start")
            for pod in pods:
                self.runner.pipe(
                    ["docker", "save", spec.image],
                    self.kube(spec, "exec", "-i", pod, "--", "nsenter", "-t", "1", "-m", "-u", "-i", "-n", "--", "ctr", "-n", "k8s.io", "images", "import", "-"),
                    timeout=1800,
                )
            record.resources["loader"] = loader_name(spec)
        # `none`: the nodes already have it (e.g. a shared image store)

    def _start_port_forward(self, spec: DeploymentSpec, record: DeploymentRecord) -> str:
        local = free_port()
        log_path = spec.work_path / PORT_FORWARD_LOG
        argv = self.kube(spec, "port-forward", f"svc/{spec.resource}", f"{local}:{spec.port}", "--address", "127.0.0.1")
        pid = self.runner.spawn(argv, log_path=log_path)
        record.resources["port_forward"] = {"pid": pid, "local_port": local, "log": str(log_path), "argv": argv}
        return f"http://127.0.0.1:{local}"

    def _nodeport_endpoint(self, spec: DeploymentSpec) -> str:
        node_port = self.runner.run(self.kube(spec, "get", "svc", spec.resource, "-o", "jsonpath={.spec.ports[0].nodePort}"), timeout=60).stdout.strip()
        ip = self.runner.run(self.kube(spec, "get", "nodes", "-o", 'jsonpath={.items[0].status.addresses[?(@.type=="InternalIP")].address}', namespaced=False), timeout=60).stdout.strip()
        if not node_port or not ip:
            raise DeploymentError("could not resolve the NodePort endpoint (service nodePort / node InternalIP)")
        return f"http://{ip}:{node_port}"

    def endpoint_for(self, spec: DeploymentSpec, record: DeploymentRecord) -> str:
        if spec.expose == "nodeport":
            return self._nodeport_endpoint(spec)
        return self._start_port_forward(spec, record)

    def up(self, spec: DeploymentSpec) -> DeploymentRecord:
        record = self.new_record(spec)
        record.resources = {"namespace": spec.namespace, "deployment": spec.resource, "service": spec.resource,
                            "manifests": str(spec.work_path / MANIFESTS), "context": spec.context}
        try:
            self.build(spec)
            self.distribute(spec, record)
            self.runner.run(self.kube(spec, "apply", "-f", str(spec.work_path / MANIFESTS)), timeout=120)
            self.runner.run(self.kube(spec, "rollout", "status", f"deployment/{spec.resource}", f"--timeout={spec.timeout}s"), timeout=spec.timeout + 30)
            endpoint = self.endpoint_for(spec, record)
            return self.finish(record, endpoint, spec.timeout)
        except Exception as e:  # noqa: BLE001
            self.failed(record, e)
            raise

    def _deployment_state(self, spec: DeploymentSpec) -> dict:
        """Replica counts through a one-line jsonpath read — a pretty-printed Deployment
        can be longer than the runner's output tail, and a JSON fragment would parse as
        "never ready"."""
        argv = self.kube(spec, "get", "deployment", spec.resource, "-o", REPLICAS_JSONPATH)
        result = self.runner.run(argv, check=False, timeout=60)
        if not result.ok:
            return {"found": False, "ready": 0, "wanted": spec.replicas, "error": (result.stderr or result.stdout).strip()[:200]}
        wanted, ready, available = (_count(f) for f in (result.stdout.strip().split("|") + ["", "", ""])[:3])
        return {"found": True, "ready": ready, "wanted": wanted or spec.replicas, "available": available}

    def ensure_endpoint(self, spec: DeploymentSpec, record: DeploymentRecord) -> bool:
        """Restart a dead port-forward; returns True when the record changed."""
        pf = (record.resources or {}).get("port_forward")
        if spec.expose != "port-forward" or (pf and self.runner.pid_alive(pf.get("pid"))):
            return False
        record.endpoint = self._start_port_forward(spec, record)
        record.touch()
        return True

    def status(self, spec: DeploymentSpec, record: DeploymentRecord) -> dict:
        state = self._deployment_state(spec)
        restarted = self.ensure_endpoint(spec, record) if state["found"] and state["ready"] else False
        health = self.health(record.endpoint) if record.endpoint else {"ok": False, "error": "no endpoint"}
        pf = (record.resources or {}).get("port_forward") or {}
        return {"ready": bool(state["found"] and state["ready"] and health.get("ok")), "endpoint": record.endpoint, "health": health,
                "replicas": {"ready": state["ready"], "wanted": state["wanted"]},
                "details": {**state, "port_forward": {**pf, "alive": self.runner.pid_alive(pf.get("pid"))} if pf else None, "restarted_port_forward": restarted}}

    def logs(self, spec: DeploymentSpec, record: DeploymentRecord, lines: int = 100) -> str:
        return self.runner.run(self.kube(spec, "logs", f"deployment/{spec.resource}", "--tail", str(lines), "--all-containers"), check=False, timeout=60).stdout

    def down(self, spec: DeploymentSpec, record: DeploymentRecord) -> None:
        pf = (record.resources or {}).get("port_forward") or {}
        self.runner.kill(pf.get("pid"))
        manifests = Path((record.resources or {}).get("manifests") or spec.work_path / MANIFESTS)
        if manifests.exists():
            self.runner.run(self.kube(spec, "delete", "-f", str(manifests), "--ignore-not-found", "--wait=false"), check=False, timeout=300)
        else:
            self.runner.run(self.kube(spec, "delete", f"deployment/{spec.resource}", f"service/{spec.resource}", "--ignore-not-found", "--wait=false"), check=False, timeout=300)
        loader = spec.work_path / LOADER
        if (record.resources or {}).get("loader") or loader.exists():
            self.runner.run(self.kube(spec, "delete", f"daemonset/{loader_name(spec)}", "--ignore-not-found", "--wait=false"), check=False, timeout=300)
