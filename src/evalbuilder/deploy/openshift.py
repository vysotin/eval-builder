"""The `openshift` target — a prototype driven by the `oc` CLI: the Kubernetes target
with `oc` in place of `kubectl`, the image pushed to a registry the cluster pulls
from (`deploy.image.registry`; `oc registry info --public` names the internal one),
and a Route as the endpoint. Exercised against a fake `oc` only (see tests)."""

from __future__ import annotations

from evalbuilder.deploy.base import DeploymentError
from evalbuilder.deploy.kubernetes import KubernetesTarget
from evalbuilder.deploy.spec import DeploymentRecord, DeploymentSpec


class OpenShiftTarget(KubernetesTarget):
    kind = "openshift"
    cli = "oc"

    def push_mode(self, spec: DeploymentSpec) -> str:
        return "registry" if spec.push == "auto" else spec.push

    def available(self) -> tuple[bool, str]:
        if not self.runner.which("oc"):
            return False, "oc is not on PATH (install the OpenShift CLI)"
        if not self.runner.which("docker"):
            return False, "docker is not on PATH (needed to build and push the image)"
        who = self.runner.run(["oc", "whoami"], check=False, timeout=60)
        if not who.ok:
            return False, f"not logged in: run `oc login` ({(who.stderr or who.stdout).strip()[:120]})"
        return True, f"oc logged in as {who.stdout.strip()}"

    def _route_endpoint(self, spec: DeploymentSpec) -> str:
        host = self.runner.run(self.kube(spec, "get", "route", spec.resource, "-o", "jsonpath={.spec.host}"), timeout=60).stdout.strip()
        if not host:
            raise DeploymentError(f"route {spec.resource} has no host yet")
        tls = self.runner.run(self.kube(spec, "get", "route", spec.resource, "-o", "jsonpath={.spec.tls.termination}"), check=False, timeout=60).stdout.strip()
        return f"{'https' if tls else 'http'}://{host}"

    def endpoint_for(self, spec: DeploymentSpec, record: DeploymentRecord) -> str:
        if spec.expose == "route":
            record.resources["route"] = spec.resource
            return self._route_endpoint(spec)
        return super().endpoint_for(spec, record)

    def distribute(self, spec: DeploymentSpec, record: DeploymentRecord) -> None:
        if self.push_mode(spec) == "registry" and not spec.registry:
            raise DeploymentError("openshift needs deploy.image.registry (a registry the cluster pulls from)")
        super().distribute(spec, record)
