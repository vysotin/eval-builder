"""Deployment targets: run the target agent (with both mock layers) behind an HTTP
endpoint — locally, in Docker Compose, on Kubernetes, or on OpenShift (prototype).

See `spec.py` (what to deploy, the `deployment.json` record), `render.py` (generated
Dockerfile / compose / manifests), `runner.py` (logged external commands), `base.py`
(the target interface) and one module per target. `target_for`, `deploy_up`,
`deploy_status` and `deploy_down` are the entry points the CLI, the pipeline and the
UI use.
"""
