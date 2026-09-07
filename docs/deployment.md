# Deploying the agent: local, Docker, Kubernetes, OpenShift

During the inference phase evalbuilder does not import the target agent into its own
process. It runs the agent — with both tool-mocking layers packed in — behind a small
HTTP server, `evalbuilder serve`, on a **deployment target**, and talks to it over
HTTP. This page is the walkthrough: what runs where, which commands and files each
target produces, how the image reaches a cluster, how the pipeline reaches the service,
and what to do when something does not come up. The config keys are in
`skills/agent-eval-pipeline/references/config-reference.md`; the design and its
alternatives in `docs/superpowers/specs/2026-09-06-deploy-infer-eval-design.md`.

## The three phases

| phase | command | pipeline stages | writes |
|---|---|---|---|
| deploy | `evalbuilder deploy up\|status\|down\|build\|render\|logs` | `deploy`, `teardown` | `deployment.json`, `work/deploy/*` |
| infer | `evalbuilder infer DATASET [--deployment DIR \| --endpoint URL]` | `infer`, `simulate` | `results/run-<id>.json`, `simulation.json` |
| eval | `evalbuilder eval RUN… --dataset D --evaluators F` | `score`, `aggregate` | `results/score-report-<id>.json`, `aggregate.json` |

`evalbuilder pipeline run CONFIG` runs them in order (preflight → … → verify → deploy →
infer → simulate → teardown → score → aggregate → …). Each is also a command, so you can
deploy once and run several datasets against the same agent, probe it with `curl`, or
score a stored run with different evaluators without touching the deployment.

```bash
uv run evalbuilder deploy render examples/weather_bot/pipeline.yaml         # the generated files, nothing runs
uv run evalbuilder deploy up examples/weather_bot/pipeline.yaml             # build + deploy + wait for /health
uv run evalbuilder deploy status eval/pipeline/weather-bot                  # exit 1 unless up and healthy
uv run evalbuilder infer eval/pipeline/weather-bot/dataset.json --deployment eval/pipeline/weather-bot \
    --scenarios eval/pipeline/weather-bot/scenarios.yaml --repeats 2 --workers 4 --out eval/pipeline/weather-bot/results
uv run evalbuilder eval eval/pipeline/weather-bot/results/run-*.json --dataset eval/pipeline/weather-bot/dataset.json \
    --evaluators eval/pipeline/weather-bot/evaluators.yaml --config examples/weather_bot/pipeline.yaml --aggregate
uv run evalbuilder deploy logs eval/pipeline/weather-bot --lines 50
uv run evalbuilder deploy down eval/pipeline/weather-bot
```

`examples/weather_bot/pipeline.yaml` is the fully offline, docker-ready example: scripted
agent, generator and mock models, `deploy.target: docker`. It needs a Docker daemon and
nothing else; switch the target to `kubernetes` for the Docker Desktop cluster or to
`local` for no Docker at all.

## The config section

```yaml
deploy:
  target: local            # local | docker | kubernetes | openshift
  image:
    name: null             # default evalbuilder-<pipeline name>
    tag: latest
    registry: null         # e.g. docker.io/me, quay.io/team — where `push: registry` pushes; required for openshift
    push: auto             # kubernetes/openshift: auto (registry when set, else load) | registry | load | none
    extras: []             # evalbuilder extras installed in the image, e.g. [llm] for API-key providers
  build:
    context: "."           # docker build context: the checkout (src/evalbuilder must be in it)
    dockerfile: null       # your own Dockerfile instead of the generated one
    include: []            # extra paths copied into the image (the target's package always is)
    requirements: null     # requirements.txt installed in the image
  port: 8080               # container port of the agent server
  host_port: null          # docker only: host port it is published on (null = the same as port)
  namespace: default       # kubernetes / openshift namespace (project)
  replicas: 1
  expose: port-forward     # kubernetes: port-forward | nodeport; openshift: route
  env: {}                  # container environment; a null value copies the variable from the host
  keep: false              # leave the deployment running after the pipeline (skips teardown)
  timeout: 240             # seconds to wait for readiness
  context: null            # kubectl / oc context (null = current)
inference:
  workers: 4               # conversations (cases, scenarios) in flight against the agent
  backend: threads         # joblib: threads | processes
  timeout: 120             # seconds per agent request
```

Two rules the config check enforces: a container target (`docker`, `kubernetes`,
`openshift`) rejects `claude-cli:` for `models.agent` and for the mock model under
`mocking.on_miss: llm` — the `claude` binary is not in the image; use an API-key provider
(`anthropic:claude-sonnet-5` with `deploy.env: {ANTHROPIC_API_KEY: null}`) or a
`scripted:` model. And `openshift` needs `image.registry`.

## What every target does

The interface is `src/evalbuilder/deploy/base.py::DeploymentTarget` — `available()`,
`render()`, `build()`, `up()`, `status()`, `logs()`, `down()` — and every external
command goes through one `CommandRunner` that appends `argv`, exit code and output to
`<out_dir>/work/deploy/commands.log` and into the record's `commands`. Rendered files are
written to `work/deploy/` *before* they are used; `evalbuilder deploy render` shows them
without running anything.

### `local` — a subprocess, no Docker

`python -m evalbuilder.cli serve --module M --factory F --host 127.0.0.1 --port <free>`
is spawned detached (its own session) with the container environment
(`EVALBUILDER_AGENT_MODEL`, `EVALBUILDER_MOCK_MODEL`, your `deploy.env`), logging to
`work/deploy/serve.log`. The record holds the pid and the port; `status` checks the pid
and `/health`; `down` kills the process group. If the server exits before it is healthy
(a broken module, a missing factory, an unbuildable mock model spec — the server
validates all three at start-up) `deploy up` fails at once with the log tail, not after
the timeout. This is what a config without a `deploy` section gets, and what the offline
tests and the UI browser tests use.

### `docker` — Docker Compose on the local daemon

Files: `work/deploy/Dockerfile` (unless `build.dockerfile`) and `work/deploy/compose.yaml`
(project `evalbuilder-<name>`, one service `agent`, `ports: ["<host_port>:<port>"]`, the
environment, a `/health` healthcheck). Commands:

```
docker compose -p evalbuilder-<name> -f work/deploy/compose.yaml up -d --build --wait --wait-timeout <timeout>
docker compose … ps --all --format json          # status
docker compose … logs --no-color --tail N agent  # logs
docker compose … down --remove-orphans           # down
```

Endpoint `http://127.0.0.1:<deploy.host_port>`. The container port (`deploy.port`) is
published on `deploy.host_port`, which defaults to the same number — so when 8080 is
already taken on the host, set `host_port: 9090` and leave the container port alone
(nothing inside the image changes). `host_port` is docker-only: the config check rejects
it on the other targets. `available()` needs `docker` on PATH, a reachable daemon
(`docker info`) and Compose v2 (`docker compose version`).

### `kubernetes` — kubectl

Files: `Dockerfile`, `manifests.yaml` (a Deployment with readiness/liveness probes on
`/health` and the environment, plus a Service — ClusterIP, or NodePort under
`expose: nodeport`) and, under `push: load`, `loader.yaml`. Commands, in order:

```
docker build -t <image> -f work/deploy/Dockerfile <build.context>
# image distribution — one of:
docker push <image>                                                          # push: registry (imagePullPolicy Always)
kubectl -n NS apply -f work/deploy/loader.yaml                               # push: load  (imagePullPolicy Never)
kubectl -n NS rollout status daemonset/evalbuilder-<name>-image-loader --timeout=<t>s
kubectl -n NS get pods -l app=evalbuilder-<name>-image-loader -o jsonpath={.items[*].metadata.name}
docker save <image> | kubectl -n NS exec -i <loader pod> -- nsenter -t 1 -m -u -i -n -- ctr -n k8s.io images import -   # per node
                                                                             # push: none (IfNotPresent): the nodes already have it
kubectl -n NS apply -f work/deploy/manifests.yaml
kubectl -n NS rollout status deployment/evalbuilder-<name> --timeout=<t>s
kubectl -n NS port-forward svc/evalbuilder-<name> <local>:<port> --address 127.0.0.1   # expose: port-forward (detached; pid recorded)
kubectl -n NS get svc … -o jsonpath={.spec.ports[0].nodePort}; kubectl get nodes … InternalIP   # expose: nodeport
```

`image.push: auto` means `registry` when `image.registry` is set, else `load`. `load`
streams `docker save` into every node's containerd through a privileged, `hostPID`
busybox DaemonSet — the same mechanism `kind load docker-image` uses, done through the
Kubernetes API so it works on clusters whose nodes you cannot `docker exec` into. It
needs privileged pods to be allowed.

**Docker Desktop (verified on its kind-based provisioner):** locally built images are
*not* visible to the nodes (a pod with `imagePullPolicy: Never` stays in
`ErrImageNeverPull`), the nodes are hidden containers, and NodePort / LoadBalancer
services are *not* reachable from the host — while `kubectl port-forward` is. So the
defaults, `load` + `port-forward`, are exactly what works there; `docker desktop
kubernetes` has no image-load command of its own.

`status` reports the Deployment's ready replicas, the port-forward process and
`/health`; when the port-forward died (a reboot, a killed terminal) it starts a new one
and updates the record — `infer --deployment DIR` and the pipeline's stages call
`status` first, so they never talk to a dead tunnel. `down` deletes the manifests, the
loader DaemonSet and the port-forward. A port-forward is one tunnel process, not a load
balancer: `replicas > 1` still funnels through it (use `nodeport` or a Route for real
fan-out).

### `openshift` — oc (prototype)

The Kubernetes target with `oc` in place of `kubectl`, `push: registry` by default
(`image.registry` is required — `oc registry info --public` names the internal one; log
Docker into it first) and `expose: route` by default: the manifests add a Route and the
endpoint is `http(s)://<route host>` (`https` when the Route has TLS termination).
`available()` checks `oc whoami`. This target is exercised against a fake `oc` only —
no live cluster was available — so treat it as a starting point: the command sequence
is complete and unit-tested, the cluster-side behaviour is not.

## The record: `deployment.json`

Written at the output root (`evalbuilder/deployment/v1`) by `deploy up` and updated by
`status` and `down`:

```json
{
  "schema": "evalbuilder/deployment/v1",
  "name": "weather-bot", "target": "kubernetes",
  "image": "evalbuilder-weather-bot:latest",
  "endpoint": "http://127.0.0.1:52286", "expose": "port-forward",
  "status": "up",
  "resources": {"namespace": "default", "deployment": "evalbuilder-weather-bot", "service": "evalbuilder-weather-bot",
                "loader": "evalbuilder-weather-bot-image-loader",
                "port_forward": {"pid": 20317, "local_port": 52286, "log": ".../work/deploy/port-forward.log"}},
  "details": {"push": "load", "health": {"module": "examples.weather_bot.agent", "tools": ["get_weather", "get_alerts"], "agent_model": "scripted:…"}},
  "commands": [{"argv": ["docker", "build", "…"], "returncode": 0, "seconds": 12.4, "output": "…"}],
  "spec": {"…the DeploymentSpec that produced it…"}
}
```

`status` is `up`, `down` or `failed` (with `details.error`). `evalbuilder infer
--deployment DIR` reads the endpoint from it; the UI's Run & review page shows it and
offers *Tear down*; `report.json` carries a summary under `deployment`. A `deploy up` on
a record that is already `up` and healthy reuses it; on an unhealthy one it tears down
and redeploys.

## The HTTP protocol

Two JSON routes, stdlib server, no framework:

```bash
curl -s http://127.0.0.1:8080/health
# {"ok": true, "module": "examples.weather_bot.agent", "factory": "build_agent", "tools": ["get_weather", "get_alerts"],
#  "agent_model": "scripted:…", "mock_model": "scripted:…", "server": "evalbuilder/serve/v1"}

curl -s -X POST http://127.0.0.1:8080/invoke -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "mocks": {"tools": {"get_weather": [{"matchArgs": {}, "response": {"temp": 5, "condition": "mocked", "city": "Paris"}}]},
            "on_miss": "strict"},
  "mocked": true}'
# {"messages": [... the full history, tool calls and tool results included ...], "response": "It is 5F and mocked in Paris.",
#  "tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}], "node_path": ["agent", "tools", "agent"],
#  "mock_calls": [{"tool": "get_weather", "layer": "rule", ...}], "error": null, "error_class": "none", "seconds": 0.02}
```

The server is **stateless**: every request carries the whole conversation
(OpenAI-format messages — what `convert_to_openai_messages` renders, and what
`convert_to_messages` reads back exactly) and the mock block to install — `tools`
(rules, the case's overrides already merged over the dataset's), `on_miss`, `strategy`,
`strategies`, `llm` (model, `on_invalid`, `max_repairs`) and `history` (the mocked
answers of earlier turns, so the LLM mock engine stays consistent across a multi-turn
conversation). The next turn is the returned `messages` plus the new user message. A
graph and fresh tool wrappers are built per request, so replicas are interchangeable
and no dataset lives in the image. The agent model is the deployment's
(`EVALBUILDER_AGENT_MODEL`); the mock model is the deployment's when configured
(`EVALBUILDER_MOCK_MODEL`), else the request's `mocks.llm.model`.

## Environment and secrets

`deploy.env` is the container environment. A `null` value copies the variable from the
host at deploy time (`ANTHROPIC_API_KEY: null`); an unset one is reported as a problem.
The model specs are added as `EVALBUILDER_AGENT_MODEL` / `EVALBUILDER_MOCK_MODEL`. Note
that the rendered `compose.yaml` and `manifests.yaml` under `work/deploy/` contain those
values in plain text — `work/` is scratch, keep it out of version control (the default
`eval/` output root is git-ignored).

## Keeping a deployment around

- `evalbuilder pipeline run CFG --until deploy` runs everything up to the deployment and
  leaves the agent running; probe it, run `evalbuilder infer … --deployment DIR`, then
  `--resume` to let the pipeline continue (it reuses the healthy deployment) or
  `evalbuilder deploy down DIR`.
- `deploy.keep: true` skips the `teardown` stage after a full run.
- The UI's Run & review page shows the deployment block with *Tear down* while it is up.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `deployment target 'docker' unavailable: docker is not on PATH` / `docker daemon not reachable` | install / start Docker (Desktop); `docker info` must work for the user running evalbuilder |
| `models.agent 'claude-cli:…' cannot run inside the agent container` | the `claude` binary is not in the image: an API-key provider with the key in `deploy.env`, or a scripted model; or `deploy.target: local` |
| `docker compose … up` fails with a port in use | the host port `deploy.host_port` (default: `deploy.port`) is taken — set `deploy.host_port` to a free one; the container port can stay 8080 |
| pod stuck in `ErrImageNeverPull` / `ImagePullBackOff` | the nodes cannot see a local image: keep `image.push: auto` without a registry (= `load`, needs privileged pods), or set `image.registry` and let it `push` |
| `image loader pods did not start` | the cluster refuses privileged / `hostPID` pods: push to a registry instead (`image.registry`) |
| `did not become healthy within …s` on Kubernetes | `evalbuilder deploy logs DIR` (container log) and `kubectl -n NS describe deployment evalbuilder-<name>`; a wrong `build.include` (the agent's imports are not in the image) shows up here |
| `the agent server behind … exited before becoming healthy: …` (local target) | the log tail names the import / factory / model-spec error; `work/deploy/serve.log` has the whole traceback |
| `deployed agent at … is not healthy` at the start of infer | the port-forward died and could not be restarted, or the pods were removed: `evalbuilder deploy status DIR`, then `deploy up` again |
| `every case failed with infrastructure errors` | the agent answered every request with an error: `evalbuilder deploy logs DIR`, and the per-turn `log` in the run artifact (`error`, `endpoint`) |
| `deploy.env X is not set on the host` (problem) | export the variable (or put it in `.env`, loaded by the CLI) before `deploy up` |
| OpenShift: `openshift needs deploy.image.registry` | set `image.registry` (e.g. `oc registry info --public` + `/<project>`), `docker login` to it, and `oc login` first |
