# Deploy → infer → eval: a containerised target, joblib-parallel phases, dedicated commands

Date: 2026-09-06. Builds on the artifact-tier layout (2026-09-03) and the two-layer
mocking design (2026-09-01).

## 1. Goal

Split the evaluation into three phases with their own CLI commands, artifacts and
pipeline stages:

1. **Deploy** — package the target agent together with the tool-mocking layers (rules
   and the LLM mock engine) into a container image and run it behind an HTTP endpoint.
   The pipeline talks to that endpoint; it never imports the agent into its own process
   any more (except under the `local` target, see §4.4). Deployment goes through one
   generic interface with four targets: `local` (a subprocess, no Docker), `docker`
   (Docker Compose on the local daemon), `kubernetes` (the Kubernetes cluster of Docker
   Desktop, or any cluster `kubectl` reaches) and `openshift` (a prototype driven by the
   `oc` CLI).
2. **Infer** — run the dataset's approved cases (single- and multi-turn) and the
   multi-turn simulation scenarios against the deployed agent, in parallel with joblib
   (threads or processes). The phase writes the run artifacts (inputs, outputs,
   trajectories, mock ledgers, an internal per-turn log) and the simulation results
   the evaluation phase consumes.
3. **Eval** — score stored run artifacts locally, splitting the cases across joblib
   workers, then aggregate.

Everything stays simple and transparent: generated Dockerfiles, Compose files and
manifests are written to disk before they are used, every external command is logged,
and the HTTP protocol is one JSON request per agent turn.

## 2. Non-goals

- No Helm, no operators, no image registries run by evalbuilder. The Kubernetes target
  loads a locally built image straight into the cluster nodes (what `kind load` does)
  or pushes to a registry the user names.
- No stateful conversations on the server: the client sends the full message history
  each turn, so replicas are interchangeable and nothing needs a session store.
- The interactive skills keep their commands; `evalbuilder run` and `evalbuilder score`
  survive as aliases of `infer` and `eval`.
- OpenShift is a prototype: the code path is complete and unit-tested against a fake
  `oc`, but no live cluster was available to run it.

## 3. Phases, stages and commands

| phase | CLI | pipeline stage(s) | artifact |
|---|---|---|---|
| deploy | `evalbuilder deploy up\|status\|down\|build\|render CONFIG` | `deploy` (after `verify`), `teardown` (after `simulate`) | `deployment.json` (`evalbuilder/deployment/v1`), rendered files under `work/deploy/` |
| infer | `evalbuilder infer DATASET [--scenarios F] …` | `infer` (replaces `run`), `simulate` (unchanged authoring, runs through the same engine) | `results/run-<id>.json`, `simulation.json` |
| eval | `evalbuilder eval RUN… --dataset D --evaluators F` | `score`, `aggregate` | `results/score-report-<id>.json`, `aggregate.json` |
| serve | `evalbuilder serve --module M …` | — (the container's entry point) | — |

New stage order: preflight → discover → map → mocks → dataset → review → verify →
**deploy** → **infer** → simulate → **teardown** → score → aggregate → publish → analyze
→ report. `deploy` and `infer` are required (a failure makes the verdict `incomplete`);
`teardown` is optional (a failure is a problem, not an incomplete verdict) and is skipped
when `deploy.keep` is true. `--until deploy` leaves the agent running for manual probing.

The stage `run` is renamed `infer` everywhere (state, report, UI, docs). Directories
written before the rename still load: readers accept `stages.run` as a fallback for
`stages.infer`, and the run artifact kind keeps its file name `results/run-<id>.json`.

## 4. Deployment

### 4.1 Config

```yaml
deploy:
  target: docker            # local | docker | kubernetes | openshift
  image:
    name: null              # default evalbuilder-<pipeline name>
    tag: latest
    registry: null          # e.g. docker.io/me, quay.io/team, <openshift registry>/<project>
    push: auto              # auto | registry | load | none  (kubernetes / openshift only)
    extras: []              # evalbuilder extras installed in the image, e.g. [llm]
  build:
    context: "."            # docker build context (the repo root by default)
    dockerfile: null        # your own Dockerfile; null = the generated one
    include: []             # extra paths copied into the image (the target's package is always included)
    requirements: null      # a requirements.txt installed in the image
  port: 8080                # container port of the agent server
  namespace: default        # kubernetes / openshift
  replicas: 1
  expose: port-forward      # port-forward | nodeport | route  (how the pipeline reaches the service)
  env: {}                   # container environment; a null value copies the variable from the host (API keys)
  keep: false               # leave the deployment running after the pipeline (teardown skipped)
  timeout: 240              # seconds to wait for readiness
  context: null             # kubectl / oc context (null = current)
```

`PipelineConfig` defaults `deploy.target` to `local` so every existing config and every
offline test keeps working without Docker; `pipeline init` writes `docker` into the
starter and the setup form offers all four targets.

Preflight checks the target's prerequisites (`docker` on PATH and the daemon answering;
`kubectl` and a reachable context; `oc` and a login) and rejects `claude-cli:` model
specs for the agent and the mock model under any container target — the `claude` binary
is not in the image. API-key providers work when the key is passed through `deploy.env`
(`ANTHROPIC_API_KEY: null` copies it from the host); `scripted:` models work because
the target package is copied into the image.

### 4.2 The interface

```python
class DeploymentTarget:                     # src/evalbuilder/deploy/base.py
    kind: str
    def available(self) -> tuple[bool, str]       # prerequisites (binaries, daemon, context)
    def render(self, spec) -> dict[str, str]      # file name → content, written to work/deploy/
    def build(self, spec) -> str                  # image reference (no-op for local)
    def up(self, spec) -> DeploymentRecord        # build (unless prebuilt) + deploy + wait + endpoint
    def status(self, spec, record) -> dict        # {ready, endpoint, health, replicas, details}
    def logs(self, spec, record, lines) -> str
    def down(self, spec, record) -> None
```

`DeploymentSpec` (`deploy/spec.py`) is built from the pipeline config (or CLI flags):
name, target, image reference, build options, port, namespace, replicas, expose, env,
timeout, context, the target module/factory, the agent and mock model specs, and the
directory for rendered files. `DeploymentRecord` is the `deployment.json` artifact:
target, image, endpoint, resources (compose project / namespace + names / pids),
`expose`, timestamps, status and the log of commands run.

Every external command goes through `deploy/runner.py::CommandRunner`, which appends
`argv`, exit code and the first lines of output to `work/deploy/commands.log` and is
replaced by a fake in unit tests. Targets never call `subprocess` directly.

### 4.3 Image and server

`deploy/render.py` generates:

- `Dockerfile` — `python:3.12-slim`, copies `pyproject.toml`, `README.md`, `src/`, the
  target's top-level package (derived from `target.module`) and `build.include`, installs
  `.[<extras>]` (+ `build.requirements`), and runs
  `evalbuilder serve --module M --factory F --port P` with `EVALBUILDER_AGENT_MODEL` /
  `EVALBUILDER_MOCK_MODEL` from the environment. A user-supplied `build.dockerfile` is
  used verbatim.
- `compose.yaml` — one service, `build`, `ports: ["<port>:<port>"]`, `environment`,
  a `/health` healthcheck; the project name is the pipeline name.
- `manifests.yaml` — a Deployment (replicas, container port, env, readiness probe on
  `/health`, `imagePullPolicy: Never` for loaded images / `Always` for pushed ones) and a
  Service (ClusterIP, or NodePort under `expose: nodeport`); under OpenShift also a Route.
- `loader.yaml` — the privileged `evalbuilder-image-loader` DaemonSet (`hostPID`,
  `busybox`) used by the `load` distribution (§4.4).

`src/evalbuilder/serve.py` is the agent server: a stdlib `ThreadingHTTPServer` (no web
framework dependency) with two routes:

- `GET /health` → `{ok, module, factory, tools, agent_model, mock_model, version}`;
- `POST /invoke` with `{messages, mocks?, mocked}` → `{messages, response, tool_calls,
  node_path, mock_calls, error, error_class, seconds}`. `messages` are OpenAI-format
  (what `convert_to_openai_messages` produces; `convert_to_messages` round-trips them
  exactly, tool calls included). `mocks` is the dataset's `mocks` block merged with the
  case's overrides: `tools` (rules), `on_miss`, `strategy`, `strategies`, `llm`. The
  server wraps the target's `TOOLS` per request, builds the graph, invokes it with the
  history and returns the full trajectory — the same code `run_case` used in-process,
  now behind `agent_client.LocalAgent`.

### 4.4 Targets

- **local** — `python -m evalbuilder.cli serve …` as a detached subprocess (same
  interpreter, cwd = project root), a free port on 127.0.0.1, pid recorded; `down` kills
  the process group. No Docker needed: this is what the offline tests and the UI browser
  tests use, and what a config without a `deploy` section gets.
- **docker** — `docker compose -p <name> -f work/deploy/compose.yaml up -d --build --wait`;
  endpoint `http://127.0.0.1:<port>`; `status` from `docker compose ps --format json`;
  `down` = `docker compose down --remove-orphans`.
- **kubernetes** — `docker build`, then image distribution by `image.push`: `registry`
  (`docker tag` + `docker push`, pods pull with `Always`), `load` (apply the loader
  DaemonSet, then for every loader pod `docker save IMAGE | kubectl exec -i POD --
  nsenter -t 1 -m -u -i -n -- ctr -n k8s.io images import -`, pods use `Never`), `none`
  (the image is already on the nodes, e.g. Docker Desktop's kubeadm provisioner), `auto`
  (= `registry` when `image.registry` is set, else `load`). Then `kubectl apply -f
  manifests.yaml`, `kubectl rollout status deploy/<name> --timeout`, and the endpoint:
  `port-forward` (a detached `kubectl port-forward svc/<name> <local>:<port>`, pid and
  local port recorded, restarted by `status`/`infer` when it died) or `nodeport`
  (`http://<node internal ip>:<nodePort>`). `down` deletes the manifests, the loader
  DaemonSet and the port-forward. Verified on Docker Desktop's kind-based cluster, where
  locally built images are *not* visible to the nodes and NodePort / LoadBalancer services
  are not reachable from the host — `load` + `port-forward` is the combination that works.
- **openshift** — `KubernetesTarget` with `cli = "oc"`: `push` defaults to `registry`
  (`image.registry` is required; `oc registry info --public` gives the internal one),
  `expose` defaults to `route` (`oc expose svc/<name>` then `oc get route -o
  jsonpath={.spec.host}`), and `available()` checks `oc whoami`. Prototype: exercised
  with a fake command runner only.

`evalbuilder deploy render CONFIG` prints the generated files without running anything;
`deploy build` builds the image only; `deploy up` writes `deployment.json`; `deploy
status` and `deploy down` read it back (CONFIG or the output directory).

## 5. Inference

### 5.1 Agent clients (`src/evalbuilder/agent_client.py`)

One interface, two implementations, both picklable for joblib's process backend:

- `LocalAgent(module, factory, agent_model_spec, mock_model_spec)` — imports the target
  lazily, builds a graph per call with the requested mocks (rules, and the LLM engine
  when `on_miss` is `llm`), invokes it, returns an `InvokeResult`. Used by the server
  and by `evalbuilder infer --local`.
- `RemoteAgent(endpoint, timeout)` — `POST /invoke` with `urllib`; `health()`.

`InvokeResult`: `messages` (OpenAI format), `response`, `tool_calls`, `node_path`,
`mock_calls` (ledger), `error`, `error_class`, `seconds`.

### 5.2 Engine (`src/evalbuilder/inference.py`)

- `infer_cases(agent, ds, cases, *, workers, backend, on_miss, strategy, progress)` —
  `joblib.Parallel(n_jobs=workers, prefer="threads"|"processes",
  return_as="generator")` over one task per case; each task replays the case's turns
  (`inputs.messages`, then every `metadata.user_turns` entry appended to the returned
  history) and records a per-turn `log` entry `{turn, seconds, tool_calls, error,
  endpoint}`. Results come back in dataset order; `progress` is called in the parent
  after every case (the same payload the run stage used for `run-progress.json`). A
  failed task is a `CaseRun` with `error`, never an exception.
- `infer_dataset(ds, dataset_path, agent, *, out_dir, …) -> RunArtifact` — selects the
  approved cases (or `ids`), runs them, writes `run-<id>.json`. `RunArtifact` gains
  `execution: {mode: local|remote, endpoint, workers, backend, seconds}` and each
  `CaseRun` gains `log`.
- `simulate_scenarios(agent, scenarios, *, mocks, user_model_spec, workers, backend)` —
  the multi-turn simulation loop of `simulate.py`, stepping through the agent client
  instead of a graph; a scenario's `mock_strategy` selects the strategy; results carry
  `mock_calls` and a per-turn `log`. The simulated user is the generator model, given as
  a spec so process workers can build their own.
- `run_dataset` in `runner.py` becomes a thin wrapper over `infer_dataset` with a
  `LocalAgent`, keeping its signature for the interactive CLI and the tests.

Intent-group parallelism goes away: the server is stateless and every case is
independent, so the unit of parallelism is the case. `runs.parallel_intents`,
`runs.parallel_scoring` and `runs.parallel_simulations` are replaced by
`inference.workers` / `inference.backend` and `evaluation.workers` /
`evaluation.backend`; `parse_config` migrates the old keys (with a recorded problem) so
old configs still load.

### 5.3 Pipeline stages

- `infer` — reads `deployment.json`, checks `/health`, runs `runs.repeats` ×
  `infer_dataset(RemoteAgent(endpoint))` with `inference.workers`; progress to
  `work/run-progress.json` as before (with `endpoint` and `workers`); details carry
  `execution`, per-layer mock totals, error counts.
- `simulate` — authors the scenarios as before, runs them through the same
  `RemoteAgent` with `inference.workers`, mines violations.
- `teardown` — `target.down()` unless `deploy.keep`; records what was removed.

## 6. Evaluation

`evaluators.score_run(run, ds, specs, judge_model, workers=1, backend="threads")`
splits the case runs into `workers` contiguous chunks, scores each chunk in a joblib
worker (evaluators are rebuilt inside the worker from the specs, so the process backend
works), and concatenates the rows in run order — the report is byte-identical to a
sequential pass. `evalbuilder eval RUN… --dataset D --evaluators F [--workers N]
[--backend B] [--out D] [--config CFG]` scores every run given, writes the score reports
and, with `--aggregate` (default when more than one run is given), `aggregate.json`
using the thresholds of `--config` or the defaults. `evalbuilder score` stays as an
alias for one run.

## 7. UI

- **Pipeline setup** — a *Deployment* block (target, image name, registry, namespace,
  expose, keep) and *Parallelism* fields (inference workers / backend, evaluation
  workers). `default_form`, `form_from_config`, `build_config` round-trip them.
- **Run & review** — a *Deployment* block: target, image, endpoint, live status
  (`deploy status`: ready, health, replicas), the rendered files' location, and
  *Tear down* (calls `deploy down`); the stage table shows deploy / infer / teardown;
  the case progress bar follows the `infer` stage.
- **Stages & problems** and **Summary** list `deployment.json`; the loader exposes
  `bundle.deployment`; `run_ids` reads the `infer` stage (falling back to `run`).

## 8. Artifacts

| kind | file | tier | stage |
|---|---|---|---|
| `deployment` | `deployment.json` (`evalbuilder/deployment/v1`) | final | deploy, teardown |
| rendered files, command log, server / port-forward logs | `work/deploy/…` | work | deploy |

`run` and `simulation` keep their names; `run` gains `execution`, `case_runs[].log`;
simulation results gain `log`.

## 9. Testing

- Unit: `deploy/` renderers and every target against a fake `CommandRunner` (argv
  sequences, `push` decisions, endpoint modes, records, `down`); the local target with a
  real subprocess; `serve.py` in a thread (health, invoke with rules, multi-turn round
  trip, LLM engine with a scripted mock model, errors); `agent_client` parity local vs
  remote; `inference.py` with threads and processes (order, progress, isolated errors,
  multi-turn, sequential == parallel, log entries); `score_run` splits == sequential;
  config migration; CLI `deploy up|status|down|render`, `infer`, `eval`, `serve`.
- Pipeline e2e (offline, `local` target): the new stage list and artifacts,
  `--until deploy` leaves the server running, `deploy.keep`, teardown.
- Integration (marker `docker`, skipped without a daemon / cluster): Compose up → health
  → infer → down; Kubernetes load + port-forward → health → infer → down. Run locally on
  Docker Desktop.
- UI: AppTest for the new form fields and the deployment block; Playwright flow for a
  local-target run showing the deployment status; existing suites updated for the
  renamed stage and the removed `parallel_*` keys.

## 10. Decisions

1. **The pipeline talks HTTP to every target, including `local`.** One inference code
   path, one protocol, one set of tests; `local` is a subprocess running the same server
   the container runs. The in-process path survives only inside `LocalAgent` (used by the
   server itself and by `--local` runs).
2. **Stateless server, full history per turn.** Replicas are interchangeable, no session
   store, and the OpenAI message format round-trips exactly in langchain-core.
3. **Mocks travel with the request.** The dataset's `mocks` block (rules, policy,
   strategies, engine settings) is sent per request, so the image contains no dataset
   and a new dataset needs no rebuild.
4. **Image loading through a loader pod, not a registry, by default on Kubernetes.**
   Docker Desktop's kind-based cluster cannot see local images; streaming `docker save`
   into `ctr` on each node needs nothing but `kubectl`. A registry push is one config key
   away and is what OpenShift uses.
5. **Port-forward is the default endpoint.** It works on every cluster; NodePort and
   LoadBalancer do not reach the host on Docker Desktop's kind cluster.
6. **joblib with `return_as="generator"`** keeps progress reporting and result order
   trivial for both backends; the unit of work is the case / scenario / score chunk.
7. **Stdlib HTTP.** No FastAPI/uvicorn/requests: the protocol is two routes of JSON, and
   the image stays small.
8. **`run` → `infer`, `score` → `eval`**, with the old names kept as aliases, because the
   phase names now match the config sections, the stages and the docs.
