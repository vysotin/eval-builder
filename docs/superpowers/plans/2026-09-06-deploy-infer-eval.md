# Deploy → infer → eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the target agent (with both mocking layers) behind an HTTP server deployed through a generic target interface (local / docker / kubernetes / openshift), run cases and simulations against it in parallel with joblib, score locally with joblib splits, and expose the three phases as `evalbuilder deploy`, `evalbuilder infer`, `evalbuilder eval` plus pipeline stages, UI, docs and tests.

**Architecture:** `serve.py` (stdlib HTTP) wraps `agent_client.LocalAgent`; `agent_client.RemoteAgent` speaks the same `InvokeResult` contract over HTTP; `inference.py` runs cases/scenarios over either client with `joblib.Parallel(return_as="generator")`; `deploy/` renders Dockerfile / compose / manifests and drives `docker`, `kubectl`, `oc` through a logged `CommandRunner`; the pipeline gains `deploy`, `infer` (ex `run`) and `teardown` stages.

**Tech Stack:** Python 3.11+, joblib ≥ 1.6 (`prefer="threads"` / loky processes, `return_as="generator"`), stdlib `http.server` + `urllib`, Docker Compose v2+, kubectl, oc (prototype), Streamlit 1.62, pytest / AppTest / Playwright.

**Spec:** `docs/superpowers/specs/2026-09-06-deploy-infer-eval-design.md`

## Global Constraints

- No new web framework or HTTP client dependency: server = `http.server.ThreadingHTTPServer`, client = `urllib.request`.
- Every external command in `deploy/` goes through `CommandRunner.run(argv, input=None, check=True)`; targets never import `subprocess`.
- Generated deployment files are written under `<out_dir>/work/deploy/` before use; `deployment.json` is a `final` artifact at the output root (`evalbuilder/deployment/v1`).
- Old configs keep loading: `runs.parallel_intents|parallel_scoring|parallel_simulations` migrate to `inference.workers` / `evaluation.workers`; `stages.run` in an old `state.json` is read as `infer`.
- `evalbuilder run` and `evalbuilder score` remain as aliases of `infer` / `eval`.
- Parallel results are identical to sequential results (order preserved; same artifacts).
- Every stage/CLI/UI change is covered by offline tests; Docker/Kubernetes integration tests carry the `docker` marker and skip when the daemon / cluster is unavailable.

---

## File structure

```
src/evalbuilder/
  agent_client.py        InvokeResult, LocalAgent (in-process graph per call), RemoteAgent (HTTP), agent_for()
  serve.py               ThreadingHTTPServer: GET /health, POST /invoke; serve(module, factory, host, port, …)
  inference.py           infer_cases / infer_dataset / simulate_scenarios (joblib), progress, logs
  runner.py              run_dataset() = LocalAgent + infer_dataset (signature kept)
  simulate.py            simulate_scenario(step, …): the loop takes a step callable; graph_step() adapter
  evaluators.py          score_run(…, workers, backend): chunked joblib scoring
  deploy/__init__.py     TARGETS, target_for, load_record, deploy_up/down/status helpers
  deploy/spec.py         DeploymentSpec (from config), DeploymentRecord (artifact)
  deploy/runner.py       CommandRunner (logged subprocess), FakeRunner lives in tests
  deploy/render.py       render_dockerfile / render_compose / render_manifests / render_loader
  deploy/base.py         DeploymentTarget ABC + shared helpers (wait_healthy)
  deploy/local.py        LocalTarget (detached `evalbuilder serve` subprocess)
  deploy/docker_compose.py  DockerComposeTarget
  deploy/kubernetes.py   KubernetesTarget (build, push|load, apply, rollout, port-forward|nodeport)
  deploy/openshift.py    OpenShiftTarget(KubernetesTarget): oc, route, registry push
  pipeline/config.py     DeployConfig, InferenceConfig, EvaluationConfig, migration, STAGE_NAMES, template
  pipeline/layout.py     `deployment` kind
  pipeline/stages.py     deploy / infer / simulate / teardown stages, preflight target checks
  pipeline/report.py     mocking summary reads infer stage; deployment in report
  pipeline/jobs.py       run_progress unchanged; stage names
  pipeline/setup.py      form fields for deploy/inference/evaluation
  cli.py                 serve, deploy (up/status/down/build/render), infer (+run alias), eval (+score alias)
  ui/app_pages/setup.py  Deployment + Parallelism form blocks
  ui/app_pages/run.py    Deployment status block + Tear down
  ui/loader.py           bundle.deployment, run_ids from infer|run
  ui/common.py           _run_stage_active → infer
examples/weather_bot/pipeline.yaml   docker-ready offline config
tests/test_agent_client.py, test_serve.py, test_inference.py, test_deploy.py, test_deploy_integration.py,
tests/test_cli_phases.py; updates in test_pipeline_config/test_layout/test_evaluators/test_runner/
test_simulate/test_pipeline_e2e/test_e2e_small_agents/test_ui_*/tests/ui/test_playwright_flows.py
docs: README.md, docs/deployment.md, docs/architecture/{02,03,04,05,06,07}.md, skills (pipeline, run), config-reference.md, scripts/cli-smoke.sh
```

---

### Task 1: Config sections, migration, stage names, template

**Files:**
- Modify: `src/evalbuilder/pipeline/config.py`
- Test: `tests/test_pipeline_config.py`

**Interfaces (produces):**
```python
class ImageConfig(BaseModel):
    name: str | None = None; tag: str = "latest"; registry: str | None = None
    push: Literal["auto", "registry", "load", "none"] = "auto"; extras: list[str] = []
class BuildConfig(BaseModel):
    context: str = "."; dockerfile: str | None = None; include: list[str] = []; requirements: str | None = None
class DeployConfig(BaseModel):
    target: Literal["local", "docker", "kubernetes", "openshift"] = "local"
    image: ImageConfig; build: BuildConfig; port: int = 8080; namespace: str = "default"; replicas: int = 1
    expose: Literal["port-forward", "nodeport", "route"] = "port-forward"; env: dict[str, str | None] = {}
    keep: bool = False; timeout: int = 240; context: str | None = None
class InferenceConfig(BaseModel): workers: int = 4; backend: Literal["threads", "processes"] = "threads"; timeout: int = 120
class EvaluationConfig(BaseModel): workers: int = 4; backend: Literal["threads", "processes"] = "threads"
class RunsConfig(BaseModel): repeats: int = 3
PipelineConfig.deploy / .inference / .evaluation
PipelineConfig.image_ref -> str   # "<registry>/<name>:<tag>" or "<name>:<tag>", name defaults to evalbuilder-<name>
PipelineConfig.container_target -> bool   # target != local
STAGE_NAMES = ("preflight","discover","map","mocks","dataset","review","verify","deploy","infer","simulate","teardown","score","aggregate","publish","analyze","report")
migrate_config(data: dict) -> tuple[dict, list[str]]   # old keys → new sections, notes
```

- [x] Write tests: defaults (`deploy.target == "local"`, `inference.workers == 4`), `image_ref` with/without registry, migration of `runs.parallel_*` (notes returned, `runs.repeats` kept), `problems()` rejects `inference.workers < 1`, `deploy.replicas < 1`, `deploy.port` out of range, `openshift` without `image.registry` under `push: registry`; template contains `deploy:` and `inference:`; `to_yaml` round-trips `deploy`.
- [x] Run tests → fail. Implement. Run → pass. Commit.

### Task 2: Schemas and artifact layout

**Files:** `src/evalbuilder/schemas.py`, `src/evalbuilder/pipeline/layout.py`, tests `tests/test_layout.py`.

**Produces:** `CaseRun.log: list[dict]`, `RunArtifact.execution: dict | None`; `ARTIFACTS["deployment"]` = `ArtifactKind("deployment", "deployment.json", "evalbuilder/deployment/v1", "deploy, teardown", …)`; `DEPLOYMENT_SCHEMA` constant in `deploy/spec.py` (Task 7) equals that id.

- [x] Tests: `kind_of_file("deployment.json")`, schema lookup, `convention_table` row, `artifact_index` finds it. Implement, commit.

### Task 3: Agent clients

**Files:** Create `src/evalbuilder/agent_client.py`; modify `src/evalbuilder/target.py` (extract `invoke_messages(graph, messages) -> (state, node_path)` and `extract(state)`); test `tests/test_agent_client.py`.

**Produces:**
```python
@dataclass
class InvokeResult:
    messages: list[dict]; response: str; tool_calls: list[dict]; node_path: list[str]
    mock_calls: list[dict]; error: str | None; error_class: str; seconds: float
    def to_dict(self) -> dict; @classmethod from_dict(cls, d) -> "InvokeResult"

class LocalAgent:
    def __init__(self, module: str, factory: str = "build_agent", *, agent_model_spec=None, mock_model_spec=None, agent_model=None, mock_model=None)
    mode = "local"; endpoint = None
    def health(self) -> dict
    def invoke(self, messages: list[dict], mocks: dict | None = None, *, mocked: bool = True) -> InvokeResult
class RemoteAgent:
    def __init__(self, endpoint: str, timeout: float = 120.0); mode = "remote"
    def health(self) -> dict; def invoke(...) -> InvokeResult
def mocks_for_case(ds_mocks: dict, case_meta: dict, *, on_miss=None, strategy=None) -> dict   # merged block sent per request
```
`mocks` block keys: `tools` (merged rules), `on_miss`, `strategy`, `strategies`, `llm` (model, on_invalid, max_repairs). `LocalAgent.invoke` builds rules → engine (if `on_miss == "llm"`; mock model from `mocks.llm.model` spec, else the instance's) → `wrap_tools` → `build_graph` → `invoke_messages` → `InvokeResult`; exceptions become `error`/`error_class` (`infrastructure` for factory/mock-engine failures, else `agent`).

- [x] Tests: local invoke with rules on weather_bot returns tool call + response; multi-turn by re-sending `result.messages + [user]`; strict miss → `error_class == "agent"`; llm policy with `scripted:examples.weather_bot.offline:mock_model` answers via engine and ledger layer `llm`; `mocks_for_case` merges case rules over dataset rules and picks case strategy; pickling `LocalAgent`/`RemoteAgent` works (`pickle.dumps`).

### Task 4: Agent server + `evalbuilder serve`

**Files:** Create `src/evalbuilder/serve.py`; modify `src/evalbuilder/cli.py`; test `tests/test_serve.py`.

**Produces:**
```python
def make_server(agent: LocalAgent, host="127.0.0.1", port=0) -> ThreadingHTTPServer   # .server_address gives the bound port
def serve(module, factory="build_agent", host="0.0.0.0", port=8080, agent_model=None, mock_model=None) -> None  # blocking
SERVER_VERSION = "evalbuilder/serve/v1"
```
Routes: `GET /health` → `agent.health()` + `{"ok": True, "server": SERVER_VERSION}`; `POST /invoke` → JSON body `{messages, mocks, mocked}` → `InvokeResult.to_dict()`; 400 on bad JSON / missing messages; 404 otherwise. CLI: `evalbuilder serve --module M [--factory F] [--host H] [--port P] [--agent-model S] [--mock-model S]` (env fallbacks `EVALBUILDER_AGENT_MODEL`, `EVALBUILDER_MOCK_MODEL`).

- [x] Tests: start `make_server` in a daemon thread; `/health` lists tools; `/invoke` with rules; multi-turn via returned messages; `RemoteAgent` against it equals `LocalAgent` result fields (tool_calls, response); bad body → 400; CLI `serve --help` mentions `--module`.

### Task 5: Inference engine (joblib) + runner/simulate refactor

**Files:** Create `src/evalbuilder/inference.py`; modify `runner.py` (wrapper), `simulate.py` (step callable); tests `tests/test_inference.py`, update `tests/test_runner.py`, `tests/test_simulate.py`.

**Produces:**
```python
def infer_cases(agent, ds: Dataset, cases: list[Case], *, workers=1, backend="threads", on_miss=None, strategy=None, mocked=True, progress=None) -> list[CaseRun]
def infer_dataset(ds, dataset_path, agent, *, out_dir, ids=None, workers=1, backend="threads", on_miss=None, strategy=None, mocked=True, progress=None, model_spec=None, mock_model_spec=None) -> RunArtifact
def simulate_scenarios(agent, scenarios: list[dict], *, mocks: dict | None, user_model_spec=None, user_model=None, workers=1, backend="threads", on_miss=None, strategy=None) -> list[dict]
def parallel(workers, backend) -> joblib.Parallel   # n_jobs=max(1, workers), prefer=…, return_as="generator"
# simulate.py
def simulate_scenario(step: Callable[[list[dict]], str] | Any, scenario, user_model=None) -> dict   # graphs still accepted (adapter)
def graph_step(graph) -> Callable
```
`CaseRun.log` entries: `{"turn": i, "seconds": s, "tool_calls": n, "error": e, "mode": agent.mode, "endpoint": agent.endpoint}`. Scenario results gain `"log"` (same shape) and keep `mock_calls`, `mock_strategy`. `RunArtifact.execution = {"mode", "endpoint", "workers", "backend", "seconds"}`. `runner.run_dataset(...)` keeps its signature (`max_workers` → `workers`), builds a `LocalAgent(module, factory, agent_model=model, mock_model=mock_model)` and calls `infer_dataset` — existing tests must pass unchanged except `max_workers` semantics (per case now).

- [x] Tests: threads and processes backends produce identical artifacts to sequential (weather_bot scripted; loky needs `scripted:` specs, so use `LocalAgent(agent_model_spec=…)` for the processes test); dataset order kept; progress events count; per-case errors isolated; multi-turn (`user_turns`) replays 2 user messages; `log` has one entry per turn; scenarios through `LocalAgent` reach `success` with followups; `simulate_scenario(graph_step(graph), …)` still works.

### Task 6: Parallel scoring, `eval` and `infer` commands

**Files:** `src/evalbuilder/evaluators.py`, `src/evalbuilder/cli.py`; tests `tests/test_evaluators.py` (add), `tests/test_cli_phases.py`.

**Produces:** `score_run(run, ds, specs, judge_model, workers=1, backend="threads", max_workers=None)` (`max_workers` kept as alias); chunking `_chunks(items, n)` contiguous; `_score_chunk(run_json, ds_json, specs, judge_model, ids)` rebuilds evaluators in the worker. CLI:
- `evalbuilder infer DATASET [--endpoint URL] [--deployment DIR] [--local/--no-local] [--scenarios F] [--repeats N] [--workers N] [--backend B] [--out D] [--ids] [--mock/--no-mock] [--model S] [--on-miss P] [--mock-model S] [--strategy S] [--mine/--no-mine]` → JSON `{runs: [{run_id, path, cases, errors}], simulation: {sim_id, path, …} | null, endpoint, mode}`; `run` = hidden alias.
- `evalbuilder eval RUN... --dataset D --evaluators F [--out D] [--workers N] [--backend B] [--aggregate/--no-aggregate] [--config CFG] [--env-file F]` → score reports + `aggregate.json`; `score` = alias for one run printing the report.

- [x] Tests: parallel scores == sequential (threads, processes) on a 5-case run; CLI `infer` local with scenarios writes run + simulation files; CLI `eval` with two runs writes aggregate with verdict; `run`/`score` aliases still work.

### Task 7: Deployment package

**Files:** Create `src/evalbuilder/deploy/{__init__,spec,runner,render,base,local,docker_compose,kubernetes,openshift}.py`; test `tests/test_deploy.py`.

**Produces:**
```python
# spec.py
DEPLOYMENT_SCHEMA = "evalbuilder/deployment/v1"
@dataclass class DeploymentSpec: name, target, image, push, extras, build_context, dockerfile, include, requirements, port, namespace, replicas, expose, env (resolved), keep, timeout, context, module, factory, agent_model, mock_model, work_dir (Path), python_version="3.12"
def spec_from_config(cfg: PipelineConfig, out_dir: Path | None = None, target: str | None = None) -> DeploymentSpec
def resolve_env(env: dict[str, str | None]) -> dict[str, str]   # None → os.environ value (skipped when unset)
@dataclass class DeploymentRecord: schema, name, target, image, endpoint, expose, status ("up"|"down"|"failed"), resources: dict, created_at, updated_at, commands: list[dict], details: dict; to_dict/from_dict; save(path)/load(path)
# runner.py
class CommandRunner:  def __init__(self, log_path: Path | None = None, log=None); def run(self, argv, *, input: bytes | None = None, check=True, timeout=None, env=None) -> CompletedProcess; def which(self, binary) -> str | None
# render.py
def render_dockerfile(spec) -> str; def render_compose(spec) -> str; def render_manifests(spec, image_pull_policy) -> str; def render_loader(spec) -> str; def package_dir(module) -> str
# base.py
class DeploymentTarget: kind; def __init__(self, runner: CommandRunner, log=None); available(); render(spec); build(spec) -> str; up(spec) -> DeploymentRecord; status(spec, record) -> dict; logs(spec, record, lines=100) -> str; down(spec, record) -> None
def wait_healthy(endpoint, timeout, interval=1.0, sleep=time.sleep) -> dict   # RemoteAgent.health() polling
# __init__.py
TARGETS = {"local": LocalTarget, "docker": DockerComposeTarget, "kubernetes": KubernetesTarget, "openshift": OpenShiftTarget}
def target_for(kind, runner=None, log=None) -> DeploymentTarget
def record_path(out_dir) -> Path; def load_record(out_dir) -> DeploymentRecord | None
def deploy_up(cfg, out_dir=None, *, target=None, runner=None, log=None) -> DeploymentRecord   # writes deployment.json
def deploy_status(cfg_or_dir, …) -> dict; def deploy_down(cfg_or_dir, …) -> dict
```
Kubernetes: names `evalbuilder-<name>`; `load` streams `docker save` (bytes from `runner.run(["docker","save",image])`) into `kubectl exec -i <pod> -- nsenter -t 1 -m -u -i -n -- ctr -n k8s.io images import -` per loader pod; port-forward = detached `subprocess.Popen` via `runner.spawn(argv, log_path) -> pid` (so the fake can stub it); record `resources = {namespace, deployment, service, loader, port_forward: {pid, local_port}}`. OpenShift: `cli="oc"`, `push` default `registry`, `expose` default `route`, `available()` = `oc whoami`.

- [x] Tests (FakeRunner records argv and returns canned stdout per prefix): renderers (Dockerfile has `evalbuilder serve --module …`, compose ports/env, manifests have probe + policy, route only for openshift); docker target argv sequence and endpoint; kubernetes `load` path (loader apply, save→import per pod, apply, rollout, port-forward spawned, record); `registry` path (`docker tag`/`push`, policy Always); nodeport endpoint from `kubectl get nodes` json; `down` argv; openshift route endpoint and missing-registry error; `LocalTarget` real: up → health ok → status ready → down kills.

### Task 8: `evalbuilder deploy` commands

**Files:** `src/evalbuilder/cli.py`; tests in `tests/test_cli_phases.py`.

- `deploy up CONFIG [--target T] [--keep] [--env-file F]`, `deploy status CONFIG|DIR`, `deploy down CONFIG|DIR`, `deploy build CONFIG`, `deploy render CONFIG [--write]`. JSON out; exit 1 when the target is unavailable or readiness times out.
- [x] Tests with the local target: up writes `deployment.json` with endpoint; status ready; down; render prints file names and contents.

### Task 9: Pipeline stages, report, jobs, examples

**Files:** `src/evalbuilder/pipeline/{stages,report,jobs,setup}.py`, `src/evalbuilder/ui/loader.py` (run_ids), examples `*/pipeline.yaml`, `examples/weather_bot/pipeline.yaml` (new); tests `tests/test_pipeline_e2e.py`, `tests/test_e2e_small_agents.py`, `tests/test_pipeline_engine.py`, `tests/test_jobs.py`, `tests/test_ui_loader.py`.

- `preflight`: `target_for(cfg.deploy.target).available()` blocking; `claude-cli:` agent/mock spec under a container target → error; `deploy.env` names not set on host → problem.
- `deploy` stage: `deploy_up(cfg, runner=CommandRunner(work/deploy/commands.log))` → details `{target, image, endpoint, expose, seconds}`; artifact `deployment`.
- `infer`: `RemoteAgent(record.endpoint, cfg.inference.timeout)`, `health()` check (restart port-forward via `deploy_status` when dead), repeats × `infer_dataset(... workers=cfg.inference.workers, backend=…)`, progress file as today + `endpoint`, `workers`, `backend`.
- `simulate`: scenarios through `inference.simulate_scenarios(RemoteAgent, …, mocks=ds.mocks, user_model_spec=cfg.models.generator, workers, backend)`; keep `user_model` object path for FakeGenerator in tests.
- `teardown`: skipped by config when `deploy.keep`; else `deploy_down`; optional stage, deps `("deploy",)`, runs after `simulate`.
- `REQUIRED_STAGES` = preflight … verify, deploy, infer, score, aggregate. `report._mocking_summary` reads `infer` (fallback `run`); report adds `"deployment": record dict | None`.
- Stage order in `build_stages()`: … verify, deploy(deps verify), infer(deps deploy), simulate(deps infer? no — deps deploy+review; optional), teardown(deps deploy, optional, after simulate), score(deps infer), aggregate, publish, analyze, report.
- [x] Tests: offline e2e statuses include `deploy/infer/teardown: ok`, `deployment.json` at root with `status: down`, `execution.mode == "remote"`; `--until deploy` leaves the server up (health ok) and `deploy down` stops it; `deploy.keep: true` skips teardown then explicit down; migration of an old config with `runs.parallel_intents` loads with a recorded problem.

### Task 10: UI

**Files:** `src/evalbuilder/ui/app_pages/setup.py`, `run.py`, `stages.py`, `summary.py`, `src/evalbuilder/ui/common.py`, `loader.py`, `pipeline/setup.py`; tests `tests/test_ui_pipeline_pages.py`, `tests/test_ui_pages.py`, `tests/ui/test_playwright_flows.py`.

- Setup: *Deployment* block (`setup_deploy_target` selectbox local/docker/kubernetes/openshift, `setup_image_name`, `setup_image_registry`, `setup_namespace`, `setup_expose`, `setup_keep`), *Parallelism* (`setup_infer_workers`, `setup_infer_backend`, `setup_eval_workers`). `default_form`/`form_from_config`/`build_config` round-trip.
- Run & review: `_deployment_section(out_dir)` — record summary, `deploy_status` (cached per render), *Tear down* button (`run_teardown`), rendered files path; stage progress follows `infer`.
- Stages page: artifacts list shows deployment; Summary: one line "deployed to <target> at <endpoint>".
- [x] Tests: AppTest setup form has the new widgets and YAML contains `deploy:`; Run page with a fabricated `deployment.json` shows the endpoint and a Tear down button; Playwright: setup shows the Deployment target select and after a local-target full run the Run page shows "Deployment" with `status down`.

### Task 11: Docs and smoke script

**Files:** `README.md`, `docs/deployment.md` (new), `docs/architecture/02-cli.md`, `03-core-modules.md`, `04-pipeline.md`, `05-ui.md`, `06-decisions.md` (decisions 22–24), `07-limitations.md`, `docs/architecture/README.md`, `skills/agent-eval-pipeline/SKILL.md`, `skills/agent-eval-pipeline/references/config-reference.md`, `skills/agent-eval-run/SKILL.md`, `scripts/cli-smoke.sh`.

- [x] Update every stage table, config block, CLI reference, repository layout, testing section, troubleshooting rows; write `docs/deployment.md` (targets, image distribution, endpoints, OpenShift prototype, env/keys, troubleshooting). Smoke script: `deploy up/status/down` local, `infer`, `eval`, artifact list with `deployment.json`.

### Task 12: Integration on Docker Desktop + memory

**Files:** `tests/test_deploy_integration.py` (marker `docker`), `pyproject.toml` markers.

- [x] Compose: `deploy up` on `examples/weather_bot/pipeline.yaml` (target docker, scripted models) → health → `infer` 2 cases → `down`. Kubernetes: same with `--target kubernetes` (load + port-forward). Run both locally; fix what breaks. Save memory notes.

## Execution notes (2026-09-06)

Executed inline in the same session, task by task, on branch `feat/deploy-infer-eval`. Verified for
real on Docker Desktop (Docker 29.4, kind-based Kubernetes 1.36): `evalbuilder pipeline run
examples/weather_bot/pipeline.yaml` (target docker) → verdict pass with deploy / infer / simulate /
teardown ok; `deploy up --target kubernetes` → loader import, port-forward, `infer --deployment`,
`deploy down`, cluster clean. The OpenShift target has no live cluster here (fake `oc` only). The
committed `docs/examples/incident-desk/report.json` is a `--until dataset` run (verdict
`incomplete`); the UI test now asserts what the loader promises rather than a pass/fail verdict.

## Self-review

- Spec §3 phases/commands → Tasks 6, 8, 9. §4 config → Task 1; interface/targets/render/server → Tasks 4, 7. §5 clients/engine/stages → Tasks 3, 5, 9. §6 → Task 6. §7 → Task 10. §8 → Task 2. §9 → each task + Task 12. Decisions 1–8 are embedded in Tasks 3–7.
- Names used consistently: `InvokeResult`, `LocalAgent`, `RemoteAgent`, `infer_cases`, `infer_dataset`, `simulate_scenarios`, `score_run(workers, backend)`, `DeploymentSpec`, `DeploymentRecord`, `CommandRunner`, `target_for`, `deploy_up/status/down`, stage names `deploy/infer/teardown`.
