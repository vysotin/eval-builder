# eval-builder

Agentic skills + a deterministic CLI that build and run evaluations for **LangGraph
agents**, an **autonomous pipeline** that does the whole job from one config file, and a
**Streamlit report UI** that visualises every artifact the pipeline produces.

- parse the agent's source and compiled graph into a test surface (graph, tools,
  prompts, **Agent Skills** — SKILL.md folders embedded in prompts or loaded on demand —
  intents, scenarios, failure modes, constraints) with cited evidence;
- generate a coverage-driven golden dataset in OpenEvals/LangSmith-compatible JSON;
- mock tools in **two layers**: deterministic ADK-style rules first, then — under
  `on_miss: llm` — an LLM mock engine that plays the backend from pre-generated
  strategies, validates every answer against the tool's output schema and logs which
  layer answered each call;
- **deploy** the agent — with both mock layers packed in — behind an HTTP server on a
  deployment target (`local` subprocess, `docker` Compose, `kubernetes` via kubectl,
  `openshift` via `oc` — prototype), **infer** against it in parallel (joblib threads or
  processes: repeated dataset runs and multi-turn simulations), then **evaluate** the stored
  runs locally (joblib splits): deterministic checks and LLM judges, pass rates vs
  thresholds, slices and stability, and a report with a `pass | fail | incomplete`
  verdict — three phases with their own commands (`evalbuilder deploy`, `infer`, `eval`);
- capture every tool's **input/output schema** (pydantic models, `args_schema`, return
  annotations) and derive deterministic **schema edge cases** — missing / mistyped /
  out-of-range input, malformed tool output — that become coverage cells;
- drive it all from the **UI**: initialise a config interactively (free-text rules and
  instructions included), run everything autonomously or stop after dataset + mocks,
  comment, regenerate, then approve and evaluate;
- use any model through one `provider:model` spec: the Claude Code CLI (subscription,
  Claude Sonnet 5 by default), or Anthropic / OpenAI / Gemini API keys.

```
                       ┌──────────────── skills (policy prose, Claude Code) ────────────────┐
your LangGraph agent ─►│ agent-eval-discover │ agent-eval-dataset │ agent-eval-mock │ agent-eval-run │
                       └──────────────────────────────┬──────────────────────────────────────┘
                                                      │ every artifact mutation goes through
                                                      ▼
                                   evalbuilder CLI (deterministic: schemas, IDs, coverage,
                                   review gate, mocking, runner, evaluators, LangSmith)
                                                      │
        one config ──► evalbuilder pipeline run ──────┤  preflight → discover → map → mocks → dataset →
                       (agent-eval-pipeline skill)    │  review → verify → deploy → infer×N → simulate →
                                                      │  teardown → score×N → aggregate → publish →
                                                      │  analyze → report
                                                      │
                          evalbuilder deploy up ──────┤  the agent + mock layers behind `evalbuilder serve`
                          evalbuilder infer     ──────┤  on local | docker | kubernetes | openshift;
                          evalbuilder eval      ──────┤  cases + scenarios over HTTP (joblib);
                                                      │  scoring + aggregation (joblib splits)
                                                      ▼
                                   eval/pipeline/<name>/   (agent-map.json, dataset.json, deployment.json,
                                   results/run-*.json, aggregate.json, report.json, …
                                   + work/ — scratch you can delete, incl. work/deploy/)
                                                      │
                       evalbuilder ui  ◄──────────────┘  Streamlit: Pipeline setup → Run & review →
                                                         report pages (graph, intents, dataset, coverage,
                                                         eval results, stability, simulation, analysis)
```

## Contents

1. [Install](#install)
2. [Quickstart](#quickstart)
3. [Repository layout](#repository-layout)
4. [The autonomous pipeline](#the-autonomous-pipeline)
   - [Config](#config) · [Stages](#stages) · [Review gate](#review-gate) ·
     [Resume, stop early, feedback loop](#resume-stop-early-feedback-loop) ·
     [Tool schemas and edge cases](#tool-schemas-and-edge-cases) · [Verdict and report](#verdict-and-report)
5. [Deployment: local, Docker, Kubernetes, OpenShift](#deployment-local-docker-kubernetes-openshift)
6. [Inference and evaluation phases](#inference-and-evaluation-phases)
7. [Models and providers](#models-and-providers)
8. [Artifacts and naming convention](#artifacts-and-naming-convention)
9. [UI: setup, run & review, report](#ui-setup-run--review-report)
10. [Interactive skills](#interactive-skills)
11. [CLI reference](#cli-reference)
12. [Dataset format, mocking, target contract](#dataset-format)
13. [Evaluators](#evaluators)
14. [LangSmith](#langsmith-optional)
15. [Testing](#testing)
16. [Troubleshooting](#troubleshooting)
17. [Extending](#extending-to-other-frameworks) · [Architecture guide](docs/architecture/README.md) ·
    [Deployment walkthrough](docs/deployment.md)

## Install

Requires Python ≥ 3.11.7 (the suite runs on 3.11 with plain pip and on 3.12 with uv). Two equivalent set-ups:

**With [uv](https://docs.astral.sh/uv/)**

```bash
uv venv --python 3.12
uv sync                              # core: langgraph, langchain, openevals, agentevals, typer, claude-cli adapter
uv sync --extra ui                   # + streamlit, pandas  (report UI)
uv sync --extra llm                  # + langchain-anthropic, langchain-openai, langchain-google-genai
uv sync --all-extras                 # everything
uv run playwright install chromium   # only for the browser tests
cp .env.example .env                 # model defaults and API keys (all optional)
uv run evalbuilder check             # capability matrix: langgraph, claude CLI, providers, judge, langsmith
```

**With a plain Python and pip** (no uv available)

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # any Python ≥ 3.11.7; Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .                     # core
pip install -e ".[ui]"               # + report UI
pip install -e ".[ui,llm,dev]"       # + API providers + pytest / playwright (the `dev` extra mirrors uv's dev group)
playwright install chromium          # only for the browser tests
cp .env.example .env
evalbuilder check                    # the console script; `python -m evalbuilder.cli check` is equivalent
```

Every `uv run <cmd>` in this README is `<cmd>` inside the activated virtualenv (the
pipeline's background jobs and the UI always spawn the interpreter they run in, never
`uv`). Extras: `anthropic`, `openai`, `gemini` (one provider each), `llm` (all three),
`ui`, `dev`. The Claude Code CLI adapter needs the `claude` binary on `PATH` and a
logged-in subscription; nothing else needs a key.

## Quickstart

```bash
# 1. a starter config for the shipped example agent (router graph + external tools)
uv run evalbuilder pipeline init eval/pipeline.yaml --name support-bot \
    --source examples/support_bot/agent.py --module examples.support_bot.agent

# 2. edit eval/pipeline.yaml: constraints, coverage, evaluators, thresholds,
#    models, deploy.target (local | docker | kubernetes | openshift), inference/evaluation workers,
#    and — to let the pipeline approve generated cases — review.auto_approve + approved_by

# 3. run (exit code 1 unless verdict == pass); ~45 min for 16 cases × 2 repeats with claude-cli:claude-sonnet-5
uv run evalbuilder pipeline run eval/pipeline.yaml
#    …or only generate the dataset + mocks, look at them, then continue:
uv run evalbuilder pipeline run eval/pipeline.yaml --until dataset
uv run evalbuilder pipeline run eval/pipeline.yaml --resume
#    …or drive the three phases yourself (the pipeline's deploy → infer → score stages, by hand):
uv run evalbuilder deploy up eval/pipeline.yaml                          # agent + mock layers behind an endpoint → deployment.json
uv run evalbuilder infer eval/pipeline/support-bot/dataset.json --deployment eval/pipeline/support-bot \
    --scenarios eval/pipeline/support-bot/scenarios.yaml --repeats 2 --workers 4 --out eval/results
uv run evalbuilder eval eval/results/run-*.json --dataset eval/pipeline/support-bot/dataset.json \
    --evaluators eval/pipeline/support-bot/evaluators.yaml --out eval/results     # score reports + aggregate.json
uv run evalbuilder deploy down eval/pipeline/support-bot

# 4. read it
uv run evalbuilder pipeline report eval/pipeline/support-bot     # terminal summary
uv run evalbuilder ui eval/pipeline/support-bot                  # Streamlit report at http://localhost:8501
```

Prefer clicking? `uv run evalbuilder ui` → **Pipeline setup** does steps 1–3 interactively
(choose the project — an existing output folder or a new one from an example agent —
discover its tools and schemas, fill the form including free-text rules, edit the YAML,
run everything or stop after the dataset), **Run & review** shows the job live and hosts
the comment → regenerate → approve → evaluate loop; every other page shows that project.

No key, no `claude` binary? Look at a finished run right away — the repository ships
the artifacts of real runs: `uv run evalbuilder ui docs/examples/support-bot`. Or run the
smallest example end to end, fully offline, with the agent in a container:
`uv run evalbuilder pipeline run examples/weather_bot/pipeline.yaml` (scripted models,
`deploy.target: docker` — needs only a Docker daemon; set `kubernetes` for the Docker
Desktop cluster, `local` for no Docker at all).
Ready-to-run configs sit next to each example agent (`examples/*/pipeline.yaml`); three
example agents ship with several subagents and external (mocked) tools:
`support_bot` (router + 2 specialists), `incident_desk` (classifier + 4 specialists,
pydantic in/out models, `args_schema`), `loan_desk` (router + 3 chained specialists,
nested pydantic models, consent/confirmation gates).

## Repository layout

```
src/evalbuilder/
  cli.py              typer CLI: dataset/agent-map/mock/review/discover/run/score/simulate/publish/check/pipeline/ui
  schemas.py          pydantic models: AgentMap, Dataset/Case, RunArtifact, Report (score report)
  artifacts.py        dataset load/save/validate, content-hash case IDs, review state machine
  discover.py         AST + live introspection of a LangGraph module → agent-map (tools, prompts, skills)
  skills.py           Agent Skills: SKILL.md loading, listing / inline prompts, the load_skill tool, map entries
  target.py           target contract: build_agent(model=None, tools=None) + TOOLS
  mocking.py          layer 1: ADK-style rules (ordered, matchArgs subset, miss policy), ledger, verify
  mock_engine.py      layer 2: LLM mock engine — strategies, schema validation, repair, fallback/strict
  tool_schemas.py     tool args/output schemas (pydantic, args_schema, annotations), JSON-schema-subset
                      validator, sample/corrupt payloads, deterministic schema edge cases
  agent_client.py     one invoke(messages, mocks) contract: LocalAgent (in-process graph per call) and
                      RemoteAgent (HTTP) — both picklable for joblib process workers
  serve.py            the agent server: stdlib HTTP, GET /health + POST /invoke (container / local-target entry point)
  inference.py        the inference phase: cases + scenarios through an agent client with joblib, per-turn logs
  runner.py           run_dataset(): the in-process convenience over inference (LocalAgent)
  evaluators.py       deterministic evaluators + OpenEvals/AgentEvals judges, score_run (joblib splits)
  simulate.py         multi-turn scenario loop over a step callable, violation mining
  deploy/
    spec.py           DeploymentSpec (from the config) + DeploymentRecord (deployment.json)
    runner.py         CommandRunner: every docker / kubectl / oc call, logged to work/deploy/commands.log
    render.py         generated Dockerfile, compose.yaml, Kubernetes manifests (+ Route), image-loader DaemonSet
    base.py           DeploymentTarget interface (available / render / build / up / status / logs / down)
    local.py          `evalbuilder serve` as a detached subprocess (no Docker)
    docker_compose.py Docker Compose on the local daemon
    kubernetes.py     kubectl: image push | load into the nodes, apply, rollout, port-forward | NodePort
    openshift.py      oc prototype: registry push + Route
  coverage.py         intent×topic×scenario×failure_mode grid, gaps
  langsmith_io.py     idempotent publish with read-back verification
  providers.py        API-key providers (anthropic/openai/google_genai), aliases, readiness, @effort
  claude_cli.py       ChatClaudeCLI: Claude Code CLI as a LangChain chat model; model_from_spec
  config.py           .env settings, capability_check
  testing.py          ScriptedChatModel + SchemaScriptedModel (offline generators) for fully offline runs
  pipeline/
    config.py         evalbuilder/pipeline-config/v1 (pydantic), template, semantic checks
    engine.py         stage runner: deps, retries, skip, awaiting_review, persisted work/state.json
    stages.py         the 16 stage functions (incl. deploy / infer / teardown) + PipelineContext
    generator.py      structured-output LLM "author" for map/mocks/cases/review/scenarios/analysis
    planning.py       coverage cells planning and achieved coverage
    taxonomy.py       failure-type gating by agent structure
    aggregate.py      repeat-aware aggregation: pass rates, thresholds, slices, stability
    report.py         report assembly + run_pipeline entry point
    layout.py         THE artifact naming convention (kinds, files, schema ids, legacy names)
    jobs.py           background pipeline jobs for the UI (work/job.json + pipeline.log, wrapper entry point)
    setup.py          interactive setup helpers: target discovery/preview, config from form, feedback/approval
  ui/
    app.py            Streamlit entry point (st.navigation, sidebar showing the project)
    project.py        THE project: the folder / uploads chosen on Pipeline setup that every page follows;
                      persistent setup form (survives navigation), bundle loading
    loader.py         Bundle: load a directory by file name or uploads by schema id
    common.py         palette, charts (Altair), tables, transcript widgets
    app_pages/        setup (pipeline setup), run (run & review), agent, intents, dataset, coverage,
                      results, stability, simulation, analysis, summary, stages
skills/               five Claude Code skills (symlinked into .claude/skills/)
examples/             weather_bot, travel_planner, support_bot, incident_desk, loan_desk — each with a
                      scripted default model; weather_bot/support_bot/incident_desk/loan_desk ship pipeline.yaml
                      (weather_bot's is docker-ready and fully offline) and offline.py (a SchemaScriptedModel
                      generator + mock model for LLM-free pipeline runs)
docs/examples/support-bot/, incident-desk/   artifacts of real pipeline runs (UI demo + test fixtures)
docs/architecture/    how every skill, CLI command, module, the pipeline and the UI work; decisions; limitations
docs/superpowers/     design specs and implementation plans
docs/deployment.md    deployment walkthrough: targets, image distribution, endpoints, the HTTP protocol
tests/                offline pytest suite (+ tests/test_deploy_integration.py, marker `docker`: real Docker /
                      Kubernetes); tests/ui/ = Playwright browser tests (report pages + interactive flows)
```

## The autonomous pipeline

`evalbuilder pipeline run CONFIG` turns one YAML file into a verdict. An LLM
**generator** plays the author role the interactive skills otherwise give to a human:
it derives intents and scenarios (evidence-cited, taxonomy-gated), writes tool
fixtures, fills a coverage plan with cases, self-reviews them, writes multi-turn
simulation scenarios and the final analysis. Every artifact it produces still goes
through the same CLI validation, and the human review gate is preserved. The
evaluation itself runs in three phases — **deploy** the agent (with both mock layers)
behind an HTTP endpoint, **infer** against it, **evaluate** the stored runs — each also
available as its own command (`evalbuilder deploy`, `infer`, `eval`).

### Config

`evalbuilder pipeline init` writes a fully commented starter. Minimal config:

```yaml
schema: evalbuilder/pipeline-config/v1
name: support-bot
target:
  source: examples/support_bot/agent.py     # agent source (AST discovery)
  module: examples.support_bot.agent        # importable module exposing TOOLS + build_agent
models:
  agent: claude-cli:claude-sonnet-5         # injected into build_agent(model=…); omit = target's own
  judge: claude-cli:claude-sonnet-5         # LLM-as-judge
  generator: claude-cli:claude-sonnet-5     # authors intents/scenarios/cases/mocks/strategies/analysis
  mock: null                                # drives the LLM mock engine (mocking.on_miss: llm); null = generator
constraints:
  - "Never call issue_refund before the customer explicitly confirms with yes."
instructions: |                             # free-text general rules for the generator
  Customers write short, impatient messages. Prefer realistic order ids.
feedback: []                                # reviewer comments appended between runs (see the feedback loop)
coverage: {total_cases: 12, per_intent: {happy: 1, failure: 1}, per_failure_category: 1, out_of_intent: 2, multi_turn_share: 0.15, per_tool_edge_cases: 2}
evaluators: [{type: expected_tools}, {type: contains}, {type: contract}, {type: correctness}]
thresholds: {default: 0.7, metrics: {expected_tools: 0.8}, slice_min: 0.5, overall_pass: 0.75}
runs: {repeats: 2}                          # repeats only — parallelism lives in inference / evaluation
deploy:                                     # where the agent (with both mock layers) runs during inference
  target: local                             # local (subprocess) | docker (compose) | kubernetes (kubectl) | openshift (oc, prototype)
  image: {name: null, tag: latest, registry: null, push: auto, extras: []}   # name → evalbuilder-<name>; push: auto|registry|load|none
  build: {context: ".", dockerfile: null, include: [], requirements: null}
  port: 8080
  namespace: default
  replicas: 1
  expose: port-forward                      # kubernetes: port-forward | nodeport; openshift: route
  env: {}                                   # container environment; a null value copies the variable from the host (API keys)
  keep: false                               # leave the deployment running after the pipeline (skips teardown)
  timeout: 240                              # seconds to wait for readiness
  context: null                             # kubectl / oc context
inference: {workers: 4, backend: threads, timeout: 120}   # cases + scenarios against the deployed agent (joblib)
evaluation: {workers: 4, backend: threads}                # scoring splits of stored runs (joblib)
mocking: {required: true, on_miss: strict, strategies: true, strategy: default, on_invalid: fallback, max_repairs: 1}
review: {auto_approve: true, approved_by: "your name"}
```

`deploy.target` defaults to `local`, so a config without a `deploy` section runs the
agent as a subprocess of the pipeline (no Docker). The container targets reject
`claude-cli:` agent and mock models — the `claude` binary is not in the image; use an
API-key provider with the key passed through `deploy.env` (`ANTHROPIC_API_KEY: null`
copies it from the host) or a `scripted:` model. Configs written before this release
still load: `runs.parallel_intents` / `parallel_scoring` / `parallel_simulations` are
migrated into `inference.workers` / `evaluation.workers` and reported as a preflight
problem. Every key, default and semantic check is documented in
`skills/agent-eval-pipeline/references/config-reference.md`.

### Stages

| stage | depends on | what it does | on failure |
|---|---|---|---|
| preflight | – | config checks, target import, model readiness (`providers.provider_ready`) | blocking |
| discover | preflight | AST + live introspection → `agent-map.json` (nodes, edges, tools with `kind`/`mockable`, prompts, skills) | blocking |
| map | discover | generator authors intents, scenarios, failure scenarios, topics, derived constraints; taxonomy gates failure types | retry once; invalid entries dropped → `problems` |
| mocks | discover | generator writes fixtures for every **mockable** tool (no wildcard under `on_miss: llm`) and the LLM mock strategies → `work/`, then the dataset stage stores both in `dataset.mocks` | generic fixture / generic default strategy |
| dataset | map, mocks | plan coverage cells → generator fills them → `dataset.json` (cases, `mocks`, `coverage.plan` + `coverage.achieved`) | invalid cases dropped, gaps reported |
| review | dataset | self-review rejects bad cases; approves the rest **only** with `review.auto_approve` | stops with `awaiting_review` |
| verify | review | every expected tool call has a rule; every tool is mocked (under `llm`: misses are counted as engine-answered, a tool is covered by a rule or the default strategy) | blocking |
| deploy | verify | the agent + both mock layers behind an HTTP endpoint on `deploy.target` (image built, distributed, rolled out, `/health` polled) → `deployment.json`, rendered files under `work/deploy/` | blocking |
| infer | deploy | `runs.repeats` × every approved case against the endpoint, `inference.workers` in parallel (joblib) → `results/run-<id>.json` (per-case `mock_calls` ledger, per-turn `log`, `execution`, `mocking` totals) | retry once |
| simulate | deploy, review (optional) | multi-turn scenarios with a simulated user, through the same endpoint and pool → `simulation.json`; violations mined into pending cases | never blocks |
| teardown | deploy (optional) | `deploy down` — the deployment is removed before scoring; skipped when `deploy.keep` is true | a problem, never `incomplete` |
| score | infer | evaluators per run, `evaluation.workers` splits (joblib) → `results/score-report-<id>.json` | judge errors recorded per metric |
| aggregate | score | pass rates vs thresholds, slices, stability, failing cases → `aggregate.json` | – |
| publish | review (optional) | LangSmith upload (`auto` = only with a key) | skipped |
| analyze | aggregate (optional) | generator explains the verdict → `analysis.json` | deterministic fallback |
| report | always | `report.json` | – |

Failures never raise: a failed stage marks its dependents `skipped`, retries once with a
recovery note, and `report.json` always lands.

### Review gate

Generated and imported cases always land as `pending`. The pipeline approves them only
when the config says `review.auto_approve: true` **and** names `approved_by` — an
explicit human authorization recorded in every case's review note. Without it the run
stops at `awaiting_review`, writes the report, and you either approve by hand
(`evalbuilder review DATASET --approve ids…`) or flip the config and `--resume`.

### Resume, stop early, feedback loop

`work/state.json` records every stage. `--resume` reuses completed stages (the report is
always rebuilt); `--resume --from STAGE` regenerates from a stage on, e.g. after
editing thresholds (`--from aggregate`) or the agent (`--from run`).

`--until STAGE` stops deliberately after a stage (later stages are `skipped` with the
reason `stopped after <stage> (--until)`, the report still lands, exit code 0).
`--until deploy` leaves the agent running for manual probing (`evalbuilder deploy
status`, `curl <endpoint>/health`, `evalbuilder infer … --deployment DIR`); tear it
down with `evalbuilder deploy down DIR` or `--resume` to let the pipeline continue.
The typical loop:

```bash
evalbuilder pipeline run cfg.yaml --until dataset          # intents, mocks, cases — nothing runs yet
# read dataset.json (cases + both mock layers) or the UI, then comment in the config:
#   feedback:
#     - {at: 2026-08-28T10:00:00, note: "more multi-turn refund cases; EU order ids", from_stage: dataset}
evalbuilder pipeline run cfg.yaml --resume --from dataset --until dataset   # regenerate with the feedback
evalbuilder pipeline run cfg.yaml --resume                                 # approve (review gate) + evaluate
```

`instructions` (free text) and every `feedback` entry are appended to the generator's
prompts (map, mocks, cases, self-review, simulation scenarios) as `USER INSTRUCTIONS`
and `REVIEWER FEEDBACK`. Cases already approved or rejected by hand (`evalbuilder review`,
the UI) are kept: the review stage only judges `pending` cases and succeeds with
`already_reviewed` when none are left.

### Tool schemas and edge cases

Discovery records for every tool its `args_schema` (explicit pydantic `args_schema=`
class or the signature, with `$defs` for nested models), `output_schema` (return
annotation: pydantic model, TypedDict, dataclass or primitive), `schema_source`
(`args_schema` / `annotations` / `ast`), the pydantic `models` involved, and whether the
docstring declares it `side_effecting`. From those schemas `evalbuilder.tool_schemas`
derives deterministic **edge cases** (no LLM):

| kind | trigger | failure mode | expected behavior |
|---|---|---|---|
| `missing_required` | a required argument | `input_validation` | ask for the field, never call the tool with a placeholder |
| `wrong_type` | non-string type, `pattern`, `format`, nested model | `input_validation` | do not pass the malformed value; ask or explain |
| `out_of_enum` | `Literal` / `enum` | `input_validation` | offer the allowed values |
| `boundary` | `ge/le/gt/lt`, `min/max_length`, `min/max_items` | `input_validation` | refuse / adjust and explain the limit |
| `malformed_output` | output schema with required fields | `tool_error_handling` | say the result was incomplete, never fabricate |

`coverage.per_tool_edge_cases` (default 2) turns the first N per tool into `schema-edge`
coverage cells; the generator writes the user message and contract, and for
`malformed_output` the pipeline itself injects a per-case mock override that drops a
required field. Generated fixtures are validated against `output_schema` (non-conforming
defaults are replaced by a schema-conformant sample, bad variants dropped, all reported
in `problems`), and `expected_tools[].args` are type-checked against `args_schema`.

### Verdict and report

`verdict` is `incomplete` when any required stage (preflight … aggregate) did not
succeed; otherwise `pass` / `fail` from thresholds: every metric's pass rate ≥ its
threshold, every intent / failure-mode / variant slice ≥ `slice_min`, and the mean of
metric pass rates ≥ `overall_pass`. `report.json` carries `verdict_reasons`, per-stage
status, metrics, slices, coverage achieved vs planned, stability across repeats,
failing cases with judge comments, the simulation, the analysis, and `problems[]`.

Stability semantics (from repeats): an unstable **case** changed its tool trajectory;
an unstable **output** kept the trajectory but a deterministic metric flipped (wording
drift); an unstable **evaluator** kept the trajectory but a judge flipped. Judge
rationales that look like placeholders are listed as `suspect_judge_comments`.

## Deployment: local, Docker, Kubernetes, OpenShift

During inference the agent does not run inside the pipeline process: it runs behind an
HTTP server — `evalbuilder serve` — with both mock layers packed in, on a **deployment
target**. One interface (`src/evalbuilder/deploy/base.py`: `available` / `render` /
`build` / `up` / `status` / `logs` / `down`), four targets:

| `deploy.target` | needs | how the image gets there | endpoint |
|---|---|---|---|
| `local` | nothing — `python -m evalbuilder.cli serve` as a detached subprocess of the same interpreter | no image | `http://127.0.0.1:<free port>` |
| `docker` | `docker` on PATH, a running daemon, Compose v2 | `docker compose -p evalbuilder-<name> up -d --build --wait` from the generated `compose.yaml` | `http://127.0.0.1:<deploy.host_port>` (default: `deploy.port`) |
| `kubernetes` | `kubectl` reaching a cluster (Docker Desktop's included) + `docker` to build | `image.push`: `registry` (docker push, pods pull with `Always`), `load` (a privileged loader DaemonSet + `docker save … \| kubectl exec -i … nsenter -t 1 … ctr -n k8s.io images import -` into every node, `Never`), `none` (already on the nodes); `auto` = `registry` when `image.registry` is set, else `load` | `expose: port-forward` (default; a detached `kubectl port-forward svc/…`, restarted by `deploy status` when it died) or `nodeport` (`http://<node InternalIP>:<nodePort>`) |
| `openshift` (prototype) | `oc` logged in + `docker`; `image.registry` required | `docker push` to the registry (`oc registry info --public` names the internal one) | `expose: route` (default): the Route's host |

Everything a target runs is transparent: `evalbuilder deploy render CONFIG` prints the
generated files, `deploy up` writes them to `<out_dir>/work/deploy/` before use
(`Dockerfile`, `compose.yaml`, `manifests.yaml`, `loader.yaml`), every external command
lands in `work/deploy/commands.log` (and in the record's `commands`), the local server
and the port-forward log to `serve.log` / `port-forward.log` next to them. `deployment.json`
(`evalbuilder/deployment/v1`) at the output root records target, image, endpoint,
resources (compose project / namespace, names, pids), status (`up | down | failed`)
and the commands — `evalbuilder infer --deployment DIR` and the UI read it back.

```bash
uv run evalbuilder deploy render cfg.yaml --target kubernetes   # Dockerfile + manifests + loader, nothing runs
uv run evalbuilder deploy up cfg.yaml                            # build, distribute, roll out, wait for /health
uv run evalbuilder deploy status eval/pipeline/<name>            # ready? replicas, health, port-forward (exit 1 unless up)
uv run evalbuilder deploy logs eval/pipeline/<name>              # the agent's recent log lines
uv run evalbuilder infer DATASET --deployment eval/pipeline/<name> --scenarios scenarios.yaml --repeats 2
uv run evalbuilder deploy down eval/pipeline/<name>              # remove everything; the record says `down`
```

The generated image is `python:3.12-slim` + `pip install .[<image.extras>]` of this
checkout (`build.context`, `.dockerignore` keeps `.venv` / `.git` / `eval` out) + the
target's top-level package + `build.include` paths (+ `build.requirements`), running
`evalbuilder serve --module … --factory … --port …`; model specs reach the server as
`EVALBUILDER_AGENT_MODEL` / `EVALBUILDER_MOCK_MODEL`, API keys through `deploy.env`.
Bring your own `build.dockerfile` to replace it.

The protocol is two JSON routes — `GET /health` (module, factory, tools, model specs)
and `POST /invoke` with `{messages, mocks, mocked}` → `{messages, response, tool_calls,
node_path, mock_calls, error, error_class, seconds}` — and it is **stateless**: every
request carries the full conversation (OpenAI-format messages, tool calls included) and
the mock block to install (rules, policy, strategy, strategies, LLM engine settings, the
engine history of earlier turns), so replicas are interchangeable and no dataset lives in
the image. Verified on Docker Desktop's kind-based Kubernetes, where locally built images
are *not* visible to the nodes and NodePort / LoadBalancer services do *not* reach the
host: `load` + `port-forward` is the combination that works there. The OpenShift target
is exercised against a fake `oc` only. Full walkthrough: [docs/deployment.md](docs/deployment.md).

## Inference and evaluation phases

**Infer** (`src/evalbuilder/inference.py`, `evalbuilder infer`): every unit of work is
one conversation — a case (its first message plus each `metadata.user_turns` entry,
re-sent as the returned history plus the new turn) or a simulation scenario — run
through an agent client (`RemoteAgent` for an endpoint, `LocalAgent` in-process) under
`joblib.Parallel(return_as="generator")`: `inference.workers` workers, `backend`
`threads` (default; the work is waiting on the agent) or `processes` (loky; each
worker rebuilds the client from its specs). Results keep dataset order and are
identical whatever the backend; progress is reported case by case into
`work/run-progress.json`. Each `CaseRun` records a per-turn `log` (`turn`, `seconds`,
`tool_calls`, `error`, `mode`, `endpoint`) next to its trajectory and mock ledger, and
the run artifact an `execution` block (`mode`, `endpoint`, `workers`, `backend`,
`seconds`). `evalbuilder infer DATASET [--endpoint URL | --deployment DIR | in-process]
[--scenarios F] [--repeats N] [--workers N] [--backend B]` writes `run-<id>.json` per
repeat and `simulation-<id>.json`; `evalbuilder run` is the same command under its old
name.

**Eval** (`evaluators.score_run`, `evalbuilder eval`): the case runs of a stored run
artifact are split into `evaluation.workers` contiguous chunks scored concurrently
(evaluators are rebuilt in each worker, so `processes` works for judges too) and
reassembled in run order — the report is byte-identical to a sequential pass.
`evalbuilder eval RUN… --dataset D --evaluators F [--workers N] [--backend B] [--config
CFG] [--aggregate/--no-aggregate]` scores every run given and, with more than one run
(or `--aggregate`), writes `aggregate.json` with the thresholds of `--config` (defaults
otherwise); `evalbuilder score` stays as the single-run form.

## Models and providers

Every model in evalbuilder — the agent under test, the generator, the judges, the
simulated user — is a `provider:model[@effort]` spec resolved by
`evalbuilder.claude_cli.model_from_spec`:

| provider (aliases) | auth | package · extra | example | `@effort` |
|---|---|---|---|---|
| `claude-cli` | Claude Code CLI on `PATH`, subscription login | bundled (`langchain-claude-code-cli`) | `claude-cli:sonnet@high` | CLI effort |
| `anthropic` (`claude`) | `ANTHROPIC_API_KEY` | `langchain-anthropic` · `anthropic` | `anthropic:claude-sonnet-5` | extended thinking (1024 / 4096 / 16000 tokens) |
| `openai` | `OPENAI_API_KEY` | `langchain-openai` · `openai` | `openai:gpt-5@medium` | `reasoning_effort` |
| `google_genai` (`gemini`, `google`) | `GOOGLE_API_KEY` | `langchain-google-genai` · `gemini` | `gemini:gemini-2.5-pro` | ignored |
| `scripted` | – | – | `scripted:examples.support_bot.agent:default_scripted_model` | – |

- Set them in the pipeline config (`models.agent|judge|generator`), on the CLI
  (`evalbuilder run --model …`), or as defaults in `.env`
  (`EVALBUILDER_JUDGE_MODEL`, `EVALBUILDER_AGENT_MODEL`, `EVALBUILDER_GENERATOR_MODEL`).
- `evalbuilder check` prints per-provider readiness (`key`, `package`, `ready`, the
  extra to install); pipeline preflight fails early with the exact missing key or
  package rather than at the first network call.
- The `claude-cli` adapter (`src/evalbuilder/claude_cli.py`) runs one isolated
  `claude -p --output-format json --json-schema …` subprocess per call, flattens
  history into a transcript, and rides tool calls and structured output on
  `--json-schema`. ~3–10 s per agent call, ~15–20 s per judge call.
- API providers go through LangChain's `init_chat_model`, so tool calling and
  `with_structured_output` are native. The OpenEvals/AgentEvals judges accept any of
  them via `judge=`.

## Artifacts and naming convention

`src/evalbuilder/pipeline/layout.py` is the single registry of artifact kinds. Rules:
root artifacts are `<kind>.json` (or `.yaml`), per-run artifacts are
`results/<kind>-<run_id>.json`, every JSON artifact embeds
`"schema": "evalbuilder/<kind>/v1"`, and kinds / files / schema ids are unique — so a
file can be identified by name (directories) or by content (uploads).

**No artifact is a copy of part of another one.** Each kind has a *tier* saying where —
and whether — it is written:

- **final** — a deliverable, at the root of the output dir (or under `results/`);
- **work** — scratch in `<out_dir>/work/`: resume state, live progress, the UI job
  record, and the mocks→dataset handoff. Nothing downstream of the pipeline reads it,
  and deleting the whole directory loses nothing;
- **derived** — data that is *part of* another artifact and so has no file at all.

```
eval/pipeline/support-bot/
├── agent-map.json  dataset.json  evaluators.yaml  scenarios.yaml     ← deliverables
├── deployment.json  aggregate.json  simulation.json  analysis.json  report.json
├── results/run-<id>.json  results/score-report-<id>.json
├── pipeline.log
└── work/           state.json  run-progress.json  job.json           ← scratch
                    mock-rules.json  mock-strategies.json
                    deploy/  Dockerfile  compose.yaml  manifests.yaml  loader.yaml
                             commands.log  serve.log  port-forward.log
```

| file | schema | stage | contents |
|---|---|---|---|
| `agent-map.json` | `evalbuilder/agent-map/v1` | discover, map | graph (AST + live), tools + arg schemas (`kind`, `mockable`), prompts, skills, intents, scenarios (`skills`), failure scenarios, constraints, topics, `applicable_failures` |
| `dataset.json` | `evalbuilder/dataset/v1` | dataset, review | cases (inputs, references, metadata, mocks), review + publication state, `mocks` (both layers), `coverage` (plan + achieved) |
| `evaluators.yaml` | – | score | evaluator specs |
| `deployment.json` | `evalbuilder/deployment/v1` | deploy, teardown | target, image, endpoint, expose, status (`up \| down \| failed`), resources, the commands run, health facts |
| `results/run-<id>.json` | `evalbuilder/run/v1` | infer | per-case `inputs` (the case inputs plus `user_turns`), outputs, trajectory, tool calls, node path, errors, `mock_calls` ledger, per-turn `log`; `mocking` totals, `execution` (mode, endpoint, workers, backend) |
| `results/score-report-<id>.json` | `evalbuilder/score-report/v1` | score | per-metric stats, slices, per-case scores/comments/errors/skips |
| `aggregate.json` | `evalbuilder/aggregate/v1` | aggregate | pass rates vs thresholds, slices, weak slices, stability, failing cases, verdict |
| `scenarios.yaml` | – | simulate | multi-turn scenarios |
| `simulation.json` | `evalbuilder/simulation/v1` | simulate | transcripts, stop reasons, violations |
| `analysis.json` | `evalbuilder/analysis/v1` | analyze | summary, failure patterns, weak slices, recommendations, evaluator issues |
| `report.json` | `evalbuilder/pipeline-report/v1` | report | everything above, condensed, plus `artifacts` index |

Scratch (`work/`), and what holds the same data for good:

| file | schema | stage | contents |
|---|---|---|---|
| `work/mock-rules.json` | `evalbuilder/mock-rules/v1` | mocks | `tools`: tool → ordered rules (layer 1) — handed to the dataset stage, kept in `dataset.mocks.tools` |
| `work/mock-strategies.json` | `evalbuilder/mock-strategies/v1` | mocks | `world` + `strategies`: per strategy, each tool's behaviour, examples, fallback response (layer 2) — kept in `dataset.mocks.strategies` |
| `work/state.json` | `evalbuilder/pipeline-state/v1` | engine | stage status/timing/details, problems; drives `--resume` |
| `work/run-progress.json` | `evalbuilder/run-progress/v1` | infer | live per-case/per-intent progress of the infer stage (endpoint, workers, repeat) |
| `work/deploy/` | – | deploy | generated `Dockerfile`, `compose.yaml`, `manifests.yaml`, `loader.yaml`; `commands.log` (every docker / kubectl / oc call), `serve.log` (local target), `port-forward.log` |
| `work/job.json` (+ `pipeline.log`) | `evalbuilder/pipeline-job/v1` | UI | background run: mode, argv, pid, timing, exit code |

Derived — no file of their own, and the kind the UI still shows for them:

| kind | lives in | contents |
|---|---|---|
| `applicable_failures` | `agent-map.json` → `applicable_failures` | failure type → gating evidence |
| `coverage_plan` | `dataset.json` → `coverage.plan` | planned cells and summary |
| `coverage` | `dataset.json` → `coverage.achieved` | planned vs covered, by kind, multi-turn, gaps, `skills` / `uncovered_skills` |

Directories written before this layout still load everywhere — stand-alone
`coverage.json` / `applicable-failures.json` / root `state.json`, the pre-convention
names (`mocks.json`, `plan.json`, `results/report-*.json`, schema
`evalbuilder/report/v1`), and single-file uploads of any of them. To bring one onto the
current layout:

```bash
uv run evalbuilder pipeline compact eval/pipeline/support-bot --dry-run   # what would change
uv run evalbuilder pipeline compact eval/pipeline/support-bot             # fold + move
```

The standalone commands write with the same names (`evalbuilder score` →
`score-report-<id>.json`, `evalbuilder simulate` → `simulation-<id>.json`).

## UI: setup, run & review, report

```bash
uv sync --extra ui
uv run evalbuilder ui                              # choose the project on Pipeline setup (scans eval/pipeline/*, docs/examples/*)
uv run evalbuilder ui eval/pipeline/support-bot    # open one directly (it becomes the project)
uv run evalbuilder ui DIR --port 8600 --headless   # server only
uv run streamlit run src/evalbuilder/ui/app.py -- --dir DIR     # equivalent
```

The app is organised around one **project** — chosen only on *Pipeline setup* — and every
other page follows it: an output folder (existing results, or where a new run will write)
or uploaded artifacts (read-only). The sidebar shows the current project (name, folder,
config, job status, artifact counts) and a *Clear project* button; without a project every
page is empty and points back to the setup page.

**Pipeline setup** initialises a run interactively and keeps its state while you visit
other pages: **Project** (open an existing folder — its config is loaded back into the
form and YAML — or start a new one), **Target agent** (an example under `examples/`
exposing `TOOLS` + `build_agent`, or a typed source/module; its output folder becomes the
project), *Discover structure* (AST + live introspection, no LLM) shows nodes, tools,
schema sources, pydantic models, side effects and the derived edge cases; the form covers
models (Claude Sonnet 5 via claude-cli by default), constraints, **free-text general rules
and instructions**, coverage (incl. edge cases per tool), evaluators, thresholds, repeats,
mock policy, **Deployment** (target, image name, registry, namespace, expose, keep
running) and **Parallelism** (inference workers / backend, evaluation workers), review
approval and paths; *Generate YAML* fills an editable YAML editor;
*Validate* / *Save config* / *Generate dataset & mocks only* / *Run full pipeline*. When
the config file changes on disk (review feedback, approval, a job) the form reloads it.

**Run & review** follows the project's background job (`work/job.json` + `pipeline.log`,
stage table refreshed every 2 s), and after a dataset-only run shows the review loop:
dataset summary (statuses, failure modes, schema-edge cases, mocked tools), a comment box
whose text is appended to the config's `feedback`, *Save feedback & regenerate*
(`--resume --from map|mocks|dataset --until dataset`), and *Approve remaining cases & run
evaluation* (records the approver, rejects selected cases, `--resume`). A **Deployment**
block shows `deployment.json` (target, image, endpoint, health, served tools), a live
readiness check, and *Tear down* (`evalbuilder deploy down`) while the agent is still up
(`deploy.keep`, or a run stopped with `--until deploy`). *Open results* jumps to the
Summary page.

Report pages (all read the project's artifacts; uploaded files are identified by their
embedded `schema` id, falling back to the file name):

| page | shows |
|---|---|
| Summary (after Analysis) | verdict + reasons, where the agent was deployed (target, endpoint, image, status), overall score, metric pass rates vs thresholds, coverage, stability, stage timeline, analysis summary, problems |
| Agent graph & tools | the graph as Graphviz (AST edges or compiled/live edges, tool links), node prompts, tools with argument/output schemas, pydantic models, side effects and schema edge cases, constraints, topics |
| Intents & scenarios | intents with evidence and case counts, scenarios by intent (happy/failure), failure scenarios, structurally applicable failure types |
| Dataset & mocks | filterable case table (incl. schema-edge kind), case detail (inputs, references, metadata, edge, per-case mocks, review note), review counts, tool fixtures |
| Coverage | planned vs covered by kind, plan cells, gaps |
| Eval results | aggregate metrics vs thresholds, per-run metrics, slice heatmaps (intent / failure mode / variant), per-case score matrix across runs, failing cases with judge comments, case drill-down with trajectory transcript and scores per run |
| Stability | unstable cases / outputs / evaluators, suspect judge comments, per-case hashes |
| Simulation | scenario outcomes, transcripts, violations |
| Analysis | summary, verdict explanation, failure patterns, weak slices, recommendations, evaluator issues |
| Stages & problems | timeline, per-stage details and artifacts, problems, generator calls, artifacts loaded, config |

Missing artifacts are explained (which file, which stage writes it) rather than hidden.

## Interactive skills

The skills are policy prose for Claude Code (`skills/agent-eval-*/SKILL.md`, exposed via
`.claude/skills/`); all determinism lives in the CLI.

| skill | use when |
|---|---|
| `agent-eval-discover` | starting eval work / agent code changed — map the test surface into `eval/agent-map.json` |
| `agent-eval-dataset` | generating or extending the golden dataset (cases land `pending` → human review) |
| `agent-eval-mock` | cases depend on nondeterministic or side-effecting tools — rules (layer 1) and LLM mock strategies (layer 2) |
| `agent-eval-run` | running/scoring experiments, publishing to LangSmith, simulating multi-turn scenarios |
| `agent-eval-pipeline` | one config → full autonomous evaluation → `report.json` (+ reading it) |

## CLI reference

```
evalbuilder check [--target-module M] [--env-file F]      capability matrix incl. providers
evalbuilder discover MODULE [--source F] [--eval-dir D]   AST + live introspection → agent-map.json (tools, nodes, skills)
evalbuilder agent-map update MAP [--intents|--scenarios|--failures|--topics|--constraints JSON|@file]
evalbuilder dataset init|add|import|validate|list|gaps    dataset lifecycle (cases always land pending)
evalbuilder review DATASET --approve ids | --reject ids   the only way a case becomes approved
evalbuilder mock set|verify                               layer-1 rules; verify expected calls match (llm: misses only counted)
evalbuilder mock strategies DATASET [--set @f] [--model SPEC] [--on-miss P] [--case ID --strategy S]   layer 2
evalbuilder mock validate DATASET --tool T --response JSON   validate a response against the tool's output schema
evalbuilder mock try DATASET --tool T --args JSON [--strategy S] [--mock-model SPEC]   one call through both layers
evalbuilder deploy up CONFIG [--target local|docker|kubernetes|openshift]   deploy phase → deployment.json
evalbuilder deploy status|down|logs DIR|CONFIG            live status (exit 1 unless up) / tear down / agent log
evalbuilder deploy build|render CONFIG [--target T] [--write]   image only / print the generated files
evalbuilder serve --module M [--factory F] [--host H] [--port P] [--agent-model S] [--mock-model S]   the agent server
evalbuilder infer DATASET [--endpoint URL | --deployment DIR] [--scenarios F] [--repeats N] [--workers N]
                          [--backend threads|processes] [--timeout S] [--mock/--no-mock] [--model SPEC]
                          [--on-miss P] [--mock-model SPEC] [--strategy S] [--ids …] [--out D]   inference phase
evalbuilder run …                                         alias of infer (the pre-phase name)
evalbuilder eval RUN… --dataset D --evaluators evaluators.yaml [--out D] [--workers N] [--backend B]
                      [--aggregate/--no-aggregate] [--config CFG]                        evaluation phase
evalbuilder score RUN --dataset D --evaluators evaluators.yaml [--out D] [--workers N]   one run, prints its report
evalbuilder simulate DATASET --scenarios scenarios.yaml [--endpoint URL | --deployment DIR] [--workers N]
                             [--backend threads|processes] [--timeout S] [--no-mine] [--mock] [--on-miss P]
                             [--mock-model SPEC] [--out D]                       scenarios only
evalbuilder publish DATASET [--dataset-name N]            LangSmith, idempotent
evalbuilder pipeline init|run|report|compact              the autonomous pipeline
evalbuilder ui [DIR] [--port P] [--headless]              Streamlit report UI
```

Every command prints JSON to stdout (`pipeline run` prints its human summary to stderr).

## Dataset format

`evalbuilder/dataset/v1` — each case maps 1:1 to a LangSmith example and to OpenEvals
evaluator kwargs; everything harness-specific lives under `metadata`:

```json
{
  "id": "case-3fa1b2c4d5",
  "inputs": {"messages": [{"role": "user", "content": "Where is order ORD-1002?"}]},
  "reference_outputs": {
    "contains": "ORD-1002",
    "expected_tools": [{"name": "lookup_order", "args": {"order_id": "ORD-1002"}}],
    "forbidden_tools": ["issue_refund"],
    "contract": "States the order id with its status and ETA without inventing details."
  },
  "metadata": {
    "intent": "intent.order-status", "topic": "order status",
    "scenario": "scenario.order-status.happy", "failure_mode": "none",
    "variant": "happy", "source": "synthetic", "user_turns": ["And the ETA?"],
    "mocks": {"tools": {"lookup_order": [{"matchArgs": {"order_id": "ORD-1002"}, "response": {"status": "Shipped"}}]}}
  },
  "review": {"status": "pending", "note": ""},
  "publication": {"langsmith_example_id": null}
}
```

Case IDs are content hashes (inputs + intent/topic/scenario/failure mode), so re-generation
never duplicates. The coverage grid is intent × topic × scenario × failure_mode.

### Mocking

Two layers (`docs/tool-mocking.md` is the full walkthrough). **Layer 1** — ADK-eval
semantics: ordered per-tool rule lists, first match wins, `matchArgs` is a recursive
subset match, `{}` is a wildcard; per-case rules override dataset-level fixtures, and
the pipeline appends dataset fixtures as fallback to every per-case list. Miss policy
`real` (call the real tool), `fallback` (a canned value), `strict` (error — the pipeline
default), or `llm`. **Layer 2** — under `on_miss: llm`, an `LLMMockEngine` (one per
case / scenario, model `mocks.llm.model`) plays the backend from the dataset's
`mocks.strategies` (a shared world + per-strategy, per-tool behaviours, examples and a
fallback response), validates every answer against the tool's `output_schema`, repairs
once, and then falls back or errors (`on_invalid`). Cases and scenarios may select an
alternate strategy (`metadata.mocks.strategy`, `mock_strategy`). Every mocked call lands
in the run artifact's per-case `mock_calls` ledger, and the report attributes
instability that coincides with LLM-mocked calls (`stability.llm_mocked_unstable`).
Subagents exposed as tool functions are mocked like tools; skill loaders
(`load_skill`) are never mocked.

### Agent skills

An agent that uses Agent Skills — `skills/<name>/SKILL.md` folders with frontmatter
(`name`, `description`, `allowed-tools`), markdown instructions and `references/` —
declares them with `evalbuilder.skills` (`load_skills`, `skills_inline_prompt` to embed a
skill in a prompt, `skills_prompt` + `skill_loader_tool` for on-demand disclosure through
a `load_skill` tool). Discovery reads the folders, renders the composed prompts,
records `skills[]` in the agent map (instructions — full, or a deterministic summary when
longer than 1500 chars; references; the resolved `tools` the skill can use, with unknown
`allowed-tools` flagged; extracted `rules`; `used_by` nodes), gives every LLM node
`capabilities` lines (which skill it can call → which tools that reaches → what result it
achieves) and marks loader tools `kind: skill_loader`; the generator sees the skills,
every scenario lists the skills it exercises (`skill:<name>` evidence), the map stage adds
per-skill `failure_cases` (beyond tool failure) and per-tool `failure_scenarios`,
`skill_misuse` becomes an applicable failure type, and coverage reports cases per skill. `incident_desk` embeds
its skills inline; `support_bot` reads them on demand.

### Target contract

The dataset's `target` names a module exposing
`build_agent(model=None, tools=None) -> CompiledStateGraph` and a `TOOLS` list. The
agent server (and the in-process `LocalAgent`) rebuilds the graph per request, wrapping
`TOOLS` with the mock block the request carries, and injects `model=` when one is
configured (`EVALBUILDER_AGENT_MODEL` in the container, `--model` in-process). Five example targets ship in `examples/`
(`weather_bot`, `travel_planner`, `support_bot`, `incident_desk`, `loan_desk`) with
scripted chat models, offline generators and offline mock models
(`offline.py::mock_model`), so the test suite — including the LLM mock layer — runs
offline.

## Evaluators

One metric per evaluator; evaluator failures are never agent failures. Deterministic:
`expected_tools` (ordered subset of tool calls, `forbidden_tools`), `contains`,
`json_valid`, `trajectory_match`. LLM judges (OpenEvals / AgentEvals): `correctness`,
`contract` (the case's behavioral contract), `openevals` with any rubric prompt
(`prompt: CONCISENESS_PROMPT` or a name like `hallucination`), `trajectory_llm`, and
`custom`. Evaluators whose reference is absent on a case are `skipped`, not errors.
See `skills/agent-eval-run/references/evaluator-selection.md`.

## LangSmith (optional)

Set `LANGSMITH_API_KEY` (see `.env.example`); `evalbuilder publish` uploads approved
cases (idempotent, read-back verified by `metadata.local_case_id`) and records example
IDs in the dataset. The pipeline's `stages.publish: auto` publishes only when a key is
configured; a missing key is degraded, never blocking.

## Testing

```bash
uv run pytest                     # offline suite (~330 tests): CLI, pipeline e2e with scripted models and
                                  # offline generators (support_bot; weather_bot with a 4-case dataset through
                                  # run_pipeline, the review loop and real background jobs — the agent behind
                                  # the local target's HTTP server), the agent server / clients / inference
                                  # engine (test_serve, test_agent_client, test_inference), the deployment
                                  # targets against a fake command runner + the local target for real
                                  # (test_deploy), the phase commands (test_cli_phases), tool schemas /
                                  # edge cases, jobs, providers, naming convention, UI loader, headless page
                                  # tests (AppTest): every page with/without a project, setup persistence
uv run pytest -m docker tests/test_deploy_integration.py   # real Docker Compose pipeline + Kubernetes deploy/infer/down
                                  # on the local daemon / cluster (Docker Desktop); excluded from the default
                                  # run by pyproject's addopts, skipped when docker / kubectl are unavailable
uv run pytest -m ui               # Playwright browser tests: report pages (tests/ui/test_playwright.py) and the
                                  # interactive flows (tests/ui/test_playwright_flows.py): project selection,
                                  # setup state across pages + Clear project; setup → discover → generate/
                                  # validate/save YAML; dataset-only run → review → feedback → regenerate →
                                  # approve → evaluate → results; full autonomous run; weather-bot small run
EVALBUILDER_UI_SHOTS=shots uv run pytest -m ui    # also saves screenshots
```

`bash scripts/cli-smoke.sh` exercises every CLI tool and the pipeline end to end, offline,
with whatever `python`/`evalbuilder` is on `PATH` (≈2 minutes, no model or key) — the
check to run in an environment without uv after `pip install -e ".[ui,dev]"`.

Browser tests skip themselves when playwright/chromium is missing
(`uv run playwright install chromium`). The flow tests run the real pipeline as a
background job with offline models (`scripted:examples.<name>.agent:default_scripted_model`
+ `scripted:examples.<name>.offline:generator_model`) and assert on the artifacts
written to disk, so they finish in seconds. The Streamlit pages are also exercised without
a browser via `streamlit.testing.v1.AppTest` (`tests/test_ui_pages.py`,
`tests/test_ui_pipeline_pages.py`); `tests/test_e2e_small_agents.py` runs the smallest
agent end to end (`-m "not slow"` skips the subprocess job test).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `generator model 'x:y' unavailable: provider 'x' not ready: set X_API_KEY; install …` | preflight readiness: export the key (or put it in `.env`) and `uv sync --extra <provider>` |
| `claude CLI not on PATH` | install Claude Code and log in; or switch specs to an API provider |
| verdict `incomplete`, stage `review` = `awaiting_review` | set `review.auto_approve: true` + `approved_by`, or `evalbuilder review … --approve`, then `--resume` |
| `tools without mock rules: […]` | `mocking.required: true` needs a fixture per mockable tool (under `llm`: a rule or a default-strategy behaviour); the generator retries once, otherwise add rules with `evalbuilder mock set` / strategies with `evalbuilder mock strategies --set` |
| `mock model 'x:y' unavailable` / `no mock model` | `mocking.on_miss: llm` needs a ready `models.mock` (or the generator); standalone: `--mock-model SPEC` or `evalbuilder mock strategies … --model SPEC` |
| `N LLM mock response(s) failed output-schema validation` | the engine's answers did not conform after the repair round: check `mock_calls` in the run artifact; tighten the strategy's behaviour or examples, or set `on_invalid: strict` to surface them as errors |
| `every case failed with infrastructure errors` | the deployed agent cannot build its graph or reach its model — `evalbuilder deploy logs DIR` shows the server's error; fix and `--resume --from deploy` |
| `deployment target 'docker' unavailable: …` | preflight: `docker` / `kubectl` / `oc` not on PATH, the daemon or cluster not reachable, `oc` not logged in — or switch `deploy.target: local` |
| `models.agent 'claude-cli:…' cannot run inside the agent container` | container targets need a model that runs in the image: an API-key provider with the key in `deploy.env` (`ANTHROPIC_API_KEY: null` copies it from the host) or a `scripted:` model |
| pods stay `ErrImageNeverPull` / `ImagePullBackOff` | the cluster cannot see the local image: `deploy.image.push: load` (the default without a registry) streams it into the nodes through a privileged loader pod; when privileged pods are forbidden set `image.registry` and `push: registry` |
| `docker compose … up` fails with a port conflict | `deploy.port` (8080) is published on the host — set `deploy.host_port` to a free port (the container port stays) |
| `deployed agent at … is not healthy` after a pause | the `kubectl port-forward` died — `evalbuilder deploy status DIR` restarts it (the infer stage does the same before every run) |
| judge scores with comment `Test.` | placeholder judge rationale; listed under `stability.suspect_judge_comments`, retried once automatically |
| `streamlit is not installed` | `uv sync --extra ui` |
| UI shows "Missing artifact …" | that stage has not run (or failed); the message names the file and the stage |

## Extending to other frameworks

The dataset schema is framework-neutral; LangGraph specifics live in
`src/evalbuilder/{discover,target,mocking}.py`. A Google ADK adapter maps 1:1
(AgentMetadata → agent-map, `before_tool_callback` mocking, `.evalset.json` import);
Dify follows via DSL rewrite. The architecture guide in `docs/architecture/` describes
every skill, command, module, the pipeline and the UI, with the decision record
(`06-decisions.md`) and the limitations register (`07-limitations.md`); the dated design
specs are in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`.
