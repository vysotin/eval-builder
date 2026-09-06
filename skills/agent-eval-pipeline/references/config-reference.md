# Pipeline config reference — `evalbuilder/pipeline-config/v1`

Minimal config (every other key has a default):

```yaml
schema: evalbuilder/pipeline-config/v1
name: support-bot
target:
  source: examples/support_bot/agent.py     # agent source (AST discovery)
  module: examples.support_bot.agent        # importable module: TOOLS + build_agent
review:
  auto_approve: true                        # explicit human authorization
  approved_by: "your name"
```

| Section | Keys (defaults) | Notes |
|---|---|---|
| `target` | `source`, `module`, `factory: build_agent`, `root_node: null` | `factory` is the graph factory ("root node path"); `root_node` is recorded in the agent map |
| `models` | `agent: null`, `judge: claude-cli:claude-sonnet-5`, `generator: claude-cli:claude-sonnet-5`, `mock: null` | `provider:model[@effort]` — `claude-cli:<model>` (Claude Code subscription; Sonnet 5 by default, `claude-cli:sonnet` = latest Sonnet), `anthropic:…` / `claude:…`, `openai:…`, `gemini:…` / `google:…` (API keys, see below), `scripted:module:factory` (tests, offline generators). `agent: null` keeps the target's own model; `mock` drives the LLM mock engine under `mocking.on_miss: llm` (`null` = the generator) |
| `constraints` | `[]` | Free-text test constraints; reach every generation prompt and become judge `contract`s |
| `instructions` | `""` | Free-text general rules for the generator (domain notes, emphasis, exclusions); appended to every generation prompt as `USER INSTRUCTIONS` |
| `feedback` | `[]` | Reviewer comments `{at, note, from_stage}` appended between runs (UI *Save feedback*, or by hand); appended to every generation prompt as `REVIEWER FEEDBACK` |
| `coverage` | `total_cases: 20`, `per_intent: {happy: 2, failure: 1}`, `per_failure_category: 1`, `out_of_intent: 2`, `multi_turn_share: 0.15`, `per_tool_edge_cases: 2` | Per-cell counts are hard minimums; `total_cases` is a floor filled with extra happy variants; `per_tool_edge_cases` = schema-derived edge cases per tool (`schema-edge` cells: missing / wrong / out-of-enum / boundary input, malformed output) |
| `evaluators` | `expected_tools, contains, contract, correctness` | Same specs as `evaluators.yaml`: deterministic (`expected_tools`, `contains`, `json_valid`, `trajectory_match`), OpenEvals (`correctness`, `contract`, `openevals` + `prompt: NAME_PROMPT`, or any rubric name like `hallucination`), AgentEvals (`trajectory_llm`), `custom` |
| `thresholds` | `default: 0.8`, `metrics: {}`, `slice_min: 0.5`, `overall_pass: 0.8` | Pass rate per metric, per slice (intent / failure_mode / variant), and overall (mean of metric pass rates) |
| `runs` | `repeats: 3` | Repeats feed stability detection. The older `runs.parallel_intents` / `parallel_scoring` / `parallel_simulations` keys still load: they are migrated to `inference.workers` / `evaluation.workers` (simulations share the inference pool; that key is dropped) and each migration is recorded as a preflight problem |
| `deploy` | `target: local`, `image: {name: null, tag: latest, registry: null, push: auto, extras: []}`, `build: {context: ".", dockerfile: null, include: [], requirements: null}`, `port: 8080`, `namespace: default`, `replicas: 1`, `expose: port-forward`, `env: {}`, `keep: false`, `timeout: 240`, `context: null` | Where the agent (with both mock layers, behind `evalbuilder serve`) runs during inference. `target`: `local` (a subprocess of this interpreter, no Docker), `docker` (Docker Compose on the local daemon, `http://127.0.0.1:<port>`), `kubernetes` (`kubectl`: Deployment + Service), `openshift` (`oc`, prototype: + Route). `image.name` defaults to `evalbuilder-<name>`; `image.registry` prefixes the reference and is where `push: registry` pushes (required for openshift); `push`: `auto` = `registry` when a registry is set, else `load` (stream `docker save` into every node through a privileged loader DaemonSet — what works on Docker Desktop), `none` = the nodes already have it; `image.extras` = evalbuilder extras installed in the image (e.g. `[llm]` for API-key providers). `build.context` is the docker build context (the checkout, which must contain `src/evalbuilder`), `build.dockerfile` replaces the generated one, `build.include` copies extra paths (the target's package always is), `build.requirements` is pip-installed. `expose`: `port-forward` (a `kubectl port-forward`, restarted when it dies; works on every cluster), `nodeport` (`http://<node InternalIP>:<nodePort>`), `route` (openshift only). `env`: container environment; a `null` value copies the variable from the host (API keys) — unset ones are reported. `keep: true` skips the `teardown` stage. `timeout`: seconds to wait for readiness. `context`: kubectl / oc context. Container targets need `models.agent` (and `models.mock` under `on_miss: llm`) to run inside the image — an API-key provider or `scripted:`; `claude-cli` is rejected by `problems()` |
| `inference` | `workers: 4`, `backend: threads`, `timeout: 120` | joblib parallelism for the infer stage (cases) and the simulate stage (scenarios) against the deployed agent: `workers` conversations at once (1 = sequential), `backend` `threads` (default; the work waits on HTTP) or `processes` (loky), `timeout` seconds per agent request |
| `evaluation` | `workers: 4`, `backend: threads` | joblib parallelism for scoring: each run's case runs are split into `workers` contiguous chunks scored concurrently (evaluators rebuilt per process worker); the report is identical to a sequential pass |
| `mocking` | `required: true`, `on_miss: strict`, `strategies: true`, `strategy: default`, `on_invalid: fallback`, `max_repairs: 1` | Layer 1: every mockable tool from `TOOLS` gets a fixture (wildcard default unless `on_miss: llm`); per-case rules inject errors. `on_miss`: `strict` (unmatched call = error), `llm` (layer 2: the LLM mock engine answers from the strategies in `dataset.mocks.strategies` under `models.mock`), `fallback`, `real`. `strategies` pre-generates the strategies; `strategy` is the dataset default (cases / scenarios may select another); `on_invalid` decides what a still-invalid engine answer becomes after `max_repairs` repair rounds (`fallback` = the strategy's `fallback_response`, `strict` = an infrastructure error). Skill loaders are never mocked |
| `stages` | `skip: []`, `max_retries: 1`, `simulate: true`, `publish: auto` | `publish: auto` runs only when `LANGSMITH_API_KEY` is set |
| `review` | `auto_approve: false`, `approved_by: ""`, `note: ""` | Without `auto_approve` the pipeline stops at `awaiting_review`; cases already approved/rejected by hand or in the UI are kept (`already_reviewed`) |
| `output` | `dir: eval/pipeline/<name>` | See "Artifacts" below |
| `langsmith` | `dataset_name: null` | Defaults to `name` |

## Model providers

| provider (aliases) | auth | package / extra | `@effort` |
|---|---|---|---|
| `claude-cli` | Claude Code CLI on PATH (subscription) | bundled | `low|medium|high` → CLI effort |
| `anthropic` (`claude`) | `ANTHROPIC_API_KEY` | `langchain-anthropic` / `--extra anthropic` | extended-thinking budget 1024/4096/16000 tokens |
| `openai` | `OPENAI_API_KEY` | `langchain-openai` / `--extra openai` | `reasoning_effort` |
| `google_genai` (`gemini`, `google`) | `GOOGLE_API_KEY` | `langchain-google-genai` / `--extra gemini` | ignored |
| `scripted` | – | – | – |

`uv sync --extra llm` (or `pip install -e '.[llm]'`) installs all three. The same spec works for `models.agent`,
`models.judge`, `models.generator` and `EVALBUILDER_*_MODEL` in `.env`.

`evalbuilder pipeline init` writes every key above with a comment; `evalbuilder deploy
render CONFIG [--target T]` prints the Dockerfile / compose file / manifests the
`deploy` section produces.

## Artifacts

One naming convention (`src/evalbuilder/pipeline/layout.py`): root artifacts are
`<kind>.json`/`.yaml`, per-run artifacts are `results/<kind>-<run_id>.json`, every JSON
artifact embeds `"schema": "evalbuilder/<kind>/v1"`. **No artifact duplicates part of
another one** — each kind is `final` (a deliverable), `work` (scratch under
`<out_dir>/work/`, deletable), or `derived` (no file: it lives inside a deliverable).

### Deliverables

| file | schema | written by | holds |
|---|---|---|---|
| `agent-map.json` | `evalbuilder/agent-map/v1` | discover, map | graph, tools (`kind`, `mockable`), prompts, skills, intents, scenarios (`skills`), failure modes, constraints, `applicable_failures` |
| `dataset.json` | `evalbuilder/dataset/v1` | dataset, review | cases with review/publication state, `mocks` (both layers), `coverage` (`plan` + `achieved`) |
| `evaluators.yaml` | – | score | evaluator specs |
| `deployment.json` | `evalbuilder/deployment/v1` | deploy, teardown | where the agent ran: `target`, `image`, `endpoint`, `expose`, `status` (`up` / `down` / `failed`), `resources` (compose project, namespace + names, loader, port-forward pid), every command executed (`commands`), health facts, the spec |
| `results/run-<id>.json` | `evalbuilder/run/v1` | infer | outputs, trajectories, tool calls, `mock_calls` ledger and a per-turn `log` (`turn`, `seconds`, `tool_calls`, `error`, `mode`, `endpoint`) per case; `mocking` totals; `execution` (`mode`, `endpoint`, `workers`, `backend`, `seconds`) |
| `results/score-report-<id>.json` | `evalbuilder/score-report/v1` | score | per-metric stats, slices, per-case scores |
| `aggregate.json` | `evalbuilder/aggregate/v1` | aggregate | pass rates vs thresholds, slices, stability, verdict |
| `scenarios.yaml` | – | simulate | multi-turn scenarios |
| `simulation.json` | `evalbuilder/simulation/v1` | simulate | transcripts, violations |
| `analysis.json` | `evalbuilder/analysis/v1` | analyze | patterns, recommendations |
| `report.json` | `evalbuilder/pipeline-report/v1` | report | the final report (also lists `artifacts`) |

### Scratch — `work/`, safe to delete

| file | schema | written by | holds |
|---|---|---|---|
| `work/mock-rules.json` | `evalbuilder/mock-rules/v1` | mocks | `tools`: tool → ordered rules (layer 1); handed to the dataset stage, kept for good in `dataset.mocks.tools` |
| `work/mock-strategies.json` | `evalbuilder/mock-strategies/v1` | mocks | `world` + `strategies`: id → description, per-tool `behavior`, `examples`, `fallback_response` (layer 2); kept in `dataset.mocks.strategies` |
| `work/state.json` | `evalbuilder/pipeline-state/v1` | engine | stage status, problems, `data.stopped_after` (drives `--resume`) |
| `work/run-progress.json` | `evalbuilder/run-progress/v1` | infer | live per-case / per-intent progress of the infer stage (+ `endpoint`, `workers`, `backend`) |
| `work/deploy/` | – | deploy | the generated `Dockerfile`, `compose.yaml` (docker), `manifests.yaml` + `loader.yaml` (kubernetes / openshift), `commands.log` (every external command with exit code and output tail), `serve.log` (local target), `port-forward.log` (kubernetes) |
| `work/job.json` (+ `pipeline.log`) | `evalbuilder/pipeline-job/v1` | UI | background run launched from the UI: mode, argv, pid, timing, exit code |

### Derived — no file; read them out of their parent

| kind | lives in |
|---|---|
| `applicable_failures` (type → gating evidence) | `agent-map.json` → `applicable_failures` |
| `coverage_plan` (planned cells + summary) | `dataset.json` → `coverage.plan` |
| `coverage` (planned vs covered, by kind, gaps, `skills` / `uncovered_skills`) | `dataset.json` → `coverage.achieved` |

`evalbuilder ui <dir>` renders all of them — including the derived kinds, read out of
their parent — and `evalbuilder ui` alone lets you pick a directory or upload files.
Directories written before this layout still load (stand-alone `coverage.json`,
`applicable-failures.json`, root `state.json`, and the pre-convention `mocks.json`,
`plan.json`, `results/report-*.json`); `evalbuilder pipeline compact <dir>` folds and
moves them onto the current one.

## Stage semantics

| Stage | Depends on | Failure handling |
|---|---|---|
| preflight | – | blocking (config, target import, generator/agent model) |
| discover | preflight | – |
| map | discover | generator repair once; invalid entries dropped and listed in `problems` |
| mocks | discover | generic fixture fallback per tool; generic default strategy when strategy generation fails |
| dataset | map, mocks | invalid cases dropped; missing cells re-requested once; gaps reported |
| review | dataset | self-review rejects; stops with `awaiting_review` unless `auto_approve` |
| verify | review | blocking when expected calls have no rule or a tool is unmocked; under `on_miss: llm` misses are counted (`llm_answered_calls`) and a tool is covered by a rule or the default strategy |
| deploy | verify | blocking: the target is unavailable (`docker` / `kubectl` / `oc` missing, daemon or cluster unreachable — preflight already checks this), the build or a command fails, or the container / server never becomes healthy within `deploy.timeout` (the failed `deployment.json` and `work/deploy/commands.log` hold the diagnosis); a deployment that is already up and healthy is reused |
| infer | deploy | `runs.repeats` × every approved case against the deployed agent (`inference.workers`); retried once; infrastructure errors per case recorded; a run where every case is an infrastructure error fails the stage |
| simulate | deploy, review (optional) | scenarios through the same deployed agent; failure never blocks the verdict |
| teardown | deploy (optional) | removes the deployment after infer / simulate; skipped by config when `deploy.keep` is true; a failure is a problem, not an incomplete verdict |
| score | infer | judge errors recorded per metric, never as agent failures; `evaluation.workers` splits |
| aggregate | score | – |
| publish | review (optional) | skipped without key |
| analyze | aggregate (optional) | deterministic fallback |
| report | always | – |

Verdict: `incomplete` if any required stage (preflight … verify, deploy, infer,
score, aggregate) did not succeed; otherwise `pass`/`fail` from thresholds.

## Run control

| flag | effect |
|---|---|
| `--until STAGE` | stop deliberately after `STAGE`; later stages are `skipped` (`stopped after <stage> (--until)`), the report lands, exit code 0. `--until dataset` = intents + mocks + cases only; `--until deploy` leaves the agent running (`evalbuilder deploy status DIR`, `deploy down DIR`) |
| `--resume` | reuse completed stages from `work/state.json` (the report is always rebuilt) |
| `--resume --from STAGE` | invalidate `STAGE` and everything after it, then rerun |

Feedback loop: `--until dataset` → append a `feedback` entry (or `instructions`) to the
config → `--resume --from dataset --until dataset` (or `--from map` / `--from mocks`) →
`--resume` to approve and evaluate. The UI's *Run & review* page drives exactly these
commands as background jobs.

Phase commands (the same code the stages run): `evalbuilder deploy up|status|down|
build|render|logs`, `evalbuilder infer DATASET [--deployment DIR | --endpoint URL]
[--scenarios F] [--repeats N] [--workers N] [--backend threads|processes]`,
`evalbuilder eval RUN… --dataset D --evaluators F [--workers N] [--aggregate]
[--config CFG]`. `deploy up` reuses a deployment that is already up and healthy.

## Tool schemas in the agent map

Every `tools[]` entry carries `args_schema` (explicit `args_schema=` pydantic class or
the signature, `$defs` for nested models), `output_schema` (return annotation),
`schema_source` (`args_schema` | `annotations` | `ast`), `models`, `side_effecting` and
`edge_cases` (`{id, kind, field, detail, failure_mode, expected_behavior, evidence}`,
kinds `missing_required`, `wrong_type`, `out_of_enum`, `boundary`, `malformed_output`).
Mock fixtures are validated against `output_schema`; `malformed_output` cases get a
pipeline-injected per-case mock override missing a required field. The LLM mock engine
validates every generated answer against the same `output_schema`.

## Agent skills in the agent map

`skills[]` entries: `{name, description, path, dir, prompt (the SKILL.md body),
chars, instruction, summarized, allowed_tools, tools, unknown_tools, rules, metadata,
references[{path, title, chars, excerpt}], scripts, tools_mentioned, used_by,
evidence: ["skill:<name>"]}`; `app.skills_dir` names the folder. `tools` are the tools
the skill can use, resolved against the agent's tools (frontmatter `allowed-tools` ∩
agent tools, then body-mentioned ones); `unknown_tools` flags allowed names no agent
tool matches; `rules` are the extracted imperative lines (never/must/always/only);
`instruction` is the full body when it is at most 1500 chars, else a deterministic
summary (`summarized: true`, description + section headlines + rules) — `prompt` always
keeps the full body. After the map stage each skill also carries `failure_cases`
(generator-authored failure cases *beyond* tool failure, validated against the skill
names and the applicable taxonomy) and each tool carries `failure_scenarios`
(per-tool failure scenarios, `failure_mode` defaulting to `tool_error_handling`).
Nodes carry `skills` and `skills_source` (`inline` — the body is embedded in the
prompt; `listing` — names and descriptions are listed and read on demand; `prompt` —
the prompt names the skill; `loader` — the prompt names none, but the node holds a
skill-loader tool so it can read any skill) plus `capabilities`: one line per skill —
which skill the node can call, which tools that reaches (scoped to the node's own
tools) and the result it achieves. Tools of kind `skill_loader` (`load_skill`,
`read_skill`, …) are `mockable: false` and get no failure analysis. Discovery
recognises `load_skills(...)`, `SKILLS_DIR`-style constants
(`Path(__file__).parent / "skills"`), `skills=` factory keywords, the prompt helpers
`skills_prompt` / `skills_inline_prompt` (rendered into the recorded prompt), and
`skill_loader_tool(...)` assignments (`evalbuilder.skills`).

## Two-layer mocking

Layer 1 answers from rules; on a miss `on_miss` decides. Under `llm` an
`LLMMockEngine` (one per case / scenario) prompts `models.mock` with the world, the
selected strategy's behaviour for the tool, its examples, the tool definition, the
previous calls of the conversation and the call itself; the answer is validated against
`output_schema`, repaired once, then the strategy's `fallback_response` is used or an
infrastructure error raised (`on_invalid`). Every call is logged
(`run.case_runs[].mock_calls`, layer `rule | llm | real | fallback | error`) and the
report's `mocking` section and `stability.llm_mocked_unstable` attribute instability
to the mock layer. The design mirrors ADK's user simulator (a described plan + a model in
config), applied to tools.
