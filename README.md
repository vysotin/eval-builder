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
- run cases repeatedly, score them with deterministic checks and LLM judges, aggregate
  pass rates vs thresholds, slices and stability, simulate multi-turn conversations,
  and write a report with a `pass | fail | incomplete` verdict;
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
                       (agent-eval-pipeline skill)    │  review → verify → run×N → score×N → aggregate →
                                                      │  simulate → publish → analyze → report
                                                      ▼
                                   eval/pipeline/<name>/   (agent-map.json, dataset.json,
                                   results/run-*.json, aggregate.json, report.json, …)
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
5. [Models and providers](#models-and-providers)
6. [Artifacts and naming convention](#artifacts-and-naming-convention)
7. [UI: setup, run & review, report](#ui-setup-run--review-report)
8. [Interactive skills](#interactive-skills)
9. [CLI reference](#cli-reference)
10. [Dataset format, mocking, target contract](#dataset-format)
11. [Evaluators](#evaluators)
12. [LangSmith](#langsmith-optional)
13. [Testing](#testing)
14. [Troubleshooting](#troubleshooting)
15. [Extending](#extending-to-other-frameworks) · [Architecture guide](docs/architecture/README.md)

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
#    models, and — to let the pipeline approve generated cases — review.auto_approve + approved_by

# 3. run (exit code 1 unless verdict == pass); ~45 min for 16 cases × 2 repeats with claude-cli:claude-sonnet-5
uv run evalbuilder pipeline run eval/pipeline.yaml
#    …or only generate the dataset + mocks, look at them, then continue:
uv run evalbuilder pipeline run eval/pipeline.yaml --until dataset
uv run evalbuilder pipeline run eval/pipeline.yaml --resume

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
the artifacts of real runs: `uv run evalbuilder ui docs/examples/support-bot`.
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
  runner.py           run approved cases, capture trajectory / tool calls / node path
  evaluators.py       deterministic evaluators + OpenEvals/AgentEvals judges, score_run
  simulate.py         multi-turn scenarios with a simulated user, violation mining
  coverage.py         intent×topic×scenario×failure_mode grid, gaps
  langsmith_io.py     idempotent publish with read-back verification
  providers.py        API-key providers (anthropic/openai/google_genai), aliases, readiness, @effort
  claude_cli.py       ChatClaudeCLI: Claude Code CLI as a LangChain chat model; model_from_spec
  config.py           .env settings, capability_check
  testing.py          ScriptedChatModel + SchemaScriptedModel (offline generators) for fully offline runs
  pipeline/
    config.py         evalbuilder/pipeline-config/v1 (pydantic), template, semantic checks
    engine.py         stage runner: deps, retries, skip, awaiting_review, persisted state.json
    stages.py         the 14 stage functions + PipelineContext (lazy, disk-backed artifacts)
    generator.py      structured-output LLM "author" for map/mocks/cases/review/scenarios/analysis
    planning.py       coverage cells planning and achieved coverage
    taxonomy.py       failure-type gating by agent structure
    aggregate.py      repeat-aware aggregation: pass rates, thresholds, slices, stability
    report.py         report assembly + run_pipeline entry point
    layout.py         THE artifact naming convention (kinds, files, schema ids, legacy names)
    jobs.py           background pipeline jobs for the UI (job.json + pipeline.log, wrapper entry point)
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
                      scripted default model; support_bot/incident_desk/loan_desk ship pipeline.yaml, and
                      weather_bot/support_bot/incident_desk/loan_desk ship offline.py (a SchemaScriptedModel
                      generator for LLM-free pipeline runs)
docs/examples/support-bot/, incident-desk/   artifacts of real pipeline runs (UI demo + test fixtures)
docs/architecture/    how every skill, CLI command, module, the pipeline and the UI work; decisions; limitations
docs/superpowers/     design specs and implementation plans
tests/                offline pytest suite; tests/ui/ = Playwright browser tests (report pages + interactive flows)
```

## The autonomous pipeline

`evalbuilder pipeline run CONFIG` turns one YAML file into a verdict. An LLM
**generator** plays the author role the interactive skills otherwise give to a human:
it derives intents and scenarios (evidence-cited, taxonomy-gated), writes tool
fixtures, fills a coverage plan with cases, self-reviews them, writes multi-turn
simulation scenarios and the final analysis. Every artifact it produces still goes
through the same CLI validation, and the human review gate is preserved.

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
runs: {repeats: 2}
mocking: {required: true, on_miss: strict, strategies: true, strategy: default, on_invalid: fallback, max_repairs: 1}
review: {auto_approve: true, approved_by: "your name"}
```

Every key, default and semantic check is documented in
`skills/agent-eval-pipeline/references/config-reference.md`.

### Stages

| stage | depends on | what it does | on failure |
|---|---|---|---|
| preflight | – | config checks, target import, model readiness (`providers.provider_ready`) | blocking |
| discover | preflight | AST + live introspection → `agent-map.json` (nodes, edges, tools with `kind`/`mockable`, prompts, skills) | blocking |
| map | discover | generator authors intents, scenarios, failure scenarios, topics, derived constraints; taxonomy gates failure types | retry once; invalid entries dropped → `problems` |
| mocks | discover | generator writes fixtures for every **mockable** tool → `mock-rules.json` (no wildcard under `on_miss: llm`), and the LLM mock strategies → `mock-strategies.json` | generic fixture / generic default strategy |
| dataset | map, mocks | plan coverage cells → generator fills them → `dataset.json`, `coverage.json` | invalid cases dropped, gaps reported |
| review | dataset | self-review rejects bad cases; approves the rest **only** with `review.auto_approve` | stops with `awaiting_review` |
| verify | review | every expected tool call has a rule; every tool is mocked (under `llm`: misses are counted as engine-answered, a tool is covered by a rule or the default strategy) | blocking |
| run | verify | `runs.repeats` executions of all approved cases → `results/run-<id>.json` (per-case `mock_calls` ledger, `mocking` totals) | retry once |
| score | run | evaluators per run → `results/score-report-<id>.json` | judge errors recorded per metric |
| aggregate | score | pass rates vs thresholds, slices, stability, failing cases → `aggregate.json` | – |
| simulate | review (optional) | multi-turn scenarios with a simulated user → `simulation.json`; violations mined into pending cases | never blocks |
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

`state.json` records every stage. `--resume` reuses completed stages (the report is
always rebuilt); `--resume --from STAGE` regenerates from a stage on, e.g. after
editing thresholds (`--from aggregate`) or the agent (`--from run`).

`--until STAGE` stops deliberately after a stage (later stages are `skipped` with the
reason `stopped after <stage> (--until)`, the report still lands, exit code 0). The
typical loop:

```bash
evalbuilder pipeline run cfg.yaml --until dataset          # intents, mocks, cases — nothing runs yet
# read dataset.json / mock-rules.json (or the UI), then add a comment to the config:
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

| file | schema | stage | contents |
|---|---|---|---|
| `agent-map.json` | `evalbuilder/agent-map/v1` | discover, map | graph (AST + live), tools + arg schemas (`kind`, `mockable`), prompts, skills, intents, scenarios (`skills`), failure scenarios, constraints, topics |
| `applicable-failures.json` | `evalbuilder/applicable-failures/v1` | map | `failure_types`: failure type → gating evidence |
| `mock-rules.json` | `evalbuilder/mock-rules/v1` | mocks | `tools`: tool → ordered rules (layer 1) |
| `mock-strategies.json` | `evalbuilder/mock-strategies/v1` | mocks | `world` + `strategies`: per strategy, each tool's behaviour, examples, fallback response (layer 2) |
| `coverage-plan.json` | `evalbuilder/coverage-plan/v1` | dataset | planned cells and summary |
| `dataset.json` | `evalbuilder/dataset/v1` | dataset, review | cases (inputs, references, metadata, mocks), review + publication state |
| `coverage.json` | `evalbuilder/coverage/v1` | dataset, review | planned vs covered, by kind, multi-turn, gaps, `skills` / `uncovered_skills` |
| `evaluators.yaml` | – | score | evaluator specs |
| `results/run-<id>.json` | `evalbuilder/run/v1` | run | per-case outputs, trajectory, tool calls, node path, errors, `mock_calls` ledger; `mocking` totals |
| `results/score-report-<id>.json` | `evalbuilder/score-report/v1` | score | per-metric stats, slices, per-case scores/comments/errors/skips |
| `aggregate.json` | `evalbuilder/aggregate/v1` | aggregate | pass rates vs thresholds, slices, weak slices, stability, failing cases, verdict |
| `scenarios.yaml` | – | simulate | multi-turn scenarios |
| `simulation.json` | `evalbuilder/simulation/v1` | simulate | transcripts, stop reasons, violations |
| `analysis.json` | `evalbuilder/analysis/v1` | analyze | summary, failure patterns, weak slices, recommendations, evaluator issues |
| `state.json` | `evalbuilder/pipeline-state/v1` | engine | stage status/timing/details, problems |
| `report.json` | `evalbuilder/pipeline-report/v1` | report | everything above, condensed, plus `artifacts` index |

Directories written before this convention (`mocks.json`, `plan.json`,
`results/report-*.json`, schema `evalbuilder/report/v1`) still load everywhere.
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
mock policy, review approval and paths; *Generate YAML* fills an editable YAML editor;
*Validate* / *Save config* / *Generate dataset & mocks only* / *Run full pipeline*. When
the config file changes on disk (review feedback, approval, a job) the form reloads it.

**Run & review** follows the project's background job (`job.json` + `pipeline.log`,
stage table refreshed every 2 s), and after a dataset-only run shows the review loop:
dataset summary (statuses, failure modes, schema-edge cases, mocked tools), a comment box
whose text is appended to the config's `feedback`, *Save feedback & regenerate*
(`--resume --from map|mocks|dataset --until dataset`), and *Approve remaining cases & run
evaluation* (records the approver, rejects selected cases, `--resume`). *Open results*
jumps to the Summary page.

Report pages (all read the project's artifacts; uploaded files are identified by their
embedded `schema` id, falling back to the file name):

| page | shows |
|---|---|
| Summary (after Analysis) | verdict + reasons, overall score, metric pass rates vs thresholds, coverage, stability, stage timeline, analysis summary, problems |
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
evalbuilder run DATASET [--mock] [--model SPEC] [--on-miss real|fallback|strict|llm] [--mock-model SPEC] [--strategy S] [--ids …] [--out D]
evalbuilder score RUN --dataset D --evaluators evaluators.yaml [--out D]
evalbuilder simulate DATASET --scenarios scenarios.yaml [--no-mine] [--mock] [--on-miss P] [--mock-model SPEC]
evalbuilder publish DATASET [--dataset-name N]            LangSmith, idempotent
evalbuilder pipeline init|run|report                      the autonomous pipeline
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
runner rebuilds the graph per case, wrapping `TOOLS` with the merged mock rules, and
injects `model=` when a run specifies one. Five example targets ship in `examples/`
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
uv run pytest                     # offline suite (~280 tests): CLI, pipeline e2e with scripted models and
                                  # offline generators (support_bot; weather_bot with a 4-case dataset through
                                  # run_pipeline, the review loop and real background jobs), tool schemas /
                                  # edge cases, jobs, providers, naming convention, UI loader, headless page
                                  # tests (AppTest): every page with/without a project, setup persistence
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
| `every case failed with infrastructure errors` | target import / model transport problem — the first error is in the message; fix and `--resume --from run` |
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
