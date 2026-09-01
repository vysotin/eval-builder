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
| `runs` | `repeats: 3` | Repeats feed stability detection |
| `mocking` | `required: true`, `on_miss: strict`, `strategies: true`, `strategy: default`, `on_invalid: fallback`, `max_repairs: 1` | Layer 1: every mockable tool from `TOOLS` gets a fixture (wildcard default unless `on_miss: llm`); per-case rules inject errors. `on_miss`: `strict` (unmatched call = error), `llm` (layer 2: the LLM mock engine answers from `mock-strategies.json` under `models.mock`), `fallback`, `real`. `strategies` writes the strategies artifact; `strategy` is the dataset default (cases / scenarios may select another); `on_invalid` decides what a still-invalid engine answer becomes after `max_repairs` repair rounds (`fallback` = the strategy's `fallback_response`, `strict` = an infrastructure error). Skill loaders are never mocked |
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

## Artifacts

One naming convention (`src/evalbuilder/pipeline/layout.py`): root artifacts are
`<kind>.json`/`.yaml`, per-run artifacts are `results/<kind>-<run_id>.json`, every JSON
artifact embeds `"schema": "evalbuilder/<kind>/v1"`. Older directories (`mocks.json`,
`plan.json`, `results/report-*.json`) still load.

| file | schema | written by | holds |
|---|---|---|---|
| `agent-map.json` | `evalbuilder/agent-map/v1` | discover, map | graph, tools (`kind`, `mockable`), prompts, skills, intents, scenarios (`skills`), failure modes, constraints |
| `applicable-failures.json` | `evalbuilder/applicable-failures/v1` | map | `failure_types`: type → gating evidence |
| `mock-rules.json` | `evalbuilder/mock-rules/v1` | mocks | `tools`: tool → ordered rules (layer 1) |
| `mock-strategies.json` | `evalbuilder/mock-strategies/v1` | mocks | `world` + `strategies`: id → description, per-tool `behavior`, `examples`, `fallback_response` (layer 2) |
| `coverage-plan.json` | `evalbuilder/coverage-plan/v1` | dataset | planned cells + summary |
| `dataset.json` | `evalbuilder/dataset/v1` | dataset, review | cases with review/publication state |
| `coverage.json` | `evalbuilder/coverage/v1` | dataset, review | planned vs covered, by kind, gaps, `skills` / `uncovered_skills` |
| `evaluators.yaml` | – | score | evaluator specs |
| `results/run-<id>.json` | `evalbuilder/run/v1` | run | outputs, trajectories, tool calls, `mock_calls` ledger per case; `mocking` totals |
| `results/score-report-<id>.json` | `evalbuilder/score-report/v1` | score | per-metric stats, slices, per-case scores |
| `aggregate.json` | `evalbuilder/aggregate/v1` | aggregate | pass rates vs thresholds, slices, stability, verdict |
| `scenarios.yaml` | – | simulate | multi-turn scenarios |
| `simulation.json` | `evalbuilder/simulation/v1` | simulate | transcripts, violations |
| `analysis.json` | `evalbuilder/analysis/v1` | analyze | patterns, recommendations |
| `state.json` | `evalbuilder/pipeline-state/v1` | engine | stage status, problems, `data.stopped_after` (drives `--resume`) |
| `report.json` | `evalbuilder/pipeline-report/v1` | report | the final report (also lists `artifacts`) |
| `job.json` (+ `pipeline.log`) | `evalbuilder/pipeline-job/v1` | UI | background run launched from the UI: mode, argv, pid, timing, exit code |

`evalbuilder ui <dir>` renders all of them; `evalbuilder ui` alone lets you pick a
directory or upload files.

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
| run | verify | retried once; infrastructure errors per case recorded |
| score | run | judge errors recorded per metric, never as agent failures |
| aggregate | score | – |
| simulate | review (optional) | failure never blocks the verdict |
| publish | review (optional) | skipped without key |
| analyze | aggregate (optional) | deterministic fallback |
| report | always | – |

Verdict: `incomplete` if any required stage (preflight … aggregate) did not succeed;
otherwise `pass`/`fail` from thresholds.

## Run control

| flag | effect |
|---|---|
| `--until STAGE` | stop deliberately after `STAGE`; later stages are `skipped` (`stopped after <stage> (--until)`), the report lands, exit code 0. `--until dataset` = intents + mocks + cases only |
| `--resume` | reuse completed stages from `state.json` (the report is always rebuilt) |
| `--resume --from STAGE` | invalidate `STAGE` and everything after it, then rerun |

Feedback loop: `--until dataset` → append a `feedback` entry (or `instructions`) to the
config → `--resume --from dataset --until dataset` (or `--from map` / `--from mocks`) →
`--resume` to approve and evaluate. The UI's *Run & review* page drives exactly these
commands as background jobs.

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
allowed_tools, metadata, references[{path, title, chars, excerpt}], scripts,
tools_mentioned, used_by, evidence: ["skill:<name>"]}`; `app.skills_dir` names the
folder. Nodes carry `skills` and `skills_source` (`inline` — the body is embedded in the
prompt; `listing` — names and descriptions are listed and read on demand; `prompt` —
the prompt names the skill). Tools of kind `skill_loader` (`load_skill`, `read_skill`,
…) are `mockable: false`. Discovery recognises `load_skills(...)`, `SKILLS_DIR`-style
constants (`Path(__file__).parent / "skills"`), `skills=` factory keywords, the prompt
helpers `skills_prompt` / `skills_inline_prompt` (rendered into the recorded prompt), and
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
