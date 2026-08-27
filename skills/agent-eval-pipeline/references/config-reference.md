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
| `models` | `agent: null`, `judge: claude-cli:sonnet`, `generator: claude-cli:sonnet` | `provider:model[@effort]` — `claude-cli:<model>` (subscription), `anthropic:…` / `claude:…`, `openai:…`, `gemini:…` / `google:…` (API keys, see below), `scripted:module:factory` (tests). `agent: null` keeps the target's own model |
| `constraints` | `[]` | Free-text test constraints; reach every generation prompt and become judge `contract`s |
| `coverage` | `total_cases: 20`, `per_intent: {happy: 2, failure: 1}`, `per_failure_category: 1`, `out_of_intent: 2`, `multi_turn_share: 0.15` | Per-cell counts are hard minimums; `total_cases` is a floor filled with extra happy variants |
| `evaluators` | `expected_tools, contains, contract, correctness` | Same specs as `evaluators.yaml`: deterministic (`expected_tools`, `contains`, `json_valid`, `trajectory_match`), OpenEvals (`correctness`, `contract`, `openevals` + `prompt: NAME_PROMPT`, or any rubric name like `hallucination`), AgentEvals (`trajectory_llm`), `custom` |
| `thresholds` | `default: 0.8`, `metrics: {}`, `slice_min: 0.5`, `overall_pass: 0.8` | Pass rate per metric, per slice (intent / failure_mode / variant), and overall (mean of metric pass rates) |
| `runs` | `repeats: 3` | Repeats feed stability detection |
| `mocking` | `required: true`, `on_miss: strict` | Every tool from `TOOLS` gets a wildcard fixture; per-case rules inject errors/injections |
| `stages` | `skip: []`, `max_retries: 1`, `simulate: true`, `publish: auto` | `publish: auto` runs only when `LANGSMITH_API_KEY` is set |
| `review` | `auto_approve: false`, `approved_by: ""`, `note: ""` | Without `auto_approve` the pipeline stops at `awaiting_review` |
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

`uv sync --extra llm` installs all three. The same spec works for `models.agent`,
`models.judge`, `models.generator` and `EVALBUILDER_*_MODEL` in `.env`.

## Artifacts

One naming convention (`src/evalbuilder/pipeline/layout.py`): root artifacts are
`<kind>.json`/`.yaml`, per-run artifacts are `results/<kind>-<run_id>.json`, every JSON
artifact embeds `"schema": "evalbuilder/<kind>/v1"`. Older directories (`mocks.json`,
`plan.json`, `results/report-*.json`) still load.

| file | schema | written by | holds |
|---|---|---|---|
| `agent-map.json` | `evalbuilder/agent-map/v1` | discover, map | graph, tools, prompts, intents, scenarios, failure modes, constraints |
| `applicable-failures.json` | `evalbuilder/applicable-failures/v1` | map | `failure_types`: type → gating evidence |
| `mock-rules.json` | `evalbuilder/mock-rules/v1` | mocks | `tools`: tool → ordered rules |
| `coverage-plan.json` | `evalbuilder/coverage-plan/v1` | dataset | planned cells + summary |
| `dataset.json` | `evalbuilder/dataset/v1` | dataset, review | cases with review/publication state |
| `coverage.json` | `evalbuilder/coverage/v1` | dataset, review | planned vs covered, by kind, gaps |
| `evaluators.yaml` | – | score | evaluator specs |
| `results/run-<id>.json` | `evalbuilder/run/v1` | run | outputs, trajectories, tool calls per case |
| `results/score-report-<id>.json` | `evalbuilder/score-report/v1` | score | per-metric stats, slices, per-case scores |
| `aggregate.json` | `evalbuilder/aggregate/v1` | aggregate | pass rates vs thresholds, slices, stability, verdict |
| `scenarios.yaml` | – | simulate | multi-turn scenarios |
| `simulation.json` | `evalbuilder/simulation/v1` | simulate | transcripts, violations |
| `analysis.json` | `evalbuilder/analysis/v1` | analyze | patterns, recommendations |
| `state.json` | `evalbuilder/pipeline-state/v1` | engine | stage status, problems (drives `--resume`) |
| `report.json` | `evalbuilder/pipeline-report/v1` | report | the final report (also lists `artifacts`) |

`evalbuilder ui <dir>` renders all of them; `evalbuilder ui` alone lets you pick a
directory or upload files.

## Stage semantics

| Stage | Depends on | Failure handling |
|---|---|---|
| preflight | – | blocking (config, target import, generator/agent model) |
| discover | preflight | – |
| map | discover | generator repair once; invalid entries dropped and listed in `problems` |
| mocks | discover | generic fixture fallback per tool |
| dataset | map, mocks | invalid cases dropped; missing cells re-requested once; gaps reported |
| review | dataset | self-review rejects; stops with `awaiting_review` unless `auto_approve` |
| verify | review | blocking when expected calls have no rule or a tool is unmocked |
| run | verify | retried once; infrastructure errors per case recorded |
| score | run | judge errors recorded per metric, never as agent failures |
| aggregate | score | – |
| simulate | review (optional) | failure never blocks the verdict |
| publish | review (optional) | skipped without key |
| analyze | aggregate (optional) | deterministic fallback |
| report | always | – |

Verdict: `incomplete` if any required stage (preflight … aggregate) did not succeed;
otherwise `pass`/`fail` from thresholds.
