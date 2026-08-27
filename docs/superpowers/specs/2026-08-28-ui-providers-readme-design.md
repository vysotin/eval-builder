# Streamlit report UI, API-key LLM providers, artifact naming, README — design

Date: 2026-08-28. Goal (set by the user): (1) a Streamlit visualisation of every
pipeline artifact — agent graph, intents, scenarios, dataset, eval results with final
stats — that loads artifacts by distinct naming conventions, with Playwright tests;
(2) a comprehensive README for pipeline usage and structure; (3) an option to use
LLMs via API key (Claude, OpenAI, Gemini) instead of `langchain-claude-code-cli`.

## 1. API-key providers (`evalbuilder/providers.py`)

Model specs stay `provider:model[@effort]`. A registry describes each API provider:

| provider (aliases) | env var | package | example spec |
|---|---|---|---|
| `anthropic` (`claude`) | `ANTHROPIC_API_KEY` | `langchain-anthropic` | `anthropic:claude-sonnet-5` |
| `openai` | `OPENAI_API_KEY` | `langchain-openai` | `openai:gpt-5` |
| `google_genai` (`gemini`, `google`) | `GOOGLE_API_KEY` | `langchain-google-genai` | `gemini:gemini-2.5-pro` |

- `normalize_spec("gemini:x")` → `google_genai:x`; `claude-cli:` and `scripted:` are
  untouched.
- `provider_ready(spec)` → `(ready, reason)`: key present *and* package importable;
  reasons name the env var and the `uv sync --extra <provider>` fix.
- `build_model(spec, **overrides)`: `init_chat_model(model, model_provider=provider)`;
  `@effort` → `reasoning_effort` for OpenAI, `thinking` budget for Anthropic
  (`low|medium|high` → 1024/4096/16000 tokens), ignored (with a note in `check`) for
  Gemini. `claude_cli.model_from_spec` delegates to it for API providers.
- `pyproject.toml` extras: `anthropic`, `openai`, `gemini`, `llm` (all three), `ui`
  (streamlit, pandas, pytest-playwright). Core deps unchanged.
- `evalbuilder check` gains `providers: {name: {key: bool, package: bool, ready: bool}}`.
- `.env.example`, config template, config reference, pipeline skill and README document it.

## 2. Artifact naming convention (`evalbuilder/pipeline/layout.py`)

One registry is the source of truth for filenames and schema ids; stages, report,
CLI, UI loader and docs read it. Rules: root-level artifacts are `<kind>.json`;
per-run artifacts are `results/<kind>-<run_id>.json`; every JSON artifact embeds
`"schema": "evalbuilder/<kind>/v1"`; kinds are unique.

| kind | file | schema | stage |
|---|---|---|---|
| agent_map | `agent-map.json` | `evalbuilder/agent-map/v1` | discover, map |
| applicable_failures | `applicable-failures.json` | `evalbuilder/applicable-failures/v1` | map |
| mock_rules | `mock-rules.json` (was `mocks.json`) | `evalbuilder/mock-rules/v1` | mocks |
| coverage_plan | `coverage-plan.json` (was `plan.json`) | `evalbuilder/coverage-plan/v1` | dataset |
| dataset | `dataset.json` | `evalbuilder/dataset/v1` | dataset, review |
| coverage | `coverage.json` | `evalbuilder/coverage/v1` | dataset, review |
| evaluators | `evaluators.yaml` | – | score |
| run | `results/run-<id>.json` | `evalbuilder/run/v1` | run |
| score_report | `results/score-report-<id>.json` (was `report-<id>.json`) | `evalbuilder/score-report/v1` (was `report/v1`) | score |
| aggregate | `aggregate.json` | `evalbuilder/aggregate/v1` | aggregate |
| scenarios | `scenarios.yaml` | – | simulate |
| simulation | `simulation.json` | `evalbuilder/simulation/v1` | simulate |
| analysis | `analysis.json` | `evalbuilder/analysis/v1` | analyze |
| pipeline_state | `state.json` | `evalbuilder/pipeline-state/v1` | engine |
| pipeline_report | `report.json` | `evalbuilder/pipeline-report/v1` | report |

Dict-shaped artifacts are wrapped: `mock-rules.json` → `{"schema", "tools": {...}}`,
`applicable-failures.json` → `{"schema", "failure_types": {...}}`. Loaders accept the
old unwrapped shape and the old `report/v1` schema id so existing output directories
still open. The standalone `evalbuilder score` writes `score-report-<id>.json` too.

## 3. Streamlit UI (`evalbuilder/ui/`)

- `loader.py`: `load_dir(path)` reads by filename via the registry; `load_files(uploads)`
  identifies uploaded JSON/YAML by embedded `schema` (filename pattern as fallback).
  Result: `Bundle` with one attribute per kind (None when absent) and `sources`.
- `app.py`: sidebar picks an output directory (scans `eval/pipeline/*` and
  `docs/examples/*`, free-text path, or file upload); pages via `st.navigation`.
- Pages (`ui/pages/*.py`, each a `render(bundle)`):
  1. Overview — verdict, overall score vs threshold, metric pass-rate chart vs
     thresholds, coverage %, stability, stage timeline, problems.
  2. Agent — graph (DOT via `st.graphviz_chart`, AST + live edges), node prompts,
     tools with arg schemas, constraints.
  3. Intents & scenarios — intents, scenarios by intent, failure scenarios, applicable
     failure types, topics, evidence.
  4. Dataset & mocks — case table with filters, case detail (inputs, references,
     metadata, per-case mocks), review status counts, mock rules per tool.
  5. Coverage — plan cells vs achieved, by-kind chart, gaps.
  6. Eval results — per-run metrics, aggregate metrics vs thresholds, slice heatmaps,
     per-case score matrix across runs, failing cases with judge comments, case
     drill-down with trajectory transcript and tool calls for each run.
  7. Stability — unstable cases / outputs / evaluators, suspect judge comments.
  8. Simulation — scenarios, transcripts, violations, stop reasons.
  9. Analysis — summary, verdict explanation, failure patterns, weak slices,
     recommendations, evaluator issues.
  10. Stages & problems — state table, generator calls, problems list.
- Every page has stable `data-testid`-like anchors (headings and `key=` widgets) for
  Playwright.
- `evalbuilder ui [DIR] [--port]` runs `streamlit run` on the app.

## 4. Tests

- Unit: providers (normalize, readiness with monkeypatched env/packages, model build
  for OpenAI without network), layout uniqueness + wrap/unwrap compatibility, loader
  (dir + uploads), page renderers via `streamlit.testing.v1.AppTest` (headless, no
  browser).
- Playwright (`tests/ui/test_playwright.py`, pytest-playwright): session fixture
  starts `streamlit run` on a free port against `docs/examples/support-bot/`, waits
  for readiness, then walks every page asserting key content; skipped when
  playwright/browsers are missing. `uv run playwright install chromium` documented.
- Fixture: `docs/examples/support-bot/` — the live support-bot pipeline output renamed
  to the new convention, with customer emails scrubbed.

## 5. README

Rewrite: what it is, architecture, install (extras), quickstart, skills, pipeline
(stages, config, models/providers, review gate, resume), artifacts table (from the
registry), UI, CLI reference, dataset format, mocking, target contract, testing
(pytest + Playwright), troubleshooting, extending.
