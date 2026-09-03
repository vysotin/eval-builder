# The autonomous pipeline (`src/evalbuilder/pipeline/`)

One config in, one `report.json` out. An LLM **generator** plays the author role the
interactive skills give to a human-guided model; every artifact it produces still goes
through the same validation code, and the review gate is preserved as a decision
recorded in the config.

```
config.py      PipelineConfig (evalbuilder/pipeline-config/v1), template, to_yaml, guidance
engine.py      Stage / PipelineRunner / PipelineState — deps, retries, skip, stop, resume
stages.py      the 14 stage functions + PipelineContext (lazy, disk-backed artifacts)
generator.py   Generator (structured-output wrapper) + prompts/schemas + validation/repair
planning.py    coverage cells (happy / failure / category / out-of-intent / schema-edge)
taxonomy.py    failure types gated by agent structure
aggregate.py   repeat-aware aggregation: pass rates, thresholds, slices, stability, verdict
report.py      report assembly + run_pipeline entry point
layout.py      THE artifact naming convention (kinds, files, schema ids, legacy names)
jobs.py        background runs for the UI (work/job.json + pipeline.log, wrapper entry point)
setup.py       interactive setup helpers (target discovery/preview, config from form, review loop)
```

## Config (`config.py`)

`PipelineConfig` is a strict pydantic model (`extra="forbid"`) with these sections and
defaults: `target{source, module, factory=build_agent, root_node}`, `models{agent=None,
judge, generator}` (default `claude-cli:claude-sonnet-5`), `constraints[]`,
`instructions` (free text), `feedback[{at, note, from_stage}]`, `models.mock` (the
LLM mock engine's model, default the generator), `coverage{total_cases=20,
per_intent{happy=2, failure=1}, per_failure_category=1, out_of_intent=2,
multi_turn_share=0.15, per_tool_edge_cases=2}`, `evaluators` (default expected_tools,
contains, contract, correctness), `thresholds{default=0.8, metrics{}, slice_min=0.5,
overall_pass=0.8}`, `runs{repeats=3, parallel_intents=4, parallel_scoring=4, parallel_simulations=4}`, `mocking{required=true,
on_miss=strict|llm|fallback|real, strategies=true, strategy=default, on_invalid=fallback|strict, max_repairs=1}`,
`stages{skip[], max_retries=1, simulate=true, publish=auto}`, `review{auto_approve=false,
approved_by, note}`, `output{dir}`, `langsmith{dataset_name}`.

`problems()` adds the semantic checks a schema cannot express (source exists, ranges,
known stage names, `report` never skipped, evaluators have a type, model specs look
like `provider:model`, `auto_approve` requires `approved_by`). `guidance()` renders
`instructions` + `feedback` into the `USER INSTRUCTIONS` / `REVIEWER FEEDBACK` block
every generator prompt receives. `to_yaml()` writes the config back with a comment per
section (`SECTION_COMMENTS`) so the UI can show and edit the same text the CLI accepts;
`parse_config`/`load_config` return one line per validation error.

**Decision — the config is the human.** `review.auto_approve: true` + `approved_by`
is the explicit authorization the skills require; feedback entries are appended, never
overwritten, so the history of what the reviewer asked for stays in the file.

## Stage engine (`engine.py`)

`Stage(name, fn, deps, optional, always, recover, retries)`; `PipelineRunner(stages,
state_path, skip, max_retries, resume, invalidate_from, stop_after)`. For each stage in
order: cached on `--resume` when already ok (unless `always`); `skipped` when in the
skip set, when the pipeline stopped earlier (`awaiting_review` or `stop_after`), or
when a dependency did not succeed (reason recorded); otherwise executed with retries
and an optional recovery hook whose note is stored. Statuses: `ok`, `recovered`
(succeeded on a retry), `failed`, `skipped`, `awaiting_review` (a deliberate
`StageStop`). `PipelineState` (`work/state.json`, `evalbuilder/pipeline-state/v1`) is saved
after every stage and carries `data.problems` and `data.stopped_after`.

**Reasoning.** Failures never raise out of the engine: the report must always land,
and dependents must be marked rather than silently skipped. `--resume --from STAGE`
pops that stage and everything after it from the state so a fix (thresholds, agent
code, feedback) reruns exactly what it invalidates; `--until STAGE` stops deliberately
with a distinguishable reason and exit code 0 so the UI's "generate only" flow is not
reported as a failure.

## The stages (`stages.py`)

| stage | deps | does | recovery |
|---|---|---|---|
| preflight | – | `config.problems()`, `capability_check` (target importable), `provider_ready` for agent/generator/mock (blocking; mock only under `on_miss: llm`) and judge (degraded → problem) | none |
| discover | preflight | AST discovery (tools, prompts, skills) + live graph + `tool_schemas.describe_tool` for every live tool (schemas, models, side effects, kind/mockable, edge cases; cleared when `per_tool_edge_cases: 0`) + live `SKILLS` merged by name → `agent-map.json` | none |
| map | discover | `taxonomy.applicable_failure_types` (incl. `skill_misuse` when the map has skills) → `agent-map.json` `applicable_failures` (saved before the generator runs, so a generator failure keeps it); generator authors intents, scenarios (with the `skills` they exercise), failure scenarios, per-skill `skill_failures` (failure cases beyond tool failure, merged into `skills[].failure_cases`), per-tool `tool_failures` (merged into `tools[].failure_scenarios`), topics, derived constraints; `_validate_map` drops invalid entries (bad slugs, unknown intents, non-applicable failure types, missing evidence, unknown skill names) → problems; a skill no scenario exercises is reported; constraints merged with the config's | one repair prompt; retry once |
| mocks | discover | generator writes a default fixture + variants per **mockable** tool; responses validated against `output_schema` (bad defaults replaced by a schema-conformant sample, bad variants dropped, all reported); every tool ends with a wildcard rule — except under `on_miss: llm` → `work/mock-rules.json`; then (`mocking.strategies`, always under `llm`) the strategies document → `work/mock-strategies.json` (validated: unknown tools dropped, invalid fallbacks replaced, invalid examples dropped, a `default` entry synthesised for every tool); fails when `mocking.required` and a tool has neither rules nor (under `llm`) a default-strategy behaviour | generic/schema fixtures, generic default strategy |
| dataset | map, mocks | `planning.plan_cells` → `dataset.coverage.plan`; the dataset's `mocks` block takes the `work/` handoff — rules, policy, `llm` settings and strategies — so a stale dataset can never shadow this run's rules; generator fills cells in batches of 6 with one re-request for missing cells (it sees the strategies and may put a case under one — `mock_strategy`, validated); each case normalised via `artifacts.add_case` (duplicates dropped, reported); per-case mock overrides get dataset fallbacks; `malformed_output` cases get a corrupted fixture injected by code → `dataset.json`, whose `coverage.achieved` (with per-skill counts) the review stage recomputes over the approved cases | retry once |
| review | dataset | generator self-review rejects unanswerable / mismatched cases; `verify_summary` rejects cases whose expected calls no rule answers (not under `llm`); approves the rest **only with** `review.auto_approve` (note records `approved_by`); no pending cases + approved cases present → `already_reviewed` | without auto_approve → `awaiting_review` (pipeline stops, report still written) |
| verify | review | approved cases' expected calls all mocked (under `llm`: counted as `llm_answered_calls`); every mockable `TOOLS` entry covered by a rule or, under `llm`, the default strategy; an `llm` dataset names a mock model; at least one approved case | none (blocking) |
| run | verify | `runs.repeats` × `run_dataset(mocked=True, on_miss, model=agent, mock_model, strategy, max_workers=runs.parallel_intents)` — intent groups run concurrently within each repeat (cases inside one intent stay sequential, results keep dataset order); under `llm` one `LLMMockEngine` per case (the case's strategy, else the config's); live progress (current repeat, per-case completions, per-intent tallies, mock-call totals) is written to `work/run-progress.json` after every case; per-layer totals land in the stage details (`mocking.calls`) and invalid engine answers become a problem; a run where every case is an infrastructure error fails the stage | retry once |
| score | run | `evaluators.yaml` written; `score_run(max_workers=runs.parallel_scoring)` per run — case runs scored concurrently in a thread pool, rows/metrics/slices aggregated in run order so the report is identical to a sequential pass → `results/score-report-<id>.json`; dead evaluators (all errors) reported; no scores at all → failure | retry once |
| aggregate | score | `aggregate.aggregate` → `aggregate.json` | none |
| simulate | review (optional) | generator writes scenarios (optionally `mock_strategy`) → `scenarios.yaml`; runs them with mocked tools (under `llm` an engine per scenario honouring its strategy; results carry `mock_calls`) and the generator model as the simulated user — scenarios run concurrently (`runs.parallel_simulations`, a fresh graph per scenario, results keep scenario order) → `simulation.json`; violations mined into pending cases | never blocks the verdict |
| publish | review (optional) | LangSmith publish (`auto` = only with a key) | skipped |
| analyze | aggregate (optional) | generator writes the analysis from a compact summary; falls back to a deterministic facts-only analysis → `analysis.json` | deterministic fallback |
| report | always | `report.json` (`evalbuilder/pipeline-report/v1`) | – |

`PipelineContext` lazily reloads every artifact from disk (by kind, with legacy names)
so a resumed run sees the same data as a fresh one; `problems` live in `state.data`
so they survive resumes. The generator model, the agent model and the mock model are
created once per run (`model_from_spec`; the mock model reuses the generator's object
when the specs match).

## The generator (`generator.py`)

`Generator.ask(system, user, schema)` calls `model.with_structured_output(schema)` and
records `{schema, seconds}` per call (the report lists them). Every authoring step is
prompt + JSON schema + validator:

- **map** — `MAP_SYSTEM` asks for intents (2–6, each with happy and failure
  scenarios), taxonomy-gated failure scenarios, topics only when visible in evidence,
  and prompt-stated constraints; the agent brief includes nodes, edges, live graph,
  tools with both schemas and side effects, constraints, and the source. Invalid
  entries are fed back once ("YOUR PREVIOUS ANSWER HAD THESE PROBLEMS") then dropped.
- **mocks** — realistic defaults + 1–3 variants keyed by arguments, self-consistent
  across tools, no error responses (errors are per-case injections), conformant to
  `output_schema`.
- **cases** — `CASES_SYSTEM`: fill each cell exactly, realistic evolved inputs using
  fixture entities, ordered `expected_tools` (args as subsets) and `forbidden_tools`,
  a `contains` literal only when guaranteed by fixtures, a one-sentence `contract`,
  multi-turn `user_turns`, failure-mode crafting via inputs or `mock_overrides`, and
  the schema-edge rules (omit / mistype / overflow the named field; malformed output
  is a normal request whose fixture the pipeline corrupts). Cases beyond a cell's
  count are trimmed so the plan, not the model, decides the size.
- **self-review** — reject when unanswerable with the fixtures, contradicting the
  prompts, mismatching the cell, or a `contains` literal not guaranteed by fixtures.
- **scenarios** — one multi-turn scenario per intent with a success literal.
- **analysis** — separates agent defects, evaluator/mock problems and instability;
  cites case ids; actionable recommendations.

`with_guidance` appends `config.guidance()` to every prompt, so user instructions and
reviewer feedback influence every regeneration.

**Reasoning.** The LLM never writes an artifact: it returns structured JSON that the
same functions the CLI uses validate (`add_case`, `validate_dataset`, taxonomy gating,
schema validation). One repair round is enough to fix most shape problems; anything
still invalid is dropped *and listed in `problems`*, which is the difference between
an honest report and a silently thinned dataset.

## Coverage planning (`planning.py`) and taxonomy gating (`taxonomy.py`)

`plan_cells(coverage, agent_map, failure_types)`:

1. per intent: one cell per happy scenario (`per_intent.happy` spread across them)
   and per failure scenario (`per_intent.failure`), topics rotating across cells (they
   never multiply counts — the LLM invents topics otherwise);
2. per applicable failure type a cross-cutting `category` cell
   (`per_failure_category`), `out_of_scope` always present as `out-of-intent`
   (`max(per_failure_category, out_of_intent)`);
3. per tool the first `per_tool_edge_cases` deterministic edges as `schema-edge`
   cells (count 1 each; `related_intent` is the intent whose evidence cites the tool);
4. `total_cases` as a floor filled with extra happy variants; `multi_turn_share` marks
   a slice of intent cases as multi-turn.

`achieved(cells, cases)` compares planned vs present per cell key and per kind; gaps
are listed, never hidden.

`applicable_failure_types` maps the taxonomy to evidence: `input_validation`,
`provider_error`, `out_of_scope` always; `branch_misrouting` with conditional edges;
`tool_misuse`, `tool_error_handling`, `prompt_injection` with tools;
`retrieval_grounding` with search/lookup tools or a retriever in the source;
`state_loss` with multi-turn or a checkpointer; `output_contract_violation` with
structured output; `constraint_violation` with constraints. The generator may only
use these; the `map` validator drops the rest.

## Aggregation, stability, verdict (`aggregate.py`)

For each `(case, metric)` across `repeats`: `pass_rate` = mean over cases of the
mean over repeats, `min_over_repeats`, error counts, threshold and pass flag per
metric (`thresholds.metrics[m]` or `default`). Slices by `intent`, `failure_mode`,
`variant` with `slice_min`. Stability classification (spec 2026-08-27 §5):

- **unstable case** — the tool trajectory hash (names + args + error state) differs
  across repeats: the agent's *behaviour* changed;
- **unstable output** — same trajectory, a deterministic metric flipped (wording drift);
- **unstable evaluator** — same trajectory, a judge flipped (judge variance, not an
  agent defect);
- `text_varies` only counts final-text differences (LLM agents reword freely);
- `suspect_judge_comments` lists placeholder-like rationales.

Verdict `pass` only if every metric passes, no slice is weak and the overall score
(mean of metric pass rates) ≥ `overall_pass`; otherwise `fail`. The report turns a
required stage that did not succeed into `incomplete` (`report._verdict`).

**Reasoning.** Repeats are the only honest way to tell an agent defect from judge
noise without a statistical model; hashing the trajectory rather than the text keeps
wording drift from masquerading as instability. Thresholds are per metric / per slice
/ overall because a defect usually lives in one slice and the aggregate hides it.

## Report (`report.py`)

`build_report` assembles verdict + reasons, per-stage status (attempts, seconds,
error, reason, artifacts, details, recovery notes), the agent view (tools, nodes,
intents, scenarios, failure scenarios, constraints), coverage, the run/score pairs,
metrics, slices, stability, per-case rows, failing cases with judge comments,
simulation, publication, analysis, generator calls, the loaded config and
`problems[]` (stage problems plus failed/awaiting/dependency-skipped stages).
`analysis_summary` is the compact view the analyze stage sends to the generator.
`run_pipeline` wires config → context → runner and always leaves a report behind
even if the report stage itself failed. `summary_text` is the terminal rendering.

## Artifact naming convention (`layout.py`)

One registry (`ARTIFACTS`) is the source of truth for kind → tier → file → schema id →
writing stage → description, with legacy names/ids kept readable: root artifacts are
`<kind>.json`/`.yaml`, per-run artifacts `results/<kind>-<run_id>.json`, every JSON
artifact embeds `"schema": "evalbuilder/<kind>/v1"`. Dict-shaped payloads are wrapped
(`mock-rules.json` → `{"schema", "tools"}`, `applicable-failures.json` →
`{"schema", "failure_types"}`) so the schema id has a place. `identify()` recognises a
file by embedded schema first, then by name — that is what lets the UI load uploaded
files. Kinds: agent_map, applicable_failures, mock_rules, coverage_plan, dataset,
coverage, evaluators, run, score_report, aggregate, scenarios, simulation, analysis,
pipeline_state, pipeline_report, pipeline_job.

Two orthogonal fields keep the output directory free of duplicated JSON:

- **`tier`** says where the file goes. `final` — a deliverable at the root (or under
  `results/`). `work` — scratch in `<out_dir>/work/`: `state.json`, `run-progress.json`,
  `job.json`, and the `mock-rules.json` / `mock-strategies.json` handoff the mocks stage
  leaves for the dataset stage. Nothing downstream of the pipeline reads `work/`, so the
  whole directory is deletable. `derived` — no file at all; `path_for` raises for these,
  which is what stops a stage from writing a copy by accident.
- **`folded_into`** is the dotted path inside another artifact that durably holds this
  kind's data: `applicable_failures` → `agent_map.applicable_failures`, `coverage_plan` /
  `coverage` → `dataset.coverage.plan` / `.achieved`, `mock_rules` / `mock_strategies` →
  `dataset.mocks.tools` / `.strategies`. Every derived kind has one; a work kind may
  too, and that is exactly why deleting `work/` loses nothing.

Folded kinds stay in the registry so pre-fold directories and single-file uploads keep
resolving. `fold_from(kind, parents)` reads one out of its parent — the UI's `Bundle`
uses it so pages never learn about the fold, and `PipelineContext` uses the same idea
for `applicable()`, `cells()`, `coverage()`, `mock_rules()` and `mock_strategies()`.
`compact_dir(out_dir)` (CLI: `evalbuilder pipeline compact`) migrates an older directory:
fold the copies in, move the scratch, re-index the report, keeping the recorded
`output_dir` prefix because those paths are provenance. It is idempotent.

## Background jobs (`jobs.py`)

The UI never runs the pipeline in the Streamlit process. `start_job(config, out_dir,
mode, from_stage)` writes `work/job.json` (`evalbuilder/pipeline-job/v1`: mode, argv, pid,
timing, exit code, log path) and spawns a detached wrapper (`python -m
evalbuilder.pipeline.jobs run job.json`, `start_new_session=True`) which runs
`python -m evalbuilder.cli pipeline run …` with stdout/stderr appended to
`pipeline.log` and then finalises `job.json` with the exit code. Modes: `full`,
`dataset` (`--until dataset`), `resume`, `regenerate` (`--resume --from STAGE --until
dataset`). `job_status` merges `work/job.json`, process liveness and `work/state.json` into
`running | finished | lost | none`, plus the stage table and `stopped_after`. A second
start on a running job is refused.

**Reasoning.** A detached child with its own record means the run survives page
reloads, Streamlit reruns and even a restarted server, and the exit code is correct
without keeping a handle in memory.

## Interactive setup helpers (`setup.py`)

Pure Python behind the *Pipeline setup* / *Run & review* pages: `discover_targets`
(any `*.py` under `examples/` defining `TOOLS` and `build_agent`, with the scripted
model spec and `pipeline.yaml` when present), `preview_target` (AST + live schemas +
edge cases, no LLM), `default_form`/`build_config`/`validate_text` (form ↔ config ↔
YAML), and the review loop: `add_feedback` (append to `feedback` and save),
`approve_in_config` (record `auto_approve` + `approved_by`), `reject_cases`
(`artifacts.set_review` on the dataset), `dataset_summary` (statuses, failure modes,
schema-edge cases, mocked tools). Projects: `discover_projects` (every directory under
`eval/pipeline/*` and `docs/examples/*` holding an artifact or a job record, plus
`<root>/*.yaml` configs whose output directory has not run yet), `project_config`
(the config behind a folder: the path recorded by the last job, then by the report —
falling back to the config embedded in `report.json` — then a sibling `<dir>.yaml`),
`form_from_config` (the inverse of `build_config`, so a saved project can be edited
in the setup page). `build_config` rejects an empty name / source / module.

## Limitations of the pipeline

- The generator is validated **structurally**, not semantically: a case can be
  well-formed and still test the wrong thing; the self-review stage and the human
  review loop exist for that reason.
- Cost and time scale with cases × repeats × judge evaluators; the Claude CLI
  adapter adds seconds per call. A usage-limit hit mid-run surfaces as judge errors
  and failed optional stages — `--resume --from score` repairs it.
- Coverage is planned from the map the generator produced; a poor map (missed
  intents) yields a confidently complete-looking but narrow dataset. Check
  `agent-map.json` first.
- Stability is binary and needs `repeats ≥ 2`; there is no statistical significance,
  and judge calibration is not measured.
- `problems` persist across `--resume` by design, so a repaired stage's earlier
  warnings still appear in the report (with the stage's later status).
- Schema-edge cells cover top-level arguments only, and their expected behaviour is a
  generic sentence per kind — the generator writes the concrete contract.
- Only tool calls are mocked; agents that call external services from nodes are not
  deterministic under the pipeline.
