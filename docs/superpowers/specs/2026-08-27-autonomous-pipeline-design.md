# Autonomous eval pipeline for LangGraph agents — design

Date: 2026-08-27. Builds on `2026-08-23-eval-builder-skills-design.md` (the four
skills + `evalbuilder` CLI). Written under the `/goal` directive, i.e. without a
synchronous approval round; decisions that would normally be questions are listed in
§9 as explicit assumptions.

## 1. Goal

One config file in, one JSON report out. `evalbuilder pipeline run eval/pipeline.yaml`
walks every stage the four skills describe — discover → map (intents/scenarios/
failures) → dataset with obligatory tool mocks → repeated runs → scoring →
aggregation → analysis → report — for **any** LangGraph agent exposing the target
contract, with no human in the loop except the decisions recorded in the config.

Non-goals: replacing the skills (they remain the interactive path), a UI, CI wiring.

## 2. Config file — `evalbuilder/pipeline-config/v1`

Minimal viable config (everything else has defaults):

```yaml
schema: evalbuilder/pipeline-config/v1
name: support-bot
target:
  source: examples/support_bot/agent.py      # agent source file (AST discovery)
  module: examples.support_bot.agent         # importable module (live discovery + runs)
  factory: build_agent                       # root graph factory ("root node path")
models:
  agent: claude-cli:sonnet                   # injected into build_agent(model=...)
  judge: claude-cli:sonnet                   # LLM-as-judge
  generator: claude-cli:sonnet               # intents/scenarios/cases/analysis author
review:
  auto_approve: true                         # explicit, recorded authorization
  approved_by: "vladimir"
```

Full surface (with defaults):

| Section | Keys | Purpose |
|---|---|---|
| `constraints` | list of free-text strings | Test constraints / behavioral rules; reach every generation prompt and become `contract` references |
| `coverage` | `total_cases: 20`, `per_intent: {happy: 2, failure: 1}`, `per_failure_category: 1`, `out_of_intent: 2`, `multi_turn_share: 0.15` | Coverage plan; `total_cases` is a floor, per-cell minimums are hard requirements |
| `evaluators` | list of evaluator specs (same shape as `evaluators.yaml`) | Chosen evals: deterministic, OpenEvals prompt judges, AgentEvals trajectory evaluators |
| `thresholds` | `metrics: {contains: 0.9, …}`, `default: 0.8`, `slice_min: 0.5`, `overall_pass: 0.8` | Pass/fail policy per metric, per slice, overall |
| `runs` | `repeats: 3`, `max_case_seconds: 120` | Repeated execution for stability detection |
| `mocking` | `required: true`, `on_miss: strict` | Obligatory mocking of every introspected tool; unmatched call = error, never a real call |
| `stages` | `skip: []`, `max_retries: 1`, `simulate: true`, `publish: auto` | Stop/skip policy and self-recovery budget |
| `output` | `dir: eval/pipeline/<name>` | Where artifacts and `report.json` go |
| `langsmith` | `dataset_name: null` | Optional publication target |

Model spec strings: `claude-cli:<model>[@effort]` (Claude Code CLI, subscription
auth), or any `init_chat_model` string (`anthropic:…`, `openai:…`). `scripted:` is
reserved for tests.

## 3. Architecture

```
eval/pipeline.yaml ──► PipelineConfig (pydantic, validated)
                              │
                     PipelineRunner (stage engine)
   preflight ─► discover ─► map ─► mocks ─► dataset ─► review ─► verify
        ─► run×N ─► score×N ─► aggregate ─► simulate? ─► publish? ─► analyze ─► report
                              │
             state.json (per-stage status, artifacts, attempts, errors)
                              │
                       report.json (evalbuilder/pipeline-report/v1)
```

Modules (all under `src/evalbuilder/`):

- `claude_cli.py` — `ChatClaudeCLI`, a `langchain_claude_code.ChatClaudeCode`
  subclass whose transport is `claude -p --output-format json` (subprocess, isolated:
  `--tools ""`, `--setting-sources ""`, `--strict-mcp-config`,
  `--no-session-persistence`, neutral cwd, `ANTHROPIC_API_KEY` scrubbed so the
  subscription is used). Tool calling and `with_structured_output` both ride on
  `--json-schema`; conversation history (incl. tool calls/results) is flattened into a
  transcript because the CLI cannot replay assistant/tool turns. `model_from_spec()`
  resolves spec strings for agent, judge and generator.
- `pipeline/config.py` — `PipelineConfig` + loader + `init` template.
- `pipeline/engine.py` — `Stage`, `StageResult`, `PipelineState`, `PipelineRunner`:
  dependency-aware stop/skip, per-stage retry with a recovery hook, persisted state,
  `--resume`.
- `pipeline/generator.py` — the "author": prompts + JSON schemas for map, mocks,
  cases, self-review, scenarios, analysis. Every LLM output is validated by the same
  code the CLI uses (`agent_map update` checks, `artifacts.add_case`,
  `validate_dataset`, coverage math); validation errors are fed back for one repair
  attempt.
- `pipeline/stages.py` — the stage functions, thin glue over existing modules
  (`discover`, `mocking`, `runner`, `evaluators`, `simulate`, `langsmith_io`).
- `pipeline/aggregate.py` — repeat-aware aggregation: per metric mean/min over
  repeats, pass-rate vs thresholds, slices, stability classification, coverage
  achieved vs planned, overall verdict.
- `pipeline/report.py` — assembles `report.json`.
- CLI: `evalbuilder pipeline init|run|report`.
- Skill: `skills/agent-eval-pipeline/SKILL.md` (5th skill) — when to use the
  autonomous path vs the interactive skills, how to read the report.

Existing modules gain small extensions: `target.build_graph(model=…)`,
`runner.run_dataset(on_miss=…, model=…)`, evaluator types `openevals` (any
`openevals.prompts` rubric), `trajectory_llm` (AgentEvals), judge resolution via
`model_from_spec`, `Settings.agent_model`/`generator_model`.

## 4. Stages

| Stage | Depends on | Does | Recovery on failure |
|---|---|---|---|
| preflight | – | validate config, `capability_check`, `claude --version` if any `claude-cli:` model, target import, source exists | none (blocking) |
| discover | preflight | AST + live discovery → `agent-map.json` | none |
| map | discover | generator authors intents (each with ≥1 happy + ≥1 failure scenario), gated failure scenarios, topics (only from constraints/source; else `unspecified`), constraints merged from config | validation errors → one repair prompt |
| mocks | discover | introspect `TOOLS` (name, description, args schema); generator writes realistic default fixtures per tool; deterministic fallback fixture guarantees every tool has a wildcard rule | fallback fixtures |
| dataset | map, mocks | coverage plan from config × map → generator writes cases in batches per cell; each case normalized via `artifacts.add_case`; per-case mocks for failure injections; gaps re-filled once | repair batch for invalid cases; residual gaps reported |
| review | dataset | generator self-reviews each case (answerable? cell matches? mocks consistent?) → rejects flagged cases; approves the rest **only if** `review.auto_approve` | without auto_approve: stop with `awaiting_review` |
| verify | review | `mock verify` + every tool has a rule when `mocking.required` | none |
| run | verify | `runs.repeats` × `run_dataset(mocked=True, on_miss, model=agent)` | infrastructure errors → retry once |
| score | run | each run scored with configured evaluators (judge = `models.judge`) | judge unavailable → deterministic-only, flagged |
| aggregate | score | §5 | none |
| simulate | review (optional) | generator writes one multi-turn scenario per intent; `simulate_scenario` with user model; violations mined as pending cases | skipped on error |
| publish | review (optional) | LangSmith publish when key configured | skipped |
| analyze | aggregate | generator reads the aggregate + failing cases → structured analysis (patterns, weak slices, unstable cases, recommendations) | deterministic analysis fallback |
| report | always | `report.json` — even after early stop, with `problems[]` | – |

Stop/skip rules: a failed stage marks every dependent stage `skipped` with
`reason: dependency failed: <stage>`; `stages.skip` marks stages `skipped` by config;
`report` always runs. Every stage records `status ∈ {ok, recovered, failed, skipped,
awaiting_review}`, attempts, elapsed seconds, error text, artifact paths.

## 5. Aggregation and stability

For each `(case, metric)` across repeats: scores `s_1..s_N`.

- `pass_rate` per metric = mean over cases of mean over repeats.
- **Unstable case** = the tool trajectory (names + args, error state) differs across
  repeats — agent behavior. With the same trajectory, a flipping deterministic metric
  is an **unstable output** (wording drift) and a flipping judge metric is an
  **unstable evaluator** (judge variance, not an agent defect). Final-text differences
  alone are only counted (`text_varies`); LLM agents reword freely. Judge rationales
  that look like placeholders are retried once and, if they persist, listed as
  `suspect_judge_comments`.
- Threshold check per metric (`thresholds.metrics[m]` or `thresholds.default`),
  per slice (`slice_min` over intent / failure_mode / variant), overall score = mean
  of metric pass-rates vs `overall_pass`. Verdict `pass` only if all three hold and no
  stage failed; otherwise `fail` (thresholds) or `incomplete` (stage failure).
- Coverage achieved vs planned per cell class (intent-happy, intent-failure,
  failure-category, out-of-intent, multi-turn).

## 6. Example target: `examples/support_bot`

Customer-support agent with **external** tools: `lookup_order` (HTTP GET to a
non-routable host), `check_refund_policy`, `issue_refund` (side-effecting POST),
`search_kb` (external search). Graph: `classify` (structured-output router) →
conditional edges to `support_agent` (ReAct: orders/refunds), `kb_agent` (ReAct:
product questions), `decline` (deterministic out-of-scope node). Exposes
`TOOLS`, `build_agent(model=None, tools=None)`; default model is scripted so the
repo test-suite stays offline, and the pipeline injects `claude-cli:` models.

## 7. Testing

- Unit: config defaults/validation; engine stop/skip/retry/resume; aggregate math
  (thresholds, slices, stability classification); `ChatClaudeCLI` prompt flattening,
  tool-call parsing, structured output (subprocess stubbed); generator validation
  loop with a fake generator.
- Offline e2e: full pipeline on `support_bot` with `scripted:` models and a canned
  generator → `report.json` with verdict computed.
- Live e2e (manual, documented in README): `claude-cli:sonnet` on `support_bot`;
  report saved under `docs/examples/`.
- Skills lint updated to 5 skills.

## 8. Report schema — `evalbuilder/pipeline-report/v1`

```
{schema, name, generated_at, config: {...as loaded...}, verdict, overall_score,
 stages: {stage: {status, attempts, seconds, error, artifacts}},
 agent: {module, tools, nodes, intents, scenarios, failure_types},
 coverage: {planned, achieved, gaps},
 runs: [{run_id, path, report_path}],
 metrics: {m: {pass_rate, min_over_repeats, threshold, passed, n}},
 slices: {dim: {value: {m: pass_rate, passed}}},
 stability: {unstable_cases: [...], unstable_evaluators: [...], repeats},
 cases: [{id, intent, scenario, failure_mode, variant, scores: {m: [s1..sN]}, outputs_hash: [...]}],
 simulation: {...}|null, publication: {...}|null,
 analysis: {summary, failure_patterns, weak_slices, unstable, recommendations},
 problems: [{stage, severity, message}]}
```

## 9. Assumptions (would have been questions)

1. `review.auto_approve: true` in the config is the explicit human authorization the
   skills require; the pipeline never approves without it (it stops with
   `awaiting_review` and still writes the report).
2. `claude-cli:` models run through the `claude` binary with the user's subscription;
   `ANTHROPIC_API_KEY` is removed from the child environment on purpose.
3. Mock fixtures are static responses (existing rule semantics); arg-dependent
   fixtures are expressed as multiple `matchArgs` rules.
4. The generator is the same model family as the judge by default; the config can
   split them.
5. Repeats default to 3; stability is binary (any trajectory/score disagreement), no statistical test.
6. Out-of-intent cases use `intent: out-of-scope`, `failure_mode: out_of_scope`.
