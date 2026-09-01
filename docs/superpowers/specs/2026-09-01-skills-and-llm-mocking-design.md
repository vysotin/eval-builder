# Agent skills in the agent map + two-layer tool mocking — design

**Date:** 2026-09-01 · **Status:** approved for implementation (autonomous `/goal` run)
· **Builds on:** `2026-08-28-interactive-pipeline-ui-design.md`, `docs/architecture/`

Two additions, delivered together because they touch the same layers (discovery,
generator, runner, config, CLI, skills prose, UI):

1. **Agent skills** — when the target agent uses *Agent Skills* (`SKILL.md` folders with
   frontmatter and instructions, optionally `references/`), discovery records them,
   node prompts that mention or embed them are linked, generation prompts see them,
   a new failure type gates on them, coverage reports them, and the UI shows them.
2. **Two-layer tool mocking** — the deterministic rule matcher stays layer 1; when no
   rule matches and the dataset says `on_miss: llm`, the call goes to an **LLM mock
   engine** that answers from a pre-generated **mock strategy** (a described backend
   world and per-tool behaviour), validates the answer against the tool's declared
   output schema (one repair round), and records every call in a ledger. Strategies
   are generated in the `mocks` stage, embedded in the dataset, and selectable per
   case and per simulation scenario. The config names the model that drives the
   engine.

## 1. Context and constraints

- Target contract stays `TOOLS` + `build_agent(model=None, tools=None)`; discovery
  stays AST-first with live enrichment; every LLM output is validated by code.
- The ADK evaluation framework (reference checked 2026-09-01) **does not inject mock
  tool responses** — its LLM-driven piece is the *user simulator*
  (`conversation_scenario.starting_prompt` + `conversation_plan`, model in
  `user_simulator_config.model`). This design mirrors that shape for tools: a
  described plan (strategy) per case, a model named in config, validated output.
- Anything the LLM mock engine produces is non-deterministic across repeats. The
  ledger makes that attributable: stability reporting names cases whose instability
  coincides with LLM-mocked calls instead of blaming the agent.
- Offline everything: examples get scripted mock models, so the whole pipeline
  (including layer 2) runs in tests without a real model.

## 2. Agent skills

### 2.1 What a skill is

An *Agent Skill* is a directory holding `SKILL.md` with YAML frontmatter (`name`,
`description`, optional `allowed-tools`, `license`, `metadata`) and a markdown body
of instructions, plus optional `references/*.md`, `scripts/`, `assets/`. Agents use
skills in two ways:

| disclosure | how the model sees it | how the pipeline sees it |
|---|---|---|
| `inline` | the body is embedded in a node's system prompt | the node prompt recorded in the map *is* the composed text |
| `on_demand` | the prompt lists name + description; the model calls a loader tool (`load_skill(name)`) to read the body | the loader is a tool of kind `skill_loader`; it is never mocked; `load_skill` appears in expected tool calls |

### 2.2 `evalbuilder.skills` (new module)

```python
@dataclass
class Skill: name, description, path, dir, body, metadata: dict, references: list[dict], scripts: list[str]
load_skills(paths) -> list[Skill]          # every subdir with SKILL.md (or a SKILL.md path)
skills_prompt(skills, loader="load_skill") -> str   # progressive-disclosure listing
skills_inline_prompt(skills, *names) -> str         # full bodies (all, or the named ones)
skill_loader_tool(skills, name="load_skill") -> BaseTool   # returns body + reference list; metadata.kind = "skill_loader"
describe_skill(skill) -> dict               # the agent-map entry
is_skill_loader(tool) -> bool               # metadata.kind or a known loader name
SKILL_LOADER_NAMES = ("load_skill", "read_skill", "get_skill", "use_skill")
```

`references[]` entries are `{path, title, chars, excerpt}` for `references/*.md`
(title = first heading, excerpt = first 300 chars). Frontmatter is parsed with
`yaml.safe_load`; a missing `name` defaults to the folder name.

### 2.3 Discovery

**AST (`discover_from_source`)** recognises, without importing:

- skill directories from module-level assignments: `X = load_skills(<path>)`,
  `SKILLS_DIR = "<path>"` / `Path(__file__).parent / "skills"` /
  `os.path.join(os.path.dirname(__file__), "skills")`, and `skills=[...]` keywords on
  agent factories. Paths resolve relative to the source file, then the cwd.
- **composed prompts**: `_prompt_text` resolves `Constant`, `Name` (module constants),
  `BinOp(+)`, `IfExp` (the true branch), `JoinedStr` (constant parts + `{…}`),
  `.format()/.strip()` receivers, and the helper calls `skills_prompt(...)` /
  `skills_inline_prompt(...)` — rendered from the loaded skills so the recorded prompt
  is what the model sees. Module constants built that way are captured too.
- **loader tools**: `x = skill_loader_tool(...)` assignments and `@tool` functions
  whose name is a known loader name → tool entry `kind: skill_loader`,
  `mockable: false`. Every other tool gets `kind: tool`, `mockable: true`.
- **links**: a node's `skills` = skills whose name appears in its prompt or that were
  rendered into it (`skills_source: inline | listing | prompt`); each skill's
  `used_by` lists those nodes; `tools_mentioned` lists tool names in the body.

**Live** (`preview_target`, pipeline `discover`): a module attribute `SKILLS` (duck-typed
`name`/`description`/`body`) is merged by name; loader tools are classified through
`tool_schemas.describe_tool` (`kind`, `mockable`).

**Agent map** gains `skills: list[dict]` (explicit field; older maps load with `[]`),
`app.skills_dir`, per-tool `kind` / `mockable`, per-node `skills` / `skills_source`.

### 2.4 Downstream

- `agent_brief` adds `skills` (name, description, body ≤ 3000 chars, references,
  used_by) and each tool's `kind`.
- Evidence token `skill:<name>` (taxonomy reference, MAP/CASES prompts, coverage).
- New failure type `skill_misuse` ("Agent ignores, mis-selects or violates an
  applicable skill's instructions"), applicable iff the map has skills; evidence
  `skill:<name>` for every skill.
- `MAP_SYSTEM`: when skills exist, every skill must be exercised by at least one
  scenario; scenarios may list `skills: [names]` (validated against the map; unknown
  names dropped with a problem).
- Coverage (`achieved(cells, cases, skills=)`): `skills: {name: {cases: n}}` and
  `uncovered_skills` — a case covers a skill when its scenario lists it, its evidence
  cites `skill:<name>`, or an expected tool call is `load_skill(name)`.
- Report `agent.skills` (names); `analysis_summary` includes them.
- CLI `evalbuilder discover` prints `skills`.
- UI: Agent page *Skills* section (description, body, references, used-by, allowed
  tools), node expanders show skills, tools table shows `kind`; setup preview shows a
  *Skills* metric + list; sidebar caption counts skills; Coverage page shows the skill
  table when present.

### 2.5 Examples

- `examples/incident_desk` — **inline** disclosure: `skills/incident-triage/SKILL.md`
  (+ `references/severity-matrix.md`) embedded into the triage prompt and
  `skills/incident-comms/SKILL.md` (+ `references/status-update-template.md`) into the
  comms prompt via `skills_inline_prompt`. `TOOLS` unchanged. The offline generator
  cites `skill:` evidence, adds a `skill_misuse` scenario ("skip the status check")
  and its case.
- `examples/support_bot` — **on-demand** disclosure: `skills/refund-policy/SKILL.md`
  and `skills/product-troubleshooting/SKILL.md`; `load_skill = skill_loader_tool(SKILLS)`
  appended to `TOOLS`; `build_agent` appends `skills_prompt(SKILLS)` to the support
  prompt only when the loader is among the injected tools (so unit tests that inject
  four plain tools are unchanged). The scripted model calls `load_skill("refund-policy")`
  first on refund requests when the listing is present; the offline generator expects
  it in `expected_tools`.
- `weather_bot`, `loan_desk`, `travel_planner` stay skill-less (regression path).

## 3. Two-layer tool mocking

### 3.1 Data model

`dataset.mocks`:

```json
{
  "tools": {"<tool>": [{"matchArgs": {...}, "response": ...}]},
  "on_miss": "strict | real | fallback | llm",
  "strategy": "default",
  "llm": {"model": "claude-cli:claude-sonnet-5", "on_invalid": "fallback | strict", "max_repairs": 1},
  "strategies": { "world": "...", "strategies": { "default": {...}, "degraded": {...} } }
}
```

Per case: `metadata.mocks.strategy: "<id>"` (optional). Per simulation scenario:
`mock_strategy: "<id>"` (optional). Dataset validation checks `on_miss` values, that
`llm.model` looks like `provider:model` when `on_miss: llm`, and that every referenced
strategy id exists.

`mock-strategies.json` (new artifact kind `mock_strategies`, schema
`evalbuilder/mock-strategies/v1`, stage `mocks`):

```json
{"schema": "evalbuilder/mock-strategies/v1",
 "world": "One paragraph: the simulated backend every tool shares (entities, ids, invariants).",
 "strategies": {
   "default": {"description": "healthy backend, consistent with the fixtures",
               "tools": {"lookup_order": {"behavior": "…rules an LLM can follow…",
                                          "examples": [{"args": {...}, "response": {...}}],
                                          "fallback_response": {...}}}},
   "degraded": {"description": "…", "tools": {...}}
 }}
```

`fallback_response` and every example response are validated against the tool's
`output_schema`; example args against `args_schema` (partial). Invalid entries are
replaced (fallback → `tool_schemas.example`) or dropped (examples), each with a
recorded problem. The `default` strategy always exists and covers every mockable
tool (a deterministic generic behaviour is synthesised from the tool description when
the generator omits one).

### 3.2 The engine (`evalbuilder.mock_engine`)

```python
class LLMMockEngine:
    def __init__(self, model, strategies: dict, tool_specs: dict[str, dict], *, strategy="default",
                 on_invalid="fallback", max_repairs=1, log=None)
    def respond(self, tool_name: str, args: dict) -> Any
    ledger: list[dict]   # one entry per call: tool, args, strategy, response, valid, repairs, seconds, fallback, error
```

`respond` builds one prompt: the world, the strategy's behaviour for the tool, its
examples, the tool definition (description, `args_schema`, `output_schema`), the last
N calls of this engine (consistency inside a case), and the call itself in a
machine-readable block (`TOOL CALL:\n{json}`). Structured output uses the tool's
object `output_schema` (retitled `mock_response`) or, for tools without one, a
`{"response_json": "<JSON text>"}` envelope. The answer is validated with
`tool_schemas.validate`; on problems the engine re-asks once with the problems
listed; still invalid → `on_invalid: fallback` returns the strategy's
`fallback_response` (ledger `valid: false, fallback: true`), `strict` raises
`MockEngineError` (a run records it as an *infrastructure* error, never an agent
error). One engine per case / per scenario (history is per conversation); the model
object is shared (stateless per call).

### 3.3 The wrapper (`mocking.py`)

`wrap_tool(tool, rules, on_miss, fallback, *, engine=None, ledger=None)`: rules first
(responses deep-copied per call); on a miss, `on_miss == "llm"` → `engine.respond`;
otherwise the existing real / fallback / strict behaviour. `wrap_tools` wraps every
**mockable** tool when an engine is given (tools without rules included), skips
`skill_loader` tools always, and appends a ledger entry per call
(`layer: rule | llm | real | fallback | error`). `mockable(tool)` is the one
predicate (`skills.is_skill_loader`).

`verify_dataset(ds)` is unchanged for the three legacy policies; under `llm` a miss
is not a defect — `verify_summary(ds)` reports `{misses, llm_answered, policy}` and
the CLI/stage treat misses as informational.

### 3.4 Config

```yaml
models:
  mock: null              # model driving LLM mock responses; null = models.generator
mocking:
  required: true
  on_miss: strict         # real | fallback | strict | llm
  strategies: true        # generate mock-strategies.json in the mocks stage (always on when on_miss: llm)
  strategy: default       # dataset-level default strategy id
  on_invalid: fallback    # after the repair round: fallback (schema-conformant) | strict (error)
  max_repairs: 1
```

`PipelineConfig.problems()` requires `on_miss: llm` → a `provider:model` mock model;
`mock_model_spec` property resolves the default. Preflight checks the mock model's
readiness when `on_miss: llm` (blocking).

### 3.5 Stages

- **mocks**: `author_mocks` writes fixtures for mockable tools only; with
  `on_miss: llm` no wildcard default is appended (keyed variants stay, the long tail
  goes to the engine). Then `author_strategies` (when enabled) → `mock-strategies.json`.
- **dataset**: `mocks` block embeds policy, model, strategy and strategies; the
  generator may set `mock_strategy` per case (validated).
- **review**: `verify_dataset` misses auto-reject cases only under non-`llm` policies.
- **verify**: under `llm`, misses are counted (`llm_answered`), and "unmocked tool"
  means *no rule and no strategy entry*; loader tools are exempt everywhere.
- **run**: `run_dataset(..., on_miss, mock_model=, tool_specs=)` builds one engine per
  case with the case's strategy; `CaseRun.mock_calls` holds the ledger; the stage
  sums layers (`rule / llm / invalid / fallback / errors`) into its details and
  `run-progress.json`, and records a problem when invalid responses occurred.
- **simulate**: engine per scenario with `scenario.mock_strategy`; results carry
  `mock_calls` counts.
- **aggregate**: per case `llm_mock_calls`; `stability.llm_mocked_unstable` lists
  unstable cases that had LLM-mocked calls.
- **report**: `mocking` section — policy, model, strategy ids, per-layer totals.

### 3.6 CLI

```
evalbuilder run DATASET --mock --on-miss llm [--mock-model SPEC] [--strategy ID]
evalbuilder simulate DATASET --scenarios F [--mock] [--on-miss P] [--mock-model SPEC]
evalbuilder mock strategies DATASET [--set @file.json] [--model SPEC] [--strategy ID] [--on-miss P]
evalbuilder mock validate DATASET --tool T --response JSON|@file      # output-schema validation
evalbuilder mock try DATASET --tool T --args JSON [--strategy ID] [--model SPEC]   # both layers, once
evalbuilder mock verify DATASET                                        # llm policy: misses informational
```

### 3.7 UI

- Setup: *Mock model* input, miss policy adds `llm`, *On invalid* select, *Generate
  strategies* checkbox (form ↔ config round trip).
- Dataset & mocks: *Mock strategies* section (world, one tab per strategy, per-tool
  behaviour / fallback / examples), case table `strategy` column, case detail shows
  the selected strategy; header shows the policy and model.
- Eval results drill-down: *Mock calls* table per run (layer, tool, args, valid,
  repairs, fallback).
- Summary: a *Mocking* line (policy, model, LLM-answered calls, invalid).
- Run & review: dataset summary lists strategies and the policy.

### 3.8 Skills prose and docs

`agent-eval-discover` (skills, `skill:` evidence, `skill_misuse`), `agent-eval-mock`
(two layers, strategies, `try`/`validate`, when to choose `llm`), `agent-eval-dataset`
(per-case `mock_strategy`), `agent-eval-run` (`--on-miss llm`, ledger), and
`agent-eval-pipeline` + `config-reference.md` (config keys, artifact table).
`docs/tool-mocking.md`, `docs/architecture/*` and the README describe both features;
decision record entries 19 (skills as first-class map entries) and 20 (LLM mocking
as a second layer behind deterministic rules, strategy-driven, schema-validated).

## 4. Testing

- Unit: `test_skills.py` (loading, prompts, loader tool, describe), discovery of both
  disclosure styles and composed prompts, `test_mock_engine.py` (layers, validation,
  repair, fallback/strict, ledger, mockability), generator `author_strategies`
  validation, config/problems, dataset validation, layout kind, runner ledger and
  per-case strategy, aggregate attribution.
- End-to-end (offline): incident_desk and support_bot pipelines with skills (report
  `agent.skills`, coverage `skills`, `load_skill` in trajectories); weather_bot with
  `on_miss: llm` and a scripted mock model (`scripted:examples.weather_bot.offline:mock_model`)
  proving layer-2 calls, validation repair and the report's `mocking` section; a
  failing-validation variant proving `fallback` vs `strict`.
- CLI: `mock strategies/validate/try`, `discover` skills output, `run --on-miss llm`.
- UI: AppTest for the Agent, Dataset and Setup changes; the Playwright flow suite
  extended for skills and strategies where it already runs.

## 5. Out of scope

Skill *execution* (scripts), ADK/Dify adapters, mock outcomes other than `return`
(raise/sequence), and LangChain middleware interception — all noted in
`docs/architecture/07-limitations.md`.
