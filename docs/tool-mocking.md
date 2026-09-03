# How tool mocking works

A walkthrough of the implementation, layer by layer — rule semantics, the LLM mock
engine behind them, when mocks are installed, where rules and strategies come from,
the verification gates, and the decision of *when and how to mock*. Code references
point at the modules that own each part.

## The design stance

Mock data is **data that lives in the dataset** (`dataset.json → mocks`), not code. At
execution time the runner **wraps the agent's tools** so the mock layers answer instead
of the real function — the wrapper keeps the original name, description and
`args_schema`, so the model's tool-*selection* behavior is untouched; only the tool's
*answer* is substituted (`src/evalbuilder/mocking.py`, `wrap_tool`). Two layers answer,
in order:

| layer | answers from | deterministic? | module |
|---|---|---|---|
| 1 — rules | ordered `{matchArgs, response}` rules per tool | yes | `mocking.py` |
| 2 — LLM mock engine | a model playing the backend from a pre-generated **strategy**, validated against the tool's output schema | no (logged, attributable) | `mock_engine.py` |

Layer 2 exists for the long tail: fixtures can name the known entities (order A1234,
service `api`), but an agent under evaluation will also call tools with inputs nobody
enumerated. Under `on_miss: strict` such a call is an error; under `on_miss: llm` the
engine answers it *in character*, consistently with the same world the fixtures
describe. The shape mirrors Google ADK's evaluation framework, whose LLM-driven piece is
the **user simulator** (a described conversation plan plus a model named in config) —
ADK does not inject mock tool responses at all; this design applies the same
plan-plus-model idea to tools. **Skill loaders** (tools of kind `skill_loader` that read
an Agent Skill's SKILL.md, e.g. `load_skill`) are local and deterministic: neither layer
ever wraps them.

## 1. Rule semantics (`mocking.py`)

A tool's rules are an **ordered list**; each rule is `{"matchArgs": {...}, "response": ...}`:

- **First matching rule wins** (`match_rule` / `match_index`).
- `matchArgs` is a **recursive subset match** (`args_subset`): every listed key must be
  present and equal in the call's args; nested dicts are matched key-by-key, so
  pydantic-model arguments match on just the fields the rule cares about.
- `matchArgs: {}` is a **wildcard** — matches any call. By convention it sits last as
  the default fixture (under `on_miss: llm` the pipeline leaves it out on purpose).
- Responses are **deep-copied** per call, so an agent mutating a payload never changes
  the fixture or a previous call's ledger entry.
- If **no rule matches**, the `on_miss` policy decides (`wrap_tool`):

  | policy | behavior |
  |---|---|
  | `real` | call the real tool (the standalone CLI default) |
  | `fallback` | return a canned fallback value |
  | `strict` | raise `MockMissError` (the pipeline default: an unplanned call is an error, never a live call) |
  | `llm` | hand the call to the `LLMMockEngine` (layer 2) |

`wrap_tools` wraps the tools that have rules — and, when an engine is given, every
mockable tool, since the engine can answer for tools with no rules at all. Subagents
exposed as tools (e.g. the travel planner's `flight_agent`) are mocked exactly like
plain tools. Every answered call can be appended to a **ledger** (`layer: rule | llm |
real | fallback | error`, plus the rule index or the engine's validation record).

## 2. The LLM mock engine (`mock_engine.py`)

`LLMMockEngine(model, strategies, tool_specs, strategy="default", on_invalid, max_repairs)`
— one engine per conversation (a case, or a simulation scenario), so its call history
stays within that conversation. `respond(tool_name, args)`:

1. builds one prompt: the shared **world** (entities, ids, invariants every tool draws
   from), the selected **strategy**'s description and its **behaviour** for this tool,
   the strategy's **examples** (args → response), the **tool definition** (description,
   `args_schema`, `output_schema`), the **previous calls** of this conversation (for
   consistency: the same id resolves to the same entity) and the call itself in a
   machine-readable `TOOL CALL:` block;
2. asks for a **structured answer** — the tool's object `output_schema` when it has one,
   else a `{"response_json": "<JSON text>"}` envelope;
3. **validates** the answer with `tool_schemas.validate` against `output_schema`
   (required fields, types, enums, limits — the same validator the mocks stage applies
   to generated fixtures);
4. on problems, re-asks **once** with the problems listed (`max_repairs`);
5. if still invalid: `on_invalid: fallback` returns the strategy's `fallback_response`
   (itself validated when the strategies were generated), `on_invalid: strict` raises
   `MockEngineError` — which the runner records as an **infrastructure** error, never an
   agent error.

Everything is written to `engine.ledger` (`tool, args, strategy, response, valid,
repairs, fallback, seconds, problems/error`) and copied into the wrapper's ledger.
Strategy lookup falls back from the selected strategy to `default` to a generic
behaviour derived from the tool's own description, so an engine is never without a
script. `validate_strategies` checks a hand-written strategies document against the
tools' schemas (the CLI uses it for `mock strategies --set`).

## 3. When mocks are installed (run time)

Three call sites, three decisions:

- **Pipeline `run` stage** (`pipeline/stages.py`): always `mocked=True`,
  `on_miss=cfg.mocking.on_miss` (config default `strict`). Per case,
  `merge_mock_rules(dataset_rules, case_rules)` is computed — **per-case rules replace
  the dataset list for that tool** — and the graph is rebuilt per case with the wrapped
  tools (`runner.py`). Under `llm` the runner builds one engine per case from the
  dataset's `mocks.strategies` and `mocks.llm` (model, `on_invalid`, `max_repairs`),
  selecting `metadata.mocks.strategy` when the case names one; the `CaseRun` keeps the
  ledger in `mock_calls`, and the run artifact sums the layers in `mocking.calls`
  (`rule`, `llm`, `real`, `fallback`, `error`, `invalid`).
- **Simulate stage**: wraps tools with the **dataset-level** rules only (no per-case
  overrides — scenarios are not cases) under the same policy; under `llm` each scenario
  gets its own engine, honouring the scenario's `mock_strategy`; results carry
  `mock_calls` totals.
- **Standalone CLI** (`evalbuilder run DATASET --mock [--on-miss …] [--mock-model SPEC]
  [--strategy ID]`, `evalbuilder simulate … --mock`): mocking is **opt-in** (`--mock`,
  default off); the policy defaults to the dataset's `mocks.on_miss`, else `real` — the
  interactive/skill workflow may touch real tools unless told otherwise; the autonomous
  pipeline never does.

## 4. Where the rules and strategies come from (generation time)

The **`mocks` pipeline stage** asks the generator LLM for fixtures per mockable tool
(`MOCK_SYSTEM` in `pipeline/generator.py`): one realistic **default success response**
plus 1–3 **variants** keyed by specific args (a known order id, a known query),
explicitly told to be *cross-tool consistent* (ids referenced by one tool must exist in
the others) and to contain **no error responses** — errors are per-case injections.
What comes back is defensively validated (`author_mocks`):

- variant `match_args` are type-checked against the tool's `args_schema`
  (subset/partial semantics), responses against its `output_schema`; violators are
  dropped, each with a recorded problem;
- a bad or missing default is replaced by a **schema-conformant sample**
  (`tool_schemas.example`) or a generic `{"ok": true, …}` fixture;
- every tool's list **ends with a wildcard default** — except under `on_miss: llm`,
  where only the keyed variants stay so the engine sees the long tail;
- if `mocking.required` (default true), any tool left without rules (or, under `llm`,
  without a default-strategy behaviour) **fails the stage**.

Then, when `mocking.strategies` is on (the default, and always under `llm`),
`author_strategies` asks the generator for the **strategies document** (`STRATEGIES_SYSTEM`):
the world; a `default` strategy with a behaviour entry, examples and a
`fallback_response` for *every* tool, consistent with the fixtures; and 1–2 alternates
named for a failure mode (`degraded`, `empty`). Validation mirrors the fixtures':
unknown tools and id-less strategies are dropped, invalid fallbacks are replaced by a
schema sample, invalid examples dropped, and a missing default behaviour is
synthesised from the tool's description — all reported as problems. The result is
`dataset.mocks.strategies` (`evalbuilder/mock-strategies/v1`; the mocks stage also leaves a
copy in `work/mock-strategies.json` on its way to the dataset stage).

The **dataset stage** embeds everything into `dataset.json → mocks: {tools, on_miss,
strategy, llm: {model, on_invalid, max_repairs}, strategies}` — the dataset is
self-contained. The case-authoring prompt lists the strategies so the generator can put
a case under an alternate one (`mock_strategy`, validated against the declared ids →
`metadata.mocks.strategy`); the scenario-authoring prompt does the same for simulation
scenarios (`mock_strategy`).

**Per-case rules** are added in two ways during case authoring:

- the generator can emit `mock_overrides` on a case (e.g. `{"error": "timeout"}` for a
  `tool_error_handling` scenario) — parsed into `case.metadata.mocks.tools`;
- **schema-edge `malformed_output` cells never trust the LLM**: code injects the
  corrupted payload itself via `tool_schemas.corrupt(output_schema)` — drop a required
  field or mistype one.

`with_fallback` (`mocking.py`) then appends the dataset-level rules after the case's
rules unless the case already ends with a wildcard — so an error injection for one arg
set keeps the tool answerable for every other call in that same case.

Interactively, the `agent-eval-mock` skill authors the same data by hand:
`evalbuilder mock set` (rules), `evalbuilder mock strategies --set @strategies.json
--model SPEC --on-miss llm` (strategies, model, policy; `--case ID --strategy X` pins a
case), `evalbuilder mock validate --tool T --response @r.json` (the engine's validator on
demand) and `evalbuilder mock try --tool T --args '{…}' [--strategy ID]` (one call
through both layers, showing which answered and whether it conforms).

## 5. Verification gates before anything runs

`verify_dataset` (`mocking.py`) simulates matching: every `expected_tools[i].args` of
every mocked case must be answered by some rule. `verify_summary` reads that through the
policy: under `strict` (and `fallback`/`real`) the **review stage** auto-rejects cases
whose expected calls have no rule and the **verify stage** hard-fails on remaining
misses and on unmocked tools when `mocking.required`; under `llm` the misses are
counted as `llm_answered_calls` (informational — the engine will answer them) and a
tool counts as mocked when a rule *or* the default strategy covers it; the verify stage
also refuses an `llm` dataset that names no mock model. The same check is exposed as
`evalbuilder mock verify`.

## 6. The "when and how to mock" decision, summarized

| Layer | Decision | Who makes it |
|---|---|---|
| Pipeline config | `mocking.required: true` + `on_miss: strict` → **every tool gets a fixture; no real call ever happens** during eval runs | default; you opt *out* per config |
| Pipeline config | `on_miss: llm` + `models.mock` → fixtures for the known entities, the engine for the long tail; `on_invalid` decides what an unfixable answer becomes | opt-in per config |
| Standalone CLI | `--mock` off by default; misses run the real tool unless the dataset says otherwise | the human running it |
| Interactive skill (`agent-eval-mock`) | mock tools that are *nondeterministic, side-effecting (`book_*`, `send_*`, payments), rate-limited, or offline-unavailable*; never leave side-effecting tools unmocked in a repeated-run dataset; never mock skill loaders; `strict` for CI, `llm` for exploratory / long-tail runs | Claude following the skill |
| Per case | error injections for `tool_error_handling`; code-injected `corrupt()` for `malformed_output` edges; an alternate strategy only when the failure mode needs it; never mock away the behavior a case tests, *except* when the mock itself is the injected fault | generator + review gate |

The philosophy in one line: mocks make repeats meaningful — stability tracking compares
trajectories across repeats, which only works when tool answers are deterministic — so
`strict` is the safe default for an autonomous pipeline, where a call nobody planned
for is a finding, not a live side effect. The LLM layer trades some of that determinism
for coverage of the long tail, and pays for it with a ledger: the report's `mocking`
section and `stability.llm_mocked_unstable` say which unstable cases had LLM-mocked
tool answers, so the instability is attributed to the mock layer before the agent.
Mock responses are *fixtures, not goldens*: `reference_outputs` still defines what the
agent should do with them.
