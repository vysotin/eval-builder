# How tool mocking works

A walkthrough of the current implementation, layer by layer — rule semantics, when
mocks are installed, where the rules come from, the verification gates, and the
decision of *when to mock*. Code references point at the modules that own each part.

## The design stance

Mock rules are **data that live in the dataset** (`dataset.json → mocks.tools`, plus
per-case overrides in `case.metadata.mocks.tools`), not code. At execution time the
runner **wraps the agent's tools** so a matching rule answers instead of the real
function — the wrapper keeps the original name, description and `args_schema`, so the
model's tool-*selection* behavior is untouched; only the tool's *answer* is substituted
(`src/evalbuilder/mocking.py`, `wrap_tool`).

## 1. Rule semantics (`mocking.py`)

A tool's rules are an **ordered list**; each rule is `{"matchArgs": {...}, "response": ...}`:

- **First matching rule wins** (`match_rule`).
- `matchArgs` is a **recursive subset match** (`args_subset`): every listed key must be
  present and equal in the call's args; nested dicts are matched key-by-key, so
  pydantic-model arguments match on just the fields the rule cares about.
- `matchArgs: {}` is a **wildcard** — matches any call. By convention it sits last as
  the default fixture.
- If **no rule matches**, the `on_miss` policy decides (`wrap_tool`):

  | policy | behavior |
  |---|---|
  | `real` | call the real tool (the standalone CLI default) |
  | `fallback` | return a canned fallback value |
  | `strict` | raise `MockMissError` (the pipeline default: an unplanned call is an error, never a live call) |

`wrap_tools` wraps **only tools that have rules**; others pass through. Subagents
exposed as tools (e.g. the travel planner's `flight_agent`) are mocked exactly like
plain tools.

## 2. When mocks are installed (run time)

Three call sites, three decisions:

- **Pipeline `run` stage** (`pipeline/stages.py`): always `mocked=True`,
  `on_miss=cfg.mocking.on_miss` (config default `strict`). Per case,
  `merge_mock_rules(dataset_rules, case_rules)` is computed — **per-case rules replace
  the dataset list for that tool** — and the graph is rebuilt per case with the wrapped
  tools (`runner.py`). A case with an injected error fixture gets *its* world; the next
  case gets the clean one.
- **Simulate stage**: wraps tools with the **dataset-level** rules only (no per-case
  overrides — scenarios are not cases) under the same `on_miss`.
- **Standalone CLI** (`evalbuilder run DATASET --mock --on-miss …`): mocking is
  **opt-in** (`--mock`, default off) and `on_miss` defaults to `real` — the
  interactive/skill workflow may touch real tools unless told otherwise; the autonomous
  pipeline never does.

## 3. Where the rules come from (generation time)

The **`mocks` pipeline stage** asks the generator LLM for fixtures per tool
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
- every tool's list **ends with a wildcard default**, so every call is answerable;
- if `mocking.required` (default true), any tool left without rules **fails the stage**.

The result is saved as `mock-rules.json` and, in the **dataset stage**, embedded into
the dataset as `mocks: {tools, on_miss}` — the dataset is self-contained.

**Per-case mocks** are added in two ways during case authoring:

- the generator can emit `mock_overrides` on a case (e.g. `{"error": "timeout"}` for a
  `tool_error_handling` scenario) — parsed into `case.metadata.mocks.tools`;
- **schema-edge `malformed_output` cells never trust the LLM**: code injects the
  corrupted payload itself via `tool_schemas.corrupt(output_schema)` — drop a required
  field or mistype one.

`with_fallback` (`mocking.py`) then appends the dataset-level rules after the case's
rules unless the case already ends with a wildcard — so an error injection for one arg
set keeps the tool answerable for every other call in that same case.

## 4. Verification gates before anything runs

`verify_dataset` (`mocking.py`) simulates matching: every `expected_tools[i].args` of
every mocked case must be answered by some rule. The **review stage** auto-rejects
cases whose expected calls have no rule; the **verify stage** hard-fails on remaining
misses and on unmocked tools when `mocking.required`. The same check is exposed as
`evalbuilder mock verify`.

## 5. The "when to mock" decision, summarized

| Layer | Decision | Who makes it |
|---|---|---|
| Pipeline config | `mocking.required: true` + `on_miss: strict` → **every tool gets a fixture; no real call ever happens** during eval runs | default; you opt *out* per config |
| Standalone CLI | `--mock` off by default; misses run the real tool | the human running it |
| Interactive skill (`agent-eval-mock`) | mock tools that are *nondeterministic, side-effecting (`book_*`, `send_*`, payments), rate-limited, or offline-unavailable*; never leave side-effecting tools unmocked in a repeated-run dataset | Claude following the skill |
| Per case | error injections for `tool_error_handling`; code-injected `corrupt()` for `malformed_output` edges; never mock away the behavior a case tests, *except* when the mock itself is the injected fault | generator + review gate |

The philosophy in one line: mocks make repeats meaningful — stability tracking compares
trajectories across repeats, which only works when tool answers are deterministic — and
`strict` is the safe default for an autonomous pipeline, where a call nobody planned
for is a finding, not a live side effect. Mock responses are *fixtures, not goldens*:
`reference_outputs` still defines what the agent should do with them.
