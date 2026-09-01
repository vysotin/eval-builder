---
name: agent-eval-mock
description: Use when eval cases depend on nondeterministic, side-effecting, rate-limited, or unavailable tools in a LangGraph agent — declares two-layer tool mocks in the dataset (deterministic rules first, an LLM mock engine driven by pre-generated strategies for the long tail) and verifies runs are answerable and repeatable.
---

# Make scenarios deterministic (or at least answerable) with tool mocks

Mock data lives in the dataset, next to the cases it serves, and is installed at
inference time by wrapping the target's tools (name, description and args schema are
preserved, so the model's tool selection is unchanged). Subagents exposed as tools are
mocked like plain tools. **Skill loaders** (`load_skill`-style tools of kind
`skill_loader` in the agent map) are local, deterministic reads of SKILL.md — they are
never mocked and need no rules.

## Two layers

| layer | what answers | when |
|---|---|---|
| 1 — rules | ordered per-tool `{matchArgs, response}` rules; first match wins; `matchArgs` is a recursive subset match; `{}` is a wildcard | always first |
| 2 — LLM mock engine | a model playing the backend from a **strategy** (shared *world* + per-tool *behaviour*, examples, `fallback_response`), validated against the tool's `output_schema`, one repair round | only under `on_miss: llm`, when no rule matched |

Miss policies (`mocks.on_miss`): `strict` (unmatched call = error, never a real call —
the CI default), `llm` (the engine answers), `fallback` (a canned value), `real` (call
the real tool; the standalone CLI default). Under `llm`, `mocks.llm` names the model
(`provider:model`), `on_invalid` (`fallback` → the strategy's `fallback_response` after
a failed repair, `strict` → an infrastructure error) and `max_repairs`.

## Workflow

1. Read the dataset and the agent map. List tools that are nondeterministic (search,
   live data), side-effecting (`book_*`, `send_*`, writes), rate-limited, or
   unavailable offline. Decide the policy: `strict` when every call a case can make is
   known (repeatable, CI); `llm` when cases legitimately reach a long tail of inputs
   the fixtures cannot enumerate (exploratory runs, simulations). Say which and why.

2. Layer 1 — author ordered rules, most-specific `matchArgs` first, a `{}` wildcard
   last only when the case needs a default fixture (under `llm` leave the wildcard
   out, so the engine sees the long tail):

   ```bash
   evalbuilder mock set eval/datasets/NAME.json --case CASE_ID --tool search_flights --rules @rules.json
   evalbuilder mock set eval/datasets/NAME.json --tool get_weather --rules @default-weather.json   # dataset-level
   ```

   Per-case rules override the dataset-level list for that tool.

3. Layer 2 — write the strategies (`{world, strategies: {default: {description, tools:
   {name: {behavior, examples, fallback_response}}}, degraded: …}}`): a `default`
   strategy covering every mockable tool with concrete rules an LLM can follow (which
   ids resolve to what, ranges, how unknown ids answer), consistent with the layer-1
   fixtures, plus 1–2 alternates named for a failure mode. Then install them with the
   model and policy:

   ```bash
   evalbuilder mock strategies eval/datasets/NAME.json --set @strategies.json --model claude-cli:claude-sonnet-5 --on-miss llm
   evalbuilder mock strategies eval/datasets/NAME.json --case CASE_ID --strategy degraded   # one case under an alternate strategy
   ```

   `--set` validates fallbacks and examples against the tools' schemas and refuses
   unknown tools; `--on-invalid strict|fallback` and `--max-repairs N` tune the engine.

4. Validate a fixture or a hand-written response against the tool's declared output
   schema (the same check the engine applies), and try a call through both layers to
   see which one answers and whether the answer conforms:

   ```bash
   evalbuilder mock validate eval/datasets/NAME.json --tool create_ticket --response @receipt.json
   evalbuilder mock try eval/datasets/NAME.json --tool lookup_order --args '{"order_id": "Z9"}' --strategy degraded
   ```

5. Verify before anyone runs. Under `strict` every expected call must match a rule;
   under `llm` the unmatched expected calls are listed as `llm_answered` (informational):

   ```bash
   evalbuilder mock verify eval/datasets/NAME.json
   ```

6. State the policy decision for this dataset explicitly in your summary, and under
   `llm` remind the user that engine answers vary between repeats: the run artifact's
   per-case `mock_calls` ledger and the report's `stability.llm_mocked_unstable` tell
   the mock layer's instability apart from the agent's.

## Rules

- Never mock the tool a case is specifically testing the error handling of — unless
  the mock itself injects the error (`{"matchArgs": {}, "response": {"error": "timeout"}}`
  is the right way to test `tool_error_handling`; a `degraded` strategy is the
  long-tail equivalent).
- Never leave side-effecting tools (`book_*`, `send_*`, payments) unmocked in a
  dataset meant for CI or repeated runs; under `llm` they are answered by the engine —
  make their `default` behaviour explicit (what a successful write returns).
- Never mock skill loaders; never let the engine answer for them.
- Mock responses are fixtures, not goldens: `reference_outputs` still defines what the
  agent should do with them. Only use `contains` literals guaranteed by a rule, a
  strategy example, or an explicit behaviour rule.
- An engine answer that fails validation after the repair round is a mock problem, not
  an agent defect: report `invalid` / `fallback` / `error` counts separately.
