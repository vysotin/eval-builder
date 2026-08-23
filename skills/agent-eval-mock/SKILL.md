---
name: agent-eval-mock
description: Use when eval cases depend on nondeterministic, side-effecting, rate-limited, or unavailable tools in a LangGraph agent — declares ADK-style tool-mock rules in the dataset and verifies runs become deterministic.
---

# Make scenarios deterministic with tool mocks

Mock rules live in the dataset, next to the cases they serve, and are installed at
inference time by wrapping the target's tools (name, description, and args schema are
preserved, so the model's tool selection is unchanged). Subagents exposed as tools
are mocked exactly like plain tools.

## Rule semantics

| Property | Behavior |
|---|---|
| Rule list | Ordered per tool; **first matching rule wins** |
| `matchArgs` | Subset equality: every listed key must equal the call's arg |
| `{}` (empty matchArgs) | Wildcard — matches any call |
| No rule matches | Miss policy: run the **real tool** (default), return a fallback, or strict error |

## Workflow

1. Read the dataset and the agent map. List tools that are nondeterministic
   (search, live data), side-effecting (`book_*`, `send_*`, writes), rate-limited,
   or unavailable offline.

2. For each affected case, author ordered rules — most-specific `matchArgs` first,
   a `{}` wildcard last only when the case needs a default fixture:

   ```bash
   evalbuilder mock set eval/datasets/NAME.json --case CASE_ID --tool search_flights --rules @rules.json
   ```

3. Shared fixtures used by many cases go dataset-level (omit `--case`); per-case
   rules override the dataset-level list for that tool:

   ```bash
   evalbuilder mock set eval/datasets/NAME.json --tool get_weather --rules @default-weather.json
   ```

4. Verify every expected tool call is answerable by a rule before anyone runs:

   ```bash
   evalbuilder mock verify eval/datasets/NAME.json
   ```

5. State the miss-policy decision for this dataset explicitly in your summary
   (default: unmatched calls run the real tool).

## Rules

- Never mock the tool a case is specifically testing the error handling of — unless
  the mock itself injects the error (`{"matchArgs": {}, "response": {"error": "timeout"}}`
  is the right way to test `tool_error_handling`).
- Never leave side-effecting tools (`book_*`, `send_*`, payments) unmocked in a
  dataset meant for CI or repeated runs.
- Mock responses are fixtures, not goldens: `reference_outputs` still defines what
  the agent should do with them.
