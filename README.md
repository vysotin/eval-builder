# eval-builder

Agentic skills + a deterministic CLI for building and running evals for **LangGraph
agents**: parse the agent's source, derive intents/scenarios/failure modes with cited
evidence, generate a coverage-driven golden dataset in OpenEvals/LangSmith-compatible
JSON, mock tools ADK-style for deterministic runs, and score experiments with
OpenEvals evaluators — local filesystem first, LangSmith optional.

```
user's LangGraph agent repo
        │
        ▼
[skill] agent-eval-discover ──► eval/agent-map.json         (hypotheses + evidence)
        ▼
[skill] agent-eval-dataset ──► eval/datasets/<name>.json    (pending cases, coverage grid)
        │        human approval gate (explicit, per-case/batch)
        ▼
[skill] agent-eval-mock ─────► ADK-style mock rules in the dataset
        ▼
[skill] agent-eval-run ──────► validate → run (mocks installed) → score with
                                OpenEvals → eval/results/*  (+ optional LangSmith)
                                simulate: multi-turn scenarios; failures mined back
```

Skills are policy prose; **all determinism lives in the `evalbuilder` CLI** — schema
validation, content-hash case IDs, coverage math, the review state machine, mock
installation, trajectory capture, scoring, and idempotent LangSmith publication.

## Quickstart

```bash
uv venv --python 3.12 && uv sync
uv run evalbuilder check                      # capability matrix
uv run evalbuilder discover examples.weather_bot.agent --source examples/weather_bot/agent.py
uv run pytest                                 # full offline test suite
```

The four skills are exposed to Claude Code via `.claude/skills/agent-eval-*`
(symlinks into `skills/`).

| Skill | Use when |
|---|---|
| `agent-eval-discover` | Starting eval work / agent code changed — map the test surface |
| `agent-eval-dataset` | Generating or extending the golden dataset (pending → human review) |
| `agent-eval-mock` | Cases depend on nondeterministic or side-effecting tools |
| `agent-eval-run` | Running/scoring experiments, publishing to LangSmith, simulating |

## Dataset format

`evalbuilder/dataset/v1` — each case maps 1:1 to a LangSmith example and to OpenEvals
evaluator kwargs; everything harness-specific lives under `metadata`:

```json
{
  "id": "case-3fa1b2c4d5",
  "inputs": {"messages": [{"role": "user", "content": "Find me a flight from SFO to JFK on 2026-09-01"}]},
  "reference_outputs": {
    "contains": "AA100",
    "expected_tools": [{"name": "flight_agent", "args": {}}],
    "contract": "Must not confirm a booking without an explicit user yes."
  },
  "metadata": {
    "intent": "intent.book-flight", "topic": "unspecified",
    "scenario": "scenario.book-flight.happy", "failure_mode": "none",
    "variant": "happy", "source": "synthetic",
    "mocks": {"tools": {"flight_agent": [
      {"matchArgs": {}, "response": "Found AA100 at $350 (mocked fixture)."}]}}
  },
  "review": {"status": "pending"},
  "publication": {"langsmith_example_id": null}
}
```

Mock rules use ADK-eval semantics: ordered per-tool rule lists, first match wins,
`matchArgs` subset equality, `{}` wildcard, unmatched calls run the real tool (or
fallback/strict).

## Target contract

The dataset's `target` names a module exposing
`build_agent(model=None, tools=None) -> CompiledStateGraph` and a `TOOLS` list. The
runner rebuilds the graph per case, wrapping `TOOLS` with the case's merged mock
rules. Two example targets ship in `examples/` (`weather_bot`, `travel_planner`) with
scripted chat models, so the whole pipeline runs offline.

## LangSmith (optional)

Set `LANGSMITH_API_KEY` (see `.env.example`), then
`evalbuilder publish eval/datasets/<name>.json` uploads approved cases (idempotent,
read-back verified) and records example IDs in the dataset file.

## Extending to other frameworks

The dataset schema is framework-neutral; LangGraph specifics live in
`src/evalbuilder/{discover,target,mocking}.py`. A Google ADK adapter maps 1:1
(AgentMetadata → agent-map, `before_tool_callback` mocking, `.evalset.json` import);
Dify follows via DSL rewrite. See the design spec:
`docs/superpowers/specs/2026-08-23-eval-builder-skills-design.md` and plan:
`docs/superpowers/plans/2026-08-23-eval-builder-mvp.md`.
