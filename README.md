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

The fifth skill, `agent-eval-pipeline`, runs all of that **unattended** from one config
file (see below).

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
| `agent-eval-pipeline` | One config → full autonomous evaluation → `report.json` |

## Autonomous pipeline

```bash
uv run evalbuilder pipeline init eval/pipeline.yaml --name support-bot \
    --source examples/support_bot/agent.py --module examples.support_bot.agent
# edit constraints / coverage / evaluators / thresholds; set review.auto_approve + approved_by
uv run evalbuilder pipeline run eval/pipeline.yaml          # exit 1 unless verdict == pass
uv run evalbuilder pipeline report eval/pipeline/support-bot
```

Stages: `preflight → discover → map → mocks → dataset → review → verify → run×N →
score×N → aggregate → simulate → publish → analyze → report`. An LLM **generator**
authors intents/scenarios (evidence-cited, taxonomy-gated), tool fixtures (every
introspected tool is mocked; unmatched calls are errors), coverage-planned cases,
a self-review, multi-turn scenarios and the final analysis — and every artifact
still passes the same CLI validation. Failures never raise: a failed stage marks its
dependents `skipped`, retries once with a recovery note, and `report.json` always
lands with `verdict ∈ pass|fail|incomplete`, per-stage status, metrics vs thresholds,
slices, coverage achieved vs planned, stability across repeats (unstable *cases* vs
unstable *evaluators*), and a `problems[]` list. `--resume` reuses completed stages; `--resume --from STAGE` regenerates from a stage on.

Model specs: `claude-cli:sonnet` uses the Claude Code CLI with your subscription (the
`langchain-claude-code-cli` package's `ChatClaudeCode` parameter surface, with a
subprocess transport and `--json-schema` structured tool calling in
`src/evalbuilder/claude_cli.py`); `anthropic:…`/`openai:…` use API keys;
`scripted:module:factory` keeps tests offline. Config reference:
`skills/agent-eval-pipeline/references/config-reference.md`; a ready-to-run example:
`examples/support_bot/pipeline.yaml`, and the report it produced with `claude-cli:sonnet`
(16 cases × 2 repeats, ~45 min): `docs/examples/support-bot-report.json`.

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
rules, and injects `model=` when a run specifies one. Three example targets ship in
`examples/` (`weather_bot`, `travel_planner`, and `support_bot` — a router graph over
two ReAct specialists with external HTTP tools) with scripted chat models, so the
whole test-suite runs offline.

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
