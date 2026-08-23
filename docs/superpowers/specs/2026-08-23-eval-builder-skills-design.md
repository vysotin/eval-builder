# eval-builder — Agentic skills for building and running evals for LangGraph agents

**Date:** 2026-08-23
**Status:** Draft for review (produced in an autonomous session; see "Assumptions & open questions")
**MVP target:** LangGraph agents. Designed for extension to Google ADK, then Dify.

## 1. Goal

A set of Claude Code agentic skills, backed by a deterministic Python CLI, that help a user:

1. **Discover** — parse a LangGraph agent's source code (graph structure, prompts, tools,
   subagents), identify data domains (from sample documents), and derive intents and
   scenarios (happy paths and structurally justified failure modes).
2. **Generate** — build a synthetic golden dataset from the agent metadata, identified
   intents/scenarios, data domains, and simulated inputs, with coverage accounting and a
   human review gate.
3. **Mock** — declare tool-mocking rules (ADK-eval style), store them in the dataset, and
   install them into the agent at inference time so complex scenarios run deterministically.
4. **Run** — validate the dataset, execute the agent over approved cases, score with
   OpenEvals evaluators (deterministic first, LLM-as-judge second), and report per-metric
   stats with coverage slices. Simulate multi-turn conversations for complex scenarios.

Storage is **local filesystem first**; a **LangSmith server is an optional, configurable
backend** for dataset publication and experiment tracking. The dataset format is
**OpenEvals/LangSmith-compatible JSON** (`inputs` / `reference_outputs` / `metadata` per
case) with additional metadata fields.

## 2. Idea sources

Synthesized from three sibling repos (explored 2026-08-23):

- `../adk-eval-ext` — agent parsing (AST + live introspection → `AgentMetadata` tree),
  task/scenario generation with distribution knobs, ADK `.evalset.json` generation,
  inference/scoring separation, and — most importantly — the **`mockResponses` schema and
  a working tool/sub-agent mocking implementation** (`poc_split_eval/mocking.py`, git HEAD):
  per-tool ordered rules, `matchArgs` subset matching, `{}` wildcard, fallback-to-real-tool,
  strict mode, reversible install/uninstall. Its
  `docs/03-cross-framework-langchain-langgraph-eval.md` prototypes LangGraph mocking
  (tool wrapping preserving `args_schema`; mock tool node).
- `../dify-agent-test-skills` — the six-skill pipeline shape (setup → discover → dataset →
  simulate → approve → evaluate), skills as short policy prose delegating all determinism
  to a `difytest` CLI, coverage grid `intent × topic × scenario × failure_mode`,
  content-hash case IDs, review state machine the model cannot bypass, idempotent
  LangSmith publish with read-back verification, OpenEvals judge wiring with
  prompt override.
- `../dify-eval` — evidence-cited analysis artifact ("hypothesis, never fact"), gated
  failure taxonomy (a failure type is only offered when structure justifies it), document →
  data-domain pipeline (chunk → context quality filter → generate → evolve), simulation
  scenarios as goal + stop conditions (not scripts), metric plans, executable skill
  contract tests.

## 3. Approaches considered

- **A. Skills-only (no CLI).** Claude does everything ad hoc, writes JSON by hand.
  Fast to build; rejected — both reference repos independently converged on "skills =
  policy prose, determinism in a CLI" because hand-written artifacts drift, validation is
  skipped, and approval gates are bypassable.
- **B. Skills + deterministic CLI/library (chosen).** A small Python package
  (`evalbuilder`) owns schemas, validation, coverage math, mocking, execution, and
  LangSmith I/O. Skills guide the model through analysis/generation policy and invoke the
  CLI for every artifact mutation. This is the shape proven by `dify-agent-test-skills`
  and matches the goal's "skills + test scripts" framing.
- **C. Full framework port of `adk_eval_tool` (library + UI).** Too big for an MVP and
  duplicates the LLM-generation logic that Claude itself performs better inside a skill.

## 4. Architecture overview

```
user's LangGraph agent repo (or examples/ here)
        │
        ▼
[skill] agent-eval-discover ──► eval/agent-map.json         (hypotheses + evidence)
        │   + sample docs ────► data domains/topics
        ▼
[skill] agent-eval-dataset ──► eval/datasets/<name>.json    (pending cases, coverage grid)
        │                              │
        │   human approval gate        ▼
        │   (explicit, per-case/batch) review.status=approved
        ▼
[skill] agent-eval-mock ─────► mock rules in dataset (dataset-level + per-case)
        ▼                       + target factory contract in agent code
[skill] agent-eval-run ──────► validate → execute (mocks installed) → score with
        │                       OpenEvals → eval/results/<run>.json (+ optional
        │                       LangSmith dataset + experiment)
        └── simulate mode ───► multi-turn simulated-user runs for complex scenarios;
                                failures mined back into the dataset as new cases
```

**Division of labor.** Claude (inside a skill) does: reading code and documents, deriving
intents/scenarios/failure hypotheses with evidence, writing case content, choosing
evaluators, judging simulation transcripts. The `evalbuilder` CLI does: schema
validation, case normalization and IDs, coverage/gap math, review-state enforcement,
mock installation, graph execution, trajectory capture, scoring, stats/slices, LangSmith
publish/read-back. **Skills never write artifact JSON directly** — every mutation goes
through a CLI command that validates the whole artifact.

## 5. Repository layout

```
eval-builder/
├── skills/                          # source of truth for the four skills
│   ├── agent-eval-discover/SKILL.md   (+ references/)
│   ├── agent-eval-dataset/SKILL.md    (+ references/)
│   ├── agent-eval-mock/SKILL.md       (+ references/)
│   └── agent-eval-run/SKILL.md        (+ references/)
├── .claude/skills/agent-eval-*  →  symlinks to ../../skills/agent-eval-*
├── src/evalbuilder/                 # the deterministic core
│   ├── schemas.py        # pydantic models for all artifacts
│   ├── artifacts.py      # load/save/validate/normalize, case IDs, import
│   ├── coverage.py       # required cells, gaps
│   ├── discover.py       # LangGraph source AST parse + live graph introspection
│   ├── target.py         # target loading contract, trajectory capture
│   ├── mocking.py        # MockRule matching, tool wrapping, install/uninstall
│   ├── runner.py         # local execution + scoring loop
│   ├── evaluators.py     # deterministic evaluators + OpenEvals judge wiring
│   ├── simulate.py       # multi-turn simulation (openevals simulated user)
│   ├── langsmith_io.py   # optional publish/experiment with read-back verification
│   ├── config.py         # .env settings + capability matrix
│   └── cli.py            # typer CLI, entry point `evalbuilder`
├── examples/
│   ├── weather_bot/         # single ReAct agent, 2 tools
│   └── travel_planner/      # supervisor + flight/hotel subagents, 4 tools
├── tests/                   # pytest; offline (scripted fake model), no network
├── docs/superpowers/specs/  # this document
└── pyproject.toml           # uv-managed, Python 3.12+
```

## 6. Artifacts and schemas

All artifacts are JSON with a `schema` version string, written only by the CLI
(`indent=2`, `ensure_ascii=False`). Local FS is the source of truth.

### 6.1 Agent map — `evalbuilder/agent-map/v1` → `eval/agent-map.json`

```json
{
  "schema": "evalbuilder/agent-map/v1",
  "framework": "langgraph",
  "source_sha256": "…",
  "app": {"name": "travel_planner", "module": "examples.travel_planner.agent"},
  "graph": {
    "nodes": [{"id": "supervisor", "kind": "llm|tool|router|subgraph", "prompt": "…",
                "tools": ["search_flights"], "evidence": ["source:agent.py:42"]}],
    "edges": [["supervisor", "flight_agent"]],
    "conditional_edges": [{"source": "supervisor", "targets": ["flight_agent", "hotel_agent", "END"]}]
  },
  "tools": [{"name": "search_flights", "description": "…", "args_schema": {…},
              "used_by": ["flight_agent"]}],
  "constraints": ["never confirm a booking without an explicit user yes"],
  "data_domains": {"topics": [], "sources": []},
  "intents":  [{"id": "intent.book-flight", "name": "…", "status": "hypothesis",
                 "evidence": ["prompt:supervisor", "tool:search_flights"]}],
  "scenarios": [{"id": "scenario.book-flight.happy", "intent": "intent.book-flight",
                  "description": "…", "status": "hypothesis", "evidence": ["…"]}],
  "failure_scenarios": [{"failure_type": "tool_error_handling", "rationale": "…",
                          "evidence": ["tool:search_flights"]}],
  "decisions_needed": ["Confirm inferred intents", "Provide sample documents or topics"]
}
```

Rules carried over from the reference repos: everything derived is a **hypothesis with
cited evidence** (`source:<file>:<line>`, `prompt:<node>`, `tool:<name>`,
`edge:<a>-><b>`); topics are **never invented** — they come from user-provided documents
or the user's description; `source_sha256` detects staleness; `constraints` are authored
by the user and merged forward.

**Failure taxonomy (structurally gated).** A failure type may only be proposed when its
precondition holds in the graph: `input_validation` and `provider_error` (always),
`branch_misrouting` (conditional edges exist), `tool_misuse` / `tool_error_handling`
(tools exist), `retrieval_grounding` (retriever/RAG node exists), `state_loss`
(multi-turn/checkpointer), `output_contract_violation` (structured output declared),
`constraint_violation` (authored constraints exist), `out_of_scope` (always),
`prompt_injection` (external content reaches the prompt).

### 6.2 Dataset — `evalbuilder/dataset/v1` → `eval/datasets/<name>.json`

OpenEvals/LangSmith-compatible per case, with metadata extensions:

```json
{
  "schema": "evalbuilder/dataset/v1",
  "name": "travel-planner-golden-v1",
  "dataset_type": "final_response | trajectory",
  "target": {"framework": "langgraph", "module": "examples.travel_planner.agent",
              "factory": "build_agent"},
  "mocks": { "tools": { … dataset-level default rules … } },
  "cases": [
    {
      "id": "case-3fa1b2c4d5",
      "inputs": {"messages": [{"role": "user", "content": "Book a flight SFO→JFK on 2026-09-01"}]},
      "reference_outputs": {
        "response": "Presents flight options incl. AA100 at $350 and asks which to book.",
        "trajectory": [ … optional OpenAI-style reference messages with tool_calls … ],
        "expected_tools": [{"name": "search_flights",
                             "args": {"origin": "SFO", "destination": "JFK"}}],
        "contract": "Must not confirm a booking without an explicit user yes."
      },
      "metadata": {
        "intent": "intent.book-flight", "topic": "unspecified",
        "scenario": "scenario.book-flight.happy", "failure_mode": "none",
        "variant": "happy | boundary | adversarial | linguistic | multi-turn",
        "source": "synthetic | import | simulation",
        "evidence": ["scenario.book-flight.happy"],
        "mocks": {"tools": {"search_flights": [
          {"matchArgs": {"origin": "SFO"}, "response": {"flights": [{"id": "AA100", "price": 350}]}},
          {"matchArgs": {}, "response": {"flights": [], "error": "no flights found"}}
        ]}}
      },
      "review": {"status": "pending | approved | rejected", "note": ""},
      "publication": {"langsmith_example_id": null}
    }
  ],
  "langsmith": {"dataset_id": null, "dataset_name": null}
}
```

Design rules:

- `inputs` / `reference_outputs` / `metadata` map 1:1 to a LangSmith example and to
  OpenEvals evaluator kwargs (`inputs=`, `outputs=`, `reference_outputs=`). Everything
  framework- or harness-specific lives under `metadata` (the additive-key pattern from
  `adk-eval-ext`'s Dify docs: the file stays a valid generic dataset with zero edits).
- **Case IDs are content hashes** over `(inputs, intent, topic, scenario, failure_mode)`,
  so the same input under a different coverage cell is a different case.
- **Review state machine**: `normalize_case` force-resets new/imported content to
  `pending`. `publish` and `run` refuse pending or unpublished-but-required cases.
  Approval only via an explicit `evalbuilder review` command the skill may run **only
  after the human explicitly approves** (never inferred from silence).
- **Coverage grid** `intent × topic × scenario × failure_mode`, `gaps` computed against
  the reviewed agent map with a `target_per_cell`.
- Import supports foreign shapes (`examples`/`goldens`/`cases` keys, LangSmith exports)
  with field mapping, always landing as `review.status=pending`, `metadata.source=import`.

### 6.3 Mock rules (ADK-style)

Semantics copied from `adk-eval-ext` (`docs/01-subagent-mocking-in-eval-datasets.md` +
`poc_split_eval/mocking.py`):

- Per-tool **ordered rule list**; first match wins.
- `matchArgs` is a **subset-equality** match over the tool's call args; `{}` is a
  wildcard.
- No matching rule → configurable: run the **real tool** (default), return a `fallback`,
  or **strict** error.
- Rules merge: per-case `metadata.mocks` override dataset-level `mocks` per tool name.
- Subagent mocking (MVP, LangGraph): subagents invoked as tools (handoff/task tools) are
  mocked with the same tool-rule mechanism. Mocking arbitrary graph nodes is out of scope
  for the MVP (documented limitation; ADK adapter will add stub agents later).

### 6.4 Results — `evalbuilder/results/v1` → `eval/results/run-<id>.json`

Per-case rows (`case_id`, outputs envelope, per-metric scores with comments, error class)
plus aggregate `metrics: {key: {n, avg, min, max, errors}}` and **slices** by
`intent`, `failure_mode`, `variant`. Execution evidence and scores are kept in one file
but produced in two phases (run → score) so re-scoring with different evaluators does not
re-run the agent. Failure classes are separated (agent failure vs infrastructure vs
evaluator error) — an evaluator problem is never reported as the agent misbehaving.

### 6.5 Evaluator config — `eval/evaluators.yaml`

```yaml
evaluators:
  - type: expected_tools          # deterministic: expected_tools ⊆ actual calls, args subset
  - type: trajectory_match        # agentevals/openevals: strict|unordered|subset|superset
    match_mode: unordered
  - type: json_valid
  - type: contract                # LLM judge over reference_outputs.contract
    model: "anthropic:claude-sonnet-5"
  - type: correctness             # OpenEvals CORRECTNESS_PROMPT unless prompt: overrides
    model: "anthropic:claude-sonnet-5"
```

One metric per evaluator; deterministic before judge; judge model/prompt overridable.

### 6.6 CLI surface (typer, entry point `evalbuilder`)

```
evalbuilder check                                  # capability matrix (json)
evalbuilder discover MODULE [--source F] [--eval-dir D]   # → agent-map.json skeleton
evalbuilder agent-map set-intents|set-scenarios|set-failures|set-topics …  # validated updates from skill-authored JSON
evalbuilder dataset init PATH --name N --type T --target MODULE[:FACTORY]
evalbuilder dataset add PATH --case JSON           # normalize + hash ID + pending
evalbuilder dataset import PATH --from FILE        # foreign formats → pending
evalbuilder dataset validate PATH
evalbuilder dataset gaps PATH --agent-map M [--target-per-cell 1]
evalbuilder review PATH --approve|--reject IDS [--note …]   # only after explicit human decision
evalbuilder mock set PATH [--case ID] --tool NAME --rules JSON
evalbuilder mock verify PATH                       # every mocked case matches ≥1 rule (dry)
evalbuilder run PATH [--mock/--no-mock] [--ids …] [--out D]      # execute → run artifact
evalbuilder score RUN --evaluators eval/evaluators.yaml          # score + slices, no re-run
evalbuilder simulate PATH --scenarios F [--max-turns N]          # failures → pending cases
evalbuilder publish PATH [--dataset-name N]        # LangSmith, approved-only, read-back verified
```

## 7. The four skills

Each skill = short imperative prose with explicit prohibitions + CLI invocations +
`references/` for depth (progressive disclosure; SKILL.md well under 500 lines). Authored
following `superpowers:writing-skills`.

1. **agent-eval-discover** — "Map a LangGraph agent's test surface." Runs
   `evalbuilder check` (capability matrix: target importable, judge key, LangSmith creds —
   report, don't narrate a healthy setup), then `evalbuilder discover` for the structural
   skeleton, then reads source/prompts to draft intents/scenarios/failures **with cited
   evidence**; ingests sample documents into data-domain topics (never invents topics);
   records `decisions_needed` and asks the user to confirm the map before generation.
   References: failure-taxonomy.md, evidence rules.
2. **agent-eval-dataset** — "Generate a coverage-driven golden dataset." Requires a
   reviewed agent map; asks for existing goldens and imports them first; allocates
   coverage across the grid + variant taxonomy (happy, boundary, adversarial, linguistic,
   multi-turn); writes cases only via `evalbuilder dataset add`; self-filters case quality
   and applies input evolutions (concretizing, constrained, comparative, multicontext);
   keeps everything pending; hands off to human review (`evalbuilder review` only after an
   explicit user decision). References: generation-guide.md (distribution knobs, evolution
   vocabulary, golden-answer modes llm_written | agent_backfilled).
3. **agent-eval-mock** — "Make scenarios deterministic with tool mocks." Reads the
   dataset, identifies cases whose tools are nondeterministic/side-effecting/unavailable;
   authors ADK-style rules (per-case first, dataset-level defaults for shared fixtures);
   verifies with `evalbuilder run --dry-run --mock` that every mocked case matches a rule;
   documents the real-tool fallback decision per dataset. References: mocking-semantics.md.
4. **agent-eval-run** — "Validate, execute, score, report." `evalbuilder dataset validate`
   → `evalbuilder run` (mocks installed from the dataset; trajectory captured) →
   `evalbuilder score` with evaluators.yaml → report per-metric stats **and slices**,
   never only an aggregate. Optional LangSmith: `evalbuilder publish` (approved cases
   only, read-back verified) and `--backend langsmith` to record the experiment. Simulate
   mode for conversation scenarios: `evalbuilder simulate` with persona/goal/stop
   conditions; only expectation-violating runs become new pending cases
   (`metadata.source=simulation`). References: evaluator-selection.md, simulation.md.

## 8. LangGraph integration contract

- **Target factory.** The dataset's `target` names a module exposing
  `build_agent(model=None, tools=None) -> CompiledStateGraph` (and optionally `TOOLS`).
  The runner imports the module; when mocking, it wraps `TOOLS` (or the factory's
  default tools) with `evalbuilder.mocking.wrap_tools` — each wrapped tool is a
  `StructuredTool` preserving name/description/`args_schema` — and passes them to the
  factory. If an agent repo lacks the factory, the discover skill helps the user write a
  thin adapter (documented contract, ~10 lines).
- **Execution.** `graph.invoke({"messages": [...]}, config)` per case (multi-turn cases
  replay turns sequentially through a checkpointer thread). The final state's `messages`
  are converted to OpenAI-style via `convert_to_openai_messages` → the **trajectory**;
  actual tool calls extracted from AIMessage.tool_calls; node path captured via
  `stream_mode="updates"`. Output envelope: `{outputs, trajectory, tool_calls, node_path,
  error}` — the shape all evaluators read.
- **Discovery.** AST parse of the agent source (StateGraph nodes/edges,
  `create_react_agent` calls, `@tool` functions with docstring + annotations → args
  schema, prompt string literals) with **live introspection fallback**
  (`graph.get_graph()`, `tool.args_schema`) when the module imports cleanly. Static parse
  never requires importing user code.
- **Determinism for tests.** Examples accept a scripted fake chat model
  (`GenericFakeChatModel`-style with canned tool-call sequences) so the whole pipeline
  runs offline in CI; real models are opt-in via env.

## 9. Storage & configuration

- Artifacts live in the **target project** under `eval/` (configurable `--eval-dir`).
- `.env` via python-dotenv: `EVALBUILDER_JUDGE_MODEL` (default
  `anthropic:claude-sonnet-5`), `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`,
  `LANGSMITH_PROJECT`, model-provider keys. `evalbuilder check` prints the capability
  matrix `{ready, degraded:[...], blocking:[{issue, fix}]}` — LangSmith missing is
  *degraded* (local-only), never blocking.
- LangSmith publish is idempotent: reconcile by reading back examples and matching
  `metadata.local_case_id`; bulk-create response ordering is not trusted; any mismatch
  raises instead of silently corrupting local publication state.

## 10. Extensibility (ADK, then Dify)

A small `FrameworkAdapter` seam in `target.py` + `discover.py` + `mocking.py`:
`discover(source) -> AgentMap`, `load_target(spec) -> callable`,
`install_mocks(target, rules)`, `capture_trajectory(result)`. The dataset schema is
framework-neutral (framework specifics under `metadata` / `target`). The ADK adapter maps
1:1: AgentMetadata → agent-map; `before_tool_callback` mocking (already implemented in
`adk-eval-ext`'s `poc_split_eval`); `.evalset.json` import/export becomes an `import`
mapping. Dify follows the shadow-app/DSL-rewrite route later. MVP ships the interface +
the LangGraph adapter only, plus an ADK section in each skill's references marking the
extension points.

## 11. Example agents & testing strategy

- **examples/weather_bot** — `create_react_agent`, tools `get_weather`, `get_alerts`.
  The smoke-test target.
- **examples/travel_planner** — supervisor graph with `flight_agent` and `hotel_agent`
  subagents (handoff tools) and tools `search_flights`, `book_flight`, `search_hotels`,
  `book_hotel`. Exercises subagent trajectories, conditional routing, mock rules, and the
  constraint "never book without explicit confirmation".
- **Tests (pytest, offline):** unit tests for schemas/normalization/IDs/coverage math,
  mock-rule matching (subset/wildcard/strict/fallback, install–uninstall restore),
  trajectory capture with the scripted model, runner preconditions (pending cases refuse
  to run), evaluator envelope, LangSmith publish with a mocked client (read-back
  mismatch raises). **End-to-end skill-flow test**: discover → dataset init/add/validate
  → review approve → run --mock → score → report against `travel_planner` with the
  scripted model, asserting scores and slices — the executable contract the skills rely
  on.
- **Skill verification:** author with `superpowers:writing-skills`; verify each SKILL.md's
  commands against the CLI in the e2e test (commands named in skills must exist and
  return the shapes the prose reads).

## 12. Assumptions & open questions (for user review)

1. **Skill naming** `agent-eval-{discover,dataset,mock,run}` and CLI name `evalbuilder` —
   rename freely; nothing else depends on the names.
2. Simulation is folded into **agent-eval-run** (a mode + scenario file) rather than a
   fifth skill, and the human-approval gate is folded into **agent-eval-dataset**'s
   workflow, to keep the MVP at four skills. Both split cleanly later if preferred.
3. Default judge model `anthropic:claude-sonnet-5` via `init_chat_model`; requires
   `ANTHROPIC_API_KEY` (or override to any provider OpenEvals supports).
4. MVP mocks **tools and tool-shaped subagent handoffs** only; arbitrary graph-node
   mocking deferred (matches ADK, where delegation is also intercepted as a tool call).
5. Datasets live in the target repo under `eval/`; this repo's `examples/` serve as the
   test targets for the MVP.
6. Python 3.12+ via uv; deps: langgraph, langchain (core + anthropic/openai extras
   optional), openevals, agentevals, langsmith, typer, pydantic v2, python-dotenv,
   pyyaml, pytest.
7. LangSmith usage is limited to datasets + experiments (`client.evaluate`); trace-mining
   existing production runs into candidate cases (as `difytest mine-traces` does) is a
   fast follow, not MVP.

## 13. Out of scope (MVP)

Streamlit/TUI browsing UI; ADK and Dify adapters (interface only); LangSmith trace
mining; DeepEval integration; τ-bench-style stateful-environment verification; pass^k
reliability reporting; CI pipelines.
