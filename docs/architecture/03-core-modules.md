# The deterministic core (`src/evalbuilder/`)

Each section: what the module does, how it is used, what it depends on, the decisions
inside it and their reasoning, and its limitations. The pipeline-only modules are in
[04-pipeline.md](04-pipeline.md).

## Artifacts and schemas (`schemas.py`, `artifacts.py`)

`schemas.py` holds the pydantic models for every interactive artifact:

- `AgentMap` (`evalbuilder/agent-map/v1`): `source_sha256`, `app`, `graph`
  (`nodes`, `edges`, `conditional_edges`, `live`), `tools[]`, `constraints`,
  `data_domains.topics`, `intents[]`, `scenarios[]`, `failure_scenarios[]`,
  `decisions_needed[]`. `extra="allow"` so discovery can add fields (tool schemas,
  edge cases) without a schema bump.
- `Dataset` (`evalbuilder/dataset/v1`) with `Case` (`inputs`, `reference_outputs`,
  `metadata`, `review{status, note}`, `publication{langsmith_example_id}`), `target`
  (`module`, `factory`), dataset-level `mocks`, `langsmith` ids.
- `RunArtifact` (`evalbuilder/run/v1`) with `CaseRun` (`outputs`, `trajectory`,
  `tool_calls`, `node_path`, `error`, `error_class ∈ none|agent|infrastructure`).
- `Report` (`evalbuilder/score-report/v1`; the legacy id `evalbuilder/report/v1` is
  still accepted by the layout registry): `metrics`, `slices`, `cases`.

`artifacts.py` is the only place that loads, saves, normalises and validates datasets:

- `case_id(raw)` — `case-<sha256[:10]>` over `(inputs, intent, topic, scenario,
  failure_mode)`. **Why a content hash:** the same input under a different coverage
  cell is a different test; duplicate cells are detected for free; ids are stable
  across regeneration so review decisions and LangSmith example ids stay attached.
- `normalize_case` — defaults the coverage cell (`unspecified` / `none`), sets
  `source=synthetic`, and **always** resets `review` to `pending` with the note
  "New/generated/imported content can never grant itself approval." That single line
  is the review state machine's invariant.
- `add_case` refuses duplicate ids; `import_cases` maps foreign shapes
  (`cases`/`examples`/`goldens`, `outputs` → `reference_outputs`, `additional_metadata`)
  and stamps `source=import`; `set_review` is the only writer of `review.status`;
  `validate_dataset` checks schema id, unique ids, non-empty inputs, review status
  values and the mock blocks (dataset-level and per-case).

Limitations: the dataset format is deliberately generic (LangSmith/OpenEvals shape),
so framework-specific facts live under `metadata`; there is no migration tooling
beyond the legacy-name/schema acceptance in the layout registry.

## Discovery (`discover.py`)

Two complementary views of a LangGraph module:

1. **AST parse** (`discover_from_source`) — never imports user code. Walks the module
   for: module-level string constants (prompts) and list constants (tool lists);
   `@tool` functions (docstring, `args_schema=` keyword, argument annotations, return
   annotation); pydantic / `TypedDict` classes defined at module level
   (`_pydantic_models`: field types, `Literal` → `enum`, `Optional`/`X | None` →
   `anyOf`, `list[X]`, nested model references as `$ref` + `$defs`, `Field(default,
   description, ge/le/gt/lt, min_length/max_length, pattern)`); `create_agent` /
   `create_react_agent` calls (LLM nodes named by the assignment target, prompt from
   `system_prompt`/`prompt`/… keywords or the first long string, tool lists resolved
   through list constants and `IfExp`); `add_node`, `add_edge`,
   `add_conditional_edges` (targets from the dict mapping). When an LLM node has no
   resolvable tool list, the tools mentioned in its prompt are linked
   (`tools_source: prompt`) — dynamically built tool lists are opaque to the AST.
   **Skills**: skill folders are resolved from `load_skills(...)` calls, `SKILLS_DIR`-style
   constants (`"skills"`, `Path(__file__).parent / "skills"`, `Path(__file__).with_name(..)`,
   `os.path.join(os.path.dirname(__file__), ..)`) and `skills=` factory keywords, loaded
   from disk (never executed) and recorded under `skills[]`; **composed prompts**
   (`BASE + skills_inline_prompt(SKILLS, "x")`, f-strings, `x if c else y`, string
   methods, module or factory-local assignments) are rendered — the skill helpers
   from the loaded skills — so the recorded prompt is what the model sees, and each
   node lists the skills it embeds or names (`skills`, `skills_source`: `inline` |
   `listing` | `prompt`); `skill_loader_tool(...)` assignments and loader-named `@tool`
   functions become tools of `kind: skill_loader`, `mockable: false`. Tools are ordered
   as the module's `TOOLS` list orders them.
2. **Live introspection** (`discover_live`) — imports the module, calls the factory,
   reads `graph.get_graph()` nodes and edges. Failures are recorded as `graph.live.error`,
   never raised: a map without a live view is still useful.

Every node and conditional edge carries `evidence: ["source:<file>:<line>"]`; the
map records `source_sha256` so a stale map is detectable.

**Decisions.** AST-first because importing user code can have side effects, need
credentials, or simply fail; live enrichment (tool schemas from `tool_call_schema`,
compiled graph edges) is layered on top when it works. LLM nodes and graph nodes are
listed separately with the same id (an LLM node created by `create_agent` and then
`add_node`d) — the UI merges them by id (`merged_nodes`).

**Limitations.** Only the patterns above are recognised: graphs built through helper
functions, factories that take config objects, `Send`-based fan-out, subgraphs added
via `add_node(name, compiled_subgraph)` (they appear as opaque graph nodes), prompts
built through templates or functions other than the skill helpers, and tool lists
computed at runtime are partially or not captured; a skills folder computed at runtime
(env vars, `glob`) is not found by the AST — the live `SKILLS` attribute still is. Node kinds are only `llm` and `graph-node` (no router/tool-node
classification from the AST beyond conditional edges).

## Agent skills (`skills.py`)

Agent Skills are `SKILL.md` folders (YAML frontmatter `name`, `description`,
`allowed-tools`, `metadata`; a markdown body of instructions; optional `references/`
and `scripts/`). The module is pure data plumbing:

- `load_skills(paths)` → `Skill(name, description, path, dir, body, metadata,
  references[{path, title, chars, excerpt}], scripts)` for a skills root, one folder or
  one file; a missing `name` defaults to the folder name.
- `skills_inline_prompt(skills, *names)` embeds bodies into a prompt (`## Skill: <name>`
  sections); `skills_prompt(skills)` lists names + descriptions for on-demand use;
  `skill_loader_tool(skills, name="load_skill")` builds the tool that returns a skill's
  body and reference list, tagged `metadata.kind = "skill_loader"`;
  `is_skill_loader(tool)` recognises that tag or a loader-like name
  (`load_skill`, `read_skill`, `get_skill`, `use_skill`).
- `describe_skill(skill, tool_names)` → the agent-map entry (`prompt` = body,
  `allowed_tools`, `tools_mentioned`, `references`, `used_by`, `evidence:
  ["skill:<name>"]`).

**Decision.** Skills are first-class map entries rather than prompt text, because the
generator must know *which* behavioural rules come from a skill (to test `skill_misuse`
and to count coverage per skill) and because a loader tool must be recognised as
local and deterministic — never mocked. The examples use the module's helpers, but
discovery does not require them: any `SKILLS_DIR`-style constant, `load_skills(...)`
call, `skills=` factory keyword or loader-named `@tool` is recognised.

## Tool schemas and edge cases (`tool_schemas.py`)

Everything here is pure Python — no LLM:

- `resolve_args_schema(tool)` → the JSON schema of `tool.tool_call_schema` plus the
  `schema_source`: `args_schema` when an explicit pydantic class was passed (its module
  is not `langchain_core.*`), else `annotations`.
- `resolve_output_schema(tool)` → the JSON schema of the return annotation via
  `pydantic.TypeAdapter` (models, TypedDicts, dataclasses, primitives; the root
  `title` is kept because it is the model name). `{}` when unknown.
- `describe_tool(tool)` → name, description, both schemas, `models` (names from
  `$defs` + titled roots), `side_effecting` (docstring regex: "side-effect", "only call
  after", "after the user confirms"), `kind` (`tool` | `skill_loader`), `mockable`
  (false for skill loaders, which also get no edge cases), `edge_cases`.
- `validate(value, schema, partial=False)` — a JSON-schema *subset*: `type`
  (int is not bool), `properties`/`required`/`additionalProperties`, `items`,
  `min/maxItems`, `enum`/`const`, `anyOf`/`oneOf`/`allOf`, `$ref → $defs`, nullable
  type lists, `min/maxLength`, `pattern`, `minimum`/`maximum`/`exclusive*`. `partial`
  skips `required` at every level for subset matchers (`matchArgs`,
  `expected_tools[].args`). Unknown keywords are ignored.
- `example(schema)` — a conformant sample (first enum value, non-null branch,
  limits respected, simple `^\d{n}$` patterns satisfied); `corrupt(schema, kind)` —
  a non-conformant one (`missing_required` drops the first required field,
  `wrong_type` mistypes it).
- `edge_cases(tool)` — deterministic edge cases from the schemas, round-robin over
  kinds so the first N are diverse: `missing_required` per required field,
  `wrong_type` for non-string / patterned / formatted / nested fields, `out_of_enum`
  for enums, `boundary` for declared limits, `malformed_output` when the output schema
  has required fields. Each carries `failure_mode` (`input_validation` for input
  kinds, `tool_error_handling` for malformed output), an `expected_behavior` sentence
  and `evidence: ["schema:<tool>.<field>"]`.

**Reasoning.** A hand-rolled subset validator avoids a `jsonschema` dependency and
keeps error messages short and path-addressed (`$.ticket.priority: 'p9' not in enum`),
which is what the generator's repair prompt and the report need. Edge cases are
derived by code so they exist for every tool with a schema regardless of what the LLM
decides to write, and their expected behaviour is stated once, consistently.

**Limitations.** Edge cases look only at top-level properties of the args schema —
constraints inside nested models are not expanded into edges (a nested `Literal` will
not produce an `out_of_enum` edge; the `loan_desk` example added a top-level Literal
argument for that reason). `example()` cannot satisfy arbitrary regex patterns. The
validator does not implement `format` semantics, `dependentRequired`, `if/then`,
`uniqueItems`, or `$ref` outside `#/$defs/`.

## Target contract and execution (`target.py`, `runner.py`)

The **target contract** is the only thing an agent repository must provide: a module
exposing `TOOLS` (a list of LangChain tools) and `build_agent(model=None, tools=None)`
returning a compiled graph. `build_graph` passes only the overrides it was given so a
factory's own defaults (e.g. a scripted model) still apply.

`run_case` streams the graph with `stream_mode=["updates", "values"]` to capture the
node path and the final state, replays `metadata.user_turns` sequentially for
multi-turn cases (appending to the same message list), converts messages to
OpenAI-style dicts (`convert_to_openai_messages`) as the **trajectory**, extracts
`tool_calls` from `AIMessage.tool_calls`, and returns the last message's content as
`outputs.response`. Any exception becomes `error_class=agent`.

`run_dataset` selects approved cases (or explicit ids, refusing non-approved ones),
rebuilds the graph **per case** with tools wrapped by the merged mock rules
(dataset-level overridden per tool by per-case rules) and the injected model, and
classifies factory failures as `infrastructure`. It writes `run-<uuid8>.json`.

**Reasoning.** Rebuilding per case is what makes per-case mock overrides (error
injections) possible with immutable compiled graphs. Separating `agent` from
`infrastructure` errors lets aggregation and the report keep "the agent misbehaved"
apart from "the harness could not run".

**Limitations.** Multi-turn replay re-invokes the graph with the accumulated message
list — graphs relying on a checkpointer thread id for state get a fresh state each
turn; `outputs.response` is the last message's text (structured/state outputs beyond
`messages` are not captured); there is no per-case timeout; parallelism is per intent
group only (`max_workers` > 1 runs intent groups in threads, cases within one intent
stay sequential, and an optional `progress` callback reports each completed case);
the graph must accept `{"messages": [...]}` as input.

## Mocking (`mocking.py`, `mock_engine.py`)

Two layers behind one wrapper. `docs/tool-mocking.md` is the full walkthrough.

**Layer 1 — rules** (`mocking.py`). `args_subset` (recursive subset), `match_index` /
`match_rule` (first match wins, `{}` wildcard), `wrap_tool(tool, rules, on_miss,
fallback, engine=, ledger=)` → a `StructuredTool` with the original name /
description / `args_schema` whose function answers from the rules (responses deep-copied
per call), else by policy: `real` (invoke the tool), `fallback` (a canned value),
`strict` (`MockMissError`), `llm` (the engine). `wrap_tools` wraps the tools that have
rules and — with an engine — every mockable tool; `mockable(tool)` excludes skill
loaders. `merge_mock_rules` (case list replaces dataset list per tool), `with_fallback`
(case rules + dataset rules unless the case ends with a wildcard), `verify_dataset`
(expected calls without a rule) and `verify_summary` (the same read through the
policy: under `llm` misses are informational `llm_answered`). Every answered call can
be appended to a ledger (`layer`, args, response, rule index or the engine's record);
`ledger_totals` sums layers plus `invalid`.

**Layer 2 — the LLM mock engine** (`mock_engine.py`). `LLMMockEngine(model,
strategies, tool_specs, strategy, on_invalid, max_repairs)` — one per conversation.
`respond(tool, args)` builds one prompt (world, strategy description and the tool's
behaviour, examples, tool definition with both schemas, the conversation's previous
calls, the call in a `TOOL CALL:` block), asks for structured output (the tool's object
`output_schema` retitled `mock_response`, or a `response_json` envelope), validates with
`tool_schemas.validate`, re-asks once with the problems, then returns the strategy's
`fallback_response` (`on_invalid: fallback`) or raises `MockEngineError` (`strict`; the
runner classifies it as an infrastructure error). Strategy lookup: the selected
strategy's entry → `default`'s → a generic behaviour from the tool description.
`validate_strategies` checks a strategies document against the tools' schemas;
`call_in_prompt` parses the call block (scripted mock models in the examples use it to
answer offline).

**Decisions.** Rules stay data in the dataset and answer first, so a deterministic
fixture is never overridden by a model. The engine gets *strategies*, not free rein:
behaviour text, examples and a validated fallback per tool, generated once in the
`mocks` stage and embedded in the dataset — the same plan-plus-model shape ADK uses
for its user simulator, applied to tools. Validation is the tool's own `output_schema`
through the same validator the fixtures pass; an unfixable answer is a mock problem,
never an agent failure. The ledger exists so the stability report can attribute
instability to LLM-mocked calls.

**Limitations.** Static responses per rule (no call counting or sequences); tool calls
only (nodes calling services directly are not intercepted); wrapped tools drop
tool-level settings such as `handle_tool_error`; the engine's history is per
conversation, not per dataset (two cases can get different answers for the same
unknown id — by design, and visible in the ledger); prompt size grows with the
strategies document and the tool schemas.

## Evaluators (`evaluators.py`)

One metric per evaluator; each is `fn(case, case_run) -> {key, score, comment}` or
raises `EvaluatorNotApplicable` (case has no reference → `skipped`) /
`EvaluatorUnavailable` (judge missing or failed → `error`). `score_run` scores every
case of a run, records per-case `scores`/`errors`/`skipped`, per-metric
`{n, avg, min, max, errors, skipped}`, and per-slice means for `intent`,
`failure_mode`, `variant`. A case that errored during the run scores 0 on every
metric with the error as comment. With `max_workers > 1` the case runs are scored
concurrently in a thread pool (each case still runs its evaluators in order); rows,
metrics and slices are aggregated in run order afterwards, so the report is identical
to a sequential pass.

Types: deterministic `expected_tools` (ordered subsequence of calls with recursive
args subset, plus `forbidden_tools`), `contains` (case-insensitive substring(s)),
`json_valid`, `trajectory_match` (AgentEvals, needs a reference `trajectory`);
judges `correctness` (OpenEvals `CORRECTNESS_PROMPT`), `contract` (a fixed prompt
around the case's `reference_outputs.contract`), any OpenEvals rubric by name or
`type: openevals` + `prompt`, `trajectory_llm` (AgentEvals trajectory judge);
`custom` (`ref: module:function`). Judges are built with `create_llm_as_judge(judge=
<LangChain model>)`, so any `provider:model` spec works. A judge rationale that looks
like a placeholder (`Test.`, `placeholder`, `lorem`) triggers one retry.

**Reasoning.** Judges accept a LangChain model object — that is the single seam that
makes the Claude CLI adapter, API-key providers and scripted models interchangeable.
Skipped ≠ error keeps "this case has no `contains` reference" from polluting error
counts; error ≠ agent failure keeps a missing API key from looking like a regression.

**Limitations.** Judges are binary by default (`continuous: true` is passed through
but thresholds assume pass rates); judge variance is only detected across repeats
(the pipeline's stability analysis), not calibrated; `contains` is a plain substring
check; `expected_tools` compares argument values by equality after subset selection
(no fuzzy matching of ids or dates).

## Simulation (`simulate.py`)

Scenarios (YAML) have `id`, `persona`, `goal`, `opening`, `followups[]`, `max_turns`
and stop conditions (`success_contains`, `expect.contains`, `expect.not_contains`).
`simulate_scenario` drives the graph turn by turn: scripted follow-ups first, then —
when a `user_model` is given — a simulated user prompted with persona, goal and the
last six transcript entries. Stop reasons: `success` (needle seen), `exhausted` (no
more user messages), `max_turns` (a truncation, reported as a violation when a
success needle was expected). `simulate_scenarios(graph_factory, scenarios,
user_model=, max_workers=)` runs a scenario list — concurrently when
`max_workers > 1`, each worker building its own graph via the factory so nothing is
shared between threads; results keep scenario order. `mine_failures` turns violating runs into **pending**
cases (`source=simulation`, `failure_mode=simulation-violation`, later turns in
`user_turns`, transcript tail kept for the reviewer).

**Limitations.** The simulated user is a plain prompt, not OpenEvals' simulated-user
harness; expectations are substring checks on the concatenated assistant text; each
turn re-invokes the graph with the accumulated transcript as plain role/content
messages (tool-call history is not replayed).

## Coverage grid (`coverage.py`)

`required_cells(agent_map)` = every scenario × every topic (or `unspecified`) plus
one cross-cutting cell per failure scenario; `coverage_gaps` counts cases per cell
key `intent/topic/scenario/failure_mode` against `target_per_cell`. This is the
interactive skills' notion of coverage; the pipeline's planner
(`pipeline/planning.py`) is richer (counts, variants, multi-turn share, schema-edge
cells) but uses the same cell key so `achieved` and `gaps` agree.

## Providers and model specs (`providers.py`, `claude_cli.py`, `config.py`)

All models are addressed as `provider:model[@effort]`:

- `claude-cli:<model>` — `ChatClaudeCLI` (`claude_cli.py`), a subclass of
  `langchain_claude_code.ChatClaudeCode` with a **subprocess transport**: each call is
  one `claude -p --output-format json --model … --tools "" --setting-sources ""
  --strict-mcp-config --disable-slash-commands --no-session-persistence` in a neutral
  temporary cwd with `ANTHROPIC_API_KEY` scrubbed (so the subscription is used, and no
  project `CLAUDE.md`/memory leaks in). Conversation history — including assistant
  tool calls and tool results — is flattened into a transcript because the CLI cannot
  replay assistant/tool turns. Tool calling and `with_structured_output` both use
  `--json-schema`: bound tools are described in the system prompt and the reply must
  match `TOOL_CALL_SCHEMA` (`content` + `tool_calls`); tool-call JSON accidentally
  placed in `content` is unwrapped. One retry with backoff on transport errors,
  `is_error` results or unparsable output; `ClaudeCLIError` otherwise. `@effort` maps
  to the CLI's `--effort`.
- `anthropic:`/`claude:`, `openai:`, `gemini:`/`google:`/`google_genai:` — API-key
  providers in a registry (`Provider`: env var, package, `uv sync --extra …`, example,
  effort mode). `build_model` → `init_chat_model(model, model_provider=…)`; `@effort`
  becomes OpenAI `reasoning_effort` or an Anthropic thinking budget
  (1024/4096/16000 tokens) and is ignored for Gemini. `provider_ready(spec)` checks
  key **and** package and returns a fix string, so preflight fails before any network call.
- `scripted:module.path:factory` — offline models for tests and the offline
  generators; always "ready"; the current directory is put on `sys.path` before the
  import.

`config.py` reads `.env`/environment into `Settings` and computes the capability
matrix (`capability_check`): LangGraph present (blocking if not), Claude CLI on PATH,
each provider's readiness, judge readiness (degraded), LangSmith key (degraded),
target importable (blocking).

**Reasoning.** The CLI transport exists because the packaged `claude-code-sdk`
transport crashed on the current CLI's event stream and its tool calling parsed the
whole reply as JSON; a single isolated subprocess per call is slower (~3–10 s per agent
call, ~15–20 s per judge call) but robust and stateless. `--bare` was rejected because
it breaks subscription auth. Keeping *every* model behind one spec string means the
agent, the generator and the judges can be mixed freely (`scripted:` agent with a
real judge, or the reverse).

**Limitations.** The CLI adapter has no streaming (chunks are synthesised after the
fact), no token-level usage accounting beyond what the CLI reports, and cost/latency
that make large runs slow (a 23-case × 2-repeat run with two judges took ~1.5 h); a
usage-limit hit surfaces as `ClaudeCLIError … is_error … total_cost_usd 0` and, in
the pipeline, as judge `errors` — rerun with `--resume --from score` once the limit
resets. API providers require their extras installed.

## LangSmith I/O (`langsmith_io.py`)

`publish_approved` creates or reads the dataset by name, uploads only approved cases
without a `langsmith_example_id` (metadata carries `local_case_id`), then **reads the
examples back** and matches by `local_case_id` because the bulk-create response order
is not trusted; a missing case raises and leaves local state untouched. Success
stores `dataset_id`/`dataset_name` on the dataset and the example id on each case.
Publishing is optional and degraded-not-blocking everywhere.

**Limitations.** Datasets only: no experiment upload (`client.evaluate`) and no trace
mining of production runs into candidate cases.

## Offline testing helpers (`testing.py`)

- `ScriptedChatModel` — a regex-scripted `BaseChatModel`: `script` rules match the
  latest text (`TOOL:<result>` for tool results) and return an `AIMessage` or a tool
  call; `structured_script` serves `with_structured_output` from the full user
  transcript (routers). `bind_tools` returns itself. Every example agent has a
  `default_scripted_model()` so the repository's tests and the UI browser flows run
  with no model at all.
- `SchemaScriptedModel` — dispatches `with_structured_output(schema)` by the schema's
  `title` to canned dicts or callables over the messages (`cells_in_prompt` parses the
  case-authoring prompt's cells) — an offline *generator* for the pipeline
  (`examples/*/offline.py`).

**Reasoning.** Deterministic models make the whole pipeline testable end to end in
seconds, which is what allows the UI flows to be browser-tested against real artifacts
on disk rather than mocks of the UI.
