# Tool mocking: implementation review and framework extension analysis

**Assessment date:** 2026-08-31  
**Scope:** `skills/agent-eval-mock/SKILL.md`, `src/evalbuilder/mocking.py`, the runner/target integration, pure LangChain agents, LangGraph alternatives, and Google ADK.

## Executive decision

Keep the dataset-level rule language, but separate it from framework-specific interception. The rule matcher is a useful, mostly framework-neutral core; the current tool wrapper and runner are LangChain/LangGraph adapters accidentally combined with that core.

For this repository:

1. **Retain ordered rules, recursive subset matching, per-case overrides, and strict misses.** They fit generated evaluation fixtures and make failures understandable.
2. **Make `strict` the safe default whenever mocking is enabled**, especially for CI and side-effecting tools. The current library default of `real` can silently make a supposedly deterministic run call production dependencies.
3. **Introduce a framework-neutral `MockEngine` and an execution record**, then add adapters:
   - LangChain `BaseTool` replacement for explicitly injected tools;
   - LangChain `wrap_tool_call` middleware for current `create_agent` applications;
   - LangGraph `ToolNode`/tool-registry replacement for custom graphs;
   - Google ADK `before_tool_callback` or plugin interception.
4. **Do not describe current LangChain `create_agent` as “pure LangChain, no LangGraph.”** The current API is a LangChain surface implemented on LangGraph. A truly LangGraph-free integration means supporting an application-owned model/tool loop or the legacy `AgentExecutor`, not current `create_agent`.
5. **Do not delegate ADK mocking to the ADK evaluation framework on the assumption that it is already implemented.** Current ADK provides a strong evaluation system and a supported tool-interception callback, but current source does not connect the historical `mock_tool_output` field to eval execution. Reuse ADK evaluation for ADK-native running/scoring when desired; supply this project's rules through a small callback/plugin adapter.

The preferred near-term architecture is therefore **dependency injection first, framework interception second, monkeypatching only as an escape hatch**.

## 1. What exists today

### 1.1 Skill behavior

`skills/agent-eval-mock/SKILL.md` tells an agent to:

- identify nondeterministic, side-effecting, rate-limited, or unavailable tools;
- author ordered rules per tool;
- use recursive subset equality through `matchArgs` and `{}` as a wildcard;
- place shared rules at dataset scope and override them at case scope;
- verify expected calls before running;
- choose among `real`, `fallback`, and `strict` miss behavior;
- avoid unmocked side effects in CI and inject failures through fixtures.

This is concise and operationally sound. It also correctly keeps mock responses separate from expected outcomes: a fixture controls the environment, while `reference_outputs` defines success.

### 1.2 Runtime implementation

`src/evalbuilder/mocking.py` contains four distinct responsibilities:

| Responsibility | Current function | Framework coupling |
|---|---|---|
| recursive argument comparison | `args_subset` | none |
| ordered rule selection | `match_rule` | none |
| dataset/case composition | `merge_mock_rules`, `with_fallback` | none |
| execution interception | `wrap_tool`, `wrap_tools` | LangChain `BaseTool` / `StructuredTool` |

At run time, `run_dataset` imports the target module, reads its module-level `TOOLS`, wraps tools having rules, and asks the target factory to rebuild the agent with `tools=`. The target contract then assumes a LangGraph-style object whose `stream()` supports `stream_mode=["updates", "values"]` and whose state contains `messages`.

The implementation is therefore not “LangGraph mocking” in itself. Tool replacement uses LangChain Core. The surrounding discovery, construction, invocation, state extraction, and node-path collection are what make the complete path LangGraph-specific.

### 1.3 What the tests prove

`tests/test_mocking.py` currently proves:

- first-match-wins and wildcard behavior;
- recursive matching for nested mappings;
- preservation of basic tool name, description, and argument schema;
- real/fallback/strict miss policies;
- whole-list per-case override semantics;
- pass-through for tools without rules;
- CLI authoring and verification of expected calls.

The focused test suite passes: `uv run pytest tests/test_mocking.py -q` reports 9 passing tests in the current worktree.

That is good unit coverage for the small synchronous happy path, but not proof of general tool compatibility or end-to-end safety.

## 2. Strengths

### Rule semantics are deterministic and easy to generate

Ordered rules plus subset matching are a good fit for LLM-generated fixtures. Authors need not repeat optional/default arguments, and specific cases can precede a wildcard. Failure messages include tool name and actual arguments.

### Tool schemas remain visible to the model

Replacing a `BaseTool` with a `StructuredTool` that retains `name`, `description`, and `args_schema` generally preserves the model-facing contract. This is materially better than replacing a structured tool with an untyped `MagicMock`.

### Rebuilding per case isolates fixture selection

The runner constructs a fresh graph for each case. This avoids mutating a shared compiled graph and makes case-level fixtures natural. It is also concurrency-safe as long as the target factory and tools do not share mutable globals.

### Preflight verification catches a useful class of mistakes

`verify_dataset` detects an expected call whose named mock list exists but has no matching rule. This is valuable before an expensive model run.

### The storage format is portable

The core shape—tool name to ordered `{matchArgs, response}` rules—does not depend on LangChain messages, LangGraph state, or ADK events. That makes cross-framework adapters realistic.

## 3. Gaps and risks

### 3.1 Safety mismatch between guidance and defaults

The skill says never to leave a side-effecting tool unmocked in CI, but:

- `wrap_tool(..., on_miss="real")` defaults to the real tool;
- `run_dataset(..., on_miss="real")` also defaults to real;
- tools without any rule pass through untouched;
- `verify_dataset` skips a case entirely when its merged mapping is empty;
- when a referenced tool name is absent from the merged mapping, verification does not report it.

Consequently, `--mock` does not mean “all external effects are mocked.” It means “wrap tools for which rules happen to exist.” A typo in a tool name or a newly added tool can escape to the real integration.

**Recommendation:** define two independent policies:

- `unmatched_call`: `error | real | fallback`;
- `unconfigured_tool`: `error | real`.

Use `error/error` for deterministic runs and allow `real/real` only for explicitly hybrid contract tests. Make the run artifact record both policies and every passthrough.

### 3.2 The wrapper loses important `BaseTool` behavior

Only name, description, and argument schema are copied. Current `BaseTool`/`StructuredTool` behavior also includes:

- async execution (`ainvoke` / coroutine);
- `return_direct`;
- `response_format="content_and_artifact"`;
- tags and metadata;
- callback configuration;
- validation/tool-error handlers;
- tool-specific config and runtime-injected arguments;
- custom `BaseTool` overrides and state/store effects.

The wrapper provides only a synchronous `func`. Async-only tools can fail, blocking tools may run on inappropriate paths, and tuple artifact responses can be reinterpreted under the default content-only format. The claim that identity is preserved is therefore only partially true.

**Recommendation:** either clone all relevant public execution attributes and supply sync plus async interceptors, or prefer execution-time middleware that leaves the original tool object intact.

### 3.3 Invocation semantics are flattened to `**kwargs`

The wrapper assumes a structured mapping input. LangChain tools can accept string input, dict input, `ToolCall` input, injected `ToolRuntime`, or provider/tool-specific structures. Runtime-injected state/context is deliberately hidden from the model schema and may not arrive through ordinary `kwargs` in a replace-and-rewrap design.

This matters for modern tools that read or write agent state, use a store, emit stream updates, return a `Command`, or access credentials from runtime context. A static JSON `response` also cannot naturally reproduce state mutation.

**Recommendation:** define mock outcomes as a tagged union rather than a raw response only:

```json
{
  "matchArgs": {"order_id": "ORD-1002"},
  "outcome": {"type": "return", "value": {"status": "Shipped"}}
}
```

Future outcome kinds can include `raise`, `delay`, `sequence`, and framework-specific state updates. Keep the portable `return`/`raise` core and reject unsupported outcome kinds explicitly in each adapter.

### 3.4 Matching is useful but underspecified

Current matching handles nested dictionaries recursively, but:

- lists require complete equality;
- numbers use Python equality (`1 == 1.0`);
- no distinction exists between a missing key and an explicit `null` beyond ordinary mapping semantics;
- Pydantic/dataclass values are not normalized;
- rule validation accepts extra keys and does not require `response` at the Pydantic model level in a meaningful way (`response` defaults to `None`);
- `{}` matches everything, so a misplaced early wildcard shadows every later rule;
- duplicate or unreachable rules are not diagnosed.

**Recommendation:** document canonicalization and add linting for early wildcards, duplicate matchers, unknown tool names, ambiguous rules, malformed outcomes, and response-schema mismatch. Preserve simple subset equality as the default; avoid adding a general expression language until a real case requires it.

### 3.5 Override semantics differ across code paths and prose

`merge_mock_rules` says a case list fully replaces the dataset list. `with_fallback` can append dataset rules after case rules, and the README says the pipeline performs that append. These can both be valid, but they are different policies and should not share the vague word “override.”

**Recommendation:** encode composition explicitly as `replace` or `prepend_then_dataset`, store the selected policy in the dataset/run artifact, and test both.

### 3.6 Verification is expectation-driven, not execution-safe

The verifier checks `reference_outputs.expected_tools`, not all calls the model might make. It cannot establish that a run is deterministic because an agent can make an unexpected call, call a configured tool with different arguments, or select a tool not represented in the references.

**Recommendation:** rename the current behavior conceptually to “expected-call coverage.” Add a stronger static check that all exposed tools have rules under strict mode, and a runtime ledger that records matched rule ID, unmatched calls, real passthroughs, and unused rules.

### 3.7 Responses are static and no call history is retained

The same matching rule returns the same object on every call. There is no `return once`, response sequence, callback, latency injection, exception injection, call count, or stateful fake. Returning a mutable object repeatedly could also leak mutation between calls.

**Recommendation:** deep-copy JSON fixtures per invocation and add a deterministic `sequence` outcome with exhaustion behavior. Treat arbitrary Python callbacks as a separate programmatic adapter feature, not part of portable JSON.

### 3.8 Error injection is not semantically faithful

The skill suggests returning `{"error": "timeout"}`. That tests an error-shaped successful tool result, not a timeout exception or the graph/framework’s tool error path. It may be appropriate for APIs that represent errors as values, but it does not test exception handling.

**Recommendation:** support `{"outcome": {"type": "raise", "class": "TimeoutError", "message": "..."}}` using an allowlisted exception registry. Keep returned error values for value-level failure contracts.

### 3.9 The target contract is much narrower than the skill implies

The runner requires:

- an importable Python module;
- module-level `TOOLS`;
- a factory accepting `tools=` and optionally `model=`;
- a rebuilt agent per case;
- a streamed LangGraph-style message state.

It cannot transparently mock a precompiled graph, tools closed over inside nodes, dynamic tools, MCP/provider-hosted tools, tools constructed per request, or agents whose factory does not expose injection. “Subagents exposed as tools” works only when they satisfy this same replaceable `BaseTool` contract.

### 3.10 Version bounds are too broad for the relied-upon behavior

`pyproject.toml` declares `langchain>=0.3`, `langchain-core>=0.3`, and `langgraph>=0.2` with no upper bounds while the current environment is LangChain 1.3.16, LangChain Core 1.6.0, and LangGraph 1.2.11. Those ranges span major API changes. A framework adapter needs a tested compatibility window or feature detection.

## 4. Extending to LangChain agents without direct LangGraph code

### 4.1 Important terminology

Current LangChain documentation states that [`create_agent` builds a graph-based runtime using LangGraph](https://docs.langchain.com/oss/python/langchain/agents). Therefore:

- **No direct LangGraph API in application code:** straightforward. Support `langchain.agents.create_agent`; users can stay on the LangChain API surface.
- **No LangGraph dependency/runtime at all:** not possible with current `create_agent`. Use an application-owned model/tool loop or legacy `AgentExecutor` where still maintained by the application.

This distinction should be explicit in the CLI and report. A `framework: langchain` label can describe the user-facing integration, but it must not promise the absence of LangGraph transitively.

### 4.2 Recommended LangChain contract

Generalize `Target` and execution behind an adapter:

```python
class TargetAdapter(Protocol):
    def discover(self, target) -> AgentMap: ...
    def build(self, target, *, model=None, tool_interceptor=None): ...
    def invoke(self, app, case) -> CaseRun: ...
```

For a LangChain-facing target, support one of these explicit factory contracts:

```python
build_agent(model=None, tools=None, middleware=None) -> Runnable
```

or a more decoupled form:

```python
get_tools() -> Sequence[BaseTool]
build_agent(model, tools) -> Runnable
```

The adapter should invoke through the standard Runnable interface (`invoke`, `ainvoke`, optionally `stream`/`astream`) and extract messages from a configurable output path rather than requiring LangGraph update/value stream modes.

### 4.3 Two viable interception methods

#### A. Replace injected `BaseTool`s

This is the smallest extension of the present code and works for application-owned loops, legacy agents, and current LangChain agents whose factory accepts tools.

Pros:

- no dependence on graph internals;
- preserves model-visible schemas when implemented carefully;
- naturally supports the existing target factory pattern;
- easy to use with `AgentExecutor` or a manual `bind_tools` loop.

Cons:

- must faithfully preserve sync/async/config/artifact behavior;
- cannot reach tools hidden inside an already-built agent;
- awkward for runtime-registered or context-dependent tools;
- replacement can break custom tool subclasses.

#### B. Use LangChain agent middleware

Current LangChain agents expose [`wrap_tool_call`](https://docs.langchain.com/oss/python/langchain/agents) middleware around tool execution. A mock middleware can inspect `request.tool_call`, resolve a rule, and return a correctly correlated `ToolMessage` without executing the handler. If no rule matches, it can error or call `handler(request)` according to policy.

Pros:

- keeps the original tool object and model schema intact;
- one interception point handles static tools and can cooperate with runtime tools;
- sees call ID and runtime context;
- naturally supports audit logging and error conversion;
- aligns with the current public LangChain agent API.

Cons:

- applies only to the current `create_agent` middleware runtime;
- the returned value must honor `ToolMessage`, artifact, and `Command` semantics;
- because `create_agent` uses LangGraph internally, it is not a truly LangGraph-free solution;
- middleware ordering can affect behavior (HITL, retries, caching, error handlers).

**Recommendation:** implement both on a shared `MockEngine`. Use middleware for current `create_agent`; use injected tool replacement for generic Runnable/manual-loop/legacy targets.

### 4.4 Truly LangGraph-free agent loops

For a manual loop, the application binds tools to the model, reads `AIMessage.tool_calls`, invokes a registry, appends `ToolMessage`s, and repeats. LangChain’s model documentation explicitly leaves tool execution to the application when a model is used outside an agent ([tool-calling flow](https://docs.langchain.com/oss/python/langchain/models)).

Here the preferred interception is a mock-aware registry:

```python
result = mock_engine.resolve(name, args)
if result.matched:
    value = result.execute()
else:
    value = real_tools[name].invoke(args, config=config)
```

This is the cleanest genuinely LangGraph-free option, but evalbuilder cannot infer every loop’s message/output format. The target adapter must allow application-provided invocation and trace extraction hooks.

### 4.5 Required work estimate

Supporting “LangChain API surface” is a moderate extension, not a rewrite:

1. extract `MockEngine` and rule models from `mocking.py`;
2. make runner dispatch by `target.framework`;
3. add a Runnable-based target adapter and configurable input/output paths;
4. add sync and async execution;
5. add middleware and replacement adapters;
6. add trace normalization into the existing `CaseRun` schema;
7. update discovery because current AST/live graph discovery is LangGraph-oriented.

Supporting arbitrary manual loops is necessarily opt-in through user hooks. Supporting current `create_agent` is easier because messages and middleware are standardized.

## 5. LangGraph tool-mocking alternatives

LangGraph’s documented [`ToolNode`](https://langchain-ai.github.io/langgraph/agents/tools/) is the standard execution node and handles parallel execution, error handling, and state injection. The best method depends on whether the graph is built for testability.

| Method | How it works | Pros | Cons | Fit here |
|---|---|---|---|---|
| **Tool dependency injection / wrapper** | Rebuild graph with mock-aware tools, as today | Public APIs; schema visible to model; simple; portable to LangChain | requires injectable factory; wrapper fidelity burden; misses hidden/dynamic tools | **Good default now**, after hardening |
| **Mock-aware `ToolNode` or registry substitution** | Rebuild graph with a test `ToolNode`, or replace the tools node before compile | central interception; preserves call IDs; can handle parallel calls/state; easy ledger | graph topology contract required; custom tool nodes vary; must reproduce `ToolNode` semantics | **Preferred for custom LangGraph adapter** |
| **LangChain `wrap_tool_call` middleware** | Intercept execution in `create_agent` | original tools unchanged; public hook; runtime context available | only prebuilt/current agent path; middleware ordering; internally LangGraph | **Preferred for `create_agent`** |
| **Patch tool function/object with `pytest`/`unittest.mock`** | Monkeypatch the symbol used by a node/tool | minimal production changes; useful for unit tests | import-path brittle; may patch wrong bound reference; global/concurrency risk; fixtures live in code; weak CLI portability | escape hatch only |
| **Replace graph node / compile a test graph** | Substitute the whole tool node or external-service node | can simulate state updates, interrupts, retries, multi-tool behavior | couples tests to graph topology; may bypass real serialization/error logic; harder generated fixtures | use for graph-component tests |
| **Replay at model/trace boundary** | Script model messages/tool results or replay a recorded trace | completely offline and deterministic; excellent regression speed | does not test live model tool selection; recordings become stale; easy to overfit | complementary lower test layer |
| **Fake service behind real tool** | Real tool calls an in-memory/HTTP fake dependency | tests schema, serialization, retries, and tool code; framework-neutral | more setup; fake fidelity/drift; slower than response fixtures | preferred contract/component layer |
| **Interrupt before tools and resume with supplied result** | Graph pauses on tool action; harness approves/edits/resumes | tests HITL and persistence semantics; explicit side-effect gate | cumbersome for batch mocking; checkpoint/thread management; not a general fixture engine | specialized HITL tests |

### Preferred combination for this context

Use a layered strategy:

1. **Generated deterministic evaluation suite:** rule engine plus strict execution interception. For custom graphs, inject a mock-aware `ToolNode`; for `create_agent`, use middleware; retain hardened wrappers as a compatibility path.
2. **Tool contract suite:** run the real LangChain tool against fake/sandbox services to catch schema, serialization, retry, and auth drift.
3. **Small staging suite:** run real integrations under restricted credentials and explicit budgets. Never infer integration correctness from mocks alone.
4. **Replay/scripted-model suite:** keep fast orchestration regression tests, as the repository already does with scripted chat models.

This is preferable to monkeypatching because evalbuilder stores and generates fixtures as data, rebuilds targets per case, and needs reproducible CLI behavior rather than test-module-specific patches.

### Why a mock-aware `ToolNode` is not the only adapter

Some graphs execute tools inside ordinary nodes, call external clients directly, nest agents, or use provider-native server-side tools. Replacing `ToolNode` cannot intercept those. Discovery should classify each capability as:

- client-executed injectable tool;
- client-executed hidden/dynamic tool;
- graph node side effect;
- provider/server-side tool;
- subagent handoff.

Only the first category is automatically safe under the current wrapper design. Hidden tools need middleware/registry cooperation, node effects need dependency injection or node substitution, and provider-hosted tools may require provider recording/replay or disabling the feature.

## 6. Google ADK extension

### 6.1 What ADK currently provides

Current Google ADK has a native evaluation data model, local/GCS eval-set managers, response and trajectory metrics, multi-turn/session support, and runner integration. Its current `EvalCase` stores expected tool uses and tool responses as intermediate data. ADK’s tool trajectory evaluator can grade exact/in-order/any-order variants depending on metric configuration.

For interception, `LlmAgent.before_tool_callback` is a supported hook. Its documented contract is decisive: returning a tool response skips the actual tool; returning `None` continues normal execution. ADK plugins also expose a `before_tool_callback`, and plugin callbacks run before agent callbacks. See the [ADK callbacks documentation](https://google.github.io/adk-docs/callbacks/types-of-callbacks/) and assessed [ADK Python source](https://github.com/google/adk-python/blob/ed1306f58796648a58b5ee2f99f80e81e92cd7de/src/google/adk/agents/llm_agent.py).

### 6.2 Correction concerning built-in eval tool mocking

Older ADK examples and secondary tutorials describe legacy `.test.json` entries with `expected_tool_use[].mock_tool_output`. That explains this repository’s “ADK-style” terminology and README statement.

However, as of the assessment date, current `google/adk-python` source at commit `ed1306f` shows:

- `MOCK_TOOL_OUTPUT = "mock_tool_output"` remains in `evaluation_constants.py`;
- the current `EvalCase`/`IntermediateData` models store tool uses and tool responses, but no mock-output rule field;
- the legacy `.test.json` converter reads only tool name and tool input into expected `FunctionCall`s;
- repository-wide source search finds no evaluation runtime use of `MOCK_TOOL_OUTPUT`;
- tool execution is bypassable through `before_tool_callback`, independently of the evaluation framework.

Therefore, **current ADK evaluation should not be assumed to mock tools from eval data automatically**. The historical field appears vestigial in the current codebase. This conclusion is based on current source inspection; pin and retest any ADK version before relying on it.

### 6.3 Is evalbuilder easily extendable to ADK?

The **mock rule engine is easily extendable**. An async callback can resolve `tool.name` plus `args` and return a fixture on match. It naturally supports ordered rules and strict/fallback policies. A plugin is preferable when the adapter must apply across a multi-agent tree without modifying every `LlmAgent`; an agent callback is simpler for one root agent but must be composed carefully with existing callbacks.

The **whole evalbuilder pipeline is not a one-to-one adapter**. Work is needed for:

- ADK agent discovery and tool/schema extraction, including `BaseTool`, functions, toolsets, built-in tools, and subagents;
- Runner/session/artifact service setup;
- async event collection and normalization;
- mapping ADK `Event`, `FunctionCall`, and `FunctionResponse` records to `CaseRun`;
- multi-agent authorship and transfer traces;
- session state and artifact isolation;
- eval-set import/export without losing ADK-specific rubrics or invocation events;
- toolsets that resolve tools dynamically;
- built-in/provider tools that do not traverse ordinary callbacks.

So the correct characterization is: **small mocking adapter, moderate target/trace adapter, substantial full-fidelity evaluation integration**.

### 6.4 ADK callback adapter shape

Conceptually:

```python
async def before_tool_callback(tool, args, tool_context):
    decision = mock_engine.resolve(tool.name, args)
    if decision.matched:
        return decision.materialize()
    if policy.unmatched_call == "real":
        return None
    raise MockMissError(tool.name, args, decision.considered_rules)
```

Important details:

- attach the callback without overwriting application callbacks; preserve ordering;
- prefer a run-scoped plugin/clone over mutating the production singleton agent;
- ensure rule and call ledgers are isolated per eval case;
- make async/concurrent tool calls safe;
- return the response type each ADK tool expects;
- explicitly detect tools that bypass callbacks;
- decide whether transfer functions and subagent handoffs are mockable tools or trajectory events.

### 6.5 Evalbuilder versus ADK-native evaluation

| Choice | Advantages | Disadvantages | When to choose |
|---|---|---|---|
| **ADK-native evaluation + evalbuilder callback plugin** | preserves ADK sessions/events/metrics/UI/CLI; less adapter code; framework behavior stays canonical | separate artifact format and workflow; evalbuilder generation/review/scoring features need bridges; two configurations | primarily ADK estate; maximize native fidelity |
| **Full evalbuilder ADK adapter** | one cross-framework dataset, review workflow, coverage generator, reports, and CLI | must track fast-moving ADK APIs; trace/session fidelity burden; duplicate native eval functionality | cross-framework organization values one evaluation control plane |
| **Export evalbuilder cases to ADK eval sets** | pragmatic separation; native execution plus shared case generation | lossy mapping risk; mock rules still need callback/plugin; result re-import needed | recommended first integration milestone |

### Recommendation for ADK

Start with an **export/import bridge plus a run-scoped ADK mock plugin**, not a replacement ADK runner:

1. generate/review cases and rules in evalbuilder;
2. export inputs, references, expected tool calls, and applicable rubrics to current ADK `EvalSet`;
3. install evalbuilder’s rule engine through `before_tool_callback` during the ADK eval run;
4. import normalized ADK results and traces for aggregate reporting;
5. use ADK-native metrics for ADK-specific trajectory/session behavior and evalbuilder metrics where cross-framework comparability matters.

This reuses ADK’s strongest part—native execution and evaluation semantics—while retaining evalbuilder’s portable fixture authoring. A full native target adapter should follow only if the cross-framework UI/pipeline value justifies its maintenance cost.

## 7. Proposed architecture

```text
Dataset rules
    │
    ▼
Rule parser + validator ──► composition policy
    │
    ▼
MockEngine.resolve(tool, args, context)
    ├── match → Outcome(return / raise / sequence)
    ├── miss  → strict error / fallback / explicit real
    └── ledger(rule id, call id, timing, passthrough)
             │
             ├── LangChain BaseTool adapter
             ├── LangChain agent middleware adapter
             ├── LangGraph ToolNode/registry adapter
             └── Google ADK callback/plugin adapter
```

Suggested portable models:

```python
class MockRule(BaseModel):
    id: str
    match_args: dict[str, Any] = {}
    outcome: MockOutcome
    times: int | None = None

class MockPolicy(BaseModel):
    unmatched_call: Literal["error", "real", "fallback"] = "error"
    unconfigured_tool: Literal["error", "real"] = "error"
    composition: Literal["replace", "prepend_then_dataset"] = "prepend_then_dataset"

class MockCallRecord(BaseModel):
    tool_name: str
    args: dict[str, Any]
    matched_rule_id: str | None
    action: Literal["return", "raise", "real", "fallback", "error"]
```

Framework-specific state update outcomes should live in adapter extensions, not the portable base schema.

## 8. Implementation roadmap

### Phase 0: documentation and safety fixes

- Correct README claims about current ADK built-in mocking.
- State that modern LangChain `create_agent` uses LangGraph internally.
- Change mocked pipeline defaults to strict, or require an explicit real-passthrough flag.
- Record mock policy in run artifacts.
- Make verification report unconfigured expected tools and, under strict mode, all exposed tools without rules.

### Phase 1: harden the rule engine

- Introduce typed rules/outcomes/policies and stable rule IDs.
- Validate unreachable wildcards, duplicates, tool names, and response schemas.
- Deep-copy static responses.
- Add exception and sequence outcomes.
- Add per-run call ledger and unused-rule reporting.
- Test null/missing/numeric/list/nested cases and invalid data.

### Phase 2: harden the LangChain tool adapter

- Preserve sync and async behavior, `return_direct`, response format/artifacts, tags, metadata, and error settings.
- Test async-only tools, `ToolCall` inputs, injected runtime/context, callback propagation, config, artifacts, and custom subclasses.
- Prefer a delegating proxy or middleware over rebuilding a generic `StructuredTool` where possible.

### Phase 3: decouple target execution

- Add framework adapter protocol and dispatch on `Target.framework`.
- Move LangGraph stream/path extraction into a LangGraph adapter.
- Add a generic Runnable/LangChain adapter with configurable input/output normalization.
- Keep discovery and execution capability flags explicit.

### Phase 4: native interception adapters

- Implement LangChain `wrap_tool_call` middleware.
- Implement a mock-aware LangGraph `ToolNode`/registry option.
- Retain injected wrappers for existing examples and compatibility.
- Add end-to-end tests for parallel calls, unexpected calls, stateful tools, errors, and multi-turn cases.

### Phase 5: Google ADK bridge

- Pin a tested ADK version range.
- Implement current `EvalSet` export/import.
- Implement run-scoped async `before_tool_callback` plugin with ledger.
- Test multi-agent callbacks, toolsets, session state, artifacts, concurrency, and callback ordering.
- Document unsupported built-in/server-side tools.

### Phase 6: integration confidence

- Add fake-service contract tests behind real tools.
- Add a small opt-in staging suite with restricted credentials.
- Add compatibility tests across the supported LangChain/LangGraph/ADK version matrix.

## 9. Skill-specific changes recommended

The skill should remain short, but its next revision should:

- say “LangChain/LangGraph tools” rather than imply universal LangGraph coverage;
- define `--mock` as strict isolation only after the implementation enforces it;
- require explicit acknowledgement for real passthrough;
- distinguish returned error values from raised exceptions;
- warn that dynamic/provider-hosted tools may not be intercepted;
- require verification of all exposed side-effecting tools, not only expected calls;
- mention the run ledger and unused rules once implemented;
- remove or qualify “ADK-style” because current ADK no longer executes the historical field natively;
- describe case/dataset composition precisely (`replace` versus fallback append).

## 10. Final evaluation

The current feature is a strong prototype for deterministic, data-driven tests of conventional synchronous LangChain tools in rebuildable LangGraph agents. Its rule core is worth keeping and is readily portable. Its main weakness is not matching logic; it is the gap between the broad safety claim and the narrow interception boundary.

The best extension is not to accumulate framework conditionals in `wrap_tool`. Split matching, policy, outcomes, and auditing from interception. Then use each framework’s supported execution seam:

- injected tools for generic/manual LangChain loops and compatibility;
- LangChain middleware for current `create_agent`;
- `ToolNode`/registry replacement for custom LangGraph graphs;
- `before_tool_callback`/plugin for Google ADK.

For Google ADK, reusing native evaluation plus a callback adapter is preferable initially to duplicating its runner, session, and event semantics. The historical built-in mock-output behavior must not be treated as current functionality without pinning and verifying a specific older ADK release.

## Primary sources

- LangChain, [Agents](https://docs.langchain.com/oss/python/langchain/agents) — current `create_agent`, LangGraph runtime, tool middleware, dynamic tool handling.
- LangChain, [Models: tool calling](https://docs.langchain.com/oss/python/langchain/models) — application responsibility for executing tool calls outside an agent.
- LangChain, [Tools](https://docs.langchain.com/oss/python/langchain/tools) — runtime/state/store and stream behavior relevant to wrapper fidelity.
- LangGraph, [Workflows and agents / ToolNode](https://langchain-ai.github.io/langgraph/agents/tools/) — standard tool execution node, parallel execution, errors, and state injection.
- Google ADK, [Types of callbacks](https://google.github.io/adk-docs/callbacks/types-of-callbacks/) — before-tool callback interception semantics.
- Google ADK Python, [`LlmAgent.before_tool_callback`](https://github.com/google/adk-python/blob/ed1306f58796648a58b5ee2f99f80e81e92cd7de/src/google/adk/agents/llm_agent.py) — callback contract in the assessed source revision.
- Google ADK Python, [`functions.py`](https://github.com/google/adk-python/blob/ed1306f58796648a58b5ee2f99f80e81e92cd7de/src/google/adk/flows/llm_flows/functions.py) — plugin callback, agent callback, then real tool execution order.
- Google ADK Python, [`eval_case.py`](https://github.com/google/adk-python/blob/ed1306f58796648a58b5ee2f99f80e81e92cd7de/src/google/adk/evaluation/eval_case.py) — current eval conversation, expected tool-use/response, and invocation-event model.
- Google ADK Python, [`local_eval_sets_manager.py`](https://github.com/google/adk-python/blob/ed1306f58796648a58b5ee2f99f80e81e92cd7de/src/google/adk/evaluation/local_eval_sets_manager.py) — current legacy-format conversion behavior.
- Google ADK Python, [`evaluation_constants.py`](https://github.com/google/adk-python/blob/ed1306f58796648a58b5ee2f99f80e81e92cd7de/src/google/adk/evaluation/evaluation_constants.py) — vestigial `MOCK_TOOL_OUTPUT` constant.
