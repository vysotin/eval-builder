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
- `RunArtifact` (`evalbuilder/run/v1`) with `CaseRun` (`inputs` — what was asked: the
  case's `inputs` plus `user_turns` for a multi-turn case, so the run stands alone —
  `outputs`, `trajectory`, `tool_calls`, `node_path`, `error`,
  `error_class ∈ none|agent|infrastructure`, `mock_calls` — the mock ledger — and `log`,
  one entry per conversation turn: `{turn, seconds, tool_calls, error, mode, endpoint}`);
  the artifact carries `mocking`
  (policy, model, strategy, per-layer call totals) and `execution` — how the inference
  phase ran: `{mode: local|remote, endpoint, workers, backend, seconds}`.
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
   `listing` | `prompt`, or `loader` when the prompt names none but the node holds a
   skill-loader tool — it can read, so might use, any skill) plus `capabilities`
   lines (skill → reachable tools scoped to the node → result); each skill entry
   resolves its usable `tools`/`unknown_tools`, extracts `rules` and carries
   `instruction` (full body, or a deterministic summary above 1500 chars); `skill_loader_tool(...)` assignments and loader-named `@tool`
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
  `is_skill_loader(tool)` recognises that tag or a loader-like name;
- analysis helpers: `skill_tools(skill, tool_names)` resolves the tools a skill can use
  against the agent's tools (allowed-tools ∩ agent tools + body mentions, unknown
  allowed names reported), `skill_rules(body)` extracts the imperative lines,
  `summarize_skill(skill)` builds the deterministic summary used when the body exceeds
  `SKILL_INSTRUCTION_CHARS` (1500), and `capability_line(entry, disclosure, node_tools)`
  writes the per-node "skill → tools → result" line
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

## Target contract and the primitives (`target.py`)

The **target contract** is the only thing an agent repository must provide: a module
exposing `TOOLS` (a list of LangChain tools) and `build_agent(model=None, tools=None)`
returning a compiled graph. `build_graph` passes only the overrides it was given so a
factory's own defaults (e.g. a scripted model) still apply.

`invoke_messages(graph, messages)` streams the graph with `stream_mode=["updates",
"values"]` and returns the final state plus the node path; `extract(state)` converts the
messages to OpenAI-style dicts (`convert_to_openai_messages`) as the **trajectory**,
collects `tool_calls` from every `AIMessage`, and returns the last message's content as
the response; `is_mock_engine_error(exc)` recognises a `MockEngineError` in the cause
chain (an infrastructure failure, not the agent's). Multi-turn replay and the
`error_class` verdict live one layer up, in the agent clients and the inference engine.

## Agent clients (`agent_client.py`)

One `invoke(messages, mocks, mocked=True) -> InvokeResult` contract with two
implementations; both pickle, so joblib's process backend can hand them to workers:

- `InvokeResult` — `messages` (the full history after the turn, OpenAI format),
  `response`, `tool_calls` (of the whole history), `node_path` (this turn), `mock_calls`
  (this turn's ledger), `error`, `error_class`, `seconds`; `to_dict` / `from_dict` are
  the wire format.
- `LocalAgent(module, factory, agent_model_spec=, mock_model_spec=, agent_model=,
  mock_model=)` — imports the target lazily and builds a graph (and fresh tool wrappers)
  **per call**: rules from `mocks.tools`, the policy from `mocks.on_miss`, and under
  `llm` an `LLMMockEngine` from `mocks.strategies` / `mocks.strategy` / `mocks.llm`,
  seeded with `mocks.history` (the tool/args/answer triples of earlier turns) so a
  stateless server keeps a conversation consistent. The engine's model is the client's
  own (an object, or `mock_model_spec` — what the deployment configured) and the
  request's `llm.model` spec is only the fallback. Building the agent is an
  `infrastructure` error; a mock-engine failure is too; anything else is `agent`.
  `health()` imports the module, checks the factory exists and builds a configured mock
  model, so a broken target fails at start-up rather than on the first request.
  Objects built lazily are dropped on pickling and rebuilt from the specs in the worker.
- `RemoteAgent(endpoint, timeout)` — `POST /invoke` and `GET /health` over `urllib`;
  connection failures and non-200 answers come back as `infrastructure` results.
- `mocks_for_case(ds_mocks, case_meta, on_miss=, strategy=)` — the block one request
  carries: dataset rules with the case's overrides on top (`merge_mock_rules`), the
  policy (flag > dataset > `real`), the strategy (the case's own `metadata.mocks.strategy`
  always wins over the run-level one), `strategies` and `llm` when the dataset has them.
- `agent_for(endpoint=, module=)` — a remote client when an endpoint is given, else local.

**The conversation is stateless.** The caller sends the full history and gets the full
history back — tool calls and tool results included — so the next turn is `messages +
[user]`. `convert_to_openai_messages` / `convert_to_messages` round-trip exactly (tool
calls included), which is what makes replicas interchangeable and the server free of
any session store.

## The agent server (`serve.py`)

A stdlib `ThreadingHTTPServer` (`AgentServer`; no web framework in the image) with two
routes: `GET /health` → `LocalAgent.health()` + `server: evalbuilder/serve/v1` (500 with
the error when the target is broken), `POST /invoke` with `{messages, mocks?, mocked?}`
→ `InvokeResult.to_dict()` (400 for a missing body, invalid JSON, an empty `messages`
list or a non-object `mocks`; 404 elsewhere). `make_server(agent, host, port=0)` binds
(port 0 = any free port; `.endpoint`, `.serve_in_thread()` for tests) and `serve(module,
factory, host, port, agent_model, mock_model)` is the blocking entry point of
`evalbuilder serve`, the agent image's `CMD` and the `local` target; it calls `health()`
before listening so a bad module, factory or mock model exits at start-up (the local
target notices the dead process instead of waiting for the readiness timeout).

## The inference engine (`inference.py`)

Every unit of work is one conversation — a case (its first message plus every
`metadata.user_turns` entry) or a scenario — run through `agent.invoke` turn by turn,
so it needs nothing but a picklable client and plain dicts:

- `parallel(workers, backend)` → `joblib.Parallel(n_jobs, prefer="threads"|"processes",
  return_as="generator")`: results arrive in submission order, which keeps progress
  reporting and artifact order trivial for both backends (`n_jobs=1` runs inline).
- `run_conversation(agent, opening, user_turns, mocks)` — replays the turns, carries
  `mocks.history` from the accumulated ledger, records one `log` entry per turn and
  stops at the first turn that errors; returns the pieces of a `CaseRun`.
- `infer_cases(agent, ds, cases, workers=, backend=, on_miss=, strategy=, progress=)` —
  one task per case, `mocks_for_case` per case, a failed task is a `CaseRun` with an
  error (never an exception), `progress` is called in the parent after every case with
  `{case_id, intent, error_class, error, completed, total}`.
- `select_cases(ds, ids)` (approved only; explicit ids must be approved) and
  `infer_dataset(ds, dataset_path, agent, out_dir=, …)` → `run-<id>.json` with `mocking`
  totals and the `execution` block.
- `simulate_scenarios(agent, scenarios, mocks=, user_model_spec=, user_model=, workers=,
  backend=, …)` — `simulate.simulate_scenario` stepping through the client; a scenario's
  `mock_strategy` selects the engine strategy; results carry `mock_calls`, `mock_strategy`
  and a per-turn `log`; an agent error becomes a result with `stop_reason: error`. The
  simulated user is passed as a spec so process workers build their own (a model object
  forces the threads backend).
- `local_agent_for(ds, model=, model_spec=, mock_model=, mock_model_spec=)` and
  `summarize_mock_calls(results)` are the small helpers the CLI and the stages share.

**Reasoning.** `threads` is the default because the work is waiting on the agent (HTTP
or model calls); `processes` (loky) exists for CPU-bound in-process agents, and the
clients rebuild their models from specs inside the worker. Intent-group parallelism went
away with the stateless server: every case is independent, so the case is the unit.

## The runner (`runner.py`) and simulation (`simulate.py`)

`run_dataset(ds, dataset_path, mocked=, ids=, out_dir=, model=, on_miss=, fallback=,
max_workers=, workers=, backend=, progress=, mock_model=, mock_model_spec=, strategy=)`
is the in-process convenience the interactive CLI and the tests use: it wraps the
target in a `LocalAgent` (objects win over specs) and hands the work to
`infer_dataset`; `max_workers` is the older name of `workers`. `tool_specs_of(module)`
and `build_engine(model, ds, specs, strategy)` are shared with the `mock` commands.

Scenarios (YAML) have `id`, `persona`, `goal`, `opening`, `followups[]`, `max_turns`
and stop conditions (`success_contains`, `expect.contains`, `expect.not_contains`).
`simulate_scenario(step, scenario, user_model=)` drives one scenario through a **step**
— a callable taking the text transcript so far and returning the assistant's reply — with
scripted follow-ups first, then a simulated user prompted with persona, goal and the
last six transcript entries. A compiled graph is accepted too (`graph_step` adapts it).
Stop reasons: `success` (needle seen), `exhausted` (no more user messages), `max_turns`
(a truncation, reported as a violation when a success needle was expected).
`mine_failures` turns violating runs into **pending** cases (`source=simulation`,
`failure_mode=simulation-violation`, later turns in `user_turns`, transcript tail kept).

**Limitations.** Multi-turn replay re-invokes the graph with the accumulated message
list — graphs relying on a checkpointer thread id for state get a fresh state each
turn; `outputs.response` is the last message's text (structured/state outputs beyond
`messages` are not captured); the per-request timeout is the HTTP client's
(`inference.timeout`), there is none for an in-process call; the graph must accept
`{"messages": [...]}` as input; the simulated user is a plain prompt, expectations are
substring checks, and each scenario turn re-sends the plain role/content transcript
(tool-call history is not replayed).

## Deployment targets (`deploy/`)

`spec.py` — `DeploymentSpec` (what to deploy: name, target, module/factory, image
reference, `push`, registry, extras, build context / Dockerfile / includes /
requirements, port, namespace, replicas, `expose`, the resolved container `env`, keep,
timeout, kube context, the agent and mock model specs, `work_dir`) built by
`spec_from_config(cfg, out_dir, target=)` — which also returns the `deploy.env` names
requested from the host but unset — and `DeploymentRecord`, the `deployment.json`
artifact (`evalbuilder/deployment/v1`: target, image, endpoint, expose, `status ∈
pending|up|failed|down`, per-target `resources`, timestamps, the `commands` executed,
`details`, the spec). `resource_name` makes DNS-1123 names (`evalbuilder-<name>`).

`runner.py` — `CommandRunner`: `run(argv, input=, check=, timeout=, env=, cwd=)`,
`pipe(producer, consumer)` (a real pipe, so an image tarball never sits in memory),
`spawn(argv, log_path=)` (a detached process in its own session; returns the pid),
`which`, `pid_alive` (reaps zombies first), `kill` (SIGTERM the group, then SIGKILL).
`run` streams: stdout and stderr are merged and read line by line as the command runs,
each line forwarded to the runner's `log` as `  | …` (so a 1–3 minute `docker build`
shows progress instead of looking hung, the pipeline's stderr logger being that `log`),
with the last 200 lines kept as `CommandResult.stdout` — `stderr` then carries only the
runner's own diagnostics (`timed out after Ns` with code 124 after the process is
killed, the OS error with code 127 when the binary is missing). Every call is appended
to `history` (what the record embeds) and to `work/deploy/commands.log`; a non-zero exit
raises `CommandError` with the output tail. Targets never import `subprocess`; tests
inject a fake runner.

`render.py` — the generated files: `render_dockerfile` (`python:3.12-slim`, copies
`pyproject.toml`, `README.md`, `src/`, the target's top-level package (`package_dir`)
and `build.include`, `pip install ".[extras]"` plus `build.requirements`, a `HEALTHCHECK`
on `/health`, `CMD evalbuilder serve --module … --factory … --host 0.0.0.0 --port …`;
model specs travel as environment, not as flags), `render_compose` (one service with
`build`, `image`, `ports: [port:port]`, `environment` from `server_env` — the user's
variables plus `EVALBUILDER_AGENT_MODEL` / `EVALBUILDER_MOCK_MODEL` — and a healthcheck),
`render_manifests` (a Deployment with readiness / liveness probes on `/health` and the
given `imagePullPolicy`, a ClusterIP or NodePort Service, and a Route under `expose:
route`), `render_loader` (the privileged, `hostPID` `busybox` DaemonSet
`<resource>-image-loader`).

`base.py` — the interface every target implements: `available() -> (ok, reason)`,
`render(spec) -> {file: content}`, `build(spec) -> image`, `up(spec) -> record`,
`status(spec, record) -> {ready, endpoint, health, replicas, details}`, `logs`, `down`;
plus `write_files` (render into `work/deploy/`), `wait_healthy(endpoint, timeout,
alive=, diagnose=)` (poll `/health`; fail at once when the process behind the endpoint
is gone, with its last log lines), `finish` (store the health facts and the command
history on the record) and `failed`. The health probe and `sleep` are injectable.

The targets (`deploy/__init__.py::TARGETS`):

- **local** (`local.py`) — `spawn(python -m evalbuilder.cli serve --module M --factory
  F --host 127.0.0.1 --port <free port>)` with the container environment, cwd = the
  project root, output in `work/deploy/serve.log`; `status` = pid alive + `/health`;
  `down` kills the process group. No Docker.
- **docker** (`docker_compose.py`) — `available` = `docker info` + `docker compose
  version`; `up` = write Dockerfile + `compose.yaml`, `docker compose -p <resource> -f
  … up -d --build --wait --wait-timeout <timeout>`, endpoint `http://127.0.0.1:<port>`;
  `status` parses `compose ps --format json`; `logs` = `compose logs --tail`; `down` =
  `compose down --remove-orphans`.
- **kubernetes** (`kubernetes.py`) — `available` = `kubectl get nodes` (+ `docker` for
  the build); `up` = `docker build`, then image distribution by `push_mode`: `registry`
  (`docker push`, pull policy `Always`), `load` (apply the loader DaemonSet, wait for its
  rollout, list its pods, and for each one `docker save IMAGE | kubectl exec -i POD --
  nsenter -t 1 -m -u -i -n -- ctr -n k8s.io images import -`; pull policy `Never`) or
  `none` (`IfNotPresent`; `auto` = `registry` when a registry is configured, else `load`),
  then `kubectl apply -f manifests.yaml`, `kubectl rollout status deployment/…
  --timeout`, and the endpoint: `port-forward` (a detached `kubectl port-forward
  svc/<resource> <free local port>:<port> --address 127.0.0.1`, pid + port recorded) or
  `nodeport` (`http://<node InternalIP>:<nodePort>`). `status` reads the Deployment JSON
  (`readyReplicas`) and **restarts a dead port-forward** (`ensure_endpoint`), so a
  later `infer` finds a live tunnel; `down` kills the tunnel, `kubectl delete -f
  manifests.yaml`, and deletes the loader DaemonSet. Every command honours
  `deploy.context` (`--context`) and `deploy.namespace` (`-n`).
- **openshift** (`openshift.py`, prototype) — the Kubernetes target with `cli = "oc"`,
  `available` = `oc whoami`, `push` defaulting to `registry` (and refusing to run
  without `deploy.image.registry`), `expose: route` adding a Route object and reading
  the endpoint from `oc get route … -o jsonpath={.spec.host}` (`https` when the route
  has TLS termination). Exercised against a fake `oc` only.

`deploy/__init__.py` — `target_for(kind, runner=, log=)`, `record_path` / `load_record`
(the `deployment` artifact kind), `default_runner` (logging to `work/deploy/commands.log`),
and the entry points the CLI, the stages and the UI share: `deploy_render`,
`deploy_build`, `deploy_up` (checks `available()`, reuses a recorded deployment that is
still up and healthy — otherwise tears it down first — writes the failed record before
re-raising), `deploy_status` (live facts, saved back onto the record), `deploy_logs`,
`deploy_down` (marks the record `down` and clears the endpoint). They accept a
`PipelineConfig` or an output directory (the spec is rebuilt from the record).

**Reasoning.** Rendering everything to disk before use and logging every command keeps
the deployment inspectable (`evalbuilder deploy render`, `work/deploy/commands.log`,
the record's `commands`). The `load` distribution and the port-forward default come
from what was verified on Docker Desktop's kind-based cluster: locally built images are
not visible to the nodes, and NodePort / LoadBalancer services are not reachable from
the host — streaming `docker save` into each node's containerd through a privileged
pod is what `kind load` does, only through the Kubernetes API, and a tunnel works on
every cluster. A registry push is one config key away and is what OpenShift uses.

**Limitations.** See [07-limitations.md](07-limitations.md#deployment): the generated
Dockerfile assumes the evalbuilder checkout as build context, `load` needs privileged
pods, a port-forward is one tunnel, the docker target binds the container port on the
host, and `claude-cli` models cannot run inside the image.

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
unknown id — by design, and visible in the ledger), and with a stateless agent server
it is carried in the request (`mocks.history`, rebuilt from the ledger of earlier
turns) rather than held by one engine object; prompt size grows with the strategies
document and the tool schemas.

## Evaluators (`evaluators.py`)

One metric per evaluator; each is `fn(case, case_run) -> {key, score, comment}` or
raises `EvaluatorNotApplicable` (case has no reference → `skipped`) /
`EvaluatorUnavailable` (judge missing or failed → `error`). `score_run` scores every
case of a run, records per-case `scores`/`errors`/`skipped`, per-metric
`{n, avg, min, max, errors, skipped}`, and per-slice means for `intent`,
`failure_mode`, `variant`. A case that errored during the run scores 0 on every
metric with the error as comment. `score_rows(run, ds, specs, judge_model, case_ids)`
is the unit of work — it builds the evaluators itself, so a process worker can run it —
and `score_run(…, workers=, backend=)` splits the case runs into contiguous chunks
(`_chunks`), scores them with `joblib.Parallel` (`threads`, or loky `processes`; each
worker rebuilds the evaluators from the specs), and assembles rows, metrics and slices
in run order afterwards, so the report is identical to a sequential pass. `max_workers`
is the older name of `workers`.

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
