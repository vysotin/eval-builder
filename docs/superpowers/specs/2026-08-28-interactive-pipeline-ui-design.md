# Interactive pipeline UI, schema-aware edge cases, two subagent examples — design

Date: 2026-08-28. Goal (set by the user): (1) a UI page that initialises a pipeline
config interactively (editable, with free-text general rules and constraints), then
either runs the whole pipeline autonomously and shows the result, or runs only until
the dataset + tool mocks are generated so the user can add comments / new
instructions on the config page and rerun the generation steps or proceed with
evaluation; (2) two more example target agents with several subagents and multiple
tools calling external services (mocked), some tools using pydantic models for
input/output and some `args_schema`, with those schemas captured in the agent map and
used for tool mocking and for generating edge cases (wrong / missing / non-compliant
inputs); (3) Playwright functional tests covering every interactive UI flow with
result validation; (4) the pipeline's internal LLM calls go through the claude-cli
subscription adapter with Sonnet 5.

Decisions below were taken autonomously (the session runs under a `/goal`); they
follow the existing conventions (spec 2026-08-27 pipeline, spec 2026-08-28 UI).

## 1. Config additions (`pipeline/config.py`)

| key | type / default | purpose |
|---|---|---|
| `instructions` | `str = ""` | user-defined general rules and constraints as free text; reaches every generator prompt as a `USER INSTRUCTIONS` block (map, mocks, cases, self-review, simulation scenarios) |
| `feedback` | `list[{at, note, from_stage}] = []` | reviewer comments added after a partial run; each entry is appended (never overwritten) and reaches the same prompts as `REVIEWER FEEDBACK` in chronological order |
| `coverage.per_tool_edge_cases` | `int = 2` | schema-derived edge cases per tool (0 disables) |
| `models.*` defaults | `claude-cli:claude-sonnet-5` | explicit Sonnet 5 through the subscription CLI (`claude-cli:sonnet` still accepted) |

`PipelineConfig.to_yaml()` renders a config back to YAML with comments for the top-level
sections so the UI can show/edit the same text the CLI accepts.

## 2. Pipeline control (`pipeline/engine.py`, `report.py`, CLI)

- `PipelineRunner.stop_after: str | None` — after that stage completes, every later
  non-`always` stage is `skipped` with reason `stopped after <stage> (--until)`;
  `report` still runs. Verdict becomes `incomplete` (required stages missing) —
  correct: nothing was evaluated.
- `run_pipeline(..., until=...)` and `evalbuilder pipeline run CFG --until STAGE`.
  The UI's "generate dataset only" is `--until dataset`; "regenerate" is
  `--resume --from map|mocks|dataset`; "proceed with evaluation" is `--resume`.
- `review` stage: when no case is pending but approved cases exist, the stage
  succeeds with `details.already_reviewed = true` (cases approved through
  `evalbuilder review` or the UI are not re-reviewed).
- Jobs (`pipeline/jobs.py`, pure Python): `start_job(config_path, mode, out_dir)` spawns
  `evalbuilder pipeline run …` as a detached subprocess, logging to
  `<out_dir>/pipeline.log`, and writes `<out_dir>/job.json` (`schema
  evalbuilder/pipeline-job/v1`: pid, argv, mode, started_at, finished_at, exit_code).
  `job_status(out_dir)` merges job.json, liveness of the pid, and `state.json` stage
  statuses. The registry gets the new kind `pipeline_job` (`job.json`).

## 3. Tool schemas in the agent map (`discover.py`, `stages.discover`, `tool_schemas.py`)

Each `tools[]` entry gains:

- `args_schema` — already present; live introspection uses `tool_call_schema` so
  pydantic `args_schema=` classes and nested models (with `$defs`) come through.
- `output_schema` — JSON schema of the return annotation (pydantic model, TypedDict,
  dataclass or primitive) resolved live from `tool.func`; the AST pass records
  `-> Name` and resolves `Name` when it is a `BaseModel` subclass defined in the source
  (fields, types, `Literal` enums, defaults, `Field(description=, ge=, le=, min_length=,
  max_length=, pattern=)`).
- `schema_source` — `args_schema` (explicit class), `annotations` (inferred from the
  signature) or `ast` (no live import).
- `side_effecting` — true when the docstring says so (`side-effect`, `only call after`).
- `edge_cases` — the deterministic list derived from the schemas (below).

`tool_schemas.py` (no LLM):

- `resolve_output_schema(tool)` / `resolve_args_schema(tool)` for live tools.
- `validate(value, schema) -> list[str]` — a JSON-schema subset (type, properties,
  required, additionalProperties, items, enum, const, minimum/maximum/exclusive*,
  minLength/maxLength, pattern, anyOf/oneOf, `$ref` → `$defs`, nullable via `type:
  [..,"null"]`).
- `example(schema)` — a conformant sample (used when a generated fixture is invalid or
  missing) and `corrupt(schema, kind)` — a non-conformant sample for `malformed_output`
  edges (drops a required field / wrong type).
- `edge_cases(tool) -> list[EdgeCase]` where `EdgeCase = {id, tool, kind, field,
  detail, failure_mode, expected_behavior, evidence}` and `kind ∈ missing_required,
  wrong_type, out_of_enum, boundary, malformed_output`. Input kinds map to
  `failure_mode: input_validation`, `malformed_output` to `tool_error_handling`.
  Boundary uses numeric/length limits when declared; `out_of_enum` needs an enum /
  Literal; `malformed_output` needs an output schema with at least one required field.

## 4. Schema-aware mocking and edge-case generation (`generator.py`, `planning.py`)

- `author_mocks` shows `args_schema` + `output_schema` per tool and instructs the
  generator that responses must conform; every response is validated with
  `tool_schemas.validate`; a non-conforming default is replaced with
  `example(output_schema)` and reported in `problems`; non-conforming variants are
  dropped and reported.
- Planning adds cells of kind `schema-edge` (`intent: cross-cutting`, `scenario:
  edge.<tool>.<kind>[.<field>]`, `failure_mode` from the edge, `tool`, `edge`) —
  `per_tool_edge_cases` edges per tool rotating over the tool's edge list; each cell
  counts 1. `Cell` gains `tool` and `edge`; `coverage-plan.json` carries them; coverage
  `by_kind` shows `schema-edge`.
- `_cell_prompt` includes the edge (tool, kind, field, expected behavior) and
  `CASES_SYSTEM` explains schema-edge cells: the user message must omit / mistype /
  overflow the named field, or (malformed_output) is a normal request whose fixture
  is corrupted; the contract states the graceful behaviour (ask for the field, refuse
  the out-of-range value, never fabricate, explain the broken tool result).
- For `malformed_output` cells the pipeline injects the per-case mock override
  itself (`corrupt(output_schema, "missing_required")`), so the LLM cannot skip it.
- `expected_tools[].args` in generated cases are validated against `args_schema`
  (type-level); mismatches become `problems` (not rejections).
- Guidance block: `user_guidance(cfg)` renders `instructions` + `feedback` and is
  appended to `agent_brief` context in every authoring prompt.

## 5. Examples (`examples/incident_desk`, `examples/loan_desk`)

Both follow the target contract (`TOOLS`, `build_agent(model=None, tools=None)`),
call `*.example.invalid` HTTP endpoints (fail fast, JSON error payloads), ship a
scripted default model (offline), a `pipeline.yaml`, and an `offline.py` with
`generator_model()` — a `SchemaScriptedModel` (new in `testing.py`: dispatches
`with_structured_output(schema)` by `schema["title"]`, plan-aware for `cases`) so the
whole pipeline runs offline from the UI via `models.generator: scripted:…`.

- **incident_desk** — on-call copilot. Graph: `classify` (structured route) →
  `triage_agent` → `remediation_agent` → `comms_agent` → END, with `decline` for
  out-of-scope. Tools: `get_service_status(service) -> ServiceStatus` (pydantic
  output), `search_runbooks(args_schema=RunbookQuery{query, severity: Literal,
  limit: int 1..10})`, `create_ticket(ticket: TicketRequest) -> TicketReceipt`
  (pydantic in and out), `page_oncall(team, severity)` (side-effecting),
  `post_status_update(update: StatusUpdate)` (side-effecting, pydantic input).
- **loan_desk** — lending assistant. Graph: `route` → `eligibility_agent` |
  `documents_agent` | `advisor_agent`, with `documents_agent → advisor_agent`.
  Tools: `get_customer_profile(customer_id) -> CustomerProfile`,
  `credit_check(args_schema=CreditQuery{customer_id, consent: bool, ssn_last4:
  pattern})`, `quote_installment(terms: LoanTerms{amount ge 1000 le 500000, months
  Literal, rate}) -> InstallmentQuote`, `document_status(doc_id)`,
  `submit_application(application: LoanApplication) -> ApplicationReceipt`
  (side-effecting).

Tests per example: routing, tool gating on confirmation/consent, injected tools,
discovery captures pydantic schemas (`args_schema` with `$defs`, `output_schema`,
`schema_source`) and edge cases, and an offline pipeline e2e through the scripted
generator with verdict `pass`.

## 6. UI (`ui/app_pages/setup.py`, `ui/app_pages/run.py`, `app.py`)

Navigation gains a **Pipeline** group: *Setup* and *Run & review*.

**Setup** (`h2#setup`):
1. Target — pick a discovered example (`examples/*/agent.py`) or type source/module;
   *Discover* runs the AST + live introspection (no LLM) and shows nodes, tools with
   arg/output schemas and derived edge cases.
2. Config form — name, models (defaults Sonnet 5 via claude-cli), constraints (one per
   line), **instructions** (free text), coverage numbers incl. per-tool edge cases,
   evaluators (multiselect), thresholds, repeats, mocking policy, review approval
   (checkbox + approver), output dir. *Generate YAML* fills the editor.
3. YAML editor — editable text; *Validate* shows schema/semantic problems; *Save*
   writes `eval/pipeline/<name>.yaml` (path editable).
4. Actions — *Run full pipeline* / *Generate dataset & mocks only* start a job and
   switch to Run & review.

**Run & review** (`h2#run`): job status (running/finished/exit code), stage table with
live statuses (fragment `run_every=2s` while running), log tail, then:
- *Review & iterate* (after a `--until dataset` job): dataset/mocks summary (cases by
  kind incl. schema-edge, tools mocked, problems), a feedback text area → *Save
  feedback* appends to `config.feedback`; *Regenerate from* `map | mocks | dataset`
  reruns with `--resume --from`; *Proceed with evaluation* needs an approver name,
  optionally rejects selected case ids (`evalbuilder review --reject`), sets
  `review.auto_approve` + `approved_by` in the config, and resumes.
- *Results* — once the job finished, "Open results" loads the output dir into the
  sidebar bundle so every report page shows it.

Widgets carry stable `key`s and headings carry anchors for Playwright.

## 7. Tests

- Unit: config additions + YAML round-trip; engine `stop_after`; jobs (spawn a
  trivial command, status transitions); `tool_schemas` (validate subset, example,
  corrupt, edge-case derivation on pydantic tools); discover AST pydantic resolution;
  planning schema-edge cells; generator mock validation + malformed_output injection +
  guidance block; review already-approved path; examples (above).
- AppTest: setup page generates/validates/saves YAML; run page renders job states.
- Playwright (`tests/ui/test_playwright_flows.py`, marker `ui`): against a temp copy of
  the repo examples with offline models — (a) setup → discover → schemas visible →
  generate YAML contains instructions → save → validation error shown for a bad
  threshold; (b) generate-dataset-only job → stages table shows dataset ok / review
  skipped → dataset page shows schema-edge cases and mock rules; (c) feedback → save
  → regenerate from dataset → config file holds the feedback, state shows dataset
  rerun; (d) proceed with evaluation → verdict pass on Overview, results/stability
  pages populated; (e) full autonomous run from setup → verdict on Overview.
- Live: one real `claude-cli:claude-sonnet-5` run of `incident_desk` (small coverage)
  committed under `docs/examples/incident-desk/` for the UI demo.

## 8. Docs

README (pipeline control: `--until`, feedback loop, UI setup/run pages, schema edge
cases, new examples), config reference, pipeline skill, memory.

## 9. Implementation notes (deviations found while building)

- Jobs run through a wrapper entry point (`python -m evalbuilder.pipeline.jobs run
  job.json`) that finalises `job.json` itself; `evalbuilder.cli` gained a `__main__`
  guard so `python -m evalbuilder.cli` works from the wrapper.
- `mocking.match_rule` / `evaluators` subset matching became recursive for nested
  (pydantic-model) arguments, and matcher validation (`matchArgs`, `expected_tools[].args`)
  uses `validate(partial=True)` — the live run showed the generator writes partial nested
  matchers, which the strict validator rejected.
- `@tool(handle_tool_error=True)` is not accepted by the installed langchain_core; the
  examples set `handle_tool_error = True` on the tool object. `loan_desk.document_status`
  gained an optional `Literal` `doc_type` so a top-level enum edge exists.
- Playwright: Streamlit 1.62 renders select widgets with react-aria (`[role='option']`),
  `st.dataframe` on canvas (no DOM text), and the status widget can appear late after a
  click; the flow suite polls `job.json` on disk before trusting the page.
- `docs/examples/incident-desk/` holds the committed live run (Sonnet 5 via claude-cli).
