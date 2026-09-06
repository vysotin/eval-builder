# Architectural decision record

Each entry: the decision, the alternatives that were considered, the reasoning, and the
consequences to keep in mind. Dates point at the spec where the decision was first
made (`docs/superpowers/specs/`).

## 1. Skills = policy prose, CLI = mechanism (2026-08-23)

**Decision.** Claude Code skills contain workflow, prohibitions and CLI invocations
only; a Python package owns schemas, IDs, validation, coverage math, the review state
machine, mocking, execution, scoring and LangSmith I/O. Skills never write artifact JSON.

**Alternatives.** (A) skills only, the model writes JSON ad hoc — rejected: artifacts
drift, validation is skipped, approval gates are bypassable. (C) port a full eval
framework with its own LLM generation — rejected: too large, and the model inside a
skill generates better than a fixed prompt library.

**Consequences.** Every mutation is a validated command; `tests/test_skills_lint.py`
keeps prose and code in sync; the same library serves the pipeline and the UI.

## 2. Hypotheses with cited evidence (2026-08-23)

**Decision.** Everything derived from code (intents, scenarios, failure types, edge
cases) is labelled a hypothesis and cites evidence tokens; topics are never invented.

**Reasoning.** Borrowed from the Dify eval design: it makes the model's claims
checkable line by line, and it stops synthetic datasets from covering behaviour the
agent does not have. **Consequence:** the `map` stage of the pipeline drops entries
without evidence and lists them in `problems`.

## 3. Structurally gated failure taxonomy (2026-08-23, code in `pipeline/taxonomy.py`)

**Decision.** A failure type may be proposed only when its precondition holds in the
graph (`branch_misrouting` needs conditional edges, `retrieval_grounding` needs a
retriever, …); `input_validation`, `provider_error`, `out_of_scope` always apply.

**Reasoning.** The commonest defect of generated evals is failure cases that cannot
fail for the claimed reason. Gating is cheap and mechanical. **Consequence:** an agent
without tools gets no `tool_*` cases; extend the gate when adding new structure
detection.

## 4. Review state machine with an explicit human act (2026-08-23)

**Decision.** Generated, imported and regenerated cases always land `pending`;
`evalbuilder review` is the only writer of `approved`/`rejected`; the pipeline
approves only with `review.auto_approve` + `approved_by` in the config; the UI records
an approver name. Rejected cases are kept.

**Reasoning.** An eval a model approved for itself is not an eval. Recording *who*
authorised keeps the audit trail. **Consequence:** an unattended pipeline stops at
`awaiting_review` unless the config says otherwise — by design, not a bug.

## 5. Content-hash case IDs over the coverage cell (2026-08-23)

**Decision.** `case-<sha256[:10]>` over `(inputs, intent, topic, scenario,
failure_mode)`.

**Reasoning.** Same input in a different cell is a different test; duplicates are
detected; ids are stable across regeneration so review decisions and LangSmith example
ids stay attached. **Consequence:** editing a case's input changes its id (and resets
its review — intended).

## 6. ADK-style tool mocking installed by wrapping tools (2026-08-23)

**Decision.** Ordered per-tool rules, subset `matchArgs` (recursive since 2026-08-28),
`{}` wildcard, miss policy `real | fallback | strict`; installed by wrapping tools with
the same name/description/args schema and rebuilding the graph per case.

**Alternatives.** Patching graph nodes or replacing the tool node — rejected for the
MVP: LangGraph graphs are compiled and heterogeneous; the tool surface is the stable
seam, and sub-agents exposed as tools are covered for free. **Consequence:** direct
external calls from nodes are not mocked (see limitations); arg-dependent responses
are several rules.

## 7. Two phases, one run artifact (2026-08-23)

**Decision.** `run` writes outputs/trajectory/tool calls/node path; `score` reads it.

**Reasoning.** Agent execution is the expensive, non-deterministic part; re-scoring
with new evaluators or a different judge must not re-run it. **Consequence:** score
reports refer to a run id; the pipeline keeps `(run, score-report)` pairs.

## 8. One metric per evaluator; deterministic before judges (2026-08-23)

**Decision.** Infrastructure, topology (`expected_tools`), parsing (`json_valid`) and
semantics (judges) are separate metrics; evaluator errors and skips are recorded, never
scored as agent failures.

**Reasoning.** A failure must localise itself; a missing API key must not look like a
regression. **Consequence:** thresholds are per metric and reports show error counts.

## 9. `provider:model[@effort]` everywhere, Claude CLI as a first-class provider (2026-08-27/28)

**Decision.** Agent, generator and judges all take a spec string; `claude-cli:` runs
the Claude Code binary (subscription auth) through an isolated subprocess transport;
`anthropic:`/`openai:`/`gemini:` go through `init_chat_model`; `scripted:` is offline.
Default `claude-cli:claude-sonnet-5`.

**Alternatives.** The packaged `claude-code-sdk` transport — rejected: it crashed on
the current CLI's event stream and parsed whole replies as JSON; `--bare` — rejected:
breaks subscription auth. **Consequence:** slower calls (seconds each), but any model
can be swapped per role; readiness is checked before any network call.

## 10. Stage engine with persisted state, never raising (2026-08-27)

**Decision.** Every stage records `ok | recovered | failed | skipped | awaiting_review`;
dependents of a failed stage are skipped with the reason; one retry with a recovery
note; `report` always runs; `--resume`, `--resume --from`, `--until`.

**Reasoning.** An unattended run must always leave a readable report saying exactly
what happened; resuming must re-run only what a fix invalidates. **Consequence:**
`problems` persist across resumes (history), so earlier warnings remain visible.

## 11. Generator output validated by the CLI's own code, one repair, drop and report (2026-08-27)

**Decision.** Structured output → `add_case` / `validate_dataset` / taxonomy gating /
schema validation; invalid entries are fed back once, then dropped and listed.

**Reasoning.** Validation logic must not be duplicated between the interactive and
autonomous paths, and silent thinning would make coverage numbers lie.
**Consequence:** `problems` can be long on a weak model; read them.

## 12. Repeats + trajectory hashing for stability (2026-08-27)

**Decision.** Instability is classified by comparing the tool-call trajectory across
repeats; text drift is only counted; judge flips on a stable trajectory are evaluator
instability.

**Alternatives.** Statistical tests over many repeats — rejected as too costly for the
default of 2–3 repeats; single-run scoring — rejected because it cannot separate agent
variance from judge variance. **Consequence:** `repeats ≥ 2` is needed for any
stability signal.

## 13. One artifact naming registry with embedded schema ids (2026-08-28)

**Decision.** `layout.ARTIFACTS` maps kind → file → schema id; every JSON artifact
embeds its schema id; legacy names/ids stay loadable.

**Reasoning.** Stages, the report, the CLI, the UI loader and the docs all read one
table; uploads can be identified by content. **Consequence:** a new artifact starts
with a registry row.

## 14. Tool schemas captured by code, edge cases derived by code (2026-08-28)

**Decision.** `tool_schemas.describe_tool` records args/output schemas, models, side
effects and deterministic edge cases; the planner turns them into `schema-edge`
cells; malformed-output fixtures are injected by the pipeline, not written by the LLM.

**Alternatives.** Asking the generator to invent edge cases from the tool
descriptions — rejected: it forgets fields, invents constraints, and cannot be
audited. **Consequence:** edge cases exist only for tools with schemas; nested-model
constraints are not expanded (limitation).

## 15. Free-text instructions and appended reviewer feedback (2026-08-28)

**Decision.** `instructions` and `feedback[]` in the config reach every generator
prompt; feedback is appended with a timestamp and the stage to rerun from.

**Reasoning.** The user's domain knowledge and review comments are the highest-value
input to generation, and keeping them in the config (not in the UI session) makes the
regeneration reproducible from the CLI. **Consequence:** the config is the single
place to look for "why does the dataset look like this".

## 16. UI runs the pipeline as a detached job with a wrapper (2026-08-28)

**Decision.** `pipeline/jobs.py` spawns `python -m evalbuilder.pipeline.jobs run
job.json`, which runs the CLI and finalises `work/job.json`; the page polls files.

**Alternatives.** Threads inside Streamlit — rejected: reruns and reloads would lose
the handle and the exit code, and a crashed page would orphan the run.
**Consequence:** jobs survive the server; status is read from disk by any tab.

## 17. Offline everything: scripted agents and schema-scripted generators (2026-08-23/28)

**Decision.** Every example agent ships a `default_scripted_model()`; the pipeline
examples ship an `offline.py` generator (`SchemaScriptedModel`); the test suite and
the browser flows never need a model.

**Reasoning.** Seconds-long deterministic runs make end-to-end testing of skills,
pipeline and UI possible in CI, and they double as demos. **Consequence:** the
scripted models must be kept consistent with fixtures and edge-case messages when an
example changes.

## 18. One project per UI session, chosen only on Pipeline setup (2026-08-30)

**Decision.** `ui/project.py` holds the single choice every page depends on — an
output folder or uploaded artifacts — in `session_state["project"]`. Only the setup
page (and the `--dir` start-up argument) sets it; the sidebar shows it and can clear
it; *Run & review* and every report page derive what they show from it, and without
a project they are empty and link back to setup. In folder mode the project *is* the
setup form's output directory: picking a target proposes `eval/pipeline/<name>`,
opening an existing folder loads its config (job → report → sibling YAML) back into
the form, and saving/running follows the YAML.

**Alternatives.** Independent pickers per page (the previous sidebar source picker +
a run-page directory selectbox + the setup form's output directory) — rejected: three
selectors could disagree, and "which folder am I looking at" had no single answer.

**Consequence.** The setup form must survive navigation, which Streamlit does not do
for keyed widgets (their values are dropped when the widget is not rendered): the
form is a plain dict, widgets are seeded from it before creation and captured back at
the start of every run, buttons act through callbacks, and the config file's mtime is
tracked so changes made by the review page or a job are reloaded instead of being
overwritten on the next save. The *Overview* page became *Summary* and moved after
*Analysis*: it summarises a finished evaluation rather than introducing the app, whose
entry point is now the setup page.


## 19. Agent Skills are first-class entries of the agent map (2026-09-01)

**Decision.** When the target agent uses Agent Skills (`SKILL.md` folders embedded in
prompts or read on demand through a loader tool), discovery loads them from disk,
renders the composed prompts so the recorded prompt is what the model sees, records
`skills[]` (instructions, references, allowed tools, the nodes that use them) and marks
loader tools `kind: skill_loader`, `mockable: false`. Downstream, the generator sees the
skills, every scenario lists the skills it exercises (`skill:<name>` evidence),
`skill_misuse` is a structurally gated failure type, coverage counts cases per skill,
and the loader is never mocked. The examples show both disclosure styles
(`incident_desk` inline, `support_bot` on demand).

**Alternatives.** Treating skill text as ordinary prompt text — rejected: the generator
could not tell which rules come from a skill (so it could not test skill misuse or
count coverage per skill), and a `load_skill` tool would have been mocked like a
backend call, feeding the agent invented instructions.

**Consequence.** Discovery renders prompt expressions (`+`, f-strings, conditionals,
the skill helpers) instead of accepting only string constants; the AST reads skill
files but never executes anything.

## 20. LLM mocking as a second layer behind deterministic rules, driven by strategies (2026-09-01)

**Decision.** Rules answer first; only under `on_miss: llm` does a miss go to an
`LLMMockEngine` that plays the backend from a pre-generated strategies document (a
shared world + per-tool behaviour, examples and a validated fallback), validates every
answer against the tool's `output_schema`, repairs once and then falls back or errors
(`on_invalid`). Strategies are generated in the `mocks` stage, embedded in the dataset
and selectable per case and per simulation scenario; the model is named in config
(`models.mock`, default the generator); every mocked call is logged and the report
attributes instability that coincides with LLM-mocked calls to the mock layer.

**Alternatives.** (a) LLM-only mocking — rejected: fixtures for the known entities are
cheaper, deterministic and verifiable, and the review gate depends on them. (b) Letting
the engine improvise without strategies — rejected: answers would drift between cases
and from the fixtures; the strategy is the script that keeps the world consistent, the
same plan-plus-model shape ADK uses for its user simulator (ADK itself does not inject
mock tool responses). (c) Making `llm` the default — rejected: stability tracking is the
pipeline's main signal and needs deterministic tool answers; `strict` stays the default
and `llm` is an explicit opt-in whose cost (a model call per unmatched tool call) and
non-determinism are visible in the report.

**Consequence.** Under `llm` the mocks stage leaves the wildcard defaults out (the long
tail must reach the engine), verification counts misses instead of failing on them, an
unfixable engine answer is an infrastructure error, and the run artifact carries a
per-case ledger the UI shows.

## 21. Artifact tiers: no file duplicates part of another artifact (2026-09-03)

**Decision.** Every artifact kind declares a `tier` — `final` (a deliverable, at the
output root or under `results/`), `work` (scratch under `<out_dir>/work/`), or `derived`
(no file at all) — and, independently, a `folded_into` path naming the place inside
another artifact that durably holds its data. Five stand-alone files that were copies of
data already in a deliverable stopped being written at the root: `applicable-failures.json`
and `coverage-plan.json` / `coverage.json` are folded into `agent-map.json`
(`applicable_failures`) and `dataset.json` (`coverage.plan` / `coverage.achieved`), while
`mock-rules.json` and `mock-strategies.json` stay on disk only as the mocks→dataset
handoff in `work/`, their durable home being `dataset.mocks`. `state.json`,
`run-progress.json` and `job.json` moved to `work/` as well. The output root went from
15 files to 8, and `work/` can be deleted without losing anything.

**Alternatives.** (a) Leave them — rejected: `mock-rules.json` was byte-identical to
`dataset.mocks.tools` and `coverage.json` to `report.json`'s `coverage`, so two files
could disagree after a hand edit and a reader had no rule for which one won.
(b) Fold everything, including the mocks handoff, by having the `mocks` stage seed
`dataset.json` early — rejected: it couples two stages through a half-built deliverable
and leaves a case-less dataset behind whenever the dataset stage fails. (c) Move
everything to `work/` without folding — rejected for `coverage`: the achieved coverage
is a result of the evaluation, so it belongs in a deliverable, not in scratch.

**Consequence.** `path_for` raises for derived kinds, so a stage cannot write a copy by
accident. Folded kinds stay in the registry, so pre-fold directories and single-file
uploads still resolve, and `fold_from` lets the UI's `Bundle` and `PipelineContext`
present the same kind whichever way they got it — no page or stage knows about the fold.
`dataset.mocks` is the single home of both mock layers from the dataset stage on, while
the dataset stage itself reads the `work/` handoff so a stale `dataset.json` cannot
shadow the rules the current run just authored. `evalbuilder pipeline compact <dir>`
migrates an older output directory in place, idempotently.

## 22. The agent runs behind an HTTP server on a deployment target (2026-09-06)

**Decision.** The pipeline never imports the target agent into its own process. The
`deploy` stage runs it — with both mock layers — behind `evalbuilder serve` on the
config's `deploy.target`: `local` (a detached subprocess of the same interpreter),
`docker` (Docker Compose on the local daemon), `kubernetes` (`kubectl`; Docker Desktop's
cluster or any other) or `openshift` (`oc`, a prototype). Every target implements one
interface (`available`, `render`, `build`, `up`, `status`, `logs`, `down`), goes through
one logged `CommandRunner`, writes what it generated to `work/deploy/` before using it,
and leaves `deployment.json` behind. The `infer` and `simulate` stages talk to the
recorded endpoint through `RemoteAgent`; the server is stateless (the full OpenAI-format
history and the mock block travel with every request, the LLM mock engine's history as
`mocks.history`), so replicas are interchangeable. `local` is the same path, which is
what lets the offline test suite cover deploy → infer → eval without Docker.

**Alternatives.** (a) Keep the in-process runner for `local` and add the HTTP path only
for containers — rejected: two inference code paths, two mock-installation paths, and
the offline tests would never exercise the one that matters. (b) A stateful server with
conversation ids — rejected: it needs a session store, pins a conversation to one
replica and complicates retries; `convert_to_openai_messages` / `convert_to_messages`
round-trip tool calls exactly, so full-history requests cost nothing. (c) Bake the
dataset into the image — rejected: mocks travel per request, so a regenerated dataset
needs no rebuild. (d) FastAPI / uvicorn / `requests` — rejected: two JSON routes do not
justify three dependencies in the image; the stdlib server and `urllib` suffice.

**Consequence.** Models that only exist on the host cannot serve the agent: a
`claude-cli:` agent or mock model is refused by `problems()` under a container target
(the generator and the judges still run on the host). The generated Dockerfile assumes
the evalbuilder checkout as build context; a foreign project brings its own
`build.dockerfile`. `teardown` is a stage of its own (optional, after `simulate`,
skipped with `deploy.keep`), and `--until deploy` leaves the agent up for probing.

## 23. Kubernetes images through a loader DaemonSet; port-forward as the endpoint (2026-09-06)

**Decision.** On Kubernetes the image is distributed by `push_mode`: `registry`
(`docker push`, pull policy `Always`), `load` (a privileged `hostPID` DaemonSet on every
node through which `docker save IMAGE` is streamed into containerd — `kubectl exec -i …
nsenter -t 1 … ctr -n k8s.io images import -` — pull policy `Never`) or `none`; `auto`
means `registry` when `deploy.image.registry` is set, else `load`. The pipeline reaches
the Service through a detached `kubectl port-forward` by default (`expose: nodeport`
uses the node's InternalIP, `route` is OpenShift's), whose pid is recorded and which
`status` restarts when it died. OpenShift defaults to `registry` and `route`.

**Alternatives.** (a) Rely on the cluster seeing local images — rejected: verified on
Docker Desktop's kind-based cluster, where `imagePullPolicy: Never` yields
`ErrImageNeverPull` for a locally built image and `docker desktop kubernetes` offers no
image-load command. (b) A local registry the nodes pull from — rejected: the nodes'
containerd would need an insecure-registry entry, which Docker Desktop manages and hides.
(c) NodePort or LoadBalancer as the endpoint — rejected: on the same cluster a
LoadBalancer gets an address on the kind network and neither it nor the NodePort is
reachable from the host; a tunnel works on every cluster, at the price of one process.
(d) Compose Bridge / Helm for the manifests — rejected: a Deployment, a Service and an
optional Route are a few dozen lines of YAML the user can read.

**Consequence.** `load` needs privileged pods (fine on Docker Desktop, minikube, kind;
not on OpenShift, hence its `registry` default). A port-forward is a single tunnel to
the Service, not a load balancer over replicas. Everything the target ran is in
`deployment.json`'s `commands` and `work/deploy/commands.log`.

## 24. joblib for the inference and evaluation phases (2026-09-06)

**Decision.** `inference.py` runs cases and scenarios with `joblib.Parallel(n_jobs,
prefer="threads"|"processes", return_as="generator")`; `evaluators.score_run` scores
contiguous case-run splits the same way (`evaluation.workers` / `backend`). The unit of
work is one conversation (a case with all its turns, or a scenario) or one scoring
split; the agent clients pickle (models are rebuilt from specs in the worker), so both
backends see identical inputs, and the ordered generator keeps results — and therefore
artifacts — identical to a sequential pass while progress is reported after every item.

**Alternatives.** (a) Keep the `ThreadPoolExecutor` pools and the intent-group
grouping — rejected: the grouping existed to keep cases of one intent sequential in a
shared process; with a stateless server every case is independent, and one library
with a process option covers the CPU-bound in-process case (scripted agents) too. (b)
`asyncio` — rejected: the agent, the judges and the mock engine are synchronous
LangChain calls; threads are the natural fit and processes are a config switch.

**Consequence.** `runs.parallel_intents` / `parallel_scoring` / `parallel_simulations`
became `inference.workers` / `evaluation.workers` (and one pool for cases and
scenarios); old configs are migrated on load with a note. A model object (an injected
test generator as the simulated user) forces the threads backend.

## 25. `run` → `infer`, `score` → `eval`; the old names stay as aliases (2026-09-06)

**Decision.** The phases are named the same everywhere: config sections `deploy` /
`inference` / `evaluation`, stages `deploy` / `infer` / `teardown` (plus `score` and
`aggregate` inside the evaluation phase), commands `evalbuilder deploy`, `evalbuilder
infer`, `evalbuilder eval`. `evalbuilder run` is a hidden alias of `infer` with identical
behaviour; `evalbuilder score` stays for one run (the interactive skills use it) and
`eval` is the multi-run, aggregating form. Readers accept both stage names: the report,
the UI loader and the progress widget fall back to `stages.run` in a directory written
before the rename, and the run artifact keeps its file name (`results/run-<id>.json`).

**Reasoning.** A phase that has its own command, config section, stage and artifact
should carry one name in all four places; keeping the pre-phase names as aliases costs
one decorator each and spares the skills and scripts an edit. **Consequence:** a
`work/state.json` from an older run resumes by re-executing `deploy` and `infer` (there
is no cached `infer` record), which is the safe outcome.
