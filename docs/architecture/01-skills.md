# The five skills

Skills live in `skills/agent-eval-*/SKILL.md` (symlinked into `.claude/skills/` so
Claude Code discovers them) with optional `references/*.md` for depth. Each skill is
short imperative prose: a workflow of numbered steps, the exact CLI commands to run, and
explicit prohibitions. They contain **no logic** — every artifact mutation is a CLI call
whose output the skill reads back.

`tests/test_skills_lint.py` is the executable contract between the prose and the code:
there must be exactly five skills with frontmatter whose `description` starts with
"Use when", every `evalbuilder …` command mentioned in a skill must resolve to a real
command (`--help` exits 0), every `references/` file mentioned must exist, and the
symlinks must resolve. If a skill drifts from the CLI, CI fails.

## Why prose + CLI rather than prose only

Both reference repositories that inspired this design (`dify-agent-test-skills`,
`dify-eval`) converged independently on the same shape after trying "the model writes
the JSON": hand-written artifacts drift from the schema, validation gets skipped when
the model is in a hurry, and — the decisive point — an approval gate that lives in
prose can be argued around. Putting IDs, validation, coverage math and the review
state machine behind commands makes those properties *mechanical*: the model cannot
approve a case without running `evalbuilder review`, and that command only changes
`review.status`, never content. The skill text then only has to say *when* to run it.

## agent-eval-discover — map the test surface

**Use when** starting eval work, before generating any dataset, or after the agent's
code changed.

Workflow (`skills/agent-eval-discover/SKILL.md`):

1. `evalbuilder check --target-module MODULE` — capability matrix. Blocking items
   (LangGraph missing, target not importable) stop; degraded items (no judge provider,
   no LangSmith key) are reported once, never narrated.
2. `evalbuilder discover MODULE --source path/to/agent.py` — the structural skeleton
   (AST + live introspection, see [03-core-modules.md §Discovery](03-core-modules.md#discovery-discoverpy)).
3. Read the source and prompts *yourself*: node names and tool names are weak
   evidence; the prompts and conditional edges say what the agent decides.
4. Draft intents and scenarios. **Every entry cites evidence** (`source:`, `prompt:`,
   `tool:`, `edge:` tokens) and is labelled a hypothesis.
5. Propose failure scenarios only from `references/failure-taxonomy.md`, and only the
   types whose structural precondition holds (conditional edges → `branch_misrouting`,
   tools → `tool_misuse`/`tool_error_handling`, retriever → `retrieval_grounding`, …).
   An unjustifiable failure type produces cases that can never fail for the claimed
   reason.
6. Data-domain topics come only from documents or descriptions the user provides —
   never invented.
7. Ask for constraints the code cannot express ("never quote a price"); they reach
   every generation prompt and typically become judge contracts.
8. `evalbuilder agent-map update eval/agent-map.json --intents @… --scenarios @… --failures @… --topics @… --constraints @…` — each list replaces its section after shape validation.
9. Present the map and its `decisions_needed`, then **stop** until the user confirms.

Rules: never hand-edit `agent-map.json`; heuristics are hypotheses; `source_sha256`
in the map detects a stale map after code changes.

**Reasoning.** Downstream generation is "behaviourally blind" without this map: cases
would be plausible text with accidental coverage. Forcing evidence tokens keeps the
model honest about what the code actually supports, and gating failure types by
structure (a rule borrowed from `dify-eval`) prevents the most common synthetic-eval
defect — failure cases that test something the agent cannot even do.

## agent-eval-dataset — coverage-driven golden cases

**Use when** the agent map is reviewed and the user wants a dataset (new or extended).
The skill is the *generator*; the CLI owns validation, IDs, coverage math and review state.

Workflow (`skills/agent-eval-dataset/SKILL.md`, `references/generation-guide.md`):

1. Import existing goldens first (`evalbuilder dataset import … --from goldens.json`) —
   they land `pending` with `metadata.source=import`.
2. `evalbuilder dataset init PATH --name N --type final_response --target MODULE:build_agent`.
3. `evalbuilder dataset gaps PATH --agent-map eval/agent-map.json` — the required
   cells of the grid `intent × topic × scenario × failure_mode`.
4. `evalbuilder dataset add PATH --case @case.json` per case. A case carries
   `inputs.messages`, testable `reference_outputs` (`response`, `expected_tools` with
   args, optional `trajectory`, `contains`, `contract`), and `metadata` (the full
   coverage cell, a `variant` from the taxonomy happy / boundary / adversarial /
   linguistic / multi-turn, `evidence`).
5. Self-filter (rewrite ambiguous or unanswerable inputs) and apply input evolutions
   (concretizing, constrained, comparative, multicontext) — an unevolved synthetic
   dataset is trivially easy.
6. Re-run `gaps`, fill, `evalbuilder dataset validate`.
7. Present pending cases and ask how the user wants to review.

**The approval gate.** Only `evalbuilder review PATH --approve "id1,id2" --note …`
records a decision, and only after the user explicitly approved *specific cases in
this conversation*. Silence, enthusiasm or a green run are not approval; rejected cases
are kept as history; imported or regenerated content returns to `pending` by design.

Golden-answer modes: `llm_written` (default) and `agent_backfilled` (run the approved
inputs once unmocked, verify by hand, copy outputs into references — never an agent's
own unverified output).

## agent-eval-mock — deterministic runs through tool mocks

**Use when** cases depend on nondeterministic, side-effecting, rate-limited or
unavailable tools.

Semantics (implemented in `src/evalbuilder/mocking.py`): per tool an **ordered** rule
list, first match wins; `matchArgs` is a subset match (recursive for nested
pydantic-model arguments); `{}` is a wildcard; on a miss the policy is `real` (call the
real tool — the interactive default), `fallback` or `strict` (error). Rules live in
the dataset — dataset-level `mocks.tools` for shared fixtures, per-case
`metadata.mocks.tools` overriding the dataset list for that tool. Sub-agents exposed
as tools are mocked exactly like tools.

Workflow: list affected tools → `evalbuilder mock set PATH [--case ID] --tool T --rules @rules.json`
(most specific first, wildcard last) → `evalbuilder mock verify PATH` (every
`expected_tools` call in a mocked case is answerable by a rule) → state the miss
policy in the summary.

Rules: never mock away the tool a case tests the error handling of — unless the mock
injects the error (`{"matchArgs": {}, "response": {"error": "timeout"}}`); never leave
side-effecting tools unmocked in a dataset meant for repeated runs; fixtures are not
goldens — `reference_outputs` still defines the expected behaviour.

**Reasoning.** The rule model is lifted from the ADK eval toolkit's `mockResponses`
because it is the smallest scheme that covers the real needs: several fixtures per
tool keyed by arguments, a default, and an explicit decision about what happens when
nothing matches. Installing mocks by *wrapping tools with the same name, description
and args schema* keeps the model's tool selection unchanged — the eval measures the
agent's behaviour, not a different tool surface.

## agent-eval-run — validate, execute, score, report

**Use when** running, scoring or re-scoring an experiment, publishing to LangSmith, or
simulating multi-turn conversations.

Workflow: `evalbuilder dataset validate` → `evalbuilder run PATH --mock --out eval/results`
(approved cases only; mocks installed) → choose evaluators per
`references/evaluator-selection.md`, write `eval/evaluators.yaml` →
`evalbuilder score RUN --dataset PATH --evaluators eval/evaluators.yaml` → report
per-metric stats **and slices** (intent / failure_mode / variant) → keep agent errors,
infrastructure errors and evaluator errors apart.

Evaluator selection order — cheapest reliable first: deterministic (`expected_tools`,
`contains`, `json_valid`), trajectory match (AgentEvals), then LLM judges
(`correctness`, `contract`, any OpenEvals rubric, `trajectory_llm`). One metric per
evaluator so a failure localises itself. Thresholds are a *reporting* decision:
start binary, inspect distributions and slices, negotiate thresholds afterwards.

Optional: `evalbuilder publish PATH` (idempotent, read-back verified) and
`evalbuilder simulate PATH --scenarios eval/scenarios.yaml` (persona / goal / opening /
followups / stop conditions; `max_turns` is a truncation, not a pass; only violating
runs are mined back as **pending** cases with `metadata.source=simulation`).

Rules: never re-run the agent to re-score (scoring reads the stored run artifact);
never run with pending or rejected cases; report error counts per metric honestly.

## agent-eval-pipeline — one config, one report

**Use when** the user wants a complete, unattended evaluation from one config file, or
wants a pipeline report read and acted on.

Workflow: confirm the target contract (`TOOLS` + `build_agent(model=None, tools=None)`)
→ `evalbuilder pipeline init eval/pipeline.yaml --name N --source … --module …` and
edit it with the user (`references/config-reference.md` documents every key) →
`evalbuilder pipeline run …` (`--resume`, `--resume --from STAGE`, `--until dataset`)
→ read `report.json` verdict-first: `verdict` + `verdict_reasons`, `stages`, `metrics`,
`slices`, `stability` (unstable cases / outputs / evaluators, suspect judge comments),
`coverage`, `analysis`, `problems[]`.

Rules: `review.auto_approve` + `approved_by` is the user's authorization — the skill
must never set it itself; never edit output artifacts by hand (rerun with `--resume`);
a `pass` with low coverage or many problems is not a clean pass; the generator's
output is structurally validated, not semantically — point the user at
`agent-map.json` and `dataset.json` for a spot check; costs scale with cases × repeats
× judge evaluators.

The same setup is available interactively in the UI (*Pipeline setup* / *Run & review*,
see [05-ui.md](05-ui.md)); the skill points the user there when they prefer clicking.

## How the skills relate to the pipeline

The pipeline does not replace the skills; it automates the *author* role the skills
give to a human-guided model, with the same validation code and the same review gate.
Use the skills when the user wants to shape intents and cases by hand; use the
pipeline when they want a verdict; use `--until dataset` + the feedback loop when they
want to look at generated cases before anything runs.
