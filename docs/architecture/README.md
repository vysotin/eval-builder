# eval-builder architecture guide

How every skill, CLI command, module, the autonomous pipeline and the UI work — with the
architectural decisions behind them, the reasoning for each choice, and the limitations
you should expect. Written against the code as of 2026-08-30, extended 2026-09-01 (Agent Skills in the
map, two-layer mocking) and 2026-09-06 (the deploy → infer → eval phases: the agent runs
behind an HTTP server on a deployment target — local, Docker Compose, Kubernetes,
OpenShift — and inference / scoring run in parallel with joblib); file paths are relative
to the repository root.

| document | covers |
|---|---|
| [01-skills.md](01-skills.md) | the five Claude Code skills (`agent-eval-discover`, `-dataset`, `-mock`, `-run`, `-pipeline`): what each one does, its workflow, its prohibitions, and why skills are prose while the CLI is code |
| [02-cli.md](02-cli.md) | every `evalbuilder` command — incl. the phase commands `deploy`, `infer`, `eval` and the agent server `serve` — its inputs/outputs, exit codes, and the module it delegates to |
| [03-core-modules.md](03-core-modules.md) | the deterministic core: artifacts and schemas, discovery (incl. Agent Skills), skills, tool schemas and edge cases, target contract, the agent clients and the agent server, the joblib inference engine, deployment targets, two-layer mocking (rules + LLM mock engine), evaluators (chunked scoring), simulation, coverage, providers and the Claude CLI adapter, LangSmith I/O, offline testing helpers |
| [04-pipeline.md](04-pipeline.md) | the autonomous pipeline: config (incl. `deploy` / `inference` / `evaluation`), stage engine, the 16 stages (deploy → infer → simulate → teardown among them), the LLM generator and its validation loop, coverage planning, failure taxonomy gating, aggregation and stability, the report, the artifact naming convention, background jobs and the interactive setup helpers |
| [05-ui.md](05-ui.md) | the Streamlit app: the project every page follows, the persistent *Pipeline setup* (with the deployment and parallelism blocks) and *Run & review* (with the deployment status) pages, the report pages (Summary last), and how it is tested |
| [06-decisions.md](06-decisions.md) | the architectural decision record: each important choice, the alternatives considered, and the reasoning |
| [07-limitations.md](07-limitations.md) | expected limitations, by area, with the workaround where one exists |

## The system in one picture

```
                 interactive path (Claude Code)                     autonomous path
 ┌──────────────────────────────────────────────────┐   ┌──────────────────────────────────┐
 │ agent-eval-discover → agent-eval-dataset →       │   │ pipeline.yaml (or the UI form)   │
 │ agent-eval-mock → agent-eval-run                 │   │   evalbuilder pipeline run       │
 │ (skills = policy prose; the human decides)       │   │   (LLM generator = the author;   │
 └───────────────────────┬──────────────────────────┘   │    config = the human decisions) │
                         │  every artifact mutation      └────────────────┬─────────────────┘
                         ▼                                                ▼
              evalbuilder CLI / library  (src/evalbuilder/)  — deterministic: schemas, IDs,
              validation, coverage math, review state machine, tool mocking, execution,
              trajectory capture, evaluators, aggregation, LangSmith I/O, artifact layout
                         │
                         │   deploy ──► the agent + both mock layers behind `evalbuilder serve`
                         │             (local subprocess · Docker Compose · Kubernetes · OpenShift)
                         │   infer ───► cases + scenarios against that endpoint, joblib-parallel
                         │   eval ────► scored locally in joblib splits, aggregated
                         ▼
              eval/agent-map.json · eval/datasets/*.json · eval/results/*   (skills)
              eval/pipeline/<name>/  agent-map.json, dataset.json, results/run-*.json,
                                     aggregate.json, report.json … + work/ (scratch) (pipeline)
                         │
                         ▼
              evalbuilder ui  — Pipeline setup → Run & review → report pages
```

Three ideas organise everything:

1. **Skills are policy, the CLI is mechanism.** Skills tell a model *what to decide and
   what it may never do*; the CLI/library owns every fact that must be reproducible
   (schemas, IDs, validation, coverage, review state, mocking, execution, scoring). No
   artifact is ever written by prose. See [06-decisions.md §1](06-decisions.md#1-skills--policy-prose-cli--mechanism).
2. **Hypotheses with evidence, decisions by humans.** Everything derived from code is a
   hypothesis that cites evidence (`source:<file>:<line>`, `prompt:<node>`,
   `tool:<name>`, `edge:<a>-><b>`, `schema:<tool>.<field>`); generated cases always land
   `pending`; approval is an explicit, recorded human act (`evalbuilder review`, or
   `review.auto_approve` + `approved_by` in the pipeline config, or the UI's approver name).
3. **One inference path, wherever the agent runs.** The pipeline never imports the
   agent into its own process: the agent, with both mock layers, is served over HTTP by
   `evalbuilder serve` on a deployment target — a local subprocess, a Docker Compose
   service, a Kubernetes Deployment reached through a port-forward, an OpenShift Route —
   and every case, scenario and mock block travels as one JSON request. The `local`
   target is the same path, which is why the offline test suite exercises the whole
   deploy → infer → eval chain without Docker. See [06-decisions.md §22](06-decisions.md#22-the-agent-runs-behind-an-http-server-on-a-deployment-target-2026-09-06).
4. **Determinism where it is cheap, LLMs where they are needed.** Tool calls are mocked
   ADK-style so runs are repeatable, and only where rules run out — and only when the
   config opts in — an LLM mock engine plays the backend from pre-generated strategies,
   validated against the tool's output schema and logged per call; deterministic
   evaluators run before judges; tool schemas produce edge cases without an LLM;
   everything an LLM writes is validated by the same code the CLI uses, with one repair
   round, and whatever is still invalid is dropped and reported rather than silently kept.

## Where the reasoning came from

The design synthesises three earlier repositories (see
`docs/superpowers/specs/2026-08-23-eval-builder-skills-design.md` §2): the ADK eval
toolkit (tool/sub-agent mocking semantics, inference/scoring split), the Dify agent-test
skills (skills-as-prose + deterministic CLI, coverage grid, content-hash IDs, review state
machine, read-back-verified LangSmith publish) and the second-generation Dify eval design
(evidence-cited analysis, structurally gated failure taxonomy, simulation as goal + stop
conditions, executable skill contract tests). The dated specs under
`docs/superpowers/specs/` are the primary sources (the deployment phases are
`2026-09-06-deploy-infer-eval-design.md`); this guide is the consolidated,
code-verified description.
