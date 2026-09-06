---
name: agent-eval-pipeline
description: Use when the user wants a complete, unattended evaluation of a LangGraph agent from one config file — runs discover → map → mocks → dataset → review → deploy (the agent with its mock layers behind an HTTP endpoint: local, docker, kubernetes, openshift) → infer (repeated runs + simulations) → teardown → scoring → aggregation → analysis and writes report.json; also for the dedicated phase commands (deploy / infer / eval) and for reading or acting on a pipeline report.
---

# Autonomous evaluation pipeline

One config in, one `report.json` out. The pipeline drives the same CLI the four
interactive skills use (discover, dataset, mock, infer/eval), but an LLM "generator"
plays the author role and every artifact still passes the CLI's validation. Use the
interactive skills when the user wants to shape intents or cases by hand; use this
when they want a verdict.

Three phases have their own commands and stages: **deploy** (the agent, with both mock
layers, behind an HTTP endpoint — `evalbuilder deploy`, stages `deploy` / `teardown`),
**infer** (cases and multi-turn simulations against that endpoint, in parallel with
joblib — `evalbuilder infer`, stages `infer` / `simulate`) and **eval** (scoring stored
runs locally with joblib splits, then aggregation — `evalbuilder eval`, stages `score`
/ `aggregate`). Stage order: preflight → discover → map → mocks → dataset → review →
verify → deploy → infer → simulate → teardown → score → aggregate → publish → analyze
→ report.

## Workflow

1. Confirm the target contract: the module exposes `TOOLS` and
   `build_agent(model=None, tools=None)`. If it does not, stop and help the user add
   that seam first — the pipeline injects models and mocked tools through it.

2. Create the config (then edit it with the user's constraints, evaluators,
   thresholds and coverage). Every key is documented in `references/config-reference.md`:

   ```bash
   evalbuilder pipeline init eval/pipeline.yaml --name NAME --source path/to/agent.py --module pkg.agent
   ```

   - `models.*` default to `claude-cli:claude-sonnet-5` (Claude Sonnet 5 through the
     Claude Code subscription via the `claude` binary; no API key). API-key providers are interchangeable:
     `anthropic:claude-sonnet-5` (`ANTHROPIC_API_KEY`), `openai:gpt-5`
     (`OPENAI_API_KEY`), `gemini:gemini-2.5-pro` (`GOOGLE_API_KEY`); each needs its
     package (`uv sync --extra anthropic|openai|gemini|llm`, or `pip install -e '.[llm]'`
     without uv). `evalbuilder check`
     prints per-provider readiness (`providers`), and preflight fails early with the
     exact missing key/package.
   - `review.auto_approve: true` + `approved_by` is the user's explicit authorization
     for the pipeline to approve generated cases. **Never set it yourself** — ask; without
     it the pipeline stops at `awaiting_review` and writes a report saying so.
   - `mocking.required: true` (default) means every mockable tool gets a fixture (skill
     loaders are never mocked); `on_miss: strict` means an unmatched call is an error,
     never a real call. `on_miss: llm` adds the second layer: the `mocks` stage also
     writes mock strategies (a described backend world + per-tool behaviours, stored in
     `dataset.mocks.strategies`), keeps the wildcard defaults out of the rules, and an LLM mock engine
     driven by `models.mock` (default: the generator) answers the calls no rule
     covers — validated against the tool's `output_schema`, one repair round, then
     `mocking.on_invalid` (`fallback` | `strict`). Cases and simulation scenarios may
     select an alternate strategy (`mocking.strategy` is the default).
   - Agent Skills (`SKILL.md` folders the agent embeds or loads on demand) are part of
     the map (`skills[]`): the generator sees their instructions, scenarios list the
     skills they exercise, `skill_misuse` becomes an applicable failure type, and
     `dataset.coverage.achieved` counts cases per skill (`uncovered_skills`).
   - `instructions` (free text) and `feedback` entries reach every generation prompt;
     `coverage.per_tool_edge_cases` adds schema-derived edge cases per tool (missing /
     wrong / out-of-enum / boundary input, malformed tool output) from the tools'
     pydantic / `args_schema` / return-annotation schemas captured in `agent-map.json`.
   - `deploy.target` says where the agent runs during inference: `local` (the default —
     `evalbuilder serve` as a subprocess, no Docker), `docker` (Docker Compose on the
     local daemon), `kubernetes` (`kubectl`; the image is loaded straight into the nodes
     or pushed to `deploy.image.registry` — `image.push: auto | registry | load | none`),
     `openshift` (`oc`, prototype; needs `image.registry`, exposes a Route). A container
     target needs an agent model that runs **inside the image**: an API-key provider with
     the key passed through `deploy.env` (`ANTHROPIC_API_KEY: null` copies it from the
     host), or a scripted model — never `claude-cli`, and preflight rejects that
     combination. `deploy.expose` is how the pipeline reaches the service (`port-forward`
     works on every cluster; `nodeport`; `route` on OpenShift); `deploy.keep: true` skips
     the `teardown` stage. `evalbuilder deploy render CONFIG` prints the generated
     Dockerfile / compose file / manifests before anything runs.
   - `inference.workers` / `inference.backend` (`threads` | `processes`) size the joblib
     pool that runs cases and simulation scenarios against the deployed agent;
     `evaluation.workers` / `evaluation.backend` size the scoring splits. `runs.repeats`
     is the only key left under `runs`; the old `runs.parallel_*` keys still load (they
     are migrated and reported as a problem).
   - The same setup is available interactively: `evalbuilder ui` → **Pipeline setup**
     (discover the target's tools and schemas, fill the form, edit the YAML, run).

3. Run it. Stages print to stderr; the JSON summary goes to stdout; exit code 1 unless
   the verdict is `pass`:

   ```bash
   evalbuilder pipeline run eval/pipeline.yaml
   evalbuilder pipeline run eval/pipeline.yaml --resume     # reuse completed stages after a fix
   evalbuilder pipeline run eval/pipeline.yaml --resume --from dataset   # regenerate from a stage on
   evalbuilder pipeline run eval/pipeline.yaml --until dataset            # stop after cases + mocks (exit 0)
   evalbuilder pipeline run eval/pipeline.yaml --until deploy             # …or leave the agent running for a manual probe
   evalbuilder pipeline report eval/pipeline/NAME           # human summary of report.json
   evalbuilder ui eval/pipeline/NAME                        # Streamlit report UI over every artifact
   ```

   The phases are also separate commands — the same code the stages run, for a
   deployment you want to inspect, re-run against, or score differently:

   ```bash
   evalbuilder deploy render eval/pipeline.yaml [--target kubernetes]   # the generated files, nothing runs
   evalbuilder deploy up eval/pipeline.yaml [--target docker]           # build + deploy + wait for /health → deployment.json
   evalbuilder deploy status eval/pipeline/NAME                         # exit 1 unless up and healthy
   evalbuilder deploy logs eval/pipeline/NAME [--lines 100]
   evalbuilder infer eval/pipeline/NAME/dataset.json --deployment eval/pipeline/NAME \
       --scenarios eval/pipeline/NAME/scenarios.yaml --repeats 2 --workers 4 --out eval/pipeline/NAME/results
   evalbuilder eval eval/pipeline/NAME/results/run-*.json --dataset eval/pipeline/NAME/dataset.json \
       --evaluators eval/pipeline/NAME/evaluators.yaml --config eval/pipeline.yaml --out eval/pipeline/NAME/results
   evalbuilder deploy down eval/pipeline/NAME
   evalbuilder deploy build eval/pipeline.yaml                          # image only
   ```

   `--until deploy` and `deploy up` leave the agent running — tear it down with
   `deploy down` (or resume the pipeline, whose `teardown` stage does it unless
   `deploy.keep` is set).

   When the user wants to look at the generated dataset before anything runs: `--until
   dataset`, read `dataset.json` with them (cases, both mock layers and coverage all
   live in it), add their comments as
   `feedback` entries in the config (`{at, note, from_stage}`), regenerate with
   `--resume --from dataset --until dataset`, then `--resume` once they approve. The UI's
   **Run & review** page does the same loop with buttons.

4. Read the report the way the run skill reads a score report — verdict first, then
   the parts that explain it:
   - `verdict` ∈ `pass | fail | incomplete` with `verdict_reasons`. `incomplete` means
     a required stage did not finish; `stages.*` says which one and why.
   - `metrics` (pass rate vs threshold; `skipped` = no reference on the case, not an
     error), `slices` (a defect usually lives in one slice), `stability`:
     unstable **cases** = the tool trajectory differs across repeats (agent
     behavior); unstable **outputs** = same trajectory, a deterministic metric flips
     (wording drift); unstable **evaluators** = same trajectory, a judge flips
     (judge variance, not an agent defect); `suspect_judge_comments` = judge
     rationales that look like placeholders — treat those scores as unverified.
   - `coverage` — planned vs achieved cells; gaps are listed, never hidden; with
     skills, `skills` / `uncovered_skills` say which skills no case exercises.
   - `mocking` — the layers in force, the mock model and strategies, and the calls
     answered per layer (`rule`, `llm`, `invalid`, `fallback`, `error`);
     `stability.llm_mocked_unstable` lists unstable cases whose tool answers came from
     the LLM engine — attribute those to the mock layer before the agent.
   - `deployment` — where the agent ran (`target`, `image`, `endpoint`, `expose`,
     `push`) and its `status` now (`down` after teardown, `up` when kept); the full
     record with every command executed is `deployment.json`, the generated files and
     `commands.log` are under `work/deploy/`. A `deploy` stage that failed says whether
     the target was unavailable (`docker` / `kubectl` / `oc` missing, cluster
     unreachable) or the container never became healthy (its log tail is in the error).
   - `stages.infer.details.execution` — `mode` (`remote`), the `endpoint`, `workers`
     and `backend`; every run artifact carries the same `execution` block and each
     case run a per-turn `log` (`turn`, `seconds`, `tool_calls`, `error`, `endpoint`),
     so a slow or failing turn can be located without re-running.
   - `analysis` — generator-written patterns and recommendations (`source:
     deterministic` means the LLM analysis failed and only facts are listed).
   - `problems[]` — everything that was dropped, degraded, or recovered.
   - Every artifact in the output directory follows one naming convention
     (`references/config-reference.md`, "Artifacts"): `<kind>.json` at the root,
     `results/<kind>-<run_id>.json` per run, each with an embedded `schema` id — the
     UI (`evalbuilder ui`) identifies uploaded files by that id.

5. Report to the user in that order and keep the three failure classes apart: agent
   defects, evaluator/mock problems, and pipeline problems.

## Rules

- Never edit artifacts under the output directory by hand; rerun with `--resume`
  after fixing the config or the agent.
- A `pass` with low coverage or many `problems` is not a clean pass — say so.
- The generator's output is a hypothesis validated structurally, not semantically;
  point the user at `agent-map.json` and `dataset.json` for a spot check before they
  trust thresholds.
- Costs: every case runs `runs.repeats` times and every judge evaluator calls the
  judge model per case per repeat; under `on_miss: llm` every unmatched tool call is
  a mock-model call too. Shrink `coverage.total_cases` or `runs.repeats` for a first
  pass.
- Never set `deploy.keep: true`, a `deploy.image.registry` (a push publishes the
  image), or `deploy.env` entries that copy host secrets without asking — say what
  each one does and let the user decide. A deployment left running is the user's
  resource: name it (`evalbuilder deploy status DIR`) and offer `deploy down`.
- Never choose a container target for an agent whose model is `claude-cli`; explain
  that the `claude` binary is not in the image and offer an API-key or scripted model.
