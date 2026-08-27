---
name: agent-eval-pipeline
description: Use when the user wants a complete, unattended evaluation of a LangGraph agent from one config file — runs discover → map → mocks → dataset → review → repeated runs → scoring → aggregation → simulation → analysis and writes report.json; also for reading or acting on a pipeline report.
---

# Autonomous evaluation pipeline

One config in, one `report.json` out. The pipeline drives the same CLI the four
interactive skills use (discover, dataset, mock, run), but an LLM "generator" plays
the author role and every artifact still passes the CLI's validation. Use the
interactive skills when the user wants to shape intents or cases by hand; use this
when they want a verdict.

## Workflow

1. Confirm the target contract: the module exposes `TOOLS` and
   `build_agent(model=None, tools=None)`. If it does not, stop and help the user add
   that seam first — the pipeline injects models and mocked tools through it.

2. Create the config (then edit it with the user's constraints, evaluators,
   thresholds and coverage). Every key is documented in `references/config-reference.md`:

   ```bash
   evalbuilder pipeline init eval/pipeline.yaml --name NAME --source path/to/agent.py --module pkg.agent
   ```

   - `models.*` default to `claude-cli:sonnet` (Claude Code subscription via the
     `claude` binary; no API key). `anthropic:` / `openai:` specs work with keys.
   - `review.auto_approve: true` + `approved_by` is the user's explicit authorization
     for the pipeline to approve generated cases. **Never set it yourself** — ask; without
     it the pipeline stops at `awaiting_review` and writes a report saying so.
   - `mocking.required: true` (default) means every introspected tool gets a fixture;
     `on_miss: strict` means an unmatched call is an error, never a real call.

3. Run it. Stages print to stderr; the JSON summary goes to stdout; exit code 1 unless
   the verdict is `pass`:

   ```bash
   evalbuilder pipeline run eval/pipeline.yaml
   evalbuilder pipeline run eval/pipeline.yaml --resume     # reuse completed stages after a fix
   evalbuilder pipeline run eval/pipeline.yaml --resume --from dataset   # regenerate from a stage on
   evalbuilder pipeline report eval/pipeline/NAME           # human summary of report.json
   ```

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
   - `coverage` — planned vs achieved cells; gaps are listed, never hidden.
   - `analysis` — generator-written patterns and recommendations (`source:
     deterministic` means the LLM analysis failed and only facts are listed).
   - `problems[]` — everything that was dropped, degraded, or recovered.

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
  judge model per case per repeat. Shrink `coverage.total_cases` or `runs.repeats`
  for a first pass.
