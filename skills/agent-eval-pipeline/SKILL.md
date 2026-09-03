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
   - The same setup is available interactively: `evalbuilder ui` → **Pipeline setup**
     (discover the target's tools and schemas, fill the form, edit the YAML, run).

3. Run it. Stages print to stderr; the JSON summary goes to stdout; exit code 1 unless
   the verdict is `pass`:

   ```bash
   evalbuilder pipeline run eval/pipeline.yaml
   evalbuilder pipeline run eval/pipeline.yaml --resume     # reuse completed stages after a fix
   evalbuilder pipeline run eval/pipeline.yaml --resume --from dataset   # regenerate from a stage on
   evalbuilder pipeline run eval/pipeline.yaml --until dataset            # stop after cases + mocks (exit 0)
   evalbuilder pipeline report eval/pipeline/NAME           # human summary of report.json
   evalbuilder ui eval/pipeline/NAME                        # Streamlit report UI over every artifact
   ```

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
