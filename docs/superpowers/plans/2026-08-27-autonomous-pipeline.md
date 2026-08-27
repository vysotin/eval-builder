# Autonomous pipeline — implementation plan

Spec: `docs/superpowers/specs/2026-08-27-autonomous-pipeline-design.md`.
Each task: tests first, run `uv run pytest`, commit.

1. **`claude_cli.py`** — `ChatClaudeCLI` (subclass of `ChatClaudeCode`; subprocess
   transport; flattened transcript; tool calls + structured output via
   `--json-schema`; unique tool-call ids), `model_from_spec()`, `claude_available()`.
   Tests stub `subprocess.run`.
2. **Model injection** — `target.build_graph(model=)`, `runner.run_dataset(model=,
   on_miss=)`, `Settings` gains agent/generator model; `evaluators` judge resolution
   through `model_from_spec` (+ `openevals`, `trajectory_llm` evaluator types).
3. **`examples/support_bot`** — router graph + external tools + scripted default
   model with `with_structured_output`; tests for happy/refund/out-of-scope routes
   and tool injection.
4. **`pipeline/config.py`** — `PipelineConfig`, YAML loader, template; tests.
5. **`pipeline/engine.py`** — stage engine with dependencies, retries, skip, state
   persistence, resume; tests with fake stages.
6. **`pipeline/generator.py`** — prompts + schemas + validation/repair loop;
   `FakeGenerator` for tests; tests for repair loop and coverage planning.
7. **`pipeline/aggregate.py`** — repeat-aware aggregation, thresholds, slices,
   stability, coverage achieved; tests on synthetic reports.
8. **`pipeline/stages.py` + `report.py` + CLI `pipeline init|run|report`** —
   offline e2e test on `support_bot` with scripted models + fake generator.
9. **Skill `agent-eval-pipeline`** + skills lint (5 skills) + README.
10. **Live e2e** with `claude-cli:sonnet` on `support_bot`; fix what breaks; save the
    report under `docs/examples/`; update memory.
