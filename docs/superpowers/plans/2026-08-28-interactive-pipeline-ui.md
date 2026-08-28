# Interactive pipeline UI + schema-aware edge cases + subagent examples — plan

Spec: `docs/superpowers/specs/2026-08-28-interactive-pipeline-ui-design.md`.
Each task: tests first or alongside, `uv run pytest`, commit.

1. **`tool_schemas.py`** — validate subset, example, corrupt, edge-case derivation,
   live args/output schema resolution; `SchemaScriptedModel` in `testing.py`.
2. **Discovery** — AST: `args_schema=`, return annotations, pydantic classes in the
   source; live enrichment (`output_schema`, `schema_source`, `side_effecting`,
   `edge_cases`).
3. **Config + control** — `instructions`, `feedback`, `per_tool_edge_cases`, Sonnet 5
   defaults, `to_yaml`; engine `stop_after`; `run_pipeline(until=)`; CLI `--until`;
   review already-approved path; `jobs.py` + `pipeline_job` artifact kind.
4. **Planning + generator** — schema-edge cells; guidance block; mock validation with
   schema fallback; malformed_output injection; expected args validation.
5. **Examples** — `incident_desk`, `loan_desk` (agent, offline generator, pipeline.yaml,
   tests, offline e2e).
6. **UI** — Setup and Run & review pages; sidebar integration; AppTest tests.
7. **Playwright flows** — `tests/ui/test_playwright_flows.py`.
8. **Docs + live run** — README, config reference, skill; live incident-desk run under
   `docs/examples/incident-desk/`; memory.
