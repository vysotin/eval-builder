# Skills in the agent map + two-layer mocking — implementation plan

Spec: `docs/superpowers/specs/2026-09-01-skills-and-llm-mocking-design.md`. Each task is
test-first (write the failing test, implement, run `pytest -q -m "not slow and not ui"`),
committed when green. File paths are repo-relative.

## Part A — agent skills

1. **`src/evalbuilder/skills.py`** (new) — `Skill`, `load_skills`, `skills_prompt`,
   `skills_inline_prompt`, `skill_loader_tool`, `describe_skill`, `is_skill_loader`,
   `SKILL_LOADER_NAMES`. Tests: `tests/test_skills.py` (frontmatter parsing, folder-name
   default, references with title/excerpt, listing/inline prompts, loader tool answers
   and unknown-name error, `metadata.kind`).
2. **Discovery** — `tool_schemas.describe_tool` adds `kind`/`mockable`; `discover.py`:
   skills-dir detection (`load_skills(...)`, `SKILLS_DIR`, `Path(__file__).parent / …`,
   `skills=` kwargs), `_prompt_text` for composed prompts (BinOp, IfExp, JoinedStr,
   helper calls rendered from loaded skills), loader-tool assignments, node↔skill links,
   `AgentMap.skills` field + `app.skills_dir`. Tests in `tests/test_discover.py`
   (inline example, on-demand example, composed prompt text, node skills, tool kinds).
3. **Examples** — `examples/incident_desk/skills/{incident-triage,incident-comms}/`
   (SKILL.md + references), agent embeds them inline; `examples/support_bot/skills/
   {refund-policy,product-troubleshooting}/`, `load_skill` loader in `TOOLS`, prompt
   listing only when the loader is wired, scripted model loads the refund skill first;
   offline generators updated (skill evidence, `skill_misuse` scenario/case, `load_skill`
   expectations). Tests: `tests/test_support_bot.py` (loader flow), `tests/test_incident_desk.py`
   (skills in map, e2e still passes), `tests/test_examples.py`.
4. **Downstream** — taxonomy `skill_misuse` (+ `references/failure-taxonomy.md`),
   `generator.agent_brief` skills + tool kind, MAP prompt/schema/validation (`skills` on
   scenarios), CASES prompt evidence, `planning.achieved(skills=)` coverage block,
   stages `discover` live skills merge, report `agent.skills`, `analysis_summary`, CLI
   `discover` output. Tests: generator validation, planning coverage, pipeline e2e
   assertions on `report["agent"]["skills"]` and `coverage["skills"]`.

## Part B — two-layer mocking

5. **Engine + wrapper** — `src/evalbuilder/mock_engine.py` (`LLMMockEngine`,
   `MockEngineError`, prompt builder, `call_in_prompt` helper for scripted models);
   `mocking.py`: `wrap_tool(engine=, ledger=)`, `wrap_tools` wraps all mockable tools
   when an engine is present, `mockable()`, deep-copied responses, `verify_summary`.
   Tests: `tests/test_mock_engine.py`, additions to `tests/test_mocking.py`.
6. **Config, schemas, layout, artifacts** — `ModelsConfig.mock`, `MockingConfig`
   (`strategies`, `strategy`, `on_invalid`, `max_repairs`, `on_miss` + `llm`),
   `mock_model_spec`, `problems()`, template + section comments; `CaseRun.mock_calls`;
   layout kind `mock_strategies`; dataset mock-block validation (policy values,
   llm.model, strategy ids). Tests: config, layout, artifacts.
7. **Generator** — `STRATEGIES_SCHEMA/SYSTEM`, `author_strategies` (validation, generic
   default fallback), `author_mocks(wildcard=)`, `mock_strategy` on cases and scenarios,
   prompt text about strategies. Tests: `tests/test_pipeline_generator.py`.
8. **Runner, simulate, stages** — `run_dataset(mock_model=, tool_specs=, strategy=)`,
   per-case engine + ledger, `run_case` infrastructure classification for engine errors;
   `simulate_scenarios(engine_factory)`; stages: preflight mock model, mocks (no wildcard
   under llm, strategies artifact), dataset (mocks block), review (no auto-reject under
   llm), verify (`llm_answered`, strategy coverage), run (layer totals), simulate,
   aggregate (`llm_mock_calls`, `llm_mocked_unstable`), report (`mocking`). Tests:
   runner, stages via e2e.
9. **Offline mock models + e2e** — `mock_model()` in every example `offline.py` (answers
   from fixtures via `call_in_prompt`; weather's can be told to answer invalidly first),
   `mock_strategies` handlers; `tests/test_e2e_small_agents.py`: weather_bot with
   `on_miss: llm` (layer-2 calls in ledgers, report mocking section, repair + fallback,
   strict variant → infrastructure error).
10. **CLI** — `run`/`simulate` flags, `mock strategies|validate|try`, `mock verify` under
    llm. Tests in `tests/test_mocking.py` / new `tests/test_cli_mock.py`.

## Part C — UI, skills prose, docs

11. **UI** — setup form fields (+ `default_form`/`form_from_config`/`build_config`
    round trip), preview skills, Agent page skills + tool kind, Dataset page strategies
    + strategy column, Results mock-calls table, Summary mocking line, sidebar skills
    count, `dataset_summary` strategies. Tests: `tests/test_ui_pages.py`,
    `tests/test_ui_pipeline_pages.py`, `tests/test_pipeline_setup.py`; Playwright flow
    additions in `tests/ui/test_playwright_flows.py`.
12. **Skills prose + docs** — the five `SKILL.md`, `config-reference.md`,
    `failure-taxonomy.md`, `generation-guide.md`, `docs/tool-mocking.md`,
    `docs/architecture/{01..07}.md`, README; `tests/test_skills_lint.py` keeps commands real.
13. **Verification** — full suite (`-m "not ui"` incl. slow), Playwright suite when
    chromium is present, `ruff`-style lint of new modules, memory note, final commit.
