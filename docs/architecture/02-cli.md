# The `evalbuilder` CLI

`src/evalbuilder/cli.py` is a typer application. Every command prints a JSON result on
stdout (`_emit`) so a skill can read it back, prints human messages on stderr, and uses
exit codes: `0` success, `1` a validation or runtime problem the caller must act on,
`2` an unusable config. Commands never write an artifact that fails validation
(`_save_valid` validates the whole dataset before saving) — the CLI is the only writer
of artifact JSON.

## Command reference

| command | what it does | module | exit 1 when |
|---|---|---|---|
| `check [--target-module M] [--env-file F]` | capability matrix `{ready, degraded, blocking, capabilities, providers, models}` | `config.capability_check`, `providers.providers_status` | – |
| `discover MODULE [--source F] [--eval-dir D]` | AST + live introspection → `eval/agent-map.json`; prints tools, nodes, live graph, `decisions_needed` | `discover.discover_from_source`, `discover.discover_live` | module source cannot be located |
| `agent-map update PATH [--intents J] [--scenarios J] [--failures J] [--topics J] [--constraints J]` | replace sections of the map after shape checks (`id` + `evidence` per entry; `failure_type` + `evidence` for failures; string lists for topics/constraints) | `schemas.AgentMap` | any entry lacks a required key |
| `dataset init PATH --name N [--type final_response\|trajectory] --target MODULE[:FACTORY]` | empty dataset with the target contract recorded | `schemas.Dataset` | – |
| `dataset add PATH --case JSON\|@file` | normalise one case (defaults for the coverage cell, content-hash id, review reset to `pending`) and append | `artifacts.add_case` | duplicate id, invalid dataset |
| `dataset import PATH --from FILE` | import foreign shapes (`cases`/`examples`/`goldens` keys, `outputs`→`reference_outputs`), all `pending` with `source=import` | `artifacts.import_cases` | invalid dataset |
| `dataset validate PATH` | schema + review-status + mock-block validation | `artifacts.validate_dataset` | invalid |
| `dataset list PATH [--status S]` | rows of id/status/cell/source | – | – |
| `dataset gaps PATH --agent-map M [--target-per-cell N]` | required cells vs present cases | `coverage.coverage_gaps` | – |
| `review PATH --approve IDS \| --reject IDS [--note …]` | record a human decision on specific case ids (exactly one of approve/reject) | `artifacts.set_review` | unknown ids, both/neither flag |
| `mock set PATH --tool T --rules JSON [--case ID]` | set the ordered rule list (dataset-level or per-case) | dataset `mocks` / `metadata.mocks` | unknown case, invalid rules |
| `mock verify PATH` | every `expected_tools` call in mocked cases matches a rule | `mocking.verify_dataset` | any miss |
| `run PATH [--mock/--no-mock] [--ids …] [--out D] [--model SPEC] [--on-miss real\|fallback\|strict]` | execute approved cases, write `run-<id>.json` | `runner.run_dataset` | pending/rejected ids selected, no approved cases |
| `score RUN --dataset PATH --evaluators evaluators.yaml [--out D]` | score a stored run with the configured evaluators, write `score-report-<id>.json` | `evaluators.score_run` | no evaluators |
| `simulate PATH --scenarios F [--out D] [--mine/--no-mine]` | multi-turn simulations; violations mined as pending cases | `simulate.*` | invalid scenario file |
| `publish PATH [--dataset-name N] [--env-file F]` | LangSmith publish of approved cases, read-back verified | `langsmith_io.publish_approved` | no key, no approved cases, read-back mismatch |
| `pipeline init PATH --name N --source F --module M [--force]` | commented starter config | `pipeline.config.template` | file exists |
| `pipeline run CONFIG [--resume] [--from STAGE] [--until STAGE] [--quiet] [--env-file F]` | the autonomous pipeline; exit 1 unless verdict `pass` (exit 0 when stopped deliberately with `--until`) | `pipeline.report.run_pipeline` | verdict ≠ pass; exit 2 on a bad config |
| `pipeline report DIR\|report.json` | terminal summary of a report | `pipeline.report.summary_text` | no report |
| `ui [DIR] [--port N] [--headless]` | start the Streamlit app (`streamlit run … -- --dir DIR`) | `ui/app.py` | streamlit not installed |

`python -m evalbuilder.cli …` is equivalent to `evalbuilder …` (a `__main__` guard);
the UI's background jobs use that form so they run with the same interpreter as the
Streamlit server.

## Conventions worth knowing

- **JSON arguments** accept inline JSON or `@file.json` (`_read_json_arg`), so a skill
  can write a file and pass it instead of escaping JSON on a command line.
- **Working directory on `sys.path`.** `discover`, `capability_check` and
  `target.load_target` insert the current directory into `sys.path` because targets
  usually live in the repository, not in site-packages. Run the CLI from the project root.
- **`.env` is optional.** `Settings.load` reads `EVALBUILDER_JUDGE_MODEL`,
  `EVALBUILDER_AGENT_MODEL`, `EVALBUILDER_GENERATOR_MODEL`, `LANGSMITH_*` from the
  environment first, then `.env` (or `--env-file`). Provider API keys are read by the
  providers registry straight from the environment.
- **Model specs** are `provider:model[@effort]` everywhere (`--model`, `evaluators.yaml`
  `model:`, pipeline `models.*`). See [03-core-modules.md §Providers](03-core-modules.md#providers-and-model-specs-providerspy-claude_clipy-configpy).
- **Interactive artifacts** default to `eval/` (`eval/agent-map.json`,
  `eval/datasets/<name>.json`, `eval/results/run-<id>.json`, `score-report-<id>.json`,
  `simulation-<id>.json`); **pipeline artifacts** follow the naming convention in
  `pipeline/layout.py` under `eval/pipeline/<name>/`. `eval/` is git-ignored.

## Reasoning behind the CLI shape

- *One command per decision*: `review` only flips `review.status` and records a note,
  so approval is a separate, auditable act; `mock set` only touches one tool's rule
  list; `agent-map update` replaces whole sections rather than patching entries —
  simpler to validate and to reason about ("this list is what the user confirmed").
- *Two phases, one artifact*: `run` writes a run artifact (outputs, trajectory, tool
  calls, node path, errors) and `score` reads it, so re-scoring with different
  evaluators or a different judge never re-executes the agent — the most expensive step.
- *JSON out, prose in*: machine-readable stdout keeps skills honest (they read the
  shapes the prose describes) and makes the commands composable from the pipeline,
  the UI and tests alike.
