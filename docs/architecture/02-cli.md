# The `evalbuilder` CLI

`src/evalbuilder/cli.py` is a typer application. Every command prints a JSON result on
stdout (`_emit`) so a skill can read it back, prints human messages on stderr, and uses
exit codes: `0` success, `1` a validation or runtime problem the caller must act on,
`2` an unusable config (for `deploy …` also an unknown `--target` or a config whose
`problems()` are not empty). Commands never write an artifact that fails validation
(`_save_valid` validates the whole dataset before saving) — the CLI is the only writer
of artifact JSON.

The evaluation is three phases with a command each — **deploy** (`evalbuilder deploy`),
**infer** (`evalbuilder infer`) and **eval** (`evalbuilder eval`) — plus `evalbuilder
serve`, the agent server the deployment targets run. `run` and `score` are the
pre-phase names and stay as aliases (`run` is hidden from `--help`).

## Command reference

| command | what it does | module | exit 1 when |
|---|---|---|---|
| `check [--target-module M] [--env-file F]` | capability matrix `{ready, degraded, blocking, capabilities, providers, models}` | `config.capability_check`, `providers.providers_status` | – |
| `discover MODULE [--source F] [--eval-dir D]` | AST + live introspection → `eval/agent-map.json`; prints tools, skills, nodes, live graph, `decisions_needed` | `discover.discover_from_source`, `discover.discover_live` | module source cannot be located |
| `agent-map update PATH [--intents J] [--scenarios J] [--failures J] [--topics J] [--constraints J]` | replace sections of the map after shape checks (`id` + `evidence` per entry; `failure_type` + `evidence` for failures; string lists for topics/constraints) | `schemas.AgentMap` | any entry lacks a required key |
| `dataset init PATH --name N [--type final_response\|trajectory] --target MODULE[:FACTORY]` | empty dataset with the target contract recorded | `schemas.Dataset` | – |
| `dataset add PATH --case JSON\|@file` | normalise one case (defaults for the coverage cell, content-hash id, review reset to `pending`) and append | `artifacts.add_case` | duplicate id, invalid dataset |
| `dataset import PATH --from FILE` | import foreign shapes (`cases`/`examples`/`goldens` keys, `outputs`→`reference_outputs`), all `pending` with `source=import` | `artifacts.import_cases` | invalid dataset |
| `dataset validate PATH` | schema + review-status + mock-block validation | `artifacts.validate_dataset` | invalid |
| `dataset list PATH [--status S]` | rows of id/status/cell/source | – | – |
| `dataset gaps PATH --agent-map M [--target-per-cell N]` | required cells vs present cases | `coverage.coverage_gaps` | – |
| `review PATH --approve IDS \| --reject IDS [--note …]` | record a human decision on specific case ids (exactly one of approve/reject) | `artifacts.set_review` | unknown ids, both/neither flag |
| `mock set PATH --tool T --rules JSON [--case ID]` | set the ordered rule list (dataset-level or per-case) | dataset `mocks` / `metadata.mocks` | unknown case, invalid rules |
| `mock verify PATH` | every `expected_tools` call in mocked cases matches a rule; under `on_miss: llm` misses are listed as `llm_answered` (informational) | `mocking.verify_summary` | any miss (non-llm policies) |
| `mock strategies PATH [--set JSON\|@file] [--model SPEC] [--on-miss P] [--strategy S] [--on-invalid P] [--max-repairs N] [--case ID --strategy S]` | show / set the LLM mock layer: strategies (validated against the tools' schemas), model, policy; or one case's strategy | `mock_engine.validate_strategies`, dataset `mocks` | invalid strategies, unknown case |
| `mock validate PATH --tool T --response JSON\|@file` | validate a response against the tool's declared output schema (the engine's validator) | `tool_schemas.validate` | non-conforming, unknown tool |
| `mock try PATH --tool T --args JSON [--strategy S] [--mock-model SPEC] [--on-miss P] [--case ID]` | answer one call through both layers; prints the layer, response, validation, repairs | `mocking.wrap_tools`, `runner.build_engine` | engine error |
| `serve --module M [--factory F] [--host H] [--port P] [--agent-model SPEC] [--mock-model SPEC] [--quiet]` | the agent server: `GET /health`, `POST /invoke` (the target with both mock layers installed per request); the entry point of the agent image and of the `local` target; model specs default to `EVALBUILDER_AGENT_MODEL` / `EVALBUILDER_MOCK_MODEL` | `serve.serve`, `agent_client.LocalAgent` | (blocking; a broken module / factory / mock model fails at start-up) |
| `deploy up CONFIG [--target local\|docker\|kubernetes\|openshift] [--env-file F] [--quiet]` | build (if needed) and deploy the agent on the config's target, wait for `/health`, write `deployment.json`; an already-running healthy deployment is reused | `deploy.deploy_up` | deployment failed or never became healthy (the failed record is written first); exit 2 on a bad config / target |
| `deploy status DIR\|CONFIG` | live status of the recorded deployment: `ready`, endpoint, health, replicas, details (a dead port-forward is restarted) | `deploy.deploy_status` | not `up` (down, failed, unhealthy, or no record) |
| `deploy down DIR\|CONFIG` | tear the recorded deployment down; the record becomes `down` (idempotent, `none` without a record) | `deploy.deploy_down` | – |
| `deploy build CONFIG [--target T]` | build the agent image only (`null` for the local target) | `deploy.deploy_build` | `docker build` failed |
| `deploy render CONFIG [--target T] [--write]` | print (and with `--write` save under `work/deploy/`) the generated Dockerfile / compose file / manifests / loader without running anything | `deploy.deploy_render` | exit 2 on a bad config |
| `deploy logs DIR\|CONFIG [--lines N]` | the deployed agent's recent log lines (plain text, not JSON) | `deploy.deploy_logs` | – |
| `infer PATH [--endpoint URL \| --deployment DIR] [--scenarios F] [--repeats N] [--workers N] [--backend threads\|processes] [--timeout S] [--out D] [--ids …] [--mock/--no-mock] [--model SPEC] [--on-miss real\|fallback\|strict\|llm] [--mock-model SPEC] [--strategy S] [--mine/--no-mine]` | the inference phase: approved cases (and, with `--scenarios`, the multi-turn simulations) against a deployed agent (`--endpoint`, or the endpoint in `DIR/deployment.json`) or in-process (default) — joblib-parallel; writes one `run-<id>.json` per repeat (`execution`, per-case `log` and `mock_calls`) and `simulation-<id>.json`; mocks default on; `--model` is ignored against a deployed agent | `inference.infer_dataset`, `inference.simulate_scenarios`, `agent_client.*` | pending/rejected ids, no approved cases, `llm` without a mock model, unhealthy endpoint, no deployment record, bad `--backend` |
| `run …` | hidden alias of `infer` (identical flags and output; the pre-phase name) | as `infer` | as `infer` |
| `eval RUN… --dataset PATH --evaluators evaluators.yaml [--out D] [--workers N] [--backend B] [--aggregate/--no-aggregate] [--config CFG] [--env-file F]` | the evaluation phase: score every run artifact given (case-run splits scored concurrently by joblib) → `score-report-<id>.json` each, then `aggregate.json` (default when more than one run is given; thresholds and judge from `--config`, else defaults / `.env`) | `evaluators.score_run`, `pipeline.aggregate.aggregate` | no evaluators |
| `score RUN --dataset PATH --evaluators evaluators.yaml [--out D] [--workers N] [--backend B]` | score one run and print its report (kept for the interactive skills; `eval` is the multi-run form) | `evaluators.score_run` | no evaluators |
| `simulate PATH --scenarios F [--out D] [--mine/--no-mine] [--mock] [--on-miss P] [--mock-model SPEC]` | multi-turn simulations in-process (`infer --scenarios` runs them against a deployment); violations mined as pending cases | `inference.simulate_scenarios` | invalid scenario file |
| `publish PATH [--dataset-name N] [--env-file F]` | LangSmith publish of approved cases, read-back verified | `langsmith_io.publish_approved` | no key, no approved cases, read-back mismatch |
| `pipeline init PATH --name N --source F --module M [--force]` | commented starter config | `pipeline.config.template` | file exists |
| `pipeline run CONFIG [--resume] [--from STAGE] [--until STAGE] [--quiet] [--env-file F]` | the autonomous pipeline; exit 1 unless verdict `pass` (exit 0 when stopped deliberately with `--until`; `--until deploy` leaves the agent running for manual probing) | `pipeline.report.run_pipeline` | verdict ≠ pass; exit 2 on a bad config |
| `pipeline compact DIR [--dry-run]` | bring an older output directory onto the current artifact layout (fold, move scratch, re-index) | `pipeline.layout.compact_dir` | not a directory |
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
  `simulation-<id>.json`, `aggregate.json`); **pipeline artifacts** follow the naming
  convention in `pipeline/layout.py` under `eval/pipeline/<name>/` — `deployment.json`
  at the root, the rendered deployment files and `commands.log` under `work/deploy/`.
  `eval/` is git-ignored.
- **`deploy` commands take a directory or a config.** `deploy status|down|logs` accept
  the output directory (existing or not) or a `.yaml`/`.yml`/`.json` config; `deploy
  up|build|render` need the config. `--target` overrides `deploy.target` for one
  invocation and the config is re-checked (`problems()`) with it.

## Reasoning behind the CLI shape

- *One command per decision*: `review` only flips `review.status` and records a note,
  so approval is a separate, auditable act; `mock set` only touches one tool's rule
  list; `agent-map update` replaces whole sections rather than patching entries —
  simpler to validate and to reason about ("this list is what the user confirmed").
- *Three phases, one artifact between each*: `deploy` leaves `deployment.json` (where
  the agent runs), `infer` reads it and writes run artifacts (outputs, trajectory, tool
  calls, node path, per-turn log, mock ledger, `execution`), `eval` reads those and
  writes score reports and the aggregate. Re-scoring with different evaluators or a
  different judge never re-executes the agent, and re-running the agent never rebuilds
  the deployment while it is up and healthy (`deploy up` reuses it).
- *JSON out, prose in*: machine-readable stdout keeps skills honest (they read the
  shapes the prose describes) and makes the commands composable from the pipeline,
  the UI and tests alike.
