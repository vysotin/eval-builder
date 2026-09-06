# Expected limitations

Grouped by area, with the workaround where one exists. "By design" marks a
deliberate trade-off (see [06-decisions.md](06-decisions.md)); the rest is scope not
yet built.

## Framework and target

| limitation | notes / workaround |
|---|---|
| LangGraph only | Discovery, mocking and execution assume LangGraph graphs; the dataset format and evaluators are framework-neutral. The ADK/Dify adapters described in the 2026-08-23 spec §10 are not implemented. |
| Target contract required | The module must expose `TOOLS` and `build_agent(model=None, tools=None)`; agents without that seam need a ~10-line adapter. |
| Input shape | The graph must accept `{"messages": [...]}`; state beyond `messages` is neither injected nor captured; `outputs.response` is the last message's text. |
| Multi-turn replay | Follow-up turns re-invoke the graph with the accumulated message list; checkpointer-thread state is not preserved between turns. |
| Per-request timeout only | Against a deployed agent every turn has the HTTP timeout (`inference.timeout`, default 120 s); an in-process call (`infer` without an endpoint, `run_dataset`) has none. Cases, scenarios and scoring splits parallelize with joblib (`inference.workers`, `evaluation.workers`; 1 = sequential; `backend: processes` for CPU-bound in-process agents). |

## Deployment

| limitation | notes / workaround |
|---|---|
| `claude-cli` models cannot run in the container | The `claude` binary and its login are not in the image: under `docker` / `kubernetes` / `openshift` the agent model and the mock model (`models.mock`, or the generator under `on_miss: llm`) must be an API-key provider (key passed through `deploy.env: {ANTHROPIC_API_KEY: null}`) or a `scripted:` model — `problems()` refuses the config otherwise. The generator and the judges run on the host and may still use `claude-cli`. |
| Generated Dockerfile assumes the evalbuilder checkout | The build context must contain `pyproject.toml`, `README.md`, `src/` and the target's top-level package (`deploy.build.include` adds paths, `requirements` a pip file); a target living in another repository brings its own `deploy.build.dockerfile` (used verbatim) that installs evalbuilder and runs `evalbuilder serve`. A `.dockerignore` keeps `.venv`, `.git`, `eval/` and the example artifacts out of the context. |
| `load` needs privileged pods | The image loader is a `hostPID`, privileged DaemonSet (what `kind load` does through the Kubernetes API); clusters that forbid it (OpenShift by default) need `image.registry` + `push: registry`. `push: none` is for clusters that already share the image store (Docker Desktop's kubeadm provisioner). |
| A port-forward is one tunnel | `expose: port-forward` runs a single `kubectl port-forward` process to the Service — `deploy status` / `infer` restart it when it died, but it is not load-balanced over replicas and it lives only as long as the host process tree. `nodeport` needs the node's InternalIP to be routable from the host (it is not on Docker Desktop's kind cluster). |
| OpenShift is a prototype | The `oc` path (registry push, Route endpoint, `oc whoami`) is exercised against a fake command runner only; no live cluster was available. |
| Docker target binds the container port on the host | `deploy.port` is mapped to the same host port (`8080:8080`); a busy port fails `compose up`. Change `deploy.port`. |
| Secrets in rendered files | `deploy.env` values — including keys copied from the host — are written in plain text into `work/deploy/compose.yaml` / `manifests.yaml` and appear in the Dockerfile-less command log; `work/` is scratch and git-ignored under `eval/`, but treat the directory as sensitive. |
| Engine history travels per request | With a stateless server the LLM mock engine is rebuilt per turn and seeded with `mocks.history` (tool/args/answer of earlier turns, rebuilt from the ledger) instead of one engine object per conversation; consistency across turns is what the history prompt gives, not shared state. |
| One deployment per output directory | `deployment.json` records one deployment; `deploy up` reuses it while healthy and replaces it otherwise. Two pipelines on the same directory would fight over it. |

## Discovery and schemas

| limitation | notes / workaround |
|---|---|
| AST recognises common patterns only | Graphs built through helpers, subgraphs added as nodes, `Send` fan-out, runtime-built tool lists and templated prompts are partially captured; live introspection adds edges and schemas when the module imports. |
| Node kinds are coarse | `llm` and `graph-node` only; routers are inferred from conditional edges. |
| Edge cases from top-level arguments only | Constraints inside nested models are not expanded into edges; add a top-level typed argument or author the case by hand. |
| JSON-schema subset | No `format` semantics, `if/then`, `dependentRequired`, `uniqueItems`, external `$ref`; `example()` cannot satisfy arbitrary regex patterns (falls back to a plain string). |
| Side effects by docstring | `side_effecting` is a regex over the docstring ("side-effect", "only call after", "after the user confirms"); undocumented side effects are not detected. |
| Skills by convention | Skill folders are found through `load_skills(...)`, `SKILLS_DIR`-style constants, `skills=` keywords and the live `SKILLS` attribute; a folder computed at runtime is missed. Composed prompts are rendered for `+`, f-strings, conditionals, string methods and the skill helpers only. Skill *execution* (`scripts/`) is out of scope: the map records the scripts but never runs them. |

## Mocking

| limitation | notes / workaround |
|---|---|
| Static responses | Arg-dependent or sequence-dependent responses are expressed as several rules; no call counting. By design. |
| Tool calls only | Nodes that call external services directly are not intercepted — route external calls through tools. By design (decision 6). |
| Wrapped tools drop tool-level settings | `handle_tool_error` and similar attributes are not carried onto the mock wrapper (mocks never raise, so this matters only for the `real` miss policy). |
| LLM mocks are not deterministic | Under `on_miss: llm` the engine's answers vary between repeats; the ledger and `stability.llm_mocked_unstable` attribute the variance, they do not remove it. Consistency is per conversation (a case or a scenario), not per dataset. |
| Engine validation is the output schema only | A tool without a return annotation (`-> dict`) is validated as "an object"; behaviour text is followed by the model, not enforced. Write examples and a `fallback_response` for such tools, or add a return model. |
| Cost | Every unmatched tool call is a mock-model call (plus one repair at most); the report's `mocking.calls` shows how many. |

## Evaluation

| limitation | notes / workaround |
|---|---|
| Judges are binary and uncalibrated | Pass rates assume 0/1 judges; judge variance is only visible across repeats; no inter-judge agreement. Start binary, inspect slices, then set thresholds. |
| Substring references | `contains` is a case-insensitive substring check; `expected_tools` compares argument values exactly after subset selection. |
| Simulation is simple | Plain-prompt simulated user, substring expectations, no tool-call replay between turns; `max_turns` is a truncation. |
| Stability needs repeats | With `repeats: 1` there is no stability signal; classification is binary, no statistics. |
| Coverage is relative to the map | A map that missed an intent produces a complete-looking but narrow dataset; check `agent-map.json` before trusting coverage percentages. |

## Generator and pipeline

| limitation | notes / workaround |
|---|---|
| Structural, not semantic, validation | Well-formed cases can still test the wrong thing; the self-review stage and the human review loop mitigate. |
| Cost and duration | Cases × repeats × judge evaluators; with the Claude CLI adapter a 23-case, 2-repeat, 2-judge run took ~1.5 h. Shrink `total_cases`/`repeats` for a first pass. |
| Usage limits mid-run | Judge calls fail (`ClaudeCLIError … is_error … total_cost_usd 0`) and optional stages fail; `--resume --from score` after the limit resets. |
| Problems persist across resumes | Earlier warnings stay in `problems` (history) even after the stage recovered; read them with the stage status. By design. |
| Batching | Cases are generated in batches of six cells with one re-request; a weak generator can leave gaps, which are reported, not filled silently. |
| Sub-agents as nodes are not mocked separately | Specialist nodes run as part of the graph; only their tools are mocked (by design, decision 6). |
| Legacy configs are rewritten on load | `runs.parallel_intents` / `parallel_scoring` / `parallel_simulations` become `inference.workers` / `evaluation.workers` (the last one is dropped); the rewrite is noted in preflight `problems`, and `to_yaml` writes the new keys. |

## Models and providers

| limitation | notes / workaround |
|---|---|
| Claude CLI adapter | One subprocess per call (~3–10 s per agent call, ~15–20 s per judge), no streaming, history flattened into a transcript; requires the `claude` binary and a logged-in subscription. |
| API providers | Need their LangChain package (`uv sync --extra anthropic|openai|gemini|llm` or `pip install -e '.[llm]'`) and key; `@effort` is ignored for Gemini. |
| `scripted:` models | Regex-scripted; they must be kept consistent with fixtures and edge-case messages when an example changes. |

## LangSmith

| limitation | notes / workaround |
|---|---|
| Datasets only | No experiment upload (`client.evaluate`) and no trace mining of production runs into candidate cases. |
| Publication is approved-only and additive | Rejected or edited cases are not removed remotely; a changed input is a new case id and a new example. |

## UI

| limitation | notes / workaround |
|---|---|
| Local, single-user | No authentication; the only coordination is refusing a second job on a running directory. |
| Form covers common keys | Per-metric thresholds, evaluator options, `stages.skip`, LangSmith settings are edited in the YAML editor. |
| Uploads are read-only | Jobs need a directory; an uploaded artifact set is a project that can only be browsed. |
| One project per browser tab | Comparing two runs means switching the project on Pipeline setup; the setup form is per session (it survives navigation, not a server restart). |
| Config reload is by mtime | When the project's config file changes on disk the form reloads it and unsaved YAML edits are dropped; a file the user pointed the form at is only tracked from then on, not loaded. |
| Log tail and tables | Last 60 log lines on screen (file is complete); no pagination for very large datasets. |

## Testing

| limitation | notes / workaround |
|---|---|
| Browser tests need chromium | `playwright install chromium` (via `uv run` or the activated venv); the suite skips itself otherwise. |
| Python ≥ 3.11.7 | Tested on 3.12 with uv and on 3.11 with a plain `venv` + `pip` (`scripts/cli-smoke.sh` is the no-uv end-to-end check); the `dev` extra exists because older pips cannot install PEP 735 dependency groups. |
| Live runs are not in CI | The committed `docs/examples/*` directories are the fixtures (written before the deploy phase, so they carry no `deployment.json` and name the stage `run`); a real model run is manual and slow. |
| Docker / Kubernetes tests are opt-in | `tests/test_deploy_integration.py` (marker `docker`, excluded by the default `addopts`) builds the weather-bot image and deploys it for real; run `pytest -m docker` on a machine with the daemon and a `kubectl` context. Everything else exercises the targets against a fake command runner and the `local` target for real. |
| Streamlit internals | The browser tests encode Streamlit 1.62 DOM details (react-aria comboboxes, canvas dataframes); a Streamlit upgrade may need locator updates. |
