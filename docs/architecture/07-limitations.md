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
| No per-case timeout | Long or hung agent calls block their intent group; cases within one intent run serially (`runs.parallel_intents` controls the run-stage thread pool), while scoring and simulations parallelize per case run / per scenario (`runs.parallel_scoring`, `runs.parallel_simulations`; 1 = fully sequential everywhere). |

## Discovery and schemas

| limitation | notes / workaround |
|---|---|
| AST recognises common patterns only | Graphs built through helpers, subgraphs added as nodes, `Send` fan-out, runtime-built tool lists and templated prompts are partially captured; live introspection adds edges and schemas when the module imports. |
| Node kinds are coarse | `llm` and `graph-node` only; routers are inferred from conditional edges. |
| Edge cases from top-level arguments only | Constraints inside nested models are not expanded into edges; add a top-level typed argument or author the case by hand. |
| JSON-schema subset | No `format` semantics, `if/then`, `dependentRequired`, `uniqueItems`, external `$ref`; `example()` cannot satisfy arbitrary regex patterns (falls back to a plain string). |
| Side effects by docstring | `side_effecting` is a regex over the docstring ("side-effect", "only call after", "after the user confirms"); undocumented side effects are not detected. |

## Mocking

| limitation | notes / workaround |
|---|---|
| Static responses | Arg-dependent or sequence-dependent responses are expressed as several rules; no call counting. By design. |
| Tool calls only | Nodes that call external services directly are not intercepted — route external calls through tools. By design (decision 6). |
| Wrapped tools drop tool-level settings | `handle_tool_error` and similar attributes are not carried onto the mock wrapper (mocks never raise, so this matters only for the `real` miss policy). |

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
| Live runs are not in CI | The committed `docs/examples/*` directories are the fixtures; a real model run is manual and slow. |
| Streamlit internals | The browser tests encode Streamlit 1.62 DOM details (react-aria comboboxes, canvas dataframes); a Streamlit upgrade may need locator updates. |
