# eval-builder MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `evalbuilder` Python CLI + four Claude Code skills that discover a LangGraph agent's test surface, generate a coverage-driven OpenEvals-format dataset, mock tools ADK-style, and run/score experiments locally with optional LangSmith publication.

**Architecture:** Skills are policy prose that invoke a deterministic typer CLI (`evalbuilder`). All artifacts are versioned JSON on the local FS, written only by the CLI. The runner rebuilds the agent per case via a `build_agent(model=None, tools=None)` factory, installing mock tools from dataset rules; scoring is a separate phase using OpenEvals/agentevals evaluators.

**Tech Stack:** Python 3.12+ via uv; langgraph, langchain-core, openevals, agentevals, langsmith, typer, pydantic v2, python-dotenv, pyyaml, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-eval-builder-skills-design.md`

## Global Constraints

- Python 3.12+; project managed with `uv`; package layout `src/evalbuilder/`; CLI entry point `evalbuilder`.
- All tests run **offline** (no network, no API keys). LLM judges and LangSmith are tested with fakes/monkeypatches.
- Artifacts carry `schema` version strings: `evalbuilder/agent-map/v1`, `evalbuilder/dataset/v1`, `evalbuilder/run/v1`, `evalbuilder/report/v1`.
- New/imported cases are always `review.status="pending"` — `normalize_case` force-resets it; `run`/`publish` refuse pending cases (run refuses unless `--allow-pending` is never offered; publish approved-only).
- Case IDs: `"case-" + sha256(json of {inputs, intent, topic, scenario, failure_mode})[:10]`.
- Mock matching: per-tool ordered rule list, first match wins, `matchArgs` subset-equality, `{}` wildcard; miss policy `real | fallback | strict`.
- One metric per evaluator; deterministic evaluators never call an LLM.
- JSON written with `indent=2, ensure_ascii=False` + trailing newline.
- Commit after every task with a conventional message ending in the Claude-Session trailer used earlier in this repo.

## File Structure (final)

```
pyproject.toml
.env.example
README.md
src/evalbuilder/{__init__,schemas,artifacts,coverage,discover,target,mocking,runner,evaluators,simulate,langsmith_io,config,testing,cli}.py
skills/agent-eval-{discover,dataset,mock,run}/SKILL.md (+ references/)
.claude/skills/agent-eval-* -> ../../skills/agent-eval-*
examples/weather_bot/{__init__.py,agent.py}
examples/travel_planner/{__init__.py,agent.py}
tests/test_{config,artifacts,coverage,examples,discover,mocking,runner,evaluators,simulate,langsmith_io,e2e_pipeline}.py
```

---

### Task 1: Scaffolding, Settings, capability check

**Files:**
- Create: `pyproject.toml`, `.env.example`, `.gitignore`, `src/evalbuilder/__init__.py`, `src/evalbuilder/config.py`, `src/evalbuilder/cli.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` (frozen dataclass: `judge_model: str`, `langsmith_api_key: str|None`, `langsmith_endpoint: str|None`, `langsmith_project: str|None`, classmethod `load(env_file: Path|None = None) -> Settings`); `capability_check(settings, target_module: str|None = None) -> dict` returning `{"ready": bool, "degraded": [str], "blocking": [{"issue": str, "fix": str}], "capabilities": {...}}`; typer app `app` in `cli.py` with command `check`.

- [ ] **Step 1: Write pyproject and scaffolding**

`pyproject.toml`:

```toml
[project]
name = "evalbuilder"
version = "0.1.0"
description = "Agentic eval/dataset builder for LangGraph agents"
requires-python = ">=3.12"
dependencies = [
  "langgraph>=0.2",
  "langchain-core>=0.3",
  "langchain>=0.3",
  "openevals>=0.0.10",
  "agentevals>=0.0.5",
  "langsmith>=0.3",
  "typer>=0.12",
  "pydantic>=2.7",
  "python-dotenv>=1.0",
  "pyyaml>=6.0",
]

[project.scripts]
evalbuilder = "evalbuilder.cli:app"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/evalbuilder"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`.gitignore`: `.venv/`, `__pycache__/`, `*.egg-info/`, `.env`, `eval/`, `.pytest_cache/`.
`.env.example`: the vars from spec §9 with comments, `EVALBUILDER_JUDGE_MODEL=anthropic:claude-sonnet-5`.

Run: `uv venv --python 3.12 && uv sync && uv pip install -e .`

- [ ] **Step 2: Write failing tests**

```python
# tests/test_config.py
import json
from pathlib import Path
from typer.testing import CliRunner
from evalbuilder.config import Settings, capability_check
from evalbuilder.cli import app

def test_settings_load_from_env_file(tmp_path, monkeypatch):
    for var in ("LANGSMITH_API_KEY", "EVALBUILDER_JUDGE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text("LANGSMITH_API_KEY=key123\nEVALBUILDER_JUDGE_MODEL=openai:gpt-test\n")
    s = Settings.load(env)
    assert s.langsmith_api_key == "key123"
    assert s.judge_model == "openai:gpt-test"

def test_capability_check_degraded_without_langsmith(monkeypatch):
    for var in ("LANGSMITH_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.load(env_file=None)
    report = capability_check(s, target_module="examples.weather_bot.agent")
    assert report["ready"] is True          # local-only is still ready
    assert "langsmith" in report["degraded"]
    assert "judge" in report["degraded"]

def test_capability_check_blocking_on_bad_module():
    s = Settings.load(env_file=None)
    report = capability_check(s, target_module="no.such.module")
    assert report["ready"] is False
    assert report["blocking"] and "fix" in report["blocking"][0]

def test_cli_check_outputs_json(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["check"])
    assert result.exit_code == 0
    assert "ready" in json.loads(result.stdout)
```

- [ ] **Step 3: Run tests, expect import failures** — `uv run pytest tests/test_config.py -v`

- [ ] **Step 4: Implement**

```python
# src/evalbuilder/config.py
from __future__ import annotations
import importlib.util, os
from dataclasses import dataclass
from pathlib import Path
from dotenv import dotenv_values

DEFAULT_JUDGE_MODEL = "anthropic:claude-sonnet-5"

@dataclass(frozen=True)
class Settings:
    judge_model: str = DEFAULT_JUDGE_MODEL
    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    langsmith_project: str | None = None

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        values: dict[str, str | None] = {}
        if env_file is None and Path(".env").exists():
            env_file = Path(".env")
        if env_file is not None:
            values.update(dotenv_values(env_file))
        get = lambda k: os.environ.get(k) or values.get(k)
        return cls(
            judge_model=get("EVALBUILDER_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
            langsmith_api_key=get("LANGSMITH_API_KEY"),
            langsmith_endpoint=get("LANGSMITH_ENDPOINT"),
            langsmith_project=get("LANGSMITH_PROJECT"),
        )

def _has_judge_key(model: str) -> bool:
    provider = model.split(":", 1)[0]
    keys = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
            "google_genai": "GOOGLE_API_KEY"}
    var = keys.get(provider)
    return bool(var and os.environ.get(var))

def capability_check(settings: Settings, target_module: str | None = None) -> dict:
    degraded: list[str] = []
    blocking: list[dict] = []
    caps: dict[str, bool] = {}
    caps["langgraph"] = importlib.util.find_spec("langgraph") is not None
    if not caps["langgraph"]:
        blocking.append({"issue": "langgraph is not installed",
                         "fix": "uv pip install langgraph"})
    caps["judge"] = _has_judge_key(settings.judge_model)
    if not caps["judge"]:
        degraded.append("judge")
    caps["langsmith"] = bool(settings.langsmith_api_key)
    if not caps["langsmith"]:
        degraded.append("langsmith")
    if target_module:
        caps["target"] = importlib.util.find_spec(target_module) is not None
        if not caps["target"]:
            blocking.append({"issue": f"target module '{target_module}' not importable",
                             "fix": "check the module path and install its dependencies"})
    return {"ready": not blocking, "degraded": degraded,
            "blocking": blocking, "capabilities": caps}
```

```python
# src/evalbuilder/cli.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import typer
from evalbuilder.config import Settings, capability_check

app = typer.Typer(help="Build and run evals for LangGraph agents.", no_args_is_help=True)

def _emit(data) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))

@app.command()
def check(target_module: Optional[str] = typer.Option(None, "--target-module"),
          env_file: Optional[Path] = typer.Option(None, "--env-file")) -> None:
    """Print the capability matrix as JSON."""
    _emit(capability_check(Settings.load(env_file), target_module))
```

- [ ] **Step 5: Run tests, expect PASS** — `uv run pytest tests/test_config.py -v`
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: scaffolding, settings, capability check"`

---

### Task 2: Dataset schemas + artifacts + dataset/review CLI

**Files:**
- Create: `src/evalbuilder/schemas.py`, `src/evalbuilder/artifacts.py`
- Modify: `src/evalbuilder/cli.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Produces (`schemas.py`, all pydantic `BaseModel` with `model_config = ConfigDict(extra="allow")` on Case/Dataset):
  - `MockRule(matchArgs: dict = {}, response: Any = None)`
  - `Review(status: Literal["pending","approved","rejected"] = "pending", note: str = "")`
  - `Publication(langsmith_example_id: str | None = None)`
  - `Case(id: str = "", inputs: dict, reference_outputs: dict = {}, metadata: dict = {}, review: Review, publication: Publication)`
  - `Target(framework: str = "langgraph", module: str, factory: str = "build_agent")`
  - `Dataset(schema_: str = Field("evalbuilder/dataset/v1", alias="schema"), name: str, dataset_type: Literal["final_response","trajectory"], target: Target, mocks: dict = {}, cases: list[Case] = [], langsmith: dict = {"dataset_id": None, "dataset_name": None})` — use `populate_by_name=True`, serialize by alias.
- Produces (`artifacts.py`): `save_json(path, model_or_dict)`, `load_dataset(path) -> Dataset`, `case_id(case: dict) -> str`, `normalize_case(raw: dict) -> Case` (fills id, forces `review.status="pending"` on new content, defaults metadata keys intent/topic/scenario/failure_mode to `"unspecified"`/`"none"`), `add_case(ds, raw) -> Case` (rejects duplicate id), `validate_dataset(ds) -> list[str]` (error strings; empty = valid), `import_cases(ds, payload: dict|list, source_path: str) -> int` (accepts `cases`/`examples`/`goldens` keys or bare list; maps `outputs`→`reference_outputs`, `additional_metadata`→`metadata`; sets `metadata.source="import"`, `metadata.source_path`), `set_review(ds, ids, status, note) -> int`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_artifacts.py
import json, pytest
from evalbuilder.schemas import Dataset, Target
from evalbuilder.artifacts import (add_case, case_id, import_cases, load_dataset,
                                   normalize_case, save_json, set_review, validate_dataset)

def _ds():
    return Dataset(name="d", dataset_type="final_response",
                   target=Target(module="examples.weather_bot.agent"))

def test_normalize_forces_pending_and_hash_id():
    raw = {"inputs": {"messages": [{"role": "user", "content": "hi"}]},
           "metadata": {"intent": "intent.x"},
           "review": {"status": "approved", "note": "sneaky"}}
    case = normalize_case(raw)
    assert case.review.status == "pending"
    assert case.id.startswith("case-") and len(case.id) == 15
    assert case.metadata["failure_mode"] == "none"
    assert case.metadata["topic"] == "unspecified"

def test_same_inputs_different_cell_is_different_case():
    a = normalize_case({"inputs": {"q": 1}, "metadata": {"intent": "a"}})
    b = normalize_case({"inputs": {"q": 1}, "metadata": {"intent": "b"}})
    assert a.id != b.id

def test_add_case_rejects_duplicates():
    ds = _ds()
    add_case(ds, {"inputs": {"q": 1}})
    with pytest.raises(ValueError, match="duplicate"):
        add_case(ds, {"inputs": {"q": 1}})

def test_validate_reports_errors():
    ds = _ds()
    add_case(ds, {"inputs": {"q": 1}})
    ds.cases[0].review.status = "bogus"
    errs = validate_dataset(ds)
    assert any("review" in e for e in errs)

def test_roundtrip_serializes_schema_alias(tmp_path):
    ds = _ds()
    add_case(ds, {"inputs": {"q": 1}})
    p = tmp_path / "ds.json"
    save_json(p, ds)
    data = json.loads(p.read_text())
    assert data["schema"] == "evalbuilder/dataset/v1"
    ds2 = load_dataset(p)
    assert ds2.cases[0].id == ds.cases[0].id

def test_import_maps_foreign_fields():
    ds = _ds()
    n = import_cases(ds, {"examples": [{"inputs": {"q": 2}, "outputs": {"a": 3},
                                        "additional_metadata": {"intent": "i"}}]}, "f.json")
    assert n == 1
    c = ds.cases[0]
    assert c.reference_outputs == {"a": 3}
    assert c.metadata["source"] == "import" and c.metadata["intent"] == "i"
    assert c.review.status == "pending"

def test_set_review():
    ds = _ds()
    c = add_case(ds, {"inputs": {"q": 1}})
    assert set_review(ds, [c.id], "approved", "ok by user") == 1
    assert ds.cases[0].review.status == "approved"
```

- [ ] **Step 2: Run, expect fail** — `uv run pytest tests/test_artifacts.py -v`
- [ ] **Step 3: Implement `schemas.py` and `artifacts.py`**

Key implementation notes (write full modules):

```python
# artifacts.py core pieces
import hashlib, json
from pathlib import Path
from evalbuilder.schemas import Case, Dataset

def save_json(path: Path, obj) -> None:
    data = obj.model_dump(by_alias=True) if hasattr(obj, "model_dump") else obj
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

def case_id(raw: dict) -> str:
    md = raw.get("metadata", {})
    identity = {"inputs": raw.get("inputs"),
                "intent": md.get("intent", "unspecified"),
                "topic": md.get("topic", "unspecified"),
                "scenario": md.get("scenario", "unspecified"),
                "failure_mode": md.get("failure_mode", "none")}
    material = json.dumps(identity, sort_keys=True, ensure_ascii=False)
    return "case-" + hashlib.sha256(material.encode()).hexdigest()[:10]

def normalize_case(raw: dict) -> Case:
    raw = dict(raw)
    md = dict(raw.get("metadata") or {})
    md.setdefault("intent", "unspecified"); md.setdefault("topic", "unspecified")
    md.setdefault("scenario", "unspecified"); md.setdefault("failure_mode", "none")
    md.setdefault("source", "synthetic")
    raw["metadata"] = md
    raw["id"] = raw.get("id") or case_id(raw)
    raw["review"] = {"status": "pending",
                     "note": "New/generated/imported content can never grant itself approval."}
    raw.setdefault("publication", {})
    return Case.model_validate(raw)
```

`validate_dataset` checks: schema string equals `evalbuilder/dataset/v1`, dataset_type membership, unique non-empty ids, `inputs` is a non-empty dict, review status in the enum, mock rules shape (each rule an object with dict `matchArgs`). `import_cases` maps per the Interfaces block. `set_review` validates status and returns matched count, raising on unknown ids.

- [ ] **Step 4: Add CLI commands** (in `cli.py`; each loads → mutates via artifacts fns → validates → saves, printing JSON summaries):

```
evalbuilder dataset init PATH --name N --type final_response --target MODULE[:FACTORY]
evalbuilder dataset add PATH --case JSON        # JSON string or @file.json
evalbuilder dataset import PATH --from FILE
evalbuilder dataset validate PATH               # exit 1 + errors JSON when invalid
evalbuilder dataset list PATH [--status pending]
evalbuilder review PATH --approve "id1,id2" | --reject "id1" [--note TEXT]
```

Use a `dataset_app = typer.Typer()` subapp; `--case` supports `@path` file reference. `dataset add` must run `validate_dataset` after adding and refuse (exit 1, no write) if invalid.

- [ ] **Step 5: Add CLI tests to `tests/test_artifacts.py`** (CliRunner: init → add → validate → review approve → list filters) and run full file: `uv run pytest tests/test_artifacts.py -v` → PASS
- [ ] **Step 6: Commit** — `"feat: dataset schema, artifacts, dataset/review CLI"`

---

### Task 3: Agent map schema + coverage math + CLI

**Files:**
- Create: `src/evalbuilder/coverage.py`; extend `src/evalbuilder/schemas.py`
- Modify: `src/evalbuilder/cli.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Produces (`schemas.py`): `AgentMap(schema_: alias "schema" = "evalbuilder/agent-map/v1", framework: str = "langgraph", source_sha256: str = "", app: dict = {}, graph: dict = {"nodes": [], "edges": [], "conditional_edges": []}, tools: list[dict] = [], constraints: list[str] = [], data_domains: dict = {"topics": [], "sources": []}, intents: list[dict] = [], scenarios: list[dict] = [], failure_scenarios: list[dict] = [], decisions_needed: list[str] = [])`
- Produces (`coverage.py`): `required_cells(agent_map: AgentMap) -> list[dict]` (each `{intent, topic, scenario, failure_mode}`: one cell per scenario × (topics or ["unspecified"]) with failure_mode "none", plus one per failure_scenario with intent="cross-cutting", scenario="failure", topic="unspecified", failure_mode=failure_type); `coverage_gaps(ds: Dataset, agent_map: AgentMap, target_per_cell: int = 1) -> dict` returning `{"required_cells": int, "covered_cells": int, "target_per_cell": int, "gaps": [cell + {"missing": int}]}`.
- Produces (CLI): `evalbuilder discover-init` is NOT here (Task 5); here: `evalbuilder agent-map update PATH --intents JSON --scenarios JSON --failures JSON --topics JSON --constraints JSON` (each optional, `@file` supported; every provided list fully replaces the section after shape validation: intents/scenarios need `id`+`evidence` non-empty, failures need `failure_type`+`evidence`), and `evalbuilder dataset gaps DS --agent-map MAP [--target-per-cell 1]`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_coverage.py
from evalbuilder.schemas import AgentMap, Dataset, Target
from evalbuilder.coverage import required_cells, coverage_gaps
from evalbuilder.artifacts import add_case

def _map():
    return AgentMap(
        intents=[{"id": "intent.a", "status": "hypothesis", "evidence": ["prompt:n1"]}],
        scenarios=[{"id": "scenario.a.happy", "intent": "intent.a",
                    "status": "hypothesis", "evidence": ["prompt:n1"]}],
        failure_scenarios=[{"failure_type": "input_validation",
                            "rationale": "r", "evidence": ["app:always"]}],
        data_domains={"topics": ["billing", "shipping"], "sources": []})

def test_required_cells_expand_topics_and_failures():
    cells = required_cells(_map())
    assert {"intent": "intent.a", "topic": "billing", "scenario": "scenario.a.happy",
            "failure_mode": "none"} in cells
    assert {"intent": "cross-cutting", "topic": "unspecified", "scenario": "failure",
            "failure_mode": "input_validation"} in cells
    assert len(cells) == 3   # 2 topics x 1 scenario + 1 failure

def test_gaps_count_missing():
    ds = Dataset(name="d", dataset_type="final_response",
                 target=Target(module="m"))
    add_case(ds, {"inputs": {"q": 1},
                  "metadata": {"intent": "intent.a", "topic": "billing",
                                "scenario": "scenario.a.happy"}})
    report = coverage_gaps(ds, _map(), target_per_cell=1)
    assert report["required_cells"] == 3 and report["covered_cells"] == 1
    assert len(report["gaps"]) == 2 and report["gaps"][0]["missing"] == 1
```

- [ ] **Step 2: Run, expect fail** → **Step 3: Implement** `coverage.py` (cell key = the 4-tuple; count cases per key from `case.metadata`; gaps = required minus counted, `missing = max(0, target - have)`), extend `schemas.py`, add the two CLI commands + CliRunner tests for `agent-map update` validation (rejects an intent without evidence, exit 1).
- [ ] **Step 4: Run PASS** — `uv run pytest tests/test_coverage.py -v`
- [ ] **Step 5: Commit** — `"feat: agent map schema, coverage grid, gaps CLI"`

---### Task 4: Scripted chat model + example agents

**Files:**
- Create: `src/evalbuilder/testing.py`, `examples/__init__.py`, `examples/weather_bot/{__init__.py,agent.py}`, `examples/travel_planner/{__init__.py,agent.py}`
- Test: `tests/test_examples.py`

**Interfaces:**
- Produces (`testing.py`): `ScriptedChatModel(BaseChatModel)` — constructor `ScriptedChatModel(script: list[ScriptRule])` where `ScriptRule = tuple[str, Callable[[re.Match, list[BaseMessage]], AIMessage]]`; on `_generate`, if the last message is a `ToolMessage` it first tries rules whose pattern matches `"TOOL:" + tool message content`, else matches the last non-tool human/AI text; first regex match wins; raises `ScriptMissError` listing the unmatched text otherwise. Implements `bind_tools(self, tools, **kwargs)` returning `self`, `_llm_type = "scripted"`. Helper `ai(text)` and `tool_call(name, args, call_id="call_1")` constructors.
- Produces (`examples/weather_bot/agent.py`): `TOOLS: list[BaseTool]` (`get_weather(city: str) -> dict`, `get_alerts(city: str) -> dict` as `@tool` functions with docstrings); `build_agent(model=None, tools=None) -> CompiledStateGraph` using `create_react_agent` (import from `langgraph.prebuilt`, fallback `from langchain.agents import create_agent as create_react_agent`); default model = `default_scripted_model()` emitting: user text matching `weather in (\w+)` → tool_call `get_weather {"city": m[1]}`; `TOOL:.*temp` → final `ai("It is {temp}F and {condition} in ...")` built from the tool JSON.
- Produces (`examples/travel_planner/agent.py`): tools `search_flights(origin, destination, date) -> dict`, `book_flight(flight_id) -> dict`, `search_hotels(city, check_in, check_out) -> dict`, `book_hotel(hotel_id) -> dict`; two subagents built with `create_react_agent` and exposed to the supervisor as tools `flight_agent(request: str) -> str` / `hotel_agent(request: str) -> str` (each invokes the subgraph and returns its final message text); `TOOLS = [flight_agent_tool, hotel_agent_tool]`; supervisor = `create_react_agent(model, tools)` with a system prompt containing the constraint sentence "Never confirm a booking without an explicit user yes."; `build_agent(model=None, tools=None)`. Scripted supervisor model: `flight from (\w+) to (\w+)` → tool_call `flight_agent`; `TOOL:.*AA100` → final listing AA100 at $350 and asking which to book; `book it` → final "Please confirm: book flight AA100? (yes/no)"; scripted subagent model: any text with `flight` → tool_call `search_flights {...}`; `TOOL:.*flights` → final "Found AA100 at $350".

- [ ] **Step 1: Write failing tests**

```python
# tests/test_examples.py
from langchain_core.messages import AIMessage
from examples.weather_bot.agent import build_agent as build_weather
from examples.travel_planner.agent import build_agent as build_travel

def _final_text(state) -> str:
    return state["messages"][-1].content

def test_weather_bot_scripted_happy_path():
    graph = build_weather()
    out = graph.invoke({"messages": [{"role": "user", "content": "What is the weather in Paris?"}]})
    text = _final_text(out)
    assert "Paris" in text and "72" in text
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert tool_msgs and tool_msgs[0].name == "get_weather"

def test_travel_planner_delegates_to_flight_agent():
    graph = build_travel()
    out = graph.invoke({"messages": [{"role": "user",
        "content": "Find me a flight from SFO to JFK on 2026-09-01"}]})
    assert "AA100" in _final_text(out)
    called = [tc["name"] for m in out["messages"] if isinstance(m, AIMessage)
              for tc in (m.tool_calls or [])]
    assert "flight_agent" in called

def test_build_agent_accepts_injected_tools():
    from langchain_core.tools import tool
    @tool
    def get_weather(city: str) -> dict:
        "Mock weather."
        return {"temp": 99, "condition": "scripted-mock"}
    @tool
    def get_alerts(city: str) -> dict:
        "Mock alerts."
        return {"alerts": []}
    graph = build_weather(tools=[get_weather, get_alerts])
    out = graph.invoke({"messages": [{"role": "user", "content": "weather in Oslo?"}]})
    assert "99" in _final_text(out)
```

- [ ] **Step 2: Run, expect fail** → **Step 3: Implement** `testing.py` + both examples. Weather tools return fixed data (`{"temp": 72, "condition": "sunny", "city": city}`). The scripted final-answer factory reads the ToolMessage JSON so injected tools change the answer (proves the factory-injection seam mocking will use). `examples/__init__.py` files make `examples` importable; tests rely on the editable install plus repo-root cwd.
- [ ] **Step 4: Run PASS** — `uv run pytest tests/test_examples.py -v`
- [ ] **Step 5: Commit** — `"feat: scripted chat model and example LangGraph agents"`

---

### Task 5: Discovery (AST + live) + CLI

**Files:**
- Create: `src/evalbuilder/discover.py`
- Modify: `src/evalbuilder/cli.py`
- Test: `tests/test_discover.py`

**Interfaces:**
- Produces: `discover_from_source(source_path: Path) -> AgentMap` — pure AST, never imports: collects `@tool`-decorated functions (name, docstring→description, annotations→`args_schema` json types str/int/float/bool/list/dict, non-default args → required), string constants assigned to `*_PROMPT`/`SYSTEM*` names or passed as `prompt=`/`state_modifier=`/system messages, `StateGraph` `add_node`/`add_edge`/`add_conditional_edges` calls, `create_react_agent(...)` / `create_agent(...)` calls (node kind "llm"); fills `graph`, `tools`, `source_sha256`, and `decisions_needed=["Confirm inferred intents and scenarios", "Provide sample documents or describe data-domain topics", "State constraints the code cannot express"]`. Intents/scenarios stay empty — Claude authors them via `agent-map update`.
- Produces: `discover_live(module: str, factory: str = "build_agent") -> dict` — imports the module, calls the factory, returns `{"nodes": [names], "edges": [[a,b],...]}` from `graph.get_graph()`; caller merges into `AgentMap.graph["live"]`. Import failure returns `{"error": str}` instead of raising.
- Produces (CLI): `evalbuilder discover MODULE [--source FILE] [--eval-dir eval]` → writes `<eval-dir>/agent-map.json` (source defaults to the module's file via `importlib.util.find_spec` origin; when both work, AST result + live merged).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_discover.py
from pathlib import Path
from evalbuilder.discover import discover_from_source, discover_live

WEATHER = Path("examples/weather_bot/agent.py")

def test_ast_finds_tools_with_schemas():
    m = discover_from_source(WEATHER)
    names = {t["name"] for t in m.tools}
    assert {"get_weather", "get_alerts"} <= names
    gw = next(t for t in m.tools if t["name"] == "get_weather")
    assert gw["args_schema"]["properties"]["city"]["type"] == "string"
    assert "city" in gw["args_schema"]["required"]
    assert gw["description"]          # docstring captured

def test_ast_finds_prompt_and_react_node():
    m = discover_from_source(Path("examples/travel_planner/agent.py"))
    assert any("Never confirm a booking" in (n.get("prompt") or "")
               for n in m.graph["nodes"])
    assert m.source_sha256

def test_live_introspection_returns_nodes():
    live = discover_live("examples.weather_bot.agent")
    assert "error" not in live
    assert any("agent" in n or "tools" in n for n in live["nodes"])

def test_live_handles_import_error():
    assert "error" in discover_live("no.such.module")
```

- [ ] **Step 2: Run, expect fail** → **Step 3: Implement** `discover.py` (~180 lines; `ast.walk` passes as described; `_annotation_to_type` maps `str→string, int→integer, float→number, bool→boolean, list→array, dict→object`, unknown → `{}`; skip params named `self/ctx/config/state` or annotations containing `Context`), CLI command + CliRunner test asserting the file is written and `decisions_needed` non-empty.
- [ ] **Step 4: Run PASS** — `uv run pytest tests/test_discover.py -v`
- [ ] **Step 5: Commit** — `"feat: AST + live discovery into agent-map"`

---

### Task 6: Mocking engine + mock CLI

**Files:**
- Create: `src/evalbuilder/mocking.py`
- Modify: `src/evalbuilder/cli.py`
- Test: `tests/test_mocking.py`

**Interfaces:**
- Produces: `match_rule(rules: list[dict], args: dict) -> dict | None` (ordered, subset-equality, `{}` wildcard wins); `MockMissError(Exception)`; `wrap_tool(tool: BaseTool, rules: list[dict], on_miss: str = "real", fallback=None) -> StructuredTool` (returned tool keeps `name`, `description`, `args_schema`; on match returns `rule["response"]`; on miss: `"real"` → `tool.invoke(args)`, `"fallback"` → `fallback`, `"strict"` → raise `MockMissError(tool.name, args, rules)`); `merge_mock_rules(dataset_mocks: dict, case_mocks: dict) -> dict[str, list[dict]]` (per tool name, case rules replace dataset rules); `wrap_tools(tools: list[BaseTool], rules_by_tool: dict, on_miss="real") -> list[BaseTool]` (tools without rules pass through unwrapped).
- Produces (CLI): `evalbuilder mock set DS [--case ID] --tool NAME --rules JSON` (validates each rule has dict `matchArgs` and a `response` key; writes to `dataset.mocks["tools"][NAME]` or `case.metadata["mocks"]["tools"][NAME]`); `evalbuilder mock verify DS` → for every case with mocks, checks each `reference_outputs.expected_tools` entry (when present) matches ≥1 rule for that tool; prints `{"ok": bool, "misses": [...]}`, exit 1 on misses.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mocking.py
import pytest
from langchain_core.tools import tool
from evalbuilder.mocking import (MockMissError, match_rule, merge_mock_rules,
                                 wrap_tool, wrap_tools)

RULES = [
    {"matchArgs": {"origin": "SFO"}, "response": {"flights": [{"id": "AA100"}]}},
    {"matchArgs": {}, "response": {"flights": []}},
]

@tool
def search_flights(origin: str, destination: str) -> dict:
    "Search flights."
    return {"flights": [{"id": "REAL"}]}

def test_first_match_wins_and_wildcard_falls_through():
    assert match_rule(RULES, {"origin": "SFO", "destination": "JFK"})["response"]["flights"]
    assert match_rule(RULES, {"origin": "LAX"})["response"] == {"flights": []}
    assert match_rule([RULES[0]], {"origin": "LAX"}) is None

def test_wrap_tool_matches_and_preserves_identity():
    wrapped = wrap_tool(search_flights, RULES)
    assert wrapped.name == "search_flights" and wrapped.description
    assert wrapped.args_schema is not None
    out = wrapped.invoke({"origin": "SFO", "destination": "JFK"})
    assert out == {"flights": [{"id": "AA100"}]}

def test_miss_policies():
    only_sfo = [RULES[0]]
    real = wrap_tool(search_flights, only_sfo, on_miss="real")
    assert real.invoke({"origin": "LAX", "destination": "JFK"}) == {"flights": [{"id": "REAL"}]}
    fb = wrap_tool(search_flights, only_sfo, on_miss="fallback", fallback={"flights": None})
    assert fb.invoke({"origin": "LAX", "destination": "JFK"}) == {"flights": None}
    strict = wrap_tool(search_flights, only_sfo, on_miss="strict")
    with pytest.raises(MockMissError):
        strict.invoke({"origin": "LAX", "destination": "JFK"})

def test_merge_case_overrides_dataset_per_tool():
    ds = {"tools": {"a": [{"matchArgs": {}, "response": 1}],
                     "b": [{"matchArgs": {}, "response": 2}]}}
    case = {"tools": {"a": [{"matchArgs": {}, "response": 9}]}}
    merged = merge_mock_rules(ds, case)
    assert merged["a"][0]["response"] == 9 and merged["b"][0]["response"] == 2

def test_wrap_tools_passthrough():
    wrapped = wrap_tools([search_flights], {})
    assert wrapped[0] is search_flights
```

- [ ] **Step 2: Run, expect fail** → **Step 3: Implement** `mocking.py` (subset match: `all(args.get(k) == v for k, v in match_args.items())`; empty dict → match; `wrap_tool` builds the mocked callable accepting `**kwargs` and uses `StructuredTool.from_function(func=..., name=..., description=..., args_schema=tool.args_schema)`), then the two CLI commands + CliRunner tests (set rules on the weather dataset; verify catches an expected_tool with no matching rule).
- [ ] **Step 4: Run PASS** — `uv run pytest tests/test_mocking.py -v`
- [ ] **Step 5: Commit** — `"feat: ADK-style mock rules, tool wrapping, mock CLI"`

---

### Task 7: Target loading, execution, run artifact + CLI run

**Files:**
- Create: `src/evalbuilder/target.py`, `src/evalbuilder/runner.py`; extend `src/evalbuilder/schemas.py`
- Modify: `src/evalbuilder/cli.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces (`schemas.py`): `CaseRun(case_id: str, outputs: dict = {}, trajectory: list[dict] = [], tool_calls: list[dict] = [], node_path: list[str] = [], error: str | None = None, error_class: Literal["none","agent","infrastructure"] = "none")`; `RunArtifact(schema_: alias "schema" = "evalbuilder/run/v1", run_id: str, dataset_path: str, dataset_name: str, mocked: bool, case_runs: list[CaseRun] = [], timestamp: str = "")`.
- Produces (`target.py`): `load_target(target: Target) -> module`; `build_graph(module, target: Target, tools=None)` calls `getattr(module, target.factory)(tools=tools)`; `run_case(graph, case: Case) -> CaseRun` — if `case.metadata.get("user_turns")` is a list of strings, attach `MemorySaver` is NOT needed (examples compile without checkpointer; instead replay turns by feeding accumulated messages: invoke with full message history each turn, carrying prior output messages forward); else single `graph.invoke(case.inputs)`. Trajectory = `convert_to_openai_messages(state["messages"])`; `tool_calls` = flattened `[{"name","args"} for each AIMessage.tool_calls]`; `outputs = {"response": last message content}`. Exceptions → `error=str(e)`, `error_class="agent"` for `MockMissError`/graph errors, `"infrastructure"` for import/factory errors.
- Produces (`runner.py`): `run_dataset(ds: Dataset, dataset_path: Path, *, mocked: bool, ids: list[str] | None = None, out_dir: Path) -> RunArtifact` — refuses (ValueError) when the dataset has zero approved cases or any selected case is pending/rejected; for each approved case builds a **fresh graph** with `wrap_tools(module.TOOLS, merge_mock_rules(ds.mocks.get("tools", {}), case.metadata.get("mocks", {}).get("tools", {})))` when `mocked`, else raw factory; writes `out_dir/run-<8hex>.json` via `save_json`; returns the artifact.
- Produces (CLI): `evalbuilder run DS [--mock/--no-mock] [--ids "a,b"] [--out eval/results]` — prints `{"run_id", "path", "cases", "errors"}`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_runner.py
import pytest
from pathlib import Path
from evalbuilder.schemas import Dataset, Target
from evalbuilder.artifacts import add_case, save_json, set_review, load_dataset
from evalbuilder.runner import run_dataset

def _weather_ds(tmp_path) -> tuple[Dataset, Path]:
    ds = Dataset(name="w", dataset_type="final_response",
                 target=Target(module="examples.weather_bot.agent"))
    c = add_case(ds, {"inputs": {"messages": [{"role": "user",
                        "content": "What is the weather in Paris?"}]},
                      "reference_outputs": {"expected_tools": [{"name": "get_weather",
                                                                 "args": {"city": "Paris"}}]},
                      "metadata": {"intent": "intent.weather",
                                    "mocks": {"tools": {"get_weather": [
                                        {"matchArgs": {}, "response":
                                          {"temp": 55, "condition": "mocked-rain", "city": "Paris"}}]}}}})
    p = tmp_path / "ds.json"
    save_json(p, ds)
    return ds, p

def test_run_refuses_pending(tmp_path):
    ds, p = _weather_ds(tmp_path)
    with pytest.raises(ValueError, match="approved"):
        run_dataset(ds, p, mocked=False, out_dir=tmp_path)

def test_run_captures_trajectory_and_applies_mocks(tmp_path):
    ds, p = _weather_ds(tmp_path)
    set_review(ds, [ds.cases[0].id], "approved", "test")
    art = run_dataset(ds, p, mocked=True, out_dir=tmp_path)
    cr = art.case_runs[0]
    assert cr.error is None
    assert "55" in cr.outputs["response"]              # mock reached the answer
    assert {"name": "get_weather", "args": {"city": "Paris"}} in \
           [{"name": t["name"], "args": t["args"]} for t in cr.tool_calls]
    roles = [m["role"] for m in cr.trajectory]
    assert "tool" in roles and roles[-1] == "assistant"
    assert (tmp_path / f"run-{art.run_id}.json").exists()

def test_run_unmocked_uses_real_tool(tmp_path):
    ds, p = _weather_ds(tmp_path)
    set_review(ds, [ds.cases[0].id], "approved", "test")
    art = run_dataset(ds, p, mocked=False, out_dir=tmp_path)
    assert "72" in art.case_runs[0].outputs["response"]
```

- [ ] **Step 2: Run, expect fail** → **Step 3: Implement** `target.py` + `runner.py` + CLI command (+ a CliRunner test running the weather dataset end-to-end from disk). `run_id = uuid4().hex[:8]`, `timestamp` from `datetime.now(timezone.utc).isoformat()`.
- [ ] **Step 4: Run PASS** — `uv run pytest tests/test_runner.py -v`
- [ ] **Step 5: Commit** — `"feat: target execution with mock installation and run artifacts"`

---

### Task 8: Evaluators + scoring + report CLI

**Files:**
- Create: `src/evalbuilder/evaluators.py`; extend `src/evalbuilder/schemas.py`
- Modify: `src/evalbuilder/cli.py`
- Test: `tests/test_evaluators.py`

**Interfaces:**
- Produces (`schemas.py`): `Report(schema_: alias "schema" = "evalbuilder/report/v1", run_id: str, dataset_name: str, metrics: dict = {}, slices: dict = {}, cases: list[dict] = [])` — `metrics[key] = {"n", "avg", "min", "max", "errors"}`; `slices[dim][value][key] = avg` for dims `intent`, `failure_mode`, `variant`; `cases[i] = {"case_id", "scores": {key: {"score", "comment"}}, "errors": {key: str}}`.
- Produces (`evaluators.py`): `build_evaluators(specs: list[dict], judge_model: str) -> list[tuple[str, callable]]` where each callable has signature `fn(case: Case, case_run: CaseRun) -> dict` returning `{"key", "score", "comment"}` (score bool or float) or raising `EvaluatorUnavailable(reason)`:
  - `expected_tools` — every `reference_outputs.expected_tools` entry appears in `case_run.tool_calls` with subset-equal args, in order (subsequence match). Deterministic.
  - `contains` — `spec["value"]` or `reference_outputs["contains"]` substring of `outputs["response"]`.
  - `json_valid` — `outputs["response"]` parses as JSON.
  - `trajectory_match` — wraps `agentevals.trajectory.match.create_trajectory_match_evaluator(trajectory_match_mode=spec.get("match_mode", "unordered"))`, called with `outputs=case_run.trajectory, reference_outputs=case.reference_outputs["trajectory"]`; skipped (raise `EvaluatorUnavailable`) when the case has no reference trajectory.
  - `correctness` / `contract` — OpenEvals `create_llm_as_judge`; `contract` uses a custom prompt template embedding `reference_outputs.contract`; both raise `EvaluatorUnavailable("judge model key missing")` when `_has_judge_key` fails. Factory kept in a module-level function `_make_judge(prompt, model, key)` so tests monkeypatch it.
  - `custom` — `spec["ref"]` `"module:function"` import.
- Produces: `score_run(run: RunArtifact, ds: Dataset, specs, judge_model) -> Report` — per case: evaluators run over non-error case_runs; case_runs with `error` get score 0 for deterministic evaluators with comment `"agent error: ..."`; `EvaluatorUnavailable` lands in `cases[i].errors`, never in scores; aggregates + slices computed from case metadata.
- Produces (CLI): `evalbuilder score RUN_PATH --dataset DS --evaluators eval/evaluators.yaml [--out eval/results]` — YAML shape from spec §6.5; writes `report-<run_id>.json`; prints metrics + slices.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluators.py
import pytest
from evalbuilder.schemas import Case, CaseRun, Dataset, RunArtifact, Target
from evalbuilder.artifacts import add_case, set_review
from evalbuilder import evaluators as ev

def _case(**md):
    return Case.model_validate({"id": "case-x", "inputs": {"q": 1},
        "reference_outputs": {"expected_tools": [{"name": "get_weather", "args": {"city": "Paris"}}],
                               "contains": "Paris"},
        "metadata": {"intent": "i.a", "topic": "t", "scenario": "s",
                      "failure_mode": "none", "variant": "happy", **md},
        "review": {"status": "approved"}, "publication": {}})

def _run(tool_calls, response="72F in Paris"):
    return CaseRun(case_id="case-x", outputs={"response": response},
                   tool_calls=tool_calls,
                   trajectory=[{"role": "user", "content": "q"},
                                {"role": "assistant", "content": response}])

def test_expected_tools_subsequence_and_args_subset():
    fns = dict(ev.build_evaluators([{"type": "expected_tools"}], "anthropic:x"))
    ok = fns["expected_tools"](_case(), _run([{"name": "get_weather",
        "args": {"city": "Paris", "units": "F"}}]))
    assert ok["score"] is True
    bad = fns["expected_tools"](_case(), _run([{"name": "get_weather", "args": {"city": "Oslo"}}]))
    assert bad["score"] is False and "get_weather" in bad["comment"]

def test_contains_and_json_valid():
    fns = dict(ev.build_evaluators([{"type": "contains"}, {"type": "json_valid"}], "m"))
    assert fns["contains"](_case(), _run([]))["score"] is True
    assert fns["json_valid"](_case(), _run([], response='{"a": 1}'))["score"] is True
    assert fns["json_valid"](_case(), _run([], response="not json"))["score"] is False

def test_judge_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fns = dict(ev.build_evaluators([{"type": "correctness"}], "anthropic:claude-x"))
    with pytest.raises(ev.EvaluatorUnavailable):
        fns["correctness"](_case(), _run([]))

def test_judge_uses_factory(monkeypatch):
    calls = {}
    def fake_make(prompt, model, key):
        def judge(**kwargs):
            calls.update(kwargs)
            return {"key": key, "score": 1.0, "comment": "ok"}
        return judge
    monkeypatch.setattr(ev, "_make_judge", fake_make)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    fns = dict(ev.build_evaluators([{"type": "correctness"}], "anthropic:claude-x"))
    out = fns["correctness"](_case(), _run([]))
    assert out["score"] == 1.0 and "outputs" in calls

def test_score_run_aggregates_and_slices():
    ds = Dataset(name="d", dataset_type="final_response", target=Target(module="m"))
    a = add_case(ds, {"inputs": {"q": 1}, "reference_outputs": {"contains": "yes"},
                      "metadata": {"intent": "i.a", "variant": "happy"}})
    b = add_case(ds, {"inputs": {"q": 2}, "reference_outputs": {"contains": "yes"},
                      "metadata": {"intent": "i.b", "variant": "adversarial"}})
    set_review(ds, [a.id, b.id], "approved", "t")
    run = RunArtifact(run_id="r1", dataset_path="p", dataset_name="d", mocked=False,
        case_runs=[CaseRun(case_id=a.id, outputs={"response": "yes!"}),
                   CaseRun(case_id=b.id, outputs={"response": "no"})])
    report = ev.score_run(run, ds, [{"type": "contains"}], "m")
    assert report.metrics["contains"]["n"] == 2
    assert report.metrics["contains"]["avg"] == 0.5
    assert report.slices["intent"]["i.a"]["contains"] == 1.0
    assert report.slices["variant"]["adversarial"]["contains"] == 0.0
```

- [ ] **Step 2: Run, expect fail** → **Step 3: Implement** `evaluators.py` (~200 lines) + `score` CLI + a CliRunner test with a YAML file. `_make_judge` body:

```python
def _make_judge(prompt, model, key):
    from openevals.llm import create_llm_as_judge
    return create_llm_as_judge(prompt=prompt, model=model, feedback_key=key)
```

`contract` prompt template (module constant):

```python
CONTRACT_PROMPT = """You are checking one agent response against an explicit behavioral contract.
Return score 1 only when the response satisfies the contract; otherwise 0. Explain briefly.

<contract>{contract}</contract>
<inputs>{inputs}</inputs>
<outputs>{outputs}</outputs>"""
```

- [ ] **Step 4: Run PASS** — `uv run pytest tests/test_evaluators.py -v`
- [ ] **Step 5: Commit** — `"feat: deterministic + judge evaluators, scoring, report CLI"`

---

### Task 9: Multi-turn simulation + CLI

**Files:**
- Create: `src/evalbuilder/simulate.py`
- Modify: `src/evalbuilder/cli.py`
- Test: `tests/test_simulate.py`

**Interfaces:**
- Produces: `load_scenarios(path: Path) -> list[dict]` (YAML `scenarios:` list; each requires `id`, `opening`, `max_turns`, and at least one of `success_contains` / `expect` where `expect` supports `contains`, `not_contains`, `tool_called`; optional `persona`, `goal`, `followups: [str]`); `simulate_scenario(graph, scenario, user_model=None) -> dict` — transcript loop: send `opening`; after each agent reply check `success_contains` (stop `"success"`); if `followups` remain send the next one, else if `user_model` given, generate the next user message from persona+goal+transcript, else stop `"exhausted"`; hard stop at `max_turns` (stop reason `"max_turns"` — a truncation, not a pass). Returns `{"scenario_id", "stop_reason", "turns", "transcript": [openai-style messages], "violations": [str]}` where violations come from `expect` checks over the full transcript.
- Produces: `mine_failures(ds: Dataset, results: list[dict]) -> int` — for each result with violations or stop `"max_turns"`, adds a pending case: `inputs={"messages": [first user message]}`, `metadata={"source": "simulation", "scenario": scenario_id, "failure_mode": "simulation-violation", "user_turns": [subsequent user texts], "violations": [...], "transcript_tail": last 4 messages}`.
- Produces (CLI): `evalbuilder simulate DS --scenarios FILE [--out eval/results] [--mine/--no-mine]` — builds the graph from the dataset target (unmocked), runs all scenarios, writes `sim-<8hex>.json`, appends mined cases when `--mine` (default on), prints summary.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_simulate.py
import yaml
from pathlib import Path
from examples.travel_planner.agent import build_agent
from evalbuilder.schemas import Dataset, Target
from evalbuilder.simulate import load_scenarios, mine_failures, simulate_scenario

SCEN = {"id": "flight-booking-confirmation",
        "persona": "busy exec", "goal": "book the cheapest SFO->JFK flight",
        "opening": "Find me a flight from SFO to JFK on 2026-09-01",
        "followups": ["book it"],
        "max_turns": 4,
        "success_contains": "confirm",
        "expect": {"not_contains": "booked AA100 for you"}}

def test_load_scenarios_validates(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"scenarios": [SCEN]}))
    assert load_scenarios(p)[0]["id"] == SCEN["id"]

def test_simulation_reaches_confirmation_gate():
    result = simulate_scenario(build_agent(), SCEN)
    assert result["stop_reason"] == "success"
    assert result["violations"] == []
    assert any(m["role"] == "assistant" and "confirm" in m["content"].lower()
               for m in result["transcript"])

def test_mine_failures_adds_pending_case():
    ds = Dataset(name="d", dataset_type="final_response", target=Target(module="m"))
    bad = {"scenario_id": "s1", "stop_reason": "max_turns", "turns": 4,
           "transcript": [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "…"},
                           {"role": "user", "content": "again"}],
           "violations": ["never reached confirmation"]}
    assert mine_failures(ds, [bad]) == 1
    c = ds.cases[0]
    assert c.review.status == "pending"
    assert c.metadata["source"] == "simulation"
    assert c.metadata["user_turns"] == ["again"]
```

- [ ] **Step 2: Run, expect fail** → **Step 3: Implement** `simulate.py` (~140 lines; transcript maintained as message dicts; each turn re-invokes the graph with the full accumulated message list, matching `run_case`'s replay approach) + CLI + CliRunner test with the travel dataset on disk.
- [ ] **Step 4: Run PASS** — `uv run pytest tests/test_simulate.py -v`
- [ ] **Step 5: Commit** — `"feat: multi-turn simulation with failure mining"`

---

### Task 10: LangSmith publication (optional backend)

**Files:**
- Create: `src/evalbuilder/langsmith_io.py`
- Modify: `src/evalbuilder/cli.py`
- Test: `tests/test_langsmith_io.py`

**Interfaces:**
- Produces: `publish_approved(ds: Dataset, dataset_path: Path, settings: Settings, client=None, dataset_name: str | None = None) -> dict` — `client` defaults to `langsmith.Client(api_key=…, api_url=…)`; refuses when no approved cases; creates the dataset when `read_dataset` raises; uploads approved-and-unpublished cases as `inputs=case.inputs, outputs=case.reference_outputs, metadata={**case.metadata, "local_case_id": case.id}`; **read-back**: `list_examples(dataset_id=…)`, match `metadata["local_case_id"]`, set `case.publication.langsmith_example_id`; raise `RuntimeError` when any uploaded case is missing from read-back; saves the dataset file; returns `{"dataset_id", "dataset_name", "uploaded", "verified"}`.
- Produces (CLI): `evalbuilder publish DS [--dataset-name N]` — exits 1 with a clear message when `settings.langsmith_api_key` is missing.

- [ ] **Step 1: Write failing tests** — a `FakeClient` class in the test module with `read_dataset` (raises `LookupError` first call, returns namespace after), `create_dataset`, `create_examples` (stores rows), `list_examples` (yields namespaces built from stored rows; one test drops a row to trigger the read-back `RuntimeError`); tests: publishes only approved+unpublished, records example ids, is idempotent on second call (uploads 0), raises on read-back mismatch, CLI exits 1 without an API key.

```python
# tests/test_langsmith_io.py — core assertions
def test_publish_uploads_approved_only_and_records_ids(tmp_path):
    ds, path = _ds_with_two_cases(tmp_path)          # one approved, one pending
    fake = FakeClient()
    out = publish_approved(ds, path, _settings(), client=fake)
    assert out["uploaded"] == 1 and out["verified"] == 1
    approved = [c for c in ds.cases if c.review.status == "approved"][0]
    assert approved.publication.langsmith_example_id
    again = publish_approved(load_dataset(path), path, _settings(), client=fake)
    assert again["uploaded"] == 0

def test_publish_raises_on_readback_mismatch(tmp_path):
    ds, path = _ds_with_two_cases(tmp_path)
    fake = FakeClient(drop_readback=True)
    with pytest.raises(RuntimeError, match="read-back"):
        publish_approved(ds, path, _settings(), client=fake)
```

- [ ] **Step 2: Run, expect fail** → **Step 3: Implement** `langsmith_io.py` (~90 lines, lazy `import langsmith` inside the default-client branch only) + CLI command.
- [ ] **Step 4: Run PASS** — `uv run pytest tests/test_langsmith_io.py -v`
- [ ] **Step 5: Commit** — `"feat: idempotent LangSmith publish with read-back verification"`

---

### Task 11: The four skills + .claude symlinks

**Files:**
- Create: `skills/agent-eval-discover/SKILL.md`, `skills/agent-eval-discover/references/failure-taxonomy.md`, `skills/agent-eval-dataset/SKILL.md`, `skills/agent-eval-dataset/references/generation-guide.md`, `skills/agent-eval-mock/SKILL.md`, `skills/agent-eval-run/SKILL.md`, `skills/agent-eval-run/references/evaluator-selection.md`, symlinks under `.claude/skills/`
- Test: covered by Task 12's e2e (commands named in skills must exist); plus `tests/test_skills_lint.py`

**Interfaces:**
- Consumes: the full CLI surface from Tasks 1–10.
- Produces: four SKILL.md files whose every ```bash``` command line starts with `evalbuilder ` and parses against the real CLI.

- [ ] **Step 1: Write the lint test**

```python
# tests/test_skills_lint.py
import re
from pathlib import Path
from typer.testing import CliRunner
from evalbuilder.cli import app

SKILLS = sorted(Path("skills").glob("agent-eval-*/SKILL.md"))

def test_four_skills_exist_with_frontmatter():
    assert len(SKILLS) == 4
    for s in SKILLS:
        text = s.read_text()
        assert text.startswith("---\n") and "name:" in text and "description:" in text

def test_every_cli_command_in_skills_is_real():
    runner = CliRunner()
    for s in SKILLS:
        for m in re.finditer(r"^evalbuilder (\S+)( (\S+))?", s.read_text(), re.M):
            args = [m.group(1)] + ([m.group(3)] if m.group(3) and not m.group(3).startswith("-")
                                   and "/" not in m.group(3) and not m.group(3).isupper() else [])
            result = runner.invoke(app, args + ["--help"])
            assert result.exit_code == 0, f"{s}: 'evalbuilder {m.group(0)}' not a real command"

def test_symlinks_resolve():
    for link in Path(".claude/skills").glob("agent-eval-*"):
        assert (link / "SKILL.md").exists()
```

- [ ] **Step 2: Write the skills.** Author following `superpowers:writing-skills` (short imperative prose, explicit prohibitions, progressive disclosure). Required content per skill — write these fully, no stubs:

**agent-eval-discover/SKILL.md** — frontmatter `name: agent-eval-discover`, `description: Map a LangGraph agent's test surface — graph, tools, prompts, intents, scenarios, structurally justified failure modes, and data-domain topics — before any dataset generation. Use when starting eval work on a LangGraph agent or after the agent's code changes.` Workflow: (1) `evalbuilder check --target-module M` — report capabilities in one line when healthy, fix `blocking` items first; (2) `evalbuilder discover M` for the structural skeleton; (3) read the agent source and prompts yourself — node titles alone are weak evidence; (4) draft intents/scenarios, **every entry cites evidence** (`source:file:line`, `prompt:node`, `tool:name`, `edge:a->b`); (5) propose failure scenarios **only from references/failure-taxonomy.md whose structural precondition holds**; (6) derive data-domain topics only from user-provided documents or the user's description — never invent topics; (7) write via `evalbuilder agent-map update eval/agent-map.json --intents @… --scenarios @… --failures @…`; (8) present the map and `decisions_needed`, then **stop — do not generate cases until the user confirms the map or explicitly authorizes assumptions**. Prohibitions: no hand-editing artifact JSON; heuristics are hypotheses, never facts.

**references/failure-taxonomy.md** — the 10-type gated table from spec §6.1 with one-line preconditions and one example each.

**agent-eval-dataset/SKILL.md** — description: `Generate a coverage-driven golden dataset for a LangGraph agent from a reviewed agent map, in OpenEvals/LangSmith-compatible JSON with metadata. Use after agent-eval-discover, when creating or extending eval datasets.` Workflow: require a reviewed agent map (else route to agent-eval-discover); ask for existing goldens and `evalbuilder dataset import` them first; `evalbuilder dataset init`; check `evalbuilder dataset gaps` and allocate across the grid + variants (happy, boundary, adversarial, linguistic, multi-turn); write each case with `evalbuilder dataset add PATH --case @case.json` including `inputs.messages`, testable `reference_outputs` (`response` description, `expected_tools`, optional `trajectory`, `contract` from constraints), full metadata cell + `evidence`; quality-filter your own cases (reject ambiguous/unanswerable ones, rewrite rather than discard) and apply input evolutions (concretizing, constrained, comparative, multicontext) without changing the answerability contract; re-check gaps; keep everything pending. Prohibitions: **never run `evalbuilder review` unless the user has explicitly approved specific cases in this conversation; never infer approval from silence or enthusiasm**; never invent topics.

**references/generation-guide.md** — variant taxonomy with 2 examples each; default distribution happy 40% / failure 30% / boundary 15% / multi-turn 15%; multi-turn cases use `metadata.user_turns`; golden-answer modes: `llm_written` (you author `reference_outputs`) vs `agent_backfilled` (run once unmocked via `evalbuilder run`, then copy verified outputs into `reference_outputs` — only for cases a human will review).

**agent-eval-mock/SKILL.md** — description: `Declare ADK-style tool-mocking rules in an eval dataset and verify they make LangGraph agent runs deterministic. Use when eval cases depend on nondeterministic, side-effecting, or unavailable tools.` Workflow: read the dataset and agent map; list tools that are nondeterministic / side-effecting / rate-limited; for each affected case author ordered rules — most-specific `matchArgs` first, `{}` wildcard last when a default belongs in the fixture; shared fixtures go dataset-level (`mock set` without `--case`), case-specific overrides per case; state the miss policy decision explicitly (default: real tool runs); `evalbuilder mock verify DS` must pass before running. Include the rule-semantics table (ordered, first match, subset equality, wildcard). Prohibitions: never mock a tool the case is specifically testing for error handling of, unless the mock injects the error; never leave `book_*`-style side-effecting tools unmocked in datasets meant for CI.

**agent-eval-run/SKILL.md** — description: `Validate an eval dataset, execute a LangGraph agent over approved cases with mocks installed, score with OpenEvals evaluators, and report metrics with coverage slices; optionally publish to LangSmith and simulate multi-turn scenarios. Use when running or re-scoring experiments.` Workflow: `evalbuilder dataset validate` → `evalbuilder run DS --mock` → choose evaluators per references/evaluator-selection.md and write `eval/evaluators.yaml` → `evalbuilder score RUN --dataset DS --evaluators eval/evaluators.yaml` → report per-metric stats **and slices, never only the aggregate**; separate agent failures from evaluator errors — an evaluator problem is never the agent misbehaving. Optional: `evalbuilder publish DS` (approved only) when LangSmith is configured; `evalbuilder simulate DS --scenarios F` for conversation scenarios — only expectation-violating runs become new pending cases. Prohibitions: never re-run the agent just to re-score; never run with pending cases; a successful API call is not semantic success.

**references/evaluator-selection.md** — the ladder (deterministic → trajectory match → judge), one-metric-per-evaluator rule, YAML example from spec §6.5, judge model/key requirements, threshold note ("an uncalibrated threshold is theatre — start binary").

- [ ] **Step 3: Create symlinks** — `mkdir -p .claude/skills && for s in discover dataset mock run; do ln -s ../../skills/agent-eval-$s .claude/skills/agent-eval-$s; done`
- [ ] **Step 4: Run PASS** — `uv run pytest tests/test_skills_lint.py -v`
- [ ] **Step 5: Commit** — `"feat: agent-eval skills with references and .claude symlinks"`

---

### Task 12: End-to-end pipeline test + README

**Files:**
- Create: `tests/test_e2e_pipeline.py`, `README.md`
- Test: `tests/test_e2e_pipeline.py`

**Interfaces:**
- Consumes: everything.

- [ ] **Step 1: Write the e2e test** — CliRunner walking the whole flow in a tmp cwd (monkeypatch.chdir; copy nothing — target module is `examples.travel_planner.agent` importable from the repo install):

```python
# tests/test_e2e_pipeline.py  (structure; write fully)
def test_full_pipeline(tmp_path, monkeypatch):
    runner = CliRunner()
    eval_dir = tmp_path / "eval"
    # 1 discover
    r = runner.invoke(app, ["discover", "examples.travel_planner.agent",
                            "--source", "examples/travel_planner/agent.py",
                            "--eval-dir", str(eval_dir)])
    assert r.exit_code == 0
    # 2 agent-map update with intents/scenarios/failures (JSON via @files in tmp)
    # 3 dataset init + add 3 cases (happy flight search w/ mocks + expected_tools,
    #    constraint case "book it" expecting confirmation via contains,
    #    failure case input_validation) + validate + gaps vs agent map
    # 4 review --approve all three (simulating explicit user approval)
    # 5 mock verify
    # 6 run --mock ; assert 0 errors in printed summary
    # 7 score with evaluators.yaml (expected_tools + contains) ;
    #    load report json: metrics non-empty, slices["intent"] has 2 keys,
    #    contains avg == 1.0
    # 8 simulate with the confirmation scenario; stop_reason success; no new cases mined
```

Every step asserts `exit_code == 0` (or expected failure) and checks the artifact JSON on disk — this is the executable contract behind the skills.

- [ ] **Step 2: Run, fix integration bugs until PASS** — `uv run pytest tests/ -v` (full suite).
- [ ] **Step 3: Write README.md** — what this is, the pipeline diagram from spec §4, quickstart (`uv sync && uv pip install -e . && evalbuilder check`), the four skills table, dataset format example (one case verbatim from spec §6.2), LangSmith setup, extension points (ADK/Dify), link to spec + plan.
- [ ] **Step 4: Full suite PASS + Commit** — `"feat: e2e pipeline test and README"`

---

## Self-Review Notes

- Spec coverage: §6.1→T3/T5, §6.2→T2, §6.3→T6, §6.4→T7/T8, §6.5→T8, §6.6→T1–T10, §7→T11, §8→T4/T5/T6/T7, §9→T1/T10, §11→T4/T12. Extensibility §10 is interface-only in MVP: satisfied by `Target.framework` field + module seams (no extra task).
- Type consistency: `Case`/`Dataset`/`CaseRun`/`RunArtifact`/`Report` names used identically across T2/T7/T8/T9/T10; `set_review`, `add_case`, `save_json`, `wrap_tools`, `merge_mock_rules` signatures consistent.
- Known risk points called out for executors: exact import paths for `create_react_agent` and `create_trajectory_match_evaluator` may vary by installed version — resolve at Task 4/8 time against the locked versions, keeping the documented fallbacks.
