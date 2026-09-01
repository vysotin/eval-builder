"""Typer CLI — the deterministic surface the agent-eval skills drive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from evalbuilder import artifacts, coverage
from evalbuilder.config import Settings, capability_check
from evalbuilder.schemas import AgentMap, Dataset, Target

app = typer.Typer(help="Build and run evals for LangGraph agents.", no_args_is_help=True)
dataset_app = typer.Typer(help="Manage dataset artifacts.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
agent_map_app = typer.Typer(help="Manage the agent-map artifact.", no_args_is_help=True)
app.add_typer(agent_map_app, name="agent-map")


def _emit(data) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))


@app.callback()
def _main() -> None:
    """Build and run evals for LangGraph agents."""


def _read_json_arg(value: str):
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text())
    return json.loads(value)


def _load_ds(path: Path) -> Dataset:
    try:
        return artifacts.load_dataset(path)
    except FileNotFoundError:
        typer.echo(f"dataset not found: {path}", err=True)
        raise typer.Exit(1)


def _save_valid(path: Path, ds: Dataset) -> None:
    errors = artifacts.validate_dataset(ds)
    if errors:
        _emit({"errors": errors})
        raise typer.Exit(1)
    artifacts.save_json(path, ds)


@dataset_app.command("init")
def dataset_init(
    path: Path,
    name: str = typer.Option(..., "--name"),
    dataset_type: str = typer.Option("final_response", "--type"),
    target: str = typer.Option(..., "--target", help="MODULE or MODULE:FACTORY"),
) -> None:
    """Create an empty dataset artifact."""
    module, _, factory = target.partition(":")
    ds = Dataset(
        name=name,
        dataset_type=dataset_type,
        target=Target(module=module, factory=factory or "build_agent"),
    )
    _save_valid(path, ds)
    _emit({"path": str(path), "name": name})


@dataset_app.command("add")
def dataset_add(
    path: Path,
    case: str = typer.Option(..., "--case", help="case JSON, or @file.json"),
) -> None:
    """Normalize and append one case (always lands as pending)."""
    ds = _load_ds(path)
    try:
        added = artifacts.add_case(ds, _read_json_arg(case))
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    _save_valid(path, ds)
    _emit({"id": added.id, "cases": len(ds.cases)})


@dataset_app.command("import")
def dataset_import(
    path: Path,
    from_file: Path = typer.Option(..., "--from"),
) -> None:
    """Import cases from a foreign dataset file (lands as pending)."""
    ds = _load_ds(path)
    payload = json.loads(from_file.read_text())
    n = artifacts.import_cases(ds, payload, str(from_file))
    _save_valid(path, ds)
    _emit({"imported": n, "cases": len(ds.cases)})


@dataset_app.command("validate")
def dataset_validate(path: Path) -> None:
    """Validate the dataset; exit 1 with errors when invalid."""
    ds = _load_ds(path)
    errors = artifacts.validate_dataset(ds)
    if errors:
        _emit({"valid": False, "errors": errors})
        raise typer.Exit(1)
    _emit({"valid": True, "cases": len(ds.cases)})


@dataset_app.command("list")
def dataset_list(
    path: Path,
    status: Optional[str] = typer.Option(None, "--status"),
) -> None:
    """List cases (optionally filtered by review status)."""
    ds = _load_ds(path)
    rows = [
        {
            "id": c.id,
            "status": c.review.status,
            "intent": c.metadata.get("intent"),
            "scenario": c.metadata.get("scenario"),
            "failure_mode": c.metadata.get("failure_mode"),
            "source": c.metadata.get("source"),
        }
        for c in ds.cases
        if status is None or c.review.status == status
    ]
    _emit(rows)


@dataset_app.command("gaps")
def dataset_gaps(
    path: Path,
    agent_map: Path = typer.Option(..., "--agent-map"),
    target_per_cell: int = typer.Option(1, "--target-per-cell"),
) -> None:
    """Coverage gaps against the reviewed agent map."""
    ds = _load_ds(path)
    amap = AgentMap.model_validate(json.loads(agent_map.read_text()))
    _emit(coverage.coverage_gaps(ds, amap, target_per_cell))


def _check_entries(kind: str, entries, required: tuple[str, ...]) -> list[str]:
    errors = []
    if not isinstance(entries, list):
        return [f"{kind} must be a JSON list"]
    for i, entry in enumerate(entries):
        for key in required:
            if not entry.get(key):
                errors.append(f"{kind}[{i}] needs non-empty {key!r}")
    return errors


@agent_map_app.command("update")
def agent_map_update(
    path: Path,
    intents: Optional[str] = typer.Option(None, "--intents"),
    scenarios: Optional[str] = typer.Option(None, "--scenarios"),
    failures: Optional[str] = typer.Option(None, "--failures"),
    topics: Optional[str] = typer.Option(None, "--topics"),
    constraints: Optional[str] = typer.Option(None, "--constraints"),
) -> None:
    """Replace agent-map sections with validated, evidence-cited JSON."""
    amap = AgentMap.model_validate(json.loads(path.read_text()))
    errors: list[str] = []
    updated: list[str] = []
    if intents is not None:
        data = _read_json_arg(intents)
        errors += _check_entries("intents", data, ("id", "evidence"))
        amap.intents = data
        updated.append("intents")
    if scenarios is not None:
        data = _read_json_arg(scenarios)
        errors += _check_entries("scenarios", data, ("id", "evidence"))
        amap.scenarios = data
        updated.append("scenarios")
    if failures is not None:
        data = _read_json_arg(failures)
        errors += _check_entries("failure_scenarios", data, ("failure_type", "evidence"))
        amap.failure_scenarios = data
        updated.append("failure_scenarios")
    if topics is not None:
        data = _read_json_arg(topics)
        if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
            errors.append("topics must be a JSON list of strings")
        else:
            amap.data_domains["topics"] = data
            updated.append("topics")
    if constraints is not None:
        data = _read_json_arg(constraints)
        if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
            errors.append("constraints must be a JSON list of strings")
        else:
            amap.constraints = data
            updated.append("constraints")
    if errors:
        _emit({"errors": errors})
        raise typer.Exit(1)
    artifacts.save_json(path, amap)
    _emit({"updated": updated})


@app.command()
def review(
    path: Path,
    approve: Optional[str] = typer.Option(None, "--approve", help="comma-separated ids"),
    reject: Optional[str] = typer.Option(None, "--reject", help="comma-separated ids"),
    note: str = typer.Option("", "--note"),
) -> None:
    """Record an explicit human review decision. Never run without one."""
    if bool(approve) == bool(reject):
        typer.echo("pass exactly one of --approve / --reject", err=True)
        raise typer.Exit(1)
    ds = _load_ds(path)
    ids = [s.strip() for s in (approve or reject).split(",") if s.strip()]
    status = "approved" if approve else "rejected"
    try:
        n = artifacts.set_review(ds, ids, status, note)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    _save_valid(path, ds)
    _emit({"updated": n, "status": status})


mock_app = typer.Typer(help="Manage tool mocks: layer 1 rules (matchArgs → response) and layer 2 LLM mock strategies.", no_args_is_help=True)
app.add_typer(mock_app, name="mock")


def _tool_specs(ds: Dataset) -> dict[str, dict]:
    from evalbuilder import target as target_mod
    from evalbuilder.runner import tool_specs_of

    return tool_specs_of(target_mod.load_target(ds.target))


def _mock_model_for(ds: Dataset, spec: Optional[str]):
    from evalbuilder.claude_cli import model_from_spec

    chosen = spec or ((ds.mocks or {}).get("llm") or {}).get("model")
    if not chosen:
        typer.echo("no mock model: pass --mock-model SPEC or set mocks.llm.model (evalbuilder mock strategies PATH --model SPEC)", err=True)
        raise typer.Exit(1)
    return model_from_spec(chosen), chosen


@mock_app.command("set")
def mock_set(
    path: Path,
    tool: str = typer.Option(..., "--tool"),
    rules: str = typer.Option(..., "--rules", help="JSON list of rules, or @file.json"),
    case: Optional[str] = typer.Option(None, "--case", help="case id; omit for dataset-level"),
) -> None:
    """Set the ordered mock-rule list for one tool (dataset-level or per-case)."""
    ds = _load_ds(path)
    rule_list = _read_json_arg(rules)
    if case is None:
        ds.mocks.setdefault("tools", {})[tool] = rule_list
    else:
        matches = [c for c in ds.cases if c.id == case]
        if not matches:
            typer.echo(f"unknown case id {case}", err=True)
            raise typer.Exit(1)
        matches[0].metadata.setdefault("mocks", {}).setdefault("tools", {})[tool] = rule_list
    _save_valid(path, ds)
    _emit({"tool": tool, "rules": len(rule_list), "scope": case or "dataset"})


@mock_app.command("verify")
def mock_verify(path: Path) -> None:
    """Check every expected tool call in mocked cases matches a rule (under `on_miss: llm`
    a miss is answered by the LLM mock engine and only counted)."""
    from evalbuilder.mocking import verify_summary

    ds = _load_ds(path)
    summary = verify_summary(ds)
    _emit(summary)
    if not summary["ok"]:
        raise typer.Exit(1)


@mock_app.command("strategies")
def mock_strategies(
    path: Path,
    set_from: Optional[str] = typer.Option(None, "--set", help="strategies JSON ({world, strategies}) or @file (mock-strategies.json accepted)"),
    model: Optional[str] = typer.Option(None, "--model", help="mock model spec for the LLM engine (mocks.llm.model)"),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="dataset-level default strategy id"),
    on_miss: Optional[str] = typer.Option(None, "--on-miss", help="mock miss policy: real|fallback|strict|llm"),
    on_invalid: Optional[str] = typer.Option(None, "--on-invalid", help="engine answer still invalid after repair: fallback|strict"),
    max_repairs: Optional[int] = typer.Option(None, "--max-repairs"),
    case: Optional[str] = typer.Option(None, "--case", help="with --strategy: select the strategy for this case only (metadata.mocks.strategy)"),
) -> None:
    """Show or update the dataset's LLM mock layer: strategies, model, policy (or one case's strategy)."""
    from evalbuilder.mock_engine import validate_strategies

    ds = _load_ds(path)
    mocks = ds.mocks if isinstance(ds.mocks, dict) else {}
    mocks.setdefault("tools", {})
    changed: list[str] = []
    if case is not None:
        matches = [c for c in ds.cases if c.id == case]
        if not matches:
            typer.echo(f"unknown case id {case}", err=True)
            raise typer.Exit(1)
        if strategy is None:
            typer.echo("--case needs --strategy ID", err=True)
            raise typer.Exit(1)
        matches[0].metadata.setdefault("mocks", {})["strategy"] = strategy
        _save_valid(path, ds)
        _emit({"path": str(path), "case": case, "strategy": strategy})
        return
    if set_from is not None:
        data = _read_json_arg(set_from)
        if isinstance(data, dict) and "schema" in data:
            data = {k: v for k, v in data.items() if k != "schema"}
        problems = validate_strategies(data, _tool_specs(ds))
        if problems:
            _emit({"errors": problems})
            raise typer.Exit(1)
        mocks["strategies"] = data
        changed.append("strategies")
    if on_miss is not None:
        mocks["on_miss"] = on_miss
        changed.append("on_miss")
    if strategy is not None:
        mocks["strategy"] = strategy
        changed.append("strategy")
    if model is not None or on_invalid is not None or max_repairs is not None:
        llm = mocks.setdefault("llm", {})
        if model is not None:
            llm["model"] = model
        if on_invalid is not None:
            llm["on_invalid"] = on_invalid
        if max_repairs is not None:
            llm["max_repairs"] = max_repairs
        llm.setdefault("on_invalid", "fallback")
        llm.setdefault("max_repairs", 1)
        changed.append("llm")
    ds.mocks = mocks
    if changed:
        _save_valid(path, ds)
    strategies = mocks.get("strategies") or {}
    _emit({
        "path": str(path), "updated": changed, "on_miss": mocks.get("on_miss", "real"), "strategy": mocks.get("strategy"),
        "llm": mocks.get("llm"), "world": strategies.get("world"),
        "strategies": {sid: sorted((s.get("tools") or {}).keys()) for sid, s in (strategies.get("strategies") or {}).items()},
    })


@mock_app.command("validate")
def mock_validate(
    path: Path,
    tool: str = typer.Option(..., "--tool"),
    response: str = typer.Option(..., "--response", help="response JSON, or @file.json"),
) -> None:
    """Validate a mock response against the tool's declared output schema (the same check the
    LLM mock engine applies); exit 1 when it does not conform."""
    from evalbuilder import tool_schemas

    ds = _load_ds(path)
    specs = _tool_specs(ds)
    if tool not in specs:
        typer.echo(f"unknown or non-mockable tool {tool!r}; mockable tools: {sorted(specs)}", err=True)
        raise typer.Exit(1)
    value = _read_json_arg(response)
    schema = specs[tool].get("output_schema") or {}
    problems = tool_schemas.validate(value, schema)
    _emit({"tool": tool, "valid": not problems, "problems": problems, "output_schema": schema})
    if problems:
        raise typer.Exit(1)


@mock_app.command("try")
def mock_try(
    path: Path,
    tool: str = typer.Option(..., "--tool"),
    args: str = typer.Option("{}", "--args", help="call args JSON, or @file.json"),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="strategy id (default: the dataset's)"),
    mock_model: Optional[str] = typer.Option(None, "--mock-model", help="model spec for the engine (default: mocks.llm.model)"),
    on_miss: Optional[str] = typer.Option(None, "--on-miss", help="policy to try (default: the dataset's, or llm)"),
    case: Optional[str] = typer.Option(None, "--case", help="use this case's rules and strategy"),
) -> None:
    """Answer one tool call through both layers (rules, then the LLM mock engine) and show
    which layer answered, the response and its validation."""
    from evalbuilder import target as target_mod
    from evalbuilder.mocking import merge_mock_rules, wrap_tools
    from evalbuilder.runner import build_engine, tool_specs_of

    ds = _load_ds(path)
    module = target_mod.load_target(ds.target)
    call_args = _read_json_arg(args)
    case_rules, case_strategy = {}, None
    if case is not None:
        matches = [c for c in ds.cases if c.id == case]
        if not matches:
            typer.echo(f"unknown case id {case}", err=True)
            raise typer.Exit(1)
        case_rules = (matches[0].metadata.get("mocks") or {}).get("tools") or {}
        case_strategy = (matches[0].metadata.get("mocks") or {}).get("strategy")
    rules = merge_mock_rules((ds.mocks or {}).get("tools", {}), case_rules)
    policy = on_miss or (ds.mocks or {}).get("on_miss") or "llm"
    engine = None
    model_spec = None
    if policy == "llm":
        model, model_spec = _mock_model_for(ds, mock_model)
        engine = build_engine(model, ds, tool_specs_of(module), strategy=strategy or case_strategy)
    ledger: list[dict] = []
    wrapped = {t.name: t for t in wrap_tools(list(getattr(module, "TOOLS")), rules, on_miss=policy, engine=engine, ledger=ledger)}
    if tool not in wrapped:
        typer.echo(f"unknown tool {tool!r}; tools: {sorted(wrapped)}", err=True)
        raise typer.Exit(1)
    try:
        response = wrapped[tool].invoke(call_args)
        error = None
    except Exception as e:  # noqa: BLE001 - reported, not raised
        response, error = None, f"{type(e).__name__}: {e}"
    entry = ledger[-1] if ledger else {}
    _emit({
        "tool": tool, "args": call_args, "policy": policy, "layer": entry.get("layer"), "strategy": entry.get("strategy"),
        "model": model_spec, "response": response, "valid": entry.get("valid"), "repairs": entry.get("repairs"),
        "fallback": entry.get("fallback"), "problems": entry.get("problems"), "error": error,
    })
    if error:
        raise typer.Exit(1)


@app.command()
def discover(
    module: str,
    source: Optional[Path] = typer.Option(None, "--source"),
    eval_dir: Path = typer.Option(Path("eval"), "--eval-dir"),
) -> None:
    """Parse a LangGraph agent into eval/agent-map.json (AST + live introspection)."""
    import importlib.util
    import sys

    from evalbuilder import discover as discovery

    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    if source is None:
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            typer.echo(f"cannot locate source for module {module}", err=True)
            raise typer.Exit(1)
        source = Path(spec.origin)
    amap = discovery.discover_from_source(source)
    amap.app["module"] = module
    amap.graph["live"] = discovery.discover_live(module)
    out = eval_dir / "agent-map.json"
    artifacts.save_json(out, amap)
    _emit(
        {
            "path": str(out),
            "tools": [t["name"] for t in amap.tools],
            "skills": [s["name"] for s in amap.skills],
            "nodes": [n["id"] for n in amap.graph["nodes"]],
            "live": amap.graph["live"],
            "decisions_needed": amap.decisions_needed,
        }
    )


@app.command()
def run(
    path: Path,
    mock: bool = typer.Option(False, "--mock/--no-mock"),
    ids: Optional[str] = typer.Option(None, "--ids", help="comma-separated case ids"),
    out: Path = typer.Option(Path("eval/results"), "--out"),
    model: Optional[str] = typer.Option(
        None, "--model", help="agent model spec, e.g. claude-cli:sonnet (default: target's own)"
    ),
    on_miss: Optional[str] = typer.Option(None, "--on-miss", help="mock miss policy: real|fallback|strict|llm (default: the dataset's, else real)"),
    mock_model: Optional[str] = typer.Option(None, "--mock-model", help="model driving the LLM mock engine with --on-miss llm (default: mocks.llm.model)"),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="mock strategy id for the engine (default: the dataset's)"),
) -> None:
    """Execute approved cases against the target agent; write a run artifact."""
    from evalbuilder.runner import run_dataset

    ds = _load_ds(path)
    id_list = [s.strip() for s in ids.split(",") if s.strip()] if ids else None
    agent_model = None
    if model:
        from evalbuilder.claude_cli import model_from_spec

        agent_model = model_from_spec(model)
    policy = on_miss or ((ds.mocks or {}).get("on_miss") if mock else None) or "real"
    engine_model, engine_spec = (None, None)
    if mock and policy == "llm":
        engine_model, engine_spec = _mock_model_for(ds, mock_model)
    try:
        art = run_dataset(
            ds, path, mocked=mock, ids=id_list, out_dir=out,
            model=agent_model, model_spec=model, on_miss=policy,
            mock_model=engine_model, mock_model_spec=engine_spec, strategy=strategy,
        )
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    errors = [cr for cr in art.case_runs if cr.error]
    _emit(
        {
            "run_id": art.run_id,
            "path": str(Path(out) / f"run-{art.run_id}.json"),
            "cases": len(art.case_runs),
            "errors": len(errors),
            "error_cases": [
                {"case_id": cr.case_id, "error": cr.error, "class": cr.error_class}
                for cr in errors
            ],
            "mocking": art.mocking,
        }
    )


@app.command()
def simulate(
    path: Path,
    scenarios: Path = typer.Option(..., "--scenarios"),
    out: Path = typer.Option(Path("eval/results"), "--out"),
    mine: bool = typer.Option(True, "--mine/--no-mine"),
    mock: bool = typer.Option(False, "--mock/--no-mock", help="install the dataset's mock layers (rules, and the LLM engine under on_miss llm)"),
    on_miss: Optional[str] = typer.Option(None, "--on-miss", help="mock miss policy with --mock (default: the dataset's, else real)"),
    mock_model: Optional[str] = typer.Option(None, "--mock-model", help="model driving the LLM mock engine (default: mocks.llm.model)"),
) -> None:
    """Run multi-turn simulation scenarios; mine violations into pending cases."""
    from uuid import uuid4

    from evalbuilder import simulate as sim
    from evalbuilder import target as target_mod

    ds = _load_ds(path)
    try:
        scenario_list = sim.load_scenarios(scenarios)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    module = target_mod.load_target(ds.target)
    if mock:
        from evalbuilder.mocking import merge_mock_rules, wrap_tools
        from evalbuilder.runner import build_engine, tool_specs_of

        policy = on_miss or (ds.mocks or {}).get("on_miss") or "real"
        engine_model = _mock_model_for(ds, mock_model)[0] if policy == "llm" else None
        specs = tool_specs_of(module) if policy == "llm" else {}

        def graph_factory(scenario):
            ledger: list[dict] = []
            engine = build_engine(engine_model, ds, specs, strategy=scenario.get("mock_strategy")) if policy == "llm" else None
            tools = wrap_tools(list(getattr(module, "TOOLS")), merge_mock_rules((ds.mocks or {}).get("tools", {}), {}),
                               on_miss=policy, engine=engine, ledger=ledger)
            return target_mod.build_graph(module, ds.target, tools=tools), ledger

        results = sim.simulate_scenarios(graph_factory, scenario_list)
    else:
        graph = target_mod.build_graph(module, ds.target)
        results = [sim.simulate_scenario(graph, s) for s in scenario_list]
    mined = sim.mine_failures(ds, results) if mine else 0
    if mined:
        _save_valid(path, ds)
    sim_id = uuid4().hex[:8]
    artifacts.save_json(Path(out) / f"simulation-{sim_id}.json", {"schema": "evalbuilder/simulation/v1", "results": results})
    _emit(
        {
            "sim_id": sim_id,
            "scenarios": len(results),
            "stop_reasons": {r["scenario_id"]: r["stop_reason"] for r in results},
            "violations": {
                r["scenario_id"]: r["violations"] for r in results if r["violations"]
            },
            "mined": mined,
            "mock_calls": {r["scenario_id"]: r["mock_calls"] for r in results if r.get("mock_calls")},
        }
    )


@app.command()
def score(
    run_path: Path,
    dataset: Path = typer.Option(..., "--dataset"),
    evaluators: Path = typer.Option(..., "--evaluators", help="evaluators.yaml"),
    out: Path = typer.Option(Path("eval/results"), "--out"),
    env_file: Optional[Path] = typer.Option(None, "--env-file"),
) -> None:
    """Score a run artifact with the configured evaluators; write a report."""
    import yaml

    from evalbuilder.evaluators import score_run
    from evalbuilder.schemas import RunArtifact

    ds = _load_ds(dataset)
    run = RunArtifact.model_validate(json.loads(run_path.read_text()))
    config = yaml.safe_load(evaluators.read_text()) or {}
    specs = config.get("evaluators", [])
    if not specs:
        typer.echo("evaluators.yaml has no evaluators", err=True)
        raise typer.Exit(1)
    settings = Settings.load(env_file)
    report = score_run(run, ds, specs, settings.judge_model)
    artifacts.save_json(Path(out) / f"score-report-{run.run_id}.json", report)
    _emit(report.model_dump(by_alias=True))


@app.command()
def publish(
    path: Path,
    dataset_name: Optional[str] = typer.Option(None, "--dataset-name"),
    env_file: Optional[Path] = typer.Option(None, "--env-file"),
) -> None:
    """Publish approved cases to LangSmith (idempotent, read-back verified)."""
    from evalbuilder.langsmith_io import publish_approved

    settings = Settings.load(env_file)
    if not settings.langsmith_api_key:
        typer.echo("LANGSMITH_API_KEY is not configured (see .env.example)", err=True)
        raise typer.Exit(1)
    ds = _load_ds(path)
    try:
        result = publish_approved(ds, path, settings, dataset_name=dataset_name)
    except (ValueError, RuntimeError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    _emit(result)


@app.command()
def check(
    target_module: Optional[str] = typer.Option(None, "--target-module"),
    env_file: Optional[Path] = typer.Option(None, "--env-file"),
) -> None:
    """Print the capability matrix as JSON."""
    _emit(capability_check(Settings.load(env_file), target_module))


pipeline_app = typer.Typer(help="Autonomous end-to-end evaluation pipeline.", no_args_is_help=True)
app.add_typer(pipeline_app, name="pipeline")


@pipeline_app.command("init")
def pipeline_init(
    path: Path,
    name: str = typer.Option(..., "--name"),
    source: Path = typer.Option(..., "--source", help="agent source file"),
    module: str = typer.Option(..., "--module", help="importable module exposing TOOLS + build_agent"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Write a commented starter pipeline config."""
    from evalbuilder.pipeline.config import template

    if path.exists() and not force:
        typer.echo(f"{path} exists (use --force to overwrite)", err=True)
        raise typer.Exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template(name, str(source), module))
    _emit({"path": str(path), "name": name})


@pipeline_app.command("run")
def pipeline_run(
    config: Path,
    resume: bool = typer.Option(False, "--resume", help="reuse completed stages from state.json"),
    from_stage: Optional[str] = typer.Option(None, "--from", help="with --resume: rerun from this stage onward"),
    until: Optional[str] = typer.Option(None, "--until", help="stop after this stage (e.g. dataset: generate cases + mocks only)"),
    quiet: bool = typer.Option(False, "--quiet"),
    env_file: Optional[Path] = typer.Option(None, "--env-file"),
) -> None:
    """Run every stage from config to report.json; exit 1 unless the verdict is pass
    (exit 0 when stopped deliberately with --until)."""
    from evalbuilder.pipeline.config import STAGE_NAMES, load_config
    from evalbuilder.pipeline.report import run_pipeline, summary_text

    try:
        cfg = load_config(config)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)
    if until and until not in STAGE_NAMES:
        typer.echo(f"--until must be one of {', '.join(STAGE_NAMES)}", err=True)
        raise typer.Exit(2)
    log = (lambda msg: None) if quiet else (lambda msg: typer.echo(msg, err=True))
    _, report = run_pipeline(
        config, resume=resume, invalidate_from=from_stage, until=until, log=log,
        settings=Settings.load(env_file), config=cfg,
    )
    typer.echo(summary_text(report), err=True)
    _emit(
        {
            "verdict": report["verdict"],
            "overall_score": report["overall_score"],
            "report": str(cfg.output_dir / "report.json"),
            "stages": {k: v["status"] for k, v in report["stages"].items()},
            "problems": len(report["problems"]),
        }
    )
    if until:
        stopped = all(v["status"] in ("ok", "recovered", "skipped") for v in report["stages"].values())
        raise typer.Exit(0 if stopped else 1)
    if report["verdict"] != "pass":
        raise typer.Exit(1)


@app.command()
def ui(
    path: Optional[Path] = typer.Argument(None, help="pipeline output directory to open as the project (default: choose on the Pipeline setup page)"),
    port: int = typer.Option(8501, "--port"),
    headless: bool = typer.Option(False, "--headless", help="do not open a browser"),
) -> None:
    """Open the Streamlit report UI over pipeline artifacts (needs the `ui` extra: `uv sync --extra ui` or `pip install -e '.\\[ui]'`)."""
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("streamlit") is None:
        typer.echo("streamlit is not installed: run `uv sync --extra ui` (or `pip install -e '.[ui]'`)", err=True)
        raise typer.Exit(1)
    if path is not None and not path.is_dir():
        typer.echo(f"not a directory: {path}", err=True)
        raise typer.Exit(1)
    app_path = Path(__file__).parent / "ui" / "app.py"
    argv = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    if headless:
        argv += ["--server.headless", "true"]
    if path is not None:
        argv += ["--", "--dir", str(path)]
    raise typer.Exit(subprocess.call(argv))


@pipeline_app.command("report")
def pipeline_report(path: Path) -> None:
    """Print a human summary of a report.json (or the output directory holding one)."""
    from evalbuilder.pipeline.report import summary_text

    report_path = path / "report.json" if path.is_dir() else path
    if not report_path.exists():
        typer.echo(f"no report at {report_path}", err=True)
        raise typer.Exit(1)
    typer.echo(summary_text(json.loads(report_path.read_text())))


if __name__ == "__main__":  # `python -m evalbuilder.cli …` (background jobs)
    app()
