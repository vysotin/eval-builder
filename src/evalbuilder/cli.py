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


mock_app = typer.Typer(help="Manage ADK-style tool-mock rules.", no_args_is_help=True)
app.add_typer(mock_app, name="mock")


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
    """Check every expected tool call in mocked cases matches at least one rule."""
    from evalbuilder.mocking import match_rule, merge_mock_rules

    ds = _load_ds(path)
    misses: list[dict] = []
    for c in ds.cases:
        merged = merge_mock_rules(
            ds.mocks.get("tools", {}), c.metadata.get("mocks", {}).get("tools", {})
        )
        if not merged:
            continue
        for expected in c.reference_outputs.get("expected_tools", []):
            name = expected.get("name")
            if name in merged and match_rule(merged[name], expected.get("args", {})) is None:
                misses.append({"case": c.id, "tool": name, "args": expected.get("args", {})})
    _emit({"ok": not misses, "misses": misses})
    if misses:
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
) -> None:
    """Execute approved cases against the target agent; write a run artifact."""
    from evalbuilder.runner import run_dataset

    ds = _load_ds(path)
    id_list = [s.strip() for s in ids.split(",") if s.strip()] if ids else None
    try:
        art = run_dataset(ds, path, mocked=mock, ids=id_list, out_dir=out)
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
        }
    )


@app.command()
def simulate(
    path: Path,
    scenarios: Path = typer.Option(..., "--scenarios"),
    out: Path = typer.Option(Path("eval/results"), "--out"),
    mine: bool = typer.Option(True, "--mine/--no-mine"),
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
    graph = target_mod.build_graph(module, ds.target)
    results = [sim.simulate_scenario(graph, s) for s in scenario_list]
    mined = sim.mine_failures(ds, results) if mine else 0
    if mined:
        _save_valid(path, ds)
    sim_id = uuid4().hex[:8]
    artifacts.save_json(Path(out) / f"sim-{sim_id}.json", {"results": results})
    _emit(
        {
            "sim_id": sim_id,
            "scenarios": len(results),
            "stop_reasons": {r["scenario_id"]: r["stop_reason"] for r in results},
            "violations": {
                r["scenario_id"]: r["violations"] for r in results if r["violations"]
            },
            "mined": mined,
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
    artifacts.save_json(Path(out) / f"report-{run.run_id}.json", report)
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
