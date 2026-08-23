"""Typer CLI — the deterministic surface the agent-eval skills drive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from evalbuilder import artifacts
from evalbuilder.config import Settings, capability_check
from evalbuilder.schemas import Dataset, Target

app = typer.Typer(help="Build and run evals for LangGraph agents.", no_args_is_help=True)
dataset_app = typer.Typer(help="Manage dataset artifacts.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")


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


@app.command()
def check(
    target_module: Optional[str] = typer.Option(None, "--target-module"),
    env_file: Optional[Path] = typer.Option(None, "--env-file"),
) -> None:
    """Print the capability matrix as JSON."""
    _emit(capability_check(Settings.load(env_file), target_module))
