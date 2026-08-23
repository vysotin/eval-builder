"""Typer CLI — the deterministic surface the agent-eval skills drive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from evalbuilder.config import Settings, capability_check

app = typer.Typer(help="Build and run evals for LangGraph agents.", no_args_is_help=True)


def _emit(data) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))


@app.callback()
def _main() -> None:
    """Build and run evals for LangGraph agents."""


@app.command()
def check(
    target_module: Optional[str] = typer.Option(None, "--target-module"),
    env_file: Optional[Path] = typer.Option(None, "--env-file"),
) -> None:
    """Print the capability matrix as JSON."""
    _emit(capability_check(Settings.load(env_file), target_module))
