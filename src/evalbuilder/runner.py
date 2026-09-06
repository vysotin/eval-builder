"""Run a dataset's approved cases against the target agent in this process.

`run_dataset` is the in-process convenience the interactive CLI and the tests use: it
wraps the target in a `LocalAgent` and hands the work to the inference engine
(`inference.infer_dataset`), which is also what runs against a deployed agent. The two
helpers below (`tool_specs_of`, `build_engine`) are shared by the mock CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from evalbuilder import tool_schemas
from evalbuilder.inference import infer_dataset, local_agent_for
from evalbuilder.mock_engine import DEFAULT_STRATEGY, LLMMockEngine
from evalbuilder.mocking import mockable
from evalbuilder.schemas import Dataset, RunArtifact


def tool_specs_of(module) -> dict[str, dict]:
    """Tool definitions (schemas) the mock engine prompts with, keyed by name."""
    specs: dict[str, dict] = {}
    for t in getattr(module, "TOOLS", []) or []:
        if not mockable(t):
            continue
        try:
            specs[t.name] = tool_schemas.describe_tool(t)
        except Exception:  # noqa: BLE001 - keep the bare minimum for odd tool types
            specs[t.name] = {"name": t.name, "description": getattr(t, "description", "") or "", "args_schema": {}, "output_schema": {}}
    return specs


def build_engine(mock_model, ds: Dataset, tool_specs: dict[str, dict], strategy: str | None = None, log=None) -> LLMMockEngine:
    """One LLM mock engine for one conversation, configured from the dataset's `mocks` block."""
    llm = (ds.mocks or {}).get("llm") or {}
    return LLMMockEngine(
        mock_model, (ds.mocks or {}).get("strategies"), tool_specs,
        strategy=strategy or (ds.mocks or {}).get("strategy") or DEFAULT_STRATEGY,
        on_invalid=llm.get("on_invalid", "fallback"), max_repairs=int(llm.get("max_repairs", 1)), log=log,
    )


def run_dataset(
    ds: Dataset,
    dataset_path: Path,
    *,
    mocked: bool,
    ids: list[str] | None = None,
    out_dir: Path,
    model=None,
    model_spec: str | None = None,
    on_miss: str = "real",
    fallback=None,
    max_workers: int = 1,
    workers: int | None = None,
    backend: str = "threads",
    progress: Callable[[dict], None] | None = None,
    mock_model=None,
    mock_model_spec: str | None = None,
    strategy: str | None = None,
) -> RunArtifact:
    """Execute cases in this process and write `run-<id>.json`.

    `model` is injected into the target factory (`build_agent(model=...)`) so the same
    agent code can run against a scripted model in tests and a real one in evals.
    `on_miss` is the mock miss policy (real / fallback / strict / llm); under `llm`
    `mock_model` drives the LLM mock engine (one per conversation). `workers` (or the
    older `max_workers`) cases run concurrently; results keep dataset order and
    `progress` is called after every case.
    """
    if mocked and on_miss == "llm" and mock_model is None and not mock_model_spec:
        raise ValueError("on_miss='llm' needs a mock model (mock_model=) to drive the LLM mock engine")
    agent = local_agent_for(ds, model=model, mock_model=mock_model, mock_model_spec=mock_model_spec)
    if fallback is not None:
        agent.default_fallback = fallback
    return infer_dataset(
        ds, dataset_path, agent, out_dir=out_dir, ids=ids, workers=workers if workers is not None else max_workers,
        backend=backend, on_miss=on_miss if mocked else None, strategy=strategy, mocked=mocked, progress=progress,
        model_spec=model_spec, mock_model_spec=mock_model_spec,
    )
