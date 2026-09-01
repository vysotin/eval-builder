"""Run a dataset's approved cases against the target agent, with mocks installed."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from evalbuilder import target as target_mod
from evalbuilder import tool_schemas
from evalbuilder.artifacts import save_json
from evalbuilder.mock_engine import DEFAULT_STRATEGY, LLMMockEngine
from evalbuilder.mocking import ledger_totals, merge_mock_rules, mockable, wrap_tools
from evalbuilder.schemas import CaseRun, Dataset, RunArtifact


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
    progress: Callable[[dict], None] | None = None,
    mock_model=None,
    mock_model_spec: str | None = None,
    strategy: str | None = None,
) -> RunArtifact:
    """Execute cases and write `run-<id>.json`.

    `model` is injected into the target factory (`build_agent(model=...)`) so the
    same agent code can run against a scripted model in tests and a real one in evals.
    `on_miss` is the mock miss policy (real / fallback / strict / llm) for wrapped tools;
    with `llm` every mockable tool is wrapped and calls no rule answers go to an
    `LLMMockEngine` driven by `mock_model` (one engine per case, strategy from the case's
    `metadata.mocks.strategy`, else `strategy`, else the dataset's). Each `CaseRun`
    carries the mock ledger (`mock_calls`); the artifact sums the layers in `mocking`.

    With `max_workers > 1` the cases are grouped by `metadata.intent` and the groups
    run concurrently (cases within one intent stay sequential); results keep dataset
    order. `progress` (optional) is called after every case with
    `{case_id, intent, error_class, error, completed, total}` — calls are serialized.
    """
    if ids:
        by_id = {c.id: c for c in ds.cases}
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ValueError(f"unknown case ids: {', '.join(unknown)}")
        selected = [by_id[i] for i in ids]
    else:
        selected = [c for c in ds.cases if c.review.status == "approved"]

    if not selected:
        raise ValueError("dataset has no approved cases to run")
    not_approved = [c.id for c in selected if c.review.status != "approved"]
    if not_approved:
        raise ValueError(
            f"cases are not approved: {', '.join(not_approved)}"
        )

    module = target_mod.load_target(ds.target)
    llm_mocking = mocked and on_miss == "llm"
    if llm_mocking and mock_model is None:
        raise ValueError("on_miss='llm' needs a mock model (mock_model=) to drive the LLM mock engine")
    tool_specs = tool_specs_of(module) if llm_mocking else {}
    llm_block = (ds.mocks or {}).get("llm") or {}

    artifact = RunArtifact(
        run_id=uuid4().hex[:8],
        dataset_path=str(dataset_path),
        dataset_name=ds.name,
        mocked=mocked,
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent_model=model_spec,
    )

    def _one(case) -> CaseRun:
        ledger: list[dict] = []
        try:
            if mocked:
                rules = merge_mock_rules(
                    ds.mocks.get("tools", {}),
                    case.metadata.get("mocks", {}).get("tools", {}),
                )
                engine = None
                if llm_mocking:
                    case_strategy = (case.metadata.get("mocks") or {}).get("strategy") or strategy
                    engine = build_engine(mock_model, ds, tool_specs, strategy=case_strategy)
                tools = wrap_tools(
                    list(getattr(module, "TOOLS")), rules, on_miss=on_miss, fallback=fallback,
                    engine=engine, ledger=ledger,
                )
                graph = target_mod.build_graph(module, ds.target, tools=tools, model=model)
            else:
                graph = target_mod.build_graph(module, ds.target, model=model)
        except Exception as e:  # noqa: BLE001 - factory failure is infrastructure
            return CaseRun(case_id=case.id, error=f"{type(e).__name__}: {e}", error_class="infrastructure")
        case_run = target_mod.run_case(graph, case)
        case_run.mock_calls = ledger
        return case_run

    results: dict[str, CaseRun] = {}
    lock = threading.Lock()

    def _record(case, case_run: CaseRun) -> None:
        with lock:
            results[case.id] = case_run
            if progress is not None:
                try:
                    progress({
                        "case_id": case.id,
                        "intent": case.metadata.get("intent") or "",
                        "error_class": case_run.error_class,
                        "error": case_run.error,
                        "completed": len(results),
                        "total": len(selected),
                    })
                except Exception:  # noqa: BLE001 - progress reporting is advisory
                    pass

    groups: dict[str, list] = {}
    for case in selected:
        groups.setdefault(case.metadata.get("intent") or "", []).append(case)

    def _run_group(cases) -> None:
        for case in cases:
            _record(case, _one(case))

    if max_workers > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(groups))) as pool:
            for future in [pool.submit(_run_group, cases) for cases in groups.values()]:
                future.result()
    else:
        _run_group(selected)
    artifact.case_runs = [results[case.id] for case in selected]
    if mocked:
        artifact.mocking = {
            "on_miss": on_miss,
            "model": (mock_model_spec or llm_block.get("model")) if llm_mocking else None,
            "strategy": (strategy or (ds.mocks or {}).get("strategy") or DEFAULT_STRATEGY) if llm_mocking else None,
            "calls": ledger_totals([e for cr in artifact.case_runs for e in cr.mock_calls]),
        }

    out_dir = Path(out_dir)
    save_json(out_dir / f"run-{artifact.run_id}.json", artifact)
    return artifact
