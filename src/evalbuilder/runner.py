"""Run a dataset's approved cases against the target agent, with mocks installed."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from evalbuilder import target as target_mod
from evalbuilder.artifacts import save_json
from evalbuilder.mocking import merge_mock_rules, wrap_tools
from evalbuilder.schemas import CaseRun, Dataset, RunArtifact


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
) -> RunArtifact:
    """Execute cases and write `run-<id>.json`.

    `model` is injected into the target factory (`build_agent(model=...)`) so the
    same agent code can run against a scripted model in tests and a real one in evals.
    `on_miss` is the mock miss policy (real / fallback / strict) for wrapped tools.

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

    artifact = RunArtifact(
        run_id=uuid4().hex[:8],
        dataset_path=str(dataset_path),
        dataset_name=ds.name,
        mocked=mocked,
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent_model=model_spec,
    )

    def _one(case) -> CaseRun:
        try:
            if mocked:
                rules = merge_mock_rules(
                    ds.mocks.get("tools", {}),
                    case.metadata.get("mocks", {}).get("tools", {}),
                )
                tools = wrap_tools(
                    list(getattr(module, "TOOLS")), rules, on_miss=on_miss, fallback=fallback
                )
                graph = target_mod.build_graph(module, ds.target, tools=tools, model=model)
            else:
                graph = target_mod.build_graph(module, ds.target, model=model)
        except Exception as e:  # noqa: BLE001 - factory failure is infrastructure
            return CaseRun(case_id=case.id, error=f"{type(e).__name__}: {e}", error_class="infrastructure")
        return target_mod.run_case(graph, case)

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

    out_dir = Path(out_dir)
    save_json(out_dir / f"run-{artifact.run_id}.json", artifact)
    return artifact
