"""Run a dataset's approved cases against the target agent, with mocks installed."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
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
) -> RunArtifact:
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
    )

    for case in selected:
        try:
            if mocked:
                rules = merge_mock_rules(
                    ds.mocks.get("tools", {}),
                    case.metadata.get("mocks", {}).get("tools", {}),
                )
                tools = wrap_tools(list(getattr(module, "TOOLS")), rules)
                graph = target_mod.build_graph(module, ds.target, tools=tools)
            else:
                graph = target_mod.build_graph(module, ds.target)
        except Exception as e:  # noqa: BLE001 - factory failure is infrastructure
            artifact.case_runs.append(
                CaseRun(
                    case_id=case.id,
                    error=f"{type(e).__name__}: {e}",
                    error_class="infrastructure",
                )
            )
            continue
        artifact.case_runs.append(target_mod.run_case(graph, case))

    out_dir = Path(out_dir)
    save_json(out_dir / f"run-{artifact.run_id}.json", artifact)
    return artifact
