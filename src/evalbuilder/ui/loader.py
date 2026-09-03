"""Load pipeline artifacts into a `Bundle` — by directory (file names) or by uploaded
files (embedded `schema` ids). Pure Python; no Streamlit imports so it is unit-testable.

The naming convention lives in `evalbuilder.pipeline.layout`; this module only applies
it: every artifact kind becomes one attribute of the bundle (None when absent), per-run
kinds become `{run_id: data}` dicts, and dict-shaped artifacts are unwrapped so pages
see the same shapes the pipeline stages work with.

Kinds the pipeline folded into a parent artifact (coverage into the dataset, applicable
failure types into the agent map, …) are *derived* here from that parent when no
stand-alone file is present, so a page asking for `coverage_plan` gets the same shape
whether it is reading a current output directory or one written before the fold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evalbuilder.pipeline.layout import (
    ARTIFACTS,
    FOLDED,
    ArtifactKind,
    artifact_index,
    fold_from,
    identify,
    unwrap,
)

DEFAULT_SCAN_ROOTS = ("eval/pipeline", "docs/examples")


@dataclass
class Bundle:
    """Every artifact of one pipeline output, keyed by kind."""

    name: str = ""
    source: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)  # kind → data | {run_id: data}
    files: dict[str, Any] = field(default_factory=dict)  # kind → path | {run_id: path}
    problems: list[str] = field(default_factory=list)

    def get(self, kind: str, default: Any = None) -> Any:
        return self.artifacts.get(kind, default)

    def has(self, kind: str) -> bool:
        return self.artifacts.get(kind) not in (None, {}, [])

    # per-kind conveniences (all Optional)
    @property
    def agent_map(self) -> dict | None:
        return self.get("agent_map")

    @property
    def dataset(self) -> dict | None:
        return self.get("dataset")

    @property
    def aggregate(self) -> dict | None:
        return self.get("aggregate")

    @property
    def report(self) -> dict | None:
        return self.get("pipeline_report")

    @property
    def state(self) -> dict | None:
        return self.get("pipeline_state")

    @property
    def runs(self) -> dict[str, dict]:
        return self.get("run") or {}

    @property
    def score_reports(self) -> dict[str, dict]:
        return self.get("score_report") or {}

    @property
    def run_ids(self) -> list[str]:
        """Run ids in scoring order when the state knows it, else sorted."""
        ordered: list[str] = []
        state = self.state or {}
        for entry in ((state.get("stages") or {}).get("run") or {}).get("artifacts", {}).get("runs", []) or []:
            if entry.get("run_id"):
                ordered.append(entry["run_id"])
        known = set(self.runs) | set(self.score_reports)
        return [r for r in ordered if r in known] + sorted(known - set(ordered))

    @property
    def verdict(self) -> str | None:
        if self.report:
            return self.report.get("verdict")
        if self.aggregate:
            return self.aggregate.get("verdict")
        return None

    @property
    def config(self) -> dict:
        return (self.report or {}).get("config") or {}

    def summary(self) -> dict:
        """Counts for the sidebar / summary page."""
        ds = self.dataset or {}
        amap = self.agent_map or {}
        return {
            "name": self.name,
            "verdict": self.verdict,
            "artifacts": sorted(k for k in self.artifacts if self.has(k)),
            "cases": len(ds.get("cases", [])),
            "intents": len(amap.get("intents", [])),
            "scenarios": len(amap.get("scenarios", [])),
            "tools": len(amap.get("tools", [])),
            "skills": len(amap.get("skills", []) or []),
            "runs": len(self.run_ids),
        }


def _read(path: Path, kind: ArtifactKind) -> Any:
    text = path.read_text()
    if kind.format == "yaml":
        return yaml.safe_load(text)
    return json.loads(text)


def load_dir(path: str | Path) -> Bundle:
    """Read every recognised artifact in a pipeline output directory."""
    out_dir = Path(path)
    bundle = Bundle(name=out_dir.name, source=str(out_dir))
    if not out_dir.is_dir():
        bundle.problems.append(f"not a directory: {out_dir}")
        return bundle
    index = artifact_index(out_dir)
    if not index:
        bundle.problems.append(f"no evalbuilder artifacts found in {out_dir}")
    for kind_name, location in index.items():
        kind = ARTIFACTS[kind_name]
        if isinstance(location, dict):
            per_run: dict[str, Any] = {}
            for run_id, file in location.items():
                try:
                    per_run[run_id] = _read(Path(file), kind)
                except Exception as e:  # noqa: BLE001 - one bad file must not hide the rest
                    bundle.problems.append(f"{file}: {type(e).__name__}: {e}")
            bundle.artifacts[kind_name] = per_run
            bundle.files[kind_name] = location
            continue
        try:
            data = _read(Path(location), kind)
        except Exception as e:  # noqa: BLE001
            bundle.problems.append(f"{location}: {type(e).__name__}: {e}")
            continue
        bundle.artifacts[kind_name] = unwrap(kind_name, data) if kind.format == "json" else data
        bundle.files[kind_name] = location
    _finish(bundle)
    return bundle


def load_files(files: list[tuple[str, bytes | str]]) -> Bundle:
    """Identify uploaded `(name, content)` pairs by embedded schema, then by file name."""
    bundle = Bundle(name="uploads", source="uploads")
    for name, content in files:
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        data: Any
        try:
            data = yaml.safe_load(text) if name.endswith((".yaml", ".yml")) else json.loads(text)
        except Exception as e:  # noqa: BLE001
            bundle.problems.append(f"{name}: not JSON/YAML ({type(e).__name__})")
            continue
        hit = identify(data, name)
        if hit is None:
            bundle.problems.append(f"{name}: unrecognised artifact (no known schema or file name)")
            continue
        kind, run_id = hit
        if kind.per_run:
            bundle.artifacts.setdefault(kind.kind, {})[run_id or name] = data
            bundle.files.setdefault(kind.kind, {})[run_id or name] = name
        else:
            bundle.artifacts[kind.kind] = unwrap(kind.kind, data) if kind.format == "json" else data
            bundle.files[kind.kind] = name
    _finish(bundle)
    return bundle


def _derive_folded(bundle: Bundle) -> None:
    """Present every kind that now lives inside another artifact (see layout.FOLDED) in
    one shape, whichever way this bundle got it.

    A stand-alone file, when one exists, still wins — an output directory written before
    the fold keeps rendering from its own files — but it sheds the `schema` stamp, which
    names a file this data no longer has. Otherwise the kind is read out of its parent.
    """
    for kind, target in FOLDED.items():
        if bundle.has(kind):
            data = bundle.artifacts[kind]
            if isinstance(data, dict) and "schema" in data:
                bundle.artifacts[kind] = {k: v for k, v in data.items() if k != "schema"}
            continue
        data = fold_from(kind, bundle.artifacts)
        if data not in (None, {}, []):
            bundle.artifacts[kind] = data
            bundle.files[kind] = f"{bundle.files.get(target.split('.')[0], '?')}#{target}"


def _finish(bundle: Bundle) -> None:
    _derive_folded(bundle)
    report = bundle.report or {}
    ds = bundle.dataset or {}
    bundle.name = report.get("name") or ds.get("name") or bundle.name
    # score reports keyed by their own run_id when the file name did not carry one
    fixed: dict[str, dict] = {}
    for key, rep in (bundle.score_reports or {}).items():
        fixed[rep.get("run_id") or key] = rep
    if fixed:
        bundle.artifacts["score_report"] = fixed
    fixed_runs: dict[str, dict] = {}
    for key, run in (bundle.runs or {}).items():
        fixed_runs[run.get("run_id") or key] = run
    if fixed_runs:
        bundle.artifacts["run"] = fixed_runs


def discover_dirs(roots: tuple[str, ...] | list[str] = DEFAULT_SCAN_ROOTS, cwd: Path | None = None) -> list[str]:
    """Pipeline output directories under the scan roots (any dir holding a known artifact)."""
    base = Path(cwd or ".")
    found: list[str] = []
    for root in roots:
        root_path = base / root
        if not root_path.is_dir():
            continue
        for child in sorted(root_path.iterdir()):
            if child.is_dir() and artifact_index(child):
                found.append(str(child))
    return found
