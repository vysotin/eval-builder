"""Artifact naming convention — the one registry of every pipeline artifact.

Every kind has a *tier* that says where — and whether — it is written:

- `final` — a deliverable of the evaluation, at the root of the output dir
  (`<kind>.json` / `.yaml`) or, for per-run kinds, `results/<kind>-<run_id>.json`;
- `work` — scratch the pipeline needs but no consumer of the evaluation does
  (resume state, live progress, the UI job record, stage-to-stage handoffs). It lives
  in `<out_dir>/work/` and the whole directory can be deleted without losing anything;
- `derived` — data that is *part of another artifact* and is therefore never written to
  a file of its own.

`folded_into` is a second, independent axis: it names the place *inside another
artifact* that durably holds this kind's data. Every `derived` kind has one — that is
what makes it derived. A `work` kind may have one too (the mocks stage hands its rules
to the dataset stage through `work/`, and `dataset.mocks.tools` is where they stay), and
that is precisely why deleting `work/` loses nothing. Folded kinds stay in the registry
so directories written before the fold — and single-file uploads — still resolve, and so
the UI can present the same kind whether it read a stand-alone file or its parent.

Other rules:
- every JSON artifact embeds `"schema": "evalbuilder/<kind>/v1"`, so a file can be
  identified by content (uploads) as well as by name (directories);
- kinds, file names and schema ids are unique.

Stages write through `path_for`/`stamp`, the report lists artifacts through
`ARTIFACTS`, the Streamlit UI loads them through `kind_of_file`/`kind_of_schema`, and
`legacy_files`/`legacy_schemas` keep output directories from earlier versions readable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_PREFIX = "evalbuilder/"
WORK_DIR = "work"  # scratch tier: resume state, live progress, stage handoffs
RESULTS_DIR = "results"  # per-run tier


@dataclass(frozen=True)
class ArtifactKind:
    kind: str
    file: str  # relative file name; per-run kinds contain `{run_id}`
    schema: str | None  # embedded schema id (None for YAML artifacts)
    stage: str  # stage(s) that write it
    description: str
    legacy_files: tuple[str, ...] = ()
    legacy_schemas: tuple[str, ...] = ()
    wrap_key: str | None = None  # dict-shaped payloads are stored under this key
    tier: str = "final"  # final | work | derived  (see the module docstring)
    folded_into: str | None = None  # dotted path in another artifact that durably holds this data

    @property
    def per_run(self) -> bool:
        return "{run_id}" in self.file

    @property
    def written(self) -> bool:
        """False for `derived` kinds — they are part of another artifact, never a file."""
        return self.tier != "derived"

    @property
    def format(self) -> str:
        return "yaml" if self.file.endswith((".yaml", ".yml")) else "json"

    def path(self, out_dir: Path, run_id: str | None = None) -> Path:
        if self.per_run:
            if not run_id:
                raise ValueError(f"artifact {self.kind} needs a run_id")
            return Path(out_dir) / self.file.format(run_id=run_id)
        return Path(out_dir) / self.file

    def matches(self, name: str) -> str | None:
        """Return the run_id ("" for root artifacts) when `name` is this kind's file."""
        for pattern in (self.file, *self.legacy_files):
            rx = "^" + re.escape(pattern).replace(r"\{run_id\}", r"(?P<run_id>[A-Za-z0-9_-]+)") + "$"
            m = re.match(rx, name.replace("\\", "/"))
            if m:
                return m.groupdict().get("run_id", "") or ""
        return None


ARTIFACTS: dict[str, ArtifactKind] = {
    a.kind: a
    for a in [
        ArtifactKind(
            "agent_map", "agent-map.json", "evalbuilder/agent-map/v1", "discover, map",
            "Agent test surface: graph, tools, prompts, intents, scenarios, failure modes, constraints.",
        ),
        ArtifactKind(
            "applicable_failures", "applicable-failures.json", "evalbuilder/applicable-failures/v1", "map",
            "Failure types that structurally apply to this agent, with the evidence that gates them.",
            wrap_key="failure_types", tier="derived", folded_into="agent_map.applicable_failures",
        ),
        ArtifactKind(
            "mock_rules", f"{WORK_DIR}/mock-rules.json", "evalbuilder/mock-rules/v1", "mocks",
            "ADK-style tool fixtures: ordered per-tool rule lists (matchArgs → response). "
            "Handed from the mocks stage to the dataset stage, which stores them for good in `dataset.mocks.tools`.",
            legacy_files=("mock-rules.json", "mocks.json"), wrap_key="tools", tier="work",
            folded_into="dataset.mocks.tools",
        ),
        ArtifactKind(
            "mock_strategies", f"{WORK_DIR}/mock-strategies.json", "evalbuilder/mock-strategies/v1", "mocks",
            "LLM mock strategies (layer 2): the shared backend world and, per strategy, each tool's behaviour, "
            "examples and fallback response. Handed from the mocks stage to the dataset stage, which stores them "
            "for good in `dataset.mocks.strategies`.",
            legacy_files=("mock-strategies.json",), tier="work",
            folded_into="dataset.mocks.strategies",
        ),
        ArtifactKind(
            "coverage_plan", "coverage-plan.json", "evalbuilder/coverage-plan/v1", "dataset",
            "Planned coverage cells (intent × topic × scenario × failure mode) and their counts.",
            legacy_files=("plan.json",), tier="derived", folded_into="dataset.coverage.plan",
        ),
        ArtifactKind(
            "dataset", "dataset.json", "evalbuilder/dataset/v1", "dataset, review",
            "Golden cases (inputs, references, metadata, mocks) with review and publication state.",
        ),
        ArtifactKind(
            "coverage", "coverage.json", "evalbuilder/coverage/v1", "dataset, review",
            "Coverage achieved against the plan: planned vs covered cells, by kind, and gaps.",
            tier="derived", folded_into="dataset.coverage.achieved",
        ),
        ArtifactKind(
            "evaluators", "evaluators.yaml", None, "score",
            "Evaluator specs used for scoring (same format as `evalbuilder score --evaluators`).",
        ),
        ArtifactKind(
            "deployment", "deployment.json", "evalbuilder/deployment/v1", "deploy, teardown",
            "Where the agent ran during inference: target, image, endpoint, resources, status, the commands executed.",
        ),
        ArtifactKind(
            "run", "results/run-{run_id}.json", "evalbuilder/run/v1", "infer",
            "One execution of every approved case: outputs, trajectory, tool calls, node path, errors, per-turn log, execution info.",
        ),
        ArtifactKind(
            "run_progress", f"{WORK_DIR}/run-progress.json", "evalbuilder/run-progress/v1", "infer",
            "Live progress of the infer stage: current repeat, per-case completions, per-intent tallies.",
            legacy_files=("run-progress.json",), tier="work",
        ),
        ArtifactKind(
            "score_report", "results/score-report-{run_id}.json", "evalbuilder/score-report/v1", "score",
            "Evaluator scores for one run: per-metric stats, slices, per-case scores and comments.",
            legacy_files=("results/report-{run_id}.json",), legacy_schemas=("evalbuilder/report/v1",),
        ),
        ArtifactKind(
            "aggregate", "aggregate.json", "evalbuilder/aggregate/v1", "aggregate",
            "Repeat-aware aggregation: pass rates vs thresholds, slices, stability, failing cases, verdict.",
        ),
        ArtifactKind(
            "scenarios", "scenarios.yaml", None, "simulate",
            "Multi-turn simulation scenarios (persona, goal, opening, follow-ups, expectations).",
        ),
        ArtifactKind(
            "simulation", "simulation.json", "evalbuilder/simulation/v1", "simulate",
            "Simulation transcripts with stop reasons and constraint violations.",
        ),
        ArtifactKind(
            "analysis", "analysis.json", "evalbuilder/analysis/v1", "analyze",
            "Generator-written (or deterministic) analysis: failure patterns, weak slices, recommendations.",
        ),
        ArtifactKind(
            "pipeline_state", f"{WORK_DIR}/state.json", "evalbuilder/pipeline-state/v1", "engine",
            "Per-stage status, timing, artifacts and problems; drives `--resume`.",
            legacy_files=("state.json",), tier="work",
        ),
        ArtifactKind(
            "pipeline_report", "report.json", "evalbuilder/pipeline-report/v1", "report",
            "The final report: verdict, stages, metrics, slices, stability, coverage, analysis, problems.",
        ),
        ArtifactKind(
            "pipeline_job", f"{WORK_DIR}/job.json", "evalbuilder/pipeline-job/v1", "ui",
            "A pipeline run launched in the background (UI): mode, argv, pid, timing, exit code, log path.",
            legacy_files=("job.json",), tier="work",
        ),
    ]
}

_BY_SCHEMA: dict[str, ArtifactKind] = {}
for _a in ARTIFACTS.values():
    for _s in ((_a.schema,) if _a.schema else ()) + _a.legacy_schemas:
        _BY_SCHEMA[_s] = _a


_SUBDIRS = sorted({a.file.rsplit("/", 1)[0] for a in ARTIFACTS.values() if "/" in a.file})


def path_for(out_dir: Path, kind: str, run_id: str | None = None) -> Path:
    a = ARTIFACTS[kind]
    if not a.written:
        raise ValueError(
            f"artifact {kind} is derived; it lives in {a.folded_into} and has no file of its own"
        )
    return a.path(out_dir, run_id)


def existing_path(out_dir: Path, kind: str, run_id: str | None = None) -> Path | None:
    """The file backing `kind` in `out_dir` — its current name or a legacy one — or None."""
    a = ARTIFACTS[kind]
    for name in (a.file, *a.legacy_files):
        if "{run_id}" in name and not run_id:
            continue
        p = Path(out_dir) / name.format(run_id=run_id or "")
        if p.exists():
            return p
    return None


def kind_of_file(name: str | Path) -> tuple[ArtifactKind, str] | None:
    """Identify an artifact by (relative) file name → (kind, run_id or "")."""
    text = str(name).replace("\\", "/")
    base = text.rsplit("/", 1)[-1]
    # full path, bare root name, and the bare name under each artifact subdirectory
    candidates = [text, base, *(f"{d}/{base}" for d in _SUBDIRS)]
    for cand in candidates:
        for a in ARTIFACTS.values():
            rid = a.matches(cand)
            if rid is not None:
                return a, rid
    return None


def kind_of_schema(schema_id: str | None) -> ArtifactKind | None:
    return _BY_SCHEMA.get(schema_id or "")


def identify(data: Any, name: str | None = None) -> tuple[ArtifactKind, str] | None:
    """Identify by embedded schema first, then by file name."""
    if isinstance(data, dict):
        a = kind_of_schema(data.get("schema"))
        if a is not None:
            rid = ""
            if a.per_run:
                rid = str(data.get("run_id") or "")
                if not rid and name:
                    hit = kind_of_file(name)
                    rid = hit[1] if hit else ""
            return a, rid
    if name:
        return kind_of_file(name)
    return None


def stamp(kind: str, payload: Any) -> dict:
    """Embed the schema id; wrap dict-shaped payloads under the kind's `wrap_key`."""
    a = ARTIFACTS[kind]
    if a.wrap_key:
        return {"schema": a.schema, a.wrap_key: payload}
    if isinstance(payload, dict):
        return {"schema": a.schema, **{k: v for k, v in payload.items() if k != "schema"}}
    raise TypeError(f"artifact {kind} payload must be a dict, got {type(payload).__name__}")


def unwrap(kind: str, data: Any) -> Any:
    """Inverse of `stamp` that also accepts the pre-convention (unstamped) shape."""
    a = ARTIFACTS[kind]
    if a.wrap_key:
        if isinstance(data, dict) and a.wrap_key in data and (
            data.get("schema") == a.schema or len(data) <= 2
        ):
            return data[a.wrap_key]
        return {k: v for k, v in data.items() if k != "schema"} if isinstance(data, dict) else data
    return data


def artifact_index(out_dir: Path) -> dict[str, Any]:
    """Which artifacts exist in `out_dir`: kind → path (root) or {run_id: path} (per run)."""
    out_dir = Path(out_dir)
    found: dict[str, Any] = {}
    for a in ARTIFACTS.values():
        if a.per_run:
            runs: dict[str, str] = {}
            results = out_dir / "results"
            if results.is_dir():
                for p in sorted(results.iterdir()):
                    rid = a.matches("results/" + p.name)
                    if rid:
                        runs[rid] = str(p)
            if runs:
                found[a.kind] = runs
            continue
        for candidate in (a.file, *a.legacy_files):
            p = out_dir / candidate
            if p.exists():
                found[a.kind] = str(p)
                break
    return found


def convention_table() -> list[dict]:
    """Rows for docs and the UI: kind, tier, file, fold target, schema, stage, description."""
    return [
        {
            "kind": a.kind,
            "tier": a.tier,
            "file": "—" if a.tier == "derived" else a.file,
            "folded_into": a.folded_into or "—",
            "schema": a.schema or "—",
            "stage": a.stage,
            "description": a.description,
        }
        for a in ARTIFACTS.values()
    ]


def kinds_of_tier(*tiers: str) -> list[str]:
    return [a.kind for a in ARTIFACTS.values() if a.tier in tiers]


FOLDED: dict[str, str] = {a.kind: a.folded_into for a in ARTIFACTS.values() if a.folded_into}


def fold_from(kind: str, parents: dict[str, Any]) -> Any:
    """Read a derived kind out of its parent artifact — `parents` maps kind → data.

    Returns None when the parent is absent or does not carry the folded data yet.
    """
    path = FOLDED.get(kind)
    if not path:
        return None
    parent_kind, *keys = path.split(".")
    node: Any = parents.get(parent_kind)
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _set_path(root: dict, dotted: str, value: Any) -> None:
    """Write `value` at a dotted path inside `root`, creating intermediate dicts."""
    node = root
    keys = dotted.split(".")
    for key in keys[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            node[key] = nxt
        node = nxt
    node[keys[-1]] = value


def compact_dir(out_dir: Path, *, dry_run: bool = False) -> dict:
    """Bring an output directory written by an older version onto the current layout.

    Folds every stand-alone copy of data that now lives inside a deliverable into that
    deliverable, moves the scratch tier into `work/`, and refreshes the report's artifact
    index. Idempotent, and a no-op on a directory the current pipeline wrote.
    """
    import json

    out_dir = Path(out_dir)
    folded: list[str] = []
    moved: list[str] = []
    removed: list[str] = []
    parents: dict[str, dict] = {}

    def parent_of(kind: str) -> dict | None:
        if kind not in parents:
            path = existing_path(out_dir, kind)
            parents[kind] = json.loads(path.read_text()) if path else None
        return parents[kind]

    for kind, target in FOLDED.items():
        src = existing_path(out_dir, kind)
        if src is None:
            continue
        parent_kind = target.split(".", 1)[0]
        parent = parent_of(parent_kind)
        if parent is None:
            continue  # nothing to fold into; leave the file alone
        if fold_from(kind, parents) in (None, {}, []):
            data = unwrap(kind, json.loads(src.read_text()))
            if isinstance(data, dict):
                data = {k: v for k, v in data.items() if k != "schema"}  # the stamp names a file; this is not one
            _set_path(parent, target.split(".", 1)[1], data)
            folded.append(f"{src.name} → {target}")
        if ARTIFACTS[kind].tier == "derived":
            removed.append(str(src.relative_to(out_dir)))
            if not dry_run:
                src.unlink()

    for kind, parent in parents.items():
        if parent is not None and not dry_run:
            path = existing_path(out_dir, kind) or path_for(out_dir, kind)
            path.write_text(json.dumps(parent, indent=2, ensure_ascii=False) + "\n")

    for kind in kinds_of_tier("work"):
        src = existing_path(out_dir, kind)
        dest = path_for(out_dir, kind)
        if src is None or src == dest:
            continue
        moved.append(f"{src.relative_to(out_dir)} → {dest.relative_to(out_dir)}")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dest)

    report_path = existing_path(out_dir, "pipeline_report")
    if report_path is not None and not dry_run and (folded or moved or removed):
        report = json.loads(report_path.read_text())
        # Re-index, but keep the prefix the run recorded: this directory may be a copy of
        # the output dir, and the report's paths are provenance, not a lookup table.
        root = Path(report.get("output_dir") or out_dir)
        index = artifact_index(out_dir)
        rebased: dict[str, Any] = {}
        for kind, location in index.items():
            if isinstance(location, dict):
                rebased[kind] = {r: str(root / Path(f).relative_to(out_dir)) for r, f in location.items()}
            else:
                rebased[kind] = str(root / Path(location).relative_to(out_dir))
        report["artifacts"] = rebased
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    return {"folded": folded, "moved": moved, "removed": removed}
