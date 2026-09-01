"""Artifact naming convention — the one registry of every pipeline artifact.

Rules:
- root-level artifacts are `<kind>.json` (or `.yaml`) inside the pipeline output dir;
- per-run artifacts are `results/<kind>-<run_id>.json`;
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

    @property
    def per_run(self) -> bool:
        return "{run_id}" in self.file

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
            wrap_key="failure_types",
        ),
        ArtifactKind(
            "mock_rules", "mock-rules.json", "evalbuilder/mock-rules/v1", "mocks",
            "ADK-style tool fixtures: ordered per-tool rule lists (matchArgs → response).",
            legacy_files=("mocks.json",), wrap_key="tools",
        ),
        ArtifactKind(
            "mock_strategies", "mock-strategies.json", "evalbuilder/mock-strategies/v1", "mocks",
            "LLM mock strategies (layer 2): the shared backend world and, per strategy, each tool's behaviour, examples and fallback response.",
        ),
        ArtifactKind(
            "coverage_plan", "coverage-plan.json", "evalbuilder/coverage-plan/v1", "dataset",
            "Planned coverage cells (intent × topic × scenario × failure mode) and their counts.",
            legacy_files=("plan.json",),
        ),
        ArtifactKind(
            "dataset", "dataset.json", "evalbuilder/dataset/v1", "dataset, review",
            "Golden cases (inputs, references, metadata, mocks) with review and publication state.",
        ),
        ArtifactKind(
            "coverage", "coverage.json", "evalbuilder/coverage/v1", "dataset, review",
            "Coverage achieved against the plan: planned vs covered cells, by kind, and gaps.",
        ),
        ArtifactKind(
            "evaluators", "evaluators.yaml", None, "score",
            "Evaluator specs used for scoring (same format as `evalbuilder score --evaluators`).",
        ),
        ArtifactKind(
            "run", "results/run-{run_id}.json", "evalbuilder/run/v1", "run",
            "One execution of every approved case: outputs, trajectory, tool calls, node path, errors.",
        ),
        ArtifactKind(
            "run_progress", "run-progress.json", "evalbuilder/run-progress/v1", "run",
            "Live progress of the run stage: current repeat, per-case completions, per-intent tallies.",
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
            "pipeline_state", "state.json", "evalbuilder/pipeline-state/v1", "engine",
            "Per-stage status, timing, artifacts and problems; drives `--resume`.",
        ),
        ArtifactKind(
            "pipeline_report", "report.json", "evalbuilder/pipeline-report/v1", "report",
            "The final report: verdict, stages, metrics, slices, stability, coverage, analysis, problems.",
        ),
        ArtifactKind(
            "pipeline_job", "job.json", "evalbuilder/pipeline-job/v1", "ui",
            "A pipeline run launched in the background (UI): mode, argv, pid, timing, exit code, log path.",
        ),
    ]
}

_BY_SCHEMA: dict[str, ArtifactKind] = {}
for _a in ARTIFACTS.values():
    for _s in ((_a.schema,) if _a.schema else ()) + _a.legacy_schemas:
        _BY_SCHEMA[_s] = _a


def path_for(out_dir: Path, kind: str, run_id: str | None = None) -> Path:
    return ARTIFACTS[kind].path(out_dir, run_id)


def kind_of_file(name: str | Path) -> tuple[ArtifactKind, str] | None:
    """Identify an artifact by (relative) file name → (kind, run_id or "")."""
    text = str(name).replace("\\", "/")
    base = text.rsplit("/", 1)[-1]
    candidates = [text, base, "results/" + base]  # full path, bare root name, bare per-run name
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
    """Rows for docs and the UI: kind, file, schema, stage, description."""
    return [
        {"kind": a.kind, "file": a.file, "schema": a.schema or "—", "stage": a.stage, "description": a.description}
        for a in ARTIFACTS.values()
    ]
