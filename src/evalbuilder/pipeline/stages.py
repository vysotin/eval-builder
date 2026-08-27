"""Pipeline stages: thin glue over the evalbuilder modules, driven by the engine.

`PipelineContext` lazily reloads every artifact from the output directory so a
resumed run (cached stages) has the same data as a fresh one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from evalbuilder import artifacts
from evalbuilder import discover as discovery
from evalbuilder import simulate as sim
from evalbuilder import target as target_mod
from evalbuilder.config import Settings, capability_check, provider_ready
from evalbuilder.evaluators import is_judge_spec, score_run
from evalbuilder.mocking import merge_mock_rules, verify_dataset, with_fallback, wrap_tools
from evalbuilder.pipeline import generator as gen_mod
from evalbuilder.pipeline.aggregate import aggregate as aggregate_runs
from evalbuilder.pipeline.config import PipelineConfig
from evalbuilder.pipeline.engine import Stage, StageStop
from evalbuilder.pipeline.planning import Cell, achieved, plan_cells, summarize_plan
from evalbuilder.pipeline.taxonomy import applicable_failure_types
from evalbuilder.runner import run_dataset
from evalbuilder.schemas import AgentMap, Dataset, Report, RunArtifact, Target

REQUIRED_STAGES = (
    "preflight", "discover", "map", "mocks", "dataset", "review", "verify",
    "run", "score", "aggregate",
)


@dataclass
class PipelineContext:
    config: PipelineConfig
    config_path: Path
    settings: Settings
    log: Callable[[str], None] = lambda msg: None
    generator_factory: Callable[[], gen_mod.Generator] | None = None
    state: Any = None
    _problems: list[dict] = field(default_factory=list)
    _cache: dict[str, Any] = field(default_factory=dict)

    @property
    def problems(self) -> list[dict]:
        """Problems live in state.data so they survive --resume; local until state exists."""
        if self.state is not None:
            stored = self.state.data.setdefault("problems", [])
            if self._problems:
                stored.extend(p for p in self._problems if p not in stored)
                self._problems = []
            return stored
        return self._problems

    # ── paths ──────────────────────────────────────────────────
    @property
    def out_dir(self) -> Path:
        return self.config.output_dir

    def path(self, name: str) -> Path:
        return self.out_dir / name

    @property
    def map_path(self) -> Path:
        return self.path("agent-map.json")

    @property
    def dataset_path(self) -> Path:
        return self.path("dataset.json")

    @property
    def results_dir(self) -> Path:
        return self.path("results")

    # ── problems ───────────────────────────────────────────────
    def problem(self, stage: str, message: str, severity: str = "warning") -> None:
        self.problems.append({"stage": stage, "severity": severity, "message": message})
        self.log(f"{stage}: {severity} — {message}")

    # ── models ─────────────────────────────────────────────────
    def generator(self) -> gen_mod.Generator:
        if "generator" not in self._cache:
            if self.generator_factory is not None:
                self._cache["generator"] = self.generator_factory()
            else:
                from evalbuilder.claude_cli import model_from_spec

                self._cache["generator"] = gen_mod.Generator(
                    model_from_spec(self.config.models.generator), log=self.log
                )
        return self._cache["generator"]

    def agent_model(self):
        spec = self.config.models.agent
        if not spec:
            return None
        if "agent_model" not in self._cache:
            from evalbuilder.claude_cli import model_from_spec

            self._cache["agent_model"] = model_from_spec(spec)
        return self._cache["agent_model"]

    # ── artifacts (lazy, disk-backed) ──────────────────────────
    def _json(self, key: str, path: Path, loader=None):
        if key not in self._cache:
            if not path.exists():
                raise FileNotFoundError(f"missing artifact {path} (stage not run?)")
            data = json.loads(path.read_text())
            self._cache[key] = loader(data) if loader else data
        return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def agent_map(self) -> AgentMap:
        return self._json("agent_map", self.map_path, AgentMap.model_validate)

    def save_agent_map(self, amap: AgentMap) -> None:
        artifacts.save_json(self.map_path, amap)
        self._cache["agent_map"] = amap

    def source_text(self) -> str:
        if "source_text" not in self._cache:
            self._cache["source_text"] = Path(self.config.target.source).read_text()
        return self._cache["source_text"]

    def dataset(self) -> Dataset:
        return self._json("dataset", self.dataset_path, Dataset.model_validate)

    def save_dataset(self, ds: Dataset) -> None:
        errors = artifacts.validate_dataset(ds)
        if errors:
            raise ValueError("dataset invalid: " + "; ".join(errors[:5]))
        artifacts.save_json(self.dataset_path, ds)
        self._cache["dataset"] = ds

    def mock_rules(self) -> dict[str, list[dict]]:
        return self._json("mock_rules", self.path("mocks.json"))

    def applicable(self) -> dict[str, list[str]]:
        return self._json("applicable", self.path("applicable-failures.json"))

    def cells(self) -> list[Cell]:
        if "cells" not in self._cache:
            data = self._json("plan", self.path("plan.json"))
            self._cache["cells"] = [
                Cell(**{k: v for k, v in row.items() if k in Cell.__dataclass_fields__})
                for row in data["cells"]
            ]
        return self._cache["cells"]

    def run_reports(self) -> list[tuple[RunArtifact, Report]]:
        if "run_reports" not in self._cache:
            score_rec = self.state.stages.get("score") if self.state else None
            pairs = (score_rec.artifacts.get("pairs") if score_rec else None) or []
            out = []
            for pair in pairs:
                run = RunArtifact.model_validate(json.loads(Path(pair["run"]).read_text()))
                report = Report.model_validate(json.loads(Path(pair["report"]).read_text()))
                out.append((run, report))
            self._cache["run_reports"] = out
        return self._cache["run_reports"]

    def aggregate(self) -> dict:
        return self._json("aggregate", self.path("aggregate.json"))

    def optional_json(self, name: str):
        p = self.path(name)
        return json.loads(p.read_text()) if p.exists() else None


# ── stage functions ────────────────────────────────────────────


def preflight(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    problems = cfg.problems()
    if problems:
        raise ValueError("config problems: " + "; ".join(problems))
    caps = capability_check(ctx.settings, cfg.target.module)
    if caps["blocking"]:
        raise ValueError("blocking: " + "; ".join(b["issue"] for b in caps["blocking"]))
    models = {}
    for role, spec in (("agent", cfg.models.agent), ("judge", cfg.models.judge), ("generator", cfg.models.generator)):
        if not spec:
            models[role] = {"spec": None, "ready": True, "note": "target default model"}
            continue
        ready, reason = provider_ready(spec)
        models[role] = {"spec": spec, "ready": ready, "reason": reason}
        if not ready:
            if role == "generator":
                raise ValueError(f"generator model {spec!r} unavailable: {reason}")
            if role == "agent":
                raise ValueError(f"agent model {spec!r} unavailable: {reason}")
            ctx.problem("preflight", f"judge model {spec!r} unavailable: {reason}; judge evaluators will error")
    judge_specs = [e for e in cfg.evaluators if is_judge_spec(e)]
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    return {"capabilities": caps["capabilities"], "degraded": caps["degraded"], "models": models,
            "judge_evaluators": len(judge_specs)}


def _live_tools(module) -> list[dict]:
    out = []
    for t in getattr(module, "TOOLS", []) or []:
        schema: dict = {}
        try:
            schema = t.tool_call_schema.model_json_schema()
        except Exception:  # noqa: BLE001
            schema = {"type": "object", "properties": dict(getattr(t, "args", {}) or {})}
        schema.pop("title", None)
        out.append({"name": t.name, "description": t.description or "", "args_schema": schema})
    return out


def discover(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    amap = discovery.discover_from_source(Path(cfg.target.source))
    amap.app.update({"module": cfg.target.module, "factory": cfg.target.factory, "root_node": cfg.target.root_node})
    amap.graph["live"] = discovery.discover_live(cfg.target.module, cfg.target.factory)
    if "error" in amap.graph["live"]:
        ctx.problem("discover", f"live introspection failed: {amap.graph['live']['error']}")
    try:
        module = target_mod.load_target(Target(module=cfg.target.module, factory=cfg.target.factory))
        live = {t["name"]: t for t in _live_tools(module)}
    except Exception as e:  # noqa: BLE001
        ctx.problem("discover", f"could not introspect TOOLS: {type(e).__name__}: {e}")
        live = {}
    for t in amap.tools:
        if t["name"] in live:
            t["args_schema"] = live[t["name"]]["args_schema"] or t["args_schema"]
            t["description"] = live[t["name"]]["description"] or t["description"]
            t["live"] = True
    known = {t["name"] for t in amap.tools}
    for name, t in live.items():
        if name not in known:
            amap.tools.append({**t, "used_by": [], "live": True})
    ctx.save_agent_map(amap)
    return {
        "artifacts": {"agent_map": str(ctx.map_path)},
        "tools": [t["name"] for t in amap.tools],
        "nodes": [n["id"] for n in amap.graph["nodes"]],
        "live_nodes": amap.graph["live"].get("nodes", []),
    }


def author_map(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    amap = ctx.agent_map()
    applicable = applicable_failure_types(
        amap, ctx.source_text(), cfg.constraints, multi_turn=cfg.coverage.multi_turn_share > 0
    )
    artifacts.save_json(ctx.path("applicable-failures.json"), applicable)
    ctx.set("applicable", applicable)
    cleaned, errors = gen_mod.author_map(ctx.generator(), amap, ctx.source_text(), cfg.constraints, applicable)
    if not cleaned["intents"] or not cleaned["scenarios"]:
        raise ValueError("generator produced no valid intents/scenarios: " + "; ".join(errors[:5]))
    for err in errors:
        ctx.problem("map", err)
    amap.intents = cleaned["intents"]
    amap.scenarios = cleaned["scenarios"]
    amap.failure_scenarios = cleaned["failure_scenarios"]
    amap.data_domains["topics"] = cleaned["topics"]
    merged = list(cfg.constraints)
    for c in cleaned["derived_constraints"]:
        if c not in merged:
            merged.append(c)
    amap.constraints = merged
    amap.decisions_needed = [f"unvalidated generator output dropped: {e}" for e in errors]
    ctx.save_agent_map(amap)
    return {
        "artifacts": {"agent_map": str(ctx.map_path)},
        "intents": [i["id"] for i in amap.intents],
        "scenarios": len(amap.scenarios),
        "failure_types": [f["failure_type"] for f in amap.failure_scenarios],
        "topics": amap.data_domains["topics"],
        "constraints": merged,
    }


def author_mocks(ctx: PipelineContext) -> dict:
    amap = ctx.agent_map()
    rules, problems = gen_mod.author_mocks(ctx.generator(), amap.tools)
    for p in problems:
        ctx.problem("mocks", p)
    if ctx.config.mocking.required:
        missing = [t["name"] for t in amap.tools if not rules.get(t["name"])]
        if missing:
            raise ValueError(f"tools without mock rules: {missing}")
    artifacts.save_json(ctx.path("mocks.json"), rules)
    ctx.set("mock_rules", rules)
    return {
        "artifacts": {"mocks": str(ctx.path("mocks.json"))},
        "tools": {name: len(r) for name, r in rules.items()},
    }


def build_dataset(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    amap = ctx.agent_map()
    rules = ctx.mock_rules()
    failure_types = [f["failure_type"] for f in amap.failure_scenarios]
    cells = plan_cells(cfg.coverage, amap, failure_types)
    artifacts.save_json(ctx.path("plan.json"), {"cells": [c.to_dict() for c in cells], "summary": summarize_plan(cells)})
    ctx.set("cells", cells)

    ds = Dataset(
        name=cfg.name,
        dataset_type="final_response",
        target=Target(module=cfg.target.module, factory=cfg.target.factory),
        mocks={"tools": rules, "on_miss": cfg.mocking.on_miss},
    )
    scenario_index = {s["id"]: s for s in amap.scenarios}
    cases, problems = gen_mod.author_cases(
        ctx.generator(), cells, amap, amap.constraints, rules, scenario_index
    )
    for p in problems:
        ctx.problem("dataset", p)
    duplicates = 0
    for raw in cases:
        raw["metadata"].pop("_cell", None)
        for tool, case_rules in (raw["metadata"].get("mocks", {}).get("tools", {}) or {}).items():
            raw["metadata"]["mocks"]["tools"][tool] = with_fallback(case_rules, rules.get(tool, []))
        try:
            artifacts.add_case(ds, raw)
        except ValueError:
            duplicates += 1
    if duplicates:
        ctx.problem("dataset", f"{duplicates} duplicate case(s) dropped")
    if not ds.cases:
        raise ValueError("generator produced no usable cases")
    ctx.save_dataset(ds)
    coverage = achieved(cells, [c.model_dump() for c in ds.cases])
    artifacts.save_json(ctx.path("coverage.json"), coverage)
    return {
        "artifacts": {"dataset": str(ctx.dataset_path), "plan": str(ctx.path("plan.json")), "coverage": str(ctx.path("coverage.json"))},
        "cases": len(ds.cases),
        "planned": coverage["planned"],
        "coverage_pct": coverage["coverage_pct"],
        "by_kind": coverage["by_kind"],
    }


def review(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    ds = ctx.dataset()
    pending = [c for c in ds.cases if c.review.status == "pending"]
    rejected: dict[str, str] = {}
    try:
        reviews = gen_mod.self_review(
            ctx.generator(), [c.model_dump(by_alias=True) for c in pending], ctx.agent_map(), ctx.mock_rules(), ctx.agent_map().constraints
        )
    except Exception as e:  # noqa: BLE001 - self-review is advisory
        ctx.problem("review", f"self-review unavailable: {type(e).__name__}: {e}")
        reviews = {}
    for c in pending:
        verdict = reviews.get(c.id, {})
        if verdict.get("verdict") == "reject":
            rejected[c.id] = verdict.get("reason", "")
    for miss in verify_dataset(ds):
        rejected.setdefault(miss["case"], f"expected call {miss['tool']}({miss['args']}) has no mock rule")
    if rejected:
        artifacts.set_review(ds, list(rejected), "rejected", "pipeline self-review")
        for cid, reason in rejected.items():
            next(c for c in ds.cases if c.id == cid).review.note = f"pipeline self-review: {reason}"
    remaining = [c.id for c in pending if c.id not in rejected]
    if not remaining:
        ctx.save_dataset(ds)
        raise ValueError("every generated case was rejected in self-review")
    if not cfg.review.auto_approve:
        ctx.save_dataset(ds)
        raise StageStop(
            "awaiting_review",
            f"{len(remaining)} case(s) pending; set review.auto_approve (with approved_by) or run "
            f"`evalbuilder review {ctx.dataset_path} --approve ...`",
        )
    note = f"auto-approved via pipeline config by {cfg.review.approved_by}"
    if cfg.review.note:
        note += f": {cfg.review.note}"
    artifacts.set_review(ds, remaining, "approved", note)
    ctx.save_dataset(ds)
    coverage = achieved(ctx.cells(), [c.model_dump() for c in ds.cases if c.review.status == "approved"])
    artifacts.save_json(ctx.path("coverage.json"), coverage)
    return {
        "artifacts": {"dataset": str(ctx.dataset_path)},
        "approved": len(remaining),
        "rejected": rejected,
        "approved_by": cfg.review.approved_by,
        "coverage_pct": coverage["coverage_pct"],
    }


def verify(ctx: PipelineContext) -> dict:
    ds = ctx.dataset()
    misses = verify_dataset(ds, only_approved=True)
    if misses:
        raise ValueError(f"{len(misses)} expected tool call(s) have no mock rule: {misses[:3]}")
    if ctx.config.mocking.required:
        module = target_mod.load_target(ds.target)
        names = [t.name for t in getattr(module, "TOOLS", [])]
        unmocked = [n for n in names if n not in ds.mocks.get("tools", {})]
        if unmocked:
            raise ValueError(f"mocking.required but tools have no rules: {unmocked}")
    approved = [c for c in ds.cases if c.review.status == "approved"]
    if not approved:
        raise ValueError("no approved cases to run")
    return {"approved": len(approved), "mocked_tools": sorted(ds.mocks.get("tools", {}))}


def run(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    ds = ctx.dataset()
    runs = []
    errors = {"agent": 0, "infrastructure": 0}
    for i in range(cfg.runs.repeats):
        ctx.log(f"run: repeat {i + 1}/{cfg.runs.repeats}")
        art = run_dataset(
            ds, ctx.dataset_path, mocked=True, out_dir=ctx.results_dir,
            model=ctx.agent_model(), model_spec=cfg.models.agent, on_miss=cfg.mocking.on_miss,
        )
        infra = [cr for cr in art.case_runs if cr.error_class == "infrastructure"]
        if infra and len(infra) == len(art.case_runs):
            raise RuntimeError(f"every case failed with infrastructure errors: {infra[0].error}")
        errors["agent"] += sum(1 for cr in art.case_runs if cr.error_class == "agent")
        errors["infrastructure"] += len(infra)
        runs.append({"run_id": art.run_id, "path": str(ctx.results_dir / f"run-{art.run_id}.json")})
    return {"artifacts": {"runs": runs}, "repeats": cfg.runs.repeats, "cases": len(ds.cases), "errors": errors}


def score(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    ds = ctx.dataset()
    ctx.path("evaluators.yaml").write_text(yaml.safe_dump({"evaluators": cfg.evaluators}))
    runs = ctx.state.stages["run"].artifacts["runs"]
    pairs = []
    reports = []
    for entry in runs:
        run_art = RunArtifact.model_validate(json.loads(Path(entry["path"]).read_text()))
        report = score_run(run_art, ds, cfg.evaluators, cfg.models.judge)
        report_path = ctx.results_dir / f"report-{run_art.run_id}.json"
        artifacts.save_json(report_path, report)
        pairs.append({"run": entry["path"], "report": str(report_path)})
        reports.append((run_art, report))
    ctx.set("run_reports", reports)
    dead = [
        m for m, v in reports[0][1].metrics.items() if v.get("n") == 0 and v.get("errors")
    ] if reports else []
    if dead:
        ctx.problem("score", f"evaluator(s) produced no scores (all errors): {dead}")
    if reports and not any(v.get("n") for v in reports[0][1].metrics.values()):
        raise RuntimeError("no evaluator produced a single score")
    return {
        "artifacts": {"pairs": pairs, "evaluators": str(ctx.path("evaluators.yaml"))},
        "metrics": {m: v["avg"] for m, v in reports[0][1].metrics.items()} if reports else {},
        "evaluator_errors": {m: v["errors"] for m, v in reports[0][1].metrics.items() if v["errors"]} if reports else {},
    }


def aggregate(ctx: PipelineContext) -> dict:
    judge_metrics = {
        e.get("name") or (e["prompt"].lower().removesuffix("_prompt") if e["type"] == "openevals" else e["type"])
        for e in ctx.config.evaluators if is_judge_spec(e)
    }
    agg = aggregate_runs(ctx.run_reports(), ctx.dataset(), ctx.config.thresholds, judge_metrics=judge_metrics)
    artifacts.save_json(ctx.path("aggregate.json"), agg)
    ctx.set("aggregate", agg)
    return {
        "artifacts": {"aggregate": str(ctx.path("aggregate.json"))},
        "verdict": agg["verdict"],
        "overall_score": agg["overall_score"],
        "metrics": {m: v["pass_rate"] for m, v in agg["metrics"].items()},
        "unstable_cases": len(agg["stability"]["unstable_cases"]),
    }


def simulate(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    ds = ctx.dataset()
    amap = ctx.agent_map()
    scenarios, problems = gen_mod.author_scenarios(ctx.generator(), amap, amap.constraints, ctx.mock_rules())
    for p in problems:
        ctx.problem("simulate", p)
    if not scenarios:
        raise ValueError("generator produced no simulation scenarios")
    scen_path = ctx.path("scenarios.yaml")
    scen_path.write_text(yaml.safe_dump({"scenarios": scenarios}, sort_keys=False))
    scenario_list = sim.load_scenarios(scen_path)
    module = target_mod.load_target(ds.target)
    tools = wrap_tools(list(getattr(module, "TOOLS")), merge_mock_rules(ds.mocks.get("tools", {}), {}), on_miss=cfg.mocking.on_miss)
    graph = target_mod.build_graph(module, ds.target, tools=tools, model=ctx.agent_model())
    user_model = getattr(ctx.generator(), "model", None)
    results = [sim.simulate_scenario(graph, s, user_model=user_model) for s in scenario_list]
    mined = sim.mine_failures(ds, results)
    if mined:
        ctx.save_dataset(ds)
    sim_path = ctx.path("simulation.json")
    artifacts.save_json(sim_path, {"results": results})
    return {
        "artifacts": {"scenarios": str(scen_path), "simulation": str(sim_path)},
        "scenarios": len(results),
        "stop_reasons": {r["scenario_id"]: r["stop_reason"] for r in results},
        "violations": {r["scenario_id"]: r["violations"] for r in results if r["violations"]},
        "mined_pending_cases": mined,
    }


def publish(ctx: PipelineContext) -> dict:
    from evalbuilder.langsmith_io import publish_approved

    ds = ctx.dataset()
    result = publish_approved(
        ds, ctx.dataset_path, ctx.settings,
        dataset_name=ctx.config.langsmith.get("dataset_name") or ctx.config.name,
    )
    ctx.set("dataset", ds)
    return {"artifacts": {"dataset": str(ctx.dataset_path)}, **result}


def analyze(ctx: PipelineContext) -> dict:
    from evalbuilder.pipeline.report import analysis_summary

    summary = analysis_summary(ctx)
    try:
        analysis = gen_mod.author_analysis(ctx.generator(), summary)
        source = "generator"
    except Exception as e:  # noqa: BLE001 - fall back to facts
        ctx.problem("analyze", f"generator analysis failed: {type(e).__name__}: {e}")
        analysis = gen_mod.deterministic_analysis(summary, f"{type(e).__name__}: {e}")
        source = "deterministic"
    analysis["source"] = source
    artifacts.save_json(ctx.path("analysis.json"), analysis)
    ctx.set("analysis", analysis)
    return {"artifacts": {"analysis": str(ctx.path("analysis.json"))}, "source": source,
            "patterns": len(analysis.get("failure_patterns", []))}


def report(ctx: PipelineContext) -> dict:
    from evalbuilder.pipeline.report import build_report

    if ctx.state is not None:  # the snapshot describes this stage too
        ctx.state.record("report").status = "ok"
    data = build_report(ctx)
    path = ctx.path("report.json")
    artifacts.save_json(path, data)
    return {"artifacts": {"report": str(path)}, "verdict": data["verdict"], "overall_score": data["overall_score"]}


def _note(stage: str):
    def recover(ctx: PipelineContext, exc: Exception, attempt: int) -> str:
        return f"attempt {attempt} failed ({type(exc).__name__}); retrying {stage}"

    return recover


def build_stages() -> list[Stage]:
    return [
        Stage("preflight", preflight, retries=0),
        Stage("discover", discover, deps=("preflight",), retries=0),
        Stage("map", author_map, deps=("discover",), recover=_note("map")),
        Stage("mocks", author_mocks, deps=("discover",), recover=_note("mocks")),
        Stage("dataset", build_dataset, deps=("map", "mocks"), recover=_note("dataset")),
        Stage("review", review, deps=("dataset",), recover=_note("review")),
        Stage("verify", verify, deps=("review",), retries=0),
        Stage("run", run, deps=("verify",), recover=_note("run")),
        Stage("score", score, deps=("run",), recover=_note("score")),
        Stage("aggregate", aggregate, deps=("score",), retries=0),
        Stage("simulate", simulate, deps=("review",), optional=True, recover=_note("simulate")),
        Stage("publish", publish, deps=("review",), optional=True),
        Stage("analyze", analyze, deps=("aggregate",), optional=True, retries=0),
        Stage("report", report, always=True, retries=0),
    ]
