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
from evalbuilder import tool_schemas
from evalbuilder.config import Settings, capability_check, provider_ready
from evalbuilder.evaluators import is_judge_spec, score_run
from evalbuilder.mocking import ledger_totals, merge_mock_rules, verify_dataset, verify_summary, with_fallback, wrap_tools
from evalbuilder.pipeline import generator as gen_mod
from evalbuilder.pipeline.aggregate import aggregate as aggregate_runs
from evalbuilder.pipeline.config import PipelineConfig
from evalbuilder.pipeline.engine import Stage, StageStop
from evalbuilder.pipeline.layout import artifact_index, path_for, stamp, unwrap
from evalbuilder.pipeline.planning import Cell, achieved, plan_cells, summarize_plan
from evalbuilder.pipeline.taxonomy import applicable_failure_types
from evalbuilder.runner import build_engine, run_dataset, tool_specs_of
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

    def artifact(self, kind: str, run_id: str | None = None) -> Path:
        """Path of an artifact by kind (see pipeline/layout.py for the convention)."""
        return path_for(self.out_dir, kind, run_id)

    def save_artifact(self, kind: str, payload, run_id: str | None = None) -> Path:
        """Write a JSON artifact with its schema id embedded; returns the path."""
        path = self.artifact(kind, run_id)
        artifacts.save_json(path, stamp(kind, payload))
        return path

    @property
    def map_path(self) -> Path:
        return self.artifact("agent_map")

    @property
    def dataset_path(self) -> Path:
        return self.artifact("dataset")

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

    def mock_model(self):
        """The model behind the LLM mock engine (only when `mocking.on_miss` is `llm`)."""
        if not self.config.llm_mocking:
            return None
        if "mock_model" not in self._cache:
            spec = self.config.mock_model_spec
            if spec == self.config.models.generator and "generator" in self._cache:
                self._cache["mock_model"] = self._cache["generator"].model
            else:
                from evalbuilder.claude_cli import model_from_spec

                self._cache["mock_model"] = model_from_spec(spec)
        return self._cache["mock_model"]

    # ── artifacts (lazy, disk-backed) ──────────────────────────
    def _json(self, key: str, path: Path, loader=None):
        if key not in self._cache:
            if not path.exists():
                raise FileNotFoundError(f"missing artifact {path} (stage not run?)")
            data = json.loads(path.read_text())
            self._cache[key] = loader(data) if loader else data
        return self._cache[key]

    def _artifact_json(self, key: str, kind: str):
        """Load a root artifact by kind, falling back to its legacy file name."""
        path = self.artifact(kind)
        if not path.exists():
            legacy = artifact_index(self.out_dir).get(kind)
            if isinstance(legacy, str):
                path = Path(legacy)
        return self._json(key, path, lambda data: unwrap(kind, data))

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
        return self._artifact_json("mock_rules", "mock_rules")

    def mock_strategies(self) -> dict | None:
        """`mock-strategies.json` without its schema stamp, or None when the stage did not write it."""
        data = self.optional_json("mock_strategies")
        if not isinstance(data, dict):
            return None
        return {k: v for k, v in data.items() if k != "schema"}

    def mock_block(self) -> dict:
        """The dataset's `mocks` block: rules, policy, the LLM engine settings and strategies."""
        cfg = self.config.mocking
        block: dict[str, Any] = {"tools": self.mock_rules(), "on_miss": cfg.on_miss, "strategy": cfg.strategy}
        strategies = self.mock_strategies()
        if strategies:
            block["strategies"] = strategies
        if self.config.llm_mocking:
            block["llm"] = {"model": self.config.mock_model_spec, "on_invalid": cfg.on_invalid, "max_repairs": cfg.max_repairs}
        return block

    def applicable(self) -> dict[str, list[str]]:
        return self._artifact_json("applicable", "applicable_failures")

    def cells(self) -> list[Cell]:
        if "cells" not in self._cache:
            data = self._artifact_json("plan", "coverage_plan")
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
        return self._artifact_json("aggregate", "aggregate")

    def optional_json(self, kind: str):
        """A root artifact by kind (unwrapped), or None when it does not exist yet."""
        try:
            return self._artifact_json(f"optional:{kind}", kind)
        except FileNotFoundError:
            return None


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
    roles = [("agent", cfg.models.agent), ("judge", cfg.models.judge), ("generator", cfg.models.generator)]
    if cfg.llm_mocking:
        roles.append(("mock", cfg.mock_model_spec))
    for role, spec in roles:
        if not spec:
            models[role] = {"spec": None, "ready": True, "note": "target default model"}
            continue
        ready, reason = provider_ready(spec)
        models[role] = {"spec": spec, "ready": ready, "reason": reason}
        if not ready:
            if role in ("generator", "agent", "mock"):
                raise ValueError(f"{role} model {spec!r} unavailable: {reason}")
            ctx.problem("preflight", f"judge model {spec!r} unavailable: {reason}; judge evaluators will error")
    judge_specs = [e for e in cfg.evaluators if is_judge_spec(e)]
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    return {"capabilities": caps["capabilities"], "degraded": caps["degraded"], "models": models,
            "judge_evaluators": len(judge_specs), "mocking": cfg.mocking.on_miss}


def _live_tools(module) -> list[dict]:
    """Schemas, models, side effects and edge cases of every live tool in `TOOLS`."""
    out = []
    for t in getattr(module, "TOOLS", []) or []:
        try:
            out.append(tool_schemas.describe_tool(t))
        except Exception:  # noqa: BLE001 - keep the bare minimum for odd tool types
            out.append({"name": t.name, "description": getattr(t, "description", "") or "",
                        "args_schema": {"type": "object", "properties": dict(getattr(t, "args", {}) or {})},
                        "output_schema": {}, "schema_source": "annotations", "models": [], "side_effecting": False,
                        "kind": "tool", "mockable": True, "edge_cases": []})
    return out


def _live_skills(module) -> list[dict]:
    """Skills a module exposes as `SKILLS` (Skill objects or duck-typed name/description/body)."""
    from evalbuilder import skills as skills_mod

    out = []
    for s in getattr(module, "SKILLS", None) or []:
        if isinstance(s, skills_mod.Skill):
            out.append(skills_mod.describe_skill(s))
            continue
        name = getattr(s, "name", None) if not isinstance(s, dict) else s.get("name")
        if not name:
            continue
        get = (lambda k, d="": s.get(k, d)) if isinstance(s, dict) else (lambda k, d="": getattr(s, k, d))
        out.append(skills_mod.describe_skill(skills_mod.Skill(
            name=str(name), description=str(get("description")), path=str(get("path")), dir=str(get("dir")),
            body=str(get("body") or get("prompt")),
        )))
    return out


def mockable_tools(amap: AgentMap) -> list[dict]:
    """Tools the mock layers may answer (skill loaders and other local tools are excluded)."""
    return [t for t in amap.tools if t.get("mockable", True)]


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
            lt = live[t["name"]]
            t["args_schema"] = lt["args_schema"] or t["args_schema"]
            t["description"] = lt["description"] or t["description"]
            if lt.get("output_schema"):
                t["output_schema"] = lt["output_schema"]
            t["schema_source"] = lt.get("schema_source") or t.get("schema_source", "ast")
            t["models"] = lt.get("models") or t.get("models", [])
            t["side_effecting"] = bool(lt.get("side_effecting", t.get("side_effecting", False)))
            if lt.get("kind"):
                t["kind"], t["mockable"] = lt["kind"], bool(lt.get("mockable", lt["kind"] == "tool"))
            t["edge_cases"] = tool_schemas.edge_cases(t) if t.get("mockable", True) else []
            t["live"] = True
    known = {t["name"] for t in amap.tools}
    for name, t in live.items():
        if name not in known:
            amap.tools.append({**t, "used_by": [], "live": True})
    if cfg.coverage.per_tool_edge_cases <= 0:
        for t in amap.tools:
            t["edge_cases"] = []
    try:
        live_skills = _live_skills(module) if live else []
    except Exception as e:  # noqa: BLE001 - AST skills still stand
        ctx.problem("discover", f"could not introspect SKILLS: {type(e).__name__}: {e}")
        live_skills = []
    known_skills = {s["name"] for s in amap.skills}
    for entry in live_skills:
        if entry["name"] not in known_skills:
            amap.skills.append({**entry, "live": True})
    ctx.save_agent_map(amap)
    return {
        "artifacts": {"agent_map": str(ctx.map_path)},
        "tools": [t["name"] for t in amap.tools],
        "skills": [s["name"] for s in amap.skills],
        "nodes": [n["id"] for n in amap.graph["nodes"]],
        "live_nodes": amap.graph["live"].get("nodes", []),
        "schemas": {
            t["name"]: {"source": t.get("schema_source"), "models": t.get("models", []),
                        "output": bool(t.get("output_schema")), "edge_cases": len(t.get("edge_cases") or [])}
            for t in amap.tools
        },
    }


def author_map(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    amap = ctx.agent_map()
    applicable = applicable_failure_types(
        amap, ctx.source_text(), cfg.constraints, multi_turn=cfg.coverage.multi_turn_share > 0
    )
    ctx.save_artifact("applicable_failures", applicable)
    ctx.set("applicable", applicable)
    cleaned, errors = gen_mod.author_map(ctx.generator(), amap, ctx.source_text(), cfg.constraints, applicable, guidance=cfg.guidance())
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
        "skills": {s["name"]: [sc["id"] for sc in amap.scenarios if s["name"] in (sc.get("skills") or [])] for s in amap.skills},
    }


def author_mocks(ctx: PipelineContext) -> dict:
    cfg = ctx.config.mocking
    amap = ctx.agent_map()
    tools = mockable_tools(amap)
    llm = ctx.config.llm_mocking
    # Under `on_miss: llm` the wildcard default stays out: keyed variants are deterministic,
    # everything else is answered by the LLM mock engine from the strategies.
    rules, problems = gen_mod.author_mocks(ctx.generator(), tools, guidance=ctx.config.guidance(), wildcard=not llm)
    for p in problems:
        ctx.problem("mocks", p)
    path = ctx.save_artifact("mock_rules", rules)
    ctx.set("mock_rules", rules)
    details: dict[str, Any] = {
        "artifacts": {"mock_rules": str(path)},
        "tools": {name: len(r) for name, r in rules.items()},
        "on_miss": cfg.on_miss,
        "skipped_tools": [t["name"] for t in amap.tools if not t.get("mockable", True)],
    }
    strategies = None
    if cfg.strategies or llm:
        strategies, sproblems = gen_mod.author_strategies(ctx.generator(), tools, rules, guidance=ctx.config.guidance())
        for p in sproblems:
            ctx.problem("mocks", p)
        spath = ctx.save_artifact("mock_strategies", strategies)
        ctx._cache.pop("optional:mock_strategies", None)
        details["artifacts"]["mock_strategies"] = str(spath)
        details["strategies"] = {sid: sorted(s.get("tools") or {}) for sid, s in strategies["strategies"].items()}
    if cfg.required:
        covered = set((strategies or {}).get("strategies", {}).get("default", {}).get("tools", {})) if llm else set()
        missing = [t["name"] for t in tools if not rules.get(t["name"]) and t["name"] not in covered]
        if missing:
            raise ValueError(f"tools without mock rules{' or a default strategy' if llm else ''}: {missing}")
    return details


def build_dataset(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    amap = ctx.agent_map()
    rules = ctx.mock_rules()
    failure_types = [f["failure_type"] for f in amap.failure_scenarios]
    cells = plan_cells(cfg.coverage, amap, failure_types)
    plan_path = ctx.save_artifact("coverage_plan", {"cells": [c.to_dict() for c in cells], "summary": summarize_plan(cells)})
    ctx.set("cells", cells)

    ds = Dataset(
        name=cfg.name,
        dataset_type="final_response",
        target=Target(module=cfg.target.module, factory=cfg.target.factory),
        mocks=ctx.mock_block(),
    )
    scenario_index = {s["id"]: s for s in amap.scenarios}
    cases, problems = gen_mod.author_cases(
        ctx.generator(), cells, amap, amap.constraints, rules, scenario_index, guidance=cfg.guidance(),
        strategies=ctx.mock_strategies(),
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
    coverage = achieved(cells, [c.model_dump() for c in ds.cases], skills=amap.skills, scenarios=amap.scenarios)
    coverage_path = ctx.save_artifact("coverage", coverage)
    return {
        "artifacts": {"dataset": str(ctx.dataset_path), "coverage_plan": str(plan_path), "coverage": str(coverage_path)},
        "cases": len(ds.cases),
        "planned": coverage["planned"],
        "coverage_pct": coverage["coverage_pct"],
        "by_kind": coverage["by_kind"],
    }


def review(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    ds = ctx.dataset()
    pending = [c for c in ds.cases if c.review.status == "pending"]
    if not pending:
        approved = [c for c in ds.cases if c.review.status == "approved"]
        if not approved:
            raise ValueError("dataset has no pending or approved cases")
        amap = ctx.agent_map()
        coverage = achieved(ctx.cells(), [c.model_dump() for c in approved], skills=amap.skills, scenarios=amap.scenarios)
        coverage_path = ctx.save_artifact("coverage", coverage)
        return {
            "artifacts": {"dataset": str(ctx.dataset_path), "coverage": str(coverage_path)},
            "already_reviewed": True, "approved": len(approved), "rejected": {},
            "approved_by": "reviewed before this run", "coverage_pct": coverage["coverage_pct"],
        }
    rejected: dict[str, str] = {}
    try:
        reviews = gen_mod.self_review(
            ctx.generator(), [c.model_dump(by_alias=True) for c in pending], ctx.agent_map(), ctx.mock_rules(),
            ctx.agent_map().constraints, guidance=cfg.guidance(),
        )
    except Exception as e:  # noqa: BLE001 - self-review is advisory
        ctx.problem("review", f"self-review unavailable: {type(e).__name__}: {e}")
        reviews = {}
    for c in pending:
        verdict = reviews.get(c.id, {})
        if verdict.get("verdict") == "reject":
            rejected[c.id] = verdict.get("reason", "")
    for miss in verify_summary(ds)["misses"]:  # under on_miss: llm a miss is answered by the engine
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
    amap = ctx.agent_map()
    coverage = achieved(ctx.cells(), [c.model_dump() for c in ds.cases if c.review.status == "approved"], skills=amap.skills, scenarios=amap.scenarios)
    coverage_path = ctx.save_artifact("coverage", coverage)
    return {
        "artifacts": {"dataset": str(ctx.dataset_path), "coverage": str(coverage_path)},
        "approved": len(remaining),
        "rejected": rejected,
        "approved_by": cfg.review.approved_by,
        "coverage_pct": coverage["coverage_pct"],
    }


def verify(ctx: PipelineContext) -> dict:
    ds = ctx.dataset()
    summary = verify_summary(ds, only_approved=True)
    if summary["misses"]:
        raise ValueError(f"{len(summary['misses'])} expected tool call(s) have no mock rule: {summary['misses'][:3]}")
    llm = summary["policy"] == "llm"
    strategy_tools = sorted((ds.mocks.get("strategies") or {}).get("strategies", {}).get("default", {}).get("tools", {})) if llm else []
    if ctx.config.mocking.required:
        from evalbuilder.mocking import mockable

        module = target_mod.load_target(ds.target)
        names = [t.name for t in getattr(module, "TOOLS", []) if mockable(t)]
        unmocked = [n for n in names if n not in ds.mocks.get("tools", {}) and n not in strategy_tools]
        if unmocked:
            raise ValueError(f"mocking.required but tools have no rules{' or a default strategy' if llm else ''}: {unmocked}")
    if llm and not (ds.mocks.get("llm") or {}).get("model"):
        raise ValueError("mocks.on_miss is llm but the dataset names no mock model (mocks.llm.model)")
    approved = [c for c in ds.cases if c.review.status == "approved"]
    if not approved:
        raise ValueError("no approved cases to run")
    return {
        "approved": len(approved), "mocked_tools": sorted(ds.mocks.get("tools", {})), "on_miss": summary["policy"],
        "llm_answered_calls": len(summary["llm_answered"]), "strategy_tools": strategy_tools,
        "strategies_used": sorted({(c.metadata.get("mocks") or {}).get("strategy") or ds.mocks.get("strategy") or "default" for c in approved}) if llm else [],
    }


def run(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    ds = ctx.dataset()
    runs = []
    errors = {"agent": 0, "infrastructure": 0}
    approved = [c for c in ds.cases if c.review.status == "approved"]
    by_intent: dict[str, dict] = {}
    for c in approved:
        by_intent.setdefault(c.metadata.get("intent") or "", {"done": 0, "total": 0, "errors": 0})["total"] += 1
    prog = {
        "repeats": cfg.runs.repeats, "repeat": 0, "parallel_intents": cfg.runs.parallel_intents,
        "cases_total": len(approved), "cases_done": 0,
        "overall_total": len(approved) * cfg.runs.repeats, "overall_done": 0,
        "by_intent": by_intent, "runs": [], "current": [], "mock_calls": {},
    }
    mock_calls: dict[str, int] = {}

    def on_case(evt: dict) -> None:  # serialized by the runner; keeps run-progress.json live
        prog["cases_done"] = evt["completed"]
        prog["overall_done"] += 1
        rec = prog["by_intent"].setdefault(evt["intent"], {"done": 0, "total": 0, "errors": 0})
        rec["done"] += 1
        if evt["error_class"] != "none":
            rec["errors"] += 1
        prog["current"].append({k: evt[k] for k in ("case_id", "intent", "error_class", "error")})
        ctx.save_artifact("run_progress", prog)

    for i in range(cfg.runs.repeats):
        prog.update(repeat=i + 1, cases_done=0, current=[])
        for rec in prog["by_intent"].values():
            rec.update(done=0, errors=0)
        ctx.save_artifact("run_progress", prog)
        ctx.log(f"run: repeat {i + 1}/{cfg.runs.repeats}")
        art = run_dataset(
            ds, ctx.dataset_path, mocked=True, out_dir=ctx.results_dir,
            model=ctx.agent_model(), model_spec=cfg.models.agent, on_miss=cfg.mocking.on_miss,
            max_workers=cfg.runs.parallel_intents, progress=on_case,
            mock_model=ctx.mock_model(), mock_model_spec=cfg.mock_model_spec if cfg.llm_mocking else None,
            strategy=cfg.mocking.strategy,
        )
        infra = [cr for cr in art.case_runs if cr.error_class == "infrastructure"]
        if infra and len(infra) == len(art.case_runs):
            raise RuntimeError(f"every case failed with infrastructure errors: {infra[0].error}")
        errors["agent"] += sum(1 for cr in art.case_runs if cr.error_class == "agent")
        errors["infrastructure"] += len(infra)
        runs.append({"run_id": art.run_id, "path": str(ctx.artifact("run", art.run_id))})
        for layer, n in ((art.mocking or {}).get("calls") or {}).items():
            mock_calls[layer] = mock_calls.get(layer, 0) + n
        prog["runs"] = list(runs)
        prog["mock_calls"] = dict(mock_calls)
        ctx.save_artifact("run_progress", prog)
    if mock_calls.get("invalid"):
        ctx.problem("run", f"{mock_calls['invalid']} LLM mock response(s) failed output-schema validation "
                           f"({mock_calls.get('fallback', 0)} answered by the strategy fallback, {mock_calls.get('error', 0)} raised)")
    return {"artifacts": {"runs": runs, "run_progress": str(ctx.artifact("run_progress"))},
            "repeats": cfg.runs.repeats, "parallel_intents": cfg.runs.parallel_intents,
            "cases": len(ds.cases), "errors": errors,
            "mocking": {"on_miss": cfg.mocking.on_miss, "model": cfg.mock_model_spec if cfg.llm_mocking else None,
                        "strategy": cfg.mocking.strategy if cfg.llm_mocking else None, "calls": mock_calls}}


def score(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    ds = ctx.dataset()
    ctx.artifact("evaluators").write_text(yaml.safe_dump({"evaluators": cfg.evaluators}))
    runs = ctx.state.stages["run"].artifacts["runs"]
    pairs = []
    reports = []
    for entry in runs:
        run_art = RunArtifact.model_validate(json.loads(Path(entry["path"]).read_text()))
        report = score_run(run_art, ds, cfg.evaluators, cfg.models.judge,
                           max_workers=cfg.runs.parallel_scoring)
        report_path = ctx.artifact("score_report", run_art.run_id)
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
        "artifacts": {"pairs": pairs, "evaluators": str(ctx.artifact("evaluators"))},
        "parallel_scoring": cfg.runs.parallel_scoring,
        "metrics": {m: v["avg"] for m, v in reports[0][1].metrics.items()} if reports else {},
        "evaluator_errors": {m: v["errors"] for m, v in reports[0][1].metrics.items() if v["errors"]} if reports else {},
    }


def aggregate(ctx: PipelineContext) -> dict:
    judge_metrics = {
        e.get("name") or (e["prompt"].lower().removesuffix("_prompt") if e["type"] == "openevals" else e["type"])
        for e in ctx.config.evaluators if is_judge_spec(e)
    }
    agg = aggregate_runs(ctx.run_reports(), ctx.dataset(), ctx.config.thresholds, judge_metrics=judge_metrics)
    agg_path = ctx.save_artifact("aggregate", agg)
    ctx.set("aggregate", agg)
    return {
        "artifacts": {"aggregate": str(agg_path)},
        "verdict": agg["verdict"],
        "overall_score": agg["overall_score"],
        "metrics": {m: v["pass_rate"] for m, v in agg["metrics"].items()},
        "unstable_cases": len(agg["stability"]["unstable_cases"]),
    }


def simulate(ctx: PipelineContext) -> dict:
    cfg = ctx.config
    ds = ctx.dataset()
    amap = ctx.agent_map()
    scenarios, problems = gen_mod.author_scenarios(ctx.generator(), amap, amap.constraints, ctx.mock_rules(), guidance=cfg.guidance(),
                                                   strategies=ctx.mock_strategies())
    for p in problems:
        ctx.problem("simulate", p)
    if not scenarios:
        raise ValueError("generator produced no simulation scenarios")
    scen_path = ctx.artifact("scenarios")
    scen_path.write_text(yaml.safe_dump({"scenarios": scenarios}, sort_keys=False))
    scenario_list = sim.load_scenarios(scen_path)
    module = target_mod.load_target(ds.target)
    agent_model = ctx.agent_model()
    llm = cfg.llm_mocking
    tool_specs = tool_specs_of(module) if llm else {}
    mock_model = ctx.mock_model()

    def graph_factory(scenario):  # a fresh graph (and tool wrappers) per scenario, so scenarios can run concurrently
        ledger: list[dict] = []
        engine = build_engine(mock_model, ds, tool_specs, strategy=scenario.get("mock_strategy") or cfg.mocking.strategy) if llm else None
        tools = wrap_tools(list(getattr(module, "TOOLS")), merge_mock_rules(ds.mocks.get("tools", {}), {}),
                           on_miss=cfg.mocking.on_miss, engine=engine, ledger=ledger)
        return target_mod.build_graph(module, ds.target, tools=tools, model=agent_model), ledger

    user_model = getattr(ctx.generator(), "model", None)
    results = sim.simulate_scenarios(graph_factory, scenario_list, user_model=user_model,
                                     max_workers=cfg.runs.parallel_simulations)
    mined = sim.mine_failures(ds, results)
    if mined:
        ctx.save_dataset(ds)
    sim_path = ctx.save_artifact("simulation", {"results": results})
    totals: dict[str, int] = {}
    for r in results:
        for layer, n in (r.get("mock_calls") or {}).items():
            totals[layer] = totals.get(layer, 0) + n
    return {
        "artifacts": {"scenarios": str(scen_path), "simulation": str(sim_path)},
        "parallel_simulations": cfg.runs.parallel_simulations,
        "scenarios": len(results),
        "stop_reasons": {r["scenario_id"]: r["stop_reason"] for r in results},
        "violations": {r["scenario_id"]: r["violations"] for r in results if r["violations"]},
        "mined_pending_cases": mined,
        "mock_calls": totals,
        "strategies": {r["scenario_id"]: r.get("mock_strategy") for r in results if r.get("mock_strategy")},
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
    analysis_path = ctx.save_artifact("analysis", analysis)
    ctx.set("analysis", analysis)
    return {"artifacts": {"analysis": str(analysis_path)}, "source": source,
            "patterns": len(analysis.get("failure_patterns", []))}


def report(ctx: PipelineContext) -> dict:
    from evalbuilder.pipeline.report import build_report

    if ctx.state is not None:  # the snapshot describes this stage too
        ctx.state.record("report").status = "ok"
    data = build_report(ctx)
    path = ctx.artifact("pipeline_report")
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
