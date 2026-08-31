"""Interactive pipeline setup (pure Python, no Streamlit): discover candidate targets,
preview a target's structure without an LLM, build a config from form values, and the
review-loop helpers (feedback, approval, rejections) the UI calls between runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from evalbuilder import artifacts, discover as discovery, tool_schemas
from evalbuilder.pipeline.config import DEFAULT_EVALUATORS, DEFAULT_MODEL, PipelineConfig, load_config
from evalbuilder.pipeline.layout import artifact_index, path_for
from evalbuilder.schemas import Target

EVALUATOR_CHOICES = ("expected_tools", "contains", "contract", "correctness", "trajectory_llm", "json_valid", "trajectory_match")
DEFAULT_TARGET_ROOTS = ("examples",)
DEFAULT_PROJECT_ROOTS = ("eval/pipeline", "docs/examples")
SCRIPTED_SUFFIX = ":default_scripted_model"


def module_for(source: Path, root: Path | None = None) -> str:
    """`examples/incident_desk/agent.py` → `examples.incident_desk.agent`."""
    source = Path(source)
    base = Path(root or ".")
    try:
        rel = source.resolve().relative_to(base.resolve())
    except ValueError:
        rel = source
    return ".".join(rel.with_suffix("").parts)


def discover_targets(roots: tuple[str, ...] | list[str] = DEFAULT_TARGET_ROOTS, cwd: Path | None = None) -> list[dict]:
    """Agent modules under the roots: every `*.py` defining `TOOLS` and `build_agent`."""
    base = Path(cwd or ".")
    found: list[dict] = []
    for root in roots:
        root_path = base / root
        if not root_path.is_dir():
            continue
        for path in sorted(root_path.rglob("*.py")):
            if path.name.startswith("_") or "__pycache__" in path.parts:
                continue
            text = path.read_text(errors="ignore")
            if re.search(r"^TOOLS\s*=", text, re.M) and re.search(r"^def build_agent\(", text, re.M):
                rel = path.relative_to(base)
                found.append({
                    "name": rel.parent.name.replace("_", "-") if rel.parent != Path(".") else rel.stem,
                    "source": str(rel),
                    "module": module_for(rel, base),
                    "scripted_model": f"scripted:{module_for(rel, base)}{SCRIPTED_SUFFIX}"
                    if "def default_scripted_model" in text else None,
                    "pipeline_yaml": str(rel.parent / "pipeline.yaml") if (path.parent / "pipeline.yaml").exists() else None,
                })
    return found


def preview_target(source: str | Path, module: str, factory: str = "build_agent") -> dict:
    """Structure of a target without any LLM: AST discovery + live tool schemas."""
    source = Path(source)
    problems: list[str] = []
    if not source.exists():
        return {"ok": False, "problems": [f"source not found: {source}"], "nodes": [], "tools": [], "edges": [], "conditional_edges": [], "models": [], "live": {}}
    try:
        amap = discovery.discover_from_source(source)
    except SyntaxError as e:
        return {"ok": False, "problems": [f"cannot parse {source}: {e}"], "nodes": [], "tools": [], "edges": [], "conditional_edges": [], "models": [], "live": {}}
    tools = list(amap.tools)
    live_graph: dict = {}
    try:
        from evalbuilder import target as target_mod

        mod = target_mod.load_target(Target(module=module, factory=factory))
        live = {t["name"]: t for t in (tool_schemas.describe_tool(t) for t in getattr(mod, "TOOLS", []) or [])}
        for t in tools:
            if t["name"] in live:
                lt = live[t["name"]]
                t.update({k: lt[k] for k in ("args_schema", "output_schema", "schema_source", "models", "side_effecting") if lt.get(k) or k == "side_effecting"})
                t["edge_cases"] = tool_schemas.edge_cases(t)
                t["live"] = True
        for name, lt in live.items():
            if name not in {t["name"] for t in tools}:
                tools.append({**lt, "used_by": [], "live": True})
        live_graph = discovery.discover_live(module, factory)
        if "error" in live_graph:
            problems.append(f"live graph: {live_graph['error']}")
    except Exception as e:  # noqa: BLE001 - AST results still stand
        problems.append(f"live introspection unavailable: {type(e).__name__}: {e}")
    return {
        "ok": True,
        "problems": problems,
        "source": str(source),
        "module": module,
        "nodes": amap.graph.get("nodes", []),
        "edges": amap.graph.get("edges", []),
        "conditional_edges": amap.graph.get("conditional_edges", []),
        "live": live_graph,
        "tools": tools,
        "models": amap.app.get("models", []),
        "edge_case_count": sum(len(t.get("edge_cases") or []) for t in tools),
    }


def default_form(target: dict | None = None) -> dict:
    """Form defaults for the setup page (a flat dict the page binds to widgets).

    Without a target nothing is selected: source/module/output/config path are empty, so
    the UI has no project until the user picks a target or a folder.
    """
    target = target or {}
    name = target.get("name") or "my-agent"
    return {
        "name": name,
        "source": target.get("source", ""),
        "module": target.get("module", ""),
        "factory": "build_agent",
        "agent_model": DEFAULT_MODEL,
        "judge_model": DEFAULT_MODEL,
        "generator_model": DEFAULT_MODEL,
        "constraints": "",
        "instructions": "",
        "total_cases": 12,
        "happy": 1,
        "failure": 1,
        "per_failure_category": 1,
        "out_of_intent": 2,
        "multi_turn_share": 0.15,
        "per_tool_edge_cases": 1,
        "evaluators": [e["type"] for e in DEFAULT_EVALUATORS],
        "threshold_default": 0.8,
        "slice_min": 0.5,
        "overall_pass": 0.8,
        "repeats": 2,
        "on_miss": "strict",
        "simulate": True,
        "auto_approve": False,
        "approved_by": "",
        "output_dir": f"eval/pipeline/{name}" if target else "",
        "config_path": f"eval/pipeline/{name}.yaml" if target else "",
    }


def form_from_config(cfg: PipelineConfig, output_dir: str | None = None, config_path: str | None = None) -> dict:
    """The inverse of `build_config`: form values for an existing config (so a saved
    project can be edited in the setup page). `output_dir` overrides the config's
    directory when the project was opened from a folder."""
    return {
        "name": cfg.name,
        "source": cfg.target.source,
        "module": cfg.target.module,
        "factory": cfg.target.factory,
        "agent_model": cfg.models.agent or "",
        "judge_model": cfg.models.judge,
        "generator_model": cfg.models.generator,
        "constraints": "\n".join(cfg.constraints),
        "instructions": cfg.instructions,
        "total_cases": cfg.coverage.total_cases,
        "happy": cfg.coverage.per_intent.happy,
        "failure": cfg.coverage.per_intent.failure,
        "per_failure_category": cfg.coverage.per_failure_category,
        "out_of_intent": cfg.coverage.out_of_intent,
        "multi_turn_share": cfg.coverage.multi_turn_share,
        "per_tool_edge_cases": cfg.coverage.per_tool_edge_cases,
        "evaluators": [e.get("type") for e in cfg.evaluators if isinstance(e, dict) and e.get("type") in EVALUATOR_CHOICES],
        "threshold_default": cfg.thresholds.default,
        "slice_min": cfg.thresholds.slice_min,
        "overall_pass": cfg.thresholds.overall_pass,
        "repeats": cfg.runs.repeats,
        "on_miss": cfg.mocking.on_miss,
        "simulate": cfg.stages.simulate,
        "auto_approve": cfg.review.auto_approve,
        "approved_by": cfg.review.approved_by,
        "output_dir": output_dir or str(cfg.output_dir),
        "config_path": config_path or f"eval/pipeline/{cfg.name}.yaml",
    }


# ── projects (an output directory + the config that produced / will produce it) ──


def _try_load(path: str | Path | None) -> PipelineConfig | None:
    if not path or not Path(path).exists():
        return None
    try:
        return load_config(Path(path))
    except Exception:  # noqa: BLE001 - an unreadable config just does not count
        return None


def project_config(out_dir: Path) -> tuple[PipelineConfig | None, str | None]:
    """(config, config_path) for a project folder: the path recorded by the last job,
    then by the report (falling back to the config embedded in it), then a sibling
    `<dir>.yaml`. `config_path` is None when the config only exists inside report.json."""
    out_dir = Path(out_dir)
    job_path = path_for(out_dir, "pipeline_job")
    if job_path.exists():
        try:
            recorded = (json.loads(job_path.read_text()) or {}).get("config_path")
        except ValueError:
            recorded = None
        cfg = _try_load(recorded)
        if cfg is not None:
            return cfg, str(recorded)
    report_path = path_for(out_dir, "pipeline_report")
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text()) or {}
        except ValueError:
            report = {}
        cfg = _try_load(report.get("config_path"))
        if cfg is not None:
            return cfg, str(report["config_path"])
        embedded = report.get("config")
        if isinstance(embedded, dict):
            try:
                return PipelineConfig.model_validate(embedded), None
            except Exception:  # noqa: BLE001
                pass
    sibling = out_dir.parent / f"{out_dir.name}.yaml"
    cfg = _try_load(sibling)
    if cfg is not None:
        return cfg, str(sibling)
    return None, None


def discover_projects(roots: tuple[str, ...] | list[str] = DEFAULT_PROJECT_ROOTS, cwd: Path | None = None) -> list[dict]:
    """Projects under the roots: every directory holding a known artifact or a job
    record, plus every `<root>/*.yaml` config whose output directory has not run yet.
    Each entry: {name, dir, config_path, has_artifacts}; paths relative to `cwd` when
    the roots are."""
    base = Path(cwd or ".")
    found: dict[str, dict] = {}
    for root in roots:
        root_path = base / root
        if not root_path.is_dir():
            continue
        for child in sorted(root_path.iterdir()):
            if child.is_dir() and (artifact_index(child) or path_for(child, "pipeline_job").exists()):
                key = str(child if cwd else child.relative_to(base))
                cfg, cfg_path = project_config(child)
                found[key] = {"name": cfg.name if cfg else child.name, "dir": key, "config_path": cfg_path, "has_artifacts": True}
        for yaml_path in sorted(root_path.glob("*.yaml")):
            cfg = _try_load(yaml_path)
            if cfg is None:
                continue
            out_dir = Path(cfg.output.dir) if cfg.output.dir else base / "eval" / "pipeline" / cfg.name
            key = str(out_dir)
            if key in found:
                found[key]["config_path"] = found[key]["config_path"] or str(yaml_path)
                continue
            found[key] = {"name": cfg.name, "dir": key, "config_path": str(yaml_path), "has_artifacts": False}
    return list(found.values())


def _lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def build_config(form: dict) -> PipelineConfig:
    """A validated `PipelineConfig` from the flat form dict (raises on schema errors)."""
    for key in ("name", "source", "module"):
        if not (form.get(key) or "").strip():
            raise ValueError(f"{key} is required" + (" (pick a target agent)" if key != "name" else ""))
    evaluators = [{"type": t} for t in form.get("evaluators") or []] or [dict(e) for e in DEFAULT_EVALUATORS]
    agent_model = (form.get("agent_model") or "").strip() or None
    data: dict[str, Any] = {
        "name": form["name"].strip(),
        "target": {"source": form["source"].strip(), "module": form["module"].strip(), "factory": form.get("factory") or "build_agent"},
        "models": {"agent": agent_model, "judge": form.get("judge_model") or DEFAULT_MODEL, "generator": form.get("generator_model") or DEFAULT_MODEL},
        "constraints": _lines(form.get("constraints", "")),
        "instructions": (form.get("instructions") or "").strip(),
        "coverage": {
            "total_cases": int(form.get("total_cases", 12)),
            "per_intent": {"happy": int(form.get("happy", 1)), "failure": int(form.get("failure", 1))},
            "per_failure_category": int(form.get("per_failure_category", 1)),
            "out_of_intent": int(form.get("out_of_intent", 2)),
            "multi_turn_share": float(form.get("multi_turn_share", 0.15)),
            "per_tool_edge_cases": int(form.get("per_tool_edge_cases", 1)),
        },
        "evaluators": evaluators,
        "thresholds": {
            "default": float(form.get("threshold_default", 0.8)),
            "slice_min": float(form.get("slice_min", 0.5)),
            "overall_pass": float(form.get("overall_pass", 0.8)),
        },
        "runs": {"repeats": int(form.get("repeats", 2))},
        "mocking": {"required": True, "on_miss": form.get("on_miss") or "strict"},
        "stages": {"simulate": bool(form.get("simulate", True)), "publish": "auto", "max_retries": 1},
        "review": {"auto_approve": bool(form.get("auto_approve", False)), "approved_by": (form.get("approved_by") or "").strip()},
        "output": {"dir": (form.get("output_dir") or "").strip() or None},
    }
    return PipelineConfig.model_validate(data)


def validate_text(text: str) -> tuple[PipelineConfig | None, list[str]]:
    """(config, problems) for YAML text: schema errors first, then semantic problems."""
    from evalbuilder.pipeline.config import parse_config

    try:
        cfg = parse_config(text, "config")
    except ValueError as e:
        return None, [line.strip() for line in str(e).splitlines()[1:]] or [str(e)]
    return cfg, cfg.problems()


# ── review loop ────────────────────────────────────────────────


def add_feedback(config_path: Path, note: str, from_stage: str = "dataset") -> PipelineConfig:
    """Append a reviewer comment to the config file (kept in `feedback`, never overwritten)."""
    cfg = load_config(config_path)
    cfg.add_feedback(note, from_stage)
    cfg.save(config_path)
    return cfg


def approve_in_config(config_path: Path, approved_by: str, note: str = "") -> PipelineConfig:
    """Record the human authorization to approve generated cases, then save."""
    if not approved_by.strip():
        raise ValueError("approved_by is required to authorize the review stage")
    cfg = load_config(config_path)
    cfg.review.auto_approve = True
    cfg.review.approved_by = approved_by.strip()
    if note.strip():
        cfg.review.note = note.strip()
    cfg.save(config_path)
    return cfg


def reject_cases(out_dir: Path, ids: list[str], note: str, by: str) -> int:
    """Mark dataset cases rejected before the evaluation resumes; returns the count."""
    if not ids:
        return 0
    path = path_for(Path(out_dir), "dataset")
    ds = artifacts.load_dataset(path)
    n = artifacts.set_review(ds, ids, "rejected", f"rejected in UI by {by}: {note}".rstrip(": "))
    artifacts.save_json(path, ds)
    return n


def dataset_summary(out_dir: Path) -> dict | None:
    """Counts the review section shows: cases by kind / status, schema-edge cases, mocked tools."""
    path = path_for(Path(out_dir), "dataset")
    if not path.exists():
        return None
    ds = artifacts.load_dataset(path)
    by_status: dict[str, int] = {}
    by_failure: dict[str, int] = {}
    edges: list[dict] = []
    for c in ds.cases:
        by_status[c.review.status] = by_status.get(c.review.status, 0) + 1
        fm = c.metadata.get("failure_mode", "none")
        by_failure[fm] = by_failure.get(fm, 0) + 1
        if c.metadata.get("edge"):
            edges.append({
                "id": c.id, "tool": c.metadata.get("tool"), "edge": c.metadata["edge"].get("kind"),
                "field": c.metadata["edge"].get("field"), "failure_mode": fm,
                "input": str(((c.inputs.get("messages") or [{}])[0]).get("content", ""))[:120],
                "mock_override": bool((c.metadata.get("mocks") or {}).get("tools")),
            })
    return {
        "cases": len(ds.cases),
        "by_status": by_status,
        "by_failure_mode": by_failure,
        "schema_edge_cases": edges,
        "mocked_tools": sorted((ds.mocks.get("tools") or {}).keys()),
        "case_ids": [c.id for c in ds.cases],
        "inputs": {c.id: str(((c.inputs.get("messages") or [{}])[0]).get("content", ""))[:90] for c in ds.cases},
    }
