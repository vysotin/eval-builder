"""Report assembly (`evalbuilder/pipeline-report/v1`) and the pipeline entry point."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from evalbuilder.config import Settings
from evalbuilder.pipeline.config import PipelineConfig, load_config
from evalbuilder.pipeline.engine import OK_STATUSES, PipelineRunner, PipelineState
from evalbuilder.pipeline.layout import artifact_index, path_for
from evalbuilder.pipeline.stages import REQUIRED_STAGES, PipelineContext, build_stages

PIPELINE_REPORT_SCHEMA = "evalbuilder/pipeline-report/v1"


def _stage_status(ctx: PipelineContext) -> dict:
    out = {}
    for name, rec in (ctx.state.stages if ctx.state else {}).items():
        out[name] = {
            "status": rec.status,
            "attempts": rec.attempts,
            "seconds": rec.seconds,
            "error": rec.error,
            "reason": rec.reason,
            "optional": rec.optional,
            "artifacts": rec.artifacts,
            "details": {k: v for k, v in rec.details.items() if k != "traceback"},
            "recovery_notes": rec.recovery_notes,
        }
    return out


def _verdict(ctx: PipelineContext, agg: dict | None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    stages = ctx.state.stages if ctx.state else {}
    for name in REQUIRED_STAGES:
        rec = stages.get(name)
        if rec is None or rec.status not in OK_STATUSES:
            status = rec.status if rec else "not run"
            reasons.append(f"stage {name} {status}" + (f": {rec.reason or rec.error}" if rec and (rec.reason or rec.error) else ""))
    if reasons:
        return "incomplete", reasons
    if agg is None:
        return "incomplete", ["no aggregate available"]
    return agg["verdict"], list(agg.get("verdict_reasons", []))


def analysis_summary(ctx: PipelineContext) -> dict:
    """Compact, LLM-sized view of the evaluation for the analyze stage."""
    agg = ctx.optional_json("aggregate") or {}
    coverage = ctx.optional_json("coverage") or {}
    verdict, reasons = _verdict(ctx, agg or None)
    amap = ctx.optional_json("agent_map") or {}
    return {
        "agent": {
            "module": ctx.config.target.module,
            "intents": [i.get("id") for i in amap.get("intents", [])],
            "tools": [t.get("name") for t in amap.get("tools", [])],
            "constraints": amap.get("constraints", []),
        },
        "verdict": verdict,
        "verdict_reasons": reasons,
        "overall_score": agg.get("overall_score"),
        "thresholds": ctx.config.thresholds.model_dump(),
        "metrics": agg.get("metrics", {}),
        "slices": agg.get("slices", {}),
        "weak_slices": agg.get("weak_slices", []),
        "stability": agg.get("stability", {}),
        "coverage": {k: v for k, v in coverage.items() if k != "gaps"},
        "coverage_gaps": coverage.get("gaps", [])[:20],
        "failing_cases": agg.get("failing_cases", [])[:40],
        "unstable_cases": agg.get("stability", {}).get("unstable_cases", []),
        "evaluator_errors": [f"{m}: {n} error(s)" for m, n in agg.get("evaluator_errors", {}).items()],
        "stages": {k: v["status"] for k, v in _stage_status(ctx).items()},
        "problems": ctx.problems,
        "simulation": (ctx.state.stages["simulate"].details if ctx.state and "simulate" in ctx.state.stages else None),
    }


def build_report(ctx: PipelineContext) -> dict:
    agg = ctx.optional_json("aggregate")
    coverage = ctx.optional_json("coverage")
    amap = ctx.optional_json("agent_map") or {}
    analysis = ctx.optional_json("analysis")
    simulation = ctx.optional_json("simulation")
    stages = _stage_status(ctx)
    verdict, reasons = _verdict(ctx, agg)
    problems = list(ctx.problems)
    for name, rec in stages.items():
        if rec["status"] == "failed":
            problems.append({"stage": name, "severity": "warning" if rec["optional"] else "error", "message": rec["error"] or "failed"})
        elif rec["status"] in ("awaiting_review",):
            problems.append({"stage": name, "severity": "error", "message": rec["reason"] or rec["status"]})
        elif rec["status"] == "skipped" and rec["reason"] and "dependency" in rec["reason"]:
            problems.append({"stage": name, "severity": "info", "message": rec["reason"]})

    generator_calls = []
    gen = ctx._cache.get("generator")
    if gen is not None:
        generator_calls = list(getattr(gen, "calls", []))

    return {
        "schema": PIPELINE_REPORT_SCHEMA,
        "name": ctx.config.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(ctx.config_path),
        "output_dir": str(ctx.out_dir),
        "artifacts": artifact_index(ctx.out_dir),
        "config": ctx.config.model_dump(by_alias=True),
        "verdict": verdict,
        "verdict_reasons": reasons,
        "overall_score": agg.get("overall_score") if agg else None,
        "stages": stages,
        "agent": {
            "module": ctx.config.target.module,
            "source": ctx.config.target.source,
            "source_sha256": amap.get("source_sha256"),
            "tools": [t.get("name") for t in amap.get("tools", [])],
            "nodes": [n.get("id") for n in amap.get("graph", {}).get("nodes", [])],
            "intents": amap.get("intents", []),
            "scenarios": amap.get("scenarios", []),
            "failure_scenarios": amap.get("failure_scenarios", []),
            "constraints": amap.get("constraints", []),
        },
        "coverage": coverage,
        "runs": (stages.get("score", {}).get("artifacts", {}) or {}).get("pairs", []),
        "metrics": agg.get("metrics", {}) if agg else {},
        "slices": agg.get("slices", {}) if agg else {},
        "stability": agg.get("stability") if agg else None,
        "cases": agg.get("cases", []) if agg else [],
        "failing_cases": agg.get("failing_cases", []) if agg else [],
        "simulation": {
            "details": stages.get("simulate", {}).get("details"),
            "results": simulation.get("results") if simulation else None,
        } if "simulate" in stages else None,
        "publication": stages.get("publish", {}).get("details") if "publish" in stages else None,
        "analysis": analysis,
        "generator_calls": generator_calls,
        "problems": problems,
    }


def summary_text(report: dict) -> str:
    lines = [
        f"{report['name']}: verdict={report['verdict']} overall={report.get('overall_score')}",
    ]
    for r in report.get("verdict_reasons", []):
        lines.append(f"  - {r}")
    lines.append("stages:")
    for name, rec in report.get("stages", {}).items():
        extra = rec.get("reason") or rec.get("error") or ""
        lines.append(f"  {name:10s} {rec['status']:16s} {rec.get('seconds', 0):>7.1f}s  {extra[:90]}")
    if report.get("metrics"):
        lines.append("metrics:")
        for m, v in report["metrics"].items():
            mark = "ok" if v.get("passed") else "FAIL"
            lines.append(f"  {m:18s} pass_rate={v.get('pass_rate')} threshold={v.get('threshold')} errors={v.get('errors')} [{mark}]")
    cov = report.get("coverage") or {}
    if cov:
        lines.append(f"coverage: {cov.get('covered')}/{cov.get('planned')} ({cov.get('coverage_pct')}%)")
    st = report.get("stability") or {}
    if st:
        lines.append(
            f"stability: repeats={st.get('repeats')} unstable_cases={len(st.get('unstable_cases', []))} "
            f"(trajectory) unstable_outputs={len(st.get('unstable_outputs', []))} "
            f"unstable_evaluators={len(st.get('unstable_evaluators', []))} "
            f"text_varies={st.get('text_varies')} suspect_judge_comments={len(st.get('suspect_judge_comments', []))}"
        )
    if report.get("analysis"):
        lines.append(f"analysis ({report['analysis'].get('source')}): {report['analysis'].get('summary', '')[:300]}")
    if report.get("problems"):
        lines.append("problems:")
        for p in report["problems"][:15]:
            lines.append(f"  [{p['severity']}] {p['stage']}: {p['message'][:120]}")
    return "\n".join(lines)


def run_pipeline(
    config_path: Path,
    *,
    resume: bool = False,
    invalidate_from: str | None = None,
    until: str | None = None,
    log: Callable[[str], None] = lambda msg: None,
    generator_factory=None,
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
) -> tuple[PipelineState, dict]:
    cfg = config or load_config(config_path)
    settings = settings or Settings.load()
    ctx = PipelineContext(
        config=cfg, config_path=Path(config_path), settings=settings, log=log,
        generator_factory=generator_factory,
    )
    skip = set(cfg.stages.skip)
    if not cfg.stages.simulate:
        skip.add("simulate")
    if cfg.stages.publish == "never" or (cfg.stages.publish == "auto" and not settings.langsmith_api_key):
        skip.add("publish")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    runner = PipelineRunner(
        stages=build_stages(),
        state_path=path_for(cfg.output_dir, "pipeline_state"),
        skip=skip,
        max_retries=cfg.stages.max_retries,
        resume=resume,
        invalidate_from=invalidate_from,
        stop_after=until,
        log=log,
    )
    state = runner.run(ctx, name=cfg.name)
    report_path = path_for(cfg.output_dir, "pipeline_report")
    if not report_path.exists():  # report stage itself failed — still leave something behind
        from evalbuilder.artifacts import save_json

        save_json(report_path, build_report(ctx))
    import json

    return state, json.loads(report_path.read_text())
