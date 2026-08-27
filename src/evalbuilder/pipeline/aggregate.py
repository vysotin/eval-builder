"""Aggregate N repeated (run, report) pairs: pass rates, thresholds, slices, stability."""

from __future__ import annotations

import hashlib
import json

from evalbuilder.evaluators import SLICE_DIMS
from evalbuilder.pipeline.config import ThresholdsConfig
from evalbuilder.schemas import Dataset, Report, RunArtifact


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _r(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def output_hash(case_run) -> str:
    material = json.dumps(
        {
            "tool_calls": [(tc["name"], tc.get("args")) for tc in case_run.tool_calls],
            "response": (case_run.outputs or {}).get("response", ""),
            "error": case_run.error,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def aggregate(
    runs: list[tuple[RunArtifact, Report]], ds: Dataset, thresholds: ThresholdsConfig
) -> dict:
    repeats = len(runs)
    cases_by_id = {c.id: c for c in ds.cases}
    per_case: dict[str, dict] = {}

    for run, report in runs:
        run_rows = {cr.case_id: cr for cr in run.case_runs}
        for row in report.cases:
            cid = row["case_id"]
            entry = per_case.setdefault(
                cid,
                {"scores": {}, "errors": {}, "outputs": [], "agent_errors": [], "comments": {}},
            )
            cr = run_rows.get(cid)
            if cr is not None:
                entry["outputs"].append(output_hash(cr))
                if cr.error:
                    entry["agent_errors"].append(f"{cr.error_class}: {cr.error}")
            for metric, value in row.get("scores", {}).items():
                entry["scores"].setdefault(metric, []).append(float(value["score"]))
                if value.get("comment") and float(value["score"]) < 1:
                    entry["comments"].setdefault(metric, []).append(str(value["comment"])[:300])
            for metric, err in row.get("errors", {}).items():
                entry["errors"].setdefault(metric, []).append(str(err))

    # metrics
    metric_names = sorted({m for e in per_case.values() for m in e["scores"]} | {m for e in per_case.values() for m in e["errors"]})
    metrics: dict[str, dict] = {}
    for m in metric_names:
        case_means = [_mean(e["scores"][m]) for e in per_case.values() if e["scores"].get(m)]
        case_mins = [min(e["scores"][m]) for e in per_case.values() if e["scores"].get(m)]
        errors = sum(len(e["errors"].get(m, [])) for e in per_case.values())
        pass_rate = _mean([x for x in case_means if x is not None])
        threshold = thresholds.for_metric(m)
        metrics[m] = {
            "pass_rate": _r(pass_rate),
            "min_over_repeats": _r(_mean(case_mins)),
            "n_cases": len(case_means),
            "n_scores": sum(len(e["scores"].get(m, [])) for e in per_case.values()),
            "errors": errors,
            "threshold": threshold,
            "passed": pass_rate is not None and pass_rate >= threshold,
        }

    # slices
    slices: dict[str, dict] = {}
    weak_slices: list[str] = []
    for dim in SLICE_DIMS:
        slices[dim] = {}
        groups: dict[str, list[str]] = {}
        for cid in per_case:
            case = cases_by_id.get(cid)
            value = str(case.metadata.get(dim, "unspecified")) if case else "unspecified"
            groups.setdefault(value, []).append(cid)
        for value, ids in groups.items():
            rates = {}
            for m in metric_names:
                means = [_mean(per_case[c]["scores"][m]) for c in ids if per_case[c]["scores"].get(m)]
                rate = _mean([x for x in means if x is not None])
                if rate is not None:
                    rates[m] = _r(rate)
            passed = all(r >= thresholds.slice_min for r in rates.values())
            slices[dim][value] = {"n": len(ids), "metrics": rates, "passed": passed}
            if not passed:
                weak_slices.append(f"{dim}={value}")

    # stability
    unstable_cases = []
    unstable_evaluators = []
    for cid, e in per_case.items():
        distinct = len(set(e["outputs"]))
        disagreeing = [m for m, s in e["scores"].items() if len(s) > 1 and len(set(s)) > 1]
        if distinct > 1:
            unstable_cases.append(
                {"id": cid, "distinct_outputs": distinct, "metrics_disagreeing": disagreeing}
            )
        elif disagreeing:
            for m in disagreeing:
                unstable_evaluators.append({"case_id": cid, "metric": m, "scores": e["scores"][m]})

    # cases + failing
    cases_out = []
    failing = []
    for cid, e in per_case.items():
        case = cases_by_id.get(cid)
        md = case.metadata if case else {}
        means = {m: _r(_mean(s)) for m, s in e["scores"].items()}
        failing_metrics = [m for m, v in means.items() if v is not None and v < 1.0]
        row = {
            "id": cid,
            "intent": md.get("intent"),
            "scenario": md.get("scenario"),
            "failure_mode": md.get("failure_mode"),
            "variant": md.get("variant"),
            "input": (case.inputs.get("messages") or [{}])[0].get("content") if case else None,
            "scores": e["scores"],
            "means": means,
            "evaluator_errors": e["errors"],
            "agent_errors": e["agent_errors"],
            "outputs_hash": e["outputs"],
            "stable": len(set(e["outputs"])) <= 1,
        }
        cases_out.append(row)
        if failing_metrics or e["agent_errors"]:
            failing.append(
                {
                    "id": cid,
                    "intent": md.get("intent"),
                    "failure_mode": md.get("failure_mode"),
                    "input": row["input"],
                    "failing_metrics": {m: means[m] for m in failing_metrics},
                    "comments": {m: e["comments"].get(m, [])[:2] for m in failing_metrics},
                    "agent_errors": e["agent_errors"][:2],
                }
            )

    rates = [
        _mean([x for x in (_mean(e["scores"][m]) for e in per_case.values() if e["scores"].get(m)) if x is not None])
        for m in metric_names
    ]
    overall = _r(_mean([r for r in rates if r is not None]))
    metrics_ok = all(v["passed"] for v in metrics.values()) if metrics else False
    verdict = (
        "pass"
        if metrics_ok and not weak_slices and overall is not None and overall >= thresholds.overall_pass
        else "fail"
    )
    reasons = []
    if not metrics:
        reasons.append("no metrics were scored")
    reasons += [f"metric {m} {v['pass_rate']} < {v['threshold']}" for m, v in metrics.items() if not v["passed"]]
    reasons += [f"slice {s} below slice_min {thresholds.slice_min}" for s in weak_slices]
    if overall is not None and overall < thresholds.overall_pass:
        reasons.append(f"overall {overall} < {thresholds.overall_pass}")

    evaluator_errors = {m: v["errors"] for m, v in metrics.items() if v["errors"]}
    return {
        "repeats": repeats,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "overall_score": overall,
        "overall_pass": thresholds.overall_pass,
        "metrics": metrics,
        "slices": slices,
        "weak_slices": weak_slices,
        "stability": {
            "repeats": repeats,
            "unstable_cases": unstable_cases,
            "unstable_evaluators": unstable_evaluators,
            "stable_case_fraction": _r(1 - len(unstable_cases) / len(per_case)) if per_case else None,
        },
        "evaluator_errors": evaluator_errors,
        "failing_cases": failing,
        "cases": cases_out,
    }
