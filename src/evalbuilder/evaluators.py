"""Evaluators: deterministic checks first, OpenEvals LLM judges second.

One metric per evaluator. Each evaluator is `fn(case, case_run) -> {key, score, comment}`
or raises EvaluatorUnavailable — an evaluator problem is never an agent failure.
"""

from __future__ import annotations

import importlib
import json

from evalbuilder.config import _has_judge_key
from evalbuilder.schemas import Case, CaseRun, Dataset, Report, RunArtifact

SLICE_DIMS = ("intent", "failure_mode", "variant")

CONTRACT_PROMPT = """You are checking one agent response against an explicit behavioral contract.
Return score 1 only when the response satisfies the contract; otherwise 0. Explain briefly.

<contract>{contract}</contract>
<inputs>{inputs}</inputs>
<outputs>{outputs}</outputs>"""


class EvaluatorUnavailable(Exception):
    pass


def _make_judge(prompt: str, model: str, key: str):
    from openevals.llm import create_llm_as_judge

    return create_llm_as_judge(prompt=prompt, model=model, feedback_key=key)


def _args_subset(expected: dict, actual: dict) -> bool:
    return all(actual.get(k) == v for k, v in (expected or {}).items())


def _expected_tools(case: Case, case_run: CaseRun) -> dict:
    expected = case.reference_outputs.get("expected_tools", [])
    actual = case_run.tool_calls
    pos = 0
    for exp in expected:
        while pos < len(actual):
            candidate = actual[pos]
            pos += 1
            if candidate["name"] == exp.get("name") and _args_subset(
                exp.get("args", {}), candidate["args"]
            ):
                break
        else:
            return {
                "key": "expected_tools",
                "score": False,
                "comment": (
                    f"expected call to {exp.get('name')} with args "
                    f"{exp.get('args', {})} not found in order; "
                    f"actual: {[t['name'] for t in actual]}"
                ),
            }
    return {"key": "expected_tools", "score": True, "comment": ""}


def _contains(spec: dict):
    def fn(case: Case, case_run: CaseRun) -> dict:
        needle = spec.get("value") or case.reference_outputs.get("contains")
        if not needle:
            raise EvaluatorUnavailable("no reference substring for contains")
        response = case_run.outputs.get("response", "")
        return {
            "key": "contains",
            "score": needle.lower() in response.lower(),
            "comment": f"looking for {needle!r}",
        }

    return fn


def _json_valid(case: Case, case_run: CaseRun) -> dict:
    try:
        json.loads(case_run.outputs.get("response", ""))
        return {"key": "json_valid", "score": True, "comment": ""}
    except (ValueError, TypeError) as e:
        return {"key": "json_valid", "score": False, "comment": str(e)}


def _trajectory_match(spec: dict):
    from agentevals.trajectory.match import create_trajectory_match_evaluator

    inner = create_trajectory_match_evaluator(
        trajectory_match_mode=spec.get("match_mode", "unordered"),
        tool_args_match_mode=spec.get("args_match", "subset"),
    )

    def fn(case: Case, case_run: CaseRun) -> dict:
        reference = case.reference_outputs.get("trajectory")
        if not reference:
            raise EvaluatorUnavailable("case has no reference trajectory")
        result = inner(outputs=case_run.trajectory, reference_outputs=reference)
        return {
            "key": "trajectory_match",
            "score": bool(result["score"]),
            "comment": str(result.get("comment") or ""),
        }

    return fn


def _judge(kind: str, spec: dict, judge_model: str):
    model = spec.get("model", judge_model)

    def fn(case: Case, case_run: CaseRun) -> dict:
        if not _has_judge_key(model):
            raise EvaluatorUnavailable(
                f"judge model {model!r} has no API key configured"
            )
        if kind == "contract":
            contract = case.reference_outputs.get("contract")
            if not contract:
                raise EvaluatorUnavailable("case has no reference contract")
            prompt = spec.get("prompt") or CONTRACT_PROMPT.replace(
                "{contract}", contract
            )
        else:
            from openevals.prompts import CORRECTNESS_PROMPT

            prompt = spec.get("prompt") or CORRECTNESS_PROMPT
        judge = _make_judge(prompt, model, kind)
        result = judge(
            inputs=json.dumps(case.inputs, ensure_ascii=False),
            outputs=case_run.outputs.get("response", ""),
            reference_outputs=json.dumps(case.reference_outputs, ensure_ascii=False),
        )
        return {
            "key": kind,
            "score": result["score"],
            "comment": str(result.get("comment") or ""),
        }

    return fn


def _custom(spec: dict):
    module_name, _, fn_name = spec["ref"].partition(":")
    return getattr(importlib.import_module(module_name), fn_name)


def build_evaluators(specs: list[dict], judge_model: str) -> list[tuple[str, callable]]:
    out: list[tuple[str, callable]] = []
    for spec in specs:
        kind = spec["type"]
        if kind == "expected_tools":
            out.append((kind, _expected_tools))
        elif kind == "contains":
            out.append((kind, _contains(spec)))
        elif kind == "json_valid":
            out.append((kind, _json_valid))
        elif kind == "trajectory_match":
            out.append((kind, _trajectory_match(spec)))
        elif kind in ("correctness", "contract"):
            out.append((kind, _judge(kind, spec, judge_model)))
        elif kind == "custom":
            out.append((spec.get("name", "custom"), _custom(spec)))
        else:
            raise ValueError(f"unknown evaluator type {kind!r}")
    return out


def score_run(
    run: RunArtifact, ds: Dataset, specs: list[dict], judge_model: str
) -> Report:
    evaluators = build_evaluators(specs, judge_model)
    cases_by_id = {c.id: c for c in ds.cases}

    report = Report(run_id=run.run_id, dataset_name=ds.name)
    per_metric: dict[str, list[float]] = {}
    metric_errors: dict[str, int] = {}
    slice_scores: dict[str, dict[str, dict[str, list[float]]]] = {
        dim: {} for dim in SLICE_DIMS
    }

    for case_run in run.case_runs:
        case = cases_by_id.get(case_run.case_id)
        if case is None:
            continue
        row = {"case_id": case_run.case_id, "scores": {}, "errors": {}}
        for key, fn in evaluators:
            if case_run.error:
                row["scores"][key] = {
                    "score": 0.0,
                    "comment": f"agent error: {case_run.error}",
                }
            else:
                try:
                    result = fn(case, case_run)
                    row["scores"][key] = {
                        "score": float(result["score"]),
                        "comment": result.get("comment", ""),
                    }
                except EvaluatorUnavailable as e:
                    row["errors"][key] = str(e)
                    metric_errors[key] = metric_errors.get(key, 0) + 1
                    continue
                except Exception as e:  # noqa: BLE001 - evaluator bug, not agent fault
                    row["errors"][key] = f"{type(e).__name__}: {e}"
                    metric_errors[key] = metric_errors.get(key, 0) + 1
                    continue
            score = row["scores"][key]["score"]
            per_metric.setdefault(key, []).append(score)
            for dim in SLICE_DIMS:
                value = str(case.metadata.get(dim, "unspecified"))
                slice_scores[dim].setdefault(value, {}).setdefault(key, []).append(
                    score
                )
        report.cases.append(row)

    for key, scores in per_metric.items():
        report.metrics[key] = {
            "n": len(scores),
            "avg": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "errors": metric_errors.get(key, 0),
        }
    for key, count in metric_errors.items():
        report.metrics.setdefault(
            key, {"n": 0, "avg": None, "min": None, "max": None, "errors": count}
        )
    for dim, values in slice_scores.items():
        report.slices[dim] = {
            value: {key: sum(s) / len(s) for key, s in metrics.items()}
            for value, metrics in values.items()
        }
    return report
