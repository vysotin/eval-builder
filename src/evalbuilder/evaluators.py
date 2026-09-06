"""Evaluators: deterministic checks first, OpenEvals/AgentEvals LLM judges second.

One metric per evaluator. Each evaluator is `fn(case, case_run) -> {key, score, comment}`
or raises EvaluatorUnavailable — an evaluator problem is never an agent failure.

Judge models are `provider:model` specs resolved by `claude_cli.model_from_spec`, so
`claude-cli:sonnet` (subscription) and `anthropic:…` / `openai:…` (API keys) are
interchangeable.
"""

from __future__ import annotations

import importlib
import json
import re
from functools import lru_cache

from joblib import Parallel, delayed

from evalbuilder.config import provider_ready
from evalbuilder.mocking import args_subset
from evalbuilder.schemas import Case, CaseRun, Dataset, Report, RunArtifact

SLICE_DIMS = ("intent", "failure_mode", "variant")

CONTRACT_PROMPT = """You are checking one agent response against an explicit behavioral contract.
Return score 1 only when the response satisfies the contract; otherwise 0. Explain briefly.

<contract>{contract}</contract>
<inputs>{inputs}</inputs>
<outputs>{outputs}</outputs>"""

SUSPECT_REASONING = re.compile(r"^\s*(test\b|placeholder|lorem)", re.I)

DETERMINISTIC_TYPES = ("expected_tools", "contains", "json_valid", "trajectory_match")
JUDGE_TYPES = ("correctness", "contract", "openevals", "trajectory_llm")


class EvaluatorUnavailable(Exception):
    """The evaluator could not run (missing key, judge failure) — an evaluator error."""


class EvaluatorNotApplicable(EvaluatorUnavailable):
    """The case carries no reference for this evaluator — skipped, not an error."""


@lru_cache(maxsize=8)
def _model(spec: str):
    from evalbuilder.claude_cli import model_from_spec

    return model_from_spec(spec)


def _make_judge(prompt: str, model: str, key: str, **kwargs):
    from openevals.llm import create_llm_as_judge

    return create_llm_as_judge(prompt=prompt, judge=_model(model), feedback_key=key, **kwargs)


def _make_trajectory_judge(prompt: str, model: str, key: str):
    from agentevals.trajectory.llm import create_trajectory_llm_as_judge

    return create_trajectory_llm_as_judge(prompt=prompt, judge=_model(model), feedback_key=key)


def openevals_prompt(name: str) -> str | None:
    """`CONCISENESS_PROMPT` (or `conciseness`) → the OpenEvals rubric text, else None."""
    from openevals import prompts

    candidates = [name, f"{name.upper()}_PROMPT"]
    for candidate in candidates:
        value = getattr(prompts, candidate, None)
        if isinstance(value, str):
            return value
    return None


def _require_ready(model: str) -> None:
    ready, reason = provider_ready(model)
    if not ready:
        raise EvaluatorUnavailable(f"judge model {model!r} unavailable: {reason}")


def _args_subset(expected: dict, actual: dict) -> bool:
    return args_subset(expected, actual)


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
    forbidden = case.reference_outputs.get("forbidden_tools", [])
    hit = [t["name"] for t in actual if t["name"] in forbidden]
    if hit:
        return {
            "key": "expected_tools",
            "score": False,
            "comment": f"forbidden tool(s) called: {hit}",
        }
    return {"key": "expected_tools", "score": True, "comment": ""}


def _contains(spec: dict):
    def fn(case: Case, case_run: CaseRun) -> dict:
        needle = spec.get("value") or case.reference_outputs.get("contains")
        if not needle:
            raise EvaluatorNotApplicable("no reference substring for contains")
        response = case_run.outputs.get("response", "")
        needles = needle if isinstance(needle, list) else [needle]
        missing = [n for n in needles if n.lower() not in response.lower()]
        return {
            "key": "contains",
            "score": not missing,
            "comment": f"missing {missing}" if missing else f"found {needles}",
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
            raise EvaluatorNotApplicable("case has no reference trajectory")
        result = inner(outputs=case_run.trajectory, reference_outputs=reference)
        return {
            "key": "trajectory_match",
            "score": bool(result["score"]),
            "comment": str(result.get("comment") or ""),
        }

    return fn


def _judge(kind: str, spec: dict, judge_model: str):
    model = spec.get("model", judge_model)
    key = spec.get("name") or kind
    judge_kwargs = {}
    if spec.get("continuous"):
        judge_kwargs["continuous"] = True

    def fn(case: Case, case_run: CaseRun) -> dict:
        _require_ready(model)
        if kind == "contract":
            contract = case.reference_outputs.get("contract")
            if not contract:
                raise EvaluatorNotApplicable("case has no reference contract")
            prompt = spec.get("prompt") or CONTRACT_PROMPT.replace("{contract}", contract)
        else:
            raw = spec.get("prompt") or kind
            prompt = openevals_prompt(raw) or raw
            if prompt == raw and "{outputs}" not in raw:
                raise EvaluatorUnavailable(
                    f"evaluator {key!r}: {raw!r} is neither an openevals prompt name "
                    "nor a template with {outputs}"
                )
        judge = _make_judge(prompt, model, key, **judge_kwargs)
        kwargs = dict(
            inputs=json.dumps(case.inputs, ensure_ascii=False),
            outputs=case_run.outputs.get("response", ""),
            reference_outputs=json.dumps(case.reference_outputs, ensure_ascii=False),
        )
        result = judge(**kwargs)
        if SUSPECT_REASONING.match(str(result.get("comment") or "")):
            result = judge(**kwargs)  # placeholder rationale: one retry, then report as is
        return {
            "key": key,
            "score": result["score"],
            "comment": str(result.get("comment") or ""),
        }

    return fn


def _trajectory_llm(spec: dict, judge_model: str):
    model = spec.get("model", judge_model)
    key = spec.get("name") or "trajectory_llm"

    def fn(case: Case, case_run: CaseRun) -> dict:
        _require_ready(model)
        from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT

        prompt = spec.get("prompt") or TRAJECTORY_ACCURACY_PROMPT
        judge = _make_trajectory_judge(prompt, model, key)
        kwargs = {"outputs": case_run.trajectory}
        reference = case.reference_outputs.get("trajectory")
        if reference:
            kwargs["reference_outputs"] = reference
        result = judge(**kwargs)
        return {
            "key": key,
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
        elif kind == "trajectory_llm":
            out.append((spec.get("name") or kind, _trajectory_llm(spec, judge_model)))
        elif kind in ("correctness", "contract"):
            out.append((spec.get("name") or kind, _judge(kind, spec, judge_model)))
        elif kind == "openevals":
            if not spec.get("prompt"):
                raise ValueError("openevals evaluator needs a prompt name or template")
            name = spec.get("name") or spec["prompt"].lower().removesuffix("_prompt")
            out.append((name, _judge(name, {**spec, "name": name}, judge_model)))
        elif openevals_prompt(kind):
            out.append((spec.get("name") or kind, _judge(kind, spec, judge_model)))
        elif kind == "custom":
            out.append((spec.get("name", "custom"), _custom(spec)))
        else:
            raise ValueError(f"unknown evaluator type {kind!r}")
    return out


def is_judge_spec(spec: dict) -> bool:
    kind = spec.get("type", "")
    return kind in JUDGE_TYPES or (kind not in DETERMINISTIC_TYPES and kind != "custom")


def _chunks(items: list, n: int) -> list[list]:
    """`n` contiguous, near-equal splits of `items` (empty splits dropped)."""
    n = max(1, min(int(n), len(items)))
    size, extra = divmod(len(items), n)
    out, start = [], 0
    for i in range(n):
        end = start + size + (1 if i < extra else 0)
        if end > start:
            out.append(items[start:end])
        start = end
    return out


def score_rows(run: RunArtifact, ds: Dataset, specs: list[dict], judge_model: str, case_ids: list[str] | None = None) -> list[dict]:
    """Score the case runs whose ids are listed (all when None), in run order — the unit
    of work of a scoring split; evaluators are built here so a process worker can run it."""
    evaluators = build_evaluators(specs, judge_model)
    cases_by_id = {c.id: c for c in ds.cases}
    wanted = set(case_ids) if case_ids is not None else None
    rows = []
    for case_run in run.case_runs:
        case = cases_by_id.get(case_run.case_id)
        if case is None or (wanted is not None and case_run.case_id not in wanted):
            continue
        row = {"case_id": case_run.case_id, "scores": {}, "errors": {}, "skipped": {}}
        for key, fn in evaluators:
            if case_run.error:
                row["scores"][key] = {"score": 0.0, "comment": f"agent error: {case_run.error}"}
                continue
            try:
                result = fn(case, case_run)
                row["scores"][key] = {"score": float(result["score"]), "comment": result.get("comment", "")}
            except EvaluatorNotApplicable as e:
                row["skipped"][key] = str(e)
            except EvaluatorUnavailable as e:
                row["errors"][key] = str(e)
            except Exception as e:  # noqa: BLE001 - evaluator bug, not agent fault
                row["errors"][key] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


def score_run(
    run: RunArtifact, ds: Dataset, specs: list[dict], judge_model: str,
    workers: int = 1, backend: str = "threads", max_workers: int | None = None,
) -> Report:
    """Score one run. With `workers > 1` the case runs are split into contiguous
    chunks scored concurrently by joblib (`threads`, or `processes` — each worker
    rebuilds the evaluators from `specs`); rows, metrics and slices are assembled in run
    order afterwards, so the report is identical to a sequential scoring pass.
    `max_workers` is the older name of `workers`."""
    if max_workers is not None:
        workers = max_workers
    if backend not in ("threads", "processes"):
        raise ValueError(f"invalid backend {backend!r}; use threads or processes")
    cases_by_id = {c.id: c for c in ds.cases}
    ids = [cr.case_id for cr in run.case_runs if cr.case_id in cases_by_id]
    if workers > 1 and len(ids) > 1:
        pool = Parallel(n_jobs=min(int(workers), len(ids)), prefer="threads" if backend == "threads" else "processes")
        batches = pool(delayed(score_rows)(run, ds, specs, judge_model, chunk) for chunk in _chunks(ids, workers))
        rows = [row for batch in batches for row in batch]
    else:
        rows = score_rows(run, ds, specs, judge_model)
    scored = [cases_by_id[row["case_id"]] for row in rows]

    report = Report(run_id=run.run_id, dataset_name=ds.name)
    per_metric: dict[str, list[float]] = {}
    metric_errors: dict[str, int] = {}
    metric_skipped: dict[str, int] = {}
    slice_scores: dict[str, dict[str, dict[str, list[float]]]] = {
        dim: {} for dim in SLICE_DIMS
    }
    for row, case in zip(rows, scored):
        report.cases.append(row)
        for key in row["errors"]:
            metric_errors[key] = metric_errors.get(key, 0) + 1
        for key in row["skipped"]:
            metric_skipped[key] = metric_skipped.get(key, 0) + 1
        for key, cell in row["scores"].items():
            score = cell["score"]
            per_metric.setdefault(key, []).append(score)
            for dim in SLICE_DIMS:
                value = str(case.metadata.get(dim, "unspecified"))
                slice_scores[dim].setdefault(value, {}).setdefault(key, []).append(
                    score
                )

    for key, scores in per_metric.items():
        report.metrics[key] = {
            "n": len(scores),
            "avg": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "errors": metric_errors.get(key, 0),
            "skipped": metric_skipped.get(key, 0),
        }
    for key in set(metric_errors) | set(metric_skipped):
        report.metrics.setdefault(
            key,
            {"n": 0, "avg": None, "min": None, "max": None,
             "errors": metric_errors.get(key, 0), "skipped": metric_skipped.get(key, 0)},
        )
    for dim, values in slice_scores.items():
        report.slices[dim] = {
            value: {key: sum(s) / len(s) for key, s in metrics.items()}
            for value, metrics in values.items()
        }
    return report
