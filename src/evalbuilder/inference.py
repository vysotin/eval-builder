"""The inference phase: dataset cases and multi-turn simulation scenarios against an
agent client, in parallel with joblib.

Every unit of work is one conversation — a case (its first message plus every
`metadata.user_turns` entry) or a scenario — run through `agent.invoke(messages,
mocks)` turn by turn, so it needs nothing but the picklable client and plain dicts.
`joblib.Parallel(..., return_as="generator")` yields results in submission order, which
keeps progress reporting and artifact order trivial for both backends:

- `threads` — the default; the work is waiting on the agent (HTTP or model calls);
- `processes` — loky workers; useful when the agent runs in this process and is CPU
  bound (scripted graphs), each worker rebuilding the client from its specs.

Outputs are the run artifact (`run-<id>.json`: outputs, trajectories, tool calls, the
mock ledger and a per-turn `log`, plus `execution` — how and where it ran) and the
simulation results the `simulate` stage stores in `simulation.json`.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from joblib import Parallel, delayed

from evalbuilder import simulate as sim
from evalbuilder.agent_client import LocalAgent, mocks_for_case
from evalbuilder.artifacts import save_json
from evalbuilder.mock_engine import DEFAULT_STRATEGY
from evalbuilder.mocking import ledger_totals
from evalbuilder.schemas import Case, CaseRun, Dataset, RunArtifact

BACKENDS = ("threads", "processes")


def parallel(workers: int, backend: str = "threads") -> Parallel:
    """A joblib pool yielding results in order (`n_jobs=1` runs inline)."""
    if backend not in BACKENDS:
        raise ValueError(f"invalid backend {backend!r}; use one of {BACKENDS}")
    return Parallel(n_jobs=max(1, int(workers)), prefer="threads" if backend == "threads" else "processes", return_as="generator")


def _history_of(entries: list[dict]) -> list[dict]:
    """What the LLM mock engine remembers of earlier turns: tool, args, answer."""
    return [{"tool": e.get("tool"), "args": e.get("args"), "response": e.get("response")}
            for e in entries if e.get("layer") in ("rule", "llm", "real", "fallback") and "response" in e]


def run_conversation(agent, opening: list[dict], user_turns: list[str], mocks: dict | None, *, mocked: bool = True) -> dict:
    """Run one conversation turn by turn; returns the pieces of a CaseRun as a dict.

    The next turn is the returned history plus the user's message; the engine history
    (`mocks.history`) carries the mocked answers of earlier turns so a stateless server
    keeps a conversation consistent. Stops at the first turn that errors.
    """
    messages = [dict(m) for m in opening]
    ledger: list[dict] = []
    node_path: list[str] = []
    log: list[dict] = []
    result = None
    for turn, text in enumerate([None, *user_turns]):
        if text is not None:
            messages = list(result.messages) + [{"role": "user", "content": text}]
        request_mocks = dict(mocks) if (mocked and mocks is not None) else None
        if request_mocks is not None and ledger:
            request_mocks["history"] = _history_of(ledger)
        seen_calls = len(result.tool_calls) if result is not None else 0
        result = agent.invoke(messages, request_mocks, mocked=mocked)
        ledger.extend(result.mock_calls)
        node_path.extend(result.node_path)
        log.append({
            "turn": turn, "seconds": result.seconds, "tool_calls": max(0, len(result.tool_calls) - seen_calls),
            "error": result.error, "mode": agent.mode, "endpoint": agent.endpoint,
        })
        if result.error:
            break
    return {
        "outputs": {"response": result.response} if result.error is None else {},
        "trajectory": result.messages if result.error is None else [],
        "tool_calls": result.tool_calls if result.error is None else [],
        "node_path": node_path,
        "error": result.error,
        "error_class": result.error_class if result.error else "none",
        "mock_calls": ledger,
        "log": log,
    }


def _infer_case(agent, case: dict, mocks: dict | None, mocked: bool) -> CaseRun:
    messages = (case.get("inputs") or {}).get("messages")
    if not isinstance(messages, list) or not messages:
        return CaseRun(case_id=case["id"], error="case inputs have no 'messages' list", error_class="infrastructure")
    turns = [str(t) for t in ((case.get("metadata") or {}).get("user_turns") or [])]
    try:
        parts = run_conversation(agent, messages, turns, mocks, mocked=mocked)
    except Exception as e:  # noqa: BLE001 - a worker never raises; the run records it
        return CaseRun(case_id=case["id"], error=f"{type(e).__name__}: {e}", error_class="infrastructure")
    return CaseRun(case_id=case["id"], **parts)


def infer_cases(
    agent,
    ds: Dataset,
    cases: list[Case],
    *,
    workers: int = 1,
    backend: str = "threads",
    on_miss: str | None = None,
    strategy: str | None = None,
    mocked: bool = True,
    progress: Callable[[dict], None] | None = None,
) -> list[CaseRun]:
    """Run `cases` through `agent` in parallel; results keep the given order.

    `progress` (optional) is called in this thread after every completed case with
    `{case_id, intent, error_class, error, completed, total}`.
    """
    payloads = [c.model_dump(by_alias=True) for c in cases]
    blocks = [mocks_for_case(ds.mocks, c.metadata, on_miss=on_miss, strategy=strategy) if mocked else None for c in cases]
    tasks = (delayed(_infer_case)(agent, payload, block, mocked) for payload, block in zip(payloads, blocks))
    results: list[CaseRun] = []
    lock = threading.Lock()
    for case, case_run in zip(cases, parallel(workers, backend)(tasks)):
        with lock:
            results.append(case_run)
            if progress is not None:
                try:
                    progress({
                        "case_id": case.id, "intent": case.metadata.get("intent") or "",
                        "error_class": case_run.error_class, "error": case_run.error,
                        "completed": len(results), "total": len(cases),
                    })
                except Exception:  # noqa: BLE001 - progress reporting is advisory
                    pass
    return results


def select_cases(ds: Dataset, ids: list[str] | None = None) -> list[Case]:
    """The approved cases (or the given ids, which must all be approved)."""
    if ids:
        by_id = {c.id: c for c in ds.cases}
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ValueError(f"unknown case ids: {', '.join(unknown)}")
        selected = [by_id[i] for i in ids]
    else:
        selected = [c for c in ds.cases if c.review.status == "approved"]
    if not selected:
        raise ValueError("dataset has no approved cases to run")
    not_approved = [c.id for c in selected if c.review.status != "approved"]
    if not_approved:
        raise ValueError(f"cases are not approved: {', '.join(not_approved)}")
    return selected


def infer_dataset(
    ds: Dataset,
    dataset_path: Path,
    agent,
    *,
    out_dir: Path,
    ids: list[str] | None = None,
    workers: int = 1,
    backend: str = "threads",
    on_miss: str | None = None,
    strategy: str | None = None,
    mocked: bool = True,
    progress: Callable[[dict], None] | None = None,
    model_spec: str | None = None,
    mock_model_spec: str | None = None,
) -> RunArtifact:
    """Execute the approved cases against `agent` and write `run-<id>.json`."""
    selected = select_cases(ds, ids)
    policy = on_miss or (ds.mocks or {}).get("on_miss") or "real"
    started = time.time()
    artifact = RunArtifact(
        run_id=uuid4().hex[:8], dataset_path=str(dataset_path), dataset_name=ds.name, mocked=mocked,
        timestamp=datetime.now(timezone.utc).isoformat(), agent_model=model_spec,
    )
    artifact.case_runs = infer_cases(agent, ds, selected, workers=workers, backend=backend, on_miss=on_miss,
                                     strategy=strategy, mocked=mocked, progress=progress)
    llm = mocked and policy == "llm"
    if mocked:
        artifact.mocking = {
            "on_miss": policy,
            "model": (mock_model_spec or ((ds.mocks or {}).get("llm") or {}).get("model") or getattr(agent, "mock_model_spec", None)) if llm else None,
            "strategy": (strategy or (ds.mocks or {}).get("strategy") or DEFAULT_STRATEGY) if llm else None,
            "calls": ledger_totals([e for cr in artifact.case_runs for e in cr.mock_calls]),
        }
    artifact.execution = {
        "mode": getattr(agent, "mode", "local"), "endpoint": getattr(agent, "endpoint", None),
        "workers": max(1, int(workers)), "backend": backend, "seconds": round(time.time() - started, 3),
    }
    out_dir = Path(out_dir)
    save_json(out_dir / f"run-{artifact.run_id}.json", artifact)
    return artifact


# ── multi-turn simulation scenarios ────────────────────────────


@lru_cache(maxsize=8)
def _user_model(spec: str):
    from evalbuilder.claude_cli import model_from_spec

    return model_from_spec(spec)


def _simulate_one(agent, scenario: dict, mocks: dict | None, mocked: bool, user_model_spec: str | None, user_model) -> dict:
    ledger: list[dict] = []
    log: list[dict] = []
    seen_calls = 0

    def step(transcript: list[dict]) -> str:
        nonlocal seen_calls
        request_mocks = dict(mocks) if (mocked and mocks is not None) else None
        if request_mocks is not None and ledger:
            request_mocks["history"] = _history_of(ledger)
        result = agent.invoke([dict(m) for m in transcript], request_mocks, mocked=mocked)
        ledger.extend(result.mock_calls)
        log.append({"turn": len(log), "seconds": result.seconds, "tool_calls": len(result.tool_calls),
                    "error": result.error, "mode": agent.mode, "endpoint": agent.endpoint})
        if result.error:
            raise RuntimeError(result.error)
        return result.response

    model = user_model if user_model is not None else (_user_model(user_model_spec) if user_model_spec else None)
    try:
        result = sim.simulate_scenario(step, scenario, user_model=model)
    except Exception as e:  # noqa: BLE001 - the scenario result records the failure
        result = {"scenario_id": scenario["id"], "stop_reason": "error", "turns": len(log), "transcript": [],
                  "violations": [f"agent error: {type(e).__name__}: {e}"], "error": f"{type(e).__name__}: {e}"}
    if mocked and mocks is not None:
        result["mock_calls"] = ledger_totals(ledger)
        result["mock_strategy"] = scenario.get("mock_strategy")
    result["log"] = log
    return result


def simulate_scenarios(
    agent,
    scenarios: list[dict],
    *,
    mocks: dict | None = None,
    user_model_spec: str | None = None,
    user_model=None,
    workers: int = 1,
    backend: str = "threads",
    on_miss: str | None = None,
    strategy: str | None = None,
    mocked: bool = True,
) -> list[dict]:
    """Run every scenario through `agent` (in parallel); results keep scenario order.

    `mocks` is the dataset's `mocks` block (None = real tools); a scenario's
    `mock_strategy` selects the engine strategy. The simulated user is `user_model`
    (an object; threads only) or `user_model_spec` (resolved in each worker).
    """
    if user_model is not None and backend == "processes":
        backend = "threads"  # a model object does not travel to loky workers
    blocks = [
        mocks_for_case(mocks, {"mocks": {"strategy": s.get("mock_strategy")}} if s.get("mock_strategy") else {}, on_miss=on_miss, strategy=strategy)
        if (mocked and mocks is not None) else None
        for s in scenarios
    ]
    tasks = (delayed(_simulate_one)(agent, s, block, mocked, user_model_spec, user_model) for s, block in zip(scenarios, blocks))
    return list(parallel(workers, backend)(tasks))


def local_agent_for(ds: Dataset, *, model=None, model_spec: str | None = None, mock_model=None, mock_model_spec: str | None = None) -> LocalAgent:
    """A `LocalAgent` for the dataset's target (objects win over specs)."""
    return LocalAgent(
        ds.target.module, ds.target.factory, agent_model=model, mock_model=mock_model,
        agent_model_spec=model_spec if model is None else None, mock_model_spec=mock_model_spec if mock_model is None else None,
    )


def summarize_mock_calls(results: list[dict]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for r in results:
        for layer, n in (r.get("mock_calls") or {}).items():
            totals[layer] = totals.get(layer, 0) + n
    return totals


__all__: list[str] = [
    "BACKENDS", "parallel", "run_conversation", "infer_cases", "select_cases", "infer_dataset",
    "simulate_scenarios", "local_agent_for", "summarize_mock_calls",
]
Any  # re-exported name kept for type hints in callers
