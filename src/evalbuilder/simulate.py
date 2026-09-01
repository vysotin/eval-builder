"""Multi-turn conversation simulation: goal + stop conditions, failures mined as cases."""

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from evalbuilder.artifacts import add_case
from evalbuilder.mocking import ledger_totals
from evalbuilder.schemas import Dataset

REQUIRED_KEYS = ("id", "opening", "max_turns")


def load_scenarios(path: Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario file needs a non-empty 'scenarios' list")
    for i, scenario in enumerate(scenarios):
        for key in REQUIRED_KEYS:
            if not scenario.get(key):
                raise ValueError(f"scenarios[{i}] needs non-empty {key!r}")
        if not scenario.get("success_contains") and not scenario.get("expect"):
            raise ValueError(
                f"scenarios[{i}] needs success_contains or an expect block"
            )
    return scenarios


def _next_user_message(scenario: dict, followups: list[str], user_model, transcript):
    if followups:
        return followups.pop(0)
    if user_model is not None:
        prompt = (
            f"You are simulating a user. Persona: {scenario.get('persona', 'a user')}. "
            f"Goal: {scenario.get('goal', 'complete the task')}. "
            "Reply with ONLY the next user message continuing this conversation:\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in transcript[-6:])
        )
        reply = user_model.invoke(prompt)
        return reply.content if hasattr(reply, "content") else str(reply)
    return None


def _check_expectations(scenario: dict, transcript: list[dict]) -> list[str]:
    violations: list[str] = []
    expect = scenario.get("expect") or {}
    assistant_text = " ".join(
        m["content"] for m in transcript if m["role"] == "assistant"
    ).lower()
    if expect.get("contains") and expect["contains"].lower() not in assistant_text:
        violations.append(f"expected assistant text containing {expect['contains']!r}")
    if expect.get("not_contains") and expect["not_contains"].lower() in assistant_text:
        violations.append(f"forbidden assistant text {expect['not_contains']!r} appeared")
    return violations


def simulate_scenario(graph, scenario: dict, user_model=None) -> dict:
    transcript: list[dict] = []
    followups = list(scenario.get("followups", []))
    success_needle = (scenario.get("success_contains") or "").lower()
    stop_reason = "exhausted"
    user_message = scenario["opening"]
    turns = 0

    while turns < int(scenario["max_turns"]):
        transcript.append({"role": "user", "content": user_message})
        state = graph.invoke({"messages": [dict(m) for m in transcript]})
        reply = state["messages"][-1].content
        reply = reply if isinstance(reply, str) else str(reply)
        transcript.append({"role": "assistant", "content": reply})
        turns += 1

        if success_needle and success_needle in reply.lower():
            stop_reason = "success"
            break

        next_message = _next_user_message(scenario, followups, user_model, transcript)
        if next_message is None:
            stop_reason = "exhausted"
            break
        user_message = next_message
    else:
        stop_reason = "max_turns"  # a truncation, not a pass

    violations = _check_expectations(scenario, transcript)
    if stop_reason == "max_turns" and success_needle:
        violations.append(
            f"never reached success condition {scenario['success_contains']!r} "
            f"within {scenario['max_turns']} turns"
        )
    return {
        "scenario_id": scenario["id"],
        "stop_reason": stop_reason,
        "turns": turns,
        "transcript": transcript,
        "violations": violations,
    }


def simulate_scenarios(
    graph_factory, scenarios: list[dict], user_model=None, max_workers: int = 1
) -> list[dict]:
    """Run every scenario, each against its own graph from `graph_factory()`.

    A factory that takes an argument receives the scenario (so it can select the
    scenario's `mock_strategy`); it may return `(graph, ledger)` to have the mock ledger
    summarised into the result (`mock_calls`). With `max_workers > 1` the scenarios run
    concurrently in a thread pool — each worker builds a fresh graph (mirroring the
    per-case graphs of `run_dataset`), so no graph or tool wrapper is shared between
    threads. Results keep scenario order and are identical to a sequential pass."""
    takes_scenario = bool(inspect.signature(graph_factory).parameters)

    def _one(scenario: dict) -> dict:
        built = graph_factory(scenario) if takes_scenario else graph_factory()
        graph, ledger = built if isinstance(built, tuple) else (built, None)
        result = simulate_scenario(graph, scenario, user_model=user_model)
        if ledger is not None:
            result["mock_calls"] = ledger_totals(ledger)
            result["mock_strategy"] = scenario.get("mock_strategy")
        return result

    if max_workers > 1 and len(scenarios) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(scenarios))) as pool:
            return list(pool.map(_one, scenarios))
    return [_one(scenario) for scenario in scenarios]


def mine_failures(ds: Dataset, results: list[dict]) -> int:
    mined = 0
    for result in results:
        if not result["violations"] and result["stop_reason"] != "max_turns":
            continue
        user_messages = [
            m["content"] for m in result["transcript"] if m["role"] == "user"
        ]
        if not user_messages:
            continue
        add_case(
            ds,
            {
                "inputs": {"messages": [{"role": "user", "content": user_messages[0]}]},
                "metadata": {
                    "source": "simulation",
                    "scenario": result["scenario_id"],
                    "failure_mode": "simulation-violation",
                    "user_turns": user_messages[1:],
                    "violations": result["violations"],
                    "transcript_tail": result["transcript"][-4:],
                },
            },
        )
        mined += 1
    return mined
