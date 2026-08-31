import json

import pytest
import yaml
from typer.testing import CliRunner

from evalbuilder.artifacts import save_json
from evalbuilder.cli import app
from evalbuilder.schemas import Dataset, Target
from evalbuilder.simulate import load_scenarios, mine_failures, simulate_scenario
from examples.travel_planner.agent import build_agent

SCEN = {
    "id": "flight-booking-confirmation",
    "persona": "busy exec",
    "goal": "book the cheapest SFO->JFK flight",
    "opening": "Find me a flight from SFO to JFK on 2026-09-01",
    "followups": ["book it"],
    "max_turns": 4,
    "success_contains": "confirm",
    "expect": {"not_contains": "booked AA100 for you"},
}


def test_load_scenarios_validates(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"scenarios": [SCEN]}))
    assert load_scenarios(p)[0]["id"] == SCEN["id"]


def test_load_scenarios_rejects_missing_stop_condition(tmp_path):
    p = tmp_path / "s.yaml"
    bad = {"id": "x", "opening": "hi", "max_turns": 2}
    p.write_text(yaml.safe_dump({"scenarios": [bad]}))
    with pytest.raises(ValueError, match="success_contains"):
        load_scenarios(p)


def test_simulation_reaches_confirmation_gate():
    result = simulate_scenario(build_agent(), SCEN)
    assert result["stop_reason"] == "success"
    assert result["violations"] == []
    assert any(
        m["role"] == "assistant" and "confirm" in m["content"].lower()
        for m in result["transcript"]
    )


def test_simulation_max_turns_is_a_violation():
    scenario = dict(SCEN, followups=[], max_turns=1, success_contains="nonexistent")
    result = simulate_scenario(build_agent(), scenario)
    assert result["stop_reason"] in ("max_turns", "exhausted")
    # success never reached -> mined as failure when max_turns


def test_mine_failures_adds_pending_case():
    ds = Dataset(name="d", dataset_type="final_response", target=Target(module="m"))
    bad = {
        "scenario_id": "s1",
        "stop_reason": "max_turns",
        "turns": 4,
        "transcript": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "…"},
            {"role": "user", "content": "again"},
        ],
        "violations": ["never reached confirmation"],
    }
    assert mine_failures(ds, [bad]) == 1
    c = ds.cases[0]
    assert c.review.status == "pending"
    assert c.metadata["source"] == "simulation"
    assert c.metadata["user_turns"] == ["again"]


def test_mine_skips_clean_results():
    ds = Dataset(name="d", dataset_type="final_response", target=Target(module="m"))
    clean = {
        "scenario_id": "s1",
        "stop_reason": "success",
        "turns": 2,
        "transcript": [{"role": "user", "content": "hi"}],
        "violations": [],
    }
    assert mine_failures(ds, [clean]) == 0


def test_cli_simulate(tmp_path):
    runner = CliRunner()
    ds = Dataset(
        name="t",
        dataset_type="final_response",
        target=Target(module="examples.travel_planner.agent"),
    )
    ds_path = tmp_path / "ds.json"
    save_json(ds_path, ds)
    scen_path = tmp_path / "scenarios.yaml"
    scen_path.write_text(yaml.safe_dump({"scenarios": [SCEN]}))

    r = runner.invoke(
        app,
        ["simulate", str(ds_path), "--scenarios", str(scen_path), "--out", str(tmp_path)],
    )
    assert r.exit_code == 0, r.output
    summary = json.loads(r.stdout)
    assert summary["scenarios"] == 1 and summary["mined"] == 0
    assert list(tmp_path.glob("simulation-*.json"))


# ── parallel scenario execution ────────────────────────────────


class _FakeGraph:
    """graph.invoke stand-in: replies with a fixed text, optionally after a barrier."""

    def __init__(self, reply, barrier=None):
        self.reply = reply
        self.barrier = barrier

    def invoke(self, state):
        from types import SimpleNamespace

        if self.barrier is not None:
            self.barrier.wait()  # raises BrokenBarrierError when scenarios run sequentially
        return {"messages": [SimpleNamespace(content=self.reply)]}


def _scen(i):
    return {"id": f"s{i}", "opening": "hi", "max_turns": 2, "success_contains": "confirm"}


def test_simulate_scenarios_run_concurrently_with_a_graph_per_scenario():
    import threading

    from evalbuilder.simulate import simulate_scenarios

    barrier = threading.Barrier(2, timeout=10)
    built = []

    def factory():
        graph = _FakeGraph("done, please confirm", barrier)
        built.append(graph)
        return graph

    results = simulate_scenarios(factory, [_scen(1), _scen(2)], max_workers=2)
    assert [r["scenario_id"] for r in results] == ["s1", "s2"]  # scenario order kept
    assert all(r["stop_reason"] == "success" for r in results)
    assert len(built) == 2  # each worker built its own graph


def test_simulate_scenarios_sequential_matches_parallel():
    from evalbuilder.simulate import simulate_scenarios

    scens = [_scen(1), _scen(2), _scen(3)]
    parallel = simulate_scenarios(lambda: _FakeGraph("ok, confirm"), scens, max_workers=3)
    sequential = simulate_scenarios(lambda: _FakeGraph("ok, confirm"), scens, max_workers=1)
    assert parallel == sequential
    assert [r["scenario_id"] for r in parallel] == ["s1", "s2", "s3"]
