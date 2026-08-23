"""Full pipeline walk: discover -> map -> dataset -> review -> mock -> run -> score -> simulate.

This is the executable contract the four agent-eval skills rely on.
"""

import json

import yaml
from typer.testing import CliRunner

from evalbuilder.cli import app

runner = CliRunner()


def _ok(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_full_pipeline(tmp_path):
    eval_dir = tmp_path / "eval"

    # 1. discover
    out = _ok(
        runner.invoke(
            app,
            ["discover", "examples.travel_planner.agent",
             "--source", "examples/travel_planner/agent.py",
             "--eval-dir", str(eval_dir)],
        )
    )
    assert "flight_agent" in out["tools"] or "search_flights" in out["tools"]
    map_path = eval_dir / "agent-map.json"

    # 2. agent-map update (Claude's authored hypotheses, evidence-cited)
    intents = [
        {"id": "intent.book-flight", "status": "hypothesis",
         "evidence": ["prompt:build_agent", "tool:flight_agent"]},
        {"id": "intent.book-hotel", "status": "hypothesis",
         "evidence": ["prompt:build_agent", "tool:hotel_agent"]},
    ]
    scenarios = [
        {"id": "scenario.book-flight.happy", "intent": "intent.book-flight",
         "status": "hypothesis", "evidence": ["tool:flight_agent"]},
        {"id": "scenario.book-flight.confirmation-gate", "intent": "intent.book-flight",
         "status": "hypothesis",
         "evidence": ["prompt:build_agent"]},
    ]
    failures = [
        {"failure_type": "constraint_violation",
         "rationale": "supervisor prompt forbids booking without explicit yes",
         "evidence": ["prompt:build_agent"]},
    ]
    _ok(
        runner.invoke(
            app,
            ["agent-map", "update", str(map_path),
             "--intents", json.dumps(intents),
             "--scenarios", json.dumps(scenarios),
             "--failures", json.dumps(failures),
             "--constraints",
             json.dumps(["Never confirm a booking without an explicit user yes."])],
        )
    )

    # 3. dataset init + add 3 cases + validate + gaps
    ds_path = eval_dir / "datasets" / "travel-v1.json"
    _ok(
        runner.invoke(
            app,
            ["dataset", "init", str(ds_path), "--name", "travel-v1",
             "--type", "final_response",
             "--target", "examples.travel_planner.agent:build_agent"],
        )
    )
    cases = [
        {  # happy flight search, mocked subagent
            "inputs": {"messages": [{"role": "user",
                "content": "Find me a flight from SFO to JFK on 2026-09-01"}]},
            "reference_outputs": {
                "contains": "AA100",
                "expected_tools": [{"name": "flight_agent", "args": {}}],
            },
            "metadata": {
                "intent": "intent.book-flight", "scenario": "scenario.book-flight.happy",
                "variant": "happy", "evidence": ["scenario.book-flight.happy"],
                "mocks": {"tools": {"flight_agent": [
                    {"matchArgs": {},
                     "response": "Found AA100 at $350 (mocked fixture)."}]}},
            },
        },
        {  # constraint case: booking must hit the confirmation gate
            "inputs": {"messages": [{"role": "user", "content": "book it"}]},
            "reference_outputs": {
                "contains": "confirm",
                "contract": "Must not confirm a booking without an explicit user yes.",
            },
            "metadata": {
                "intent": "intent.book-flight",
                "scenario": "scenario.book-flight.confirmation-gate",
                "variant": "boundary",
                "failure_mode": "constraint_violation",
            },
        },
        {  # multi-turn: search then attempt booking
            "inputs": {"messages": [{"role": "user",
                "content": "Find me a flight from SFO to JFK on 2026-09-01"}]},
            "reference_outputs": {"contains": "confirm"},
            "metadata": {
                "intent": "intent.book-flight",
                "scenario": "scenario.book-flight.confirmation-gate",
                "variant": "multi-turn",
                "user_turns": ["book it"],
            },
        },
    ]
    ids = []
    for case in cases:
        out = _ok(
            runner.invoke(
                app, ["dataset", "add", str(ds_path), "--case", json.dumps(case)]
            )
        )
        ids.append(out["id"])
    _ok(runner.invoke(app, ["dataset", "validate", str(ds_path)]))

    gaps = _ok(
        runner.invoke(
            app, ["dataset", "gaps", str(ds_path), "--agent-map", str(map_path)]
        )
    )
    assert gaps["required_cells"] == 3  # 2 scenarios + 1 failure
    assert gaps["covered_cells"] >= 1

    # 4. run refuses pending cases
    r = runner.invoke(app, ["run", str(ds_path), "--out", str(eval_dir / "results")])
    assert r.exit_code == 1 and "approved" in r.output

    # 5. review approve (simulating the explicit user decision the skill requires)
    _ok(
        runner.invoke(
            app,
            ["review", str(ds_path), "--approve", ",".join(ids),
             "--note", "approved by user in e2e scenario"],
        )
    )

    # 6. mock verify
    verify = _ok(runner.invoke(app, ["mock", "verify", str(ds_path)]))
    assert verify["ok"] is True

    # 7. run mocked
    summary = _ok(
        runner.invoke(
            app,
            ["run", str(ds_path), "--mock", "--out", str(eval_dir / "results")],
        )
    )
    assert summary["cases"] == 3 and summary["errors"] == 0
    run_path = summary["path"]
    from pathlib import Path

    run_data = json.loads(Path(run_path).read_text())
    mocked_case = next(cr for cr in run_data["case_runs"] if cr["case_id"] == ids[0])
    assert any(tc["name"] == "flight_agent" for tc in mocked_case["tool_calls"])
    tool_msgs = [m for m in mocked_case["trajectory"] if m["role"] == "tool"]
    assert any("mocked fixture" in str(m.get("content", "")) for m in tool_msgs)

    # 8. score with deterministic evaluators
    ev_yaml = tmp_path / "evaluators.yaml"
    ev_yaml.write_text(
        yaml.safe_dump(
            {"evaluators": [{"type": "expected_tools"}, {"type": "contains"}]}
        )
    )
    report = _ok(
        runner.invoke(
            app,
            ["score", run_path, "--dataset", str(ds_path),
             "--evaluators", str(ev_yaml), "--out", str(eval_dir / "results")],
        )
    )
    assert report["metrics"]["contains"]["n"] == 3
    assert report["metrics"]["contains"]["avg"] == 1.0
    assert len(report["slices"]["intent"]) == 1
    assert set(report["slices"]["variant"]) == {"happy", "boundary", "multi-turn"}

    # 9. simulate the confirmation scenario; clean run mines nothing
    scen_path = tmp_path / "scenarios.yaml"
    scen_path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "id": "confirmation-gate",
                        "persona": "busy exec",
                        "goal": "book the cheapest flight",
                        "opening": "Find me a flight from SFO to JFK on 2026-09-01",
                        "followups": ["book it"],
                        "max_turns": 4,
                        "success_contains": "confirm",
                        "expect": {"not_contains": "booked AA100 for you"},
                    }
                ]
            }
        )
    )
    sim = _ok(
        runner.invoke(
            app,
            ["simulate", str(ds_path), "--scenarios", str(scen_path),
             "--out", str(eval_dir / "results")],
        )
    )
    assert sim["stop_reasons"]["confirmation-gate"] == "success"
    assert sim["mined"] == 0
