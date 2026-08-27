import json

import pytest
from typer.testing import CliRunner

from evalbuilder.artifacts import add_case, save_json, set_review
from evalbuilder.cli import app
from evalbuilder.runner import run_dataset
from evalbuilder.schemas import Dataset, Target


def _weather_ds(tmp_path):
    ds = Dataset(
        name="w",
        dataset_type="final_response",
        target=Target(module="examples.weather_bot.agent"),
    )
    add_case(
        ds,
        {
            "inputs": {
                "messages": [
                    {"role": "user", "content": "What is the weather in Paris?"}
                ]
            },
            "reference_outputs": {
                "expected_tools": [{"name": "get_weather", "args": {"city": "Paris"}}]
            },
            "metadata": {
                "intent": "intent.weather",
                "mocks": {
                    "tools": {
                        "get_weather": [
                            {
                                "matchArgs": {},
                                "response": {
                                    "temp": 55,
                                    "condition": "mocked-rain",
                                    "city": "Paris",
                                },
                            }
                        ]
                    }
                },
            },
        },
    )
    p = tmp_path / "ds.json"
    save_json(p, ds)
    return ds, p


def test_run_refuses_pending(tmp_path):
    ds, p = _weather_ds(tmp_path)
    with pytest.raises(ValueError, match="approved"):
        run_dataset(ds, p, mocked=False, out_dir=tmp_path)


def test_run_captures_trajectory_and_applies_mocks(tmp_path):
    ds, p = _weather_ds(tmp_path)
    set_review(ds, [ds.cases[0].id], "approved", "test")
    art = run_dataset(ds, p, mocked=True, out_dir=tmp_path)
    cr = art.case_runs[0]
    assert cr.error is None
    assert "55" in cr.outputs["response"]  # mock reached the answer
    assert {"name": "get_weather", "args": {"city": "Paris"}} in [
        {"name": t["name"], "args": t["args"]} for t in cr.tool_calls
    ]
    roles = [m["role"] for m in cr.trajectory]
    assert "tool" in roles and roles[-1] == "assistant"
    assert cr.node_path  # captured graph steps
    assert (tmp_path / f"run-{art.run_id}.json").exists()


def test_run_unmocked_uses_real_tool(tmp_path):
    ds, p = _weather_ds(tmp_path)
    set_review(ds, [ds.cases[0].id], "approved", "test")
    art = run_dataset(ds, p, mocked=False, out_dir=tmp_path)
    assert "72" in art.case_runs[0].outputs["response"]


def test_multi_turn_replay(tmp_path):
    ds = Dataset(
        name="t",
        dataset_type="final_response",
        target=Target(module="examples.travel_planner.agent"),
    )
    add_case(
        ds,
        {
            "inputs": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Find me a flight from SFO to JFK on 2026-09-01",
                    }
                ]
            },
            "metadata": {"user_turns": ["book it"]},
        },
    )
    p = tmp_path / "ds.json"
    save_json(p, ds)
    set_review(ds, [ds.cases[0].id], "approved", "test")
    art = run_dataset(ds, p, mocked=False, out_dir=tmp_path)
    cr = art.case_runs[0]
    assert cr.error is None
    assert "confirm" in cr.outputs["response"].lower()
    user_msgs = [m for m in cr.trajectory if m["role"] == "user"]
    assert len(user_msgs) == 2


def test_cli_run(tmp_path):
    runner = CliRunner()
    ds, p = _weather_ds(tmp_path)
    set_review(ds, [ds.cases[0].id], "approved", "test")
    save_json(p, ds)
    r = runner.invoke(app, ["run", str(p), "--mock", "--out", str(tmp_path)])
    assert r.exit_code == 0, r.output
    summary = json.loads(r.stdout)
    assert summary["cases"] == 1 and summary["errors"] == 0
    assert (tmp_path / f"run-{summary['run_id']}.json").exists()


def test_cli_run_refuses_pending(tmp_path):
    runner = CliRunner()
    _, p = _weather_ds(tmp_path)
    r = runner.invoke(app, ["run", str(p), "--out", str(tmp_path)])
    assert r.exit_code == 1
    assert "approved" in r.output


def test_run_injects_model_and_records_spec(tmp_path):
    from evalbuilder.testing import ScriptedChatModel, ai

    ds, p = _weather_ds(tmp_path)
    set_review(ds, [ds.cases[0].id], "approved", "test")
    injected = ScriptedChatModel(script=[(r".*", lambda m, _msgs: ai("INJECTED MODEL"))])
    art = run_dataset(
        ds, p, mocked=True, out_dir=tmp_path, model=injected, model_spec="scripted:test"
    )
    assert art.agent_model == "scripted:test"
    assert art.case_runs[0].outputs["response"] == "INJECTED MODEL"


def test_run_strict_miss_policy_records_agent_error(tmp_path):
    from evalbuilder.testing import ScriptedChatModel, ai, tool_call

    ds, p = _weather_ds(tmp_path)
    set_review(ds, [ds.cases[0].id], "approved", "test")
    # rules only match {} wildcard in the fixture; make them specific so Oslo misses
    ds.cases[0].metadata["mocks"]["tools"]["get_weather"] = [
        {"matchArgs": {"city": "Paris"}, "response": {"temp": 1, "condition": "x", "city": "Paris"}}
    ]
    oslo = ScriptedChatModel(
        script=[
            (r"TOOL:", lambda m, _msgs: ai("done")),
            (r".*", lambda m, _msgs: tool_call("get_weather", {"city": "Oslo"})),
        ]
    )
    art = run_dataset(ds, p, mocked=True, out_dir=tmp_path, model=oslo, on_miss="strict")
    cr = art.case_runs[0]
    assert cr.error and "no mock rule matched" in cr.error and cr.error_class == "agent"


def test_cli_run_accepts_model_spec(tmp_path):
    ds, p = _weather_ds(tmp_path)
    set_review(ds, [ds.cases[0].id], "approved", "test")
    save_json(p, ds)
    result = CliRunner().invoke(
        app,
        ["run", str(p), "--mock", "--out", str(tmp_path),
         "--model", "scripted:examples.weather_bot.agent:default_scripted_model",
         "--on-miss", "strict"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["errors"] == 0
