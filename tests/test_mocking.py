import json

import pytest
from langchain_core.tools import tool
from typer.testing import CliRunner

from evalbuilder.cli import app
from evalbuilder.mocking import (
    MockMissError,
    match_rule,
    merge_mock_rules,
    wrap_tool,
    wrap_tools,
)

RULES = [
    {"matchArgs": {"origin": "SFO"}, "response": {"flights": [{"id": "AA100"}]}},
    {"matchArgs": {}, "response": {"flights": []}},
]


@tool
def search_flights(origin: str, destination: str) -> dict:
    """Search flights."""
    return {"flights": [{"id": "REAL"}]}


def test_first_match_wins_and_wildcard_falls_through():
    assert match_rule(RULES, {"origin": "SFO", "destination": "JFK"})["response"][
        "flights"
    ]
    assert match_rule(RULES, {"origin": "LAX"})["response"] == {"flights": []}
    assert match_rule([RULES[0]], {"origin": "LAX"}) is None


def test_wrap_tool_matches_and_preserves_identity():
    wrapped = wrap_tool(search_flights, RULES)
    assert wrapped.name == "search_flights" and wrapped.description
    assert wrapped.args_schema is not None
    out = wrapped.invoke({"origin": "SFO", "destination": "JFK"})
    assert out == {"flights": [{"id": "AA100"}]}


def test_miss_policies():
    only_sfo = [RULES[0]]
    real = wrap_tool(search_flights, only_sfo, on_miss="real")
    assert real.invoke({"origin": "LAX", "destination": "JFK"}) == {
        "flights": [{"id": "REAL"}]
    }
    fb = wrap_tool(search_flights, only_sfo, on_miss="fallback", fallback={"flights": None})
    assert fb.invoke({"origin": "LAX", "destination": "JFK"}) == {"flights": None}
    strict = wrap_tool(search_flights, only_sfo, on_miss="strict")
    with pytest.raises(MockMissError):
        strict.invoke({"origin": "LAX", "destination": "JFK"})


def test_merge_case_overrides_dataset_per_tool():
    ds = {
        "a": [{"matchArgs": {}, "response": 1}],
        "b": [{"matchArgs": {}, "response": 2}],
    }
    case = {"a": [{"matchArgs": {}, "response": 9}]}
    merged = merge_mock_rules(ds, case)
    assert merged["a"][0]["response"] == 9 and merged["b"][0]["response"] == 2


def test_wrap_tools_passthrough():
    wrapped = wrap_tools([search_flights], {})
    assert wrapped[0] is search_flights


def _init_ds(runner, ds_path):
    runner.invoke(
        app,
        ["dataset", "init", str(ds_path), "--name", "d", "--type", "final_response",
         "--target", "examples.weather_bot.agent"],
    )


def test_cli_mock_set_and_verify(tmp_path):
    runner = CliRunner()
    ds_path = tmp_path / "ds.json"
    _init_ds(runner, ds_path)
    r = runner.invoke(
        app,
        ["dataset", "add", str(ds_path), "--case", json.dumps(
            {"inputs": {"q": 1},
             "reference_outputs": {"expected_tools": [
                 {"name": "get_weather", "args": {"city": "Paris"}}]}}
        )],
    )
    case_id = json.loads(r.stdout)["id"]

    r = runner.invoke(
        app,
        ["mock", "set", str(ds_path), "--case", case_id, "--tool", "get_weather",
         "--rules", json.dumps([{"matchArgs": {"city": "Paris"},
                                  "response": {"temp": 1}}])],
    )
    assert r.exit_code == 0, r.output

    r = runner.invoke(app, ["mock", "verify", str(ds_path)])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["ok"] is True

    # Now a rule that can never match the expected tool call args
    r = runner.invoke(
        app,
        ["mock", "set", str(ds_path), "--case", case_id, "--tool", "get_weather",
         "--rules", json.dumps([{"matchArgs": {"city": "Oslo"},
                                  "response": {"temp": 2}}])],
    )
    r = runner.invoke(app, ["mock", "verify", str(ds_path)])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["misses"]


def test_cli_mock_set_dataset_level(tmp_path):
    runner = CliRunner()
    ds_path = tmp_path / "ds.json"
    _init_ds(runner, ds_path)
    r = runner.invoke(
        app,
        ["mock", "set", str(ds_path), "--tool", "get_weather",
         "--rules", json.dumps([{"matchArgs": {}, "response": {"temp": 5}}])],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(ds_path.read_text())
    assert data["mocks"]["tools"]["get_weather"][0]["response"] == {"temp": 5}


def test_with_fallback_and_verify_only_approved():
    from evalbuilder.artifacts import add_case, set_review
    from evalbuilder.mocking import verify_dataset, with_fallback
    from evalbuilder.schemas import Dataset, Target

    default = [{"matchArgs": {}, "response": {"ok": True}}]
    inject = [{"matchArgs": {"order_id": "X"}, "response": {"error": "down"}}]
    assert with_fallback(inject, default) == inject + default
    assert with_fallback(default, default) == default  # already a wildcard
    ds = Dataset(name="d", dataset_type="final_response", target=Target(module="m"), mocks={"tools": {"t": default}})
    bad = add_case(ds, {"inputs": {"q": 1}, "reference_outputs": {"expected_tools": [{"name": "t", "args": {"a": 1}}]},
                        "metadata": {"mocks": {"tools": {"t": [{"matchArgs": {"a": 2}, "response": 1}]}}}})
    assert [m["case"] for m in verify_dataset(ds)] == [bad.id]
    set_review(ds, [bad.id], "rejected", "no rule")
    assert verify_dataset(ds, only_approved=True) == []
