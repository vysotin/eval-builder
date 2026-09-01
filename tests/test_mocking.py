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


def test_nested_subset_matching_for_model_arguments():
    from evalbuilder.mocking import args_subset, match_rule

    rules = [
        {"matchArgs": {"ticket": {"service": "api", "severity": "sev1"}}, "response": {"ticket_id": "INC-1"}},
        {"matchArgs": {}, "response": {"ticket_id": "INC-0"}},
    ]
    full = {"ticket": {"title": "API down", "service": "api", "severity": "sev1", "summary": "500s"}}
    assert match_rule(rules, full)["response"] == {"ticket_id": "INC-1"}
    assert match_rule(rules, {"ticket": {"service": "api", "severity": "sev2"}})["response"] == {"ticket_id": "INC-0"}
    assert match_rule(rules, {"ticket": "text"})["response"] == {"ticket_id": "INC-0"}
    assert args_subset({"a": {"b": 1}}, {"a": {"b": 1, "c": 2}}) and not args_subset({"a": {"b": 1}}, {"a": {"c": 2}})
    assert args_subset({}, {"x": 1}) and not args_subset({"x": 1}, {})


# ── CLI: layer 2 (strategies, validate, try, run --on-miss llm) ─────────

WEATHER_MOCK = "scripted:examples.weather_bot.offline:mock_model"
STRATEGIES = {
    "world": "Paris only.",
    "strategies": {
        "default": {"description": "sunny", "tools": {
            "get_weather": {"behavior": "72F sunny", "examples": [{"args": {"city": "Paris"}, "response": {"temp": 72, "condition": "sunny", "city": "Paris"}}],
                            "fallback_response": {"temp": 70, "condition": "sunny", "city": "Paris"}},
            "get_alerts": {"behavior": "heat advisory in Paris", "examples": [], "fallback_response": {"alerts": [], "city": "Paris"}},
        }},
        "stormy": {"description": "storms", "tools": {"get_alerts": {"behavior": "storm warning everywhere", "examples": []}}},
    },
}


def _llm_dataset(tmp_path):
    runner = CliRunner()
    ds_path = tmp_path / "ds.json"
    _init_ds(runner, ds_path)
    r = runner.invoke(app, ["dataset", "add", str(ds_path), "--case", json.dumps(
        {"inputs": {"messages": [{"role": "user", "content": "What is the weather in Paris?"}]},
         "reference_outputs": {"expected_tools": [{"name": "get_weather", "args": {"city": "Paris"}}]}})])
    case_id = json.loads(r.stdout)["id"]
    runner.invoke(app, ["review", str(ds_path), "--approve", case_id, "--note", "test"])
    return runner, ds_path, case_id


def test_cli_mock_strategies_sets_the_llm_layer_and_validates(tmp_path):
    runner, ds_path, _ = _llm_dataset(tmp_path)
    strategies_file = tmp_path / "strategies.json"
    strategies_file.write_text(json.dumps({"schema": "evalbuilder/mock-strategies/v1", **STRATEGIES}))
    r = runner.invoke(app, ["mock", "strategies", str(ds_path), "--set", f"@{strategies_file}", "--model", WEATHER_MOCK, "--on-miss", "llm", "--strategy", "default"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert out["updated"] == ["strategies", "on_miss", "strategy", "llm"] and out["on_miss"] == "llm"
    assert out["llm"] == {"model": WEATHER_MOCK, "on_invalid": "fallback", "max_repairs": 1}
    assert out["strategies"] == {"default": ["get_alerts", "get_weather"], "stormy": ["get_alerts"]} and out["world"] == "Paris only."
    data = json.loads(ds_path.read_text())
    assert data["mocks"]["strategies"]["strategies"]["default"]["tools"]["get_weather"]["behavior"] == "72F sunny"
    # show without changes
    r = runner.invoke(app, ["mock", "strategies", str(ds_path)])
    assert r.exit_code == 0 and json.loads(r.stdout)["updated"] == []
    # invalid strategies are refused with the schema problems
    bad = {"world": "w", "strategies": {"default": {"tools": {"get_weather": {"behavior": "x", "examples": [{"args": {"city": 5}, "response": {}}]}, "nope": {"behavior": "y"}}}}}
    r = runner.invoke(app, ["mock", "strategies", str(ds_path), "--set", json.dumps(bad)])
    assert r.exit_code == 1 and "unknown tool 'nope'" in r.stdout and "example 0 args" in r.stdout
    r = runner.invoke(app, ["mock", "strategies", str(ds_path), "--on-invalid", "shrug"])
    assert r.exit_code == 1 and "on_invalid" in r.stdout


def test_cli_mock_validate_and_try_and_verify_under_llm(tmp_path):
    runner, ds_path, case_id = _llm_dataset(tmp_path)
    runner.invoke(app, ["mock", "strategies", str(ds_path), "--set", json.dumps(STRATEGIES), "--model", WEATHER_MOCK, "--on-miss", "llm"])
    # validate: weather tools return dicts (no output schema) → anything JSON is valid; a schema-carrying tool would be checked
    r = runner.invoke(app, ["mock", "validate", str(ds_path), "--tool", "get_weather", "--response", json.dumps({"temp": 1})])
    out = json.loads(r.stdout)
    assert r.exit_code == 0 and out["valid"] and out["problems"] == [] and out["output_schema"]["type"] == "object"
    r = runner.invoke(app, ["mock", "validate", str(ds_path), "--tool", "get_weather", "--response", json.dumps("not an object")])
    assert r.exit_code == 1 and "expected object" in r.stdout
    r = runner.invoke(app, ["mock", "validate", str(ds_path), "--tool", "nope", "--response", "{}"])
    assert r.exit_code == 1 and "unknown or non-mockable" in r.output
    # try: no rule → the engine answers under the selected strategy
    r = runner.invoke(app, ["mock", "try", str(ds_path), "--tool", "get_alerts", "--args", json.dumps({"city": "Oslo"}), "--strategy", "stormy"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert out["layer"] == "llm" and out["strategy"] == "stormy" and out["response"] == {"alerts": ["storm warning"], "city": "Oslo"}
    assert out["valid"] is True and out["repairs"] == 0 and out["model"] == WEATHER_MOCK
    # try: a rule wins
    runner.invoke(app, ["mock", "set", str(ds_path), "--tool", "get_weather", "--rules", json.dumps([{"matchArgs": {"city": "Paris"}, "response": {"temp": 5, "condition": "snow", "city": "Paris"}}])])
    r = runner.invoke(app, ["mock", "try", str(ds_path), "--tool", "get_weather", "--args", json.dumps({"city": "Paris"})])
    assert json.loads(r.stdout)["layer"] == "rule" and json.loads(r.stdout)["response"]["condition"] == "snow"
    # verify: the expected Oslo call has no rule but the engine answers → informational
    runner.invoke(app, ["dataset", "add", str(ds_path), "--case", json.dumps(
        {"inputs": {"messages": [{"role": "user", "content": "What is the weather in Oslo?"}]},
         "reference_outputs": {"expected_tools": [{"name": "get_weather", "args": {"city": "Oslo"}}]}})])
    r = runner.invoke(app, ["mock", "verify", str(ds_path)])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert out["ok"] and out["policy"] == "llm" and out["misses"] == [] and out["llm_answered"][0]["tool"] == "get_weather"
    # run --mock picks the dataset's policy and model; the artifact carries the ledger
    r = runner.invoke(app, ["run", str(ds_path), "--mock", "--out", str(tmp_path), "--ids", case_id,
                            "--model", "scripted:examples.weather_bot.agent:default_scripted_model"])
    assert r.exit_code == 0, r.output
    summary = json.loads(r.stdout)
    assert summary["errors"] == 0 and summary["mocking"]["on_miss"] == "llm" and summary["mocking"]["calls"]["rule"] == 1
    # strict policy without a model for llm
    data = json.loads(ds_path.read_text())
    data["mocks"].pop("llm")
    ds_path.write_text(json.dumps(data))
    r = runner.invoke(app, ["run", str(ds_path), "--mock", "--on-miss", "llm", "--out", str(tmp_path), "--ids", case_id])
    assert r.exit_code == 1 and "no mock model" in r.output
    r = runner.invoke(app, ["mock", "try", str(ds_path), "--tool", "get_alerts", "--mock-model", WEATHER_MOCK, "--args", json.dumps({"city": "Rome"})])
    assert r.exit_code == 0 and json.loads(r.stdout)["response"] == {"alerts": [], "city": "Rome"}


def test_cli_simulate_with_mocks(tmp_path):
    import yaml

    runner, ds_path, _ = _llm_dataset(tmp_path)
    runner.invoke(app, ["mock", "strategies", str(ds_path), "--set", json.dumps(STRATEGIES), "--model", WEATHER_MOCK, "--on-miss", "llm"])
    scen = tmp_path / "scen.yaml"
    scen.write_text(yaml.safe_dump({"scenarios": [
        {"id": "alerts", "opening": "Any alerts for Paris?", "max_turns": 2, "success_contains": "heat advisory"},
        {"id": "storm", "opening": "Any alerts for Paris?", "max_turns": 2, "success_contains": "storm warning", "mock_strategy": "stormy"},
    ]}))
    r = runner.invoke(app, ["simulate", str(ds_path), "--scenarios", str(scen), "--out", str(tmp_path), "--mock", "--no-mine"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert out["stop_reasons"] == {"alerts": "success", "storm": "success"}
    assert out["mock_calls"]["alerts"]["llm"] == 1 and out["mock_calls"]["storm"]["llm"] == 1
