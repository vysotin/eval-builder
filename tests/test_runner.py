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


def _two_intent_ds(tmp_path):
    """Two approved cases in different intents against the weather bot."""
    ds = Dataset(name="w2", dataset_type="final_response",
                 target=Target(module="examples.weather_bot.agent"))
    for city, intent in (("Paris", "intent.weather"), ("Oslo", "intent.travel")):
        add_case(ds, {
            "inputs": {"messages": [{"role": "user", "content": f"What is the weather in {city}?"}]},
            "metadata": {"intent": intent, "mocks": {"tools": {"get_weather": [
                {"matchArgs": {}, "response": {"temp": 5, "condition": "mocked", "city": city}}]}}},
        })
    set_review(ds, [c.id for c in ds.cases], "approved", "test")
    p = tmp_path / "ds.json"
    save_json(p, ds)
    return ds, p


def test_run_parallel_intent_groups_execute_concurrently(tmp_path):
    """With max_workers=2 both intent groups run at once: a 2-party barrier inside the
    model only passes when the two cases execute concurrently."""
    import threading

    from evalbuilder.testing import ScriptedChatModel, ai, tool_call

    ds, p = _two_intent_ds(tmp_path)
    barrier = threading.Barrier(2, timeout=10)

    def first(city):
        def rule(m, _msgs):
            barrier.wait()  # raises BrokenBarrierError when cases run sequentially
            return tool_call("get_weather", {"city": city})
        return rule

    model = ScriptedChatModel(script=[
        (r"TOOL:", lambda m, _msgs: ai("done")),
        (r"Paris", first("Paris")),
        (r"Oslo", first("Oslo")),
    ])
    art = run_dataset(ds, p, mocked=True, out_dir=tmp_path, model=model, max_workers=2)
    assert [cr.error for cr in art.case_runs] == [None, None]
    assert [cr.case_id for cr in art.case_runs] == [c.id for c in ds.cases]  # dataset order kept


def test_run_progress_callback_reports_each_case(tmp_path):
    from evalbuilder.testing import ScriptedChatModel, ai, tool_call

    ds, p = _two_intent_ds(tmp_path)
    model = ScriptedChatModel(script=[
        (r"TOOL:", lambda m, _msgs: ai("done")),
        (r"Paris", lambda m, _msgs: tool_call("get_weather", {"city": "Paris"})),
        (r"Oslo", lambda m, _msgs: tool_call("get_weather", {"city": "Oslo"})),
    ])
    events = []
    art = run_dataset(ds, p, mocked=True, out_dir=tmp_path, model=model,
                      max_workers=2, progress=events.append)
    assert len(events) == 2
    assert {e["case_id"] for e in events} == {c.id for c in ds.cases}
    assert sorted(e["completed"] for e in events) == [1, 2]
    assert all(e["total"] == 2 and e["intent"] and e["error_class"] == "none" for e in events)
    assert len(art.case_runs) == 2


def test_run_parallel_keeps_per_case_errors_isolated(tmp_path):
    from evalbuilder.testing import ScriptedChatModel, ai, tool_call

    ds, p = _two_intent_ds(tmp_path)
    model = ScriptedChatModel(script=[  # no rule for Oslo → that case errors, Paris passes
        (r"TOOL:", lambda m, _msgs: ai("done")),
        (r"Paris", lambda m, _msgs: tool_call("get_weather", {"city": "Paris"})),
    ])
    art = run_dataset(ds, p, mocked=True, out_dir=tmp_path, model=model, max_workers=2)
    by_id = {cr.case_id: cr for cr in art.case_runs}
    paris, oslo = ds.cases[0].id, ds.cases[1].id
    assert by_id[paris].error is None
    assert by_id[oslo].error and by_id[oslo].error_class == "agent"


# ── layer 2: the LLM mock engine inside a run ──────────────────


def _llm_ds(tmp_path, strategy=None, rules=None):
    from evalbuilder.mock_engine import DEFAULT_STRATEGY

    ds = Dataset(
        name="w", dataset_type="final_response", target=Target(module="examples.weather_bot.agent"),
        mocks={
            "tools": rules or {}, "on_miss": "llm", "strategy": DEFAULT_STRATEGY,
            "llm": {"model": "scripted:x:y", "on_invalid": "fallback", "max_repairs": 1},
            "strategies": {"world": "Weather world.", "strategies": {
                "default": {"description": "sunny", "tools": {"get_weather": {"behavior": "Always sunny, 70F.", "examples": [], "fallback_response": {"temp": 70, "condition": "sunny", "city": "?"}}}},
                "stormy": {"description": "storms", "tools": {"get_weather": {"behavior": "Thunderstorms everywhere.", "examples": []}}},
            }},
        },
    )
    md = {"intent": "intent.weather"}
    if strategy:
        md["mocks"] = {"strategy": strategy}
    case = add_case(ds, {"inputs": {"messages": [{"role": "user", "content": "What is the weather in Paris?"}]},
                         "reference_outputs": {"expected_tools": [{"name": "get_weather", "args": {"city": "Paris"}}]}, "metadata": md})
    set_review(ds, [case.id], "approved", "test")
    p = tmp_path / "ds.json"
    save_json(p, ds)
    return ds, p


def _mock_model(answer):
    from evalbuilder.mock_engine import MOCK_RESPONSE_TITLE, call_in_prompt
    from evalbuilder.testing import SchemaScriptedModel

    prompts = []

    def handler(messages):
        prompts.append(messages[-1].content)
        return answer(call_in_prompt(messages[-1].content))

    return SchemaScriptedModel(handlers={MOCK_RESPONSE_TITLE: handler}), prompts


def test_run_uses_the_engine_on_a_miss_and_records_the_ledger(tmp_path):
    ds, p = _llm_ds(tmp_path)
    model, prompts = _mock_model(lambda call: {"response_json": json.dumps({"temp": 61, "condition": "cloudy", "city": call["args"]["city"]})})
    art = run_dataset(ds, p, mocked=True, out_dir=tmp_path, on_miss="llm", mock_model=model)
    cr = art.case_runs[0]
    assert cr.error is None and "61" in cr.outputs["response"] and "cloudy" in cr.outputs["response"]
    assert [m["layer"] for m in cr.mock_calls] == ["llm"] and cr.mock_calls[0]["strategy"] == "default" and cr.mock_calls[0]["valid"]
    assert "Always sunny, 70F." in prompts[0] and "Weather world." in prompts[0]
    assert art.mocking == {"on_miss": "llm", "model": "scripted:x:y", "strategy": "default", "calls": {"rule": 0, "llm": 1, "real": 0, "fallback": 0, "error": 0, "invalid": 0}}
    saved = json.loads((tmp_path / f"run-{art.run_id}.json").read_text())
    assert saved["case_runs"][0]["mock_calls"][0]["tool"] == "get_weather" and saved["mocking"]["calls"]["llm"] == 1


def test_run_selects_the_case_strategy_and_rules_still_win(tmp_path):
    rules = {"get_weather": [{"matchArgs": {"city": "Oslo"}, "response": {"temp": 1, "condition": "snow", "city": "Oslo"}}]}
    ds, p = _llm_ds(tmp_path, strategy="stormy", rules=rules)
    model, prompts = _mock_model(lambda call: {"response_json": json.dumps({"temp": 50, "condition": "storm", "city": call["args"]["city"]})})
    art = run_dataset(ds, p, mocked=True, out_dir=tmp_path, on_miss="llm", mock_model=model)
    cr = art.case_runs[0]
    assert cr.mock_calls[0]["strategy"] == "stormy" and "Thunderstorms everywhere." in prompts[0] and "storm" in cr.outputs["response"]
    add_case(ds, {"inputs": {"messages": [{"role": "user", "content": "What is the weather in Oslo?"}]}, "metadata": {"intent": "intent.weather"}})
    set_review(ds, [ds.cases[1].id], "approved", "test")
    art2 = run_dataset(ds, p, mocked=True, out_dir=tmp_path, on_miss="llm", mock_model=model)
    oslo = art2.case_runs[1]
    assert oslo.mock_calls[0]["layer"] == "rule" and "snow" in oslo.outputs["response"] and len(prompts) == 2


def test_invalid_engine_answers_fall_back_or_become_infrastructure_errors(tmp_path):
    ds, p = _llm_ds(tmp_path)
    model, _ = _mock_model(lambda call: {"response_json": "nope {"})
    art = run_dataset(ds, p, mocked=True, out_dir=tmp_path, on_miss="llm", mock_model=model)
    cr = art.case_runs[0]
    assert cr.error is None and "70" in cr.outputs["response"]  # strategy fallback answered
    assert cr.mock_calls[0]["fallback"] is True and cr.mock_calls[0]["valid"] is False and cr.mock_calls[0]["repairs"] == 1
    assert art.mocking["calls"]["fallback"] == 1 and art.mocking["calls"]["invalid"] == 1
    ds.mocks["llm"]["on_invalid"] = "strict"
    art2 = run_dataset(ds, p, mocked=True, out_dir=tmp_path, on_miss="llm", mock_model=model)
    cr2 = art2.case_runs[0]
    assert cr2.error_class == "infrastructure" and "MockEngineError" in cr2.error and cr2.mock_calls[-1]["error"]
    assert art2.mocking["calls"]["error"] == 1


def test_llm_policy_without_a_model_is_an_error(tmp_path):
    ds, p = _llm_ds(tmp_path)
    with pytest.raises(ValueError, match="mock model"):
        run_dataset(ds, p, mocked=True, out_dir=tmp_path, on_miss="llm")
