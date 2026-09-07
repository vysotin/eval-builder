"""The inference engine: cases and scenarios through an agent client with joblib —
order, progress, isolation of errors, multi-turn replay, per-turn logs, and identical
artifacts whatever the backend."""

import json
import threading

import pytest

from evalbuilder.agent_client import LocalAgent, RemoteAgent
from evalbuilder.artifacts import add_case, save_json, set_review
from evalbuilder.inference import (
    infer_cases,
    infer_dataset,
    local_agent_for,
    run_conversation,
    select_cases,
    simulate_scenarios,
    summarize_mock_calls,
)
from evalbuilder.schemas import Dataset, Target
from evalbuilder.serve import make_server

WEATHER = "examples.weather_bot.agent"
SCRIPTED = "scripted:examples.weather_bot.agent:default_scripted_model"
MOCK_MODEL = "scripted:examples.weather_bot.offline:mock_model"


def _ds(tmp_path, n=3, turns=False):
    ds = Dataset(name="w", dataset_type="final_response", target=Target(module=WEATHER),
                 mocks={"tools": {"get_weather": [{"matchArgs": {}, "response": {"temp": 5, "condition": "mocked", "city": "X"}}],
                                  "get_alerts": [{"matchArgs": {}, "response": {"alerts": ["heat advisory"], "city": "X"}}]}, "on_miss": "strict"})
    cities = ["Paris", "Oslo", "Rome", "Lima", "Cairo"]
    for i in range(n):
        md = {"intent": f"intent.{i % 2}"}
        if turns:
            md["user_turns"] = [f"Any alerts for {cities[i]}?"]
        add_case(ds, {"inputs": {"messages": [{"role": "user", "content": f"What is the weather in {cities[i]}?"}]}, "metadata": md})
    set_review(ds, [c.id for c in ds.cases], "approved", "t")
    p = tmp_path / "ds.json"
    save_json(p, ds)
    return ds, p


def _strip(artifact) -> dict:
    """The artifact without the parts that legitimately differ between runs."""
    data = json.loads(artifact.model_dump_json(by_alias=True))
    data.pop("run_id"), data.pop("timestamp"), data.pop("execution")
    for cr in data["case_runs"]:
        for entry in cr["log"]:
            entry.pop("seconds")
        for entry in cr["mock_calls"]:
            entry.pop("seconds", None)
    return data


def test_run_conversation_replays_every_turn_and_logs_each_one():
    agent = LocalAgent(WEATHER)
    parts = run_conversation(agent, [{"role": "user", "content": "What is the weather in Paris?"}], ["Any alerts for Paris?"],
                             {"tools": {"get_weather": [{"matchArgs": {}, "response": {"temp": 1, "condition": "c", "city": "Paris"}}],
                                        "get_alerts": [{"matchArgs": {}, "response": {"alerts": ["storm"], "city": "Paris"}}]}, "on_miss": "strict"})
    assert parts["error"] is None and "storm" in parts["outputs"]["response"]
    assert [tc["name"] for tc in parts["tool_calls"]] == ["get_weather", "get_alerts"]
    assert [m["role"] for m in parts["trajectory"]].count("user") == 2
    assert [e["turn"] for e in parts["log"]] == [0, 1] and [e["tool_calls"] for e in parts["log"]] == [1, 1]
    assert all(e["mode"] == "local" and e["endpoint"] is None and e["error"] is None for e in parts["log"])
    assert [e["layer"] for e in parts["mock_calls"]] == ["rule", "rule"] and len(parts["node_path"]) >= 2


def test_run_conversation_stops_at_the_first_failing_turn():
    agent = LocalAgent(WEATHER)
    parts = run_conversation(agent, [{"role": "user", "content": "What is the weather in Paris?"}], ["Any alerts for Paris?"],
                             {"tools": {"get_weather": [{"matchArgs": {}, "response": {"temp": 1, "condition": "c", "city": "Paris"}}],
                                        "get_alerts": [{"matchArgs": {"city": "Nowhere"}, "response": {}}]}, "on_miss": "strict"})
    assert parts["error"] and "get_alerts" in parts["error"] and parts["error_class"] == "agent"
    assert [e["error"] is None for e in parts["log"]] == [True, False] and parts["outputs"] == {}


def test_infer_cases_keeps_order_reports_progress_and_isolates_errors(tmp_path):
    ds, _ = _ds(tmp_path, n=4)
    ds.cases[2].metadata["mocks"] = {"tools": {"get_weather": [{"matchArgs": {"city": "Nowhere"}, "response": {}}]}}  # Rome misses → error
    events = []
    results = infer_cases(LocalAgent(WEATHER), ds, ds.cases, workers=3, progress=events.append)
    assert [r.case_id for r in results] == [c.id for c in ds.cases]
    assert [r.error is None for r in results] == [True, True, False, True]
    assert results[2].error_class == "agent" and "no mock rule matched" in results[2].error
    assert [e["completed"] for e in events] == [1, 2, 3, 4] and all(e["total"] == 4 for e in events)
    assert {e["case_id"] for e in events} == {c.id for c in ds.cases} and events[2]["error_class"] == "agent"


def test_infer_cases_runs_concurrently_with_threads(tmp_path):
    ds, _ = _ds(tmp_path, n=2)
    barrier = threading.Barrier(2, timeout=10)

    class Gate:
        mode, endpoint = "local", None

        def __init__(self):
            self.inner = LocalAgent(WEATHER)

        def invoke(self, messages, mocks=None, *, mocked=True):
            barrier.wait()  # BrokenBarrierError when the two cases run one after the other
            return self.inner.invoke(messages, mocks, mocked=mocked)

    results = infer_cases(Gate(), ds, ds.cases, workers=2, backend="threads")
    assert [r.error for r in results] == [None, None]


def test_infer_dataset_backends_produce_identical_artifacts(tmp_path):
    ds, p = _ds(tmp_path, n=5, turns=True)
    agent = LocalAgent(WEATHER, agent_model_spec=SCRIPTED)  # specs, so loky workers rebuild the model
    sequential = infer_dataset(ds, p, agent, out_dir=tmp_path / "seq", workers=1, model_spec=SCRIPTED)
    threads = infer_dataset(ds, p, agent, out_dir=tmp_path / "thr", workers=4, backend="threads", model_spec=SCRIPTED)
    processes = infer_dataset(ds, p, agent, out_dir=tmp_path / "proc", workers=2, backend="processes", model_spec=SCRIPTED)
    assert _strip(threads) == _strip(sequential) == _strip(processes)
    assert sequential.execution["mode"] == "local" and sequential.execution["workers"] == 1 and sequential.execution["backend"] == "threads"
    assert processes.execution["workers"] == 2 and processes.execution["backend"] == "processes" and processes.execution["seconds"] >= 0
    assert all(cr.error is None and len(cr.log) == 2 for cr in processes.case_runs)
    assert processes.mocking["on_miss"] == "strict" and processes.mocking["calls"]["rule"] == 10
    saved = json.loads((tmp_path / "proc" / f"run-{processes.run_id}.json").read_text())
    assert saved["execution"]["backend"] == "processes" and saved["case_runs"][0]["log"][1]["turn"] == 1


def test_infer_dataset_against_a_remote_agent_records_the_endpoint(tmp_path):
    ds, p = _ds(tmp_path, n=2, turns=True)
    srv = make_server(LocalAgent(WEATHER), "127.0.0.1", 0)
    srv.serve_in_thread()
    try:
        remote = RemoteAgent(srv.endpoint, timeout=30)
        art = infer_dataset(ds, p, remote, out_dir=tmp_path, workers=2)
        local = infer_dataset(ds, p, LocalAgent(WEATHER), out_dir=tmp_path / "local", workers=1)
    finally:
        srv.shutdown()
        srv.server_close()
    assert art.execution["mode"] == "remote" and art.execution["endpoint"] == srv.endpoint
    assert all(cr.error is None and cr.log[0]["endpoint"] == srv.endpoint and cr.log[0]["mode"] == "remote" for cr in art.case_runs)
    for a, b in zip(art.case_runs, local.case_runs):
        assert (a.outputs, a.tool_calls, a.trajectory) == (b.outputs, b.tool_calls, b.trajectory)


def test_select_cases_rules(tmp_path):
    ds, _ = _ds(tmp_path, n=2)
    assert [c.id for c in select_cases(ds)] == [c.id for c in ds.cases]
    assert select_cases(ds, [ds.cases[1].id])[0].id == ds.cases[1].id
    with pytest.raises(ValueError, match="unknown case ids"):
        select_cases(ds, ["nope"])
    set_review(ds, [ds.cases[0].id], "rejected", "t")
    with pytest.raises(ValueError, match="not approved"):
        select_cases(ds, [ds.cases[0].id])
    empty = Dataset(name="e", dataset_type="final_response", target=Target(module=WEATHER))
    with pytest.raises(ValueError, match="no approved cases"):
        select_cases(empty)


def test_llm_policy_keeps_engine_history_across_turns(tmp_path):
    ds = Dataset(name="w", dataset_type="final_response", target=Target(module=WEATHER),
                 mocks={"tools": {}, "on_miss": "llm", "strategy": "default", "llm": {"model": MOCK_MODEL, "on_invalid": "fallback", "max_repairs": 1},
                        "strategies": {"world": "w", "strategies": {"default": {"description": "d", "tools": {
                            "get_weather": {"behavior": "sunny", "examples": []}, "get_alerts": {"behavior": "advisory", "examples": []}}}}}})
    add_case(ds, {"inputs": {"messages": [{"role": "user", "content": "What is the weather in Paris?"}]},
                  "metadata": {"intent": "i", "user_turns": ["Any alerts for Paris?"]}})
    set_review(ds, [ds.cases[0].id], "approved", "t")
    art = infer_dataset(ds, tmp_path / "ds.json", LocalAgent(WEATHER), out_dir=tmp_path, workers=1)
    cr = art.case_runs[0]
    assert cr.error is None and [m["layer"] for m in cr.mock_calls] == ["llm", "llm"] and "heat advisory" in cr.outputs["response"]
    assert art.mocking == {"on_miss": "llm", "model": MOCK_MODEL, "strategy": "default", "calls": {"rule": 0, "llm": 2, "real": 0, "fallback": 0, "error": 0, "invalid": 0}}


SCENARIO = {"id": "weather-then-alerts", "opening": "What is the weather in Paris?", "followups": ["Any alerts for Paris?"],
            "max_turns": 3, "success_contains": "heat advisory", "expect": {"contains": "mocked"}}


def test_simulate_scenarios_through_an_agent_keeps_order_and_logs(tmp_path):
    ds, _ = _ds(tmp_path)
    scenarios = [SCENARIO, {**SCENARIO, "id": "second", "mock_strategy": "stormy"}, {**SCENARIO, "id": "never", "success_contains": "nonexistent", "max_turns": 1}]
    results = simulate_scenarios(LocalAgent(WEATHER), scenarios, mocks=ds.mocks, workers=3)
    assert [r["scenario_id"] for r in results] == ["weather-then-alerts", "second", "never"]
    assert results[0]["stop_reason"] == "success" and results[0]["violations"] == []
    assert results[1]["mock_strategy"] == "stormy" and results[1]["mock_calls"]["rule"] == 2
    assert results[2]["stop_reason"] == "max_turns" and results[2]["violations"]
    assert [e["turn"] for e in results[0]["log"]] == [0, 1] and results[0]["log"][1]["mode"] == "local"
    assert summarize_mock_calls(results)["rule"] == 5
    same = simulate_scenarios(LocalAgent(WEATHER, agent_model_spec=SCRIPTED), scenarios, mocks=ds.mocks, workers=2, backend="processes")
    assert [(r["scenario_id"], r["stop_reason"], r["transcript"]) for r in same] == [(r["scenario_id"], r["stop_reason"], r["transcript"]) for r in results]


def test_simulate_scenarios_reports_agent_errors_as_results():
    results = simulate_scenarios(LocalAgent(WEATHER, factory="no_such_factory"), [SCENARIO], mocks={"tools": {}, "on_miss": "real"})
    assert results[0]["stop_reason"] == "error" and "no_such_factory" in results[0]["error"] and results[0]["violations"]


def test_local_agent_for_prefers_objects_over_specs(tmp_path):
    from evalbuilder.testing import ScriptedChatModel, ai

    ds, p = _ds(tmp_path, n=1)
    injected = ScriptedChatModel(script=[(r".*", lambda m, _msgs: ai("INJECTED"))])
    agent = local_agent_for(ds, model=injected, model_spec="scripted:ignored:x")
    assert agent.agent_model_spec is None and agent.invoke([{"role": "user", "content": "hi"}], None).response == "INJECTED"


def test_case_runs_record_the_inputs_that_were_asked(tmp_path):
    ds, p = _ds(tmp_path, n=2, turns=True)
    art = infer_dataset(ds, p, LocalAgent(WEATHER), out_dir=tmp_path, workers=2)
    for case, cr in zip(ds.cases, art.case_runs):
        assert cr.error is None
        assert cr.inputs == {**case.inputs, "user_turns": case.metadata["user_turns"]}
        assert cr.inputs["messages"] == case.inputs["messages"]
    saved = json.loads((tmp_path / f"run-{art.run_id}.json").read_text())
    assert saved["case_runs"][0]["inputs"]["user_turns"] == ds.cases[0].metadata["user_turns"]


def test_single_turn_case_runs_carry_inputs_without_user_turns(tmp_path):
    ds, p = _ds(tmp_path, n=2)
    art = infer_dataset(ds, p, LocalAgent(WEATHER), out_dir=tmp_path, workers=1)
    for case, cr in zip(ds.cases, art.case_runs):
        assert cr.inputs == case.inputs and "user_turns" not in cr.inputs


def test_error_case_runs_still_record_the_inputs():
    from evalbuilder.inference import _infer_case

    case = {"id": "c1", "inputs": {"question": "no messages here"}, "metadata": {"user_turns": ["and then?"]}}
    cr = _infer_case(LocalAgent(WEATHER), case, None, False)
    assert cr.error_class == "infrastructure" and "no 'messages' list" in cr.error
    assert cr.inputs == {"question": "no messages here", "user_turns": ["and then?"]}

    boom = {"id": "c2", "inputs": {"messages": [{"role": "user", "content": "hi"}]}}
    cr = _infer_case(LocalAgent(WEATHER, factory="no_such_factory"), boom, None, False)
    assert cr.error and cr.error_class == "infrastructure" and cr.inputs == boom["inputs"]
