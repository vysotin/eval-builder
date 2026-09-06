"""Agent clients: the in-process `LocalAgent` and the HTTP `RemoteAgent` share one
`InvokeResult` contract; `mocks_for_case` builds the per-request mock block."""

import pickle

from evalbuilder.agent_client import InvokeResult, LocalAgent, RemoteAgent, mocks_for_case

WEATHER = "examples.weather_bot.agent"
RULES = {"get_weather": [{"matchArgs": {}, "response": {"temp": 55, "condition": "mocked-rain", "city": "Paris"}}]}
USER = [{"role": "user", "content": "What is the weather in Paris?"}]


def test_local_agent_answers_from_rules_and_reports_the_trajectory():
    agent = LocalAgent(WEATHER)
    assert agent.mode == "local" and agent.endpoint is None
    result = agent.invoke(USER, {"tools": RULES, "on_miss": "strict"})
    assert isinstance(result, InvokeResult) and result.error is None and result.error_class == "none"
    assert "55" in result.response and "mocked-rain" in result.response
    assert result.tool_calls == [{"name": "get_weather", "args": {"city": "Paris"}}]
    assert [m["role"] for m in result.messages][-3:] == ["assistant", "tool", "assistant"]
    assert result.node_path and result.seconds >= 0
    assert [e["layer"] for e in result.mock_calls] == ["rule"]
    health = agent.health()
    assert health["module"] == WEATHER and "get_weather" in health["tools"] and health["mode"] == "local"


def test_local_agent_multi_turn_continues_from_the_returned_history():
    agent = LocalAgent(WEATHER)
    first = agent.invoke(USER, {"tools": RULES})
    history = first.messages + [{"role": "user", "content": "And alerts in Paris?"}]
    second = agent.invoke(history, {"tools": {**RULES, "get_alerts": [{"matchArgs": {}, "response": {"alerts": ["storm"], "city": "Paris"}}]}})
    assert second.error is None and "storm" in second.response
    assert [m["role"] for m in second.messages].count("user") == 2
    assert second.tool_calls[-1]["name"] == "get_alerts"


def test_local_agent_strict_miss_is_an_agent_error_and_no_mocks_means_real_tools():
    agent = LocalAgent(WEATHER)
    strict = agent.invoke(USER, {"tools": {"get_weather": [{"matchArgs": {"city": "Oslo"}, "response": {}}]}, "on_miss": "strict"})
    assert strict.error and "no mock rule matched" in strict.error and strict.error_class == "agent"
    real = agent.invoke(USER, None, mocked=False)
    assert real.error is None and "72" in real.response and real.mock_calls == []


def test_local_agent_llm_policy_uses_the_engine_from_the_request_block():
    agent = LocalAgent(WEATHER)
    mocks = {
        "tools": {}, "on_miss": "llm", "strategy": "stormy",
        "llm": {"model": "scripted:examples.weather_bot.offline:mock_model", "on_invalid": "fallback", "max_repairs": 1},
        "strategies": {"world": "w", "strategies": {
            "default": {"description": "d", "tools": {"get_weather": {"behavior": "sunny", "examples": []}}},
            "stormy": {"description": "s", "tools": {"get_alerts": {"behavior": "storm warning", "examples": []}}},
        }},
    }
    result = agent.invoke([{"role": "user", "content": "Any alerts in Paris?"}], mocks)
    assert result.error is None and "storm warning" in result.response
    assert result.mock_calls[0]["layer"] == "llm" and result.mock_calls[0]["strategy"] == "stormy"


def test_local_agent_factory_failure_is_infrastructure():
    result = LocalAgent(WEATHER, factory="no_such_factory").invoke(USER, {"tools": RULES})
    assert result.error and result.error_class == "infrastructure"


def test_mocks_for_case_merges_case_rules_and_selects_the_case_strategy():
    ds_mocks = {"tools": {"get_weather": [{"matchArgs": {}, "response": {"temp": 1}}], "get_alerts": [{"matchArgs": {}, "response": {"alerts": []}}]},
                "on_miss": "llm", "strategy": "default", "strategies": {"world": "w", "strategies": {}}, "llm": {"model": "scripted:x:y"}}
    meta = {"mocks": {"tools": {"get_weather": [{"matchArgs": {"city": "Oslo"}, "response": {"temp": -5}}]}, "strategy": "stormy"}}
    block = mocks_for_case(ds_mocks, meta)
    assert block["tools"]["get_weather"] == [{"matchArgs": {"city": "Oslo"}, "response": {"temp": -5}}]  # case rules replace
    assert block["tools"]["get_alerts"] == ds_mocks["tools"]["get_alerts"]
    assert block["on_miss"] == "llm" and block["strategy"] == "stormy" and block["llm"] == {"model": "scripted:x:y"}
    assert block["strategies"] == ds_mocks["strategies"]
    override = mocks_for_case(ds_mocks, meta, on_miss="strict", strategy="other")
    assert override["on_miss"] == "strict" and override["strategy"] == "stormy"  # the case's own choice still wins
    plain = mocks_for_case(ds_mocks, {}, strategy="other")
    assert plain["strategy"] == "other"


def test_invoke_result_round_trips_and_clients_pickle():
    result = InvokeResult(messages=USER, response="r", tool_calls=[], node_path=["agent"], mock_calls=[], error=None, error_class="none", seconds=0.1)
    assert InvokeResult.from_dict(result.to_dict()) == result
    assert InvokeResult.from_dict({"response": "x"}).error_class == "none"
    local = pickle.loads(pickle.dumps(LocalAgent(WEATHER, agent_model_spec="scripted:examples.weather_bot.agent:default_scripted_model")))
    assert local.invoke(USER, {"tools": RULES}).error is None
    remote = pickle.loads(pickle.dumps(RemoteAgent("http://127.0.0.1:9/", timeout=3)))
    assert remote.endpoint == "http://127.0.0.1:9" and remote.mode == "remote"


def test_remote_agent_reports_connection_failures_as_infrastructure_errors():
    result = RemoteAgent("http://127.0.0.1:9", timeout=2).invoke(USER, {"tools": RULES})
    assert result.error and result.error_class == "infrastructure" and result.response == ""
