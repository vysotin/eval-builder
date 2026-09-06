"""The agent server (`evalbuilder serve`): health, invoke with mocks, multi-turn round
trips, error responses — and the RemoteAgent talking to it."""

import json
import urllib.error
import urllib.request

import pytest
from typer.testing import CliRunner

from evalbuilder.agent_client import LocalAgent, RemoteAgent
from evalbuilder.cli import app
from evalbuilder.serve import SERVER_VERSION, make_server

WEATHER = "examples.weather_bot.agent"
RULES = {"get_weather": [{"matchArgs": {}, "response": {"temp": 55, "condition": "mocked-rain", "city": "Paris"}}]}
USER = [{"role": "user", "content": "What is the weather in Paris?"}]


@pytest.fixture(scope="module")
def server():
    srv = make_server(LocalAgent(WEATHER), "127.0.0.1", 0)
    srv.serve_in_thread()
    yield srv
    srv.shutdown()
    srv.server_close()


def _post(endpoint: str, path: str, payload) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
    req = urllib.request.Request(f"{endpoint}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_health_lists_the_target(server):
    with urllib.request.urlopen(f"{server.endpoint}/health", timeout=10) as resp:
        body = json.loads(resp.read().decode())
    assert body["ok"] is True and body["module"] == WEATHER and body["server"] == SERVER_VERSION
    assert body["tools"] == ["get_weather", "get_alerts"] and body["factory"] == "build_agent"
    assert server.endpoint.startswith("http://127.0.0.1:") and server.port > 0


def test_invoke_installs_the_request_mocks(server):
    status, body = _post(server.endpoint, "/invoke", {"messages": USER, "mocks": {"tools": RULES, "on_miss": "strict"}})
    assert status == 200 and body["error"] is None
    assert "55" in body["response"] and body["tool_calls"] == [{"name": "get_weather", "args": {"city": "Paris"}}]
    assert [m["role"] for m in body["messages"]][-2:] == ["tool", "assistant"]
    assert body["mock_calls"][0]["layer"] == "rule" and body["node_path"] and body["seconds"] >= 0


def test_remote_agent_matches_local_agent_and_continues_multi_turn(server):
    remote = RemoteAgent(server.endpoint, timeout=10)
    assert remote.health()["ok"] is True and remote.health()["mode"] == "remote"
    local = LocalAgent(WEATHER).invoke(USER, {"tools": RULES})
    first = remote.invoke(USER, {"tools": RULES})
    assert (first.response, first.tool_calls, first.error) == (local.response, local.tool_calls, None)
    history = first.messages + [{"role": "user", "content": "And alerts in Paris?"}]
    second = remote.invoke(history, {"tools": {**RULES, "get_alerts": [{"matchArgs": {}, "response": {"alerts": ["storm"], "city": "Paris"}}]}})
    assert second.error is None and "storm" in second.response
    assert [m["role"] for m in second.messages].count("user") == 2 and second.tool_calls[-1]["name"] == "get_alerts"


def test_invoke_without_mocks_calls_real_tools_and_agent_errors_are_reported(server):
    status, body = _post(server.endpoint, "/invoke", {"messages": USER, "mocked": False})
    assert status == 200 and body["error"] is None and "72" in body["response"]
    status, body = _post(server.endpoint, "/invoke", {"messages": USER, "mocks": {"tools": {"get_weather": [{"matchArgs": {"city": "Oslo"}, "response": {}}]}, "on_miss": "strict"}})
    assert status == 200 and body["error"] and body["error_class"] == "agent"


def test_bad_requests_get_400_and_unknown_routes_404(server):
    assert _post(server.endpoint, "/invoke", {"mocks": {}})[0] == 400
    assert _post(server.endpoint, "/invoke", b"not json")[0] == 400
    assert _post(server.endpoint, "/invoke", {"messages": USER, "mocks": []})[0] == 400
    assert _post(server.endpoint, "/nope", {"messages": USER})[0] == 404
    try:
        urllib.request.urlopen(f"{server.endpoint}/nope", timeout=10)
    except urllib.error.HTTPError as e:
        assert e.code == 404
    else:
        raise AssertionError("expected 404")


def test_health_reports_a_broken_target_as_500():
    srv = make_server(LocalAgent("examples.no_such_module"), "127.0.0.1", 0)
    srv.serve_in_thread()
    try:
        urllib.request.urlopen(f"{srv.endpoint}/health", timeout=10)
    except urllib.error.HTTPError as e:
        assert e.code == 500 and "no_such_module" in json.loads(e.read().decode())["error"]
    else:
        raise AssertionError("expected 500")
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_serve_help_and_env_defaults(monkeypatch):
    result = CliRunner().invoke(app, ["serve", "--help"])
    assert result.exit_code == 0 and "--module" in result.output and "--port" in result.output
    seen = {}

    def fake_serve(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr("evalbuilder.serve.serve", fake_serve)
    monkeypatch.setenv("EVALBUILDER_AGENT_MODEL", "scripted:examples.weather_bot.agent:default_scripted_model")
    result = CliRunner().invoke(app, ["serve", "--module", WEATHER, "--port", "0"])
    assert result.exit_code == 0, result.output
    assert seen["module"] == WEATHER and seen["port"] == 0 and seen["agent_model"] == "scripted:examples.weather_bot.agent:default_scripted_model"
