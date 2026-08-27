import json
from pathlib import Path

from typer.testing import CliRunner

from evalbuilder.cli import app
from evalbuilder.discover import discover_from_source, discover_live

WEATHER = Path("examples/weather_bot/agent.py")


def test_ast_finds_tools_with_schemas():
    m = discover_from_source(WEATHER)
    names = {t["name"] for t in m.tools}
    assert {"get_weather", "get_alerts"} <= names
    gw = next(t for t in m.tools if t["name"] == "get_weather")
    assert gw["args_schema"]["properties"]["city"]["type"] == "string"
    assert "city" in gw["args_schema"]["required"]
    assert gw["description"]  # docstring captured


def test_ast_finds_prompt_and_react_node():
    m = discover_from_source(Path("examples/travel_planner/agent.py"))
    assert any(
        "Never confirm a booking" in (n.get("prompt") or "") for n in m.graph["nodes"]
    )
    assert m.source_sha256


def test_ast_links_tools_to_nodes():
    m = discover_from_source(WEATHER)
    gw = next(t for t in m.tools if t["name"] == "get_weather")
    assert "build_agent" in gw["used_by"]


def test_live_introspection_returns_nodes():
    live = discover_live("examples.weather_bot.agent")
    assert "error" not in live
    assert any("agent" in n or "tools" in n for n in live["nodes"])


def test_live_handles_import_error():
    assert "error" in discover_live("no.such.module")


def test_cli_discover_writes_agent_map(tmp_path):
    runner = CliRunner()
    r = runner.invoke(
        app,
        ["discover", "examples.weather_bot.agent", "--source", str(WEATHER),
         "--eval-dir", str(tmp_path)],
    )
    assert r.exit_code == 0, r.output
    data = json.loads((tmp_path / "agent-map.json").read_text())
    assert data["schema"] == "evalbuilder/agent-map/v1"
    assert data["decisions_needed"]
    assert data["tools"]
    assert "live" in data["graph"] and "error" not in data["graph"]["live"]


def test_ast_names_llm_nodes_by_assignment_and_skips_shim():
    m = discover_from_source(Path("examples/support_bot/agent.py"))
    llm_ids = [n["id"] for n in m.graph["nodes"] if n["kind"] == "llm"]
    assert llm_ids == ["support_agent", "kb_agent"]
    graph_ids = [n["id"] for n in m.graph["nodes"] if n["kind"] == "graph-node"]
    assert {"classify", "support_agent", "kb_agent", "decline"} <= set(graph_ids)
    assert m.graph["conditional_edges"] and m.graph["conditional_edges"][0]["source"] == "classify"
    support = next(n for n in m.graph["nodes"] if n["id"] == "support_agent")
    assert "issue_refund ONLY" in support["prompt"]
    assert "lookup_order" in support["tools"]
