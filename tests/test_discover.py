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


PYDANTIC_SOURCE = '''
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class Query(BaseModel):
    """Search query."""
    text: str = Field(description="what to search")
    severity: Literal["low", "high"] = "low"
    limit: int = Field(3, ge=1, le=10)

class Receipt(BaseModel):
    ticket_id: str
    url: str | None = None

class Ticket(BaseModel):
    title: str = Field(..., min_length=3)
    priority: Literal["p1", "p2"]
    tags: list[str] = []
    receipt: Receipt | None = None

@tool(args_schema=Query)
def search(text: str, severity: str = "low", limit: int = 3) -> dict:
    """Search things."""
    return {}

@tool
def create(ticket: Ticket, note: str | None = None) -> Receipt:
    """Create a ticket. Side-effecting: only call after the user confirms."""
    return Receipt(ticket_id="T1")

TOOLS = [search, create]
'''


def test_ast_discovery_resolves_pydantic_schemas(tmp_path):
    src = tmp_path / "agent.py"
    src.write_text(PYDANTIC_SOURCE)
    amap = discover_from_source(src)
    by_name = {t["name"]: t for t in amap.tools}
    search = by_name["search"]
    assert search["schema_source"] == "args_schema"
    assert search["args_schema"]["properties"]["severity"]["enum"] == ["low", "high"]
    assert search["args_schema"]["properties"]["limit"] == {"type": "integer", "default": 3, "minimum": 1, "maximum": 10}
    assert search["args_schema"]["required"] == ["text"] and search["args_schema"]["description"] == "Search query."
    create = by_name["create"]
    assert create["schema_source"] == "ast" and create["side_effecting"] is True
    assert create["args_schema"]["properties"]["ticket"] == {"$ref": "#/$defs/Ticket"}
    assert set(create["args_schema"]["$defs"]) == {"Ticket", "Receipt"}
    ticket = create["args_schema"]["$defs"]["Ticket"]
    assert ticket["required"] == ["title", "priority"] and ticket["properties"]["title"]["minLength"] == 3
    assert ticket["properties"]["tags"] == {"type": "array", "items": {"type": "string"}, "default": []}
    assert create["args_schema"]["properties"]["note"] == {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}
    assert create["output_schema"]["title"] == "Receipt" and create["output_schema"]["required"] == ["ticket_id"]
    assert create["models"] == ["Receipt", "Ticket"]
    kinds = [e["kind"] for e in create["edge_cases"]]
    assert kinds[:3] == ["missing_required", "wrong_type", "malformed_output"]
    assert amap.app["models"] == ["Query", "Receipt", "Ticket"]
