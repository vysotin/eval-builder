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


# ── agent skills ───────────────────────────────────────────────

TRIAGE_SKILL = """---
name: incident-triage
description: Triage an incident before remediation.
---
# Triage
Call get_service_status first, then search_runbooks.
"""
COMMS_SKILL = """---
name: incident-comms
description: Word a status update.
---
Always name the ticket id in the update.
"""

INLINE_SOURCE = '''
from pathlib import Path
from langchain_core.tools import tool
from langchain.agents import create_agent
from evalbuilder.skills import load_skills, skills_inline_prompt

SKILLS_DIR = Path(__file__).parent / "skills"
SKILLS = load_skills(SKILLS_DIR)

@tool
def get_service_status(service: str) -> dict:
    """status"""
    return {}

@tool
def search_runbooks(query: str) -> dict:
    """runbooks"""
    return {}

TOOLS = [get_service_status, search_runbooks]
TRIAGE_BASE = "You are the triage specialist. Never invent status values."
TRIAGE_PROMPT = TRIAGE_BASE + "\\n\\n" + skills_inline_prompt(SKILLS, "incident-triage")
COMMS_PROMPT = f"You are the comms specialist. {skills_inline_prompt(SKILLS, 'incident-comms')}"

def build_agent(model=None, tools=None):
    triage_agent = create_agent(model, TOOLS, system_prompt=TRIAGE_PROMPT)
    comms_agent = create_agent(model, [], system_prompt=COMMS_PROMPT)
    return triage_agent
'''

ON_DEMAND_SOURCE = '''
from pathlib import Path
from langchain_core.tools import tool
from langchain.agents import create_agent
from evalbuilder.skills import load_skills, skills_prompt, skill_loader_tool

SKILLS = load_skills(Path(__file__).with_name("skills"))
load_skill = skill_loader_tool(SKILLS)

@tool
def lookup_order(order_id: str) -> dict:
    """lookup"""
    return {}

TOOLS = [lookup_order, load_skill]
SUPPORT_PROMPT = "You are support. Always call lookup_order first."

def build_agent(model=None, tools=None):
    tools = TOOLS if tools is None else tools
    by_name = {t.name: t for t in tools}
    prompt = SUPPORT_PROMPT + ("\\n\\n" + skills_prompt(SKILLS) if "load_skill" in by_name else "")
    support_agent = create_agent(model, tools, system_prompt=prompt)
    return support_agent
'''

KWARG_SOURCE = '''
from langchain.agents import create_agent
SKILLS_DIR = "skills"

def build_agent(model=None, tools=None):
    agent = create_agent(model, [], system_prompt="Use the incident-triage skill when an incident is reported.", skills=[SKILLS_DIR])
    return agent
'''


def _write_skills(root: Path) -> None:
    (root / "skills" / "incident-triage").mkdir(parents=True)
    (root / "skills" / "incident-triage" / "SKILL.md").write_text(TRIAGE_SKILL)
    (root / "skills" / "incident-comms").mkdir(parents=True)
    (root / "skills" / "incident-comms" / "SKILL.md").write_text(COMMS_SKILL)


def test_ast_discovers_inline_skills_and_composed_prompts(tmp_path):
    _write_skills(tmp_path)
    src = tmp_path / "agent.py"
    src.write_text(INLINE_SOURCE)
    amap = discover_from_source(src)
    assert [s["name"] for s in amap.skills] == ["incident-comms", "incident-triage"]
    assert amap.app["skills_dir"] == str(tmp_path / "skills")
    triage = next(s for s in amap.skills if s["name"] == "incident-triage")
    assert triage["prompt"].startswith("# Triage") and triage["tools_mentioned"] == ["get_service_status", "search_runbooks"]
    assert triage["used_by"] == ["triage_agent"] and triage["evidence"] == ["skill:incident-triage"]
    nodes = {n["id"]: n for n in amap.graph["nodes"] if n["kind"] == "llm"}
    assert nodes["triage_agent"]["prompt"].startswith("You are the triage specialist. Never invent status values.\n\n## Skill: incident-triage")
    assert "Call get_service_status first" in nodes["triage_agent"]["prompt"]
    assert nodes["triage_agent"]["skills"] == ["incident-triage"] and nodes["triage_agent"]["skills_source"] == "inline"
    assert nodes["triage_agent"]["tools"] == ["get_service_status", "search_runbooks"]
    assert nodes["comms_agent"]["prompt"] == "You are the comms specialist. ## Skill: incident-comms\nAlways name the ticket id in the update."
    assert nodes["comms_agent"]["skills"] == ["incident-comms"]
    tools = {t["name"]: t for t in amap.tools}
    assert all(t["kind"] == "tool" and t["mockable"] for t in tools.values())


def test_ast_discovers_on_demand_skills_and_the_loader_tool(tmp_path):
    _write_skills(tmp_path)
    src = tmp_path / "agent.py"
    src.write_text(ON_DEMAND_SOURCE)
    amap = discover_from_source(src)
    assert [s["name"] for s in amap.skills] == ["incident-comms", "incident-triage"]
    tools = {t["name"]: t for t in amap.tools}
    assert list(tools) == ["lookup_order", "load_skill"]
    assert tools["lookup_order"]["kind"] == "tool" and tools["lookup_order"]["mockable"] is True
    loader = tools["load_skill"]
    assert loader["kind"] == "skill_loader" and loader["mockable"] is False and loader["edge_cases"] == []
    assert loader["args_schema"]["properties"]["name"]["type"] == "string" and loader["args_schema"]["required"] == ["name"]
    assert "skill" in loader["description"].lower()
    node = next(n for n in amap.graph["nodes"] if n["id"] == "support_agent")
    assert node["prompt"].startswith("You are support. Always call lookup_order first.\n\nSkills available on demand")
    assert "- incident-triage: Triage an incident before remediation." in node["prompt"]
    assert node["skills"] == ["incident-comms", "incident-triage"] and node["skills_source"] == "listing"
    assert node["tools"] == ["lookup_order", "load_skill"]
    assert all(s["used_by"] == ["support_agent"] for s in amap.skills)


def test_ast_discovers_skills_from_factory_kwarg_and_prompt_mentions(tmp_path):
    _write_skills(tmp_path)
    src = tmp_path / "agent.py"
    src.write_text(KWARG_SOURCE)
    amap = discover_from_source(src)
    assert [s["name"] for s in amap.skills] == ["incident-comms", "incident-triage"]
    node = next(n for n in amap.graph["nodes"] if n["kind"] == "llm")
    assert node["skills"] == ["incident-triage"] and node["skills_source"] == "prompt"
    assert next(s for s in amap.skills if s["name"] == "incident-triage")["used_by"] == [node["id"]]
    assert next(s for s in amap.skills if s["name"] == "incident-comms")["used_by"] == []


def test_agents_without_skills_have_no_skill_fields():
    amap = discover_from_source(WEATHER)
    assert amap.skills == [] and "skills_dir" not in amap.app
    assert all("skills" not in n for n in amap.graph["nodes"])
    assert all(t["kind"] == "tool" and t["mockable"] for t in amap.tools)


def test_live_tool_description_classifies_skill_loaders(tmp_path):
    from evalbuilder import skills as sk
    from evalbuilder.tool_schemas import describe_tool

    _write_skills(tmp_path)
    loader = sk.skill_loader_tool(sk.load_skills(tmp_path / "skills"))
    entry = describe_tool(loader)
    assert entry["kind"] == "skill_loader" and entry["mockable"] is False and entry["edge_cases"] == []
    assert entry["args_schema"]["required"] == ["name"]
    from examples.weather_bot.agent import get_weather

    plain = describe_tool(get_weather)
    assert plain["kind"] == "tool" and plain["mockable"] is True and plain["edge_cases"]
