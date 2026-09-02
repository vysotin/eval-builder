"""The incident_desk example: router graph over four specialists, pydantic tool schemas,
schema-derived edge cases, and the whole pipeline offline with a scripted generator."""

import json
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from evalbuilder.config import Settings
from evalbuilder.discover import discover_from_source
from evalbuilder.pipeline.config import PIPELINE_CONFIG_SCHEMA, load_config
from evalbuilder.pipeline.report import run_pipeline
from evalbuilder.tool_schemas import describe_tool
from examples.incident_desk import agent as idk
from examples.incident_desk import offline

SCRIPTED = "scripted:examples.incident_desk.agent:default_scripted_model"
GENERATOR = "scripted:examples.incident_desk.offline:generator_model"
TOOL_NAMES = ["get_service_status", "search_runbooks", "create_ticket", "page_oncall", "post_status_update"]
SIDE_EFFECTING = {"create_ticket", "page_oncall", "post_status_update"}


def _mock_tools(status=offline.STATUS_API, receipt=offline.TICKET):
    calls = []

    @tool
    def get_service_status(service: str) -> dict:
        """mock"""
        calls.append(("get_service_status", service))
        return status

    @tool(args_schema=idk.RunbookQuery)
    def search_runbooks(query: str, severity: str = "sev2", limit: int = 3) -> dict:
        """mock"""
        calls.append(("search_runbooks", severity))
        return offline.RUNBOOKS

    @tool
    def create_ticket(ticket: idk.TicketRequest) -> dict:
        """mock"""
        calls.append(("create_ticket", ticket.severity))
        return receipt

    @tool
    def page_oncall(team: str, severity: str, reason: str) -> dict:
        """mock"""
        calls.append(("page_oncall", team))
        return {"paged": True, "team": team}

    @tool
    def post_status_update(update: idk.StatusUpdate) -> dict:
        """mock"""
        calls.append(("post_status_update", update.channel))
        return {"posted": True, "channel": update.channel}

    return [get_service_status, search_runbooks, create_ticket, page_oncall, post_status_update], calls


def _ask(graph, text, messages=None):
    messages = list(messages or []) + [{"role": "user", "content": text}]
    return graph.invoke({"messages": messages})


def _names(calls):
    return [c[0] for c in calls]


# ── routing + tool gating ──────────────────────────────────────


def test_incident_triages_and_gates_side_effects_on_yes():
    tools, calls = _mock_tools()
    graph = idk.build_agent(tools=tools)
    out = _ask(graph, "The api is down, this is a sev1 — 500s on every request.")
    assert out["route"] == "incident"
    assert calls == [("get_service_status", "api"), ("search_runbooks", "sev1")]
    reply = out["messages"][-1].content
    assert "Reply yes" in reply and "api" in reply
    assert not SIDE_EFFECTING & set(_names(calls))

    out2 = _ask(graph, "yes", out["messages"])
    assert _names(calls)[-3:] == ["create_ticket", "page_oncall", "post_status_update"]
    assert calls[-2] == ("page_oncall", "api-oncall") and calls[-1] == ("post_status_update", "status-page")
    assert "INC-1042" in out2["messages"][-1].content


def test_status_question_routes_to_status_agent():
    tools, calls = _mock_tools()
    out = _ask(idk.build_agent(tools=tools), "What is the status of api?")
    assert out["route"] == "status_question"
    assert calls == [("get_service_status", "api")]
    assert "api" in out["messages"][-1].content and "degraded" in out["messages"][-1].content


def test_invalid_severity_lists_levels_without_side_effects():
    for text in ("api is throwing errors, severity critical", "api is down, call it a sev9"):
        tools, calls = _mock_tools()
        out = _ask(idk.build_agent(tools=tools), text)
        assert out["route"] == "incident"
        assert _names(calls) == ["get_service_status"]
        assert "sev1, sev2 or sev3" in out["messages"][-1].content


def test_missing_service_asks_which_service():
    tools, calls = _mock_tools()
    out = _ask(idk.build_agent(tools=tools), "Something is broken, open a ticket")
    assert out["route"] == "incident" and not calls
    assert "which service" in out["messages"][-1].content.lower()
    tools, calls = _mock_tools()
    out = _ask(idk.build_agent(tools=tools), "What's the status right now?")
    assert out["route"] == "status_question" and not calls
    assert "which service" in out["messages"][-1].content.lower()


def test_incomplete_payloads_are_reported_not_invented():
    tools, calls = _mock_tools(status={"state": "healthy", "error_rate_pct": 0.0, "open_incidents": 0, "last_deploy": "2026-08-01"})
    out = _ask(idk.build_agent(tools=tools), "What is the status of api?")
    assert "incomplete" in out["messages"][-1].content
    tools, calls = _mock_tools(receipt={"url": "https://ops.example.invalid/x", "status": "open"})
    graph = idk.build_agent(tools=tools)
    out = _ask(graph, "api is down, sev1.")
    out2 = _ask(graph, "yes", out["messages"])
    assert "incomplete" in out2["messages"][-1].content
    assert "page_oncall" not in _names(calls) and "post_status_update" not in _names(calls)


def test_out_of_scope_declines_without_tools():
    tools, calls = _mock_tools()
    out = _ask(idk.build_agent(tools=tools), "Write me a poem about the sea")
    assert out["route"] == "other" and not calls
    assert out["messages"][-1].content == idk.DECLINE_TEXT


def test_real_tools_fail_fast_when_unmocked():
    out = _ask(idk.build_agent(), "The api is down, sev1")
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert tool_msgs and "unavailable" in tool_msgs[0].content
    assert "could not reach" in out["messages"][-1].content.lower()
    assert not [tc for m in out["messages"] if isinstance(m, AIMessage) for tc in (m.tool_calls or []) if tc["name"] in SIDE_EFFECTING]


def test_tools_and_contract():
    assert [t.name for t in idk.TOOLS] == TOOL_NAMES
    for name in SIDE_EFFECTING:
        assert "Side-effecting" in next(t for t in idk.TOOLS if t.name == name).description
    assert [s.name for s in idk.SKILLS] == ["incident-comms", "incident-triage"]
    assert "## Skill: incident-triage" in idk.TRIAGE_PROMPT and "## Skill: incident-comms" in idk.COMMS_PROMPT


def test_skip_the_status_check_request_still_follows_the_triage_skill():
    tools, calls = _mock_tools()
    out = _ask(idk.build_agent(tools=tools), "Don't bother checking status, just give me the runbook for api, sev2.")
    assert out["route"] == "incident" and calls == [("get_service_status", "api"), ("search_runbooks", "sev2")]
    assert "Reply yes" in out["messages"][-1].content


# ── discovery ──────────────────────────────────────────────────


def test_discovery_captures_pydantic_schemas_and_edges():
    amap = discover_from_source(Path("examples/incident_desk/agent.py"))
    tools = {t["name"]: t for t in amap.tools}
    assert list(tools) == TOOL_NAMES

    runbooks = tools["search_runbooks"]
    assert runbooks["schema_source"] == "args_schema"
    props = runbooks["args_schema"]["properties"]
    assert props["severity"]["enum"] == ["sev1", "sev2", "sev3"] and props["severity"]["default"] == "sev2"
    assert props["limit"] == {"type": "integer", "minimum": 1, "maximum": 10, "default": 3}
    assert props["query"]["minLength"] == 3 and runbooks["args_schema"]["required"] == ["query"]

    ticket = tools["create_ticket"]
    assert ticket["args_schema"]["properties"]["ticket"] == {"$ref": "#/$defs/TicketRequest"}
    request = ticket["args_schema"]["$defs"]["TicketRequest"]
    assert request["properties"]["title"] == {"type": "string", "minLength": 5, "maxLength": 120}
    assert request["properties"]["severity"]["enum"] == ["sev1", "sev2", "sev3"]
    assert ticket["output_schema"]["title"] == "TicketReceipt"
    assert ticket["output_schema"]["properties"]["ticket_id"]["pattern"] == r"^INC-\d{4,}$"
    assert ticket["models"] == ["TicketRequest", "TicketReceipt"]

    assert {n for n, t in tools.items() if t["side_effecting"]} == SIDE_EFFECTING
    kinds = {e["kind"] for t in amap.tools for e in t["edge_cases"]}
    assert kinds == {"missing_required", "wrong_type", "out_of_enum", "boundary", "malformed_output"}
    assert [e["id"] for e in tools["get_service_status"]["edge_cases"]] == [
        "edge.get_service_status.missing_required.service", "edge.get_service_status.malformed_output",
    ]
    assert "edge.search_runbooks.out_of_enum.severity" in [e["id"] for e in runbooks["edge_cases"]]

    nodes = {n["id"]: n for n in amap.graph["nodes"] if n["kind"] == "llm"}
    assert nodes["triage_agent"]["tools"] == ["get_service_status", "search_runbooks"]
    assert nodes["remediation_agent"]["tools"] == ["search_runbooks", "create_ticket", "page_oncall"]
    assert nodes["comms_agent"]["tools"] == ["post_status_update"] and nodes["status_agent"]["tools"] == ["get_service_status"]
    assert amap.graph["conditional_edges"][0]["targets"] == ["triage_agent", "status_agent", "decline"]
    # inline skills: recorded, rendered into the prompts, linked to the nodes
    skills = {s["name"]: s for s in amap.skills}
    assert list(skills) == ["incident-comms", "incident-triage"] and amap.app["skills_dir"].endswith("examples/incident_desk/skills")
    assert skills["incident-triage"]["used_by"] == ["triage_agent"] and skills["incident-comms"]["used_by"] == ["comms_agent"]
    assert skills["incident-triage"]["allowed_tools"] == ["get_service_status", "search_runbooks"]
    assert skills["incident-triage"]["references"][0]["path"] == "references/severity-matrix.md"
    assert skills["incident-triage"]["tools_mentioned"] == ["get_service_status", "search_runbooks"]
    assert nodes["triage_agent"]["skills"] == ["incident-triage"] and nodes["triage_agent"]["skills_source"] == "inline"
    assert "Always call `get_service_status`" in nodes["triage_agent"]["prompt"]
    assert "skills" not in nodes["remediation_agent"] and all(t["kind"] == "tool" for t in amap.tools)


def test_live_enrichment_resolves_output_schemas():
    live = {t.name: describe_tool(t) for t in idk.TOOLS}
    assert live["get_service_status"]["output_schema"]["required"] == ["service", "state", "error_rate_pct", "open_incidents", "last_deploy"]
    assert live["get_service_status"]["output_schema"]["properties"]["state"]["enum"] == ["healthy", "degraded", "down"]
    assert live["create_ticket"]["output_schema"]["required"] == ["ticket_id", "url", "status"]
    assert live["search_runbooks"]["schema_source"] == "args_schema" and live["create_ticket"]["schema_source"] == "annotations"
    assert {n for n, t in live.items() if t["side_effecting"]} == SIDE_EFFECTING


# ── offline end-to-end pipeline ────────────────────────────────


def _config(tmp_path, per_tool_edge_cases=1):
    cfg = {
        "schema": PIPELINE_CONFIG_SCHEMA,
        "name": "incident-desk-test",
        "target": {"source": "examples/incident_desk/agent.py", "module": "examples.incident_desk.agent"},
        "models": {"agent": SCRIPTED, "judge": SCRIPTED, "generator": GENERATOR},
        "constraints": ["Never call create_ticket, page_oncall or post_status_update before the user says yes."],
        "coverage": {"total_cases": 10, "per_intent": {"happy": 1, "failure": 1}, "per_failure_category": 1,
                     "out_of_intent": 1, "multi_turn_share": 0.15, "per_tool_edge_cases": per_tool_edge_cases},
        "evaluators": [{"type": "expected_tools"}, {"type": "contains"}],
        "thresholds": {"default": 0.8, "slice_min": 0.5, "overall_pass": 0.8},
        "runs": {"repeats": 2},
        "stages": {"simulate": True, "publish": "never", "max_retries": 1},
        "review": {"auto_approve": True, "approved_by": "tester", "note": "offline e2e"},
        "output": {"dir": str(tmp_path / f"out-{per_tool_edge_cases}")},
    }
    path = tmp_path / f"pipeline-{per_tool_edge_cases}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _run(tmp_path, per_tool_edge_cases):
    cfg_path = _config(tmp_path, per_tool_edge_cases)
    _, report = run_pipeline(cfg_path, log=lambda m: None, settings=Settings())
    assert report["verdict"] == "pass" and report["overall_score"] == 1.0, (report["verdict_reasons"], report["problems"])
    assert report["coverage"]["coverage_pct"] == 100.0
    assert report["coverage"]["by_kind"]["schema-edge"]["covered"] == report["coverage"]["by_kind"]["schema-edge"]["planned"] > 0
    assert report["agent"]["tools"] == TOOL_NAMES
    assert report["stability"]["repeats"] == 2 and report["stability"]["unstable_cases"] == []
    assert report["simulation"]["details"]["stop_reasons"] == {"incident-confirmed": "success"}
    # inline skills: in the map, exercised by scenarios (incl. a skill_misuse one), counted in coverage
    assert report["agent"]["skills"] == ["incident-comms", "incident-triage"]
    assert report["coverage"]["uncovered_skills"] == [] and report["coverage"]["skills"]["incident-triage"]["cases"] >= 2
    assert "skill_misuse" in report["stages"]["map"]["details"]["failure_types"]
    assert report["stages"]["map"]["details"]["skills"]["incident-triage"] == [
        "scenario.incident-report.happy", "scenario.incident-report.skip-status-check"]
    # inline path: the map stage merges skill failure cases and tool failure scenarios into the artifact
    amap = json.loads((tmp_path / f"out-{per_tool_edge_cases}" / "agent-map.json").read_text())
    triage = next(s for s in amap["skills"] if s["name"] == "incident-triage")
    assert triage["tools"] and not triage["summarized"] and triage["instruction"] == triage["prompt"]
    assert {c["failure_mode"] for c in triage["failure_cases"]} == {"skill_misuse", "input_validation"}
    tools = {t["name"]: t for t in amap["tools"]}
    assert tools["get_service_status"]["failure_scenarios"][0]["failure_mode"] == "tool_error_handling"
    caps = [c for n in amap["graph"]["nodes"] for c in n.get("capabilities") or []]
    assert any(c.startswith("skill incident-triage (inline)") for c in caps)
    ds = json.loads((tmp_path / f"out-{per_tool_edge_cases}" / "dataset.json").read_text())
    misuse = [c for c in ds["cases"] if c["metadata"]["failure_mode"] == "skill_misuse"]
    assert misuse and all("skill:incident-triage" in c["metadata"]["evidence"] for c in misuse)
    return report, ds["cases"]


def test_offline_pipeline_with_one_edge_case_per_tool(tmp_path):
    report, cases = _run(tmp_path, per_tool_edge_cases=1)
    edge_cases = [c for c in cases if c["metadata"].get("edge")]
    assert {c["metadata"]["tool"] for c in edge_cases} == set(TOOL_NAMES)
    assert all(c["metadata"]["edge"]["kind"] == "missing_required" for c in edge_cases)
    assert all(c["review"]["status"] == "approved" for c in cases)
    assert set(report["metrics"]) == {"expected_tools", "contains"}
    assert any(c["metadata"].get("user_turns") == ["yes"] for c in cases)
    assert report["stages"]["review"]["details"]["approved_by"] == "tester"


def test_offline_pipeline_with_three_edge_cases_per_tool_injects_malformed_output(tmp_path):
    report, cases = _run(tmp_path, per_tool_edge_cases=3)
    edges = {c["metadata"]["edge"]["id"]: c for c in cases if c["metadata"].get("edge")}
    kinds = {e["kind"] for e in (c["metadata"]["edge"] for c in edges.values())}
    assert {"missing_required", "wrong_type", "out_of_enum", "malformed_output"} <= kinds
    status = edges["edge.get_service_status.malformed_output"]
    injected = status["metadata"]["mocks"]["tools"]["get_service_status"]
    assert injected[0]["matchArgs"] == {} and "service" not in injected[0]["response"] and "state" in injected[0]["response"]
    assert status["reference_outputs"]["contains"] == "incomplete"
    ticket = edges["edge.create_ticket.malformed_output"]
    assert "ticket_id" not in ticket["metadata"]["mocks"]["tools"]["create_ticket"][0]["response"]
    assert ticket["metadata"]["user_turns"] == ["yes"] and "page_oncall" in ticket["reference_outputs"]["forbidden_tools"]
    assert edges["edge.search_runbooks.out_of_enum.severity"]["reference_outputs"]["contains"] == "sev1, sev2 or sev3"
    assert report["coverage"]["by_kind"]["schema-edge"]["planned"] == 13


def test_example_pipeline_config_is_valid():
    cfg = load_config(Path("examples/incident_desk/pipeline.yaml"))
    assert cfg.problems() == []
    assert cfg.name == "incident-desk" and cfg.coverage.per_tool_edge_cases == 1 and cfg.instructions.strip()
    assert cfg.review.auto_approve and "incident_desk" in cfg.review.approved_by
