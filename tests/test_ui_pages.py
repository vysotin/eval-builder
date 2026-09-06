"""Report-UI pages rendered headlessly with streamlit's AppTest (no browser).

Every page is rendered the way the app does it: `project.begin_run()` derives the
bundle from the project in session state, then the page renders."""

import re

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from evalbuilder.ui import loader  # noqa: E402

PAGES = ("summary", "agent", "intents", "dataset", "coverage", "results", "stability", "simulation", "analysis", "stages")
EXPECTED_HEADERS = {
    "summary": "Summary", "agent": "Agent", "intents": "Intents & scenarios", "dataset": "Dataset & mocks",
    "coverage": "Coverage", "results": "Eval results", "stability": "Stability", "simulation": "Simulation",
    "analysis": "Analysis", "stages": "Stages & problems",
}
EXAMPLE = "docs/examples/support-bot"
INCIDENT_EXAMPLE = "docs/examples/incident-desk"


@pytest.fixture(scope="module")
def llm_bundle_dir(tmp_path_factory):
    """A support_bot pipeline output produced offline with agent skills (on-demand loader)
    and the LLM mock layer (`on_miss: llm`, scripted mock model) — one run per module."""
    from evalbuilder.config import Settings
    from evalbuilder.pipeline import setup as setup_mod
    from evalbuilder.pipeline.report import run_pipeline

    root = tmp_path_factory.mktemp("llm")
    form = setup_mod.default_form({"name": "support-llm", "source": "examples/support_bot/agent.py", "module": "examples.support_bot.agent"})
    form.update(agent_model="scripted:examples.support_bot.agent:default_scripted_model",
                judge_model="scripted:examples.support_bot.agent:default_scripted_model",
                generator_model="scripted:examples.support_bot.offline:generator_model",
                mock_model="scripted:examples.support_bot.offline:mock_model", on_miss="llm",
                evaluators=["expected_tools", "contains"], total_cases=6, happy=1, failure=1, per_failure_category=1, out_of_intent=1,
                multi_turn_share=0.0, per_tool_edge_cases=0, repeats=1, simulate=True, auto_approve=True, approved_by="tester",
                output_dir=str(root / "out"), config_path=str(root / "support-llm.yaml"))
    cfg = setup_mod.build_config(form)
    cfg.stages.publish = "never"
    cfg.save(root / "support-llm.yaml")
    _, report = run_pipeline(root / "support-llm.yaml", settings=Settings())
    assert report["stages"]["infer"]["status"] == "ok", report["stages"]
    return str(root / "out")


def _page_script(page):
    import importlib

    from evalbuilder.ui import project

    project.begin_run()
    importlib.import_module(f"evalbuilder.ui.app_pages.{page}").render()


def _run(page: str, project=None) -> AppTest:
    at = AppTest.from_function(_page_script, kwargs={"page": page}, default_timeout=120)
    at.session_state["project"] = project
    at.run()
    return at


def _dir(path: str) -> dict:
    return {"mode": "dir", "dir": path}


def _errors(at: AppTest) -> list[str]:
    return [re.sub(r"\x1b\[[0-9;]*m", "", e.value) for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = _run(page, _dir(EXAMPLE))
    assert _errors(at) == []
    assert [h.value for h in at.header] == [EXPECTED_HEADERS[page]]
    assert len(at.subheader) >= 3
    assert not at.warning, [w.value for w in at.warning]


@pytest.mark.parametrize("page", PAGES)
def test_page_without_project_points_to_setup(page):
    at = _run(page, None)
    assert _errors(at) == []
    assert at.header == [] and at.subheader == []
    assert any("No project selected" in i.value and "Pipeline setup" in i.value for i in at.info)


@pytest.mark.parametrize("page", PAGES)
def test_page_for_unrun_folder_says_no_artifacts(page, tmp_path):
    at = _run(page, _dir(str(tmp_path / "fresh")))
    assert _errors(at) == []
    assert at.header == []
    assert any("No artifacts yet" in i.value for i in at.info)


@pytest.mark.parametrize("page", PAGES)
def test_page_explains_missing_artifacts(page):
    partial = loader.Bundle(name="partial", source="uploads", artifacts={"pipeline_config": {"name": "partial"}})
    at = _run(page, {"mode": "uploads", "bundle": partial})
    assert _errors(at) == []
    if page not in ("summary", "results", "stages"):
        assert at.warning, "pages needing an artifact must say which one is missing"
        assert "written by stage" in at.warning[0].value


def test_summary_shows_verdict_and_metrics():
    at = _run("summary", _dir(EXAMPLE))
    labels = [m.label for m in at.metric]
    assert "Overall score" in labels and "Coverage" in labels
    assert at.metric[0].value == "0.59"
    assert any("metric contains 0.5 < 0.7" in m.value for m in at.markdown)


def test_dataset_page_filters_and_details():
    at = _run("dataset", _dir(EXAMPLE))
    assert at.metric[0].value == "21"
    at.multiselect(key="ds_failure").select("tool_error_handling").run()
    assert _errors(at) == []
    assert any(re.match(r"\d+ of 21 cases", c.value) for c in at.caption)
    assert at.selectbox(key="ds_case").value.startswith("case-")


def test_results_page_drilldown_switches_case():
    at = _run("results", _dir(EXAMPLE))
    box = at.selectbox(key="results_case")
    first = box.value
    box.select_index(1).run()
    assert _errors(at) == []
    assert at.selectbox(key="results_case").value != first
    assert len(at.tabs) >= 2  # slices + per-run tabs


def test_agent_graph_dot_covers_nodes_and_tools():
    from evalbuilder.ui.app_pages.agent import graph_dot

    amap = loader.load_dir(EXAMPLE).agent_map
    dot = graph_dot(amap["graph"], amap["tools"], live=False, show_tools=True)
    for node in ("classify", "support_agent", "kb_agent", "decline"):
        assert f'"{node}"' in dot
    assert '"classify" -> "support_agent" [style=dashed' in dot
    assert '"tool:lookup_order"' in dot and "digraph agent" in dot
    live = graph_dot(amap["graph"], amap["tools"], live=True, show_tools=False)
    assert "START -> \"classify\"" in live and "tool:" not in live


def test_incident_desk_example_loads_with_schema_data():
    incident = loader.load_dir(INCIDENT_EXAMPLE)
    assert incident.problems == [] and incident.name == "incident-desk"
    tools = {t["name"]: t for t in incident.agent_map["tools"]}
    assert tools["search_runbooks"]["schema_source"] == "args_schema"
    assert tools["create_ticket"]["output_schema"]["title"] == "TicketReceipt" and "TicketRequest" in tools["create_ticket"]["models"]
    assert tools["create_ticket"]["side_effecting"] and any(e["kind"] == "malformed_output" for e in tools["create_ticket"]["edge_cases"])
    edges = [c for c in incident.dataset["cases"] if c["metadata"].get("edge")]
    assert edges and {c["metadata"]["failure_mode"] for c in edges} <= {"input_validation", "tool_error_handling"}
    assert incident.report["config"]["models"]["generator"] == "claude-cli:claude-sonnet-5"
    assert incident.report["config"]["instructions"].startswith("Ops domain")
    assert "schema-edge" in incident.report["coverage"]["by_kind"]
    assert incident.verdict == incident.report["verdict"]  # the committed run stopped after dataset (--until): incomplete


@pytest.mark.parametrize("page", PAGES)
def test_incident_desk_pages_render(page):
    at = _run(page, _dir(INCIDENT_EXAMPLE))
    assert _errors(at) == []
    if at.warning:  # an optional stage did not produce its artifact: the page must say which one
        assert "written by stage" in at.warning[0].value
    else:
        assert [h.value for h in at.header] == [EXPECTED_HEADERS[page]]
    if page == "agent":
        text = "\n".join(m.value for m in at.markdown)
        assert "Output schema" in text and "Schema edge cases" in text
    if page == "dataset":
        assert at.selectbox(key="ds_case").value.startswith("case-") and len(at.subheader) >= 3


def test_uploaded_bundle_is_a_read_only_project():
    bundle = loader.load_dir(EXAMPLE)
    at = _run("summary", {"mode": "uploads", "bundle": bundle})
    assert _errors(at) == [] and [h.value for h in at.header] == ["Summary"]
    at = _run("run", {"mode": "uploads", "bundle": bundle})
    assert any("read-only" in i.value for i in at.info)



# ── skills + the LLM mock layer on every page ──────────────────


@pytest.mark.parametrize("page", PAGES)
def test_llm_and_skills_bundle_pages_render(page, llm_bundle_dir):
    at = _run(page, _dir(llm_bundle_dir))
    assert _errors(at) == []
    text = "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)
    if page == "agent":
        assert any(h.value == "Skills" for h in at.subheader)
        assert "refund-policy" in text and "product-troubleshooting" in text and "skill_loader" in "\n".join(str(d.value) for d in at.dataframe)
        assert any("skills: product-troubleshooting, refund-policy" in e.label for e in at.expander)
        assert "skill refund-policy (listing) → tools lookup_order, check_refund_policy, issue_refund" in text
        assert "(not wired on this node)" in text  # kb_agent lacks the refund tools
        assert "Failure cases beyond tool failure" in text and "Failure scenarios" in text
        assert "full instructions are" in text  # refund-policy is summarized
    if page == "dataset":
        assert any(h.value.startswith("Mock strategies") for h in at.subheader)
        assert "mock miss policy `llm`" in text and "scripted:examples.support_bot.offline:mock_model" in text
        assert "Acme Store" in text  # the world
    if page == "results":
        assert any("Mock calls" in e.label for e in at.expander)
    if page == "summary":
        assert any(h.value == "Mocking" for h in at.subheader) and "rules → llm_engine" in text
    if page == "coverage":
        assert any(h.value == "Skills" for h in at.subheader)
    if page == "intents":
        assert any("skills: refund-policy" in e.label for e in at.expander)


def test_llm_bundle_agent_map_carries_skill_analysis(llm_bundle_dir):
    import json as _json
    from pathlib import Path as _P

    amap = _json.loads((_P(llm_bundle_dir) / "agent-map.json").read_text())
    by_name = {s["name"]: s for s in amap["skills"]}
    refund = by_name["refund-policy"]
    assert refund["tools"] == ["lookup_order", "check_refund_policy", "issue_refund"]
    assert refund["summarized"] and len(refund["instruction"]) <= 1500 < refund["chars"]
    assert {c["failure_mode"] for c in refund["failure_cases"]} == {"skill_misuse", "tool_misuse"}
    assert by_name["product-troubleshooting"]["failure_cases"][0]["failure_mode"] == "skill_misuse"
    tools = {t["name"]: t for t in amap["tools"]}
    assert tools["issue_refund"]["failure_scenarios"][0]["failure_mode"] == "tool_error_handling"
    assert tools["search_kb"]["failure_scenarios"][0]["failure_mode"] == "retrieval_grounding"
    assert "failure_scenarios" not in tools["load_skill"]  # the loader is local: no tool-failure analysis
    nodes = {n["id"]: n for n in amap["graph"]["nodes"] if n.get("kind") == "llm"}
    assert any(c.startswith("skill refund-policy (listing) → tools lookup_order") for c in nodes["support_agent"]["capabilities"])
    assert any("(not wired on this node)" in c for c in nodes["kb_agent"]["capabilities"])


def test_llm_bundle_loader_summary_counts_skills(llm_bundle_dir):
    b = loader.load_dir(llm_bundle_dir)
    assert b.problems == [] and b.summary()["skills"] == 2 and b.has("mock_strategies")
    assert b.get("mock_strategies")["strategies"]["default"]["tools"]["lookup_order"]["behavior"]
    assert b.agent_map["skills"][0]["name"] == "product-troubleshooting"
    run = next(iter(b.runs.values()))
    assert run["mocking"]["on_miss"] == "llm" and any(m["layer"] == "llm" for cr in run["case_runs"] for m in cr["mock_calls"])
