"""Report-UI pages rendered headlessly with streamlit's AppTest (no browser)."""

import re

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from evalbuilder.ui import loader  # noqa: E402

PAGES = ("overview", "agent", "intents", "dataset", "coverage", "results", "stability", "simulation", "analysis", "stages")
EXPECTED_HEADERS = {
    "overview": "Overview", "agent": "Agent", "intents": "Intents & scenarios", "dataset": "Dataset & mocks",
    "coverage": "Coverage", "results": "Eval results", "stability": "Stability", "simulation": "Simulation",
    "analysis": "Analysis", "stages": "Stages & problems",
}


def _page_script(page):
    import importlib

    importlib.import_module(f"evalbuilder.ui.app_pages.{page}").render()


def _run(page: str, bundle) -> AppTest:
    at = AppTest.from_function(_page_script, kwargs={"page": page}, default_timeout=120)
    at.session_state["bundle"] = bundle
    at.run()
    return at


def _errors(at: AppTest) -> list[str]:
    return [re.sub(r"\x1b\[[0-9;]*m", "", e.value) for e in at.exception]


@pytest.fixture(scope="module")
def example():
    return loader.load_dir("docs/examples/support-bot")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page, example):
    at = _run(page, example)
    assert _errors(at) == []
    assert [h.value for h in at.header] == [EXPECTED_HEADERS[page]]
    assert len(at.subheader) >= 3
    assert not at.warning, [w.value for w in at.warning]


@pytest.mark.parametrize("page", PAGES)
def test_page_explains_missing_artifacts(page):
    at = _run(page, loader.Bundle(name="empty", source="empty"))
    assert _errors(at) == []
    if page not in ("overview", "results", "stages"):
        assert at.warning, "pages needing an artifact must say which one is missing"
        assert "written by stage" in at.warning[0].value


def test_overview_shows_verdict_and_metrics(example):
    at = _run("overview", example)
    labels = [m.label for m in at.metric]
    assert "Overall score" in labels and "Coverage" in labels
    assert at.metric[0].value == "0.59"
    assert any("metric contains 0.5 < 0.7" in m.value for m in at.markdown)


def test_dataset_page_filters_and_details(example):
    at = _run("dataset", example)
    assert at.metric[0].value == "21"
    at.multiselect(key="ds_failure").select("tool_error_handling").run()
    assert _errors(at) == []
    assert any(re.match(r"\d+ of 21 cases", c.value) for c in at.caption)
    assert at.selectbox(key="ds_case").value.startswith("case-")


def test_results_page_drilldown_switches_case(example):
    at = _run("results", example)
    box = at.selectbox(key="results_case")
    first = box.value
    box.select_index(1).run()
    assert _errors(at) == []
    assert at.selectbox(key="results_case").value != first
    assert len(at.tabs) >= 2  # slices + per-run tabs


def test_agent_graph_dot_covers_nodes_and_tools(example):
    from evalbuilder.ui.app_pages.agent import graph_dot

    amap = example.agent_map
    dot = graph_dot(amap["graph"], amap["tools"], live=False, show_tools=True)
    for node in ("classify", "support_agent", "kb_agent", "decline"):
        assert f'"{node}"' in dot
    assert '"classify" -> "support_agent" [style=dashed' in dot
    assert '"tool:lookup_order"' in dot and "digraph agent" in dot
    live = graph_dot(amap["graph"], amap["tools"], live=True, show_tools=False)
    assert "START -> \"classify\"" in live and "tool:" not in live


INCIDENT_EXAMPLE = "docs/examples/incident-desk"


@pytest.fixture(scope="module")
def incident():
    return loader.load_dir(INCIDENT_EXAMPLE)


def test_incident_desk_example_loads_with_schema_data(incident):
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
    assert incident.verdict in ("pass", "fail")


@pytest.mark.parametrize("page", PAGES)
def test_incident_desk_pages_render(page, incident):
    at = _run(page, incident)
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
