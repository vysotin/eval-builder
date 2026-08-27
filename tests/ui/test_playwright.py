"""Browser tests for the Streamlit report UI (pytest-playwright, chromium, headless).

A session fixture starts `streamlit run` on a free port against the committed
support-bot example artifacts; each test navigates the sidebar and asserts what the
page shows. Skipped when playwright or its chromium build is missing:

    uv sync --extra ui && uv run playwright install chromium
    uv run pytest -m ui
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui

playwright = pytest.importorskip("playwright.sync_api")
pytest.importorskip("streamlit")
expect = playwright.expect
expect.set_options(timeout=30_000)

EXAMPLE = Path("docs/examples/support-bot").resolve()
APP = Path("src/evalbuilder/ui/app.py").resolve()
SHOTS = Path(os.environ.get("EVALBUILDER_UI_SHOTS", "")) if os.environ.get("EVALBUILDER_UI_SHOTS") else None


def _chromium_available() -> bool:
    try:
        with playwright.sync_playwright() as p:
            return Path(p.chromium.executable_path).exists()
    except Exception:  # noqa: BLE001
        return False


if not _chromium_available():
    pytest.skip("playwright chromium not installed (uv run playwright install chromium)", allow_module_level=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def app_url():
    port = _free_port()
    env = {**os.environ, "EVALBUILDER_UI_DIR": str(EXAMPLE), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"streamlit exited early:\n{proc.stdout.read()}")
        try:
            if urllib.request.urlopen(f"{url}/_stcore/health", timeout=1).read() == b"ok":
                break
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    else:
        proc.kill()
        raise RuntimeError("streamlit did not become healthy in 60s")
    yield url
    proc.kill()
    proc.wait(timeout=10)


@pytest.fixture
def report_page(page, app_url):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(app_url)
    page.locator("h2#overview").wait_for(timeout=60_000)
    page.locator("[data-testid='stStatusWidget']").wait_for(state="hidden", timeout=60_000)
    return page


def _expand(page, text: str) -> None:
    """Open a Streamlit expander (rendered as <details><summary>) by its label text."""
    page.get_by_test_id("stExpander").filter(has_text=text).first.locator("summary").click()


def _open(page, title: str, anchor: str) -> None:
    page.get_by_test_id("stSidebarNav").get_by_role("link", name=title).click()
    page.locator(f"h2#{anchor}").wait_for(timeout=60_000)
    page.locator("[data-testid='stStatusWidget']").wait_for(state="hidden", timeout=60_000)
    if SHOTS:
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / f"{anchor}.png"), full_page=True)


def test_overview_shows_verdict_headline_and_charts(report_page):
    page = report_page
    if SHOTS:
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / "overview.png"), full_page=True)
    expect(page.get_by_test_id("stSidebar").get_by_text("support-bot").first).to_be_visible()
    expect(page.locator("h2#overview")).to_be_visible()
    assert page.locator("[data-testid='stMetric']").count() >= 6
    expect(page.get_by_text("Overall score")).to_be_visible()
    _expand(page, "verdict reasons")
    expect(page.get_by_text("metric contains 0.5 < 0.7")).to_be_visible()
    expect(page.locator("h3#metric-pass-rates")).to_be_visible()
    assert page.locator("[data-testid='stVegaLiteChart']").count() >= 2
    assert page.locator("[data-testid='stDataFrame']").count() >= 2


def test_agent_page_renders_graph_and_tools(report_page):
    page = report_page
    _open(page, "Agent graph & tools", "agent")
    expect(page.locator("[data-testid='stGraphVizChart']").first).to_be_visible()
    expect(page.locator("h3#agent-tools")).to_be_visible()
    _expand(page, "lookup_order — argument schema")
    expect(page.get_by_text('"order_id"').first).to_be_visible()
    page.get_by_test_id("stButtonGroup").get_by_text("live (compiled)").click()
    page.locator("[data-testid='stStatusWidget']").wait_for(state="hidden", timeout=60_000)
    expect(page.locator("[data-testid='stGraphVizChart']").first).to_be_visible()


def test_intents_page_lists_intents_and_scenarios(report_page):
    page = report_page
    _open(page, "Intents & scenarios", "intents")
    expect(page.locator("h3#scenarios")).to_be_visible()
    expect(page.get_by_text("intent.order-status").first).to_be_visible()
    assert page.locator("[data-testid='stExpander']").count() >= 8


def test_dataset_page_filters_and_case_detail(report_page):
    page = report_page
    _open(page, "Dataset & mocks", "dataset")
    expect(page.get_by_text("21", exact=True).first).to_be_visible()
    expect(page.locator("h3#case-detail")).to_be_visible()
    assert page.locator("[data-testid='stChatMessage']").count() >= 1
    expect(page.locator("h3#mock-rules")).to_be_visible()
    _expand(page, "lookup_order — 4 rule(s)")
    assert page.locator("[data-testid='stDataFrame']").count() >= 3


def test_coverage_page_shows_plan_and_gaps(report_page):
    page = report_page
    _open(page, "Coverage", "coverage")
    expect(page.locator("h3#coverage-by-kind")).to_be_visible()
    expect(page.locator("h3#coverage-gaps")).to_be_visible()
    assert page.locator("[data-testid='stVegaLiteChart']").count() >= 1


def test_results_page_metrics_slices_and_drilldown(report_page):
    page = report_page
    _open(page, "Eval results", "results")
    expect(page.locator("h3#aggregate-metrics")).to_be_visible()
    expect(page.locator("h3#slices")).to_be_visible()
    for tab in ("intent", "failure_mode", "variant"):
        assert page.get_by_role("tab", name=tab).is_visible()
    page.get_by_role("tab", name="failure_mode").click()
    expect(page.locator("h3#failing-cases")).to_be_visible()
    expect(page.locator("h3#case-drilldown")).to_be_visible()
    assert page.get_by_role("tab", name="run 093d879e").is_visible()
    page.get_by_role("tab", name="run 8a6166d8").click()
    expect(page.get_by_text("tool calls:").first).to_be_visible()


def test_stability_simulation_analysis_pages(report_page):
    page = report_page
    _open(page, "Stability", "stability")
    expect(page.locator("h3#unstable-cases")).to_be_visible()
    _open(page, "Simulation", "simulation")
    expect(page.locator("h3#simulation-outcomes")).to_be_visible()
    assert page.locator("[data-testid='stChatMessage']").count() >= 2
    _open(page, "Analysis", "analysis")
    expect(page.locator("h3#failure-patterns")).to_be_visible()
    expect(page.locator("h3#recommendations")).to_be_visible()


def test_stages_page_timeline_and_artifact_table(report_page):
    page = report_page
    _open(page, "Stages & problems", "stages")
    expect(page.locator("h3#stage-timeline")).to_be_visible()
    expect(page.locator("h3#artifacts-loaded")).to_be_visible()
    _expand(page, "score — ok")
    expect(page.get_by_text("score-report-093d879e.json").first).to_be_visible()


def test_upload_mode_identifies_artifacts_by_schema(report_page):
    page = report_page
    page.get_by_test_id("stSidebar").get_by_text("uploaded files").click()
    files = [str(p) for p in sorted(EXAMPLE.glob("*.json"))] + [str(p) for p in sorted((EXAMPLE / "results").glob("*.json"))]
    page.locator("input[type='file']").set_input_files(files)
    page.locator("[data-testid='stStatusWidget']").wait_for(state="hidden", timeout=60_000)
    page.locator("h2#overview").wait_for(timeout=60_000)
    sidebar = page.get_by_test_id("stSidebar")
    sidebar.get_by_text("artifact kinds").wait_for(timeout=60_000)
    assert "support-bot" in sidebar.inner_text()
    expect(page.get_by_text("Overall score")).to_be_visible()
