"""Browser tests for the interactive pipeline flows (pytest-playwright, chromium, headless).

Setup page → discover → config form → YAML validate/save; dataset-only run → review →
feedback → regenerate → approve → evaluation → results; full autonomous run. The pipeline
itself runs offline through the support bot's scripted agent + offline generator, so each
flow completes in seconds and the assertions check real artifacts on disk.

    uv sync --extra ui && uv run playwright install chromium
    uv run pytest -m ui tests/ui/test_playwright_flows.py
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.ui

playwright = pytest.importorskip("playwright.sync_api")
pytest.importorskip("streamlit")
expect = playwright.expect
expect.set_options(timeout=30_000)

APP = Path("src/evalbuilder/ui/app.py").resolve()
EXAMPLE = Path("docs/examples/support-bot").resolve()
SCRIPTED_AGENT = "scripted:examples.support_bot.agent:default_scripted_model"
OFFLINE_GENERATOR = "scripted:examples.support_bot.offline:generator_model"
JOB_TIMEOUT = 180_000
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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return tmp_path_factory.mktemp("flows")


# ── helpers ────────────────────────────────────────────────────


def _settle(page) -> None:
    """Wait for a rerun that may still be starting (the status widget appears with a delay), then for it to finish."""
    status = page.locator("[data-testid='stStatusWidget']")
    try:
        status.wait_for(state="visible", timeout=1_500)
    except playwright.TimeoutError:
        pass
    status.wait_for(state="hidden", timeout=60_000)


def _shot(page, name: str) -> None:
    if SHOTS:
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)


def _fill(page, label: str, text: str, kind: str = "stTextInput") -> None:
    """Type into a Streamlit text input / text area by label, commit with Tab, wait for the rerun."""
    widget = page.get_by_test_id(kind).filter(has_text=label).first
    box = widget.locator("input, textarea").first
    box.fill(text)
    box.press("Tab")
    _settle(page)


def _fill_area(page, label: str, text: str) -> None:
    _fill(page, label, text, kind="stTextArea")


def _select(page, label: str, option: str) -> None:
    widget = page.get_by_test_id("stSelectbox").filter(has_text=label).first
    box = widget.locator("input")
    if option in (box.input_value() or ""):
        return  # already selected; re-clicking a selected react-aria option keeps re-rendering it
    box.click()
    page.locator("[role='option']").filter(has_text=option).first.click()
    _settle(page)


def _click(page, name: str) -> None:
    page.get_by_role("button", name=name).first.click()
    _settle(page)


def _yaml_area(page):
    return page.get_by_test_id("stTextArea").filter(has_text="pipeline config").first.locator("textarea")


def _set_yaml(page, text: str) -> None:
    area = _yaml_area(page)
    area.fill(text)
    area.press("Tab")
    _settle(page)


def _open_setup(page, app_url: str) -> None:
    page.set_viewport_size({"width": 1500, "height": 1100})
    page.goto(app_url)  # Pipeline setup is the default page: it lives at the root URL
    page.locator("h2#setup").wait_for(timeout=60_000)
    _settle(page)
    # every flow starts from a clean project
    sidebar = page.get_by_test_id("stSidebar")
    if sidebar.get_by_role("button", name="Clear project").count():
        sidebar.get_by_role("button", name="Clear project").click()
        _settle(page)
    expect(sidebar.get_by_text("No project selected")).to_be_visible()


def _nav(page, title: str, anchor: str) -> None:
    page.get_by_test_id("stSidebarNav").get_by_role("link", name=title).click()
    page.locator(f"h2#{anchor}").wait_for(timeout=60_000)
    _settle(page)


def _configure(page, work: Path, name: str, *, auto_approve: bool, instructions: str, target: str = "support-bot",
               module: str = "examples.support_bot.agent", agent_model: str = SCRIPTED_AGENT,
               generator_model: str = OFFLINE_GENERATOR,
               constraints: str = "Never call issue_refund before the customer explicitly confirms.\nAlways mention the order id.",
               edge_cases_per_tool: int | None = None) -> Path:
    """Fill the setup form for an example agent with offline models; returns the config path."""
    _select(page, "Example agent", target)
    expect(page.get_by_test_id("stTextInput").filter(has_text="Importable module").locator("input")).to_have_value(module)
    _fill(page, "Pipeline name", name)
    cfg_path = work / f"{name}.yaml"
    _fill(page, "Config file path", str(cfg_path))
    _fill(page, "Agent model", agent_model)
    _fill(page, "Judge model", agent_model)
    _fill(page, "Generator model", generator_model)
    _fill_area(page, "Constraints", constraints)
    _fill_area(page, "General rules", instructions)
    _fill(page, "Output directory", str(work / name))
    if auto_approve:
        page.get_by_text("Auto-approve generated cases").click()
        _fill(page, "Approved by", "playwright")
    _click(page, "Generate YAML")
    page.locator("h2#setup").wait_for()
    expect(_yaml_area(page)).to_have_value(re.compile(r"instructions: " + re.escape(instructions)))
    # drop the judge evaluators (the scripted judge cannot score) by editing the YAML directly
    text = _yaml_area(page).input_value()
    text = text.replace("- type: contract\n", "").replace("- type: correctness\n", "")
    if edge_cases_per_tool is not None:
        text = re.sub(r"per_tool_edge_cases: \d+", f"per_tool_edge_cases: {edge_cases_per_tool}", text)
    _set_yaml(page, text)
    return cfg_path


def _wait_job_finished(page, out_dir: Path, mode: str) -> dict:
    """Wait for the expected job to finish on disk, then for the page to show that job as finished."""
    page.locator("h2#run").wait_for(timeout=60_000)
    deadline = time.time() + JOB_TIMEOUT / 1000
    job: dict = {}
    while time.time() < deadline:
        job_file = out_dir / "job.json"
        if job_file.exists():
            try:
                job = json.loads(job_file.read_text())
            except ValueError:
                job = {}
            if job.get("mode") == mode and job.get("finished_at"):
                break
        time.sleep(0.2)
    else:
        raise AssertionError(f"job {mode} did not finish in time: {job}")
    started = job["started_at"][:19].replace("T", " ")
    expect(page.get_by_text(f"started {started}").first).to_be_visible(timeout=JOB_TIMEOUT)
    expect(page.get_by_text("job finished").first).to_be_visible(timeout=JOB_TIMEOUT)
    _settle(page)
    return job


def _expect_results_verdict(page, verdict: str) -> None:
    expect(page.locator("h3#results")).to_be_visible()
    expect(page.locator("span.stMarkdownBadge").filter(has_text=verdict).first).to_be_visible()
    expect(page.get_by_text("overall score").first).to_be_visible()


# ── flows ──────────────────────────────────────────────────────


def test_setup_discover_generate_validate_and_save(page, app_url, work):
    _open_setup(page, app_url)
    _select(page, "Example agent", "support-bot")
    _click(page, "Discover structure")
    expect(page.get_by_text("Tools and schemas")).to_be_visible()
    metrics = page.locator("[data-testid='stMetric']")
    assert metrics.filter(has_text="Tools").first.inner_text().strip().endswith("4")
    assert int(metrics.filter(has_text="Schema edge cases").first.locator("[data-testid='stMetricValue']").inner_text()) >= 4
    refund_expander = page.get_by_test_id("stExpander").filter(has_text="issue_refund — schemas & edge cases")
    refund_expander.locator("summary").click()
    expect(refund_expander.get_by_text("missing_required").first).to_be_visible()
    expect(refund_expander.get_by_text('"amount"').first).to_be_visible()
    _shot(page, "setup-discover")

    cfg_path = _configure(page, work, "ui-setup", auto_approve=False, instructions="Customers are impatient; keep replies short.")
    _click(page, "Validate")
    expect(page.get_by_text("is valid")).to_be_visible()
    good = _yaml_area(page).input_value()
    assert "- Never call issue_refund before the customer explicitly confirms." in good
    assert "per_tool_edge_cases: 1" in good and "generator: " + OFFLINE_GENERATOR in good
    # a semantic error is reported inline and nothing is saved
    _set_yaml(page, good.replace("overall_pass: 0.8", "overall_pass: 2"))
    _click(page, "Save config")
    expect(page.get_by_text("thresholds.overall_pass must be within [0, 1]")).to_be_visible()
    assert not cfg_path.exists()
    _shot(page, "setup-invalid")
    _set_yaml(page, good)
    _click(page, "Save config")
    expect(page.get_by_text("Saved to")).to_be_visible()
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["instructions"] == "Customers are impatient; keep replies short."
    assert saved["models"]["generator"] == OFFLINE_GENERATOR and saved["coverage"]["per_tool_edge_cases"] == 1
    assert [e["type"] for e in saved["evaluators"]] == ["expected_tools", "contains"]
    _shot(page, "setup-saved")


def test_dataset_only_run_review_feedback_regenerate_and_evaluate(page, app_url, work):
    _open_setup(page, app_url)
    cfg_path = _configure(page, work, "ui-review", auto_approve=False, instructions="Focus on refunds and order tracking.")
    out_dir = work / "ui-review"
    _click(page, "Generate dataset & mocks only")
    _wait_job_finished(page, out_dir, "dataset")
    _shot(page, "run-dataset-done")
    expect(page.get_by_text("stopped after").first).to_be_visible()
    assert (out_dir / "dataset.json").exists() and (out_dir / "mock-rules.json").exists()
    state = json.loads((out_dir / "state.json").read_text())
    assert state["stages"]["dataset"]["status"] == "ok" and state["stages"]["review"]["status"] == "skipped"
    assert state["data"]["stopped_after"] == "dataset"
    dataset = json.loads((out_dir / "dataset.json").read_text())
    assert all(c["review"]["status"] == "pending" for c in dataset["cases"])
    edge_cases = [c for c in dataset["cases"] if c["metadata"].get("edge")]
    assert edge_cases and {c["metadata"]["edge"]["kind"] for c in edge_cases} == {"missing_required"}

    # review section reflects the dataset
    expect(page.locator("h3#review")).to_be_visible()
    review_metrics = page.locator("[data-testid='stMetric']")
    assert review_metrics.filter(has_text="Schema-edge cases").first.locator("[data-testid='stMetricValue']").inner_text() == str(len(edge_cases))
    assert review_metrics.filter(has_text="Pending").first.locator("[data-testid='stMetricValue']").inner_text() == str(len(dataset["cases"]))
    expect(page.get_by_text("Schema edge cases (from tool input/output schemas)")).to_be_visible()

    # feedback → config, regenerate from dataset → new dataset run
    _fill_area(page, "Comment / new instruction", "Use European order ids and add a multi-turn refund case.")
    _select(page, "Regenerate from", "dataset")
    first_updated = state["updated_at"]
    _click(page, "Save feedback & regenerate")
    _wait_job_finished(page, out_dir, "regenerate")
    _shot(page, "run-regenerated")
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["feedback"][0]["note"] == "Use European order ids and add a multi-turn refund case."
    assert saved["feedback"][0]["from_stage"] == "dataset"
    state2 = json.loads((out_dir / "state.json").read_text())
    assert state2["updated_at"] > first_updated and state2["stages"]["dataset"]["status"] == "ok"
    assert state2["stages"]["map"]["status"] == "ok"  # cached, not regenerated
    log = (out_dir / "pipeline.log").read_text()
    assert "--resume --from dataset" in log and "map: cached" in log

    # proceed: reject one case, approve the rest, run the evaluation
    expect(page.locator("h3#proceed")).to_be_visible()
    _fill(page, "Approved by", "playwright reviewer")
    reject_widget = page.get_by_test_id("stMultiSelect").filter(has_text="Reject these cases").first
    tags = reject_widget.get_by_test_id("stMultiSelectTagsContainer")
    for _attempt in range(3):  # the option click can race a pending rerun; verify the tag landed
        reject_widget.locator("input").click(force=True)
        options = page.locator("[role='option']")
        expect(options.first).to_be_visible()
        expect(options.first).to_have_text("Select all")
        options.nth(1).click()  # the first real case
        _settle(page)
        if "case-" in tags.inner_text():
            break
    page.locator("h3#proceed").click()  # close the multiselect menu (it stays open for multi-selection)
    _settle(page)
    expect(tags).to_contain_text("case-")
    _click(page, "Approve remaining cases & run evaluation")
    _wait_job_finished(page, out_dir, "resume")
    _shot(page, "run-evaluated")
    _expect_results_verdict(page, "pass")
    report = json.loads((out_dir / "report.json").read_text())
    assert report["verdict"] == "pass" and report["stages"]["review"]["details"]["approved_by"] == "playwright reviewer"
    final = json.loads((out_dir / "dataset.json").read_text())
    statuses = {c["review"]["status"] for c in final["cases"]}
    assert statuses == {"approved", "rejected"} and sum(c["review"]["status"] == "rejected" for c in final["cases"]) == 1
    assert yaml.safe_load(cfg_path.read_text())["review"] == {"auto_approve": True, "approved_by": "playwright reviewer", "note": "approved in the UI"}

    # open the results in the report pages: the project never changed, so they simply show it
    expect(page.get_by_test_id("stSidebar").get_by_text("ui-review").first).to_be_visible()
    _click(page, "Open results in the report pages")
    page.locator("h2#summary").wait_for(timeout=60_000)
    _settle(page)
    expect(page.get_by_test_id("stSidebar").get_by_text("ui-review").first).to_be_visible()
    expect(page.get_by_text("Overall score")).to_be_visible()
    page.get_by_test_id("stSidebarNav").get_by_role("link", name="Dataset & mocks").click()
    page.locator("h2#dataset").wait_for(timeout=60_000)
    _settle(page)
    expect(page.locator("h3#mock-rules")).to_be_visible()
    _shot(page, "results-dataset")


def test_full_autonomous_run_from_setup(page, app_url, work):
    _open_setup(page, app_url)
    _configure(page, work, "ui-full", auto_approve=True, instructions="Run everything end to end.")
    out_dir = work / "ui-full"
    _click(page, "Run full pipeline")
    _wait_job_finished(page, out_dir, "full")
    _shot(page, "run-full")
    _expect_results_verdict(page, "pass")
    report = json.loads((out_dir / "report.json").read_text())
    assert report["verdict"] == "pass" and report["overall_score"] == 1.0
    assert report["stages"]["review"]["details"]["approved_by"] == "playwright"
    assert report["config"]["instructions"] == "Run everything end to end."
    assert "schema-edge" in report["coverage"]["by_kind"]
    assert json.loads((out_dir / "job.json").read_text())["exit_code"] == 0
    _click(page, "Open results in the report pages")
    page.locator("h2#summary").wait_for(timeout=60_000)
    _settle(page)
    expect(page.get_by_text("Overall score")).to_be_visible()
    page.get_by_test_id("stSidebarNav").get_by_role("link", name="Coverage").click()
    page.locator("h2#coverage").wait_for(timeout=60_000)
    _settle(page)
    expect(page.get_by_text("schema-edge").first).to_be_visible()


def test_pydantic_agent_dataset_run_shows_schema_edge_cases(page, app_url, work):
    """incident_desk: pydantic in/out models + args_schema → discovery preview, richer edge kinds,
    pipeline-injected malformed-output fixtures, then approval and evaluation."""
    _open_setup(page, app_url)
    _select(page, "Example agent", "incident-desk")
    _click(page, "Discover structure")
    metrics = page.locator("[data-testid='stMetric']")
    assert int(metrics.filter(has_text="Pydantic models").first.locator("[data-testid='stMetricValue']").inner_text()) >= 4
    assert int(metrics.filter(has_text="Schema edge cases").first.locator("[data-testid='stMetricValue']").inner_text()) >= 10
    runbooks = page.get_by_test_id("stExpander").filter(has_text="search_runbooks — schemas & edge cases")
    runbooks.locator("summary").click()
    expect(runbooks.get_by_text("out_of_enum").first).to_be_visible()
    expect(runbooks.get_by_text('"sev1"').first).to_be_visible()
    ticket = page.get_by_test_id("stExpander").filter(has_text="create_ticket — schemas & edge cases")
    ticket.locator("summary").click()
    expect(ticket.get_by_text("malformed_output").first).to_be_visible()
    expect(ticket.get_by_text('"TicketRequest"').first).to_be_visible()
    _shot(page, "setup-incident-discover")

    _configure(
        page, work, "ui-incident", auto_approve=False, instructions="Ops on-call copilot; sev1 means outage.",
        target="incident-desk", module="examples.incident_desk.agent",
        agent_model="scripted:examples.incident_desk.agent:default_scripted_model",
        generator_model="scripted:examples.incident_desk.offline:generator_model",
        constraints="Never page on-call before the user explicitly confirms.", edge_cases_per_tool=3,
    )
    out_dir = work / "ui-incident"
    _click(page, "Generate dataset & mocks only")
    _wait_job_finished(page, out_dir, "dataset")
    _shot(page, "run-incident-dataset")
    dataset = json.loads((out_dir / "dataset.json").read_text())
    edges = [c for c in dataset["cases"] if c["metadata"].get("edge")]
    kinds = {c["metadata"]["edge"]["kind"] for c in edges}
    assert {"missing_required", "wrong_type", "out_of_enum", "malformed_output"} <= kinds, kinds
    malformed = [c for c in edges if c["metadata"]["edge"]["kind"] == "malformed_output"]
    assert malformed and all((c["metadata"].get("mocks") or {}).get("tools") for c in malformed)
    override = next(iter(malformed[0]["metadata"]["mocks"]["tools"].values()))[0]["response"]
    amap = json.loads((out_dir / "agent-map.json").read_text())
    tool = next(t for t in amap["tools"] if t["name"] == malformed[0]["metadata"]["tool"])
    assert set(tool["output_schema"]["required"]) - set(override)  # a required field was dropped
    mocks = json.loads((out_dir / "mock-rules.json").read_text())["tools"]
    assert set(mocks) == {t["name"] for t in amap["tools"]}
    review_metrics = page.locator("[data-testid='stMetric']")
    assert review_metrics.filter(has_text="Schema-edge cases").first.locator("[data-testid='stMetricValue']").inner_text() == str(len(edges))

    _fill(page, "Approved by", "playwright ops")
    _click(page, "Approve remaining cases & run evaluation")
    _wait_job_finished(page, out_dir, "resume")
    _expect_results_verdict(page, "pass")
    report = json.loads((out_dir / "report.json").read_text())
    assert report["verdict"] == "pass" and report["coverage"]["by_kind"]["schema-edge"]["covered"] == len(edges)
    _click(page, "Open results in the report pages")
    page.locator("h2#summary").wait_for(timeout=60_000)
    _settle(page)
    page.get_by_test_id("stSidebarNav").get_by_role("link", name="Agent graph & tools").click()
    page.locator("h2#agent").wait_for(timeout=60_000)
    _settle(page)
    expander = page.get_by_test_id("stExpander").filter(has_text="create_ticket — argument schema")
    expander.locator("summary").click()
    expect(expander.get_by_text("Output schema").first).to_be_visible()
    expect(expander.get_by_text('"TicketReceipt"').first).to_be_visible()
    _shot(page, "results-incident-agent")


# ── project selection, persistence, consistency ────────────────


def test_setup_state_persists_across_pages_and_clear_empties_everything(page, app_url, work):
    _open_setup(page, app_url)
    sidebar = page.get_by_test_id("stSidebar")
    # no project: every page is empty and points back to the setup page
    for title in ("Run & review", "Summary", "Dataset & mocks"):
        page.get_by_test_id("stSidebarNav").get_by_role("link", name=title).click()
        _settle(page)
        expect(page.get_by_text("No project selected").first).to_be_visible()
    _nav(page, "Pipeline setup", "setup")
    # pick a target: its output folder becomes the project everywhere
    _select(page, "Example agent", "weather-bot")
    expect(page.get_by_test_id("stTextInput").filter(has_text="Output directory").locator("input")).to_have_value("eval/pipeline/weather-bot")
    expect(sidebar.get_by_text("weather-bot").first).to_be_visible()
    out = work / "ui-persist"
    _fill(page, "Output directory", str(out))
    expect(sidebar.get_by_text(str(out)).first).to_be_visible()
    _click(page, "Discover structure")
    expect(page.get_by_text("Tools and schemas")).to_be_visible()
    _fill_area(page, "Constraints", "Always name the city.")
    _fill_area(page, "General rules", "Travellers ask short questions.")
    _fill(page, "Agent model", "scripted:examples.weather_bot.agent:default_scripted_model")
    _click(page, "Generate YAML")
    expect(_yaml_area(page)).to_have_value(re.compile("instructions: Travellers ask short questions."))
    edited = _yaml_area(page).input_value() + "# edited in the browser\n"
    _set_yaml(page, edited)
    _shot(page, "setup-before-leaving")
    # leave: the run page follows the same folder, the summary explains there is nothing yet
    _nav(page, "Run & review", "run")
    expect(page.get_by_text("Nothing has run in").first).to_be_visible()
    expect(page.get_by_text(str(out)).first).to_be_visible()
    page.get_by_test_id("stSidebarNav").get_by_role("link", name="Summary").click()
    _settle(page)
    expect(page.get_by_text("No artifacts yet").first).to_be_visible()
    # come back: every field, the preview and the edited YAML survived
    _nav(page, "Pipeline setup", "setup")
    expect(page.get_by_test_id("stSelectbox").filter(has_text="Example agent").locator("input")).to_have_value(re.compile("weather-bot"))
    expect(page.get_by_test_id("stTextArea").filter(has_text="Constraints").first.locator("textarea")).to_have_value("Always name the city.")
    expect(page.get_by_test_id("stTextArea").filter(has_text="General rules").first.locator("textarea")).to_have_value("Travellers ask short questions.")
    expect(page.get_by_test_id("stTextInput").filter(has_text="Agent model").locator("input")).to_have_value("scripted:examples.weather_bot.agent:default_scripted_model")
    expect(page.get_by_test_id("stTextInput").filter(has_text="Output directory").locator("input")).to_have_value(str(out))
    expect(_yaml_area(page)).to_have_value(edited)
    expect(page.get_by_text("Tools and schemas")).to_be_visible()
    _shot(page, "setup-after-returning")
    # clear from the sidebar: the form is empty and every page is empty again
    sidebar.get_by_role("button", name="Clear project").click()
    _settle(page)
    expect(sidebar.get_by_text("No project selected")).to_be_visible()
    expect(page.get_by_test_id("stTextArea").filter(has_text="Constraints").first.locator("textarea")).to_have_value("")
    expect(_yaml_area(page)).to_have_value("")
    expect(page.get_by_text("Tools and schemas")).to_have_count(0)
    _nav(page, "Run & review", "run")
    expect(page.get_by_text("No project selected").first).to_be_visible()


def test_opening_an_existing_project_loads_config_and_results(page, app_url):
    _open_setup(page, app_url)
    _select(page, "Project folder", "support-bot (docs/examples/support-bot)")
    expect(page.get_by_text("Opened").first).to_be_visible()
    expect(page.get_by_test_id("stTextInput").filter(has_text="Pipeline name").locator("input")).to_have_value("support-bot")
    expect(page.get_by_test_id("stTextInput").filter(has_text="Config file path").locator("input")).to_have_value("examples/support_bot/pipeline.yaml")
    expect(_yaml_area(page)).to_have_value(re.compile("name: support-bot"))
    expect(page.get_by_test_id("stSidebar").get_by_text("artifact kinds")).to_be_visible()
    _nav(page, "Summary", "summary")
    expect(page.get_by_text("Overall score")).to_be_visible()
    _nav(page, "Run & review", "run")
    expect(page.locator("h3#results")).to_be_visible()
    _nav(page, "Coverage", "coverage")
    expect(page.locator("h3#coverage-by-kind")).to_be_visible()
    # new project… resets everything
    _nav(page, "Pipeline setup", "setup")
    _select(page, "Project folder", "new project…")
    expect(page.get_by_test_id("stSidebar").get_by_text("No project selected")).to_be_visible()
    expect(_yaml_area(page)).to_have_value("")


# ── the smallest agent, end to end in the browser ──────────────


def test_weather_bot_small_dataset_review_and_evaluate(page, app_url, work):
    """weather_bot (one node, two tools): dataset-only run → review → approve → evaluation →
    every report page, with a 4-case dataset, all through the browser."""
    _open_setup(page, app_url)
    _select(page, "Example agent", "weather-bot")
    _click(page, "Discover structure")
    metrics = page.locator("[data-testid='stMetric']")
    assert metrics.filter(has_text="Tools").first.inner_text().strip().endswith("2")
    _configure(
        page, work, "ui-weather", auto_approve=False, instructions="Travellers ask short questions.",
        target="weather-bot", module="examples.weather_bot.agent",
        agent_model="scripted:examples.weather_bot.agent:default_scripted_model",
        generator_model="scripted:examples.weather_bot.offline:generator_model",
        constraints="Always name the city in the answer.", edge_cases_per_tool=1,
    )
    text = re.sub(r"total_cases: \d+", "total_cases: 4", _yaml_area(page).input_value())
    text = re.sub(r"repeats: \d+", "repeats: 1", text)
    _set_yaml(page, text)
    out_dir = work / "ui-weather"
    _click(page, "Generate dataset & mocks only")
    _wait_job_finished(page, out_dir, "dataset")
    expect(page.get_by_test_id("stSidebar").get_by_text("ui-weather").first).to_be_visible()
    dataset = json.loads((out_dir / "dataset.json").read_text())
    assert 4 <= len(dataset["cases"]) <= 9 and all(c["review"]["status"] == "pending" for c in dataset["cases"])
    edges = [c for c in dataset["cases"] if c["metadata"].get("edge")]
    assert {c["metadata"]["tool"] for c in edges} == {"get_weather", "get_alerts"}
    review_metrics = page.locator("[data-testid='stMetric']")
    assert review_metrics.filter(has_text="Mocked tools").first.locator("[data-testid='stMetricValue']").inner_text() == "2"
    _shot(page, "run-weather-dataset")
    _fill(page, "Approved by", "playwright weather")
    _click(page, "Approve remaining cases & run evaluation")
    _wait_job_finished(page, out_dir, "resume")
    _expect_results_verdict(page, "pass")
    report = json.loads((out_dir / "report.json").read_text())
    assert report["verdict"] == "pass" and report["agent"]["tools"] == ["get_weather", "get_alerts"]
    assert report["stages"]["review"]["details"]["approved_by"] == "playwright weather"
    _click(page, "Open results in the report pages")
    page.locator("h2#summary").wait_for(timeout=60_000)
    _settle(page)
    expect(page.get_by_text("Overall score")).to_be_visible()
    for title, anchor in (("Agent graph & tools", "agent"), ("Intents & scenarios", "intents"), ("Dataset & mocks", "dataset"),
                          ("Coverage", "coverage"), ("Eval results", "results"), ("Stability", "stability"),
                          ("Simulation", "simulation"), ("Analysis", "analysis"), ("Stages & problems", "stages")):
        _nav(page, title, anchor)
        expect(page.get_by_test_id("stSidebar").get_by_text("ui-weather").first).to_be_visible()
    _shot(page, "results-weather-stages")
