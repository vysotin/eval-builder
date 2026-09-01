"""Setup and Run & review pages rendered headlessly with AppTest (no browser, no jobs).

One AppTest renders whichever page `session_state["page"]` names, after the same
per-run bookkeeping the app does (`project.begin_run`), so switching the page between
runs is exactly what navigating the sidebar does to session state — including
Streamlit dropping the values of widgets that were not rendered."""

import importlib
import json
import re
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from evalbuilder.pipeline import jobs  # noqa: E402
from evalbuilder.pipeline import setup as setup_mod  # noqa: E402

SUPPORT = "examples/support_bot/agent.py"
EXAMPLE_DIR = "docs/examples/support-bot"

# a fake page whose render() is the sidebar's "Clear project" callback
_probe = types.ModuleType("evalbuilder.ui.app_pages.clear_probe")
_probe.render = lambda: importlib.import_module("evalbuilder.ui.project").clear()
sys.modules.setdefault("evalbuilder.ui.app_pages.clear_probe", _probe)


def _script():
    import importlib

    import streamlit as st

    from evalbuilder.ui import project

    project.begin_run()
    importlib.import_module(f"evalbuilder.ui.app_pages.{st.session_state['page']}").render()


def _app(page: str = "setup", **state) -> AppTest:
    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["page"] = page
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _goto(at: AppTest, page: str) -> AppTest:
    at.session_state["page"] = page
    return at.run()


def _errors(at: AppTest) -> list[str]:
    return [re.sub(r"\x1b\[[0-9;]*m", "", e.value) for e in at.exception]


def _texts(elements) -> list[str]:
    return [e.value for e in elements]


# ── project selection drives every page ─────────────────────────


def test_setup_starts_without_a_project_and_other_pages_are_empty():
    at = _app("setup")
    assert _errors(at) == []
    assert _texts(at.header) == ["Pipeline setup"]
    assert at.session_state["project"] is None
    assert at.selectbox(key="setup_target").value is None and at.selectbox(key="setup_project").value is None
    assert at.text_input(key="setup_output_dir").value == "" and at.text_area(key="setup_yaml").value == ""
    assert any("No project yet" in i.value for i in at.info)
    for page in ("run", "summary", "dataset"):
        _goto(at, page)
        assert _errors(at) == [] and _texts(at.header) == [] if page != "run" else _texts(at.header) == ["Run & review"]
        assert any("No project selected" in i.value for i in at.info), page


def test_choosing_a_target_selects_its_output_folder_everywhere(tmp_path):
    at = _app("setup")
    at.selectbox(key="setup_target").select(SUPPORT).run()
    assert _errors(at) == []
    assert at.text_input(key="setup_module").value == "examples.support_bot.agent"
    assert at.text_input(key="setup_output_dir").value == "eval/pipeline/support-bot"
    assert at.session_state["project"] == {"mode": "dir", "dir": "eval/pipeline/support-bot"}
    # the form is the master: a different output directory moves the project
    out = str(tmp_path / "fresh")
    at.text_input(key="setup_output_dir").set_value(out).run()
    assert at.session_state["project"] == {"mode": "dir", "dir": out}
    assert at.selectbox(key="setup_project").value == out
    _goto(at, "run")
    assert _texts(at.header) == ["Run & review"]
    assert any("Nothing has run in" in i.value and out in i.value for i in at.info)
    assert any(f"`{out}`" in c.value for c in at.caption)
    _goto(at, "summary")
    assert _texts(at.header) == [] and any("No artifacts yet" in i.value and out in i.value for i in at.info)


def test_opening_an_existing_folder_loads_its_config_and_results():
    at = _app("setup")
    at.selectbox(key="setup_project").select(EXAMPLE_DIR).run()
    assert _errors(at) == []
    assert at.session_state["project"] == {"mode": "dir", "dir": EXAMPLE_DIR}
    assert at.text_input(key="setup_name").value == "support-bot"
    assert at.selectbox(key="setup_target").value == SUPPORT
    assert at.text_input(key="setup_output_dir").value == EXAMPLE_DIR
    assert at.text_input(key="setup_config_path").value == "examples/support_bot/pipeline.yaml"
    assert "Never call issue_refund" in at.text_area(key="setup_constraints").value
    assert "name: support-bot" in at.text_area(key="setup_yaml").value
    assert any("Opened **support-bot**" in i.value for i in at.info)
    _goto(at, "summary")
    assert _texts(at.header) == ["Summary"] and any(m.label == "Overall score" for m in at.metric)
    _goto(at, "run")
    assert "Results" in _texts(at.subheader) and "Job status" in _texts(at.subheader)
    _goto(at, "dataset")
    assert _texts(at.header) == ["Dataset & mocks"]


def test_open_any_path_button_and_new_project_option(tmp_path):
    out = tmp_path / "eval" / "pipeline" / "mine"
    out.mkdir(parents=True)
    form = setup_mod.default_form({"name": "mine", "source": SUPPORT, "module": "examples.support_bot.agent"})
    setup_mod.build_config({**form, "output_dir": str(out), "instructions": "from the sibling config"}).save(out.parent / "mine.yaml")
    at = _app("setup")
    at.text_input(key="setup_project_path").set_value(str(out))
    at.button(key="setup_open_path").click().run()
    assert _errors(at) == []
    assert at.session_state["project"] == {"mode": "dir", "dir": str(out)}
    assert at.text_area(key="setup_instructions").value == "from the sibling config"
    assert at.text_input(key="setup_config_path").value == str(out.parent / "mine.yaml")
    # a folder without any config: empty form pointed at it
    bare = tmp_path / "bare"
    bare.mkdir()
    at.text_input(key="setup_project_path").set_value(str(bare))
    at.button(key="setup_open_path").click().run()
    assert at.session_state["project"] == {"mode": "dir", "dir": str(bare)}
    assert at.selectbox(key="setup_target").value is None and any("no pipeline config found" in i.value for i in at.info)
    # "new project…" clears everything
    at.selectbox(key="setup_project").select("new project…").run()
    assert at.session_state["project"] is None and at.text_input(key="setup_output_dir").value == ""


def _clear(at: AppTest) -> AppTest:
    """Press the sidebar's Clear project button (its callback runs inside the app script)."""
    page = at.session_state["page"]
    at.session_state["page"] = "clear_probe"
    at.run()
    at.session_state["page"] = page
    return at.run()


def test_clear_project_resets_every_page():
    at = _app("setup")
    at.selectbox(key="setup_project").select(EXAMPLE_DIR).run()
    at.text_area(key="setup_instructions").set_value("keep me?").run()
    _goto(at, "summary")
    assert _texts(at.header) == ["Summary"]
    _goto(at, "setup")
    _clear(at)
    assert _errors(at) == []
    assert at.session_state["project"] is None and at.session_state["bundle"] is None
    assert at.text_area(key="setup_instructions").value == "" and at.selectbox(key="setup_target").value is None
    assert at.selectbox(key="setup_project").value is None and at.text_area(key="setup_yaml").value == ""
    assert not any("Opened" in i.value for i in at.info)
    _goto(at, "summary")
    assert _texts(at.header) == [] and any("No project selected" in i.value for i in at.info)
    _goto(at, "run")
    assert any("No project selected" in i.value for i in at.info)


def test_clear_drops_every_widget_copy_of_the_form():
    at = _app("setup")
    at.selectbox(key="setup_target").select(SUPPORT).run()
    at.text_area(key="setup_constraints").set_value("Never guess.").run()
    assert at.session_state["project"] is not None and at.session_state["setup_form"]["constraints"] == "Never guess."
    _clear(at)
    assert _errors(at) == []
    assert at.session_state["setup_form"]["constraints"] == "" and at.session_state["setup_form"]["output_dir"] == ""
    assert at.text_area(key="setup_constraints").value == "" and at.text_input(key="setup_output_dir").value == ""


# ── persistence across navigation ───────────────────────────────


def test_setup_page_keeps_its_state_across_navigation(tmp_path):
    at = _app("setup")
    at.selectbox(key="setup_target").select(SUPPORT).run()
    at.button(key="setup_discover").click().run()
    assert _errors(at) == []
    assert next(m for m in at.metric if m.label == "Tools").value == "5"  # four API tools + load_skill
    at.text_area(key="setup_constraints").set_value("Never refund without a yes.\nMention the order id.").run()
    at.text_area(key="setup_instructions").set_value("Customers are impatient; keep replies short.").run()
    at.text_input(key="setup_agent_model").set_value("scripted:examples.support_bot.agent:default_scripted_model").run()
    at.number_input(key="setup_total_cases").set_value(7).run()
    at.multiselect(key="setup_evaluators").unselect("correctness").run()
    at.checkbox(key="setup_simulate").uncheck().run()
    at.text_input(key="setup_config_path").set_value(str(tmp_path / "cfg.yaml")).run()
    at.button(key="setup_generate").click().run()
    assert _errors(at) == []
    yaml_text = at.text_area(key="setup_yaml").value
    assert "instructions: Customers are impatient; keep replies short." in yaml_text
    assert "total_cases: 7" in yaml_text and "simulate: false" in yaml_text and "- type: correctness" not in yaml_text
    edited = yaml_text.replace("total_cases: 7", "total_cases: 9")
    at.text_area(key="setup_yaml").set_value(edited).run()

    # leave for two other pages and come back: every widget, the preview and the YAML survive
    _goto(at, "summary")
    _goto(at, "run")
    assert _texts(at.header) == ["Run & review"]
    _goto(at, "setup")
    assert _errors(at) == []
    assert at.selectbox(key="setup_target").value == SUPPORT
    assert at.text_area(key="setup_constraints").value == "Never refund without a yes.\nMention the order id."
    assert at.text_area(key="setup_instructions").value == "Customers are impatient; keep replies short."
    assert at.text_input(key="setup_agent_model").value == "scripted:examples.support_bot.agent:default_scripted_model"
    assert at.number_input(key="setup_total_cases").value == 7 and not at.checkbox(key="setup_simulate").value
    assert "correctness" not in at.multiselect(key="setup_evaluators").value
    assert at.text_input(key="setup_config_path").value == str(tmp_path / "cfg.yaml")
    assert at.text_area(key="setup_yaml").value == edited
    assert next(m for m in at.metric if m.label == "Tools").value == "5"  # four API tools + load_skill  # discovery preview kept
    # the edited YAML is what gets validated / saved
    at.button(key="setup_validate").click().run()
    assert any("is valid" in s.value for s in at.success)
    at.button(key="setup_save").click().run()
    assert _errors(at) == [] and any("Saved to" in i.value for i in at.info)
    assert "total_cases: 9" in (tmp_path / "cfg.yaml").read_text()
    # an invalid YAML is reported inline and nothing is written
    at.text_area(key="setup_yaml").set_value(edited.replace("overall_pass: 0.8", "overall_pass: 2")).run()
    at.button(key="setup_save").click().run()
    assert any("thresholds.overall_pass must be within [0, 1]" in e.value for e in at.error)
    assert "total_cases: 9" in (tmp_path / "cfg.yaml").read_text()


def test_yaml_output_dir_moves_the_project(tmp_path):
    at = _app("setup")
    at.selectbox(key="setup_target").select(SUPPORT).run()
    at.text_input(key="setup_config_path").set_value(str(tmp_path / "cfg.yaml")).run()
    at.button(key="setup_generate").click().run()
    text = at.text_area(key="setup_yaml").value
    moved = str(tmp_path / "moved")
    at.text_area(key="setup_yaml").set_value(text.replace("dir: eval/pipeline/support-bot", f"dir: {moved}")).run()
    at.button(key="setup_validate").click().run()
    assert _errors(at) == []
    assert at.session_state["project"] == {"mode": "dir", "dir": moved}
    assert at.text_input(key="setup_output_dir").value == moved
    assert any(f"`{moved}`" in s.value for s in at.success)


def test_generate_yaml_reports_invalid_form_inline():
    at = _app("setup")
    at.selectbox(key="setup_target").select("custom target…").run()
    assert at.session_state["project"] is None and at.text_input(key="setup_source").value == ""
    at.button(key="setup_generate").click().run()
    assert _errors(at) == [] and any("Config invalid" in e.value and "source is required" in e.value for e in at.error)
    assert at.session_state["project"] is None
    # typing the target by hand works like picking an example
    at.text_input(key="setup_source").set_value("examples/weather_bot/agent.py").run()
    at.text_input(key="setup_module").set_value("examples.weather_bot.agent").run()
    at.text_input(key="setup_name").set_value("weather-custom").run()
    at.button(key="setup_generate").click().run()
    assert _errors(at) == [] and not at.error
    assert at.session_state["project"] == {"mode": "dir", "dir": "eval/pipeline/weather-custom"}
    assert at.text_input(key="setup_output_dir").value == "eval/pipeline/weather-custom"
    assert "module: examples.weather_bot.agent" in at.text_area(key="setup_yaml").value


# ── Run & review with a stopped-after-dataset folder ───────────


def _stopped_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    ds = json.loads(Path("docs/examples/support-bot/dataset.json").read_text())
    for c in ds["cases"]:
        c["review"] = {"status": "pending", "note": ""}
    ds["cases"][0]["metadata"]["edge"] = {"id": "edge.lookup_order.missing_required.order_id", "kind": "missing_required", "field": "order_id"}
    ds["cases"][0]["metadata"]["tool"] = "lookup_order"
    (out / "dataset.json").write_text(json.dumps(ds))
    (out / "state.json").write_text(json.dumps({
        "schema": "evalbuilder/pipeline-state/v1", "name": "support-bot",
        "stages": {"preflight": {"status": "ok", "seconds": 0.1}, "discover": {"status": "ok", "seconds": 0.2},
                   "map": {"status": "ok", "seconds": 3.0}, "mocks": {"status": "ok", "seconds": 2.0},
                   "dataset": {"status": "ok", "seconds": 5.0}, "review": {"status": "skipped", "reason": "stopped after dataset (--until)"},
                   "report": {"status": "ok", "seconds": 0.1}},
        "data": {"stopped_after": "dataset"},
    }))
    (out / "job.json").write_text(json.dumps({
        "schema": jobs.JOB_SCHEMA, "mode": "dataset", "config_path": str(tmp_path / "cfg.yaml"), "output_dir": str(out),
        "argv": ["x"], "log": str(out / "pipeline.log"), "pid": 1, "started_at": "2026-08-28T10:00:00", "finished_at": "2026-08-28T10:01:00", "exit_code": 0,
    }))
    (out / "pipeline.log").write_text("dataset: ok (5.0s)\n")
    form = setup_mod.default_form({"name": "support-bot", "source": SUPPORT, "module": "examples.support_bot.agent"})
    setup_mod.build_config({**form, "output_dir": str(out)}).save(tmp_path / "cfg.yaml")
    return out


def test_run_page_shows_review_section_after_dataset_stop(tmp_path):
    out = _stopped_dir(tmp_path)
    at = _app("run", project={"mode": "dir", "dir": str(out)})
    assert _errors(at) == []
    subheaders = _texts(at.subheader)
    assert "Job status" in subheaders and "Review & iterate" in subheaders and "Proceed with evaluation" in subheaders
    assert any("job finished" in m.value and "stopped after **dataset**" in m.value for m in at.markdown)
    assert any(f"`{tmp_path / 'cfg.yaml'}`" in c.value for c in at.caption)  # config resolved through job.json
    labels = {m.label: m.value for m in at.metric}
    assert labels["Cases"] == "21" and labels["Schema-edge cases"] == "1" and labels["Pending"] == "21"
    # feedback is appended to the config file
    at.text_area(key="review_feedback").set_value("Add more multi-turn refund cases.")
    at.selectbox(key="review_from").select("map")
    at.button(key="review_save").click().run()
    assert _errors(at) == []
    from evalbuilder.pipeline.config import load_config

    cfg = load_config(tmp_path / "cfg.yaml")
    assert cfg.feedback[0].note == "Add more multi-turn refund cases." and cfg.feedback[0].from_stage == "map"
    assert any("Feedback saved" in s.value for s in at.success)
    # proceeding needs an approver
    at.button(key="proceed_run").click().run()
    assert any("approver" in e.value for e in at.error)
    # the setup page shows the same project, loaded from the job's config — and picks up
    # the feedback the review page just appended to that file
    _goto(at, "setup")
    assert at.text_input(key="setup_output_dir").value == str(out) and at.selectbox(key="setup_target").value == SUPPORT
    assert at.text_input(key="setup_config_path").value == str(tmp_path / "cfg.yaml")
    assert "Add more multi-turn refund cases." in at.text_area(key="setup_yaml").value
    assert any("reloaded from" in i.value for i in at.info)


def test_setup_reloads_config_changed_on_disk_but_keeps_local_edits_otherwise(tmp_path):
    at = _app("setup")
    at.selectbox(key="setup_target").select(SUPPORT).run()
    cfg_path = tmp_path / "cfg.yaml"
    at.text_input(key="setup_config_path").set_value(str(cfg_path)).run()
    at.button(key="setup_generate").click().run()
    at.button(key="setup_save").click().run()
    assert cfg_path.exists()
    # local YAML edits survive navigation while the file is untouched
    at.text_area(key="setup_yaml").set_value(at.text_area(key="setup_yaml").value + "# local edit\n").run()
    _goto(at, "run")
    _goto(at, "setup")
    assert at.text_area(key="setup_yaml").value.endswith("# local edit\n")
    # a change on disk (as the review page makes) is reloaded
    setup_mod.add_feedback(cfg_path, "from disk", "map")
    _goto(at, "run")
    _goto(at, "setup")
    assert "note: from disk" in at.text_area(key="setup_yaml").value and not at.text_area(key="setup_yaml").value.endswith("# local edit\n")
    assert any("reloaded from" in i.value for i in at.info)


# ── job progress, partial results and stop ──────────────────────


def _running_job(tmp_path, sleep=60):
    """A real background job (sleeping) plus a state/run-progress snapshot mid-run."""
    out = tmp_path / "out"
    jobs.start_job(tmp_path / "cfg.yaml", out, mode="full",
                   argv=[sys.executable, "-c", f"import time; time.sleep({sleep})"])
    (out / "state.json").write_text(json.dumps({
        "schema": "evalbuilder/pipeline-state/v1", "name": "x",
        "stages": {"preflight": {"status": "ok"}, "discover": {"status": "ok"},
                   "verify": {"status": "ok"}, "run": {"status": "running"},
                   "score": {"status": "pending"}, "report": {"status": "pending"}},
        "data": {},
    }))
    (out / "run-progress.json").write_text(json.dumps({
        "schema": "evalbuilder/run-progress/v1", "repeat": 2, "repeats": 3,
        "cases_total": 4, "cases_done": 1, "overall_total": 12, "overall_done": 5,
        "by_intent": {"intent.orders": {"done": 1, "total": 2, "errors": 1},
                      "intent.refunds": {"done": 0, "total": 2, "errors": 0}},
        "runs": [], "current": [{"case_id": "case-1", "intent": "intent.orders",
                                 "error_class": "agent", "error": "boom"}],
    }))
    deadline = __import__("time").time() + 20
    while __import__("time").time() < deadline:
        if jobs.job_status(out)["status"] == "running":
            return out
        __import__("time").sleep(0.1)
    raise AssertionError("job did not start")


def test_run_page_shows_progress_bar_stage_and_partial_results_while_running(tmp_path):
    out = _running_job(tmp_path)
    try:
        at = _app("run", project={"mode": "dir", "dir": str(out)})
        assert _errors(at) == []
        bars = at.get("progress")
        assert len(bars) >= 2  # stage-level + case-level bars
        texts = " ".join(str(getattr(b, "text", "")) for b in bars)
        assert "run" in texts and "repeat 2/3" in texts and "5/12" in texts
        assert any("running **run**" in m.value for m in at.markdown)
        # partial per-intent results are on the page while the job runs
        page = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        assert "intent.orders" in page or any("intent.orders" in str(d.value) for d in at.dataframe)
        assert at.button(key="run_stop") is not None
    finally:
        jobs.stop_job(out)


def test_run_page_stop_button_stops_the_job(tmp_path):
    out = _running_job(tmp_path)
    try:
        at = _app("run", project={"mode": "dir", "dir": str(out)})
        at.button(key="run_stop").click().run()
        assert _errors(at) == []
        assert jobs.job_status(out)["status"] == "stopped"
        assert any("job stopped" in m.value for m in at.markdown)
        assert any("interrupted" in m.value and "run" in m.value for m in at.markdown)
    finally:
        jobs.stop_job(out)


# a fake page whose render() is the shared job-progress widget (what the sidebar shows)
_jobbox = types.ModuleType("evalbuilder.ui.app_pages.jobbox_probe")


def _jobbox_render():
    from pathlib import Path as _P

    import streamlit as st

    from evalbuilder.pipeline import jobs as _jobs
    from evalbuilder.ui.common import job_progress

    job_progress(_jobs.job_status(_P(st.session_state["jobbox_dir"])), compact=True)


_jobbox.render = _jobbox_render
sys.modules.setdefault("evalbuilder.ui.app_pages.jobbox_probe", _jobbox)


def test_sidebar_job_widget_shows_badge_bar_and_stage(tmp_path):
    out = _running_job(tmp_path)
    try:
        at = _app("jobbox_probe", jobbox_dir=str(out))
        assert _errors(at) == []
        assert any("job running" in m.value for m in at.markdown)
        bars = at.get("progress")
        assert bars and "run" in str(getattr(bars[0], "text", ""))
    finally:
        jobs.stop_job(out)



def test_setup_form_carries_the_mock_layer_and_skills_into_the_yaml(tmp_path):
    at = _app("setup")
    at.selectbox(key="setup_target").select(SUPPORT).run()
    at.button(key="setup_discover").click().run()
    assert _errors(at) == []
    assert next(m for m in at.metric if m.label == "Skills").value == "2"
    text = "\n".join(m.value for m in at.markdown)
    assert "Agent skills" in text
    at.selectbox(key="setup_on_miss").select("llm").run()
    at.selectbox(key="setup_on_invalid").select("strict").run()
    at.text_input(key="setup_mock_model").set_value("scripted:examples.support_bot.offline:mock_model").run()
    at.checkbox(key="setup_strategies").uncheck().run()
    at.text_input(key="setup_config_path").set_value(str(tmp_path / "cfg.yaml")).run()
    at.button(key="setup_generate").click().run()
    yaml_text = at.text_area(key="setup_yaml").value
    assert "on_miss: llm" in yaml_text and "on_invalid: strict" in yaml_text and "strategies: false" in yaml_text
    assert "mock: scripted:examples.support_bot.offline:mock_model" in yaml_text
    at.button(key="setup_validate").click().run()
    assert any("is valid" in s.value for s in at.success)
    # the form survives navigation and comes back from the YAML
    _goto(at, "run")
    _goto(at, "setup")
    assert at.selectbox(key="setup_on_miss").value == "llm" and at.selectbox(key="setup_on_invalid").value == "strict"
    assert at.text_input(key="setup_mock_model").value == "scripted:examples.support_bot.offline:mock_model"
    assert not at.checkbox(key="setup_strategies").value
