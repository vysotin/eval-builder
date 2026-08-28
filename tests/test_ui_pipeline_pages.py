"""Setup and Run & review pages rendered headlessly with AppTest (no browser, no jobs)."""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from evalbuilder.pipeline import jobs  # noqa: E402


def _page_script(page):
    import importlib

    importlib.import_module(f"evalbuilder.ui.app_pages.{page}").render()


def _run(page: str, state: dict | None = None) -> AppTest:
    at = AppTest.from_function(_page_script, kwargs={"page": page}, default_timeout=120)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    return at


def _errors(at: AppTest) -> list[str]:
    return [re.sub(r"\x1b\[[0-9;]*m", "", e.value) for e in at.exception]


def test_setup_page_discovers_target_and_generates_yaml():
    at = _run("setup")
    assert _errors(at) == []
    assert [h.value for h in at.header] == ["Pipeline setup"]
    assert at.selectbox(key="setup_target").value == "examples/support_bot/agent.py" or at.selectbox(key="setup_target").value.startswith("examples/")
    at.selectbox(key="setup_target").select("examples/support_bot/agent.py").run()
    assert at.text_input(key="setup_module").value == "examples.support_bot.agent"
    at.button(key="setup_discover").click().run()
    assert _errors(at) == []
    labels = [m.label for m in at.metric]
    assert "Tools" in labels and "Schema edge cases" in labels
    assert next(m for m in at.metric if m.label == "Tools").value == "4"
    at.text_area(key="setup_constraints").set_value("Never refund without a yes.\nMention the order id.")
    at.text_area(key="setup_instructions").set_value("Customers are impatient; keep replies short.")
    at.text_input(key="setup_agent_model").set_value("scripted:examples.support_bot.agent:default_scripted_model")
    at.button(key="FormSubmitter:setup_form-Generate YAML").click().run()
    assert _errors(at) == []
    yaml_text = at.text_area(key="setup_yaml").value
    assert "instructions: Customers are impatient; keep replies short." in yaml_text
    assert "- Never refund without a yes." in yaml_text and "agent: scripted:examples.support_bot.agent:default_scripted_model" in yaml_text
    assert "per_tool_edge_cases: 1" in yaml_text and "judge: claude-cli:claude-sonnet-5" in yaml_text
    at.button(key="setup_validate").click().run()
    assert any("is valid" in s.value for s in at.success)
    at.text_area(key="setup_yaml").set_value(yaml_text.replace("overall_pass: 0.8", "overall_pass: 2"))
    at.button(key="setup_validate").click().run()
    assert any("thresholds.overall_pass must be within [0, 1]" in e.value for e in at.error)


def test_setup_page_saves_config_to_path(tmp_path):
    at = _run("setup")
    at.selectbox(key="setup_target").select("examples/support_bot/agent.py").run()
    at.text_input(key="setup_config_path").set_value(str(tmp_path / "cfg.yaml"))
    at.button(key="FormSubmitter:setup_form-Generate YAML").click().run()
    at.button(key="setup_save").click().run()
    assert _errors(at) == []
    assert (tmp_path / "cfg.yaml").exists() and any("Saved to" in i.value for i in at.info)


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
    from evalbuilder.pipeline import setup as s

    s.build_config(s.default_form({"name": "support-bot", "source": "examples/support_bot/agent.py", "module": "examples.support_bot.agent"})).save(tmp_path / "cfg.yaml")
    return out


def test_run_page_without_dirs_explains():
    at = _run("run", {"job_dir": None})
    assert _errors(at) == []
    assert [h.value for h in at.header] == ["Run & review"]


def test_run_page_shows_review_section_after_dataset_stop(tmp_path):
    out = _stopped_dir(tmp_path)
    at = _run("run", {"job_dir": str(out)})
    assert _errors(at) == []
    subheaders = [h.value for h in at.subheader]
    assert "Job status" in subheaders and "Review & iterate" in subheaders and "Proceed with evaluation" in subheaders
    assert any("job finished" in m.value and "stopped after **dataset**" in m.value for m in at.markdown)
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
