"""Interactive setup helpers (pure Python): target discovery/preview, config building,
feedback / approval / rejection between runs."""

import json
from pathlib import Path

import pytest

from evalbuilder.pipeline import setup as s
from evalbuilder.pipeline.config import DEFAULT_MODEL, load_config


def test_discover_targets_finds_example_agents():
    targets = {t["name"]: t for t in s.discover_targets()}
    assert "support-bot" in targets
    sb = targets["support-bot"]
    assert sb["source"] == "examples/support_bot/agent.py" and sb["module"] == "examples.support_bot.agent"
    assert sb["scripted_model"] == "scripted:examples.support_bot.agent:default_scripted_model"
    assert sb["pipeline_yaml"] == "examples/support_bot/pipeline.yaml"
    assert s.module_for(Path("examples/weather_bot/agent.py")) == "examples.weather_bot.agent"
    assert s.discover_targets(("nope",)) == []


def test_preview_target_reports_structure_without_llm():
    preview = s.preview_target("examples/support_bot/agent.py", "examples.support_bot.agent")
    assert preview["ok"] and preview["problems"] == []
    assert [sk["name"] for sk in preview["skills"]] == ["product-troubleshooting", "refund-policy"] and preview["skills_dir"].endswith("support_bot/skills")
    loader = next(t for t in preview["tools"] if t["name"] == "load_skill")
    assert loader["kind"] == "skill_loader" and loader["mockable"] is False and loader["live"]
    assert s.preview_target("examples/weather_bot/agent.py", "examples.weather_bot.agent")["skills"] == []
    assert {n["id"] for n in preview["nodes"]} >= {"classify", "support_agent", "kb_agent", "decline"}
    tools = {t["name"]: t for t in preview["tools"]}
    assert tools["issue_refund"]["side_effecting"] and tools["issue_refund"]["live"]
    assert tools["lookup_order"]["schema_source"] == "annotations" and tools["lookup_order"]["args_schema"]["required"] == ["order_id"]
    assert preview["edge_case_count"] >= 4 and preview["live"].get("nodes")
    assert preview["models"] == ["Route"]
    missing = s.preview_target("nowhere/agent.py", "nowhere.agent")
    assert not missing["ok"] and "source not found" in missing["problems"][0]


def test_build_config_and_validate_text(tmp_path):
    form = s.default_form({"name": "demo", "source": "examples/support_bot/agent.py", "module": "examples.support_bot.agent"})
    assert form["judge_model"] == DEFAULT_MODEL and form["config_path"] == "eval/pipeline/demo.yaml"
    form.update(constraints="Never guess.\n\nAlways cite.", instructions="Be terse.", agent_model="", evaluators=["expected_tools"],
                auto_approve=True, approved_by="me", per_tool_edge_cases=3)
    cfg = s.build_config(form)
    assert cfg.constraints == ["Never guess.", "Always cite."] and cfg.instructions == "Be terse."
    assert cfg.models.agent is None and cfg.coverage.per_tool_edge_cases == 3 and cfg.evaluators == [{"type": "expected_tools"}]
    assert cfg.problems() == []
    text = cfg.to_yaml()
    parsed, problems = s.validate_text(text)
    assert parsed == cfg and problems == []
    _, problems = s.validate_text(text.replace("default: 0.8", "default: 1.8"))
    assert problems == ["thresholds.default must be within [0, 1]"]
    none, problems = s.validate_text("name: only\n")
    assert none is None and any("target" in p for p in problems)


def test_feedback_approval_and_rejection_round_trip(tmp_path):
    form = s.default_form({"name": "demo", "source": "examples/support_bot/agent.py", "module": "examples.support_bot.agent"})
    path = s.build_config(form).save(tmp_path / "demo.yaml")
    cfg = s.add_feedback(path, "more refunds", "map")
    assert cfg.feedback[0].note == "more refunds" and load_config(path).feedback[0].from_stage == "map"
    s.add_feedback(path, "and fewer poems")
    assert len(load_config(path).feedback) == 2
    with pytest.raises(ValueError):
        s.approve_in_config(path, "  ")
    cfg = s.approve_in_config(path, "vlad", note="looked fine")
    assert cfg.review.auto_approve and load_config(path).review.approved_by == "vlad" and cfg.review.note == "looked fine"

    out = tmp_path / "out"
    out.mkdir()
    ds = json.loads(Path("docs/examples/support-bot/dataset.json").read_text())
    (out / "dataset.json").write_text(json.dumps(ds))
    already = sum(1 for c in ds["cases"] if c["review"]["status"] == "rejected")
    ids = [c["id"] for c in ds["cases"][:2]]
    assert s.reject_cases(out, ids, "dup", "vlad") == 2 and s.reject_cases(out, [], "x", "y") == 0
    after = json.loads((out / "dataset.json").read_text())
    assert [c["review"]["status"] for c in after["cases"][:2]] == ["rejected", "rejected"]
    assert after["cases"][0]["review"]["note"] == "rejected in UI by vlad: dup"
    summary = s.dataset_summary(out)
    assert summary["cases"] == len(ds["cases"]) and summary["by_status"]["rejected"] == 2 + already
    assert summary["mocked_tools"] == ["check_refund_policy", "issue_refund", "lookup_order", "search_kb"]
    assert s.dataset_summary(tmp_path / "empty") is None


# ── projects (folder = output directory; the UI's single source of truth) ──


def test_default_form_without_target_selects_nothing():
    form = s.default_form()
    assert form["source"] == "" and form["module"] == "" and form["output_dir"] == "" and form["config_path"] == ""
    with_target = s.default_form({"name": "weather-bot", "source": "examples/weather_bot/agent.py", "module": "examples.weather_bot.agent"})
    assert with_target["output_dir"] == "eval/pipeline/weather-bot" and with_target["config_path"] == "eval/pipeline/weather-bot.yaml"


def test_form_from_config_round_trips_build_config():
    form = s.default_form({"name": "demo", "source": "examples/support_bot/agent.py", "module": "examples.support_bot.agent"})
    form.update(constraints="Never guess.\nAlways cite.", instructions="Be terse.", agent_model="", evaluators=["expected_tools", "contains"],
                auto_approve=True, approved_by="me", per_tool_edge_cases=3, repeats=1, simulate=False, on_miss="llm",
                mock_model="openai:gpt-5", on_invalid="strict", strategies=False,
                total_cases=5, multi_turn_share=0.25, threshold_default=0.6)
    cfg = s.build_config(form)
    back = s.form_from_config(cfg, config_path="eval/pipeline/demo.yaml")
    assert back == {**form, "output_dir": "eval/pipeline/demo"}
    assert s.build_config(back) == cfg
    assert cfg.mocking.on_miss == "llm" and cfg.models.mock == "openai:gpt-5" and cfg.mocking.on_invalid == "strict" and not cfg.mocking.strategies
    assert s.default_form()["mock_model"] == "" and s.default_form()["on_invalid"] == "fallback" and s.default_form()["strategies"] is True
    # a folder overrides the config's output dir (the project is the folder)
    assert s.form_from_config(cfg, output_dir="docs/examples/demo")["output_dir"] == "docs/examples/demo"
    # unknown evaluators (with options) are kept out of the multiselect but survive in the YAML round trip only
    cfg.evaluators.append({"type": "openevals", "rubric": "x"})
    assert s.form_from_config(cfg)["evaluators"] == ["expected_tools", "contains"]


def test_project_config_prefers_job_then_report_then_sibling(tmp_path):
    out = tmp_path / "eval" / "pipeline" / "demo"
    out.mkdir(parents=True)
    assert s.project_config(out) == (None, None)
    form = s.default_form({"name": "demo", "source": "examples/support_bot/agent.py", "module": "examples.support_bot.agent"})
    cfg = s.build_config(form)
    # 3. a sibling <dir>.yaml
    sibling = tmp_path / "eval" / "pipeline" / "demo.yaml"
    cfg.save(sibling)
    found, path = s.project_config(out)
    assert found == cfg and path == str(sibling)
    # 2. report.json: config_path when it exists, else the embedded config
    other = tmp_path / "other.yaml"
    (out / "report.json").write_text(json.dumps({"schema": "evalbuilder/pipeline-report/v1", "name": "demo",
                                                 "config_path": str(other), "config": {**cfg.model_dump(by_alias=True), "instructions": "embedded"}}))
    found, path = s.project_config(out)
    assert found.instructions == "embedded" and path is None
    cfg.model_copy(update={"instructions": "from other"}).save(other)
    found, path = s.project_config(out)
    assert found.instructions == "from other" and path == str(other)
    # 1. job.json wins
    job_cfg = tmp_path / "job.yaml"
    cfg.model_copy(update={"instructions": "from job"}).save(job_cfg)
    (out / "job.json").write_text(json.dumps({"schema": "evalbuilder/pipeline-job/v1", "config_path": str(job_cfg)}))
    found, path = s.project_config(out)
    assert found.instructions == "from job" and path == str(job_cfg)
    # a broken config file is reported as None rather than raising
    job_cfg.write_text("name: only\n")
    assert s.project_config(out) == (None, None) or s.project_config(out)[0].instructions == "from other"


def test_discover_projects_lists_output_dirs_and_unrun_configs(tmp_path):
    root = tmp_path / "eval" / "pipeline"
    root.mkdir(parents=True)
    form = s.default_form({"name": "ran", "source": "examples/support_bot/agent.py", "module": "examples.support_bot.agent"})
    ran = s.build_config({**form, "output_dir": str(root / "ran")})
    ran.save(root / "ran.yaml")
    (root / "ran").mkdir()
    (root / "ran" / "state.json").write_text(json.dumps({"schema": "evalbuilder/pipeline-state/v1", "name": "ran", "stages": {}}))
    unrun = s.build_config({**form, "name": "unrun", "output_dir": str(root / "unrun")})
    unrun.save(root / "unrun.yaml")
    (root / "noise").mkdir()  # no artifacts, no config → not a project
    (root / "broken.yaml").write_text("nope: 1\n")
    projects = s.discover_projects(("eval/pipeline",), cwd=tmp_path)
    by_name = {p["name"]: p for p in projects}
    assert set(by_name) == {"ran", "unrun"}
    assert by_name["ran"]["dir"] == str(root / "ran") and by_name["ran"]["has_artifacts"] and by_name["ran"]["config_path"] == str(root / "ran.yaml")
    assert by_name["unrun"]["dir"] == str(root / "unrun") and not by_name["unrun"]["has_artifacts"] and by_name["unrun"]["config_path"] == str(root / "unrun.yaml")
    # committed examples are projects too
    real = {p["dir"]: p for p in s.discover_projects()}
    assert "docs/examples/support-bot" in real and real["docs/examples/support-bot"]["has_artifacts"]
    assert real["docs/examples/support-bot"]["name"] == "support-bot"
