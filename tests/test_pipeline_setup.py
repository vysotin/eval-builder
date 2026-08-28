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
