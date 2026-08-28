import pytest
import yaml

from evalbuilder.pipeline.config import (
    PIPELINE_CONFIG_SCHEMA,
    PipelineConfig,
    load_config,
    template,
)

MINIMAL = {
    "schema": PIPELINE_CONFIG_SCHEMA,
    "name": "support-bot",
    "target": {"source": "examples/support_bot/agent.py", "module": "examples.support_bot.agent"},
}


def test_minimal_config_gets_defaults():
    cfg = PipelineConfig.model_validate(MINIMAL)
    assert cfg.models.judge.startswith("claude-cli:")
    assert cfg.coverage.per_intent.happy == 2
    assert cfg.runs.repeats == 3
    assert cfg.mocking.on_miss == "strict" and cfg.mocking.required
    assert [e["type"] for e in cfg.evaluators] == ["expected_tools", "contains", "contract", "correctness"]
    assert str(cfg.output_dir) == "eval/pipeline/support-bot"
    assert cfg.thresholds.for_metric("contains") == 0.8
    assert cfg.problems() == []


def test_problems_flag_semantic_issues():
    cfg = PipelineConfig.model_validate(
        {
            **MINIMAL,
            "target": {"source": "nope.py", "module": "x"},
            "runs": {"repeats": 0},
            "stages": {"skip": ["report", "bogus"]},
            "thresholds": {"default": 1.5},
            "review": {"auto_approve": True},
            "models": {"judge": "sonnet"},
        }
    )
    problems = "\n".join(cfg.problems())
    for needle in ("target.source", "repeats", "bogus", "report", "thresholds.default", "approved_by", "models.judge"):
        assert needle in problems


def test_unknown_keys_rejected():
    with pytest.raises(Exception):
        PipelineConfig.model_validate({**MINIMAL, "coverge": {}})


def test_load_yaml_and_json(tmp_path):
    y = tmp_path / "p.yaml"
    y.write_text(yaml.safe_dump({**MINIMAL, "constraints": ["Never guess"]}))
    assert load_config(y).constraints == ["Never guess"]
    j = tmp_path / "p.json"
    j.write_text('{"schema": "%s", "name": "n", "target": {"source": "a.py", "module": "a"}}' % PIPELINE_CONFIG_SCHEMA)
    assert load_config(j).name == "n"


def test_load_reports_readable_errors(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\n")
    with pytest.raises(ValueError, match="target"):
        load_config(p)


def test_template_is_loadable(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(template("demo", "examples/weather_bot/agent.py", "examples.weather_bot.agent"))
    cfg = load_config(p)
    assert cfg.name == "demo" and cfg.problems() == []


def test_new_fields_defaults_and_yaml_round_trip(tmp_path):
    from evalbuilder.pipeline.config import DEFAULT_MODEL, PipelineConfig, load_config, parse_config

    cfg = PipelineConfig(name="x", target={"source": "examples/weather_bot/agent.py", "module": "examples.weather_bot.agent"})
    assert cfg.models.judge == DEFAULT_MODEL == "claude-cli:claude-sonnet-5"
    assert cfg.coverage.per_tool_edge_cases == 2 and cfg.instructions == "" and cfg.feedback == []
    assert cfg.guidance() == ""
    cfg.instructions = "Focus on refunds.\nNever invent order ids."
    entry = cfg.add_feedback("More multi-turn cases please", from_stage="dataset")
    assert entry.at and cfg.feedback[0].note == "More multi-turn cases please"
    text = cfg.guidance()
    assert text.startswith("USER INSTRUCTIONS") and "REVIEWER FEEDBACK" in text and "(rerun from dataset)" in text
    yaml_text = cfg.to_yaml()
    assert yaml_text.startswith("schema: evalbuilder/pipeline-config/v1\nname: x\n") and "# free-text general rules" in yaml_text
    assert parse_config(yaml_text) == cfg
    path = cfg.save(tmp_path / "p.yaml")
    assert load_config(path) == cfg
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_config("")
    with pytest.raises(ValueError, match="thresholds.default"):
        parse_config(yaml_text.replace("default: 0.8", "default: nope"))
    cfg.coverage.per_tool_edge_cases = -1
    assert "coverage.per_tool_edge_cases must be >= 0" in cfg.problems()
