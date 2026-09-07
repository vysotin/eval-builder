import pytest
import yaml

from evalbuilder.pipeline.config import (
    DEFAULT_MODEL,
    PIPELINE_CONFIG_SCHEMA,
    STAGE_NAMES,
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


def test_runs_keeps_only_repeats_and_parallelism_moved_to_phases():
    cfg = PipelineConfig.model_validate(MINIMAL)
    assert cfg.runs.repeats == 3
    assert cfg.inference.workers == 4 and cfg.inference.backend == "threads" and cfg.inference.timeout == 120
    assert cfg.evaluation.workers == 4 and cfg.evaluation.backend == "threads"
    cfg.inference.workers = 0
    cfg.evaluation.workers = 0
    problems = cfg.problems()
    assert any("inference.workers must be >= 1" in p for p in problems)
    assert any("evaluation.workers must be >= 1" in p for p in problems)
    with pytest.raises(Exception):
        PipelineConfig.model_validate({**MINIMAL, "inference": {"backend": "gpu"}})


def test_old_parallel_keys_migrate_into_the_phase_sections():
    cfg = PipelineConfig.model_validate({**MINIMAL, "runs": {"repeats": 2, "parallel_intents": 6, "parallel_scoring": 3, "parallel_simulations": 5}})
    assert cfg.runs.repeats == 2
    assert cfg.inference.workers == 6 and cfg.evaluation.workers == 3
    assert any("runs.parallel_intents" in note for note in cfg.migrated)
    assert any("runs.parallel_simulations" in note for note in cfg.migrated)
    assert "migrated" not in cfg.to_yaml() and "parallel_intents" not in cfg.to_yaml()
    # an explicit new section wins over the legacy key
    cfg2 = PipelineConfig.model_validate({**MINIMAL, "runs": {"parallel_scoring": 9}, "evaluation": {"workers": 2}})
    assert cfg2.evaluation.workers == 2 and cfg2.migrated == ["runs.parallel_scoring ignored: evaluation.workers is set"]


# ── deployment ─────────────────────────────────────────────────


def test_deploy_defaults_and_image_ref():
    cfg = PipelineConfig.model_validate(MINIMAL)
    d = cfg.deploy
    assert d.target == "local" and not cfg.container_target
    assert (d.port, d.namespace, d.replicas, d.expose, d.keep, d.timeout, d.context) == (8080, "default", 1, "port-forward", False, 240, None)
    assert d.image.name is None and d.image.tag == "latest" and d.image.registry is None and d.image.push == "auto" and d.image.extras == []
    assert d.build.context == "." and d.build.dockerfile is None and d.build.include == [] and d.build.requirements is None
    assert cfg.image_ref == "evalbuilder-support-bot:latest"
    docker = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "docker", "image": {"name": "acme/bot", "tag": "v2"}}})
    assert docker.container_target and docker.image_ref == "acme/bot:v2"
    reg = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "kubernetes", "image": {"registry": "quay.io/team"}}})
    assert reg.image_ref == "quay.io/team/evalbuilder-support-bot:latest"
    with pytest.raises(Exception):
        PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "heroku"}})


def test_deploy_problems():
    bad = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "kubernetes", "port": 70000, "replicas": 0, "timeout": 0}})
    problems = "\n".join(bad.problems())
    for needle in ("deploy.port", "deploy.replicas", "deploy.timeout"):
        assert needle in problems
    ocp = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "openshift"}})
    assert any("image.registry" in p for p in ocp.problems())
    push = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "kubernetes", "image": {"push": "registry"}}})
    assert any("image.registry" in p for p in push.problems())
    route = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "kubernetes", "expose": "route"}})
    assert any("expose: route" in p for p in route.problems())
    cli = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "docker"}, "models": {"agent": "claude-cli:sonnet"}})
    assert any("claude-cli" in p and "container" in p for p in cli.problems())
    ok = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "docker"}, "models": {"agent": "scripted:examples.support_bot.agent:default_scripted_model"}})
    assert ok.problems() == []


def test_deploy_host_port_default_round_trip_and_problems():
    cfg = PipelineConfig.model_validate(MINIMAL)
    assert cfg.deploy.host_port is None  # null = the container port
    docker = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "docker", "host_port": 9090}})
    assert docker.deploy.host_port == 9090 and docker.deploy.port == 8080 and docker.problems() == []
    yaml_text = docker.to_yaml()
    assert "host_port: 9090" in yaml_text
    assert load_config_text(yaml_text).deploy == docker.deploy
    out_of_range = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "docker", "host_port": 70000}})
    assert any("deploy.host_port must be within [1, 65535]" in p for p in out_of_range.problems())
    other_target = PipelineConfig.model_validate({**MINIMAL, "deploy": {"target": "kubernetes", "host_port": 9090}})
    assert any("deploy.host_port is only meaningful for the docker target" in p for p in other_target.problems())
    text = template("demo", "examples/weather_bot/agent.py", "examples.weather_bot.agent")
    assert "host_port: null" in text and load_config_text(text).deploy.host_port is None


def test_template_spells_out_deploy_inference_evaluation():
    text = template("demo", "examples/weather_bot/agent.py", "examples.weather_bot.agent")
    for needle in ("deploy:", "target: local", "| docker (compose)", "inference:", "workers: 4", "backend: threads", "evaluation:", "expose: port-forward"):
        assert needle in text, needle
    assert "parallel_scoring" not in text
    cfg = load_config_text(text)
    assert cfg.deploy.target == "local" and cfg.inference.workers == 4 and cfg.problems() == []
    docker = load_config_text(text.replace("target: local ", "target: docker").replace(f"agent: {DEFAULT_MODEL}", "agent: null"))
    assert docker.deploy.target == "docker" and docker.problems() == []
    back = load_config_text(cfg.to_yaml())
    assert back.deploy == cfg.deploy and back.inference == cfg.inference and back.evaluation == cfg.evaluation
    assert STAGE_NAMES.index("deploy") < STAGE_NAMES.index("infer") < STAGE_NAMES.index("simulate") < STAGE_NAMES.index("teardown") < STAGE_NAMES.index("score")
    assert "run" not in STAGE_NAMES


# ── two-layer mocking config ───────────────────────────────────


def test_mocking_and_mock_model_defaults_and_problems():
    cfg = PipelineConfig.model_validate(MINIMAL)
    m = cfg.mocking
    assert (m.on_miss, m.strategies, m.strategy, m.on_invalid, m.max_repairs) == ("strict", True, "default", "fallback", 1)
    assert cfg.models.mock is None and cfg.mock_model_spec == cfg.models.generator
    llm = PipelineConfig.model_validate({**MINIMAL, "mocking": {"on_miss": "llm"}, "models": {"mock": "openai:gpt-5"}})
    assert llm.mock_model_spec == "openai:gpt-5" and llm.problems() == []
    bad = PipelineConfig.model_validate({**MINIMAL, "mocking": {"on_miss": "llm", "max_repairs": -1, "strategy": ""}, "models": {"mock": "gpt"}})
    problems = bad.problems()
    assert any("models.mock must look like provider:model" in p for p in problems)
    assert any("mocking.max_repairs" in p for p in problems) and any("mocking.strategy" in p for p in problems)
    with pytest.raises(Exception):
        PipelineConfig.model_validate({**MINIMAL, "mocking": {"on_miss": "guess"}})
    text = template("x", "examples/support_bot/agent.py", "examples.support_bot.agent")
    assert "mock:" in text and "on_invalid" in text and "strategies:" in text and "on_miss: strict" in text
    back = load_config_text(text)
    assert back.mocking.on_invalid == "fallback" and back.models.mock is None
    round_trip = load_config_text(llm.to_yaml())
    assert round_trip.mocking.on_miss == "llm" and round_trip.models.mock == "openai:gpt-5"


def load_config_text(text: str) -> PipelineConfig:
    from evalbuilder.pipeline.config import parse_config

    return parse_config(text, "test")
