import json

from typer.testing import CliRunner

from evalbuilder.cli import app
from evalbuilder.config import Settings, capability_check


def test_settings_load_from_env_file(tmp_path, monkeypatch):
    for var in ("LANGSMITH_API_KEY", "EVALBUILDER_JUDGE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text("LANGSMITH_API_KEY=key123\nEVALBUILDER_JUDGE_MODEL=openai:gpt-test\n")
    s = Settings.load(env)
    assert s.langsmith_api_key == "key123"
    assert s.judge_model == "openai:gpt-test"


def test_capability_check_degraded_without_langsmith(monkeypatch):
    for var in ("LANGSMITH_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.load(env_file=None)
    report = capability_check(s)
    assert report["ready"] is True  # local-only is still ready
    assert "langsmith" in report["degraded"]
    assert "judge" in report["degraded"]


def test_capability_check_blocking_on_bad_module():
    s = Settings.load(env_file=None)
    report = capability_check(s, target_module="no.such.module")
    assert report["ready"] is False
    assert report["blocking"] and "fix" in report["blocking"][0]


def test_cli_check_outputs_json(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["check"])
    assert result.exit_code == 0
    assert "ready" in json.loads(result.stdout)


def test_capability_check_claude_cli_judge(monkeypatch):
    from evalbuilder import claude_cli

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(claude_cli, "claude_available", lambda cli_path=None: True)
    report = capability_check(Settings(judge_model="claude-cli:sonnet"))
    assert report["capabilities"]["judge"] is True
    assert report["capabilities"]["claude_cli"] is True
    assert "judge" not in report["degraded"]


def test_settings_agent_and_generator_models(tmp_path, monkeypatch):
    for var in ("EVALBUILDER_AGENT_MODEL", "EVALBUILDER_GENERATOR_MODEL"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text("EVALBUILDER_AGENT_MODEL=claude-cli:sonnet\nEVALBUILDER_GENERATOR_MODEL=claude-cli:opus\n")
    s = Settings.load(env)
    assert s.agent_model == "claude-cli:sonnet" and s.generator_model == "claude-cli:opus"
