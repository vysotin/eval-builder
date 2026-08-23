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
