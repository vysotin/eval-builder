"""API-key providers: spec normalization, readiness diagnostics, model construction."""

import pytest

from evalbuilder import providers
from evalbuilder.claude_cli import model_from_spec
from evalbuilder.config import Settings, capability_check
from evalbuilder.providers import (
    PROVIDERS,
    build_model,
    effort_kwargs,
    normalize_spec,
    parse_spec,
    provider_ready,
    providers_status,
)


def test_parse_spec_aliases_and_effort():
    assert parse_spec("gemini:gemini-2.5-pro") == ("google_genai", "gemini-2.5-pro", None)
    assert parse_spec("google:gemini-2.5-flash@low") == ("google_genai", "gemini-2.5-flash", "low")
    assert parse_spec("claude:claude-sonnet-5") == ("anthropic", "claude-sonnet-5", None)
    assert parse_spec("openai:gpt-5@high") == ("openai", "gpt-5", "high")
    assert parse_spec("claude-cli:opus@high") == ("claude-cli", "opus", "high")
    assert parse_spec("scripted:examples.weather_bot.agent:default_scripted_model") == (
        "scripted", "examples.weather_bot.agent:default_scripted_model", None,
    )
    with pytest.raises(ValueError):
        parse_spec("no-colon")


def test_normalize_spec():
    assert normalize_spec("gemini:gemini-2.5-pro") == "google_genai:gemini-2.5-pro"
    assert normalize_spec("claude:x@medium") == "anthropic:x@medium"
    assert normalize_spec("claude-cli:sonnet") == "claude-cli:sonnet"


def test_effort_kwargs_per_provider():
    assert effort_kwargs("openai", "high") == {"reasoning_effort": "high"}
    assert effort_kwargs("anthropic", "low")["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert effort_kwargs("google_genai", "high") == {}
    assert effort_kwargs("openai", None) == {}
    with pytest.raises(ValueError):
        effort_kwargs("openai", "max")


def _all_keys_unset(monkeypatch):
    for p in PROVIDERS.values():
        monkeypatch.delenv(p.env_var, raising=False)


def test_provider_ready_reports_missing_key_and_package(monkeypatch):
    _all_keys_unset(monkeypatch)
    monkeypatch.setattr(providers, "package_installed", lambda module: False)
    ready, reason = provider_ready("openai:gpt-5")
    assert not ready
    assert "OPENAI_API_KEY" in reason and "uv sync --extra openai" in reason
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    ready, reason = provider_ready("gemini:gemini-2.5-pro")
    assert not ready and "langchain_google_genai" in reason and "GOOGLE_API_KEY" not in reason


def test_provider_ready_when_key_and_package_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(providers, "package_installed", lambda module: True)
    assert provider_ready("claude:claude-sonnet-5") == (True, "")
    assert provider_ready("scripted:x:y") == (True, "")
    ok, reason = provider_ready("mystery:model")
    assert not ok and "unknown provider" in reason


def test_providers_status_shape(monkeypatch):
    _all_keys_unset(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    status = providers_status()
    assert set(status) == {"anthropic", "openai", "google_genai"}
    assert status["openai"]["key"] is True and status["anthropic"]["key"] is False
    assert status["google_genai"]["aliases"] == ["gemini", "google"]
    for entry in status.values():
        assert {"env_var", "package", "ready", "example", "extra"} <= set(entry)


def test_build_model_fails_early_without_key(monkeypatch):
    _all_keys_unset(monkeypatch)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_model("openai:gpt-5")
    with pytest.raises(ValueError, match="unknown API provider"):
        build_model("claude-cli:sonnet")


@pytest.mark.skipif(not providers.package_installed("langchain_openai"), reason="langchain-openai not installed")
def test_model_from_spec_builds_openai_model_with_effort(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = model_from_spec("openai:gpt-5@low")
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "gpt-5" and llm.reasoning_effort == "low"


@pytest.mark.skipif(not providers.package_installed("langchain_anthropic"), reason="langchain-anthropic not installed")
def test_model_from_spec_builds_anthropic_model_with_thinking(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    llm = model_from_spec("claude:claude-sonnet-5@medium")
    assert type(llm).__name__ == "ChatAnthropic"
    assert llm.model == "claude-sonnet-5" and llm.thinking["budget_tokens"] == 4096


@pytest.mark.skipif(not providers.package_installed("langchain_google_genai"), reason="langchain-google-genai not installed")
def test_model_from_spec_builds_gemini_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    llm = model_from_spec("gemini:gemini-2.5-pro")
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert "gemini-2.5-pro" in llm.model


def test_capability_check_lists_providers(monkeypatch):
    _all_keys_unset(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(providers, "package_installed", lambda module: module == "langchain_openai")
    report = capability_check(Settings(judge_model="openai:gpt-5"))
    assert report["providers"]["openai"]["ready"] is True
    assert report["capabilities"]["provider_openai"] is True
    assert report["capabilities"]["provider_anthropic"] is False
    assert report["capabilities"]["judge"] is True and "judge" not in report["degraded"]
    assert report["models"]["judge"] == "openai:gpt-5"
