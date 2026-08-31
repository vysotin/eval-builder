"""API-key LLM providers (Anthropic, OpenAI, Google Gemini) behind `provider:model` specs.

Every model spec in evalbuilder is `provider:model[@effort]`. Two providers are
special: `claude-cli:` (Claude Code CLI, subscription auth) and `scripted:` (offline
test models). Everything else is an API-key provider described here and instantiated
through LangChain's `init_chat_model`, so `anthropic:claude-sonnet-5`,
`openai:gpt-5` and `gemini:gemini-2.5-pro` are interchangeable with
`claude-cli:sonnet` for the agent, the generator and the judges.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any

CLAUDE_CLI_PROVIDER = "claude-cli"
SCRIPTED_PROVIDER = "scripted"

EFFORT_LEVELS = ("low", "medium", "high")

# Anthropic extended-thinking budgets per effort level (tokens).
ANTHROPIC_THINKING_BUDGET = {"low": 1024, "medium": 4096, "high": 16000}


@dataclass(frozen=True)
class Provider:
    name: str  # canonical `init_chat_model` provider name
    env_var: str  # API key environment variable
    package: str  # importable module that must be installed
    extra: str  # `uv sync --extra <extra>` that installs it
    example: str  # example model spec for docs and diagnostics
    aliases: tuple[str, ...] = ()
    effort: str = "ignored"  # how `@effort` is applied: reasoning_effort | thinking | ignored
    default_kwargs: dict[str, Any] = field(default_factory=dict)


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        package="langchain_anthropic",
        extra="anthropic",
        example="anthropic:claude-sonnet-5",
        aliases=("claude",),
        effort="thinking",
    ),
    "openai": Provider(
        name="openai",
        env_var="OPENAI_API_KEY",
        package="langchain_openai",
        extra="openai",
        example="openai:gpt-5",
        effort="reasoning_effort",
    ),
    "google_genai": Provider(
        name="google_genai",
        env_var="GOOGLE_API_KEY",
        package="langchain_google_genai",
        extra="gemini",
        example="gemini:gemini-2.5-pro",
        aliases=("gemini", "google"),
        effort="ignored",
    ),
}

_ALIASES: dict[str, str] = {
    alias: p.name for p in PROVIDERS.values() for alias in (p.name, *p.aliases)
}

# Backwards-compatible view used by `config.capability_check` and tests.
PROVIDER_KEYS = {name: p.env_var for name, p in PROVIDERS.items()}


def canonical_provider(name: str) -> str:
    """`gemini` → `google_genai`, `claude` → `anthropic`; unknown names pass through."""
    return _ALIASES.get(name, name)


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """`provider:model[@effort]` → (canonical provider, model, effort or None).

    `scripted:module.path:factory` keeps everything after the first colon as the
    model; `@effort` is only split off when it is a known level.
    """
    if ":" not in spec:
        raise ValueError(f"model spec must look like provider:model, got {spec!r}")
    provider, model = spec.split(":", 1)
    provider = canonical_provider(provider.strip())
    effort: str | None = None
    if provider != SCRIPTED_PROVIDER and "@" in model:
        model, _, suffix = model.rpartition("@")
        effort = suffix or None
    return provider, model.strip(), effort


def normalize_spec(spec: str) -> str:
    """Rewrite the provider part to its canonical name (`gemini:x` → `google_genai:x`)."""
    provider, model, effort = parse_spec(spec)
    return f"{provider}:{model}" + (f"@{effort}" if effort else "")


def package_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def provider_status(name: str) -> dict:
    """Readiness of one API provider: key present? package installed? example spec."""
    provider = PROVIDERS[canonical_provider(name)]
    has_key = bool(os.environ.get(provider.env_var))
    has_package = package_installed(provider.package)
    return {
        "provider": provider.name,
        "aliases": list(provider.aliases),
        "env_var": provider.env_var,
        "key": has_key,
        "package": has_package,
        "extra": provider.extra,
        "ready": has_key and has_package,
        "example": provider.example,
    }


def providers_status() -> dict[str, dict]:
    return {name: provider_status(name) for name in PROVIDERS}


def provider_ready(spec: str) -> tuple[bool, str]:
    """(ready, reason) for any model spec.

    `claude-cli:` needs the claude binary; `scripted:` is always ready; API providers
    need their key in the environment and their LangChain package installed.
    """
    try:
        provider, _, _ = parse_spec(spec)
    except ValueError as e:
        return False, str(e)
    if provider == CLAUDE_CLI_PROVIDER:
        from evalbuilder.claude_cli import claude_available

        return (True, "") if claude_available() else (False, "claude CLI not on PATH")
    if provider == SCRIPTED_PROVIDER:
        return True, ""
    if provider not in PROVIDERS:
        return False, (
            f"unknown provider {provider!r}; use claude-cli, scripted, or one of "
            f"{sorted(_ALIASES)}"
        )
    status = provider_status(provider)
    missing = []
    if not status["key"]:
        missing.append(f"set {status['env_var']}")
    if not status["package"]:
        missing.append(f"install {PROVIDERS[provider].package} (uv sync --extra {status['extra']} or pip install -e '.[{status['extra']}]')")
    if missing:
        return False, f"provider {provider!r} not ready: " + "; ".join(missing)
    return True, ""


def effort_kwargs(provider: str, effort: str | None) -> dict[str, Any]:
    """Provider-specific kwargs that realize an `@effort` suffix (empty when ignored)."""
    if not effort:
        return {}
    mode = PROVIDERS[provider].effort if provider in PROVIDERS else "ignored"
    if mode != "ignored" and effort not in EFFORT_LEVELS:
        raise ValueError(f"unknown effort {effort!r} for provider {provider!r}; use one of {EFFORT_LEVELS}")
    if mode == "reasoning_effort":
        return {"reasoning_effort": effort}
    if mode == "thinking":
        return {
            "thinking": {"type": "enabled", "budget_tokens": ANTHROPIC_THINKING_BUDGET[effort]},
            "max_tokens": max(ANTHROPIC_THINKING_BUDGET[effort] * 2, 8192),
        }
    return {}


def build_model(spec: str, **overrides: Any):
    """Instantiate an API-key chat model for `provider:model[@effort]`.

    Raises ValueError with the same reason `provider_ready` would give when the key or
    package is missing, so callers fail before the first network call.
    """
    provider, model, effort = parse_spec(spec)
    if provider not in PROVIDERS:
        raise ValueError(f"{spec!r}: unknown API provider {provider!r}")
    ready, reason = provider_ready(spec)
    if not ready:
        raise ValueError(f"{spec!r}: {reason}")
    from langchain.chat_models import init_chat_model

    kwargs: dict[str, Any] = dict(PROVIDERS[provider].default_kwargs)
    kwargs.update(effort_kwargs(provider, effort))
    kwargs.update(overrides)
    return init_chat_model(model, model_provider=provider, **kwargs)
