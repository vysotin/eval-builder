"""Settings from .env and the capability matrix."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_JUDGE_MODEL = "anthropic:claude-sonnet-5"


@dataclass(frozen=True)
class Settings:
    judge_model: str = DEFAULT_JUDGE_MODEL
    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    langsmith_project: str | None = None

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        values: dict[str, str | None] = {}
        if env_file is None and Path(".env").exists():
            env_file = Path(".env")
        if env_file is not None:
            values.update(dotenv_values(env_file))

        def get(key: str) -> str | None:
            return os.environ.get(key) or values.get(key)

        return cls(
            judge_model=get("EVALBUILDER_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
            langsmith_api_key=get("LANGSMITH_API_KEY"),
            langsmith_endpoint=get("LANGSMITH_ENDPOINT"),
            langsmith_project=get("LANGSMITH_PROJECT"),
        )


def _has_judge_key(model: str) -> bool:
    provider = model.split(":", 1)[0]
    keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
    }
    var = keys.get(provider)
    return bool(var and os.environ.get(var))


def capability_check(settings: Settings, target_module: str | None = None) -> dict:
    degraded: list[str] = []
    blocking: list[dict] = []
    caps: dict[str, bool] = {}

    caps["langgraph"] = importlib.util.find_spec("langgraph") is not None
    if not caps["langgraph"]:
        blocking.append(
            {"issue": "langgraph is not installed", "fix": "uv pip install langgraph"}
        )

    caps["judge"] = _has_judge_key(settings.judge_model)
    if not caps["judge"]:
        degraded.append("judge")

    caps["langsmith"] = bool(settings.langsmith_api_key)
    if not caps["langsmith"]:
        degraded.append("langsmith")

    if target_module:
        try:
            caps["target"] = importlib.util.find_spec(target_module) is not None
        except ModuleNotFoundError:
            caps["target"] = False
        if not caps["target"]:
            blocking.append(
                {
                    "issue": f"target module '{target_module}' not importable",
                    "fix": "check the module path and install its dependencies",
                }
            )

    return {
        "ready": not blocking,
        "degraded": degraded,
        "blocking": blocking,
        "capabilities": caps,
    }
