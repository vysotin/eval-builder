"""Claude Code CLI as a LangChain chat model (subscription auth, no API key).

`ChatClaudeCLI` keeps `langchain_claude_code.ChatClaudeCode`'s parameter surface but
replaces its transport: the bundled `claude-code-sdk` cannot parse the current CLI's
event stream, and its tool calling `json.loads` the whole reply. Here every call is one
isolated `claude -p --output-format json` subprocess; tool calls and structured output
both use `--json-schema`, and conversation history (including tool calls/results) is
flattened into a transcript because the CLI cannot replay assistant/tool turns.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Sequence
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.messages import convert_to_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_claude_code.chat_models import ChatClaudeCode, _tool_to_anthropic_schema

CLAUDE_CLI_PROVIDER = "claude-cli"
SCRIPTED_PROVIDER = "scripted"

TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "Your reply to the user. Empty when you are calling tools.",
        },
        "tool_calls": {
            "type": "array",
            "description": "Tools to call now, in order. Empty when you answer directly.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "args": {"type": "object"},
                },
                "required": ["name", "args"],
            },
        },
    },
    "required": ["content", "tool_calls"],
}

TOOL_INSTRUCTION = (
    "You can call the tools listed below (JSON schemas). Reply with a JSON object "
    "matching the required output schema: put tool invocations in `tool_calls` "
    "(name + args exactly per schema) and leave `content` empty, or put your final "
    "answer to the user in `content` with an empty `tool_calls` list. Never invent "
    "tool results; call the tool instead.\n\nTools:\n"
)

TRANSCRIPT_INSTRUCTION = (
    "The conversation so far is below. Produce the assistant's next turn only."
)

_NEUTRAL_CWD: str | None = None


class ClaudeCLIError(RuntimeError):
    """The claude CLI failed, returned an error result, or produced unparsable output."""


def _neutral_cwd() -> str:
    """A directory with no CLAUDE.md/memory so project context never leaks in."""
    global _NEUTRAL_CWD
    if _NEUTRAL_CWD is None or not os.path.isdir(_NEUTRAL_CWD):
        _NEUTRAL_CWD = tempfile.mkdtemp(prefix="evalbuilder-claude-")
    return _NEUTRAL_CWD


def claude_available(cli_path: str | None = None) -> bool:
    return shutil.which(cli_path or "claude") is not None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def render_transcript(messages: list[BaseMessage]) -> tuple[str | None, str]:
    """Split messages into (system prompt, flattened transcript)."""
    system_parts: list[str] = []
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(_text(msg.content))
        elif isinstance(msg, AIMessage):
            text = _text(msg.content)
            if text:
                lines.append(f"[assistant]\n{text}")
            for tc in msg.tool_calls or []:
                lines.append(
                    f"[assistant tool_call id={tc.get('id')}] {tc['name']}("
                    f"{json.dumps(tc.get('args', {}), ensure_ascii=False)})"
                )
        elif isinstance(msg, ToolMessage):
            lines.append(
                f"[tool result id={msg.tool_call_id} name={msg.name or ''}]\n"
                f"{_text(msg.content)}"
            )
        else:
            role = getattr(msg, "type", "user")
            role = "user" if role == "human" else role
            lines.append(f"[{role}]\n{_text(msg.content)}")
    system = "\n\n".join(p for p in system_parts if p) or None
    return system, "\n\n".join(lines)


def _json_schema_of(schema: Any) -> tuple[dict, Any]:
    """Return (json schema dict, pydantic class or None)."""
    if isinstance(schema, dict):
        return schema, None
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema(), schema
    from langchain_core.utils.function_calling import convert_to_openai_tool

    tool_schema = convert_to_openai_tool(schema)
    return tool_schema["function"]["parameters"], None


class ChatClaudeCLI(ChatClaudeCode):
    """LangChain chat model backed by isolated `claude -p` subprocess calls."""

    model: str = "sonnet"
    max_retries: int = 1
    timeout_seconds: float = 240.0
    strip_api_key: bool = True
    retry_backoff_seconds: float = 2.0

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    # ── tools / structured output ──────────────────────────────

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        new = self.model_copy()
        new._bound_tools = [_tool_to_anthropic_schema(t) for t in tools]
        return new

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        json_schema, model_cls = _json_schema_of(schema)

        def _call(inputs: Any) -> Any:
            messages = self._coerce_messages(inputs)
            system, transcript = render_transcript(messages)
            data = self._run_cli(transcript, system, json_schema)
            parsed = data.get("structured_output")
            if parsed is None:
                parsed = _loads_lenient(data.get("result", ""))
            if parsed is None:
                raise ClaudeCLIError(
                    f"claude returned no structured output: {data.get('result')!r}"
                )
            value = model_cls.model_validate(parsed) if model_cls else parsed
            if include_raw:
                return {"raw": AIMessage(content=str(data.get("result", ""))), "parsed": value, "parsing_error": None}
            return value

        return RunnableLambda(_call)

    @staticmethod
    def _coerce_messages(inputs: Any) -> list[BaseMessage]:
        if isinstance(inputs, str):
            return convert_to_messages([{"role": "user", "content": inputs}])
        if hasattr(inputs, "to_messages"):
            return list(inputs.to_messages())
        return convert_to_messages(list(inputs))

    # ── transport ──────────────────────────────────────────────

    def _argv(self, system: str | None, json_schema: dict | None) -> list[str]:
        argv = [
            self.cli_path or "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--tools",
            "",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        if system:
            argv += ["--system-prompt", system]
        if json_schema is not None:
            argv += ["--json-schema", json.dumps(json_schema)]
        if self.effort:
            argv += ["--effort", self.effort]
        if self.permission_mode:
            argv += ["--permission-mode", self.permission_mode]
        return argv

    def _run_cli(self, prompt: str, system: str | None, json_schema: dict | None) -> dict:
        env = dict(os.environ)
        if self.strip_api_key:
            env.pop("ANTHROPIC_API_KEY", None)
        argv = self._argv(system, json_schema)
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                proc = subprocess.run(
                    argv,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=self.cwd or _neutral_cwd(),
                    timeout=self.timeout_seconds,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                last_error = f"{type(e).__name__}: {e}"
            else:
                if proc.returncode != 0:
                    last_error = (
                        f"claude exited {proc.returncode}: "
                        f"{(proc.stderr or proc.stdout or '').strip()[:500]}"
                    )
                else:
                    try:
                        data = json.loads(proc.stdout)
                    except ValueError:
                        last_error = f"unparsable claude output: {proc.stdout[:300]!r}"
                    else:
                        if data.get("is_error"):
                            last_error = f"claude error: {data.get('result')}"
                        else:
                            self._last_result = data
                            return data
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_seconds * (attempt + 1))
        raise ClaudeCLIError(last_error)

    # ── generation ─────────────────────────────────────────────

    def _system_with_tools(self, system: str | None) -> str | None:
        if not self._bound_tools:
            return system
        block = TOOL_INSTRUCTION + json.dumps(self._bound_tools, indent=1, ensure_ascii=False)
        return f"{system}\n\n{block}" if system else block

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, transcript = render_transcript(messages)
        system = self._system_with_tools(system)
        prompt = f"{TRANSCRIPT_INSTRUCTION}\n\n{transcript}" if "\n\n" in transcript or transcript.startswith("[") else transcript
        schema = TOOL_CALL_SCHEMA if self._bound_tools else None
        data = self._run_cli(prompt, system, schema)

        gen_info = {
            "model": self.model,
            "backend": "claude-cli",
            "session_id": data.get("session_id"),
            "duration_ms": data.get("duration_ms"),
            "num_turns": data.get("num_turns"),
            "total_cost_usd": data.get("total_cost_usd"),
            "usage": data.get("usage"),
        }
        text = str(data.get("result") or "")
        if not self._bound_tools:
            message = AIMessage(content=text, response_metadata=gen_info)
            return ChatResult(generations=[ChatGeneration(message=message, generation_info=gen_info)])

        parsed = data.get("structured_output")
        if not isinstance(parsed, dict):
            parsed = _loads_lenient(text) or {"content": text, "tool_calls": []}
        tool_calls = [
            {
                "name": tc["name"],
                "args": tc.get("args") or {},
                "id": f"call_{uuid4().hex[:8]}",
                "type": "tool_call",
            }
            for tc in parsed.get("tool_calls") or []
            if isinstance(tc, dict) and tc.get("name")
        ]
        message = AIMessage(
            content=str(parsed.get("content") or ""),
            tool_calls=tool_calls,
            response_metadata=gen_info,
        )
        return ChatResult(generations=[ChatGeneration(message=message, generation_info=gen_info)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        result = self._generate(messages, stop=stop, **kwargs)
        msg = result.generations[0].message
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=msg.content, tool_call_chunks=[
                {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                for i, tc in enumerate(msg.tool_calls)
            ])
        )

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self._stream(messages, stop=stop, **kwargs):
            yield chunk


def _loads_lenient(text: str) -> dict | None:
    """Parse a JSON object from text, tolerating fences and surrounding prose."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate[4:] if candidate.startswith("json") else candidate
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(candidate[start : end + 1])
            return value if isinstance(value, dict) else None
        except ValueError:
            return None
    return None


# ── model spec resolution ──────────────────────────────────────


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """`provider:model[@effort]` → (provider, model, effort)."""
    provider, _, rest = spec.partition(":")
    if not rest:
        raise ValueError(f"model spec {spec!r} must look like provider:model")
    model, _, effort = rest.partition("@")
    return provider, model, effort or None


def provider_ready(spec: str) -> tuple[bool, str]:
    """(ready, reason) for a model spec without instantiating it."""
    provider, model, _ = parse_spec(spec)
    if provider == CLAUDE_CLI_PROVIDER:
        return (True, "") if claude_available() else (False, "claude CLI not on PATH")
    if provider == SCRIPTED_PROVIDER:
        return True, ""
    from evalbuilder.config import _has_judge_key

    return (True, "") if _has_judge_key(spec) else (False, f"no API key for provider {provider!r}")


def model_from_spec(spec: str, **overrides: Any):
    """Instantiate a chat model from `claude-cli:…`, `scripted:module:fn`, or an
    `init_chat_model` string."""
    provider, model, effort = parse_spec(spec)
    if provider == CLAUDE_CLI_PROVIDER:
        kwargs: dict[str, Any] = {"model": model}
        if effort:
            kwargs["effort"] = effort
        kwargs.update(overrides)
        return ChatClaudeCLI(**kwargs)
    if provider == SCRIPTED_PROVIDER:
        module_name, _, fn_name = model.rpartition(":")
        if not module_name:
            raise ValueError("scripted spec must be scripted:module.path:factory")
        return getattr(importlib.import_module(module_name), fn_name)()
    from langchain.chat_models import init_chat_model

    return init_chat_model(spec, **overrides)
