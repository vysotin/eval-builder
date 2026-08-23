"""Deterministic scripted chat model for offline agent runs and tests."""

from __future__ import annotations

import re
from typing import Any, Callable

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

ScriptRule = tuple[str, Callable[[re.Match, list[BaseMessage]], AIMessage]]


class ScriptMissError(Exception):
    def __init__(self, text: str, patterns: list[str]):
        super().__init__(
            f"no script rule matched {text!r}; patterns: {patterns}"
        )


def ai(text: str) -> AIMessage:
    return AIMessage(content=text)


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _match_text(messages: list[BaseMessage]) -> str:
    """The text the script matches against for the current step."""
    last = messages[-1]
    if isinstance(last, ToolMessage):
        content = last.content if isinstance(last.content, str) else str(last.content)
        return f"TOOL:{content}"
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if content:
            return content
    return ""


class ScriptedChatModel(BaseChatModel):
    """Regex-scripted chat model: first matching rule produces the AIMessage."""

    script: list[Any]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        text = _match_text(messages)
        for pattern, factory in self.script:
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                return ChatResult(
                    generations=[ChatGeneration(message=factory(m, messages))]
                )
        raise ScriptMissError(text, [p for p, _ in self.script])
