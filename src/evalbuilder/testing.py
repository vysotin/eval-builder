"""Deterministic scripted chat model for offline agent runs and tests."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

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


def _user_transcript(messages: list[BaseMessage]) -> str:
    return "\n".join(
        m.content if isinstance(m.content, str) else str(m.content)
        for m in messages
        if isinstance(m, HumanMessage)
    )


class ScriptedChatModel(BaseChatModel):
    """Regex-scripted chat model: first matching rule produces the AIMessage.

    `structured_script` serves `with_structured_output`: rules are matched against
    the full user transcript and their factories return a dict (or an AIMessage whose
    content is JSON).
    """

    script: list[Any]
    structured_script: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        rules = self.structured_script or self.script

        def _call(inputs):
            from langchain_core.messages import convert_to_messages

            if isinstance(inputs, str):
                messages = [HumanMessage(content=inputs)]
            elif hasattr(inputs, "to_messages"):
                messages = list(inputs.to_messages())
            else:
                messages = convert_to_messages(list(inputs))
            text = _user_transcript(messages) if self.structured_script else _match_text(messages)
            for pattern, factory in rules:
                m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if m:
                    value = factory(m, messages)
                    if isinstance(value, AIMessage):
                        value = json.loads(value.content)
                    if hasattr(schema, "model_validate"):
                        return schema.model_validate(value)
                    return value
            raise ScriptMissError(text, [p for p, _ in rules])

        return RunnableLambda(_call)

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
