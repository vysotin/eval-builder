"""Weather bot — a single ReAct agent with two tools.

Target contract for evalbuilder: exposes TOOLS and build_agent(model=None, tools=None).
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

try:
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - older langchain/langgraph installs
    from langgraph.prebuilt import create_react_agent

    def create_agent(model, tools, system_prompt=None, **kwargs):
        return create_react_agent(model, tools, prompt=system_prompt, **kwargs)


from evalbuilder.testing import ScriptedChatModel, ai, tool_call

SYSTEM_PROMPT = (
    "You are a weather assistant. Use get_weather for conditions and "
    "get_alerts for warnings. Always name the city in your answer."
)


@tool
def get_weather(city: str) -> dict:
    """Get current weather conditions for a city."""
    return {"temp": 72, "condition": "sunny", "city": city}


@tool
def get_alerts(city: str) -> dict:
    """Get active weather alerts for a city."""
    return {"alerts": [], "city": city}


TOOLS = [get_weather, get_alerts]


def _final_from_tool(match, messages):
    data = json.loads(messages[-1].content)
    return ai(
        f"It is {data.get('temp')}F and {data.get('condition')} in {data.get('city')}."
    )


def default_scripted_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        script=[
            (r"TOOL:.*temp", _final_from_tool),
            (
                r"weather in (\w+)",
                lambda m, _msgs: tool_call("get_weather", {"city": m.group(1)}),
            ),
            (
                r"alerts? (?:in|for) (\w+)",
                lambda m, _msgs: tool_call("get_alerts", {"city": m.group(1)}),
            ),
            (r"TOOL:.*alerts", lambda m, msgs: ai(f"Alerts: {msgs[-1].content}")),
        ]
    )


def build_agent(model=None, tools=None):
    """Build the compiled weather-bot graph."""
    return create_agent(
        model or default_scripted_model(),
        tools if tools is not None else TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
