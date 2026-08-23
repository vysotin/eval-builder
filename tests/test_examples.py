from langchain_core.messages import AIMessage

from examples.travel_planner.agent import build_agent as build_travel
from examples.weather_bot.agent import build_agent as build_weather


def _final_text(state) -> str:
    return state["messages"][-1].content


def test_weather_bot_scripted_happy_path():
    graph = build_weather()
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in Paris?"}]}
    )
    text = _final_text(out)
    assert "Paris" in text and "72" in text
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert tool_msgs and tool_msgs[0].name == "get_weather"


def test_travel_planner_delegates_to_flight_agent():
    graph = build_travel()
    out = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Find me a flight from SFO to JFK on 2026-09-01",
                }
            ]
        }
    )
    assert "AA100" in _final_text(out)
    called = [
        tc["name"]
        for m in out["messages"]
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    ]
    assert "flight_agent" in called


def test_travel_planner_gates_booking_on_confirmation():
    graph = build_travel()
    out = graph.invoke({"messages": [{"role": "user", "content": "book it"}]})
    assert "confirm" in _final_text(out).lower()


def test_build_agent_accepts_injected_tools():
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> dict:
        """Mock weather."""
        return {"temp": 99, "condition": "scripted-mock", "city": city}

    @tool
    def get_alerts(city: str) -> dict:
        """Mock alerts."""
        return {"alerts": [], "city": city}

    graph = build_weather(tools=[get_weather, get_alerts])
    out = graph.invoke({"messages": [{"role": "user", "content": "weather in Oslo?"}]})
    assert "99" in _final_text(out)
