"""Travel planner — supervisor agent delegating to flight/hotel subagents via tools.

Target contract for evalbuilder: exposes TOOLS and build_agent(model=None, tools=None).
Subagent delegation is tool-shaped (like ADK's transfer_to_agent), so mock rules
apply to subagents the same way they apply to plain tools.
"""

from __future__ import annotations

from langchain_core.tools import tool

try:
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - older langchain/langgraph installs
    from langgraph.prebuilt import create_react_agent

    def create_agent(model, tools, system_prompt=None, **kwargs):
        return create_react_agent(model, tools, prompt=system_prompt, **kwargs)


from evalbuilder.testing import ScriptedChatModel, ai, tool_call

SUPERVISOR_PROMPT = (
    "You are a travel-planning supervisor. Delegate flight requests to flight_agent "
    "and hotel requests to hotel_agent, then summarize for the user. "
    "Never confirm a booking without an explicit user yes."
)


@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search available flights between two cities on a date."""
    return {"flights": [{"id": "AA100", "airline": "AA", "price": 350,
                          "origin": origin, "destination": destination, "date": date}]}


@tool
def book_flight(flight_id: str) -> dict:
    """Book a flight by id. Side-effecting: only call after user confirmation."""
    return {"status": "booked", "flight_id": flight_id}


@tool
def search_hotels(city: str, check_in: str, check_out: str) -> dict:
    """Search hotels in a city for a date range."""
    return {"hotels": [{"id": "H7", "name": "Grand", "price": 180, "city": city}]}


@tool
def book_hotel(hotel_id: str) -> dict:
    """Book a hotel by id. Side-effecting: only call after user confirmation."""
    return {"status": "booked", "hotel_id": hotel_id}


def _flight_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        script=[
            (r"TOOL:.*flights", lambda m, msgs: ai("Found AA100 at $350.")),
            (
                r"from (\w+) to (\w+).*?on ([\d-]+)",
                lambda m, _msgs: tool_call(
                    "search_flights",
                    {"origin": m.group(1), "destination": m.group(2), "date": m.group(3)},
                ),
            ),
            (
                r"from (\w+) to (\w+)",
                lambda m, _msgs: tool_call(
                    "search_flights",
                    {"origin": m.group(1), "destination": m.group(2), "date": "unknown"},
                ),
            ),
        ]
    )


def _hotel_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        script=[
            (r"TOOL:.*hotels", lambda m, msgs: ai("Found Grand at $180/night.")),
            (
                r"hotel in (\w+)",
                lambda m, _msgs: tool_call(
                    "search_hotels",
                    {"city": m.group(1), "check_in": "unknown", "check_out": "unknown"},
                ),
            ),
        ]
    )


def _build_flight_agent():
    return create_agent(
        _flight_model(), [search_flights, book_flight],
        system_prompt="You are a flight specialist.",
    )


def _build_hotel_agent():
    return create_agent(
        _hotel_model(), [search_hotels, book_hotel],
        system_prompt="You are a hotel specialist.",
    )


@tool
def flight_agent(request: str) -> str:
    """Delegate a flight search or booking request to the flight specialist."""
    result = _build_flight_agent().invoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    return result["messages"][-1].content


@tool
def hotel_agent(request: str) -> str:
    """Delegate a hotel search or booking request to the hotel specialist."""
    result = _build_hotel_agent().invoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    return result["messages"][-1].content


TOOLS = [flight_agent, hotel_agent]


def default_scripted_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        script=[
            (
                r"TOOL:.*AA100",
                lambda m, _msgs: ai(
                    "I found flight AA100 at $350. Would you like me to book it?"
                ),
            ),
            (
                r"TOOL:.*Grand",
                lambda m, _msgs: ai("I found the Grand hotel at $180/night."),
            ),
            (
                r"^(yes|confirm)",
                lambda m, _msgs: ai("Booking confirmed for flight AA100."),
            ),
            (
                r"book it",
                lambda m, _msgs: ai("Please confirm: book flight AA100? (yes/no)"),
            ),
            (
                r"flight from .*",
                lambda m, _msgs: tool_call("flight_agent", {"request": m.group(0)}),
            ),
            (
                r"hotel in .*",
                lambda m, _msgs: tool_call("hotel_agent", {"request": m.group(0)}),
            ),
        ]
    )


def build_agent(model=None, tools=None):
    """Build the compiled travel-planner supervisor graph."""
    return create_agent(
        model or default_scripted_model(),
        tools if tools is not None else TOOLS,
        system_prompt=SUPERVISOR_PROMPT,
    )
