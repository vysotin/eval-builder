"""Support bot — a router graph over two ReAct specialists with EXTERNAL tools.

Target contract for evalbuilder: exposes TOOLS and build_agent(model=None, tools=None).

Graph:  START → classify ─┬─ order_status / refund ─► support_agent ─► END
                          ├─ product_question ──────► kb_agent ──────► END
                          └─ other ─────────────────► decline ───────► END

Every tool talks to a network host that does not exist (`*.example.invalid`), so an
unmocked run returns error payloads fast instead of silently succeeding — the pipeline must mock
every tool (`issue_refund` is also side-effecting).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel

try:
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - older langchain/langgraph installs
    from langgraph.prebuilt import create_react_agent

    def create_agent(model, tools, system_prompt=None, **kwargs):
        return create_react_agent(model, tools, prompt=system_prompt, **kwargs)


from evalbuilder.testing import ScriptedChatModel, ai, tool_call

ORDER_API = "http://orders.example.invalid/api"
KB_API = "http://kb.example.invalid/search"
HTTP_TIMEOUT_SECONDS = 3


def _http(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        # Error payloads (not exceptions) so the agent's error handling is testable.
        return {"error": f"external service unavailable: {url} ({e})"}


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by id: status, product category, total, shipping ETA (external API)."""
    return _http(f"{ORDER_API}/orders/{urllib.parse.quote(order_id)}")


@tool
def check_refund_policy(category: str) -> dict:
    """Refund policy for a product category: window in days and restocking fee percent."""
    return _http(f"{ORDER_API}/policy/{urllib.parse.quote(category)}")


@tool
def issue_refund(order_id: str, amount: float, reason: str) -> dict:
    """Issue a refund for an order. Side-effecting: call only after the user explicitly confirms."""
    return _http(
        f"{ORDER_API}/refunds",
        {"order_id": order_id, "amount": amount, "reason": reason},
    )


@tool
def search_kb(query: str) -> dict:
    """Search the product knowledge base; returns matching articles with titles and snippets."""
    return _http(f"{KB_API}?q={urllib.parse.quote(query)}")


TOOLS = [lookup_order, check_refund_policy, issue_refund, search_kb]

CLASSIFY_PROMPT = (
    "Classify the customer's latest request in the context of the conversation. "
    "order_status: anything about an existing order's status, delivery, or tracking. "
    "refund: refund, return, cancellation, or money back — including confirmations "
    "('yes') that continue a refund conversation. product_question: how a product "
    "works, compatibility, warranty, specs. other: anything else."
)

SUPPORT_PROMPT = (
    "You are the support specialist for Acme Store. Always call lookup_order before "
    "answering anything about an order, and mention the order id in your answer. "
    "For refunds: look up the order, call check_refund_policy for its category, tell "
    "the customer the refundable amount and window, then ask for an explicit yes. "
    "Call issue_refund ONLY after the customer explicitly confirms in this "
    "conversation. Never promise a refund amount before checking the policy. "
    "If a tool fails, apologize, explain you could not reach the system, and offer "
    "to retry later — never guess order details."
)

KB_PROMPT = (
    "You are the product expert for Acme Store. Use search_kb to ground every answer "
    "in knowledge-base articles and cite the article title. If nothing relevant is "
    "found, say so plainly instead of guessing. Do not discuss orders or refunds; "
    "point the customer to support for those."
)

DECLINE_TEXT = (
    "I can help with Acme Store orders, refunds, and product questions. "
    "That request is outside what I can do here."
)


class Route(BaseModel):
    intent: Literal["order_status", "refund", "product_question", "other"]


class SupportState(MessagesState):
    route: str


ROUTES = {
    "order_status": "support_agent",
    "refund": "support_agent",
    "product_question": "kb_agent",
    "other": "decline",
}


def build_agent(model=None, tools=None):
    """Build the compiled support-bot graph."""
    model = model or default_scripted_model()
    tools = TOOLS if tools is None else tools
    by_name = {t.name: t for t in tools}
    support_tools = [
        by_name[n] for n in ("lookup_order", "check_refund_policy", "issue_refund")
        if n in by_name
    ]
    kb_tools = [by_name["search_kb"]] if "search_kb" in by_name else []

    support_agent = create_agent(model, support_tools, system_prompt=SUPPORT_PROMPT)
    kb_agent = create_agent(model, kb_tools, system_prompt=KB_PROMPT)
    classifier = model.with_structured_output(Route)

    def classify(state: SupportState) -> dict:
        recent = [m for m in state["messages"] if m.type in ("human", "ai")][-6:]
        route = classifier.invoke([SystemMessage(content=CLASSIFY_PROMPT), *recent])
        return {"route": route.intent}

    def decline(state: SupportState) -> dict:
        return {"messages": [AIMessage(content=DECLINE_TEXT)]}

    def pick(state: SupportState) -> str:
        return ROUTES.get(state.get("route", "other"), "decline")

    graph = StateGraph(SupportState)
    graph.add_node("classify", classify)
    graph.add_node("support_agent", support_agent)
    graph.add_node("kb_agent", kb_agent)
    graph.add_node("decline", decline)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        pick,
        {"support_agent": "support_agent", "kb_agent": "kb_agent", "decline": "decline"},
    )
    graph.add_edge("support_agent", END)
    graph.add_edge("kb_agent", END)
    graph.add_edge("decline", END)
    return graph.compile()


# ── scripted default model (offline tests) ──────────────────────

_ORDER_ID = re.compile(r"order\s*#?\s*([A-Za-z]\d{3,}|\d{4,})", re.IGNORECASE)


def _order_id_from_history(messages) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            found = _ORDER_ID.search(m.content if isinstance(m.content, str) else "")
            if found:
                return found.group(1)
    return "unknown"


def _last_tool_json(messages, key: str) -> dict:
    for m in reversed(messages):
        if m.type == "tool":
            try:
                data = json.loads(m.content)
            except (ValueError, TypeError):
                continue
            if key in data:
                return data
    return {}


def _mentions_refund(messages) -> bool:
    return any(
        isinstance(m, HumanMessage) and re.search(r"refund|return|money back", str(m.content), re.I)
        for m in messages
    )


def _after_lookup(match, messages):
    order = json.loads(messages[-1].content)
    if "error" in order:
        return ai(
            f"Sorry — I could not reach the order system for order {order.get('order_id', '')}. "
            "Please try again later."
        )
    if _mentions_refund(messages):
        return tool_call("check_refund_policy", {"category": order.get("category", "general")})
    return ai(
        f"Order {order.get('order_id')} is {order.get('status')}; "
        f"estimated delivery {order.get('eta', 'unknown')}."
    )


def _after_policy(match, messages):
    policy = json.loads(messages[-1].content)
    order = _last_tool_json(messages, "order_id")
    return ai(
        f"Order {order.get('order_id')} is eligible for a refund of ${order.get('total')} "
        f"within {policy.get('window_days')} days. Reply yes to confirm the refund."
    )


def _confirm(match, messages):
    order = _last_tool_json(messages, "order_id")
    if not order:
        return ai("Which order would you like refunded? Please share the order id.")
    return tool_call(
        "issue_refund",
        {"order_id": order.get("order_id"), "amount": order.get("total", 0), "reason": "customer request"},
    )


def _after_refund(match, messages):
    refund = json.loads(messages[-1].content)
    order = _last_tool_json(messages, "order_id")
    return ai(f"Refund {refund.get('refund_id')} issued for order {order.get('order_id')}.")


def _after_kb(match, messages):
    result = json.loads(messages[-1].content)
    articles = result.get("articles") or []
    if not articles:
        return ai("I could not find anything relevant in the knowledge base.")
    first = articles[0]
    return ai(f"According to '{first.get('title')}': {first.get('snippet')}")


def default_scripted_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        structured_script=[
            (r"refund|return|money back", lambda m, _msgs: {"intent": "refund"}),
            (r"order\s*#?\s*[A-Za-z]?\d{3,}|track|deliver", lambda m, _msgs: {"intent": "order_status"}),
            (r"how|does|compatible|warranty|work|support", lambda m, _msgs: {"intent": "product_question"}),
            (r".*", lambda m, _msgs: {"intent": "other"}),
        ],
        script=[
            (r"TOOL:.*refund_id", _after_refund),
            (r"TOOL:.*window_days", _after_policy),
            (r"TOOL:.*\"status\"", _after_lookup),
            (r"TOOL:.*articles", _after_kb),
            (r"TOOL:.*error", lambda m, msgs: ai("Sorry — I could not reach the system right now. Please try again later.")),
            (r"^\s*(yes|confirm|go ahead)", _confirm),
            (
                r"order\s*#?\s*([A-Za-z]\d{3,}|\d{4,})",
                lambda m, _msgs: tool_call("lookup_order", {"order_id": m.group(1)}),
            ),
            (r"refund|return", lambda m, _msgs: ai("Which order would you like refunded? Please share the order id.")),
            (r"order|track|deliver", lambda m, _msgs: ai("Which order would you like me to check? Please share the order id.")),
            (r"(.+)", lambda m, _msgs: tool_call("search_kb", {"query": m.group(1).strip()[:80]})),
        ],
    )
