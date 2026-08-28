"""The support_bot example: router graph, external tools, scripted offline model."""


from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from examples.support_bot import agent as sb

ORDER = {"order_id": "A1234", "status": "delivered", "category": "electronics", "total": 59.99, "eta": "2026-08-20"}
POLICY = {"category": "electronics", "window_days": 30, "restocking_fee_pct": 0}
REFUND = {"refund_id": "R77", "status": "issued"}
KB = {"articles": [{"title": "Pairing the X1 headset", "snippet": "Hold the button for 5 seconds."}]}


def _mock_tools(order=ORDER, policy=POLICY, refund=REFUND, kb=KB):
    calls = []

    @tool
    def lookup_order(order_id: str) -> dict:
        """mock"""
        calls.append(("lookup_order", order_id))
        return order

    @tool
    def check_refund_policy(category: str) -> dict:
        """mock"""
        calls.append(("check_refund_policy", category))
        return policy

    @tool
    def issue_refund(order_id: str, amount: float, reason: str) -> dict:
        """mock"""
        calls.append(("issue_refund", order_id))
        return refund

    @tool
    def search_kb(query: str) -> dict:
        """mock"""
        calls.append(("search_kb", query))
        return kb

    return [lookup_order, check_refund_policy, issue_refund, search_kb], calls


def _ask(graph, text, messages=None):
    messages = list(messages or []) + [{"role": "user", "content": text}]
    return graph.invoke({"messages": messages})


def _tool_calls(state):
    return [tc["name"] for m in state["messages"] if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]


def test_order_status_routes_to_support_and_looks_up():
    tools, calls = _mock_tools()
    out = _ask(sb.build_agent(tools=tools), "Where is my order A1234?")
    assert out["route"] == "order_status"
    assert calls == [("lookup_order", "A1234")]
    assert "A1234" in out["messages"][-1].content and "delivered" in out["messages"][-1].content


def test_refund_flow_gates_issue_refund_on_confirmation():
    tools, calls = _mock_tools()
    graph = sb.build_agent(tools=tools)
    out = _ask(graph, "I want a refund for order A1234, the headphones are broken")
    assert [c[0] for c in calls] == ["lookup_order", "check_refund_policy"]
    assert "yes" in out["messages"][-1].content.lower()
    out2 = _ask(graph, "yes", out["messages"])
    assert calls[-1] == ("issue_refund", "A1234")
    assert "R77" in out2["messages"][-1].content


def test_product_question_routes_to_kb():
    tools, calls = _mock_tools()
    out = _ask(sb.build_agent(tools=tools), "How do I pair the X1 headset?")
    assert out["route"] == "product_question"
    assert calls and calls[0][0] == "search_kb"
    assert "Pairing the X1 headset" in out["messages"][-1].content


def test_out_of_scope_declines_without_tools():
    tools, calls = _mock_tools()
    out = _ask(sb.build_agent(tools=tools), "Write me a poem about the sea")
    assert out["route"] == "other" and not calls
    assert out["messages"][-1].content == sb.DECLINE_TEXT


def test_real_tools_fail_fast_when_unmocked():
    out = _ask(sb.build_agent(), "Where is my order A1234?")
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert tool_msgs and "unavailable" in tool_msgs[0].content
    assert "could not reach" in out["messages"][-1].content.lower()


def test_tools_and_contract():
    assert [t.name for t in sb.TOOLS] == ["lookup_order", "check_refund_policy", "issue_refund", "search_kb"]
    assert "Side-effecting" in sb.issue_refund.description
