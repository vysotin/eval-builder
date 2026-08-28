"""Incident desk — an on-call incident copilot: a router graph over four ReAct specialists.

Target contract for evalbuilder: exposes TOOLS and build_agent(model=None, tools=None).

Graph:  START → classify ─┬─ incident ────────► triage_agent ► remediation_agent ► comms_agent ► END
                          ├─ status_question ─► status_agent ► END
                          └─ other ───────────► decline ► END

Every tool talks to an ops host that does not exist (`ops.example.invalid`), so an unmocked
run fails fast with an error payload instead of silently succeeding — the pipeline must mock
every tool. Tools declare pydantic input/output models so the schema-derived edge cases
(missing/invalid input, malformed output) are exercised; `create_ticket`, `page_oncall` and
`post_status_update` are side-effecting and gated on an explicit user confirmation.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

try:
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - older langchain/langgraph installs
    from langgraph.prebuilt import create_react_agent

    def create_agent(model, tools, system_prompt=None, **kwargs):
        return create_react_agent(model, tools, prompt=system_prompt, **kwargs)


from evalbuilder.testing import ScriptedChatModel, ai, tool_call

OPS_API = "http://ops.example.invalid"
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


def _payload(url: str, payload: dict | None = None) -> dict:
    """`_http` for tools that build a pydantic result: an error payload becomes a ToolException
    (surfaced as the ToolMessage text via `handle_tool_error`) instead of a validation crash."""
    data = _http(url, payload)
    if "error" in data:
        raise ToolException(str(data["error"]))
    return data


# ── schemas ─────────────────────────────────────────────────────


class ServiceStatus(BaseModel):
    """Current health of one service as reported by the status API."""

    service: str
    state: Literal["healthy", "degraded", "down"]
    error_rate_pct: float = Field(ge=0, le=100, description="5-minute error rate in percent")
    open_incidents: int = Field(ge=0)
    last_deploy: str


class RunbookQuery(BaseModel):
    query: str = Field(min_length=3, description="Symptom or service to search runbooks for")
    severity: Literal["sev1", "sev2", "sev3"] = "sev2"
    limit: int = Field(3, ge=1, le=10)


class TicketRequest(BaseModel):
    title: str = Field(min_length=5, max_length=120)
    service: str
    severity: Literal["sev1", "sev2", "sev3"]
    summary: str


class TicketReceipt(BaseModel):
    ticket_id: str = Field(pattern=r"^INC-\d{4,}$")
    url: str
    status: Literal["open", "acknowledged"]


class StatusUpdate(BaseModel):
    channel: Literal["status-page", "slack"]
    message: str = Field(min_length=10, max_length=500)
    severity: Literal["sev1", "sev2", "sev3"]


# ── tools ───────────────────────────────────────────────────────


@tool
def get_service_status(service: str) -> ServiceStatus:
    """Current health of a service: state, error rate, open incidents, last deploy (external status API)."""
    return ServiceStatus.model_validate(_payload(f"{OPS_API}/status/{urllib.parse.quote(service)}"))


@tool(args_schema=RunbookQuery)
def search_runbooks(query: str, severity: str = "sev2", limit: int = 3) -> dict:
    """Search the runbook library for remediation steps matching a symptom or service."""
    return _http(
        f"{OPS_API}/runbooks/search", {"query": query, "severity": severity, "limit": limit}
    )


@tool
def create_ticket(ticket: TicketRequest) -> TicketReceipt:
    """Open an incident ticket. Side-effecting: only call after the user explicitly confirms."""
    return TicketReceipt.model_validate(_payload(f"{OPS_API}/tickets", ticket.model_dump()))


@tool
def page_oncall(team: str, severity: Literal["sev1", "sev2", "sev3"], reason: str) -> dict:
    """Page the on-call engineer of a team. Side-effecting: page a human only after the user explicitly confirms."""
    return _http(f"{OPS_API}/pages", {"team": team, "severity": severity, "reason": reason})


@tool
def post_status_update(update: StatusUpdate) -> dict:
    """Publish an incident update to the status page or Slack. Side-effecting: post only after the user explicitly confirms."""
    return _http(f"{OPS_API}/updates", update.model_dump())


# Error payloads from pydantic-returning tools become the ToolMessage text, not a crash.
get_service_status.handle_tool_error = True
create_ticket.handle_tool_error = True

TOOLS = [get_service_status, search_runbooks, create_ticket, page_oncall, post_status_update]

# ── prompts ─────────────────────────────────────────────────────

CLASSIFY_PROMPT = (
    "Classify the on-call engineer's latest message in the context of the conversation. "
    "incident: a service is down, degraded, erroring, or the user wants a ticket, a page, "
    "a runbook, or a status update — including confirmations ('yes') that continue an "
    "incident conversation. status_question: asking how a service is doing right now. "
    "other: anything else."
)

TRIAGE_PROMPT = (
    "You are the triage specialist of the incident desk. Always call get_service_status "
    "for the affected service before anything else, then call search_runbooks with the "
    "reported severity to find remediation steps. If the user did not name the service, "
    "ask which service is affected instead of guessing. Severity must be one of sev1, "
    "sev2 or sev3: if it is missing or anything else (critical, P0, sev9), offer those "
    "three values and stop. If a tool errors or returns an incomplete payload, say so "
    "plainly and never invent status values. Summarize the triage in one message."
)

REMEDIATION_PROMPT = (
    "You are the remediation specialist of the incident desk. Based on the triage, propose "
    "one ticket (title, service, severity) and ask the user for an explicit yes. Call "
    "create_ticket and then page_oncall ONLY after the user explicitly confirms with yes "
    "in this conversation — never on the first request, however urgent. Use "
    "search_runbooks if the triage found no runbook. If create_ticket errors or returns "
    "an incomplete receipt (no ticket id), report that, do not page, and never invent a "
    "ticket id."
)

COMMS_PROMPT = (
    "You are the communications specialist of the incident desk. Call post_status_update "
    "only after a ticket has been created and on-call paged in this conversation; before "
    "that, restate what is pending and post nothing. Mention the ticket id in the final "
    "message. If the update fails, say the status page could not be reached."
)

STATUS_PROMPT = (
    "You are the status specialist of the incident desk. Always call get_service_status "
    "for the named service and answer with its state, error rate, open incidents and last "
    "deploy. If no service was named, ask which service to check. If the tool errors or "
    "returns an incomplete payload, say so and never guess the state."
)

DECLINE_TEXT = (
    "I can help with production incidents and service status for the ops team. "
    "That request is outside what I can do here."
)


class Route(BaseModel):
    intent: Literal["incident", "status_question", "other"]


class IncidentState(MessagesState):
    route: str


ROUTES = {
    "incident": "triage_agent",
    "status_question": "status_agent",
    "other": "decline",
}


def build_agent(model=None, tools=None):
    """Build the compiled incident-desk graph."""
    model = model or default_scripted_model()
    tools = TOOLS if tools is None else tools
    by_name = {t.name: t for t in tools}

    def subset(*names: str) -> list:
        return [by_name[n] for n in names if n in by_name]

    triage_agent = create_agent(
        model, subset("get_service_status", "search_runbooks"), system_prompt=TRIAGE_PROMPT
    )
    remediation_agent = create_agent(
        model, subset("search_runbooks", "create_ticket", "page_oncall"), system_prompt=REMEDIATION_PROMPT
    )
    comms_agent = create_agent(model, subset("post_status_update"), system_prompt=COMMS_PROMPT)
    status_agent = create_agent(model, subset("get_service_status"), system_prompt=STATUS_PROMPT)
    classifier = model.with_structured_output(Route)

    def classify(state: IncidentState) -> dict:
        # User turns plus the specialists' replies (tool-call stubs have empty content).
        recent = [m for m in state["messages"] if m.type == "human" or (m.type == "ai" and m.content)][-10:]
        route = classifier.invoke([SystemMessage(content=CLASSIFY_PROMPT), *recent])
        return {"route": route.intent}

    def decline(state: IncidentState) -> dict:
        return {"messages": [AIMessage(content=DECLINE_TEXT)]}

    def pick(state: IncidentState) -> str:
        return ROUTES.get(state.get("route", "other"), "decline")

    graph = StateGraph(IncidentState)
    graph.add_node("classify", classify)
    graph.add_node("triage_agent", triage_agent)
    graph.add_node("remediation_agent", remediation_agent)
    graph.add_node("comms_agent", comms_agent)
    graph.add_node("status_agent", status_agent)
    graph.add_node("decline", decline)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        pick,
        {"triage_agent": "triage_agent", "status_agent": "status_agent", "decline": "decline"},
    )
    graph.add_edge("triage_agent", "remediation_agent")
    graph.add_edge("remediation_agent", "comms_agent")
    graph.add_edge("comms_agent", END)
    graph.add_edge("status_agent", END)
    graph.add_edge("decline", END)
    return graph.compile()


# ── scripted default model (offline tests) ──────────────────────
#
# One script serves every specialist; factories tell the nodes apart by the system prompt
# at messages[0]. Tool results are read from JSON (mocked fixtures) or from the pydantic
# repr (`service='api' state='degraded' …`) that a live pydantic-returning tool produces.

KNOWN_SERVICES = ("api", "checkout", "payments", "auth", "search", "database", "gateway", "billing", "web", "notifications")
_SERVICE_KNOWN = re.compile(r"\b(" + "|".join(KNOWN_SERVICES) + r")\b", re.IGNORECASE)
_SERVICE_PHRASE = re.compile(r"\b([a-z][a-z0-9_-]{1,30})[ -]service\b|\bservice[:\s]+(?:named\s+)?['\"]?([a-z][a-z0-9_-]{1,30})", re.IGNORECASE)
_SERVICE_STOPWORDS = {"is", "are", "was", "were", "the", "a", "an", "that", "which", "name", "named", "went", "seems", "has", "status", "this", "it", "to", "be", "our", "your", "affected"}
_SEVERITY = re.compile(r"\b(?:sev|severity)\s*-?\s*(\d+)\b", re.IGNORECASE)
_SEVERITY_WORDS = re.compile(r"\b(critical|urgent|blocker|emergency|p[0-4]|high|medium|low|minor|major)\b", re.IGNORECASE)
_CONFIRM = re.compile(r"^\s*(yes|yep|yeah|confirm(ed)?|go ahead|do it|proceed|approved?)\b", re.IGNORECASE)
_REPR_FIELDS = re.compile(r"(\w+)=('[^']*'|[^\s']+)")
_REPR_LINE = re.compile(r"(\s*\w+=(?:'[^']*'|[^\s']+))+\s*")
_NODE_BY_PROMPT = {
    TRIAGE_PROMPT: "triage",
    REMEDIATION_PROMPT: "remediation",
    COMMS_PROMPT: "comms",
    STATUS_PROMPT: "status",
}


def _node(messages) -> str:
    first = messages[0] if messages else None
    return _NODE_BY_PROMPT.get(str(first.content), "") if isinstance(first, SystemMessage) else ""


def _human_texts(messages) -> list[str]:
    return [str(m.content) for m in messages if isinstance(m, HumanMessage)]


def _service_in(text: str) -> str | None:
    for phrase in _SERVICE_PHRASE.finditer(text):
        name = (phrase.group(1) or phrase.group(2) or "").lower()
        if name and name not in _SERVICE_STOPWORDS:
            return name
    known = _SERVICE_KNOWN.search(text)
    return known.group(1).lower() if known else None


def _service(messages) -> str | None:
    """The affected service: the latest user message first, then the whole transcript."""
    texts = _human_texts(messages)
    for text in ([texts[-1]] if texts else []) + ["\n".join(texts)]:
        found = _service_in(text)
        if found:
            return found
    return None


def _severity(messages) -> tuple[str | None, str | None]:
    """(valid severity, invalid token): `sev2` → ("sev2", None); `sev9`/`critical` → (None, token)."""
    texts = _human_texts(messages)
    for text in ([texts[-1]] if texts else []) + ["\n".join(texts)]:
        levels = _SEVERITY.findall(text)
        valid = [n for n in levels if n in ("1", "2", "3")]
        if valid:
            return f"sev{valid[0]}", None
        if levels:
            return None, f"sev{levels[0]}"
        word = _SEVERITY_WORDS.search(text)
        if word:
            return None, word.group(1)
    return None, None


def _confirmed(messages) -> bool:
    """The latest user turn is a yes that answers an earlier ticket proposal."""
    texts = _human_texts(messages)
    if not texts or not _CONFIRM.search(texts[-1]):
        return False
    return any(isinstance(m, AIMessage) and "Reply yes" in str(m.content) for m in messages)


def _tool_fields(content: str) -> dict:
    """Fields of a tool result: JSON (mocked fixture / dict tool), pydantic repr (live
    model-returning tool), or `{"error": text}` for anything else (ToolException text)."""
    try:
        data = json.loads(content)
    except ValueError:
        data = None
    if isinstance(data, dict):
        return data
    if data is None and _REPR_LINE.fullmatch(content):
        return {k: v.strip("'") for k, v in _REPR_FIELDS.findall(content)}
    return {"error": content}


def _last_tool_fields(messages, name: str) -> dict:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == name:
            return _tool_fields(str(m.content))
    return {}


def _echo(messages) -> AIMessage:
    """A specialist with nothing to add restates the previous specialist's answer."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and str(m.content).strip():
            return ai(str(m.content))
    return ai("How can I help with this incident?")


def _ask_service(node: str) -> AIMessage:
    if node == "status":
        return ai("Which service do you want me to check? Please name it and I will look up its status.")
    return ai("Which service is affected? Please name the service so I can check its status first.")


def _ticket_title(service: str, severity: str) -> str:
    return f"{service} incident ({severity})"


def _after_status(fields: dict, messages, node: str) -> AIMessage:
    service = fields.get("service") or _service(messages) or "the service"
    missing = [f for f in ("service", "state") if f not in fields]
    if missing:
        return ai(
            f"The status check for {service} came back incomplete (missing {', '.join(missing)}), "
            "so I cannot report its health. Please retry in a moment."
        )
    state = fields["state"]
    if node == "status":
        return ai(
            f"{service} is currently {state}: error rate {fields.get('error_rate_pct', '?')}%, "
            f"{fields.get('open_incidents', '?')} open incident(s), last deploy {fields.get('last_deploy', 'unknown')}."
        )
    severity, invalid = _severity(messages)
    if invalid:
        return ai(
            f"Severity must be one of sev1, sev2 or sev3 — '{invalid}' is not a level I can use. "
            f"Which one applies to {service}?"
        )
    if not severity:
        return ai(f"{service} is {state} ({fields.get('error_rate_pct', '?')}% errors). Which severity applies: sev1, sev2 or sev3?")
    return tool_call("search_runbooks", {"query": f"{service} {state}", "severity": severity, "limit": 3})


def _after_runbooks(fields: dict, messages) -> AIMessage:
    status = _last_tool_fields(messages, "get_service_status")
    service = status.get("service") or _service(messages) or "the service"
    head = (
        f"Triage: {service} is {status.get('state', 'unknown')} with a {status.get('error_rate_pct', '?')}% "
        f"error rate and {status.get('open_incidents', '?')} open incident(s)."
    )
    runbooks = fields.get("runbooks") or []
    if not runbooks:
        return ai(head + " No matching runbook was found.")
    first = runbooks[0]
    return ai(head + f" Runbook '{first.get('title')}': {'; '.join(first.get('steps') or [])}.")


def _after_ticket(fields: dict, messages) -> AIMessage:
    if "ticket_id" not in fields:
        return ai(
            "The ticket system returned an incomplete receipt (no ticket id), so I have not paged "
            "on-call. Please retry the ticket in a moment."
        )
    service = _service(messages) or "service"
    severity = _severity(messages)[0] or "sev2"
    return tool_call(
        "page_oncall",
        {"team": f"{service}-oncall", "severity": severity, "reason": f"{fields['ticket_id']}: {service} {severity} incident"},
    )


def _after_page(fields: dict, messages) -> AIMessage:
    ticket = _last_tool_fields(messages, "create_ticket")
    service = _service(messages) or "service"
    severity = _severity(messages)[0] or "sev2"
    return ai(f"Created ticket {ticket.get('ticket_id')} and paged {service}-oncall on-call ({severity}).")


def _after_update(fields: dict, messages) -> AIMessage:
    ticket = _last_tool_fields(messages, "create_ticket")
    service = _service(messages) or "service"
    severity = _severity(messages)[0] or "sev2"
    return ai(
        f"Created ticket {ticket.get('ticket_id')}, paged {service}-oncall on-call and posted a "
        f"{severity} update to {fields.get('channel', 'status-page')}."
    )


def _after_tool(match, messages) -> AIMessage:
    last = messages[-1]
    fields = _tool_fields(str(last.content))
    if "error" in fields:
        return ai(
            f"Sorry — I could not reach the ops system ({last.name} failed), so I have not changed "
            "anything. Please try again in a few minutes."
        )
    handlers = {
        "get_service_status": lambda: _after_status(fields, messages, _node(messages)),
        "search_runbooks": lambda: _after_runbooks(fields, messages),
        "create_ticket": lambda: _after_ticket(fields, messages),
        "page_oncall": lambda: _after_page(fields, messages),
        "post_status_update": lambda: _after_update(fields, messages),
    }
    return handlers.get(last.name, lambda: ai("Done."))()


def _remediate(match, messages) -> AIMessage:
    """Remediation acts on the triage summary: propose, or create + page once confirmed."""
    if _node(messages) != "remediation":
        return _echo(messages)
    service = _service(messages) or "service"
    severity = _severity(messages)[0] or "sev2"
    title = _ticket_title(service, severity)
    if _confirmed(messages):
        summary = _human_texts(messages)[0][:200]
        return tool_call(
            "create_ticket",
            {"ticket": {"title": title, "service": service, "severity": severity, "summary": summary}},
        )
    return ai(f"Proposed {severity} ticket for {service}: '{title}'. Reply yes to create the ticket and page on-call.")


def _announce(match, messages) -> AIMessage:
    """Comms posts the status update once remediation reports the ticket and page."""
    if _node(messages) != "comms":
        return _echo(messages)
    service = _service(messages) or "service"
    severity = _severity(messages)[0] or "sev2"
    ticket_id = _last_tool_fields(messages, "create_ticket").get("ticket_id") or match.group(1)
    return tool_call(
        "post_status_update",
        {
            "update": {
                "channel": "status-page",
                "message": f"We are investigating a {severity} incident on {service}. Ticket {ticket_id} is open and on-call has been paged.",
                "severity": severity,
            }
        },
    )


def _turn(match, messages) -> AIMessage:
    """Start of a specialist's turn on the latest user message (or the previous specialist's text)."""
    node = _node(messages)
    if node in ("triage", "status"):
        service = _service(messages)
        if not service:
            return _ask_service(node)
        return tool_call("get_service_status", {"service": service})
    if node in ("remediation", "comms"):
        return _echo(messages)
    return ai(DECLINE_TEXT)


def default_scripted_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        structured_script=[
            (r"\bstatus update\b|\bstatus page\b|\bpost\b.*\bstatus\b", lambda m, _msgs: {"intent": "incident"}),
            (
                r"\bstatus\s+(of|for)\b|what('s| is) the status|\bhealth(y)?\b.*\?|\bis [\w-]+ (up|down|ok|healthy)\b.*\?|how (is|are) [\w -]+ (doing|looking)",
                lambda m, _msgs: {"intent": "status_question"},
            ),
            (
                r"\b(down|outage|incident|broken|failing|fails?|degraded|errors?|5\d\d|latency|timeouts?|timing out|ticket|page|on-?call|alert|crash(ed|ing)?|unavailable|runbook)\b|\bsev\s*-?\s*\d",
                lambda m, _msgs: {"intent": "incident"},
            ),
            (r".*", lambda m, _msgs: {"intent": "other"}),
        ],
        script=[
            (r"^TOOL:", _after_tool),
            (r"^Triage:", _remediate),
            (r"^Created ticket (\S+) and paged", _announce),
            (r".*", _turn),
        ],
    )
