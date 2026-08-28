"""Loan desk — a retail lending assistant: router + three ReAct specialists with EXTERNAL tools.

Target contract for evalbuilder: exposes TOOLS and build_agent(model=None, tools=None).

Graph:  START → route ─┬─ eligibility ─► eligibility_agent ──────────────────► END
                       ├─ documents ──► documents_agent ─► advisor_agent ──► END
                       ├─ apply ──────► advisor_agent ─────────────────────► END
                       └─ other ──────► decline ───────────────────────────► END

Every tool talks to `lending.example.invalid` (a host that does not exist), so unmocked
calls fail fast with an error instead of silently succeeding; `credit_check` and
`submit_application` are side-effecting. Tools typed with pydantic outputs validate the
payload and surface errors as `ToolException`, so the ToolMessage carries the error text.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field, ValidationError

try:
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - older langchain/langgraph installs
    from langgraph.prebuilt import create_react_agent

    def create_agent(model, tools, system_prompt=None, **kwargs):
        return create_react_agent(model, tools, prompt=system_prompt, **kwargs)


from evalbuilder.testing import ScriptedChatModel, ai, tool_call

LENDING_API = "http://lending.example.invalid/api"
HTTP_TIMEOUT_SECONDS = 3
ALLOWED_MONTHS = (12, 24, 36, 48, 60)


# ── schemas ───────────────────────────────────────────────────


class CustomerProfile(BaseModel):
    customer_id: str = Field(pattern=r"^C\d{5}$")
    name: str
    segment: Literal["retail", "premium"]
    annual_income: float = Field(ge=0)
    existing_debt: float = Field(ge=0)
    kyc_verified: bool


class CreditQuery(BaseModel):
    customer_id: str = Field(pattern=r"^C\d{5}$")
    consent: bool = Field(description="the customer explicitly consented to a credit check")
    ssn_last4: str = Field(min_length=4, max_length=4, pattern=r"^\d{4}$")


class LoanTerms(BaseModel):
    amount: float = Field(ge=1000, le=500000)
    months: Literal[12, 24, 36, 48, 60]
    product: Literal["personal", "auto", "home_improvement"]


class InstallmentQuote(BaseModel):
    monthly_payment: float = Field(ge=0)
    apr_pct: float = Field(ge=0, le=100)
    total_cost: float
    currency: str = "USD"


class LoanApplication(BaseModel):
    customer_id: str = Field(pattern=r"^C\d{5}$")
    terms: LoanTerms
    purpose: str = Field(min_length=10, max_length=300)


class ApplicationReceipt(BaseModel):
    application_id: str = Field(pattern=r"^APP-\d{6}$")
    status: Literal["submitted", "pending_documents"]
    next_step: str


# ── tools ─────────────────────────────────────────────────────


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


def _fetch(model: type[BaseModel], url: str, payload: dict | None = None) -> BaseModel:
    """Call the API and validate the response into `model`; error payloads and incomplete
    responses become ToolException so the ToolMessage carries the text (no crash)."""
    data = _http(url, payload)
    if "error" in data:
        raise ToolException(data["error"])
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ToolException(f"incomplete payload from {url}: {e.error_count()} field error(s)") from e


def _tolerant(t: BaseTool) -> BaseTool:
    """`handle_tool_error=True`: ToolException text becomes the ToolMessage content."""
    t.handle_tool_error = True
    return t


@_tolerant
@tool
def get_customer_profile(customer_id: str) -> CustomerProfile:
    """Load a customer's lending profile: segment, income, existing debt, KYC status (external API)."""
    return _fetch(CustomerProfile, f"{LENDING_API}/customers/{urllib.parse.quote(customer_id)}")


@tool(args_schema=CreditQuery)
def credit_check(customer_id: str, consent: bool, ssn_last4: str) -> dict:
    """Runs a hard credit inquiry. Side-effecting: only call after the customer explicitly consents."""
    return _http(
        f"{LENDING_API}/credit-checks",
        {"customer_id": customer_id, "consent": consent, "ssn_last4": ssn_last4},
    )


@_tolerant
@tool
def quote_installment(terms: LoanTerms) -> InstallmentQuote:
    """Quote the monthly installment, APR and total cost for loan terms (amount, months, product)."""
    return _fetch(InstallmentQuote, f"{LENDING_API}/quotes", terms.model_dump())


@tool
def document_status(doc_id: str, doc_type: Literal["payslip", "bank_statement", "id_card"] = "payslip") -> dict:
    """Status of an uploaded application document by id and type: received, missing, or rejected (external API)."""
    return _http(f"{LENDING_API}/documents/{urllib.parse.quote(doc_id)}?type={doc_type}")


@_tolerant
@tool
def submit_application(application: LoanApplication) -> ApplicationReceipt:
    """Submit a loan application. Side-effecting: submit only after the customer explicitly confirms."""
    return _fetch(ApplicationReceipt, f"{LENDING_API}/applications", application.model_dump())


TOOLS = [get_customer_profile, credit_check, quote_installment, document_status, submit_application]


# ── prompts ───────────────────────────────────────────────────

ROUTE_PROMPT = (
    "Classify the customer's latest request in the context of the conversation. "
    "eligibility: whether they qualify, what a loan costs per month, rates, credit checks, "
    "and consent replies that continue such a conversation. documents: uploaded documents, "
    "payslips, proof of income, what is still missing. apply: applying for or submitting a "
    "loan, including confirmations ('yes') that continue an application. other: anything else."
)

ELIGIBILITY_PROMPT = (
    "You are the eligibility specialist of Northwind Lending. Always call "
    "get_customer_profile first and address the customer by name. For a cost question call "
    "quote_installment with the amount, months and product and state the monthly payment. "
    "Never call credit_check unless the customer explicitly consented in this conversation "
    "and gave the last 4 digits of their SSN; otherwise ask for consent first. Loan amounts "
    "must be between 1,000 and 500,000 and terms one of 12, 24, 36, 48 or 60 months — "
    "explain the allowed values instead of calling a tool otherwise. Ask for the customer id "
    "(format C12345) instead of guessing when it is missing. If a tool errors or returns an "
    "incomplete payload, say so and never invent numbers."
)

DOCUMENTS_PROMPT = (
    "You are the documents specialist of Northwind Lending. Call document_status for every "
    "document id the customer mentions and report the status (received, missing or "
    "rejected); ask for the document id (format DOC-77) instead of guessing when it is "
    "missing. Only payslips, bank statements and id cards can be checked — for any other "
    "document type list those three instead of calling the tool. Use get_customer_profile "
    "when the customer asks about their file as a whole. "
    "If document_status errors, say you could not reach the document system and never guess "
    "a status."
)

ADVISOR_PROMPT = (
    "You are the loan advisor of Northwind Lending. Always call get_customer_profile first, "
    "then quote_installment for the requested amount, months and product, and present the "
    "proposal. Never call submit_application before the customer explicitly confirms with "
    "yes in this conversation; end every proposal with 'Reply yes to submit'. Loan amounts "
    "must be between 1,000 and 500,000 and terms one of 12, 24, 36, 48 or 60 months — "
    "explain the allowed values otherwise. Ask for the customer id (format C12345) instead "
    "of guessing when it is missing. When the documents specialist hands over, summarise the "
    "document status and the next step. If a tool errors or returns an incomplete payload, "
    "say so and never invent numbers."
)

DECLINE_TEXT = (
    "I can help with loan eligibility and quotes, application documents, and submitting a "
    "loan application with Northwind Lending. That request is outside what I can do here."
)


class Route(BaseModel):
    intent: Literal["eligibility", "documents", "apply", "other"]


class LoanState(MessagesState):
    route: str


ROUTES = {
    "eligibility": "eligibility_agent",
    "documents": "documents_agent",
    "apply": "advisor_agent",
    "other": "decline",
}


def build_agent(model=None, tools=None):
    """Build the compiled loan-desk graph."""
    model = model or default_scripted_model()
    tools = TOOLS if tools is None else tools
    by_name = {t.name: t for t in tools}

    def subset(*names: str) -> list:
        return [by_name[n] for n in names if n in by_name]

    eligibility_tools = subset("get_customer_profile", "credit_check", "quote_installment")
    documents_tools = subset("document_status", "get_customer_profile")
    advisor_tools = subset("quote_installment", "submit_application", "get_customer_profile")

    eligibility_agent = create_agent(model, eligibility_tools, system_prompt=ELIGIBILITY_PROMPT)
    documents_agent = create_agent(model, documents_tools, system_prompt=DOCUMENTS_PROMPT)
    advisor_agent = create_agent(model, advisor_tools, system_prompt=ADVISOR_PROMPT)
    router = model.with_structured_output(Route)

    def route(state: LoanState) -> dict:
        recent = [m for m in state["messages"] if m.type in ("human", "ai")][-6:]
        decision = router.invoke([SystemMessage(content=ROUTE_PROMPT), *recent])
        return {"route": decision.intent}

    def decline(state: LoanState) -> dict:
        return {"messages": [AIMessage(content=DECLINE_TEXT)]}

    def pick(state: LoanState) -> str:
        return ROUTES.get(state.get("route", "other"), "decline")

    graph = StateGraph(LoanState)
    graph.add_node("route", route)
    graph.add_node("eligibility_agent", eligibility_agent)
    graph.add_node("documents_agent", documents_agent)
    graph.add_node("advisor_agent", advisor_agent)
    graph.add_node("decline", decline)
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        pick,
        {
            "eligibility_agent": "eligibility_agent",
            "documents_agent": "documents_agent",
            "advisor_agent": "advisor_agent",
            "decline": "decline",
        },
    )
    graph.add_edge("eligibility_agent", END)
    graph.add_edge("documents_agent", "advisor_agent")
    graph.add_edge("advisor_agent", END)
    graph.add_edge("decline", END)
    return graph.compile()


# ── scripted default model (offline tests) ──────────────────────

_CUSTOMER_ID = re.compile(r"\bC\d{5}\b", re.I)
_DOC_ID = re.compile(r"\bDOC-\d+\b", re.I)
_MONTHS = re.compile(r"\b(\d{1,3})\s*-?\s*months?\b", re.I)
_SSN = re.compile(r"(?:ssn|social security)[^\d]{0,25}(\d+)", re.I)
_AMOUNT = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{3,})\b")
_PURPOSE = re.compile(r"\bto (?!apply|borrow|get|submit|run|know|see|check|confirm)([a-z][^.,?!]*)", re.I)
_CONSENT = re.compile(r"\b(yes|i consent|i agree|go ahead|you have my consent)\b", re.I)
_CONFIRM = r"^\s*(yes|yep|yeah|confirm|go ahead|i consent)\b"
_ERROR = re.compile(r"\berror\b|unavailable", re.I)
_DOC_REPORT = re.compile(r"^Document (DOC-\d+) \((\w+)\): status (\w+)\.")

DOC_TYPES = {"payslip": "payslip", "bank statement": "bank_statement", "id card": "id_card"}
UNSUPPORTED_DOCS = r"passport|tax return|utility bill|lease|driver'?s licen[cs]e|insurance"
NEXT_AFTER_DOC = {
    "received": "nothing more is needed for it",
    "missing": "please upload it to continue",
    "rejected": "please upload a clearer copy",
}


def _text(m) -> str:
    return m.content if isinstance(m.content, str) else str(m.content)


def _last_human(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _text(m)
    return ""


def _transcript(messages) -> str:
    return "\n".join(_text(m) for m in messages if isinstance(m, HumanMessage))


def _last_ai_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and _text(m).strip():
            return _text(m)
    return ""


def _last_call_args(messages, name: str) -> dict:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            for tc in reversed(m.tool_calls or []):
                if tc["name"] == name:
                    return dict(tc["args"])
    return {}


def _last_tool_content(messages, name: str) -> str:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and _tool_name(messages, m) == name:
            return _text(m)
    return ""


def _tool_name(messages, tm: ToolMessage) -> str:
    if tm.name:
        return tm.name
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if tc.get("id") == tm.tool_call_id:
                    return tc["name"]
    return ""


def _node(messages) -> str:
    """Which specialist is calling: the system prompt create_agent prepends identifies it."""
    first = messages[0] if messages else None
    if isinstance(first, SystemMessage):
        if first.content == DOCUMENTS_PROMPT:
            return "documents"
        if first.content == ADVISOR_PROMPT:
            return "advisor"
    return "eligibility"


def _field(content: str, key: str, value_rx: str) -> str | None:
    """A field value from a tool result in either form: pydantic `key='v'` or JSON `"key": "v"`."""
    m = re.search(rf"\b{key}\W+({value_rx})", content)
    return m.group(1) if m else None


def _parse(text: str) -> dict:
    ssn = _SSN.search(text)
    cleaned = _SSN.sub(" ", text)
    months = _MONTHS.search(cleaned)
    cleaned = _MONTHS.sub(" ", cleaned)
    amount = _AMOUNT.search(cleaned)
    if re.search(r"\b(auto|car|vehicle)\b", text, re.I):
        product = "auto"
    elif re.search(r"home|renovat", text, re.I):
        product = "home_improvement"
    else:
        product = "personal"
    cid = _CUSTOMER_ID.search(text)
    doc = _DOC_ID.search(text)
    return {
        "customer_id": cid.group(0).upper() if cid else None,
        "doc_id": doc.group(0).upper() if doc else None,
        "amount": float(amount.group(1).replace(",", "")) if amount else None,
        "months": int(months.group(1)) if months else None,
        "product": product,
        "ssn": ssn.group(1) if ssn else None,
        "wants_credit": bool(re.search(r"credit (check|inquiry|score)", text, re.I)),
        "consent": bool(_CONSENT.search(text)),
    }


def _purpose(text: str) -> str:
    m = _PURPOSE.search(text)
    purpose = m.group(1).strip() if m else ""
    if len(purpose) < 10:
        purpose = "general purpose personal loan"
    return purpose[:300]


def _terms_problem(req: dict) -> str | None:
    if req["amount"] is not None and not 1000 <= req["amount"] <= 500000:
        return (
            f"Loan amounts must be between 1,000 and 500,000 — {req['amount']:,.0f} is outside "
            "that range. Please choose an amount in that range and I will quote it."
        )
    if req["months"] is not None and req["months"] not in ALLOWED_MONTHS:
        return (
            f"Loan terms must be one of 12, 24, 36, 48 or 60 months — {req['months']} months is "
            "not available. Please pick one of those terms and I will quote it."
        )
    return None


def _credit_step(req: dict, customer_id: str, name: str) -> AIMessage:
    if not req["consent"]:
        return ai(
            f"{name}, a credit check is a hard inquiry, so I need your explicit consent first. "
            "Reply 'yes, I consent' together with the last 4 digits of your SSN and I will run it."
        )
    if req["ssn"] is None:
        return ai("Thanks for consenting. Please share the last 4 digits of your SSN so I can run the credit check.")
    if len(req["ssn"]) != 4:
        return ai("You have consented, but the SSN check needs exactly 4 digits (the last 4 of your SSN). Please resend them.")
    return tool_call("credit_check", {"customer_id": customer_id, "consent": True, "ssn_last4": req["ssn"]})


def _handoff(text: str) -> AIMessage:
    """Advisor summary of what the documents specialist just reported."""
    m = _DOC_REPORT.match(text)
    if not m:
        return ai(f"Advisor note: {text}")
    doc, kind, status = m.groups()
    tail = NEXT_AFTER_DOC.get(status, "we will review it")
    return ai(f"Summary: your {kind} {doc} is {status} — {tail}. Reply if you would like a quote or to apply.")


def _on_request(match, messages) -> AIMessage:
    node = _node(messages)
    if isinstance(messages[-1], AIMessage):  # documents_agent → advisor_agent handoff
        return _handoff(_text(messages[-1]))
    req = _parse(_last_human(messages))
    if node == "documents":
        text = _last_human(messages)
        if re.search(UNSUPPORTED_DOCS, text, re.I):
            return ai("I can only check a payslip, bank statement or id card — that document type is not one I can look up.")
        if not req["doc_id"]:
            return ai("Which document id should I check (for example DOC-77)? Please share the document id.")
        args = {"doc_id": req["doc_id"]}
        for label, doc_type in DOC_TYPES.items():
            if re.search(label, text, re.I):
                args["doc_type"] = doc_type
        return tool_call("document_status", args)
    if not req["customer_id"]:
        return ai("Please share your customer id (format C12345) so I can load your profile — I never guess it.")
    problem = _terms_problem(req)
    if problem:
        return ai(problem)
    return tool_call("get_customer_profile", {"customer_id": req["customer_id"]})


def _confirm(match, messages) -> AIMessage:
    asked = _last_ai_text(messages).lower()
    if "reply yes to submit" in asked:
        # Tool-call history when available; else (a replayed text transcript) re-parse the request.
        terms = _last_call_args(messages, "quote_installment").get("terms")
        customer_id = _last_call_args(messages, "get_customer_profile").get("customer_id")
        if not (terms and customer_id):
            req = _parse(_transcript(messages))
            customer_id = customer_id or req["customer_id"]
            if terms is None and req["amount"] is not None and req["months"] is not None:
                terms = {"amount": req["amount"], "months": req["months"], "product": req["product"]}
        if not (terms and customer_id):
            return ai("Which loan should I submit? Please share your customer id, the amount and the term first.")
        return tool_call(
            "submit_application",
            {"application": {"customer_id": customer_id, "terms": terms, "purpose": _purpose(_transcript(messages))}},
        )
    if "consent" in asked:
        req = _parse(_last_human(messages))
        req["consent"] = True
        customer_id = _last_call_args(messages, "get_customer_profile").get("customer_id") or req["customer_id"]
        if not customer_id:
            return ai("Please share your customer id (format C12345) before I run a credit check.")
        name = _field(_last_tool_content(messages, "get_customer_profile"), "name", r"[A-Za-z][A-Za-z .-]*") or "there"
        return _credit_step(req, customer_id, name)
    return _on_request(match, messages)


def _after_profile(content: str, messages, node: str) -> AIMessage:
    customer_id = _field(content, "customer_id", r"C\d{5}")
    name = _field(content, "name", r"[A-Za-z][A-Za-z .-]*")
    if not (customer_id and name):
        return ai("The customer profile I received was incomplete, so I cannot continue safely — please try again in a moment.")
    req = _parse(_last_human(messages))
    if req["amount"] is not None and req["months"] is not None:
        return tool_call(
            "quote_installment",
            {"terms": {"amount": req["amount"], "months": req["months"], "product": req["product"]}},
        )
    if req["wants_credit"] and node == "eligibility":
        return _credit_step(req, customer_id, name)
    return ai(
        f"Thanks {name}. How much would you like to borrow (an amount between 1,000 and 500,000) "
        "and over how many months (12, 24, 36, 48 or 60)?"
    )


def _after_quote(content: str, messages, node: str) -> AIMessage:
    monthly = _field(content, "monthly_payment", r"[\d.]+")
    apr = _field(content, "apr_pct", r"[\d.]+")
    if monthly is None:
        return ai("The quote I received was incomplete (no monthly payment), so I cannot give you a number — please try again shortly.")
    terms = _last_call_args(messages, "quote_installment").get("terms", {})
    name = _field(_last_tool_content(messages, "get_customer_profile"), "name", r"[A-Za-z][A-Za-z .-]*") or "there"
    loan = (
        f"a {terms.get('amount', 0):,.0f} {str(terms.get('product', 'personal')).replace('_', ' ')} loan "
        f"over {terms.get('months')} months at about {float(monthly):,.2f} USD per month (APR {apr or '?'}%)"
    )
    if node == "advisor":
        return ai(f"Proposal for {name}: {loan}. Reply yes to submit the application.")
    return ai(
        f"{name}, {loan}. If you would like a credit check to confirm eligibility, reply "
        "'yes, I consent' with the last 4 digits of your SSN — I only run it with your explicit consent."
    )


def _after_credit(content: str) -> AIMessage:
    score = _field(content, "score", r"\d+")
    band = _field(content, "band", r"[a-z_]+") or "unknown"
    if score is None:
        return ai("The credit check result was incomplete, so I cannot report a score — please try again later.")
    return ai(f"Your credit check is done: score {score} ({band} band). You can go ahead and apply for the quoted terms.")


def _after_document(content: str) -> AIMessage:
    doc = _field(content, "doc_id", r"DOC-\d+")
    kind = _field(content, "type", r"[a-z_]+") or "document"
    status = _field(content, "status", r"[a-z_]+")
    if not (doc and status):
        return ai("The document record I received was incomplete, so I cannot confirm its status — please try again.")
    return ai(f"Document {doc} ({kind}): status {status}.")


def _after_submit(content: str) -> AIMessage:
    app_id = _field(content, "application_id", r"APP-\d+")
    next_step = _field(content, "next_step", r"[^'\"]+") or "we will be in touch"
    if app_id is None:
        return ai("The application receipt was incomplete (no application id), so I cannot confirm the submission — please try again.")
    return ai(f"Application {app_id} submitted. Next step: {next_step}.")


def _after_tool(match, messages) -> AIMessage:
    tm = messages[-1]
    content = _text(tm)
    name = _tool_name(messages, tm)
    if getattr(tm, "status", "") == "error" or _ERROR.search(content):
        return ai(f"Sorry — I could not reach the lending system ({name or 'tool'} failed). Please try again later.")
    node = _node(messages)
    if name == "get_customer_profile":
        return _after_profile(content, messages, node)
    if name == "quote_installment":
        return _after_quote(content, messages, node)
    if name == "credit_check":
        return _after_credit(content)
    if name == "document_status":
        return _after_document(content)
    if name == "submit_application":
        return _after_submit(content)
    return ai("The tool result was incomplete, so I cannot continue — please try again.")


def default_scripted_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        structured_script=[
            (r"\b(document|payslip|upload|proof of|doc-\d+)", lambda m, _msgs: {"intent": "documents"}),
            (r"\b(apply|application|submit)", lambda m, _msgs: {"intent": "apply"}),
            (
                r"\b(eligib|qualif|afford|loan|borrow|monthly|installment|credit|apr|rate)",
                lambda m, _msgs: {"intent": "eligibility"},
            ),
            (r".*", lambda m, _msgs: {"intent": "other"}),
        ],
        script=[
            (r"^TOOL:", _after_tool),
            (_CONFIRM, _confirm),
            (r".*", _on_request),
        ],
    )
