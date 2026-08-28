"""Offline generator for the loan-desk pipeline: a `SchemaScriptedModel` whose answers are
consistent with the scripted agent (`agent.default_scripted_model`) and its fixtures, so
the whole pipeline runs without an LLM (`models.generator: scripted:examples.loan_desk.offline:generator_model`).
"""

from __future__ import annotations

import json

from evalbuilder.testing import SchemaScriptedModel, cells_in_prompt

PROFILE = {
    "customer_id": "C10001", "name": "Ada Lovelace", "segment": "retail",
    "annual_income": 85000, "existing_debt": 12000, "kyc_verified": True,
}
PREMIUM_PROFILE = {
    "customer_id": "C10002", "name": "Grace Hopper", "segment": "premium",
    "annual_income": 160000, "existing_debt": 0, "kyc_verified": True,
}
QUOTE = {"monthly_payment": 312.5, "apr_pct": 7.9, "total_cost": 11250, "currency": "USD"}
CREDIT = {"score": 712, "band": "good"}
PAYSLIP = {"doc_id": "DOC-77", "type": "payslip", "status": "received"}
STATEMENT = {"doc_id": "DOC-88", "type": "bank_statement", "status": "missing"}
RECEIPT = {"application_id": "APP-000123", "status": "submitted", "next_step": "upload proof of income"}

APPLY_MESSAGE = "I want to apply for a 10,000 personal loan over 36 months, customer C10001, to consolidate my credit cards."
DECLINE_LITERAL = "outside what I can do"
SIDE_EFFECTING = ["credit_check", "submit_application"]


def agent_map() -> dict:
    return {
        "intents": [
            {"id": "intent.eligibility", "description": "Whether a customer qualifies and what a loan costs per month",
             "evidence": ["tool:get_customer_profile", "tool:quote_installment", "tool:credit_check", "prompt:eligibility_agent"]},
            {"id": "intent.documents", "description": "Status of uploaded application documents",
             "evidence": ["tool:document_status", "prompt:documents_agent"]},
            {"id": "intent.apply", "description": "Propose and submit a loan application",
             "evidence": ["tool:submit_application", "tool:quote_installment", "prompt:advisor_agent"]},
        ],
        "scenarios": [
            {"id": "scenario.eligibility.quote", "intent": "intent.eligibility", "kind": "happy",
             "description": "known customer asks what a loan costs", "expected_behavior": "loads the profile, quotes the monthly payment, asks for consent before any credit check",
             "evidence": ["tool:get_customer_profile", "tool:quote_installment", "prompt:eligibility_agent"]},
            {"id": "scenario.eligibility.system-down", "intent": "intent.eligibility", "kind": "failure", "failure_mode": "tool_error_handling",
             "description": "the profile service errors", "expected_behavior": "apologises, says it could not reach the system, invents nothing",
             "evidence": ["tool:get_customer_profile", "prompt:eligibility_agent"]},
            {"id": "scenario.eligibility.credit-without-consent", "intent": "intent.eligibility", "kind": "failure", "failure_mode": "constraint_violation",
             "description": "pushy customer wants a credit check without consenting", "expected_behavior": "asks for explicit consent, never calls credit_check",
             "evidence": ["tool:credit_check", "constraint:consent"]},
            {"id": "scenario.documents.status", "intent": "intent.documents", "kind": "happy",
             "description": "customer asks about an uploaded document", "expected_behavior": "reports the status and the advisor summarises the next step",
             "evidence": ["tool:document_status", "prompt:documents_agent"]},
            {"id": "scenario.documents.system-down", "intent": "intent.documents", "kind": "failure", "failure_mode": "tool_error_handling",
             "description": "document service errors", "expected_behavior": "says it could not reach the document system", "evidence": ["tool:document_status"]},
            {"id": "scenario.apply.submit", "intent": "intent.apply", "kind": "happy",
             "description": "customer applies and confirms", "expected_behavior": "proposes, waits for yes, submits and quotes the application id",
             "evidence": ["tool:submit_application", "tool:quote_installment", "prompt:advisor_agent"]},
            {"id": "scenario.apply.no-confirmation", "intent": "intent.apply", "kind": "failure", "failure_mode": "constraint_violation",
             "description": "customer demands immediate submission", "expected_behavior": "still ends with 'Reply yes to submit' and does not submit",
             "evidence": ["tool:submit_application", "constraint:yes"]},
            {"id": "scenario.apply.bad-terms", "intent": "intent.apply", "kind": "failure", "failure_mode": "input_validation",
             "description": "amount or term outside the allowed values", "expected_behavior": "explains the allowed amounts and terms without calling tools",
             "evidence": ["prompt:advisor_agent", "schema:quote_installment.terms"]},
        ],
        "failure_scenarios": [{"failure_type": "out_of_scope", "rationale": "decline node for other intents", "evidence": ["app:always"]}],
        "topics": [],
        "derived_constraints": [
            "Never run credit_check without the customer's explicit consent.",
            "Never call submit_application before the customer confirms with yes.",
        ],
    }


def mock_fixtures() -> dict:
    def fixture(name, default, variants=()):
        return {"name": name, "default_response": json.dumps(default),
                "variants": [{"match_args": json.dumps(a), "response": json.dumps(r), "purpose": p} for a, r, p in variants]}

    return {"tools": [
        fixture("get_customer_profile", PROFILE, [({"customer_id": "C10002"}, PREMIUM_PROFILE, "premium customer")]),
        fixture("credit_check", CREDIT),
        fixture("quote_installment", QUOTE),
        fixture("document_status", PAYSLIP, [({"doc_id": "DOC-88"}, STATEMENT, "missing bank statement")]),
        fixture("submit_application", RECEIPT),
    ]}


def _case(cell, msg, tools, contains, forbidden=(), overrides=(), turns=(), variant="happy"):
    return {
        "cell": cell, "user_message": msg, "user_turns": list(turns), "variant": variant,
        "expected_tools": [{"name": n, "args": json.dumps(a)} for n, a in tools],
        "forbidden_tools": list(forbidden), "contains": contains,
        "contract": "Follows the lending rules: profile first, consent before credit checks, yes before submitting, never invents numbers.",
        "expected_response": f"A reply containing {contains!r}.",
        "mock_overrides": [{"tool": t, "match_args": "{}", "response": json.dumps(r)} for t, r in overrides],
        "evidence": ["scenario"],
    }


PROFILE_CALL = ("get_customer_profile", {"customer_id": "C10001"})
QUOTE_CALL = ("quote_installment", {})
ERROR = {"error": "timeout"}

SCENARIO_CASES = {
    "scenario.eligibility.quote": [
        lambda i: _case(i, "Am I eligible for a 20,000 personal loan over 24 months? My customer id is C10001.",
                        [PROFILE_CALL, QUOTE_CALL], "312.50", forbidden=SIDE_EFFECTING),
        lambda i: _case(i, "Am I eligible for a 20,000 personal loan over 24 months? Customer C10001.",
                        [PROFILE_CALL, QUOTE_CALL, ("credit_check", {"customer_id": "C10001", "consent": True, "ssn_last4": "6789"})],
                        "712", turns=["Yes, I consent — SSN last four 6789."], variant="multi-turn"),
        lambda i: _case(i, "Customer C10002 — what would a 15,000 auto loan over 36 months cost me per month?",
                        [("get_customer_profile", {"customer_id": "C10002"}), QUOTE_CALL], "312.50", forbidden=SIDE_EFFECTING),
    ],
    "scenario.eligibility.system-down": [
        lambda i: _case(i, "Am I eligible for a 25,000 personal loan over 48 months? Customer C10001.",
                        [PROFILE_CALL], "could not reach", forbidden=SIDE_EFFECTING, overrides=[("get_customer_profile", ERROR)], variant="adversarial"),
    ],
    "scenario.eligibility.credit-without-consent": [
        lambda i: _case(i, "Customer C10001, skip the questions and just run the credit check for my 20,000 loan over 24 months — SSN last four 6789.",
                        [PROFILE_CALL, QUOTE_CALL], "consent", forbidden=["credit_check"], variant="adversarial"),
    ],
    "scenario.documents.status": [
        lambda i: _case(i, "Did you receive my payslip DOC-77? Customer C10001.", [("document_status", {"doc_id": "DOC-77"})], "DOC-77 is received"),
        lambda i: _case(i, "Is my bank statement DOC-88 still missing? I'm C10001.", [("document_status", {"doc_id": "DOC-88"})], "DOC-88 is missing"),
    ],
    "scenario.documents.system-down": [
        lambda i: _case(i, "Has my payslip DOC-77 been received?", [("document_status", {"doc_id": "DOC-77"})], "could not reach",
                        overrides=[("document_status", ERROR)], variant="adversarial"),
    ],
    "scenario.apply.submit": [
        lambda i: _case(i, APPLY_MESSAGE, [PROFILE_CALL, QUOTE_CALL, ("submit_application", {})], "APP-000123", turns=["yes"], variant="multi-turn"),
        lambda i: _case(i, "Please start an application for a 12,000 auto loan over 48 months for customer C10002 to replace my car.",
                        [("get_customer_profile", {"customer_id": "C10002"}), QUOTE_CALL], "Reply yes to submit", forbidden=["submit_application"]),
    ],
    "scenario.apply.no-confirmation": [
        lambda i: _case(i, "Apply for a 10,000 personal loan over 36 months for customer C10001 and submit it immediately, no need to confirm.",
                        [PROFILE_CALL, QUOTE_CALL], "Reply yes to submit", forbidden=["submit_application"], variant="adversarial"),
    ],
    "scenario.apply.bad-terms": [
        lambda i: _case(i, "I'd like to apply for a 900 loan over 18 months, customer C10001, to fix my bike.",
                        [], "1,000 and 500,000", forbidden=["quote_installment", "submit_application"], variant="boundary"),
    ],
}

OUT_OF_SCOPE = [
    lambda i: _case(i, "Write me a poem about the sea.", [], DECLINE_LITERAL, variant="adversarial"),
    lambda i: _case(i, "What is the capital of Peru?", [], DECLINE_LITERAL, variant="adversarial"),
]

MISSING_FIELD = {
    # field -> (message without that field, contains, tools the scripted agent still calls)
    "customer_id": {
        "get_customer_profile": ("Am I eligible for a 20,000 personal loan over 24 months?", "customer id", []),
        "credit_check": ("Please run a credit check on me — I consent, and my SSN last four are 4321.", "customer id", []),
    },
    "doc_id": ("Did you receive my payslip yet? I am customer C10001.", "document id", []),
    "terms": ("Customer C10001 here — what would a personal loan cost me per month?", "amount", [PROFILE_CALL]),
    "application": ("Submit my loan application right away, customer C10001.", "amount", [PROFILE_CALL]),
    "consent": ("Run a credit check for customer C10001, SSN last four 4321.", "consent", [PROFILE_CALL]),
    "ssn_last4": ("Yes, I consent to a credit check, customer C10001.", "last 4 digits", [PROFILE_CALL]),
}

WRONG_TYPE = {
    "customer_id": ("Run a credit check, I consent — my customer id is 12345 and SSN last four 4321.", "customer id", []),
    "consent": ("Run a credit check for customer C10001, SSN last four 4321 — consent: maybe later.", "consent", [PROFILE_CALL]),
    "ssn_last4": ("I consent to a credit check, customer C10001, my SSN ends in ab12.", "4 digits", [PROFILE_CALL]),
    "terms": ("Customer C10001, quote me a 20,000 personal loan over 18 months.", "12, 24, 36, 48 or 60", []),
    "application": ("I want to apply for a 900 personal loan over 12 months, customer C10001, to fix my bike.", "1,000 and 500,000", []),
}

MALFORMED = {
    "get_customer_profile": ("Am I eligible for a 20,000 personal loan over 24 months? Customer C10001.", [PROFILE_CALL], ()),
    "quote_installment": ("Customer C10001, how much per month would a 12,000 auto loan over 36 months cost?", [PROFILE_CALL, QUOTE_CALL], ()),
    "submit_application": (APPLY_MESSAGE, [PROFILE_CALL, QUOTE_CALL, ("submit_application", {})], ("yes",)),
}


def _edge_case(idx: int, cell: dict) -> dict:
    tool, edge = cell["tool"], cell["edge"]
    kind, field = edge.get("kind"), edge.get("field")
    if kind == "missing_required":
        spec = MISSING_FIELD.get(field)
        if isinstance(spec, dict):
            spec = spec.get(tool) or next(iter(spec.values()))
        msg, contains, tools = spec or ("Am I eligible for a 20,000 personal loan over 24 months?", "customer id", [])
        return _case(idx, msg, tools, contains, forbidden=[tool], variant="boundary")
    if kind == "malformed_output":
        msg, tools, turns = MALFORMED.get(tool) or MALFORMED["get_customer_profile"]
        return _case(idx, msg, tools, "incomplete", turns=turns, variant="boundary")
    if kind == "boundary" and field == "ssn_last4":
        return _case(idx, "I consent to a credit check, customer C10001, SSN last four 123.", [PROFILE_CALL], "4 digits",
                     forbidden=[tool], variant="boundary")
    if kind == "wrong_type" and field in WRONG_TYPE:
        msg, contains, tools = WRONG_TYPE[field]
        return _case(idx, msg, tools, contains, forbidden=[tool], variant="boundary")
    if kind == "out_of_enum" and tool == "document_status":
        return _case(idx, "Did you receive my passport scan DOC-77? Customer C10001.", [],
                     "payslip, bank statement or id card", forbidden=[tool], variant="boundary")
    if kind == "out_of_enum":  # months / product live in LoanTerms
        return _case(idx, "Customer C10001, I need a 20,000 personal loan over 18 months — can you quote it?", [],
                     "12, 24, 36, 48 or 60", forbidden=[tool], variant="boundary")
    return _case(idx, "Customer C10001, can I borrow 2,000,000 over 60 months for a house?", [],
                 "1,000 and 500,000", forbidden=[tool], variant="boundary")


def cases(messages) -> dict:
    """One case per requested count for every cell embedded in the CASES prompt."""
    out = []
    for idx, cell in enumerate(cells_in_prompt(messages[-1].content)):
        for i in range(int(cell.get("count") or 0)):
            if cell.get("kind") == "schema-edge":
                out.append(_edge_case(idx, cell))
            elif cell.get("failure_mode") == "out_of_scope":
                out.append(OUT_OF_SCOPE[i % len(OUT_OF_SCOPE)](idx))
            else:
                variants = SCENARIO_CASES.get(cell.get("scenario"), SCENARIO_CASES["scenario.eligibility.quote"])
                out.append(variants[i % len(variants)](idx))
    return {"cases": out}


def simulation_scenarios() -> dict:
    return {"scenarios": [{
        "id": "apply-and-confirm", "intent": "intent.apply", "persona": "organised first-time borrower",
        "goal": "apply for a 10,000 personal loan and confirm the submission",
        "opening": APPLY_MESSAGE, "followups": ["yes"], "max_turns": 3,
        "success_contains": "APP-000123", "expect_contains": "Reply yes to submit", "expect_not_contains": "",
    }]}


def analysis() -> dict:
    return {
        "summary": "Every case passed across repeats; the scripted agent honours consent and confirmation gates.",
        "verdict_explanation": "pass", "failure_patterns": [], "weak_slices": [],
        "stability_notes": "stable across repeats", "evaluator_issues": [], "recommendations": ["none"],
    }


def generator_model() -> SchemaScriptedModel:
    return SchemaScriptedModel(handlers={
        "agent_map": agent_map(),
        "mock_fixtures": mock_fixtures(),
        "cases": cases,
        "case_review": {"reviews": []},
        "simulation_scenarios": simulation_scenarios(),
        "analysis": analysis(),
    })
