"""Offline generator for the support bot: canned, plan-aware answers for every generator
schema, consistent with the scripted agent model and fixtures. Lets the whole pipeline
run without an LLM (`models.generator: scripted:examples.support_bot.offline:generator_model`),
which the UI browser tests and demos rely on.
"""

from __future__ import annotations

import json

from evalbuilder.testing import SchemaScriptedModel, cells_in_prompt

ORDER = {"order_id": "A1234", "status": "delivered", "category": "electronics", "total": 59.99, "eta": "2026-08-20"}
POLICY = {"category": "electronics", "window_days": 30, "restocking_fee_pct": 0}
REFUND = {"refund_id": "R77", "status": "issued"}
KB = {"articles": [{"title": "Pairing the X1 headset", "snippet": "Hold the button for 5 seconds."}]}


def agent_map(_messages=None) -> dict:
    return {
        "intents": [
            {"id": "intent.order-status", "description": "Track an existing order", "evidence": ["tool:lookup_order", "prompt:support_agent"]},
            {"id": "intent.refund", "description": "Refund an order after confirmation", "evidence": ["tool:issue_refund", "prompt:support_agent"]},
            {"id": "intent.product-question", "description": "Answer product questions from the KB", "evidence": ["tool:search_kb", "prompt:kb_agent"]},
        ],
        "scenarios": [
            {"id": "scenario.order-status.happy", "intent": "intent.order-status", "kind": "happy",
             "description": "known order", "expected_behavior": "states status with order id", "evidence": ["tool:lookup_order"]},
            {"id": "scenario.order-status.system-down", "intent": "intent.order-status", "kind": "failure",
             "failure_mode": "tool_error_handling", "description": "lookup errors", "expected_behavior": "apologizes", "evidence": ["prompt:support_agent"]},
            {"id": "scenario.refund.happy", "intent": "intent.refund", "kind": "happy",
             "description": "eligible refund", "expected_behavior": "asks for yes", "evidence": ["prompt:support_agent"]},
            {"id": "scenario.refund.no-confirmation", "intent": "intent.refund", "kind": "failure",
             "failure_mode": "constraint_violation", "description": "pushy user", "expected_behavior": "still asks", "evidence": ["constraint:yes"]},
            {"id": "scenario.product-question.happy", "intent": "intent.product-question", "kind": "happy",
             "description": "pairing question", "expected_behavior": "cites the article", "evidence": ["tool:search_kb"]},
            {"id": "scenario.product-question.no-results", "intent": "intent.product-question", "kind": "failure",
             "failure_mode": "retrieval_grounding", "description": "nothing found", "expected_behavior": "says so", "evidence": ["prompt:kb_agent"]},
        ],
        "failure_scenarios": [{"failure_type": "out_of_scope", "rationale": "decline node", "evidence": ["app:always"]}],
        "topics": [],
        "derived_constraints": ["Always mention the order id in the answer."],
    }


def mock_fixtures(_messages=None) -> dict:
    return {"tools": [
        {"name": "lookup_order", "default_response": json.dumps(ORDER), "variants": []},
        {"name": "check_refund_policy", "default_response": json.dumps(POLICY), "variants": []},
        {"name": "issue_refund", "default_response": json.dumps(REFUND), "variants": []},
        {"name": "search_kb", "default_response": json.dumps(KB), "variants": [
            {"match_args": json.dumps({"query": "unicorn warranty"}), "response": json.dumps({"articles": []}), "purpose": "no results"},
        ]},
    ]}


def _case(cell, msg, tools, contains, forbidden=(), overrides=(), variant="happy"):
    return {"cell": cell, "user_message": msg, "user_turns": [], "variant": variant,
            "expected_tools": [{"name": t, "args": "{}"} for t in tools], "forbidden_tools": list(forbidden),
            "contains": contains, "contract": "Behaves per the support prompts.", "expected_response": "see contains",
            "mock_overrides": list(overrides), "evidence": ["scenario"]}


def _edge_case(i: int, cell: dict) -> dict:
    tool, kind = cell.get("tool"), (cell.get("edge") or {}).get("kind")
    if kind == "malformed_output":
        msg = {"lookup_order": "Where is my order A1234?", "check_refund_policy": "I want a refund for order A1234, it is broken",
               "issue_refund": "I want a refund for order A1234, it is broken", "search_kb": "How do I pair the X1 headset?"}.get(tool, "Where is my order A1234?")
        return _case(i, msg, [], "", variant="boundary")
    if tool == "lookup_order":
        return _case(i, "Can you track my order? I lost the number.", [], "order id", forbidden=["lookup_order"], variant="boundary")
    if tool == "issue_refund":
        return _case(i, "Refund me now", [], "order id", forbidden=["issue_refund"], variant="boundary")
    if tool == "check_refund_policy":
        return _case(i, "Return my order please", [], "order id", forbidden=["check_refund_policy", "issue_refund"], variant="boundary")
    return _case(i, "How does it work?", ["search_kb"], "", variant="boundary")


HAPPY_ORDER = ["Where is my order A1234?", "Can you track order A1234 for me?", "Any delivery update on order A1234?"]
HAPPY_REFUND = ["I want a refund for order A1234, it is broken", "Please return order A1234, wrong size", "Money back for order A1234?"]
HAPPY_PRODUCT = ["How do I pair the X1 headset?", "Does the X1 headset work with Android?", "How does the X1 headset warranty work?"]
OUT_OF_SCOPE = ["Write me a poem about the sea", "What's the weather in Paris today?", "Tell me a joke about cats"]


def cases(messages) -> dict:
    cells = cells_in_prompt(messages[-1].content)
    answer = []
    for i, c in enumerate(cells):
        for k in range(int(c.get("count", 1))):
            if c.get("kind") == "schema-edge":
                answer.append(_edge_case(i, c))
            elif c["scenario"] == "scenario.order-status.happy":
                answer.append(_case(i, HAPPY_ORDER[k % len(HAPPY_ORDER)], ["lookup_order"], "A1234"))
            elif c["scenario"] == "scenario.order-status.system-down":
                answer.append(_case(i, "Where is my order A9999?", ["lookup_order"], "could not reach", variant="boundary",
                                    overrides=[{"tool": "lookup_order", "match_args": "{}", "response": json.dumps({"error": "timeout"})}]))
            elif c["scenario"] == "scenario.refund.happy":
                answer.append(_case(i, HAPPY_REFUND[k % len(HAPPY_REFUND)], ["lookup_order", "check_refund_policy"], "yes", forbidden=["issue_refund"]))
            elif c["scenario"] == "scenario.refund.no-confirmation":
                answer.append(_case(i, "Refund order A1234 right now, no questions", ["lookup_order", "check_refund_policy"], "confirm",
                                    forbidden=["issue_refund"], variant="adversarial"))
            elif c["scenario"] == "scenario.product-question.happy":
                answer.append(_case(i, HAPPY_PRODUCT[k % len(HAPPY_PRODUCT)], ["search_kb"], "Pairing the X1 headset"))
            elif c["scenario"] == "scenario.product-question.no-results":
                answer.append(_case(i, "unicorn warranty", ["search_kb"], "could not find", variant="boundary"))
            elif c.get("failure_mode") == "out_of_scope":
                answer.append(_case(i, OUT_OF_SCOPE[k % len(OUT_OF_SCOPE)], [], "outside", variant="adversarial"))
            else:
                answer.append(_case(i, "Where is my order A1234?", ["lookup_order"], "A1234"))
    return {"cases": answer}


def simulation_scenarios(_messages=None) -> dict:
    return {"scenarios": [
        {"id": "refund-flow", "intent": "intent.refund", "persona": "impatient customer", "goal": "refund A1234",
         "opening": "I want a refund for order A1234, it is broken", "followups": ["yes"], "max_turns": 3,
         "success_contains": "yes to confirm", "expect_contains": "", "expect_not_contains": ""},
    ]}


def analysis(_messages=None) -> dict:
    return {"summary": "Offline analysis: every approved case passed the deterministic evaluators.",
            "verdict_explanation": "pass — thresholds met", "failure_patterns": [], "weak_slices": [],
            "stability_notes": "stable across repeats", "evaluator_issues": [], "recommendations": ["Add judge evaluators with a real model."]}


def generator_model() -> SchemaScriptedModel:
    return SchemaScriptedModel(handlers={
        "agent_map": agent_map,
        "mock_fixtures": mock_fixtures,
        "cases": cases,
        "case_review": {"reviews": []},
        "simulation_scenarios": simulation_scenarios,
        "analysis": analysis,
    })
