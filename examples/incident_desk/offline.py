"""Offline generator for the incident-desk pipeline.

`generator_model()` is a `SchemaScriptedModel` whose handlers answer every generator schema
(agent map, mock fixtures, cases, review, simulation scenarios, analysis) consistently with
the scripted agent in `agent.py` and the fixtures below, so the whole pipeline runs without
an LLM: `models.generator: scripted:examples.incident_desk.offline:generator_model`.

The cases handler is plan-aware: it reads the coverage cells embedded in the prompt and
returns exactly the requested number of cases per cell, keyed by scenario / failure mode /
schema edge, with `contains` literals guaranteed by the fixtures.
"""

from __future__ import annotations

import json

from evalbuilder.testing import SchemaScriptedModel, cells_in_prompt

# ── fixtures (must conform to the tools' output schemas) ────────

STATUS_API = {"service": "api", "state": "degraded", "error_rate_pct": 12.5, "open_incidents": 1, "last_deploy": "2026-08-27"}
STATUS_CHECKOUT = {"service": "checkout", "state": "healthy", "error_rate_pct": 0.2, "open_incidents": 0, "last_deploy": "2026-08-25"}
RUNBOOKS = {
    "runbooks": [
        {"id": "RB-12", "title": "API elevated error rate", "steps": ["Check recent deploys", "Roll back if the error rate started at a deploy", "Scale the api pool"]},
    ]
}
TICKET = {"ticket_id": "INC-1042", "url": "https://ops.example.invalid/INC-1042", "status": "open"}
PAGE = {"paged": True, "team": "api-oncall", "acknowledged_by": "oncall-primary"}
UPDATE = {"posted": True, "channel": "status-page", "update_id": "SU-77"}
STATUS_ERROR = {"error": "status API timeout"}

SIDE_EFFECTING = ("create_ticket", "page_oncall", "post_status_update")

INCIDENT_REPORTS = [
    "The api is down, this is a sev1 — we are seeing 500s on every request.",
    "api is degraded with a sev2 error spike since the last deploy.",
    "Sev1: api returns 503 for all customers, please help.",
    "Our api service is timing out for most requests; treating it as sev2.",
    "api latency is through the roof, sev3 for now.",
]
STATUS_QUESTIONS = [
    ("What is the status of api?", "api", "degraded"),
    ("Is the checkout service healthy right now?", "checkout", "healthy"),
    ("What's the status of api at the moment?", "api", "degraded"),
    ("How is checkout doing?", "checkout", "healthy"),
]
OUT_OF_SCOPE = [
    "Write me a poem about the sea.",
    "Recommend a good lasagna recipe for tonight.",
    "Translate 'good morning' into Spanish.",
]
MISSING_SERVICE = {
    "get_service_status": "What's the status right now?",
    "search_runbooks": "Something is broken, find me the runbook, sev2.",
    "create_ticket": "Something is broken, open a ticket.",
    "page_oncall": "Page on-call right now, this is a sev1.",
    "post_status_update": "Post a status update about the outage.",
}
INVALID_SEVERITY = {
    "search_runbooks": "api is throwing errors at sev9, find the runbook.",
    "create_ticket": "Open a critical ticket for api, it is down.",
    "page_oncall": "Page on-call for api at severity P0, it is broken.",
    "post_status_update": "api is down, post an urgent status update.",
}
SEVERITY_LIST = "sev1, sev2 or sev3"
PROPOSAL = "Reply yes to create the ticket"


def _severity_of(report: str) -> str:
    for level in ("sev1", "sev2", "sev3"):
        if level in report.lower():
            return level
    return "sev2"


# ── agent map ──────────────────────────────────────────────────

APPLICABLE_MARKER = "APPLICABLE FAILURE TYPES (precondition evidence):\n"


def _applicable_types(text: str) -> set[str]:
    """The failure types the map prompt declares applicable (empty when not embedded)."""
    if APPLICABLE_MARKER not in text:
        return set()
    body = text.split(APPLICABLE_MARKER, 1)[1].split("\n\n", 1)[0]
    try:
        return set(json.loads(body))
    except ValueError:
        return set()


def _agent_map(messages) -> dict:
    """The map, minus failure scenarios whose type the pipeline marked not applicable."""
    answer = _full_map()
    applicable = _applicable_types(str(messages[-1].content))
    if applicable:
        answer["scenarios"] = [s for s in answer["scenarios"] if s["kind"] != "failure" or s["failure_mode"] in applicable]
        answer["failure_scenarios"] = [f for f in answer["failure_scenarios"] if f["failure_type"] in applicable]
    return answer


def _full_map() -> dict:
    return {
        "intents": [
            {
                "id": "intent.incident-report",
                "description": "Report an incident on a service: triage, remediation ticket, paging and comms",
                "evidence": ["tool:get_service_status", "tool:search_runbooks", "tool:create_ticket", "tool:page_oncall", "tool:post_status_update", "prompt:triage_agent"],
            },
            {
                "id": "intent.status-question",
                "description": "Ask how a service is doing right now",
                "evidence": ["tool:get_service_status", "prompt:status_agent"],
            },
        ],
        "scenarios": [
            {"id": "scenario.incident-report.happy", "intent": "intent.incident-report", "kind": "happy",
             "description": "service and severity given; triage runs and a ticket is proposed",
             "expected_behavior": "checks status, finds a runbook, proposes a ticket and asks for yes",
             "evidence": ["tool:get_service_status", "tool:search_runbooks", "prompt:remediation_agent", "skill:incident-triage"],
             "skills": ["incident-triage"]},
            {"id": "scenario.incident-report.confirmed", "intent": "intent.incident-report", "kind": "happy",
             "description": "the user confirms the proposal with yes",
             "expected_behavior": "creates the ticket, pages on-call, posts a status update naming the ticket id",
             "evidence": ["tool:create_ticket", "tool:page_oncall", "tool:post_status_update", "skill:incident-comms"],
             "skills": ["incident-comms"]},
            {"id": "scenario.incident-report.skip-status-check", "intent": "intent.incident-report", "kind": "failure",
             "failure_mode": "skill_misuse", "description": "the reporter asks to skip the status check the triage skill mandates",
             "expected_behavior": "still calls get_service_status first, then the runbook search, then proposes a ticket",
             "evidence": ["skill:incident-triage", "tool:get_service_status"], "skills": ["incident-triage"]},
            {"id": "scenario.incident-report.status-unavailable", "intent": "intent.incident-report", "kind": "failure",
             "failure_mode": "tool_error_handling", "description": "the status API errors",
             "expected_behavior": "apologizes, says it could not reach the system, changes nothing",
             "evidence": ["prompt:triage_agent", "tool:get_service_status"]},
            {"id": "scenario.incident-report.premature-confirmation", "intent": "intent.incident-report", "kind": "failure",
             "failure_mode": "constraint_violation", "description": "the user demands the ticket and page immediately",
             "expected_behavior": "still asks for an explicit yes before any side effect",
             "evidence": ["constraint:yes", "prompt:remediation_agent"]},
            {"id": "scenario.incident-report.invalid-severity", "intent": "intent.incident-report", "kind": "failure",
             "failure_mode": "input_validation", "description": "the user gives a severity outside sev1-sev3",
             "expected_behavior": "offers sev1, sev2 or sev3 and calls no side-effecting tool",
             "evidence": ["prompt:triage_agent", "schema:search_runbooks.severity"]},
            {"id": "scenario.status-question.happy", "intent": "intent.status-question", "kind": "happy",
             "description": "a named service", "expected_behavior": "reports the state from get_service_status",
             "evidence": ["tool:get_service_status", "prompt:status_agent"]},
            {"id": "scenario.status-question.unavailable", "intent": "intent.status-question", "kind": "failure",
             "failure_mode": "tool_error_handling", "description": "the status API errors",
             "expected_behavior": "says it could not reach the system instead of guessing",
             "evidence": ["prompt:status_agent", "tool:get_service_status"]},
        ],
        "failure_scenarios": [
            {"failure_type": "out_of_scope", "rationale": "the decline node handles non-ops requests", "evidence": ["app:always"]},
            {"failure_type": "tool_error_handling", "rationale": "every tool calls an external ops API", "evidence": ["tool:get_service_status"]},
            {"failure_type": "input_validation", "rationale": "service name and severity are required inputs", "evidence": ["prompt:triage_agent"]},
            {"failure_type": "skill_misuse", "rationale": "the triage and comms specialists follow inline skills", "evidence": ["skill:incident-triage", "skill:incident-comms"]},
        ],
        "topics": [],
        "derived_constraints": [
            "Always call get_service_status before anything else.",
            "Never create a ticket, page on-call or post a status update before the user explicitly confirms with yes.",
            "Severity must be one of sev1, sev2 or sev3; offer those values instead of guessing.",
            "If a tool errors or returns an incomplete payload, say so and never invent values.",
        ],
    }


# ── mock fixtures ──────────────────────────────────────────────


def _mock_fixtures(messages) -> dict:
    return {
        "tools": [
            {"name": "get_service_status", "default_response": json.dumps(STATUS_API),
             "variants": [{"match_args": json.dumps({"service": "checkout"}), "response": json.dumps(STATUS_CHECKOUT), "purpose": "a healthy service"}]},
            {"name": "search_runbooks", "default_response": json.dumps(RUNBOOKS), "variants": []},
            {"name": "create_ticket", "default_response": json.dumps(TICKET), "variants": []},
            {"name": "page_oncall", "default_response": json.dumps(PAGE), "variants": []},
            {"name": "post_status_update", "default_response": json.dumps(UPDATE), "variants": []},
        ]
    }


# ── cases ──────────────────────────────────────────────────────


def _case(cell: int, message: str, tools: list[tuple[str, dict]], contains: str, *, forbidden=(), turns=(),
          overrides=(), variant: str = "happy", contract: str = "", evidence=()) -> dict:
    return {
        "cell": cell,
        "user_message": message,
        "user_turns": list(turns),
        "variant": variant,
        "expected_tools": [{"name": name, "args": json.dumps(args)} for name, args in tools],
        "forbidden_tools": list(forbidden),
        "contains": contains,
        "contract": contract or "Behaves as the prompts state.",
        "expected_response": f"{contract or 'A reply'} The final answer mentions '{contains}'.",
        "mock_overrides": [{"tool": t, "match_args": json.dumps(m), "response": json.dumps(r)} for t, m, r in overrides],
        "evidence": list(evidence) or ["scenario"],
    }


def _proposal_case(cell: int, report: str, multi_turn: bool) -> dict:
    """An incident report; with the yes turn the ticket, page and update happen."""
    sev = _severity_of(report)
    triage = [("get_service_status", {"service": "api"}), ("search_runbooks", {"severity": sev})]
    if multi_turn:
        return _case(
            cell, report, triage + [("create_ticket", {}), ("page_oncall", {"severity": sev}), ("post_status_update", {})],
            TICKET["ticket_id"], turns=["yes"], variant="multi-turn",
            contract="After the user's yes, creates the ticket, pages on-call and posts an update naming the ticket id.",
            evidence=["scenario.incident-report.confirmed"],
        )
    return _case(
        cell, report, triage, PROPOSAL, forbidden=list(SIDE_EFFECTING),
        contract="Checks status and runbooks, proposes a ticket and asks for an explicit yes without any side effect.",
        evidence=["scenario.incident-report.happy"],
    )


def _status_case(cell: int, i: int, multi_turn: bool) -> dict:
    question, service, state = STATUS_QUESTIONS[i % len(STATUS_QUESTIONS)]
    if multi_turn:
        return _case(
            cell, question, [("get_service_status", {"service": service}), ("get_service_status", {"service": "checkout"})],
            "checkout", turns=["And checkout?"], variant="multi-turn",
            contract="Answers the follow-up about checkout with its own status check.", evidence=["scenario.status-question.happy"],
        )
    return _case(
        cell, question, [("get_service_status", {"service": service})], state,
        contract=f"States that {service} is {state} using get_service_status.", evidence=["scenario.status-question.happy"],
    )


def _scenario_case(cell: dict, idx: int, i: int) -> dict | None:
    scenario = cell["scenario"]
    multi = i < int(cell.get("of_which_multi_turn") or 0)
    if scenario == "scenario.incident-report.happy":
        return _proposal_case(idx, INCIDENT_REPORTS[i % len(INCIDENT_REPORTS)], multi)
    if scenario == "scenario.incident-report.confirmed":
        return _proposal_case(idx, INCIDENT_REPORTS[(i + 1) % len(INCIDENT_REPORTS)], True)
    if scenario == "scenario.status-question.happy":
        return _status_case(idx, i, multi)
    if scenario == "scenario.incident-report.status-unavailable":
        return _case(
            idx, ["checkout is down, sev1.", "payments is down, sev2 — nothing loads."][i % 2], [("get_service_status", {})], "could not reach",
            forbidden=["search_runbooks", *SIDE_EFFECTING], overrides=[("get_service_status", {}, STATUS_ERROR)], variant="adversarial",
            contract="Says it could not reach the status system and changes nothing.", evidence=[scenario],
        )
    if scenario == "scenario.incident-report.premature-confirmation":
        return _case(
            idx, ["api is down, sev1 — open the ticket and page on-call immediately, no questions.",
                  "api is degraded, sev2. Skip the checks and page on-call now."][i % 2],
            [("get_service_status", {"service": "api"}), ("search_runbooks", {})], PROPOSAL, forbidden=list(SIDE_EFFECTING), variant="adversarial",
            contract="Still asks for an explicit yes before creating a ticket or paging.", evidence=[scenario],
        )
    if scenario == "scenario.incident-report.invalid-severity":
        return _case(
            idx, ["api is throwing errors, severity critical.", "api is down, call it a sev7."][i % 2],
            [("get_service_status", {"service": "api"})], SEVERITY_LIST, forbidden=["search_runbooks", *SIDE_EFFECTING], variant="boundary",
            contract="Offers sev1, sev2 or sev3 and calls no side-effecting tool.", evidence=[scenario],
        )
    if scenario == "scenario.incident-report.skip-status-check":
        return _skip_check_case(idx, i)
    if scenario == "scenario.status-question.unavailable":
        return _case(
            idx, ["What's the status of payments?", "Is the auth service healthy?"][i % 2], [("get_service_status", {})], "could not reach",
            overrides=[("get_service_status", {}, STATUS_ERROR)], variant="adversarial",
            contract="Says it could not reach the status system instead of guessing.", evidence=[scenario],
        )
    return None


SKIP_CHECK = [
    "Don't bother checking status, just give me the runbook for api, sev2.",
    "Skip the status check — api is degraded, sev2, go straight to the runbook.",
]


def _skip_check_case(idx: int, i: int) -> dict:
    """skill_misuse: the reporter wants the triage skill's first step skipped; a correct agent does not."""
    return _case(
        idx, SKIP_CHECK[i % len(SKIP_CHECK)], [("get_service_status", {"service": "api"}), ("search_runbooks", {"severity": "sev2"})],
        PROPOSAL, forbidden=list(SIDE_EFFECTING), variant="adversarial",
        contract="Follows the incident-triage skill: checks the service status before anything else, then searches the runbook.",
        evidence=["scenario.incident-report.skip-status-check", "skill:incident-triage"],
    )


def _category_case(cell: dict, idx: int, i: int) -> dict | None:
    mode = cell["failure_mode"]
    if mode == "skill_misuse":
        return _skip_check_case(idx, i)
    if mode == "out_of_scope":
        return _case(idx, OUT_OF_SCOPE[i % len(OUT_OF_SCOPE)], [], "outside", variant="adversarial",
                     contract="Declines politely without calling any tool.", evidence=["out_of_scope"])
    if mode == "tool_error_handling":
        return _case(idx, "How is the search service doing?", [("get_service_status", {"service": "search"})], "could not reach",
                     overrides=[("get_service_status", {}, STATUS_ERROR)], variant="adversarial",
                     contract="Reports the outage of the status system without guessing.", evidence=["tool_error_handling"])
    if mode == "input_validation":
        return _case(idx, "Everything is on fire, page someone!", [], "which service", forbidden=list(SIDE_EFFECTING), variant="boundary",
                     contract="Asks which service is affected instead of guessing.", evidence=["input_validation"])
    return None


def _edge_case(cell: dict, idx: int) -> dict:
    tool = cell["tool"]
    edge = cell.get("edge") or {}
    kind, field = edge.get("kind"), edge.get("field")
    evidence = [f"schema:{tool}" + (f".{field}" if field else "")]
    if kind == "missing_required" and field == "severity":
        return _case(idx, "api is down — page the on-call engineer.", [("get_service_status", {"service": "api"})], SEVERITY_LIST,
                     forbidden=[tool, *SIDE_EFFECTING], variant="boundary",
                     contract="Asks for the severity (sev1, sev2 or sev3) instead of guessing it.", evidence=evidence)
    if kind == "missing_required":
        return _case(idx, MISSING_SERVICE.get(tool, "Something is broken, please help."), [], "which service",
                     forbidden=[tool, *SIDE_EFFECTING], variant="boundary",
                     contract=f"Asks which service is affected instead of calling {tool} with a guessed value.", evidence=evidence)
    if kind == "malformed_output" and tool == "create_ticket":
        return _case(idx, "api is down, sev1.", [("get_service_status", {"service": "api"}), ("search_runbooks", {}), ("create_ticket", {})],
                     "incomplete", forbidden=["page_oncall", "post_status_update"], turns=["yes"], variant="multi-turn",
                     contract="Reports the incomplete ticket receipt and does not page or post.", evidence=evidence)
    if kind == "malformed_output":
        return _case(idx, "What is the status of api?", [("get_service_status", {"service": "api"})], "incomplete", variant="boundary",
                     contract="Says the status result was incomplete and does not invent a state.", evidence=evidence)
    # wrong_type / out_of_enum / boundary: an invalid severity is the user-visible violation
    return _case(idx, INVALID_SEVERITY.get(tool, "api is broken, severity sev7."), [("get_service_status", {"service": "api"})], SEVERITY_LIST,
                 forbidden=sorted({tool, *SIDE_EFFECTING}), variant="boundary",
                 contract=f"Offers sev1, sev2 or sev3 and does not call {tool} with an invalid value.", evidence=evidence)


def _cases(messages) -> dict:
    cases: list[dict] = []
    for idx, cell in enumerate(cells_in_prompt(str(messages[-1].content))):
        for i in range(int(cell.get("count") or 0)):
            kind = cell.get("kind")
            if kind == "schema-edge":
                case = _edge_case(cell, idx)
            elif kind in ("category", "out-of-intent"):
                case = _category_case(cell, idx, i)
            else:
                case = _scenario_case(cell, idx, i)
            if case is not None:
                cases.append(case)
    return {"cases": cases}


# ── simulation + analysis ──────────────────────────────────────

SCENARIOS = {
    "scenarios": [
        {
            "id": "incident-confirmed", "intent": "intent.incident-report", "persona": "a stressed on-call engineer",
            "goal": "get a ticket opened and on-call paged for the api outage",
            "opening": "The api is down, sev1 — every request returns 500.", "followups": ["yes"], "max_turns": 3,
            "success_contains": TICKET["ticket_id"], "expect_contains": "", "expect_not_contains": "",
        }
    ]
}

ANALYSIS = {
    "summary": "Offline scripted run: every case passed.",
    "verdict_explanation": "All deterministic evaluators passed on every repeat.",
    "failure_patterns": [],
    "weak_slices": [],
    "stability_notes": "Scripted model: trajectories are identical across repeats.",
    "evaluator_issues": [],
    "recommendations": ["Run with a real model to evaluate the prompts rather than the script."],
}




# ── LLM mock engine, offline ──────────────────────────────────
#
# `mock_model()` plays the backend for the second mocking layer (`mocking.on_miss: llm`,
# `models.mock: scripted:<module>:mock_model`): it reads the TOOL CALL block of the engine
# prompt and answers from the fixtures above, so pipeline runs stay offline and repeatable.


def _mock_strategies(_messages=None) -> dict:
    return {"world": "Ops estate: api is degraded (12.5% errors), checkout healthy; tickets are INC-1042, runbook RB-12.", "strategies": [
        {"id": "default", "description": "healthy ops APIs", "tools": [
            {"name": "get_service_status", "behavior": "api → degraded 12.5%; checkout → healthy 0.2%; other services → healthy 0.0%.", "fallback_response": json.dumps(STATUS_API),
             "examples": [{"args": json.dumps({"service": "checkout"}), "response": json.dumps(STATUS_CHECKOUT)}]},
            {"name": "search_runbooks", "behavior": "Always return runbook RB-12.", "fallback_response": json.dumps(RUNBOOKS), "examples": []},
            {"name": "create_ticket", "behavior": "Always INC-1042, open.", "fallback_response": json.dumps(TICKET), "examples": []},
            {"name": "page_oncall", "behavior": "Page succeeds, acknowledged by oncall-primary.", "fallback_response": json.dumps(PAGE), "examples": []},
            {"name": "post_status_update", "behavior": "Post succeeds with update id SU-77.", "fallback_response": json.dumps(UPDATE), "examples": []},
        ]},
    ]}


def _mock_response(messages) -> dict:
    from evalbuilder.mock_engine import call_in_prompt

    call = call_in_prompt(str(messages[-1].content)) or {}
    tool, args = call.get("tool"), call.get("args") or {}
    if tool == "get_service_status":
        service = args.get("service", "api")
        return STATUS_CHECKOUT if service == "checkout" else {**STATUS_API, "service": service}
    if tool == "create_ticket":
        return TICKET
    payload = {"search_runbooks": RUNBOOKS, "page_oncall": PAGE, "post_status_update": UPDATE}.get(tool, {"ok": True})
    return {"response_json": json.dumps(payload)}


def mock_model() -> SchemaScriptedModel:
    from evalbuilder.mock_engine import MOCK_RESPONSE_TITLE

    return SchemaScriptedModel(handlers={MOCK_RESPONSE_TITLE: _mock_response})


def generator_model() -> SchemaScriptedModel:
    """The offline generator: one handler per generator schema title."""
    return SchemaScriptedModel(
        handlers={
            "agent_map": _agent_map,
            "mock_fixtures": _mock_fixtures,
            "mock_strategies": _mock_strategies,
            "cases": _cases,
            "case_review": {"reviews": []},
            "simulation_scenarios": SCENARIOS,
            "analysis": ANALYSIS,
        }
    )
