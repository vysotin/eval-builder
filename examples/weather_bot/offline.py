"""Offline generator for the weather bot: canned, plan-aware answers for every generator
schema, consistent with the scripted agent model and fixtures. The smallest end-to-end
target (one ReAct node, two tools) — used by the small-agent e2e and UI tests:
`models.generator: scripted:examples.weather_bot.offline:generator_model`.
"""

from __future__ import annotations

import json

from evalbuilder.testing import SchemaScriptedModel, cells_in_prompt

WEATHER = {"temp": 72, "condition": "sunny", "city": "Paris"}
ALERTS = {"alerts": ["heat advisory"], "city": "Paris"}


def agent_map(_messages=None) -> dict:
    return {
        "intents": [
            {"id": "intent.conditions", "description": "Current weather conditions for a city", "evidence": ["tool:get_weather", "prompt:build_agent"]},
            {"id": "intent.alerts", "description": "Active weather alerts for a city", "evidence": ["tool:get_alerts", "prompt:build_agent"]},
        ],
        "scenarios": [
            {"id": "scenario.conditions.happy", "intent": "intent.conditions", "kind": "happy",
             "description": "asks for the weather in a named city", "expected_behavior": "names the city with temperature and condition", "evidence": ["tool:get_weather"]},
            {"id": "scenario.conditions.service-down", "intent": "intent.conditions", "kind": "failure",
             "failure_mode": "tool_error_handling", "description": "the weather service errors", "expected_behavior": "apologises without inventing data", "evidence": ["prompt:build_agent"]},
            {"id": "scenario.alerts.happy", "intent": "intent.alerts", "kind": "happy",
             "description": "asks for alerts in a named city", "expected_behavior": "lists the alerts", "evidence": ["tool:get_alerts"]},
            {"id": "scenario.alerts.service-down", "intent": "intent.alerts", "kind": "failure",
             "failure_mode": "tool_error_handling", "description": "the alerts service errors", "expected_behavior": "apologises", "evidence": ["prompt:build_agent"]},
        ],
        "failure_scenarios": [{"failure_type": "out_of_scope", "rationale": "the prompt limits the bot to weather", "evidence": ["prompt:build_agent"]}],
        "topics": [],
        "derived_constraints": ["Always name the city in the answer."],
    }


def mock_fixtures(_messages=None) -> dict:
    return {"tools": [
        {"name": "get_weather", "default_response": json.dumps(WEATHER), "variants": []},
        {"name": "get_alerts", "default_response": json.dumps(ALERTS), "variants": []},
    ]}


def _case(cell, msg, tools, contains, forbidden=(), overrides=(), variant="happy", turns=()):
    return {"cell": cell, "user_message": msg, "user_turns": list(turns), "variant": variant,
            "expected_tools": [{"name": t, "args": "{}"} for t in tools], "forbidden_tools": list(forbidden),
            "contains": contains, "contract": "Names the city; never invents weather data.", "expected_response": "see contains",
            "mock_overrides": list(overrides), "evidence": ["scenario"]}


ERROR = [{"tool": "get_weather", "match_args": "{}", "response": json.dumps({"error": "service unavailable"})}]
ALERT_ERROR = [{"tool": "get_alerts", "match_args": "{}", "response": json.dumps({"error": "service unavailable"})}]
HAPPY = ["What is the weather in Paris?", "How is the weather in Paris right now?", "Tell me the weather in Paris please"]
OUT_OF_SCOPE = ["Book me a table for two tonight", "What is the capital of Peru?", "Write a haiku about rain"]


def cases(messages) -> dict:
    cells = cells_in_prompt(messages[-1].content)
    answer = []
    for i, c in enumerate(cells):
        multi = int(c.get("of_which_multi_turn", 0) or 0)
        for k in range(int(c.get("count", 1))):
            turns = ["And are there any alerts for Paris?"] if k < multi else []
            if c.get("kind") == "schema-edge":
                if c.get("tool") == "get_alerts":
                    answer.append(_case(i, "Any weather alerts right now?", [], "city", forbidden=["get_alerts"], variant="boundary"))
                else:
                    answer.append(_case(i, "What is the weather like today?", [], "city", forbidden=["get_weather"], variant="boundary"))
            elif c["scenario"] == "scenario.conditions.happy":
                answer.append(_case(i, HAPPY[k % len(HAPPY)], ["get_weather", "get_alerts"] if turns else ["get_weather"], "Paris", turns=turns))
            elif c["scenario"] == "scenario.conditions.service-down":
                answer.append(_case(i, "What is the weather in Paris?", ["get_weather"], "could not", overrides=ERROR, variant="boundary"))
            elif c["scenario"] == "scenario.alerts.happy":
                answer.append(_case(i, "Any alerts for Paris?", ["get_alerts"], "heat advisory"))
            elif c["scenario"] == "scenario.alerts.service-down":
                answer.append(_case(i, "Any alerts for Paris?", ["get_alerts"], "could not", overrides=ALERT_ERROR, variant="boundary"))
            elif c.get("failure_mode") == "out_of_scope":
                answer.append(_case(i, OUT_OF_SCOPE[k % len(OUT_OF_SCOPE)], [], "weather", forbidden=["get_weather", "get_alerts"], variant="adversarial"))
            else:
                answer.append(_case(i, HAPPY[k % len(HAPPY)], ["get_weather"], "Paris"))
    return {"cases": answer}


def simulation_scenarios(_messages=None) -> dict:
    return {"scenarios": [
        {"id": "weather-then-alerts", "intent": "intent.conditions", "persona": "traveller", "goal": "weather and alerts for Paris",
         "opening": "What is the weather in Paris?", "followups": ["Any alerts for Paris?"], "max_turns": 3,
         "success_contains": "heat advisory", "expect_contains": "Paris", "expect_not_contains": ""},
    ]}


def analysis(_messages=None) -> dict:
    return {"summary": "Offline analysis: the weather bot names the city and never invents data.",
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
