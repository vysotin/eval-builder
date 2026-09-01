"""Two-layer mocking: deterministic rules first, the LLM mock engine on a miss — with
strategy selection, output-schema validation, one repair round, fallback/strict policies,
a per-call ledger, and skill loaders left untouched."""

import json

import pytest
from langchain_core.tools import tool

from evalbuilder import mock_engine as me
from evalbuilder import skills as sk
from evalbuilder.mocking import MockMissError, mockable, wrap_tool, wrap_tools
from evalbuilder.testing import SchemaScriptedModel
from evalbuilder.tool_schemas import describe_tool


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order."""
    return {"order_id": order_id, "status": "REAL"}


@tool
def get_weather(city: str) -> dict:
    """Weather for a city."""
    return {"temp": 1, "city": city}


ORDER_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}, "status": {"type": "string", "enum": ["shipped", "delivered", "lost"]},
                   "total": {"type": "number"}},
    "required": ["order_id", "status", "total"],
}
TOOL_SPECS = {
    "lookup_order": {**describe_tool(lookup_order), "output_schema": ORDER_SCHEMA},
    "get_weather": describe_tool(get_weather),  # dict return → no output schema
}
STRATEGIES = {
    "world": "Acme Store: orders A1000-A1999 exist; anything else is unknown.",
    "strategies": {
        "default": {"description": "healthy backend", "tools": {
            "lookup_order": {"behavior": "Known ids are shipped, total 10-500.", "examples": [{"args": {"order_id": "A1000"}, "response": {"order_id": "A1000", "status": "shipped", "total": 42.0}}],
                             "fallback_response": {"order_id": "A0000", "status": "lost", "total": 0.0}},
            "get_weather": {"behavior": "Mild weather everywhere.", "examples": [], "fallback_response": {"temp": 20, "city": "?"}},
        }},
        "degraded": {"description": "orders API slow, statuses stale", "tools": {
            "lookup_order": {"behavior": "Every order is 'lost'.", "examples": [], "fallback_response": {"order_id": "A0000", "status": "lost", "total": 0.0}},
        }},
    },
}
RULES = {"lookup_order": [{"matchArgs": {"order_id": "A1000"}, "response": {"order_id": "A1000", "status": "shipped", "total": 42.0}}]}


def _model(answers):
    """A scripted mock model: `answers(call, prompt)` decides what the LLM returns."""
    seen = []

    def handler(messages):
        prompt = messages[-1].content
        call = me.call_in_prompt(prompt)
        seen.append((call, prompt))
        return answers(call, prompt)

    model = SchemaScriptedModel(handlers={me.MOCK_RESPONSE_TITLE: handler})
    return model, seen


def _engine(answers, **kwargs):
    model, seen = _model(answers)
    return me.LLMMockEngine(model, STRATEGIES, TOOL_SPECS, **kwargs), seen


def test_call_in_prompt_round_trips_the_call_block():
    prompt = me.build_prompt("lookup_order", {"order_id": "A1"}, TOOL_SPECS["lookup_order"], STRATEGIES["world"], "default",
                             STRATEGIES["strategies"]["default"], [], [])
    call = me.call_in_prompt(prompt)
    assert call == {"tool": "lookup_order", "args": {"order_id": "A1"}, "strategy": "default"}
    assert "WORLD:" in prompt and "Known ids are shipped" in prompt and '"enum"' in prompt and "A1000" in prompt
    assert me.call_in_prompt("no call here") is None


def test_rule_hit_never_reaches_the_engine_and_responses_are_copied():
    engine, seen = _engine(lambda call, prompt: {"order_id": "X", "status": "lost", "total": 1.0})
    ledger = []
    wrapped = wrap_tool(lookup_order, RULES["lookup_order"], on_miss="llm", engine=engine, ledger=ledger)
    out = wrapped.invoke({"order_id": "A1000"})
    assert out == {"order_id": "A1000", "status": "shipped", "total": 42.0} and seen == []
    out["status"] = "mutated"
    assert RULES["lookup_order"][0]["response"]["status"] == "shipped"
    assert ledger == [{"tool": "lookup_order", "args": {"order_id": "A1000"}, "layer": "rule", "rule": 0,
                       "response": {"order_id": "A1000", "status": "shipped", "total": 42.0}}]


def test_miss_goes_to_the_engine_and_is_validated_and_logged():
    engine, seen = _engine(lambda call, prompt: {"order_id": call["args"]["order_id"], "status": "delivered", "total": 99.5})
    ledger = []
    wrapped = wrap_tool(lookup_order, RULES["lookup_order"], on_miss="llm", engine=engine, ledger=ledger)
    out = wrapped.invoke({"order_id": "A1500"})
    assert out == {"order_id": "A1500", "status": "delivered", "total": 99.5}
    assert len(seen) == 1 and seen[0][0]["tool"] == "lookup_order"
    assert "PREVIOUS CALLS IN THIS CONVERSATION:\n[]" in seen[0][1]
    entry = ledger[-1]
    assert entry["layer"] == "llm" and entry["strategy"] == "default" and entry["valid"] is True
    assert entry["repairs"] == 0 and entry["fallback"] is False and entry["response"]["status"] == "delivered"
    assert entry["seconds"] >= 0 and engine.ledger == [entry]
    # consistency: the next call sees the previous one in its prompt
    wrapped.invoke({"order_id": "A1501"})
    assert "A1500" in seen[1][1] and "PREVIOUS CALLS" in seen[1][1]
    assert engine.stats() == {"calls": 2, "valid": 2, "repaired": 0, "fallback": 0, "errors": 0}


def test_invalid_answer_is_repaired_once_with_the_problems_listed():
    attempts = []

    def answers(call, prompt):
        attempts.append(prompt)
        if len(attempts) == 1:
            return {"order_id": "A2", "status": "teleported", "total": "n/a"}  # enum + type violations
        return {"order_id": "A2", "status": "shipped", "total": 5}

    engine, _ = _engine(answers)
    out = engine.respond("lookup_order", {"order_id": "A2"})
    assert out == {"order_id": "A2", "status": "shipped", "total": 5}
    assert len(attempts) == 2 and "PROBLEMS WITH YOUR PREVIOUS ANSWER" in attempts[1]
    assert "not in enum" in attempts[1] and "expected number" in attempts[1]
    assert engine.ledger[-1]["repairs"] == 1 and engine.ledger[-1]["valid"] is True
    assert engine.stats()["repaired"] == 1


def test_still_invalid_after_repair_falls_back_or_raises():
    bad = lambda call, prompt: {"order_id": "A3", "status": "teleported", "total": 1}  # noqa: E731
    engine, seen = _engine(bad, on_invalid="fallback")
    out = engine.respond("lookup_order", {"order_id": "A3"})
    assert out == STRATEGIES["strategies"]["default"]["tools"]["lookup_order"]["fallback_response"]
    assert len(seen) == 2 and engine.ledger[-1] == {**engine.ledger[-1], "valid": False, "fallback": True, "repairs": 1}
    assert engine.ledger[-1]["problems"] and engine.stats()["fallback"] == 1

    strict, _ = _engine(bad, on_invalid="strict", max_repairs=0)
    with pytest.raises(me.MockEngineError) as info:
        strict.respond("lookup_order", {"order_id": "A3"})
    assert "lookup_order" in str(info.value) and "not in enum" in str(info.value)
    assert strict.ledger[-1]["valid"] is False and strict.ledger[-1]["fallback"] is False and strict.stats()["errors"] == 1

    # without a strategy fallback the schema sample is used
    no_fallback = {"world": "w", "strategies": {"default": {"description": "d", "tools": {"lookup_order": {"behavior": "b"}}}}}
    model, _ = _model(bad)
    eng = me.LLMMockEngine(model, no_fallback, TOOL_SPECS, on_invalid="fallback", max_repairs=0)
    out = eng.respond("lookup_order", {"order_id": "A3"})
    assert out["status"] == "shipped" and set(out) == {"order_id", "status", "total"}


def test_tools_without_an_output_schema_use_the_json_envelope():
    engine, seen = _engine(lambda call, prompt: {"response_json": json.dumps({"temp": 31, "city": call["args"]["city"]})})
    assert engine.respond("get_weather", {"city": "Rome"}) == {"temp": 31, "city": "Rome"}
    assert engine.ledger[-1]["valid"] is True
    broken, _ = _engine(lambda call, prompt: {"response_json": "not json {"}, on_invalid="fallback", max_repairs=0)
    assert broken.respond("get_weather", {"city": "Rome"}) == {"temp": 20, "city": "?"}
    assert broken.ledger[-1]["valid"] is False and "invalid JSON" in broken.ledger[-1]["problems"][0]


def test_strategy_selection_and_generic_behaviour_without_strategies():
    engine, seen = _engine(lambda call, prompt: {"order_id": "A9", "status": "lost", "total": 0.0}, strategy="degraded")
    engine.respond("lookup_order", {"order_id": "A9"})
    assert "Every order is 'lost'" in seen[0][1] and engine.ledger[-1]["strategy"] == "degraded"
    # a tool the alternate strategy does not describe falls back to the default strategy's behaviour
    assert engine.tool_strategy("get_weather")["behavior"] == "Mild weather everywhere."
    unknown, _ = _engine(lambda call, prompt: {"order_id": "A9", "status": "lost", "total": 0.0}, strategy="nope")
    assert unknown.strategy == "default" and unknown.notes == ["unknown mock strategy 'nope'; using 'default'"]
    model, seen2 = _model(lambda call, prompt: {"order_id": "A9", "status": "lost", "total": 0.0})
    bare = me.LLMMockEngine(model, None, TOOL_SPECS)
    bare.respond("lookup_order", {"order_id": "A9"})
    assert "Look up an order." in seen2[0][1] and "realistic" in bare.tool_strategy("lookup_order")["behavior"]


def test_wrap_tools_with_an_engine_wraps_every_mockable_tool_and_skips_skill_loaders(tmp_path):
    (tmp_path / "s" / "SKILL.md").parent.mkdir(parents=True)
    (tmp_path / "s" / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nbody")
    loader = sk.skill_loader_tool(sk.load_skills(tmp_path))
    engine, seen = _engine(lambda call, prompt: {"response_json": json.dumps({"temp": 5, "city": "x"})})
    ledger = []
    wrapped = wrap_tools([lookup_order, get_weather, loader], RULES, on_miss="llm", engine=engine, ledger=ledger)
    assert [t.name for t in wrapped] == ["lookup_order", "get_weather", "load_skill"]
    assert wrapped[2] is loader and not mockable(loader) and mockable(get_weather)
    assert wrapped[1] is not get_weather  # no rules, but the engine answers for it
    assert wrapped[1].invoke({"city": "x"}) == {"temp": 5, "city": "x"} and ledger[-1]["layer"] == "llm"
    assert wrapped[2].invoke({"name": "s"}) == "body" and len(ledger) == 1
    # without an engine the legacy behaviour holds: only tools with rules are wrapped
    legacy = wrap_tools([lookup_order, get_weather, loader], RULES, on_miss="strict")
    assert legacy[1] is get_weather and legacy[2] is loader
    with pytest.raises(MockMissError):
        legacy[0].invoke({"order_id": "nope"})
    with pytest.raises(ValueError, match="engine"):
        wrap_tool(lookup_order, [], on_miss="llm")


def test_legacy_policies_are_logged_in_the_ledger():
    ledger = []
    real = wrap_tool(lookup_order, [], on_miss="real", ledger=ledger)
    assert real.invoke({"order_id": "Z"})["status"] == "REAL" and ledger[-1]["layer"] == "real"
    fb = wrap_tool(lookup_order, [], on_miss="fallback", fallback={"x": 1}, ledger=ledger)
    assert fb.invoke({"order_id": "Z"}) == {"x": 1} and ledger[-1]["layer"] == "fallback"
    strict = wrap_tool(lookup_order, [], on_miss="strict", ledger=ledger)
    with pytest.raises(MockMissError):
        strict.invoke({"order_id": "Z"})
    assert ledger[-1]["layer"] == "error" and "no mock rule" in ledger[-1]["error"]
