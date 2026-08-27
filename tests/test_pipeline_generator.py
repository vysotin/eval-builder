"""Taxonomy gating, coverage planning, and the generator's validation/repair loops."""

import json
from pathlib import Path

from evalbuilder.artifacts import add_case
from evalbuilder.discover import discover_from_source
from evalbuilder.pipeline import generator as g
from evalbuilder.pipeline.config import CoverageConfig, PerIntent
from evalbuilder.pipeline.planning import CROSS_CUTTING, achieved, plan_cells, summarize_plan
from evalbuilder.pipeline.taxonomy import applicable_failure_types
from evalbuilder.schemas import Dataset, Target

SOURCE = Path("examples/support_bot/agent.py")


def _map():
    amap = discover_from_source(SOURCE)
    amap.intents = [
        {"id": "intent.order-status", "evidence": ["tool:lookup_order"]},
        {"id": "intent.refund", "evidence": ["tool:issue_refund"]},
    ]
    amap.scenarios = [
        {"id": "scenario.order-status.happy", "intent": "intent.order-status", "kind": "happy", "evidence": ["x"]},
        {"id": "scenario.order-status.unknown-order", "intent": "intent.order-status", "kind": "failure",
         "failure_mode": "tool_error_handling", "evidence": ["x"]},
        {"id": "scenario.refund.happy", "intent": "intent.refund", "kind": "happy", "evidence": ["x"]},
        {"id": "scenario.refund.no-confirmation", "intent": "intent.refund", "kind": "failure",
         "failure_mode": "constraint_violation", "evidence": ["x"]},
    ]
    return amap


def test_applicable_failure_types_are_structurally_gated():
    amap = discover_from_source(SOURCE)
    app = applicable_failure_types(amap, SOURCE.read_text(), ["Never refund without yes"], multi_turn=True)
    assert {"input_validation", "provider_error", "out_of_scope"} <= set(app)
    assert "branch_misrouting" in app and app["branch_misrouting"][0].startswith("edge:classify")
    assert "tool_error_handling" in app and "constraint_violation" in app
    assert "output_contract_violation" in app  # with_structured_output in source
    assert "retrieval_grounding" in app  # search_kb / lookup_order
    weather = discover_from_source(Path("examples/weather_bot/agent.py"))
    app2 = applicable_failure_types(weather, Path("examples/weather_bot/agent.py").read_text(), [], multi_turn=False)
    assert "branch_misrouting" not in app2 and "constraint_violation" not in app2 and "state_loss" not in app2


def test_plan_cells_respects_config_and_floor():
    cfg = CoverageConfig(total_cases=14, per_intent=PerIntent(happy=2, failure=1),
                         per_failure_category=1, out_of_intent=2, multi_turn_share=0.2)
    cells = plan_cells(cfg, _map(), ["out_of_scope", "tool_misuse"])
    summary = summarize_plan(cells)
    assert summary["cases"] >= 14
    kinds = summary["by_kind"]
    assert kinds["failure"] == 2 and kinds["out-of-intent"] == 2 and kinds["category"] == 1
    oos = next(c for c in cells if c.failure_mode == "out_of_scope")
    assert oos.intent == CROSS_CUTTING and oos.scenario == "failure"
    assert summary["multi_turn"] >= 1


def test_achieved_reports_gaps_and_multi_turn():
    cfg = CoverageConfig(total_cases=4, per_intent=PerIntent(happy=1, failure=1), out_of_intent=0, per_failure_category=0)
    cells = plan_cells(cfg, _map(), [])
    cases = [
        {"metadata": {"intent": "intent.refund", "topic": "unspecified", "scenario": "scenario.refund.happy",
                      "failure_mode": "none", "user_turns": ["yes"]}},
    ]
    result = achieved(cells, cases)
    assert result["covered"] == 1 and result["planned"] == 4
    assert result["multi_turn"]["have"] == 1
    assert any(gap["scenario"] == "scenario.order-status.happy" for gap in result["gaps"])


def _good_map_answer():
    return {
        "intents": [
            {"id": "intent.order-status", "description": "track", "evidence": ["tool:lookup_order"]},
        ],
        "scenarios": [
            {"id": "scenario.order-status.happy", "intent": "intent.order-status", "kind": "happy",
             "description": "d", "expected_behavior": "e", "evidence": ["tool:lookup_order"]},
            {"id": "scenario.order-status.error", "intent": "intent.order-status", "kind": "failure",
             "failure_mode": "tool_error_handling", "description": "d", "expected_behavior": "e",
             "evidence": ["tool:lookup_order"]},
        ],
        "failure_scenarios": [
            {"failure_type": "out_of_scope", "rationale": "r", "evidence": ["app:always"]},
        ],
        "topics": [],
        "derived_constraints": ["Always mention the order id"],
    }


def test_author_map_repairs_once_then_filters():
    bad = _good_map_answer()
    bad["scenarios"].append(
        {"id": "scenario.x", "intent": "intent.missing", "kind": "happy", "description": "",
         "expected_behavior": "", "evidence": ["x"]}
    )
    bad["failure_scenarios"].append({"failure_type": "retrieval_grounding", "rationale": "", "evidence": []})
    fake = g.FakeGenerator({"agent_map": [bad, _good_map_answer()]})
    amap = discover_from_source(SOURCE)
    applicable = {"out_of_scope": ["app:always"], "tool_error_handling": ["tool:lookup_order"]}
    cleaned, errors = g.author_map(fake, amap, "", [], applicable)
    assert errors == [] and len(fake.prompts) == 2
    assert "PROBLEMS" in fake.prompts[1][2] and "intent.missing" in fake.prompts[1][2]
    assert [s["id"] for s in cleaned["scenarios"]] == ["scenario.order-status.happy", "scenario.order-status.error"]

    # still-bad second answer: invalid entries are filtered and errors reported
    fake2 = g.FakeGenerator({"agent_map": [bad, bad]})
    cleaned2, errors2 = g.author_map(fake2, amap, "", [], applicable)
    assert errors2 and all(s["intent"] == "intent.order-status" for s in cleaned2["scenarios"])
    assert [f["failure_type"] for f in cleaned2["failure_scenarios"]] == ["out_of_scope"]


def test_author_mocks_wildcard_for_every_tool_and_generic_fallback():
    tools = [{"name": "lookup_order", "description": "", "args_schema": {}}, {"name": "search_kb", "description": "", "args_schema": {}}]
    fake = g.FakeGenerator({"mock_fixtures": [{"tools": [
        {"name": "lookup_order", "default_response": json.dumps({"order_id": "A1", "status": "shipped"}),
         "variants": [{"match_args": json.dumps({"order_id": "B2"}), "response": json.dumps({"order_id": "B2", "status": "lost"}), "purpose": "x"},
                      {"match_args": "not json", "response": "{}", "purpose": "bad"}]},
    ]}]})
    rules, problems = g.author_mocks(fake, tools)
    assert [r["matchArgs"] for r in rules["lookup_order"]] == [{"order_id": "B2"}, {}]
    assert rules["search_kb"][-1]["matchArgs"] == {} and rules["search_kb"][-1]["response"]["note"].startswith("generic")
    assert any("search_kb" in p for p in problems) and any("invalid JSON" in p for p in problems)

    broken = g.FakeGenerator({})  # raises → generic everywhere
    rules2, problems2 = g.author_mocks(broken, tools)
    assert all(r[-1]["response"]["ok"] for r in rules2.values()) and problems2


def _cases_answer(cell_idx, n, tool="lookup_order", extra=None):
    return {"cases": [
        {"cell": cell_idx, "user_message": f"Where is my order A100{i}?", "user_turns": [], "variant": "happy",
         "expected_tools": [{"name": tool, "args": json.dumps({"order_id": f"A100{i}"})}],
         "forbidden_tools": ["issue_refund"], "contains": f"A100{i}",
         "contract": "Mentions the order id and status.", "expected_response": "status", "mock_overrides": [],
         "evidence": ["scenario.order-status.happy"], **(extra or {})}
        for i in range(n)
    ]}


def test_author_cases_validates_and_repairs_missing_cells():
    amap = _map()
    cfg = CoverageConfig(total_cases=3, per_intent=PerIntent(happy=2, failure=1), out_of_intent=0, per_failure_category=0)
    cells = [c for c in plan_cells(cfg, amap, []) if c.intent == "intent.order-status"]
    happy = next(i for i, c in enumerate(cells) if c.kind == "happy")
    first = _cases_answer(happy, 1)
    first["cases"].append({**_cases_answer(happy, 1)["cases"][0], "expected_tools": [{"name": "nope", "args": "{}"}]})
    repair = _cases_answer(0, 1)
    repair["cases"][0]["mock_overrides"] = [{"tool": "lookup_order", "match_args": "{}", "response": json.dumps({"error": "timeout"})}]
    repair["cases"][0]["user_turns"] = ["and my other order?"]
    repair["cases"][0]["user_message"] = "Order A1000 is late — where is it now?"
    fake = g.FakeGenerator({"cases": [first, repair]})
    rules = {"lookup_order": [{"matchArgs": {}, "response": {"order_id": "A1000", "status": "x"}}]}
    cases, problems = g.author_cases(fake, cells, amap, [], rules, {s["id"]: s for s in amap.scenarios}, batch_size=10)
    assert len(fake.prompts) == 2 and "missing or invalid" in fake.prompts[1][2]
    assert any("unknown tool" in p for p in problems)
    multi = [c for c in cases if c["metadata"].get("user_turns")]
    assert multi and multi[0]["metadata"]["variant"] == "multi-turn"
    assert multi[0]["metadata"]["mocks"]["tools"]["lookup_order"][0]["response"] == {"error": "timeout"}
    assert all(c["reference_outputs"]["forbidden_tools"] == ["issue_refund"] for c in cases)
    # every produced case is addable through the CLI's normalizer
    ds = Dataset(name="t", dataset_type="final_response", target=Target(module="examples.support_bot.agent"))
    for c in cases:
        add_case(ds, c)
    assert len(ds.cases) == len(cases)


def test_self_review_and_scenarios_and_analysis_fallback():
    amap = _map()
    fake = g.FakeGenerator({
        "case_review": [{"reviews": [{"id": "case-1", "verdict": "reject", "reason": "ambiguous"}]}],
        "simulation_scenarios": [{"scenarios": [
            {"id": "Refund Flow", "intent": "intent.refund", "persona": "p", "goal": "g", "opening": "refund A1",
             "followups": ["yes"], "max_turns": 9, "success_contains": "R77", "expect_contains": "", "expect_not_contains": "issued without"},
            {"id": "bad", "intent": "x", "persona": "p", "goal": "g", "opening": "hi", "followups": [], "max_turns": 3,
             "success_contains": "", "expect_contains": "", "expect_not_contains": ""},
        ]}],
    })
    cases = [{"id": "case-1", "inputs": {}, "reference_outputs": {}, "metadata": {}}]
    reviews = g.self_review(fake, cases, amap, {}, [])
    assert reviews["case-1"]["verdict"] == "reject"
    scenarios, problems = g.author_scenarios(fake, amap, [], {})
    assert len(scenarios) == 1 and scenarios[0]["id"] == "refund-flow" and scenarios[0]["max_turns"] == 6
    assert scenarios[0]["expect"] == {"not_contains": "issued without"} and problems
    fallback = g.deterministic_analysis({"verdict": "fail", "failing_cases": [{"id": "case-1"}]}, "no model")
    assert fallback["failure_patterns"][0]["affected_cases"] == ["case-1"]


def test_plan_topics_rotate_and_out_of_intent_is_guaranteed():
    amap = _map()
    amap.data_domains["topics"] = ["orders", "returns", "warranty"]
    cfg = CoverageConfig(total_cases=1, per_intent=PerIntent(happy=1, failure=1), per_failure_category=0, out_of_intent=2, multi_turn_share=0)
    cells = plan_cells(cfg, amap, [])  # out_of_scope not in the map's failure list
    intent_cells = [c for c in cells if c.kind in ("happy", "failure")]
    assert sum(c.count for c in intent_cells) == 4  # not multiplied by 3 topics
    assert {c.topic for c in intent_cells} == {"orders", "returns", "warranty"}
    oos = [c for c in cells if c.failure_mode == "out_of_scope"]
    assert len(oos) == 1 and oos[0].count == 2 and oos[0].kind == "out-of-intent"
