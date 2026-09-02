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
    cfg = CoverageConfig(total_cases=14, per_intent=PerIntent(happy=2, failure=1), per_tool_edge_cases=0,
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
    cfg = CoverageConfig(total_cases=4, per_intent=PerIntent(happy=1, failure=1), out_of_intent=0, per_failure_category=0, per_tool_edge_cases=0)
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
             "description": "d", "expected_behavior": "e", "evidence": ["tool:lookup_order", "skill:product-troubleshooting"],
             "skills": ["refund-policy"]},
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
    # skills: listed names and `skill:` evidence are merged and validated against the map
    assert cleaned["scenarios"][0]["skills"] == ["refund-policy", "product-troubleshooting"]
    assert "skills" not in cleaned["scenarios"][1]
    assert '"skills"' in fake.prompts[0][2] and "refund-policy" in fake.prompts[0][2]  # the brief carries the skills

    # an unknown skill name is an error (repair), an unexercised skill only a reported problem
    unknown = _good_map_answer()
    unknown["scenarios"][0]["skills"] = ["nope"]
    fake3 = g.FakeGenerator({"agent_map": [unknown, _good_map_answer()]})
    cleaned3, errors3 = g.author_map(fake3, amap, "", [], applicable)
    assert len(fake3.prompts) == 2 and "unknown skill" in fake3.prompts[1][2] and errors3 == []
    unexercised = _good_map_answer()
    unexercised["scenarios"][0].pop("skills")
    unexercised["scenarios"][0]["evidence"] = ["tool:lookup_order"]
    fake4 = g.FakeGenerator({"agent_map": [unexercised]})
    _, errors4 = g.author_map(fake4, amap, "", [], applicable)
    assert len(fake4.prompts) == 1 and [e for e in errors4 if "not exercised" in e] == [
        "skill product-troubleshooting is not exercised by any scenario (list it in `skills` and cite skill:product-troubleshooting)",
        "skill refund-policy is not exercised by any scenario (list it in `skills` and cite skill:refund-policy)",
    ]

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
    cfg = CoverageConfig(total_cases=3, per_intent=PerIntent(happy=2, failure=1), out_of_intent=0, per_failure_category=0, per_tool_edge_cases=0)
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
    cfg = CoverageConfig(total_cases=1, per_intent=PerIntent(happy=1, failure=1), per_failure_category=0, out_of_intent=2, multi_turn_share=0, per_tool_edge_cases=0)
    cells = plan_cells(cfg, amap, [])  # out_of_scope not in the map's failure list
    intent_cells = [c for c in cells if c.kind in ("happy", "failure")]
    assert sum(c.count for c in intent_cells) == 4  # not multiplied by 3 topics
    assert {c.topic for c in intent_cells} == {"orders", "returns", "warranty"}
    oos = [c for c in cells if c.failure_mode == "out_of_scope"]
    assert len(oos) == 1 and oos[0].count == 2 and oos[0].kind == "out-of-intent"


def test_schema_scripted_model_dispatches_on_schema_title():
    from evalbuilder.pipeline.generator import Generator
    from evalbuilder.testing import SchemaScriptedModel, ScriptMissError, cells_in_prompt

    seen = {}

    def cases(messages):
        seen["cells"] = cells_in_prompt(messages[-1].content)
        return {"cases": []}

    model = SchemaScriptedModel(handlers={"agent_map": {"intents": []}, "cases": cases})
    gen = Generator(model)
    assert gen.ask("s", "u", {"title": "agent_map"}) == {"intents": []}
    gen.ask("s", "CELLS TO FILL:\n[{\"cell\": 0}]", {"title": "cases"})
    assert seen["cells"] == [{"cell": 0}] and model.calls == ["agent_map", "cases"]
    import pytest

    with pytest.raises(ScriptMissError):
        gen.ask("s", "u", {"title": "unknown"})
    assert model.invoke("hi").content == "ok"


def _edge_map():
    from evalbuilder.schemas import AgentMap
    from evalbuilder.tool_schemas import edge_cases

    tools = [
        {"name": "lookup_order", "description": "", "used_by": ["agent"],
         "args_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "limit": {"type": "integer", "maximum": 5}}, "required": ["order_id"]},
         "output_schema": {"type": "object", "title": "Order", "properties": {"order_id": {"type": "string"}, "total": {"type": "number"}}, "required": ["order_id", "total"]}},
        {"name": "search_kb", "description": "", "used_by": ["agent"],
         "args_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, "output_schema": {"type": "object"}},
    ]
    for t in tools:
        t["edge_cases"] = edge_cases(t)
    return AgentMap(
        tools=tools,
        intents=[{"id": "intent.order", "description": "", "evidence": ["tool:lookup_order"]}],
        scenarios=[{"id": "scenario.order.happy", "intent": "intent.order", "kind": "happy", "description": "", "expected_behavior": "", "evidence": ["tool:lookup_order"]}],
    )


def test_plan_adds_schema_edge_cells_per_tool():
    from evalbuilder.pipeline.config import CoverageConfig
    from evalbuilder.pipeline.planning import SCHEMA_EDGE, plan_cells, summarize_plan

    cfg = CoverageConfig(total_cases=1, per_intent={"happy": 1, "failure": 0}, per_failure_category=0, out_of_intent=0,
                         multi_turn_share=0, per_tool_edge_cases=2)
    cells = plan_cells(cfg, _edge_map(), [])
    edges = [c for c in cells if c.kind == SCHEMA_EDGE]
    assert [c.scenario for c in edges] == [
        "edge.lookup_order.missing_required.order_id", "edge.lookup_order.wrong_type.limit", "edge.search_kb.missing_required.query",
    ]
    assert edges[0].tool == "lookup_order" and edges[0].edge["kind"] == "missing_required"
    assert edges[0].related_intent == "intent.order" and edges[0].failure_mode == "input_validation"
    assert summarize_plan(cells)["by_kind"][SCHEMA_EDGE] == 3
    assert cells[0].to_dict()["edge"] is None and edges[1].to_dict()["edge"]["field"] == "limit"
    assert not [c for c in plan_cells(CoverageConfig(per_tool_edge_cases=0), _edge_map(), []) if c.kind == SCHEMA_EDGE]


def test_author_cases_injects_malformed_output_and_validates_expected_args():
    from evalbuilder.pipeline.generator import FakeGenerator, author_cases
    from evalbuilder.pipeline.planning import Cell
    from evalbuilder.tool_schemas import edge_cases

    amap = _edge_map()
    tool = amap.tools[0]
    malformed = next(e for e in edge_cases(tool) if e["kind"] == "malformed_output")
    missing = edge_cases(tool)[0]
    cells = [
        Cell(intent="cross-cutting", scenario=malformed["id"], failure_mode="tool_error_handling", kind="schema-edge", count=1,
             tool="lookup_order", edge=malformed),
        Cell(intent="cross-cutting", scenario=missing["id"], failure_mode="input_validation", kind="schema-edge", count=1,
             tool="lookup_order", edge=missing),
    ]
    gen = FakeGenerator({"cases": [{"cases": [
        {"cell": 0, "user_message": "Where is order A1?", "user_turns": [], "variant": "boundary",
         "expected_tools": [{"name": "lookup_order", "args": "{\"order_id\": \"A1\"}"}], "forbidden_tools": [],
         "contains": "", "contract": "Explains the incomplete result.", "expected_response": "…", "mock_overrides": [], "evidence": []},
        {"cell": 1, "user_message": "Where is my order?", "user_turns": [], "variant": "boundary",
         "expected_tools": [{"name": "lookup_order", "args": "{\"limit\": \"ten\"}"}], "forbidden_tools": ["lookup_order"],
         "contains": "", "contract": "Asks for the order id.", "expected_response": "…", "mock_overrides": [], "evidence": []},
    ]}]})
    cases, problems = author_cases(gen, cells, amap, [], {"lookup_order": []}, {}, guidance="USER INSTRUCTIONS:\nbe terse")
    assert len(cases) == 2
    injected = cases[0]["metadata"]["mocks"]["tools"]["lookup_order"]
    assert injected == [{"matchArgs": {}, "response": {"total": 1.0}}]  # order_id (required) dropped
    assert cases[0]["metadata"]["edge"] == {"id": malformed["id"], "kind": "malformed_output", "field": None}
    assert cases[0]["metadata"]["tool"] == "lookup_order"
    assert "mocks" not in cases[1]["metadata"]
    assert any("args violate args_schema" in p and "$.limit" in p for p in problems)
    title, system, user = gen.prompts[0]
    assert "USER INSTRUCTIONS:\nbe terse" in user and '"edge": {' in user and "malformed_output" in user
    assert "Schema-edge cells" in system


def test_author_mocks_validates_against_output_schema():
    from evalbuilder.pipeline.generator import FakeGenerator, author_mocks

    tools = _edge_map().tools
    gen = FakeGenerator({"mock_fixtures": [{"tools": [
        {"name": "lookup_order", "default_response": json.dumps({"order_id": "A1"}),  # missing total
         "variants": [
             {"match_args": json.dumps({"order_id": "A2"}), "response": json.dumps({"order_id": "A2", "total": 5}), "purpose": "ok"},
             {"match_args": json.dumps({"order_id": "A3"}), "response": json.dumps({"order_id": "A3", "total": "five"}), "purpose": "bad type"},
             {"match_args": json.dumps({"limit": 99}), "response": json.dumps({"order_id": "A4", "total": 1}), "purpose": "bad args"},
         ]},
        {"name": "search_kb", "default_response": json.dumps({"articles": []}), "variants": []},
    ]}]})
    rules, problems = author_mocks(gen, tools, guidance="REVIEWER FEEDBACK:\n- use EU order ids")
    assert rules["lookup_order"][0] == {"matchArgs": {"order_id": "A2"}, "response": {"order_id": "A2", "total": 5}}
    assert rules["lookup_order"][-1] == {"matchArgs": {}, "response": {"order_id": "order_id", "total": 1.0}}
    assert len(rules["lookup_order"]) == 2 and rules["search_kb"][-1]["response"] == {"articles": []}
    assert sum("dropped (schema)" in p for p in problems) == 2 and any("violated output_schema" in p for p in problems)
    assert "REVIEWER FEEDBACK" in gen.prompts[0][2] and "output_schema" in gen.prompts[0][2]


# ── mock strategies (layer 2) ──────────────────────────────────

ORDER_TOOL = {"name": "lookup_order", "description": "Look up an order.", "args_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
              "output_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "status": {"type": "string", "enum": ["shipped", "lost"]}}, "required": ["order_id", "status"]}}
KB_TOOL = {"name": "search_kb", "description": "Search articles.", "args_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}


def _strategies_answer():
    return {"world": "Acme: orders A1xxx exist.", "strategies": [
        {"id": "default", "description": "healthy", "tools": [
            {"name": "lookup_order", "behavior": "A1xxx are shipped.", "fallback_response": json.dumps({"order_id": "A1000", "status": "shipped"}),
             "examples": [{"args": json.dumps({"order_id": "A1001"}), "response": json.dumps({"order_id": "A1001", "status": "shipped"})},
                          {"args": json.dumps({"order_id": "A1002"}), "response": json.dumps({"order_id": "A1002", "status": "teleported"})},
                          {"args": "not json", "response": "{}"}]},
            {"name": "search_kb", "behavior": "Three articles about headsets.", "fallback_response": json.dumps({"articles": []}), "examples": []},
            {"name": "ghost_tool", "behavior": "n/a", "fallback_response": "{}", "examples": []},
        ]},
        {"id": "degraded", "description": "orders API stale", "tools": [
            {"name": "lookup_order", "behavior": "Everything is lost.", "fallback_response": json.dumps({"order_id": "A", "status": "gone"}), "examples": []},
        ]},
        {"id": "", "description": "no id", "tools": []},
    ]}


def test_author_strategies_validates_and_completes_the_default_strategy():
    fake = g.FakeGenerator({"mock_strategies": [_strategies_answer()]})
    strategies, problems = g.author_strategies(fake, [ORDER_TOOL, KB_TOOL], {"lookup_order": []})
    assert strategies["world"] == "Acme: orders A1xxx exist."
    assert list(strategies["strategies"]) == ["default", "degraded"]
    default = strategies["strategies"]["default"]
    assert set(default["tools"]) == {"lookup_order", "search_kb"}  # unknown tool dropped, every mockable tool present
    order = default["tools"]["lookup_order"]
    assert order["behavior"] == "A1xxx are shipped." and order["fallback_response"] == {"order_id": "A1000", "status": "shipped"}
    assert order["examples"] == [{"args": {"order_id": "A1001"}, "response": {"order_id": "A1001", "status": "shipped"}}]  # bad enum + bad JSON dropped
    degraded = strategies["strategies"]["degraded"]["tools"]["lookup_order"]
    assert degraded["behavior"] == "Everything is lost." and degraded["fallback_response"] == {"order_id": "order_id", "status": "shipped"}  # replaced by a schema sample
    assert any("ghost_tool" in p for p in problems) and any("teleported" in p or "not in enum" in p for p in problems)
    assert any("fallback_response" in p and "degraded" in p for p in problems) and any("without an id" in p for p in problems)

    # generator failure → a deterministic default strategy still covers every tool
    broken = g.FakeGenerator({})
    strategies2, problems2 = g.author_strategies(broken, [ORDER_TOOL, KB_TOOL], {})
    assert list(strategies2["strategies"]) == ["default"] and set(strategies2["strategies"]["default"]["tools"]) == {"lookup_order", "search_kb"}
    assert "Look up an order." in strategies2["strategies"]["default"]["tools"]["lookup_order"]["behavior"]
    assert strategies2["strategies"]["default"]["tools"]["lookup_order"]["fallback_response"] == {"order_id": "order_id", "status": "shipped"}
    assert problems2 and "strategy generation failed" in problems2[0]
    # the prompt carries the fixtures so strategies stay consistent with them
    assert "MOCK FIXTURES" in fake.prompts[0][2]


def test_author_mocks_can_skip_the_wildcard_default_for_the_llm_layer():
    tools = [{"name": "lookup_order", "description": "", "args_schema": {}}]
    answer = {"tools": [{"name": "lookup_order", "default_response": json.dumps({"order_id": "A1", "status": "shipped"}),
                         "variants": [{"match_args": json.dumps({"order_id": "B2"}), "response": json.dumps({"order_id": "B2", "status": "lost"}), "purpose": "x"}]}]}
    rules, _ = g.author_mocks(g.FakeGenerator({"mock_fixtures": [answer]}), tools, wildcard=False)
    assert rules["lookup_order"] == [{"matchArgs": {"order_id": "B2"}, "response": {"order_id": "B2", "status": "lost"}}]
    rules2, _ = g.author_mocks(g.FakeGenerator({"mock_fixtures": [answer]}), tools)
    assert rules2["lookup_order"][-1]["matchArgs"] == {}


def test_cases_and_scenarios_may_select_a_mock_strategy():
    amap = _map()
    cfg = CoverageConfig(total_cases=1, per_intent=PerIntent(happy=1, failure=0), out_of_intent=0, per_failure_category=0, per_tool_edge_cases=0)
    cells = [c for c in plan_cells(cfg, amap, []) if c.intent == "intent.order-status"][:1]
    answer = _cases_answer(0, 2)
    answer["cases"][0]["mock_strategy"] = "degraded"
    answer["cases"][1]["mock_strategy"] = "nope"
    fake = g.FakeGenerator({"cases": [answer]})
    cases, problems = g.author_cases(fake, cells, amap, [], {"lookup_order": []}, {s["id"]: s for s in amap.scenarios},
                                     strategies={"world": "w", "strategies": {"default": {}, "degraded": {}}})
    assert cases[0]["metadata"]["mocks"]["strategy"] == "degraded"
    assert "MOCK STRATEGIES" in fake.prompts[0][2] and "degraded" in fake.prompts[0][2]
    # the second case was trimmed by the cell count; an unknown id would be a reported problem
    trimmed = _cases_answer(0, 1)
    trimmed["cases"][0]["mock_strategy"] = "nope"
    cases2, problems2 = g.author_cases(g.FakeGenerator({"cases": [trimmed]}), cells, amap, [], {}, {s["id"]: s for s in amap.scenarios},
                                       strategies={"world": "w", "strategies": {"default": {}}})
    assert "mocks" not in cases2[0]["metadata"] and any("unknown mock strategy" in p for p in problems2)
    scen = {"scenarios": [{"id": "s1", "intent": "i", "persona": "p", "goal": "g", "opening": "o", "followups": [], "max_turns": 3,
                           "success_contains": "x", "expect_contains": "", "expect_not_contains": "", "mock_strategy": "degraded"},
                          {"id": "s2", "intent": "i", "persona": "p", "goal": "g", "opening": "o", "followups": [], "max_turns": 3,
                           "success_contains": "x", "expect_contains": "", "expect_not_contains": "", "mock_strategy": "nope"}]}
    scenarios, sproblems = g.author_scenarios(g.FakeGenerator({"simulation_scenarios": [scen]}), amap, [], {},
                                              strategies={"world": "w", "strategies": {"default": {}, "degraded": {}}})
    assert scenarios[0]["mock_strategy"] == "degraded" and "mock_strategy" not in scenarios[1] and any("unknown mock strategy" in p for p in sproblems)


def test_validate_map_skill_and_tool_failure_sections():
    answer = _good_map_answer()
    answer["skill_failures"] = [
        {"skill": "refund-policy", "failure_cases": [
            {"description": "issues the refund without confirmation", "failure_mode": "constraint_violation",
             "expected_behavior": "asks for an explicit yes", "evidence": ["skill:refund-policy"]},
            {"description": "bad mode", "failure_mode": "branch_misrouting", "expected_behavior": "x", "evidence": []},
            {"description": "no evidence given", "expected_behavior": "y", "evidence": []},
        ]},
        {"skill": "nope", "failure_cases": [{"description": "d", "expected_behavior": "e", "evidence": []}]},
    ]
    answer["tool_failures"] = [
        {"tool": "lookup_order", "scenarios": [
            {"description": "times out", "expected_behavior": "apologizes", "evidence": []},
        ]},
        {"tool": "ghost", "scenarios": [{"description": "d", "expected_behavior": "e", "evidence": []}]},
    ]
    applicable = {"out_of_scope": ["app:always"], "tool_error_handling": ["tool:lookup_order"],
                  "constraint_violation": ["constraint:yes"]}
    cleaned, errors = g._validate_map(
        answer, applicable, ["refund-policy", "product-troubleshooting"], ["lookup_order"]
    )
    # valid cases kept; missing evidence defaulted to skill:<name>; invalid failure_mode dropped
    cases = cleaned["skill_failures"]
    assert [sf["skill"] for sf in cases] == ["refund-policy"]
    assert [c["description"] for c in cases[0]["failure_cases"]] == [
        "issues the refund without confirmation", "no evidence given"]
    assert cases[0]["failure_cases"][1]["evidence"] == ["skill:refund-policy"]
    # tool scenarios: failure_mode defaults to tool_error_handling, evidence to tool:<name>
    tf = cleaned["tool_failures"]
    assert [t["tool"] for t in tf] == ["lookup_order"]
    assert tf[0]["scenarios"][0]["failure_mode"] == "tool_error_handling"
    assert tf[0]["scenarios"][0]["evidence"] == ["tool:lookup_order"]
    assert any("unknown skill 'nope'" in e for e in errors)
    assert any("unknown tool 'ghost'" in e for e in errors)
    assert any("non-applicable failure_mode 'branch_misrouting'" in e for e in errors)


def test_skill_brief_uses_instruction_tools_and_rules():
    brief = g.skill_brief({
        "name": "s", "description": "d", "prompt": "FULL BODY", "instruction": "THE SUMMARY",
        "summarized": True, "tools": ["a"], "rules": ["Never do X."], "used_by": ["n1"],
    })
    assert brief["instructions"] == "THE SUMMARY" and brief["instructions_summarized"] is True
    assert brief["tools"] == ["a"] and brief["rules"] == ["Never do X."]
