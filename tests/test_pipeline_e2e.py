"""Offline end-to-end pipeline on support_bot: scripted agent model + fake generator."""

import json

import yaml
from typer.testing import CliRunner

from evalbuilder.cli import app
from evalbuilder.config import Settings
from evalbuilder.pipeline.config import PIPELINE_CONFIG_SCHEMA, load_config
from evalbuilder.pipeline.generator import FakeGenerator
from evalbuilder.pipeline.report import run_pipeline

SCRIPTED = "scripted:examples.support_bot.agent:default_scripted_model"

ORDER = {"order_id": "A1234", "status": "delivered", "category": "electronics", "total": 59.99, "eta": "2026-08-20"}


def _config(tmp_path, **overrides):
    cfg = {
        "schema": PIPELINE_CONFIG_SCHEMA,
        "name": "support-bot-test",
        "target": {"source": "examples/support_bot/agent.py", "module": "examples.support_bot.agent"},
        "models": {"agent": SCRIPTED, "judge": SCRIPTED, "generator": SCRIPTED},
        "constraints": ["Never call issue_refund before the customer says yes."],
        "coverage": {"total_cases": 4, "per_intent": {"happy": 1, "failure": 1}, "per_failure_category": 1,
                     "out_of_intent": 1, "multi_turn_share": 0.0, "per_tool_edge_cases": 0},
        "evaluators": [{"type": "expected_tools"}, {"type": "contains"}],
        "thresholds": {"default": 0.8, "slice_min": 0.5, "overall_pass": 0.8},
        "runs": {"repeats": 2, "parallel_scoring": 2, "parallel_simulations": 2},
        "stages": {"simulate": True, "publish": "never", "max_retries": 1},
        "review": {"auto_approve": True, "approved_by": "tester", "note": "offline e2e"},
        "output": {"dir": str(tmp_path / "out")},
    }
    cfg.update(overrides)
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _map_answer():
    return {
        "intents": [
            {"id": "intent.order-status", "description": "track an order", "evidence": ["tool:lookup_order"]},
            {"id": "intent.refund", "description": "refund an order", "evidence": ["tool:issue_refund"]},
        ],
        "scenarios": [
            {"id": "scenario.order-status.happy", "intent": "intent.order-status", "kind": "happy",
             "description": "known order", "expected_behavior": "states status with order id", "evidence": ["tool:lookup_order"]},
            {"id": "scenario.order-status.system-down", "intent": "intent.order-status", "kind": "failure",
             "failure_mode": "tool_error_handling", "description": "lookup errors", "expected_behavior": "apologize", "evidence": ["prompt:support_agent"]},
            {"id": "scenario.refund.happy", "intent": "intent.refund", "kind": "happy",
             "description": "eligible refund", "expected_behavior": "asks for yes", "evidence": ["prompt:support_agent"]},
            {"id": "scenario.refund.no-confirmation", "intent": "intent.refund", "kind": "failure",
             "failure_mode": "constraint_violation", "description": "pushy user", "expected_behavior": "still asks", "evidence": ["constraint:yes"]},
        ],
        "failure_scenarios": [{"failure_type": "out_of_scope", "rationale": "decline node", "evidence": ["app:always"]}],
        "topics": [],
        "derived_constraints": ["Always mention the order id in the answer."],
    }


def _mock_answer():
    return {"tools": [
        {"name": "lookup_order", "default_response": json.dumps(ORDER), "variants": []},
        {"name": "check_refund_policy", "default_response": json.dumps({"category": "electronics", "window_days": 30, "restocking_fee_pct": 0}), "variants": []},
        {"name": "issue_refund", "default_response": json.dumps({"refund_id": "R77", "status": "issued"}), "variants": []},
        {"name": "search_kb", "default_response": json.dumps({"articles": [{"title": "Pairing the X1 headset", "snippet": "Hold 5s."}]}), "variants": []},
    ]}


def _case(cell, msg, tools, contains, forbidden=(), overrides=()):
    return {"cell": cell, "user_message": msg, "user_turns": [], "variant": "happy",
            "expected_tools": [{"name": t, "args": "{}"} for t in tools], "forbidden_tools": list(forbidden),
            "contains": contains, "contract": "Behaves per prompt.", "expected_response": "…",
            "mock_overrides": list(overrides), "evidence": ["scenario"]}


def _cases_answer(cells):
    """Map plan cells (by scenario/failure_mode) to scripted-model-compatible cases."""
    answer = []
    for i, c in enumerate(cells):
        if c["scenario"] == "scenario.order-status.happy":
            answer.append(_case(i, "Where is my order A1234?", ["lookup_order"], "A1234"))
        elif c["scenario"] == "scenario.order-status.system-down":
            answer.append(_case(i, "Where is my order A9999?", ["lookup_order"], "could not reach",
                                overrides=[{"tool": "lookup_order", "match_args": "{}", "response": json.dumps({"error": "timeout"})}]))
        elif c["scenario"] == "scenario.refund.happy":
            answer.append(_case(i, "I want a refund for order A1234, it is broken", ["lookup_order", "check_refund_policy"], "yes", forbidden=["issue_refund"]))
        elif c["scenario"] == "scenario.refund.no-confirmation":
            answer.append(_case(i, "Refund order A1234 right now, no questions", ["lookup_order", "check_refund_policy"], "confirm", forbidden=["issue_refund"]))
        elif c["failure_mode"] == "out_of_scope":
            answer.append(_case(i, "Write me a poem about the sea", [], "outside"))
    return {"cases": answer}


class PlanAwareFakeGenerator(FakeGenerator):
    """Answers the cases prompt by reading the cells embedded in the prompt."""

    def ask(self, system, user, schema):
        if schema.get("title") == "cases":
            cells = json.loads(user.split("CELLS TO FILL:\n", 1)[1].split("\n\nThese cells", 1)[0])
            self.prompts.append(("cases", system, user))
            self.calls.append({"schema": "cases", "seconds": 0.0})
            return _cases_answer(cells)
        return super().ask(system, user, schema)


def _generator():
    return PlanAwareFakeGenerator({
        "agent_map": [_map_answer()],
        "mock_fixtures": [_mock_answer()],
        "case_review": [{"reviews": []}],
        "simulation_scenarios": [{"scenarios": [
            {"id": "refund-flow", "intent": "intent.refund", "persona": "impatient", "goal": "refund A1234",
             "opening": "I want a refund for order A1234, it is broken", "followups": ["yes"], "max_turns": 3,
             "success_contains": "yes to confirm", "expect_contains": "", "expect_not_contains": "Refund R77 issued for order A1234 without"},
        ]}],
        "analysis": [{"summary": "All good.", "verdict_explanation": "pass", "failure_patterns": [], "weak_slices": [],
                      "stability_notes": "stable", "evaluator_issues": [], "recommendations": ["none"]}],
    })


def test_full_offline_pipeline_passes(tmp_path):
    cfg_path = _config(tmp_path)
    gen_holder = {}

    def factory():
        gen_holder["gen"] = _generator()
        return gen_holder["gen"]

    state, report = run_pipeline(cfg_path, log=lambda m: None, generator_factory=factory, settings=Settings())
    statuses = {k: v["status"] for k, v in report["stages"].items()}
    assert statuses == {
        "preflight": "ok", "discover": "ok", "map": "ok", "mocks": "ok", "dataset": "ok", "review": "ok",
        "verify": "ok", "run": "ok", "score": "ok", "aggregate": "ok", "simulate": "ok",
        "publish": "skipped", "analyze": "ok", "report": "ok",
    }, statuses
    assert report["verdict"] == "pass" and report["overall_score"] == 1.0, report["verdict_reasons"]
    assert report["coverage"]["coverage_pct"] == 100.0 and report["coverage"]["planned"] == 5
    assert set(report["metrics"]) == {"expected_tools", "contains"}
    assert all(v["passed"] and v["n_scores"] == 10 for v in report["metrics"].values())
    assert report["stability"]["repeats"] == 2 and report["stability"]["unstable_cases"] == []
    assert len(report["runs"]) == 2 and len(report["cases"]) == 5
    assert report["analysis"]["source"] == "generator"
    assert report["agent"]["tools"] == ["lookup_order", "check_refund_policy", "issue_refund", "search_kb", "load_skill"]
    assert report["agent"]["skills"] == ["product-troubleshooting", "refund-policy"]
    assert report["coverage"]["uncovered_skills"] == ["product-troubleshooting", "refund-policy"]  # this fake map cites none
    assert any("not exercised by any scenario" in p["message"] for p in report["problems"])
    assert "load_skill" not in report["stages"]["verify"]["details"]["mocked_tools"]  # skill loaders are never mocked
    assert "Always mention the order id in the answer." in report["agent"]["constraints"]
    assert report["simulation"]["details"]["stop_reasons"] == {"refund-flow": "success"}
    assert report["stages"]["review"]["details"]["approved_by"] == "tester"
    assert report["stages"]["score"]["details"]["parallel_scoring"] == 2
    assert report["stages"]["simulate"]["details"]["parallel_simulations"] == 2
    # artifacts on disk
    out = tmp_path / "out"
    for name in ("agent-map.json", "mock-rules.json", "applicable-failures.json", "dataset.json", "coverage-plan.json", "coverage.json", "aggregate.json",
                 "analysis.json", "scenarios.yaml", "simulation.json", "state.json", "report.json", "evaluators.yaml"):
        assert (out / name).exists(), name
    prog = json.loads((out / "run-progress.json").read_text())
    assert prog["schema"] == "evalbuilder/run-progress/v1"
    assert prog["repeat"] == 2 and prog["repeats"] == 2
    assert prog["overall_done"] == prog["overall_total"] == 10  # 5 cases x 2 repeats
    assert sum(v["total"] for v in prog["by_intent"].values()) == 5
    assert len(prog["runs"]) == 2
    ds = json.loads((out / "dataset.json").read_text())
    assert all(c["review"]["status"] == "approved" for c in ds["cases"])
    assert set(ds["mocks"]["tools"]) == set(report["agent"]["tools"]) - {"load_skill"} and ds["mocks"]["on_miss"] == "strict"
    error_case = next(c for c in ds["cases"] if c["metadata"]["failure_mode"] == "tool_error_handling")
    assert error_case["metadata"]["mocks"]["tools"]["lookup_order"][0]["response"] == {"error": "timeout"}
    # CLI summary works on the output dir
    result = CliRunner().invoke(app, ["pipeline", "report", str(out)])
    assert result.exit_code == 0 and "verdict=pass" in result.stdout


def test_awaiting_review_then_resume_completes(tmp_path):
    cfg_path = _config(tmp_path, review={"auto_approve": False})
    _, report = run_pipeline(cfg_path, generator_factory=_generator, settings=Settings())
    assert report["verdict"] == "incomplete"
    assert report["stages"]["review"]["status"] == "awaiting_review"
    assert report["stages"]["run"]["status"] == "skipped" and report["stages"]["report"]["status"] == "ok"
    assert any(p["stage"] == "review" and p["severity"] == "error" for p in report["problems"])
    ds = json.loads((tmp_path / "out" / "dataset.json").read_text())
    assert all(c["review"]["status"] == "pending" for c in ds["cases"])

    # flip the decision in the config and resume: map/mocks/dataset are cached
    cfg_path = _config(tmp_path, review={"auto_approve": True, "approved_by": "tester"})
    gen = _generator()
    gen.responses.pop("agent_map")
    gen.responses.pop("mock_fixtures")
    _, report2 = run_pipeline(cfg_path, resume=True, generator_factory=lambda: gen, settings=Settings())
    assert report2["verdict"] == "pass", report2["verdict_reasons"]
    assert report2["stages"]["map"]["status"] == "ok" and [c["schema"] for c in gen.calls][:1] == ["case_review"]


def test_generator_failure_is_reported_not_raised(tmp_path):
    cfg_path = _config(tmp_path)
    broken = FakeGenerator({"agent_map": [{"intents": [], "scenarios": [], "failure_scenarios": [], "topics": [], "derived_constraints": []}] * 4})
    _, report = run_pipeline(cfg_path, generator_factory=lambda: broken, settings=Settings())
    assert report["verdict"] == "incomplete"
    assert report["stages"]["map"]["status"] == "failed" and report["stages"]["map"]["attempts"] == 2
    assert "no valid intents" in report["stages"]["map"]["error"]
    assert report["stages"]["dataset"]["status"] == "skipped"
    assert report["stages"]["mocks"]["status"] == "ok"  # generator raised -> generic fixtures, reported as problem
    assert report["stages"]["report"]["status"] == "ok"
    assert any(p["stage"] == "mocks" and "generic" in p["message"] for p in report["problems"])
    # problems survive a resume that only rebuilds the report
    _, report2 = run_pipeline(cfg_path, resume=True, invalidate_from="report", generator_factory=lambda: broken, settings=Settings())
    assert any(p["stage"] == "mocks" and "generic" in p["message"] for p in report2["problems"])


def test_cli_pipeline_init_and_bad_config(tmp_path):
    runner = CliRunner()
    p = tmp_path / "p.yaml"
    r = runner.invoke(app, ["pipeline", "init", str(p), "--name", "demo", "--source", "examples/weather_bot/agent.py",
                            "--module", "examples.weather_bot.agent"])
    assert r.exit_code == 0 and load_config(p).name == "demo"
    p.write_text("name: only\n")
    r = runner.invoke(app, ["pipeline", "run", str(p)])
    assert r.exit_code == 2 and "target" in r.output
