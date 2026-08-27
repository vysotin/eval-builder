import json

import pytest
import yaml
from typer.testing import CliRunner

from evalbuilder import evaluators as ev
from evalbuilder.artifacts import add_case, save_json, set_review
from evalbuilder.cli import app
from evalbuilder.schemas import Case, CaseRun, Dataset, RunArtifact, Target


def _case(**md):
    return Case.model_validate(
        {
            "id": "case-x",
            "inputs": {"q": 1},
            "reference_outputs": {
                "expected_tools": [{"name": "get_weather", "args": {"city": "Paris"}}],
                "contains": "Paris",
            },
            "metadata": {
                "intent": "i.a",
                "topic": "t",
                "scenario": "s",
                "failure_mode": "none",
                "variant": "happy",
                **md,
            },
            "review": {"status": "approved"},
            "publication": {},
        }
    )


def _run(tool_calls, response="72F in Paris"):
    return CaseRun(
        case_id="case-x",
        outputs={"response": response},
        tool_calls=tool_calls,
        trajectory=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": response},
        ],
    )


def test_expected_tools_subsequence_and_args_subset():
    fns = dict(ev.build_evaluators([{"type": "expected_tools"}], "anthropic:x"))
    ok = fns["expected_tools"](
        _case(), _run([{"name": "get_weather", "args": {"city": "Paris", "units": "F"}}])
    )
    assert ok["score"] is True
    bad = fns["expected_tools"](
        _case(), _run([{"name": "get_weather", "args": {"city": "Oslo"}}])
    )
    assert bad["score"] is False and "get_weather" in bad["comment"]


def test_contains_and_json_valid():
    fns = dict(ev.build_evaluators([{"type": "contains"}, {"type": "json_valid"}], "m"))
    assert fns["contains"](_case(), _run([]))["score"] is True
    assert fns["json_valid"](_case(), _run([], response='{"a": 1}'))["score"] is True
    assert fns["json_valid"](_case(), _run([], response="not json"))["score"] is False


def test_trajectory_match_skips_without_reference():
    fns = dict(ev.build_evaluators([{"type": "trajectory_match"}], "m"))
    with pytest.raises(ev.EvaluatorUnavailable):
        fns["trajectory_match"](_case(), _run([]))


def test_trajectory_match_scores_with_reference():
    case = _case()
    case.reference_outputs["trajectory"] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "72F in Paris"},
    ]
    fns = dict(ev.build_evaluators([{"type": "trajectory_match"}], "m"))
    result = fns["trajectory_match"](case, _run([]))
    assert result["score"] is True


def test_judge_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fns = dict(ev.build_evaluators([{"type": "correctness"}], "anthropic:claude-x"))
    with pytest.raises(ev.EvaluatorUnavailable):
        fns["correctness"](_case(), _run([]))


def test_judge_uses_factory(monkeypatch):
    calls = {}

    def fake_make(prompt, model, key):
        def judge(**kwargs):
            calls.update(kwargs)
            return {"key": key, "score": 1.0, "comment": "ok"}

        return judge

    monkeypatch.setattr(ev, "_make_judge", fake_make)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    fns = dict(ev.build_evaluators([{"type": "correctness"}], "anthropic:claude-x"))
    out = fns["correctness"](_case(), _run([]))
    assert out["score"] == 1.0 and "outputs" in calls


def test_score_run_aggregates_and_slices():
    ds = Dataset(name="d", dataset_type="final_response", target=Target(module="m"))
    a = add_case(
        ds,
        {
            "inputs": {"q": 1},
            "reference_outputs": {"contains": "yes"},
            "metadata": {"intent": "i.a", "variant": "happy"},
        },
    )
    b = add_case(
        ds,
        {
            "inputs": {"q": 2},
            "reference_outputs": {"contains": "yes"},
            "metadata": {"intent": "i.b", "variant": "adversarial"},
        },
    )
    set_review(ds, [a.id, b.id], "approved", "t")
    run = RunArtifact(
        run_id="r1",
        dataset_path="p",
        dataset_name="d",
        mocked=False,
        case_runs=[
            CaseRun(case_id=a.id, outputs={"response": "yes!"}),
            CaseRun(case_id=b.id, outputs={"response": "no"}),
        ],
    )
    report = ev.score_run(run, ds, [{"type": "contains"}], "m")
    assert report.metrics["contains"]["n"] == 2
    assert report.metrics["contains"]["avg"] == 0.5
    assert report.slices["intent"]["i.a"]["contains"] == 1.0
    assert report.slices["variant"]["adversarial"]["contains"] == 0.0


def test_score_run_separates_evaluator_errors():
    ds = Dataset(name="d", dataset_type="final_response", target=Target(module="m"))
    a = add_case(ds, {"inputs": {"q": 1}})  # no contains reference
    set_review(ds, [a.id], "approved", "t")
    run = RunArtifact(
        run_id="r1", dataset_path="p", dataset_name="d", mocked=False,
        case_runs=[CaseRun(case_id=a.id, outputs={"response": "hi"})],
    )
    report = ev.score_run(run, ds, [{"type": "contains"}], "m")
    assert report.cases[0]["errors"]["contains"]
    assert report.metrics["contains"]["errors"] == 1
    assert report.metrics["contains"]["n"] == 0


def test_cli_score(tmp_path):
    runner = CliRunner()
    ds = Dataset(
        name="w",
        dataset_type="final_response",
        target=Target(module="examples.weather_bot.agent"),
    )
    c = add_case(
        ds,
        {
            "inputs": {
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}]
            },
            "reference_outputs": {
                "contains": "Paris",
                "expected_tools": [{"name": "get_weather", "args": {"city": "Paris"}}],
            },
            "metadata": {"intent": "i.weather", "variant": "happy"},
        },
    )
    set_review(ds, [c.id], "approved", "t")
    ds_path = tmp_path / "ds.json"
    save_json(ds_path, ds)

    r = runner.invoke(app, ["run", str(ds_path), "--out", str(tmp_path)])
    assert r.exit_code == 0, r.output
    run_path = json.loads(r.stdout)["path"]

    ev_yaml = tmp_path / "evaluators.yaml"
    ev_yaml.write_text(
        yaml.safe_dump(
            {"evaluators": [{"type": "expected_tools"}, {"type": "contains"}]}
        )
    )
    r = runner.invoke(
        app,
        ["score", run_path, "--dataset", str(ds_path), "--evaluators", str(ev_yaml),
         "--out", str(tmp_path)],
    )
    assert r.exit_code == 0, r.output
    report = json.loads(r.stdout)
    assert report["metrics"]["contains"]["avg"] == 1.0
    assert report["metrics"]["expected_tools"]["avg"] == 1.0
    assert (tmp_path / "report-r1.json").exists() or list(tmp_path.glob("report-*.json"))


def test_generic_openevals_type_resolves_named_prompt(monkeypatch):
    from openevals.prompts import CONCISENESS_PROMPT

    seen = {}

    def fake_make(prompt, model, key, **kwargs):
        seen.update(prompt=prompt, model=model, key=key)
        return lambda **kw: {"key": key, "score": True, "comment": "ok"}

    monkeypatch.setattr(ev, "_make_judge", fake_make)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fns = dict(
        ev.build_evaluators(
            [{"type": "openevals", "prompt": "CONCISENESS_PROMPT"}, {"type": "hallucination"}],
            "openai:gpt-test",
        )
    )
    assert set(fns) == {"conciseness", "hallucination"}
    assert fns["conciseness"](_case(), _run([]))["score"] is True
    assert seen["prompt"] == CONCISENESS_PROMPT and seen["key"] == "conciseness"


def test_unknown_evaluator_type_rejected():
    with pytest.raises(ValueError, match="unknown evaluator"):
        ev.build_evaluators([{"type": "no_such_thing"}], "openai:x")


def test_trajectory_llm_passes_trajectory(monkeypatch):
    seen = {}

    def fake_make(prompt, model, key):
        def judge(**kw):
            seen.update(kw)
            return {"key": key, "score": False, "comment": "wrong tool"}

        return judge

    monkeypatch.setattr(ev, "_make_trajectory_judge", fake_make)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fns = dict(ev.build_evaluators([{"type": "trajectory_llm"}], "openai:gpt-test"))
    out = fns["trajectory_llm"](_case(), _run([]))
    assert out["score"] is False and seen["outputs"][0]["role"] == "user"


def test_claude_cli_judge_ready_without_api_key(monkeypatch):
    from evalbuilder import claude_cli

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(claude_cli, "claude_available", lambda cli_path=None: True)
    monkeypatch.setattr(
        ev, "_make_judge",
        lambda prompt, model, key, **kw: (lambda **_: {"key": key, "score": 1, "comment": ""}),
    )
    fns = dict(ev.build_evaluators([{"type": "correctness"}], "claude-cli:sonnet"))
    assert fns["correctness"](_case(), _run([]))["score"] == 1


def test_expected_tools_forbidden():
    case = _case()
    case.reference_outputs = {"expected_tools": [], "forbidden_tools": ["issue_refund"]}
    fns = dict(ev.build_evaluators([{"type": "expected_tools"}], "openai:x"))
    bad = fns["expected_tools"](case, _run([{"name": "issue_refund", "args": {}}]))
    assert bad["score"] is False and "forbidden" in bad["comment"]
    assert fns["expected_tools"](case, _run([]))["score"] is True
