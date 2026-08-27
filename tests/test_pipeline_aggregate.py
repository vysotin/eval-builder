from evalbuilder.pipeline.aggregate import aggregate, output_hash
from evalbuilder.pipeline.config import ThresholdsConfig
from evalbuilder.schemas import Case, CaseRun, Dataset, Report, RunArtifact, Target


def _ds():
    def case(cid, intent, fm="none", variant="happy"):
        return Case(id=cid, inputs={"messages": [{"role": "user", "content": f"q-{cid}"}]},
                    metadata={"intent": intent, "failure_mode": fm, "variant": variant})

    return Dataset(name="d", dataset_type="final_response", target=Target(module="m"),
                   cases=[case("c1", "i.a"), case("c2", "i.b", "out_of_scope", "adversarial"), case("c3", "i.a")])


def _run(run_id, responses, errors=None, tools=None):
    errors = errors or {}
    tools = tools or {}
    return RunArtifact(run_id=run_id, dataset_path="p", dataset_name="d", mocked=True, case_runs=[
        CaseRun(case_id=cid, outputs={"response": text}, tool_calls=tools.get(cid, [{"name": "t", "args": {}}]),
                error=errors.get(cid), error_class="agent" if errors.get(cid) else "none")
        for cid, text in responses.items()
    ])


def _report(run_id, scores, errors=None, comments=None):
    errors = errors or {}
    comments = comments or {}
    return Report(run_id=run_id, dataset_name="d", cases=[
        {"case_id": cid, "scores": {m: {"score": s, "comment": comments.get((cid, m), "" if s else "bad")} for m, s in ms.items()},
         "errors": errors.get(cid, {})}
        for cid, ms in scores.items()
    ])


def test_aggregate_pass_rates_thresholds_slices_and_stability():
    runs = [
        (_run("r1", {"c1": "A", "c2": "decline", "c3": "same"}, tools={"c1": [{"name": "t", "args": {"a": 1}}]}),
         _report("r1", {"c1": {"contains": 1, "contract": 1}, "c2": {"contains": 0, "contract": 1}, "c3": {"contains": 1, "contract": 0}},
                 comments={("c3", "contract"): "Test reasoning. Thus, the score should be: false."})),
        (_run("r2", {"c1": "B", "c2": "decline", "c3": "same"}, tools={"c1": [{"name": "t", "args": {"a": 2}}]}),
         _report("r2", {"c1": {"contains": 0, "contract": 1}, "c2": {"contains": 0, "contract": 1}, "c3": {"contains": 1, "contract": 1}},
                 errors={"c2": {"correctness": "judge missing"}})),
    ]
    agg = aggregate(runs, _ds(), ThresholdsConfig(default=0.6, metrics={"contract": 0.9}, slice_min=0.5, overall_pass=0.7),
                    judge_metrics={"contract", "correctness"})
    m = agg["metrics"]
    assert m["contains"]["pass_rate"] == 0.5 and m["contains"]["passed"] is False
    assert m["contract"]["pass_rate"] == 0.8333 and m["contract"]["passed"] is False
    assert m["correctness"]["errors"] == 1 and m["correctness"]["pass_rate"] is None
    assert agg["overall_score"] == 0.6667
    assert agg["verdict"] == "fail" and any("contains" in r for r in agg["verdict_reasons"])
    # slices
    assert agg["slices"]["intent"]["i.b"]["metrics"]["contains"] == 0.0
    assert "intent=i.b" in agg["weak_slices"] and "failure_mode=out_of_scope" in agg["weak_slices"]
    # stability: c1 tool args differ (agent), c3 same trajectory but the judge disagrees (evaluator)
    st = agg["stability"]
    assert [u["id"] for u in st["unstable_cases"]] == ["c1"]
    assert st["unstable_cases"][0]["metrics_disagreeing"] == ["contains"]
    assert st["unstable_evaluators"] == [{"case_id": "c3", "metric": "contract", "scores": [0.0, 1.0]}]
    assert st["unstable_outputs"] == [] and st["text_varies"] == 1
    assert st["stable_case_fraction"] == 0.6667
    assert st["suspect_judge_comments"][0]["case_id"] == "c3"
    failing_ids = {f["id"] for f in agg["failing_cases"]}
    assert failing_ids == {"c1", "c2", "c3"}
    assert agg["evaluator_errors"] == {"correctness": 1}


def test_aggregate_pass_verdict_and_agent_errors():
    runs = [
        (_run("r1", {"c1": "ok", "c2": "ok", "c3": "ok"}, errors={"c3": "ValueError: boom"}),
         _report("r1", {"c1": {"contains": 1}, "c2": {"contains": 1}, "c3": {"contains": 0}})),
    ]
    agg = aggregate(runs, _ds(), ThresholdsConfig(default=0.6, slice_min=0.0, overall_pass=0.6))
    assert agg["verdict"] == "pass" and agg["metrics"]["contains"]["pass_rate"] == 0.6667
    c3 = next(f for f in agg["failing_cases"] if f["id"] == "c3")
    assert c3["agent_errors"] == ["agent: ValueError: boom"]


def test_wording_only_differences_are_not_trajectory_instability():
    from evalbuilder.pipeline.aggregate import trajectory_hash

    a = CaseRun(case_id="x", outputs={"response": "hi"}, tool_calls=[{"name": "t", "args": {"a": 1}}])
    b = CaseRun(case_id="x", outputs={"response": "hello"}, tool_calls=[{"name": "t", "args": {"a": 1}}])
    c = CaseRun(case_id="x", outputs={"response": "hi"}, tool_calls=[{"name": "t", "args": {"a": 2}}])
    assert output_hash(a) != output_hash(b) and trajectory_hash(a) == trajectory_hash(b)
    assert trajectory_hash(a) != trajectory_hash(c)
    runs = [(_run("r1", {"c1": "A"}), _report("r1", {"c1": {"contains": 1}})),
            (_run("r2", {"c1": "B"}), _report("r2", {"c1": {"contains": 0}}))]
    st = aggregate(runs, _ds(), ThresholdsConfig())["stability"]
    assert st["unstable_cases"] == [] and st["unstable_outputs"] == [{"case_id": "c1", "metric": "contains", "scores": [1.0, 0.0]}]
