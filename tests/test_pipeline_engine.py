from types import SimpleNamespace

from evalbuilder.pipeline.engine import PipelineRunner, PipelineState, Stage, StageStop


def _ctx():
    return SimpleNamespace(calls=[])


def _ok(name, details=None):
    def fn(ctx):
        ctx.calls.append(name)
        return details or {}

    return fn


def _boom(name, times):
    counter = {"n": 0}

    def fn(ctx):
        ctx.calls.append(name)
        counter["n"] += 1
        if counter["n"] <= times:
            raise RuntimeError(f"{name} exploded {counter['n']}")
        return {"recovered_after": counter["n"]}

    return fn


def test_dependency_failure_skips_dependents_but_report_always_runs(tmp_path):
    stages = [
        Stage("a", _ok("a")),
        Stage("b", _boom("b", 5), deps=("a",)),
        Stage("c", _ok("c"), deps=("b",)),
        Stage("report", _ok("report"), always=True),
    ]
    ctx = _ctx()
    state = PipelineRunner(stages, tmp_path / "state.json", max_retries=1).run(ctx)
    assert state.stages["a"].status == "ok"
    assert state.stages["b"].status == "failed" and state.stages["b"].attempts == 2
    assert "exploded" in state.stages["b"].error
    assert state.stages["c"].status == "skipped" and "dependency b is failed" in state.stages["c"].reason
    assert state.stages["report"].status == "ok"
    assert ctx.calls == ["a", "b", "b", "report"]


def test_retry_with_recovery_hook_marks_recovered(tmp_path):
    notes = []

    def recover(ctx, exc, attempt):
        notes.append(attempt)
        return f"adjusted after {exc}"

    stages = [Stage("a", _boom("a", 1), recover=recover)]
    state = PipelineRunner(stages, tmp_path / "s.json", max_retries=1).run(_ctx())
    rec = state.stages["a"]
    assert rec.status == "recovered" and rec.attempts == 2 and rec.error is None
    assert rec.recovery_notes == ["adjusted after a exploded 1"] and notes == [1]
    assert rec.details == {"recovered_after": 2}


def test_config_skip_and_optional_failure_do_not_block(tmp_path):
    stages = [
        Stage("a", _ok("a")),
        Stage("sim", _boom("sim", 9), deps=("a",), optional=True),
        Stage("pub", _ok("pub"), deps=("a",)),
        Stage("z", _ok("z"), deps=("a",)),
    ]
    ctx = _ctx()
    state = PipelineRunner(stages, tmp_path / "s.json", skip={"pub"}, max_retries=0).run(ctx)
    assert state.stages["sim"].status == "failed" and state.stages["sim"].optional
    assert state.stages["pub"].status == "skipped" and "config" in state.stages["pub"].reason
    assert state.stages["z"].status == "ok"


def test_stage_stop_halts_pipeline_except_always(tmp_path):
    def gate(ctx):
        raise StageStop("awaiting_review", "no auto_approve in config")

    stages = [
        Stage("review", gate),
        Stage("run", _ok("run"), deps=("review",)),
        Stage("report", _ok("report"), always=True),
    ]
    ctx = _ctx()
    state = PipelineRunner(stages, tmp_path / "s.json").run(ctx)
    assert state.stages["review"].status == "awaiting_review"
    assert state.stages["run"].status == "skipped" and "review" in state.stages["run"].reason
    assert state.stages["report"].status == "ok"


def test_resume_uses_cached_successes(tmp_path):
    path = tmp_path / "s.json"
    stages = [Stage("a", _ok("a", {"artifacts": {"p": "x"}})), Stage("b", _boom("b", 9), deps=("a",))]
    ctx = _ctx()
    PipelineRunner(stages, path, max_retries=0).run(ctx)
    assert ctx.calls == ["a", "b"]

    stages2 = [Stage("a", _ok("a")), Stage("b", _ok("b"), deps=("a",))]
    ctx2 = _ctx()
    state = PipelineRunner(stages2, path, resume=True).run(ctx2)
    assert ctx2.calls == ["b"]  # a was cached
    assert state.stages["a"].artifacts == {"p": "x"} and state.stages["b"].status == "ok"
    assert PipelineState.load(path).stages["b"].status == "ok"


def test_details_and_artifacts_are_split(tmp_path):
    stages = [Stage("a", _ok("a", {"artifacts": {"map": "m.json"}, "count": 3}))]
    state = PipelineRunner(stages, tmp_path / "s.json").run(_ctx())
    assert state.stages["a"].artifacts == {"map": "m.json"}
    assert state.stages["a"].details == {"count": 3}


def test_resume_invalidate_from_reruns_later_stages(tmp_path):
    path = tmp_path / "s.json"
    stages = [Stage("a", _ok("a")), Stage("b", _ok("b"), deps=("a",)), Stage("c", _ok("c"), deps=("b",))]
    PipelineRunner(stages, path).run(_ctx())
    ctx = _ctx()
    PipelineRunner(stages, path, resume=True, invalidate_from="b").run(ctx)
    assert ctx.calls == ["b", "c"]
