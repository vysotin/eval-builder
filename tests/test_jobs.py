"""Background pipeline jobs: argv per mode, spawn/track/finalise, status reading."""

import json
import sys
import time
from pathlib import Path

import pytest

from evalbuilder.pipeline import jobs


def test_job_argv_per_mode():
    cfg = Path("eval/p.yaml")
    assert jobs.job_argv(cfg, "full")[-3:] == ["pipeline", "run", "eval/p.yaml"]
    assert jobs.job_argv(cfg, "dataset")[-2:] == ["--until", "dataset"]
    assert jobs.job_argv(cfg, "resume")[-1] == "--resume"
    assert jobs.job_argv(cfg, "regenerate", from_stage="map")[-5:] == ["--resume", "--from", "map", "--until", "dataset"]
    with pytest.raises(ValueError):
        jobs.job_argv(cfg, "nope")
    with pytest.raises(ValueError):
        jobs.job_argv(cfg, "regenerate", from_stage="bogus")


def _wait(out_dir, status, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = jobs.job_status(out_dir)
        if st["status"] == status:
            return st
        time.sleep(0.1)
    raise AssertionError(f"job did not reach {status}: {jobs.job_status(out_dir)}")


def test_start_job_runs_wrapper_and_finalises(tmp_path):
    out = tmp_path / "out"
    assert jobs.job_status(out)["status"] == "none"
    job = jobs.start_job(tmp_path / "cfg.yaml", out, mode="full",
                         argv=[sys.executable, "-c", "import sys; print('hello from job'); sys.exit(3)"])
    assert job["pid"] and job["mode"] == "full" and (out / "job.json").exists()
    st = _wait(out, "finished")
    assert st["exit_code"] == 3 and st["job"]["finished_at"]
    assert "hello from job" in jobs.tail_log(out)
    assert json.loads((out / "job.json").read_text())["schema"] == jobs.JOB_SCHEMA


def test_running_job_blocks_a_second_start(tmp_path):
    out = tmp_path / "out"
    jobs.start_job(tmp_path / "cfg.yaml", out, mode="dataset",
                   argv=[sys.executable, "-c", "import time; time.sleep(2)"])
    st = _wait(out, "running")
    assert st["job"]["until"] == "dataset" and st["exit_code"] is None
    with pytest.raises(RuntimeError, match="already running"):
        jobs.start_job(tmp_path / "cfg.yaml", out, mode="full", argv=[sys.executable, "-c", "pass"])
    _wait(out, "finished")
    jobs.start_job(tmp_path / "cfg.yaml", out, mode="full", argv=[sys.executable, "-c", "pass"])
    assert _wait(out, "finished")["exit_code"] == 0


def test_stage_statuses_and_stopped_after(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "state.json").write_text(json.dumps({
        "schema": "evalbuilder/pipeline-state/v1", "name": "x",
        "stages": {"discover": {"status": "ok", "seconds": 1.5, "attempts": 1}, "map": {"status": "pending"}},
        "data": {"stopped_after": "dataset"},
    }))
    st = jobs.job_status(out)
    assert st["status"] == "none" and st["stages"]["discover"]["status"] == "ok" and st["stopped_after"] == "dataset"
    (out / "job.json").write_text(json.dumps({"schema": jobs.JOB_SCHEMA, "pid": 99999999, "finished_at": None}))
    assert jobs.job_status(out)["status"] == "lost"


def test_real_job_argv_runs_the_cli_module(tmp_path):
    """`python -m evalbuilder.cli pipeline run` must actually run (module entry point)."""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("name: only\n")
    out = tmp_path / "out"
    jobs.start_job(cfg, out, mode="dataset")
    st = _wait(out, "finished", timeout=60)
    assert st["exit_code"] == 2, jobs.tail_log(out)
    assert "invalid pipeline config" in jobs.tail_log(out)


def _state(out, stages, data=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "state.json").write_text(json.dumps({
        "schema": "evalbuilder/pipeline-state/v1", "name": "x",
        "stages": {k: {"status": v} for k, v in stages.items()},
        "data": data or {},
    }))


def test_job_status_reports_running_stage_and_progress(tmp_path):
    import os

    out = tmp_path / "out"
    _state(out, {"discover": "ok", "map": "running", "mocks": "pending", "dataset": "pending"})
    st = jobs.job_status(out)
    assert st["progress"] == {"done": 1, "total": 4, "fraction": 0.25, "stage": "map"}
    (out / "job.json").write_text(json.dumps(
        {"schema": jobs.JOB_SCHEMA, "pid": os.getpid(), "finished_at": None}))
    st = jobs.job_status(out)
    assert st["status"] == "running" and st["running_stage"] == "map"
    assert st["progress"]["stage"] == "map" and st["progress"]["total"] == 4


def test_job_status_flags_interrupted_stage_when_job_is_gone(tmp_path):
    out = tmp_path / "out"
    _state(out, {"discover": "ok", "run": "running"})
    (out / "job.json").write_text(json.dumps(
        {"schema": jobs.JOB_SCHEMA, "pid": 99999999, "finished_at": None}))
    st = jobs.job_status(out)
    assert st["status"] == "lost" and st["running_stage"] is None
    assert st["interrupted_stage"] == "run"


def test_stop_job_kills_the_process_and_finalises(tmp_path):
    out = tmp_path / "out"
    assert jobs.stop_job(out)["status"] == "none"  # nothing to stop
    jobs.start_job(tmp_path / "cfg.yaml", out, mode="full",
                   argv=[sys.executable, "-c", "import time; time.sleep(60)"])
    st = _wait(out, "running")
    pid = st["job"]["pid"]
    res = jobs.stop_job(out)
    assert res["status"] == "stopped"
    st = jobs.job_status(out)
    assert st["status"] == "stopped" and st["job"]["stopped"] and st["job"]["finished_at"]
    assert not jobs._alive(pid)
    assert jobs.stop_job(out)["status"] == "stopped"  # idempotent
    assert "stopped by user" in jobs.tail_log(out)


def test_stop_job_after_finish_is_a_noop(tmp_path):
    out = tmp_path / "out"
    jobs.start_job(tmp_path / "cfg.yaml", out, mode="full", argv=[sys.executable, "-c", "pass"])
    _wait(out, "finished")
    assert jobs.stop_job(out)["status"] == "finished"
    assert not jobs.read_job(out).get("stopped")


def test_job_status_includes_run_progress(tmp_path):
    out = tmp_path / "out"
    _state(out, {"verify": "ok", "run": "running"})
    (out / "run-progress.json").write_text(json.dumps({
        "schema": "evalbuilder/run-progress/v1", "repeat": 1, "repeats": 3,
        "cases_total": 4, "cases_done": 2, "overall_total": 12, "overall_done": 2,
        "by_intent": {"intent.a": {"done": 2, "total": 4, "errors": 1}}, "runs": [], "current": [],
    }))
    st = jobs.job_status(out)
    assert st["run_progress"]["cases_done"] == 2 and st["run_progress"]["overall_total"] == 12
    (out / "run-progress.json").write_text("{broken")
    assert jobs.job_status(out)["run_progress"] is None
