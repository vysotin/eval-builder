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
