"""Background pipeline jobs for the UI: spawn `evalbuilder pipeline run …` detached,
track it through `<out_dir>/job.json` (`evalbuilder/pipeline-job/v1`) and
`<out_dir>/pipeline.log`, and read progress back from `state.json`.

The child is a tiny wrapper (`python -m evalbuilder.pipeline.jobs run <job.json>`) that
executes the real command and finalises job.json with the exit code, so the status is
correct even when the process that started it (a Streamlit session) is gone.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from evalbuilder.pipeline.config import STAGE_NAMES
from evalbuilder.pipeline.layout import path_for

JOB_SCHEMA = "evalbuilder/pipeline-job/v1"
MODES = ("full", "dataset", "resume", "regenerate")
GENERATION_STAGES = ("map", "mocks", "dataset")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def job_argv(config_path: Path, mode: str, from_stage: str | None = None, until: str | None = None) -> list[str]:
    """The `evalbuilder pipeline run` command for a job mode.

    full: everything · dataset: `--until dataset` (cases + mocks, no review/run) ·
    resume: `--resume` (continue after the stop / a decision) · regenerate: `--resume
    --from <stage>` (rerun the generation stages from `from_stage`, default `dataset`).
    """
    if mode not in MODES:
        raise ValueError(f"unknown job mode {mode!r} (expected one of {MODES})")
    argv = [sys.executable, "-m", "evalbuilder.cli", "pipeline", "run", str(config_path)]
    if mode == "dataset":
        argv += ["--until", until or "dataset"]
    elif mode == "resume":
        argv += ["--resume"]
        if until:
            argv += ["--until", until]
    elif mode == "regenerate":
        stage = from_stage or "dataset"
        if stage not in STAGE_NAMES:
            raise ValueError(f"unknown stage {stage!r}")
        argv += ["--resume", "--from", stage, "--until", until or "dataset"]
    return argv


def job_path(out_dir: Path) -> Path:
    return path_for(Path(out_dir), "pipeline_job")


def log_path(out_dir: Path) -> Path:
    return Path(out_dir) / "pipeline.log"


def read_job(out_dir: Path) -> dict | None:
    path = job_path(out_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def _write_job(out_dir: Path, job: dict) -> None:
    path = job_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2) + "\n")


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def start_job(
    config_path: Path,
    out_dir: Path,
    mode: str = "full",
    from_stage: str | None = None,
    until: str | None = None,
    env: dict | None = None,
    cwd: Path | None = None,
    argv: list[str] | None = None,
) -> dict:
    """Spawn the pipeline in the background; returns the job record (also on disk).
    `argv` overrides the command (tests)."""
    out_dir = Path(out_dir)
    current = read_job(out_dir)
    if current and current.get("finished_at") is None and _alive(current.get("pid")):
        raise RuntimeError(f"a job is already running for {out_dir} (pid {current['pid']})")
    argv = list(argv) if argv else job_argv(Path(config_path), mode, from_stage, until)
    out_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "schema": JOB_SCHEMA,
        "mode": mode,
        "from_stage": from_stage if mode == "regenerate" else None,
        "until": until or ("dataset" if mode in ("dataset", "regenerate") else None),
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "argv": argv,
        "log": str(log_path(out_dir)),
        "pid": None,
        "started_at": _now(),
        "finished_at": None,
        "exit_code": None,
    }
    _write_job(out_dir, job)
    with open(log_path(out_dir), "a") as log:
        log.write(f"\n=== {job['started_at']} {mode} :: {' '.join(argv)}\n")
    wrapper = [sys.executable, "-m", "evalbuilder.pipeline.jobs", "run", str(job_path(out_dir))]
    proc = subprocess.Popen(
        wrapper,
        cwd=str(cwd or Path.cwd()),
        env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    job["pid"] = proc.pid
    _write_job(out_dir, job)
    return job


def _run_wrapper(job_file: Path) -> int:
    """Child side: run argv with output to the log, then finalise job.json."""
    job = json.loads(Path(job_file).read_text())
    out_dir = Path(job["output_dir"])
    with open(job["log"], "a") as log:
        proc = subprocess.run(job["argv"], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    job = read_job(out_dir) or job
    job["exit_code"] = proc.returncode
    job["finished_at"] = _now()
    _write_job(out_dir, job)
    return proc.returncode


def stage_statuses(out_dir: Path) -> dict[str, dict]:
    """stage → {status, seconds, attempts, reason, error} from state.json ({} when absent)."""
    path = path_for(Path(out_dir), "pipeline_state")
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text())
    except ValueError:
        return {}
    out = {}
    for name, rec in (state.get("stages") or {}).items():
        out[name] = {
            "status": rec.get("status"), "seconds": rec.get("seconds", 0.0), "attempts": rec.get("attempts", 0),
            "reason": rec.get("reason"), "error": rec.get("error"), "details": rec.get("details") or {},
        }
    return out


def job_status(out_dir: Path) -> dict:
    """One dict the UI can render: status ∈ none | running | finished | lost, plus the
    job record, stage statuses and the stage the pipeline stopped after (if any)."""
    out_dir = Path(out_dir)
    job = read_job(out_dir)
    stages = stage_statuses(out_dir)
    if job is None:
        return {"status": "none", "job": None, "stages": stages, "stopped_after": _stopped_after(out_dir)}
    if job.get("finished_at") is not None:
        status = "finished"
    elif _alive(job.get("pid")):
        status = "running"
    else:
        status = "lost"  # started but the wrapper died without finalising
    running_stage = next((n for n, r in stages.items() if r.get("status") == "pending"), None)
    return {
        "status": status,
        "job": job,
        "stages": stages,
        "running_stage": running_stage if status == "running" else None,
        "stopped_after": _stopped_after(out_dir),
        "exit_code": job.get("exit_code"),
    }


def _stopped_after(out_dir: Path) -> str | None:
    path = path_for(Path(out_dir), "pipeline_state")
    if not path.exists():
        return None
    try:
        return (json.loads(path.read_text()).get("data") or {}).get("stopped_after")
    except ValueError:
        return None


def tail_log(out_dir: Path, lines: int = 80) -> str:
    path = log_path(out_dir)
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "run":
        return _run_wrapper(Path(argv[1]))
    print("usage: python -m evalbuilder.pipeline.jobs run <job.json>", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - wrapper entry point
    sys.exit(main())
