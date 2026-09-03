"""Background pipeline jobs for the UI: spawn `evalbuilder pipeline run …` detached,
track it through `<out_dir>/work/job.json` (`evalbuilder/pipeline-job/v1`) and
`<out_dir>/pipeline.log`, and read progress back from `work/state.json`.

The child is a tiny wrapper (`python -m evalbuilder.pipeline.jobs run <job.json>`) that
executes the real command and finalises job.json with the exit code, so the status is
correct even when the process that started it (a Streamlit session) is gone.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evalbuilder.pipeline.config import STAGE_NAMES
from evalbuilder.pipeline.layout import existing_path, path_for

JOB_SCHEMA = "evalbuilder/pipeline-job/v1"
TERMINAL_STAGE_STATUSES = ("ok", "recovered", "failed", "skipped", "awaiting_review")
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
    path = existing_path(Path(out_dir), "pipeline_job")
    if path is None:
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
    try:  # reap the wrapper when it is our zombie child (kill(0) succeeds on zombies)
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, PermissionError, OSError):
        pass
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
    """stage → {status, seconds, attempts, reason, error} from the state file ({} when absent)."""
    path = existing_path(Path(out_dir), "pipeline_state")
    if path is None:
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


def stage_progress(stages: dict[str, dict]) -> dict:
    """Overall stage progress for a progress bar: done/total/fraction + the running stage."""
    total = len(stages)
    done = sum(1 for r in stages.values() if r.get("status") in TERMINAL_STAGE_STATUSES)
    running = next((n for n, r in stages.items() if r.get("status") == "running"), None)
    return {"done": done, "total": total, "fraction": (done / total) if total else 0.0, "stage": running}


def run_progress(out_dir: Path) -> dict | None:
    """The run stage's live progress (work/run-progress.json), or None."""
    path = existing_path(Path(out_dir), "run_progress")
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def job_status(out_dir: Path) -> dict:
    """One dict the UI can render: status ∈ none | running | finished | stopped | lost,
    plus the job record, stage statuses, progress and the stage the pipeline stopped
    after (if any). `interrupted_stage` is a stage left 'running' by a dead job."""
    out_dir = Path(out_dir)
    job = read_job(out_dir)
    stages = stage_statuses(out_dir)
    progress = stage_progress(stages)
    if job is None:
        return {"status": "none", "job": None, "stages": stages, "progress": progress,
                "run_progress": run_progress(out_dir), "stopped_after": _stopped_after(out_dir)}
    if job.get("finished_at") is not None:
        status = "stopped" if job.get("stopped") else "finished"
    elif _alive(job.get("pid")):
        status = "running"
    else:
        status = "lost"  # started but the wrapper died without finalising
    running_stage = progress["stage"] or next((n for n, r in stages.items() if r.get("status") == "pending"), None)
    return {
        "status": status,
        "job": job,
        "stages": stages,
        "progress": progress,
        "run_progress": run_progress(out_dir),
        "running_stage": running_stage if status == "running" else None,
        "interrupted_stage": progress["stage"] if status != "running" else None,
        "stopped_after": _stopped_after(out_dir),
        "exit_code": job.get("exit_code"),
    }


def _terminate(pid: int, timeout: float = 5.0) -> None:
    """SIGTERM the job's process group (wrapper + pipeline + model subprocesses),
    escalating to SIGKILL when it does not die within `timeout`."""

    def _signal(sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    _signal(signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.1)
    _signal(signal.SIGKILL)
    deadline = time.time() + timeout
    while time.time() < deadline and _alive(pid):
        time.sleep(0.1)


def stop_job(out_dir: Path) -> dict:
    """Stop the running job (kill its process group) and finalise job.json with
    `stopped: true`. No-op when there is no job or it already finished."""
    out_dir = Path(out_dir)
    job = read_job(out_dir)
    if job is None:
        return {"status": "none", "job": None}
    if job.get("finished_at") is not None:
        return {"status": "stopped" if job.get("stopped") else "finished", "job": job}
    pid = job.get("pid")
    if pid and _alive(pid):
        _terminate(pid)
    job = read_job(out_dir) or job  # the wrapper may have finalised in the meantime
    job["stopped"] = True
    job["stopped_at"] = _now()
    if job.get("finished_at") is None:
        job["finished_at"] = _now()
    if job.get("exit_code") is None:
        job["exit_code"] = -signal.SIGTERM
    _write_job(out_dir, job)
    with open(log_path(out_dir), "a") as log:
        log.write(f"\n=== {job['stopped_at']} stopped by user (pid {pid})\n")
    return {"status": "stopped", "job": job}


def _stopped_after(out_dir: Path) -> str | None:
    path = existing_path(Path(out_dir), "pipeline_state")
    if path is None:
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
