"""Every external command a deployment target runs goes through `CommandRunner`, which
logs `argv`, exit code, timing and the first lines of output to `commands.log` and keeps
a history the `deployment.json` record embeds. Tests replace it with a fake that
records the calls and answers with canned output — the targets never touch
`subprocess` themselves.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def summary(self) -> dict:
        return {"argv": list(self.argv), "returncode": self.returncode, "seconds": round(self.seconds, 3),
                "output": (self.stdout or self.stderr).strip()[-400:]}


class CommandError(RuntimeError):
    def __init__(self, result: CommandResult):
        self.result = result
        tail = (result.stderr or result.stdout).strip().splitlines()[-6:]
        super().__init__(f"`{' '.join(result.argv)}` exited {result.returncode}" + (": " + " | ".join(tail) if tail else ""))


@dataclass
class CommandRunner:
    """Logged, replaceable subprocess access for the deployment targets."""

    log_path: Path | None = None
    log: Callable[[str], None] = field(default=lambda msg: None)
    history: list[dict] = field(default_factory=list)

    def _record(self, result: CommandResult) -> None:
        self.history.append(result.summary())
        self.log(f"$ {' '.join(result.argv)} → {result.returncode} ({result.seconds:.1f}s)")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as fh:
                stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                fh.write(f"[{stamp}] $ {' '.join(result.argv)}\n  → exit {result.returncode} in {result.seconds:.1f}s\n")
                tail = (result.stdout or result.stderr).strip()
                if tail:
                    fh.write("  " + "\n  ".join(tail.splitlines()[-12:]) + "\n")

    def which(self, binary: str) -> str | None:
        return shutil.which(binary)

    def run(self, argv: list[str], *, input: bytes | None = None, check: bool = True, timeout: float | None = None,
            env: dict | None = None, cwd: str | Path | None = None) -> CommandResult:
        started = time.time()
        try:
            proc = subprocess.run(
                [str(a) for a in argv], input=input, capture_output=True, timeout=timeout,
                env={**os.environ, **(env or {})}, cwd=str(cwd) if cwd else None,
            )
            result = CommandResult(list(argv), proc.returncode, proc.stdout.decode(errors="replace"),
                                   proc.stderr.decode(errors="replace"), time.time() - started)
        except FileNotFoundError as e:
            result = CommandResult(list(argv), 127, "", f"{e}", time.time() - started)
        except subprocess.TimeoutExpired as e:
            result = CommandResult(list(argv), 124, (e.stdout or b"").decode(errors="replace"),
                                   f"timed out after {timeout}s", time.time() - started)
        self._record(result)
        if check and not result.ok:
            raise CommandError(result)
        return result

    def pipe(self, producer: list[str], consumer: list[str], *, check: bool = True, timeout: float | None = None,
             env: dict | None = None) -> CommandResult:
        """`producer | consumer` streamed through a pipe (no tarball in memory)."""
        started = time.time()
        full_env = {**os.environ, **(env or {})}
        try:
            src = subprocess.Popen([str(a) for a in producer], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=full_env)
            dst = subprocess.Popen([str(a) for a in consumer], stdin=src.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=full_env)
            src.stdout.close()  # let the consumer see EOF when the producer finishes
            out, err = dst.communicate(timeout=timeout)
            src_err = src.stderr.read().decode(errors="replace") if src.stderr else ""
            src.wait(timeout=timeout)
            code = dst.returncode if dst.returncode else src.returncode
            result = CommandResult([*producer, "|", *consumer], code, out.decode(errors="replace"),
                                   (err.decode(errors="replace") + src_err).strip(), time.time() - started)
        except FileNotFoundError as e:
            result = CommandResult([*producer, "|", *consumer], 127, "", f"{e}", time.time() - started)
        self._record(result)
        if check and not result.ok:
            raise CommandError(result)
        return result

    def spawn(self, argv: list[str], *, log_path: Path, env: dict | None = None, cwd: str | Path | None = None) -> int:
        """Start a long-lived process detached from this one (its own session); returns the pid."""
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as log:
            log.write(f"\n=== {datetime.now(timezone.utc).isoformat(timespec='seconds')} $ {' '.join(str(a) for a in argv)}\n")
            proc = subprocess.Popen(
                [str(a) for a in argv], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"}, cwd=str(cwd) if cwd else None,
                start_new_session=True,
            )
        self.history.append({"argv": [str(a) for a in argv], "returncode": None, "seconds": 0.0, "output": f"spawned pid {proc.pid} (log {log_path})"})
        self.log(f"$ {' '.join(str(a) for a in argv)} → pid {proc.pid}")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as fh:
                stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                fh.write(f"[{stamp}] $ {' '.join(str(a) for a in argv)}\n  → spawned pid {proc.pid}, output in {log_path}\n")
        return proc.pid

    def pid_alive(self, pid: int | None) -> bool:
        if not pid:
            return False
        try:  # reap a zombie child first: kill(0) succeeds on zombies
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

    def kill(self, pid: int | None, timeout: float = 5.0) -> None:
        """SIGTERM the process group, escalate to SIGKILL after `timeout`."""
        if not pid or not self.pid_alive(pid):
            return

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
        while time.time() < deadline and self.pid_alive(pid):
            time.sleep(0.1)
        if self.pid_alive(pid):
            _signal(signal.SIGKILL)
            deadline = time.time() + timeout
            while time.time() < deadline and self.pid_alive(pid):
                time.sleep(0.1)
        self.log(f"killed pid {pid}")
