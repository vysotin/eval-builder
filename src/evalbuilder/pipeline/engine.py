"""Stage engine: ordered stages with dependencies, retries, skip/stop, persisted state.

Every stage records a status so the report can say exactly what happened:
  ok · recovered (succeeded on retry) · failed · skipped (config or dependency) ·
  awaiting_review (stopped on purpose; needs a human decision)
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

OK_STATUSES = ("ok", "recovered")


class StageStop(Exception):
    """Raised by a stage to stop the pipeline deliberately (not a failure)."""

    def __init__(self, status: str, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


class StageRecord(BaseModel):
    status: str = "pending"
    attempts: int = 0
    seconds: float = 0.0
    error: str | None = None
    reason: str | None = None
    optional: bool = False
    artifacts: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    recovery_notes: list[str] = Field(default_factory=list)


class PipelineState(BaseModel):
    name: str
    started_at: str = ""
    updated_at: str = ""
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)

    def record(self, stage: str) -> StageRecord:
        return self.stages.setdefault(stage, StageRecord())

    def succeeded(self, stage: str) -> bool:
        return self.stages.get(stage, StageRecord()).status in OK_STATUSES

    def save(self, path: Path) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        return cls.model_validate(json.loads(Path(path).read_text()))


@dataclass
class Stage:
    name: str
    fn: Callable[[Any], dict | None]
    deps: tuple[str, ...] = ()
    optional: bool = False
    always: bool = False
    recover: Callable[[Any, Exception, int], str | None] | None = None
    retries: int | None = None  # None → runner default


@dataclass
class PipelineRunner:
    stages: list[Stage]
    state_path: Path
    skip: set[str] = field(default_factory=set)
    max_retries: int = 1
    resume: bool = False
    log: Callable[[str], None] = lambda msg: None

    def run(self, ctx: Any, name: str = "pipeline") -> PipelineState:
        if self.resume and Path(self.state_path).exists():
            state = PipelineState.load(self.state_path)
        else:
            state = PipelineState(name=name, started_at=datetime.now(timezone.utc).isoformat())
        ctx.state = state
        stopped: str | None = None

        for stage in self.stages:
            rec = state.record(stage.name)
            rec.optional = stage.optional

            if self.resume and rec.status in OK_STATUSES and not stage.always:
                self.log(f"{stage.name}: cached ({rec.status})")
                continue

            if stage.name in self.skip and not stage.always:
                self._mark(state, stage, "skipped", reason="skipped by config")
                continue

            if stopped and not stage.always:
                self._mark(state, stage, "skipped", reason=f"pipeline stopped at {stopped}")
                continue

            failed_deps = [d for d in stage.deps if not state.succeeded(d)]
            if failed_deps and not stage.always:
                blocker = failed_deps[0]
                blocker_status = state.stages.get(blocker, StageRecord()).status
                self._mark(
                    state, stage, "skipped",
                    reason=f"dependency {blocker} is {blocker_status}",
                )
                continue

            self._execute(ctx, state, stage)
            if state.stages[stage.name].status not in OK_STATUSES and not stage.optional:
                if state.stages[stage.name].status == "awaiting_review":
                    stopped = stage.name
        return state

    def _mark(self, state: PipelineState, stage: Stage, status: str, reason: str | None = None) -> None:
        rec = state.record(stage.name)
        rec.status = status
        rec.reason = reason
        self.log(f"{stage.name}: {status}" + (f" — {reason}" if reason else ""))
        state.save(self.state_path)

    def _execute(self, ctx: Any, state: PipelineState, stage: Stage) -> None:
        rec = state.record(stage.name)
        retries = self.max_retries if stage.retries is None else stage.retries
        rec.attempts = 0
        rec.error = None
        rec.reason = None
        started = time.time()
        while True:
            rec.attempts += 1
            try:
                self.log(f"{stage.name}: running (attempt {rec.attempts})")
                details = stage.fn(ctx) or {}
                rec.status = "ok" if rec.attempts == 1 else "recovered"
                rec.details = {k: v for k, v in details.items() if k != "artifacts"}
                rec.artifacts = dict(details.get("artifacts", {}))
                rec.error = None
                break
            except StageStop as stop:
                rec.status = stop.status
                rec.reason = stop.message
                break
            except Exception as e:  # noqa: BLE001 - every failure is recorded, never raised
                rec.error = f"{type(e).__name__}: {e}"
                rec.details.setdefault("traceback", traceback.format_exc()[-2000:])
                self.log(f"{stage.name}: error — {rec.error}")
                if rec.attempts > retries:
                    rec.status = "failed"
                    break
                if stage.recover is not None:
                    try:
                        note = stage.recover(ctx, e, rec.attempts)
                    except Exception as re_:  # noqa: BLE001
                        note = f"recovery hook failed: {type(re_).__name__}: {re_}"
                    if note:
                        rec.recovery_notes.append(note)
                        self.log(f"{stage.name}: recovery — {note}")
        rec.seconds = round(time.time() - started, 3)
        self.log(f"{stage.name}: {rec.status} ({rec.seconds}s)")
        state.save(self.state_path)
