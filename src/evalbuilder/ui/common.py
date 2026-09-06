"""Shared helpers for the report UI pages: palette, formatting, charts, small widgets."""

from __future__ import annotations

import json
from typing import Any, Iterable

import altair as alt
import pandas as pd
import streamlit as st

from evalbuilder.pipeline.layout import ARTIFACTS
from evalbuilder.ui.loader import Bundle

# Palette (dataviz reference instance): categorical slots 1-3, status colors, blue ramp.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARNING, CRITICAL, NEUTRAL = "#0ca30c", "#fab219", "#d03b3b", "#9a9892"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

VERDICT_COLORS = {"pass": "green", "fail": "red", "incomplete": "orange"}
STATUS_COLORS = {
    "ok": "green", "recovered": "blue", "failed": "red", "skipped": "grey",
    "awaiting_review": "orange", "pending": "grey", "running": "blue",
}
STATUS_ICONS = {
    "ok": ":material/check_circle:", "recovered": ":material/autorenew:", "failed": ":material/error:",
    "skipped": ":material/skip_next:", "awaiting_review": ":material/pending:", "pending": ":material/schedule:",
    "running": ":material/play_circle:",
}
JOB_COLORS = {"running": "blue", "finished": "green", "stopped": "red", "lost": "orange", "none": "grey"}


def get_bundle() -> Bundle | None:
    return st.session_state.get("bundle")


def require(bundle: Bundle | None, *kinds: str) -> bool:
    """Show a hint and return False when there is no project, the project has no
    artifacts yet, or any required artifact is missing."""
    from evalbuilder.ui import project

    if bundle is None:
        project.no_project_hint()
        return False
    if not any(bundle.has(k) for k in bundle.artifacts):
        st.info(f"No artifacts yet in `{bundle.source}` — run the pipeline from **Pipeline setup** (or wait for the running job).", icon=":material/hourglass_empty:")
        project.setup_link()
        return False
    missing = [k for k in kinds if not bundle.has(k)]
    if missing:
        for k in missing:
            a = ARTIFACTS[k]
            where = a.folded_into or a.file  # derived kinds live inside another artifact
            st.warning(
                f"Missing artifact **{where}** (`{k}`, written by stage *{a.stage}*): {a.description}",
                icon=":material/inventory_2:",
            )
        return False
    return True


def verdict_badge(verdict: str | None) -> None:
    if not verdict:
        st.badge("no verdict", color="grey")
        return
    icon = {"pass": ":material/check_circle:", "fail": ":material/cancel:", "incomplete": ":material/pending:"}
    st.badge(verdict, icon=icon.get(verdict), color=VERDICT_COLORS.get(verdict, "grey"))


def job_badge_line(status: dict) -> str:
    """`:color-badge[job <status>]` plus the running / interrupted stage."""
    color = JOB_COLORS.get(status.get("status"), "grey")
    line = f":{color}-badge[job {status.get('status')}]"
    if status.get("running_stage"):
        line += f" · running **{status['running_stage']}**"
    elif status.get("interrupted_stage"):
        line += f" · interrupted during **{status['interrupted_stage']}**"
    return line


def _run_stage_active(status: dict) -> bool:
    prog = status.get("progress") or {}
    return prog.get("stage") in ("infer", "run") or status.get("interrupted_stage") in ("infer", "run")


def job_progress(status: dict, compact: bool = False) -> None:
    """Visual job progress from `jobs.job_status`: stage-level progress bar and, while
    the run stage executes, a case-level bar. `compact=True` (sidebar) also renders the
    badge line and skips the per-intent breakdown; the Run & review page renders its own
    richer badge line instead."""
    if compact:
        st.markdown(job_badge_line(status))
    prog = status.get("progress") or {}
    if prog.get("total"):
        text = f"stages {prog['done']}/{prog['total']}" + (f" · {prog['stage']}" if prog.get("stage") else "")
        st.progress(min(1.0, prog["done"] / prog["total"]), text=text)
    rp = status.get("run_progress")
    if rp and _run_stage_active(status):
        total = rp.get("overall_total") or 0
        if total:
            st.progress(min(1.0, (rp.get("overall_done") or 0) / total),
                        text=f"cases {rp.get('overall_done', 0)}/{total} · repeat {rp.get('repeat')}/{rp.get('repeats')}")
        if not compact and rp.get("by_intent"):
            parts = [
                f"`{intent or '(no intent)'}` {v.get('done', 0)}/{v.get('total', 0)}"
                + (f" · {v['errors']} error(s)" if v.get("errors") else "")
                for intent, v in rp["by_intent"].items()
            ]
            st.caption("by intent: " + " · ".join(parts))


def status_md(status: str | None) -> str:
    """Inline badge markdown for a stage status."""
    color = STATUS_COLORS.get(status or "", "grey")
    return f":{color}-badge[{status or 'not run'}]"


def pct(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def frame(rows: Iterable[dict], columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if columns:
        for c in columns:
            if c not in df.columns:
                df[c] = None
        df = df[columns]
    return df


def table(rows: Iterable[dict], columns: list[str] | None = None, **kwargs: Any) -> None:
    df = frame(rows, columns)
    if df.empty:
        st.caption("nothing to show")
        return
    st.dataframe(df, hide_index=True, width="stretch", **kwargs)


def json_expander(label: str, obj: Any, expanded: bool = False) -> None:
    with st.expander(label, expanded=expanded):
        st.json(obj, expanded=2 if isinstance(obj, (dict, list)) else True)


def code_json(obj: Any) -> None:
    st.code(json.dumps(obj, indent=2, ensure_ascii=False, default=str), language="json")


# ── charts ────────────────────────────────────────────────────────


def pass_rate_chart(rows: list[dict], title: str = "") -> alt.Chart:
    """Horizontal bars of `pass_rate` per `metric`, with a threshold tick per bar.

    Rows: {metric, pass_rate, threshold, passed}.
    """
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(y=alt.Y("metric:N", title=None, sort=None))
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("pass_rate:Q", title="pass rate", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.condition(alt.datum.passed, alt.value(GOOD), alt.value(CRITICAL)),
        tooltip=[
            alt.Tooltip("metric:N"),
            alt.Tooltip("pass_rate:Q", format=".1%", title="pass rate"),
            alt.Tooltip("threshold:Q", format=".0%"),
        ],
    )
    ticks = base.mark_tick(color="#0b0b0b", thickness=2, size=22).encode(x="threshold:Q")
    labels = base.mark_text(align="left", dx=4, color="#52514e").encode(
        x="pass_rate:Q", text=alt.Text("pass_rate:Q", format=".0%")
    )
    return (bars + ticks + labels).properties(title=title, height=alt.Step(34))


def heatmap(df: pd.DataFrame, x: str, y: str, value: str, title: str = "", fmt: str = ".2f", label_angle: int = 0) -> alt.Chart:
    """Sequential (single-hue) heatmap with value labels; `value` in [0, 1]."""
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:N", title=None, sort=None, axis=alt.Axis(labelAngle=label_angle, labelLimit=260)),
        y=alt.Y(f"{y}:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
    )
    cells = base.mark_rect(stroke="#ffffff", strokeWidth=2).encode(
        color=alt.Color(
            f"{value}:Q", scale=alt.Scale(domain=[0, 1], range=SEQ_BLUE), legend=alt.Legend(title=value, format="%")
        ),
        tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{value}:Q", format=fmt)],
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text(f"{value}:Q", format=fmt),
        color=alt.condition(alt.datum[value] > 0.55, alt.value("#ffffff"), alt.value("#0b0b0b")),
    )
    return (cells + text).properties(title=title, height=alt.Step(30))


def grouped_bars(df: pd.DataFrame, category: str, series: str, value: str, title: str = "") -> alt.Chart:
    """Grouped bars (fixed categorical order: blue, orange, aqua)."""
    order = list(dict.fromkeys(df[series]))
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{category}:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset(f"{series}:N", sort=order),
            y=alt.Y(f"{value}:Q", title=value),
            color=alt.Color(f"{series}:N", scale=alt.Scale(domain=order, range=[BLUE, ORANGE, AQUA]), title=None),
            tooltip=[category, series, value],
        )
        .properties(title=title, height=220)
    )


def duration_chart(rows: list[dict]) -> alt.Chart:
    """Stage durations as horizontal bars colored by status (status colors)."""
    df = pd.DataFrame(rows)
    domain = ["ok", "recovered", "failed", "skipped", "awaiting_review"]
    palette = [GOOD, BLUE, CRITICAL, NEUTRAL, WARNING]
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=16)
        .encode(
            y=alt.Y("stage:N", sort=None, title=None),
            x=alt.X("seconds:Q", title="seconds"),
            color=alt.Color("status:N", scale=alt.Scale(domain=domain, range=palette), title="status"),
            tooltip=["stage", "status", alt.Tooltip("seconds:Q", format=".1f"), "attempts"],
        )
        .properties(height=alt.Step(24))
    )


# ── domain widgets ────────────────────────────────────────────────


def transcript(messages: list[dict], key: str = "") -> None:
    """Render a run trajectory / simulation transcript as chat messages."""
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        avatar = {"user": ":material/person:", "assistant": ":material/smart_toy:", "tool": ":material/build:", "system": ":material/settings:"}
        with st.chat_message("user" if role == "user" else "assistant", avatar=avatar.get(role)):
            if role == "tool":
                st.caption(f"tool result · {msg.get('name', '')}")
                st.code(str(msg.get("content", "")), language="json")
                continue
            content = msg.get("content")
            if content:
                st.markdown(str(content))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name")
                args = fn.get("arguments") if fn else tc.get("args")
                st.code(f"{name}({args})", language="text")


def tool_calls_inline(tool_calls: list[dict]) -> str:
    if not tool_calls:
        return "—"
    return " → ".join(f"`{tc.get('name')}({json.dumps(tc.get('args', {}), ensure_ascii=False)})`" for tc in tool_calls)


def evidence_md(evidence: list[str] | None) -> str:
    return " ".join(f"`{e}`" for e in evidence or []) or "—"


def case_input(case: dict) -> str:
    msgs = (case.get("inputs") or {}).get("messages") or []
    return str(msgs[0].get("content", "")) if msgs else json.dumps(case.get("inputs"), ensure_ascii=False)[:200]
