"""Stages & problems — per-stage status/timing/details, generator calls, artifacts, config."""

from __future__ import annotations

import streamlit as st

from evalbuilder.pipeline.layout import ARTIFACTS
from evalbuilder.ui.common import code_json, duration_chart, get_bundle, require, status_md, table


def render() -> None:
    bundle = get_bundle()
    if not require(bundle):
        return
    report = bundle.report or {}
    state = bundle.state or {}
    stages = report.get("stages") or state.get("stages") or {}
    if not stages:
        require(bundle, "pipeline_state")
        return

    st.header("Stages & problems", anchor="stages")
    st.markdown(
        f"started {state.get('started_at', '—')[:19].replace('T', ' ')} · updated {state.get('updated_at', '—')[:19].replace('T', ' ')} · "
        f"config `{report.get('config_path', '—')}` · output `{report.get('output_dir') or bundle.source}`"
    )
    total = sum((rec.get("seconds") or 0) for rec in stages.values())
    with st.container(horizontal=True):
        st.metric("Stages", len(stages), border=True)
        st.metric("Succeeded", sum(1 for r in stages.values() if r.get("status") in ("ok", "recovered")), border=True)
        st.metric("Failed", sum(1 for r in stages.values() if r.get("status") == "failed"), border=True)
        st.metric("Skipped", sum(1 for r in stages.values() if r.get("status") == "skipped"), border=True)
        st.metric("Total time", f"{total / 60:.1f} min", border=True)

    with st.container(border=True):
        st.subheader("Timeline", anchor="stage-timeline")
        rows = [{"stage": n, "status": r.get("status"), "seconds": r.get("seconds") or 0.0, "attempts": r.get("attempts", 0)} for n, r in stages.items()]
        st.altair_chart(duration_chart(rows), width="stretch")
        table([{"stage": n, "status": r.get("status"), "attempts": r.get("attempts"), "seconds": r.get("seconds"),
                "optional": bool(r.get("optional")), "reason / error": r.get("reason") or r.get("error") or "",
                "recovery notes": "; ".join(r.get("recovery_notes") or [])} for n, r in stages.items()],
              column_config={"optional": st.column_config.CheckboxColumn()})

    with st.container(border=True):
        st.subheader("Stage details", anchor="stage-details")
        for name, rec in stages.items():
            with st.expander(f"{name} — {rec.get('status')}"):
                st.markdown(status_md(rec.get("status")) + (f" {rec.get('reason') or rec.get('error') or ''}"))
                if rec.get("artifacts"):
                    st.markdown("**Artifacts**")
                    code_json(rec["artifacts"])
                if rec.get("details"):
                    st.markdown("**Details**")
                    code_json(rec["details"])

    problems = report.get("problems") or (state.get("data") or {}).get("problems") or []
    with st.container(border=True):
        st.subheader(f"Problems ({len(problems)})", anchor="problems")
        if problems:
            table(problems, ["severity", "stage", "message"])
        else:
            st.success("No problems recorded.", icon=":material/check_circle:")

    calls = report.get("generator_calls") or []
    if calls:
        with st.container(border=True):
            st.subheader("Generator calls (this process)", anchor="generator-calls")
            table(calls, ["schema", "seconds"])

    with st.container(border=True):
        st.subheader("Artifacts loaded", anchor="artifacts-loaded")
        rows = []
        for kind, a in ARTIFACTS.items():
            loc = bundle.files.get(kind)
            rows.append({"kind": kind, "file": a.file, "schema": a.schema or "—", "stage": a.stage,
                         "loaded": bool(loc), "from": (", ".join(loc.values()) if isinstance(loc, dict) else loc) or ""})
        table(rows, column_config={"loaded": st.column_config.CheckboxColumn()})

    if bundle.config:
        with st.expander("Pipeline config"):
            code_json(bundle.config)
    if bundle.get("evaluators"):
        with st.expander("Evaluators"):
            code_json(bundle.get("evaluators"))
