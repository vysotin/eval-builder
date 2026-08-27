"""Coverage — planned cells vs achieved, by kind, gaps."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evalbuilder.ui.common import get_bundle, grouped_bars, require, table


def render() -> None:
    bundle = get_bundle()
    if not require(bundle, "coverage"):
        return
    coverage = bundle.get("coverage") or {}
    plan = bundle.get("coverage_plan") or {}
    cells = plan.get("cells") or []
    summary = plan.get("summary") or {}

    st.header("Coverage", anchor="coverage")
    with st.container(horizontal=True):
        st.metric("Planned cells", coverage.get("planned", summary.get("cells", "—")), border=True)
        st.metric("Covered", coverage.get("covered", "—"), border=True)
        st.metric("Coverage", f"{coverage.get('coverage_pct', '—')}%", border=True)
        mt = coverage.get("multi_turn") or {}
        st.metric("Multi-turn", f"{mt.get('have', '—')} / {mt.get('planned', '—')}", border=True)
        st.metric("Gaps", len(coverage.get("gaps") or []), border=True)

    by_kind = coverage.get("by_kind") or {}
    if by_kind:
        with st.container(border=True):
            st.subheader("Planned vs covered by kind", anchor="coverage-by-kind")
            df = pd.DataFrame(
                [{"kind": k, "series": s, "cases": v.get(s, 0)} for k, v in by_kind.items() for s in ("planned", "covered")]
            )
            st.altair_chart(grouped_bars(df, "kind", "series", "cases"), width="stretch")

    if cells:
        with st.container(border=True):
            st.subheader("Plan cells", anchor="coverage-plan")
            st.caption("Each cell is intent × topic × scenario × failure mode; `count` cases were requested from the generator.")
            table(
                [{"intent": c.get("intent"), "topic": c.get("topic"), "scenario": c.get("scenario"),
                  "failure_mode": c.get("failure_mode"), "kind": c.get("kind"), "count": c.get("count"),
                  "variants": ", ".join(c.get("variants") or []), "multi_turn": c.get("multi_turn"),
                  "related_intent": c.get("related_intent")} for c in cells]
            )

    with st.container(border=True):
        st.subheader("Gaps", anchor="coverage-gaps")
        gaps = coverage.get("gaps") or []
        if gaps:
            table(
                [{"intent": g.get("intent"), "topic": g.get("topic"), "scenario": g.get("scenario"),
                  "failure_mode": g.get("failure_mode"), "kind": g.get("kind"), "wanted": g.get("count"),
                  "have": g.get("have"), "missing": g.get("missing")} for g in gaps]
            )
        else:
            st.success("Every planned cell is covered.", icon=":material/check_circle:")
