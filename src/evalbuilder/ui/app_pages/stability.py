"""Stability — unstable cases / outputs / evaluators across repeats, suspect judge comments."""

from __future__ import annotations

import streamlit as st

from evalbuilder.ui.common import get_bundle, pct, require, table


def render() -> None:
    bundle = get_bundle()
    if not require(bundle, "aggregate"):
        return
    st_ = (bundle.aggregate or {}).get("stability") or {}

    st.header("Stability", anchor="stability")
    st.markdown(
        "Repeated runs separate **agent behavior** from **evaluator noise**: an unstable *case* changed its tool "
        "trajectory between repeats; an unstable *output* kept the trajectory but a deterministic metric flipped "
        "(wording drift); an unstable *evaluator* kept the trajectory but an LLM judge flipped (judge variance)."
    )
    with st.container(horizontal=True):
        st.metric("Repeats", st_.get("repeats", "—"), border=True)
        st.metric("Stable cases", pct(st_.get("stable_case_fraction")), border=True)
        st.metric("Unstable cases", len(st_.get("unstable_cases") or []), border=True)
        st.metric("Unstable outputs", len(st_.get("unstable_outputs") or []), border=True)
        st.metric("Unstable evaluators", len(st_.get("unstable_evaluators") or []), border=True)
        st.metric("Wording varies", st_.get("text_varies", 0), border=True, help="cases whose response text differed (not counted as instability)")

    with st.container(border=True):
        st.subheader("Unstable cases (trajectory differs)", anchor="unstable-cases")
        table([{"case": u.get("id"), "distinct trajectories": u.get("distinct_trajectories"),
                "metrics disagreeing": ", ".join(u.get("metrics_disagreeing") or [])} for u in st_.get("unstable_cases") or []])

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Unstable outputs", anchor="unstable-outputs")
            table([{"case": u.get("case_id"), "metric": u.get("metric"), "scores": str(u.get("scores"))} for u in st_.get("unstable_outputs") or []])
    with c2:
        with st.container(border=True):
            st.subheader("Unstable evaluators", anchor="unstable-evaluators")
            table([{"case": u.get("case_id"), "metric": u.get("metric"), "scores": str(u.get("scores"))} for u in st_.get("unstable_evaluators") or []])

    with st.container(border=True):
        st.subheader("Suspect judge comments", anchor="suspect-judge-comments")
        st.caption("Judge rationales that look like placeholders (\"Test.\", \"placeholder\"…) — treat those scores as unverified.")
        table([{"case": s.get("case_id"), "run": s.get("run_id"), "metric": s.get("metric"), "comment": s.get("comment")}
               for s in st_.get("suspect_judge_comments") or []])

    cases = (bundle.aggregate or {}).get("cases") or []
    if cases:
        with st.container(border=True):
            st.subheader("Per-case hashes", anchor="case-hashes")
            table([{"case": c.get("id"), "stable": bool(c.get("stable")), "trajectory hashes": ", ".join(c.get("trajectory_hash") or []),
                    "output hashes": ", ".join(c.get("outputs_hash") or [])} for c in cases],
                  column_config={"stable": st.column_config.CheckboxColumn()})
