"""Analysis — generator (or deterministic) explanation of the verdict."""

from __future__ import annotations

import streamlit as st

from evalbuilder.ui.common import get_bundle, require

SEVERITY = {"high": "red", "medium": "orange", "low": "blue"}


def render() -> None:
    bundle = get_bundle()
    if not require(bundle, "analysis"):
        return
    a = bundle.get("analysis") or {}

    st.header("Analysis", anchor="analysis")
    st.markdown(f"source: :{'blue' if a.get('source') == 'generator' else 'grey'}-badge[{a.get('source', '?')}]")
    with st.container(border=True):
        st.subheader("Summary", anchor="analysis-text")
        st.markdown(a.get("summary", "") or "_no summary_")
        if a.get("verdict_explanation"):
            st.markdown("**Verdict explanation**")
            st.markdown(a["verdict_explanation"])

    patterns = a.get("failure_patterns") or []
    with st.container(border=True):
        st.subheader(f"Failure patterns ({len(patterns)})", anchor="failure-patterns")
        for p in patterns:
            sev = (p.get("severity") or "medium").lower()
            with st.expander(f"{p.get('pattern', '')[:120]}"):
                st.markdown(f":{SEVERITY.get(sev, 'orange')}-badge[{sev}] {p.get('pattern', '')}")
                if p.get("evidence"):
                    st.markdown(f"**Evidence:** {p['evidence']}")
                if p.get("affected_cases"):
                    st.markdown("**Affected cases:** " + " ".join(f"`{c}`" for c in p["affected_cases"]))
        if not patterns:
            st.caption("none")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Weak slices", anchor="analysis-weak-slices")
            for w in a.get("weak_slices") or []:
                st.markdown(f"- {w}")
            if not a.get("weak_slices"):
                st.caption("none")
    with c2:
        with st.container(border=True):
            st.subheader("Evaluator issues", anchor="evaluator-issues")
            for e in a.get("evaluator_issues") or []:
                st.markdown(f"- {e}")
            if not a.get("evaluator_issues"):
                st.caption("none")

    with st.container(border=True):
        st.subheader("Recommendations", anchor="recommendations")
        for i, r in enumerate(a.get("recommendations") or [], 1):
            st.markdown(f"{i}. {r}")
        if not a.get("recommendations"):
            st.caption("none")
    if a.get("stability_notes"):
        st.caption(f"stability notes: {a['stability_notes']}")
