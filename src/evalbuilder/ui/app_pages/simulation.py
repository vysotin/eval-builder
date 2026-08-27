"""Simulation — multi-turn scenarios, transcripts, violations, stop reasons."""

from __future__ import annotations

import streamlit as st

from evalbuilder.ui.common import get_bundle, require, table, transcript


def render() -> None:
    bundle = get_bundle()
    if not require(bundle, "simulation"):
        return
    results = (bundle.get("simulation") or {}).get("results") or []
    scenarios = {s.get("id"): s for s in ((bundle.get("scenarios") or {}).get("scenarios") or [])}
    details = ((bundle.report or {}).get("stages") or {}).get("simulate", {}).get("details") or {}

    st.header("Simulation", anchor="simulation")
    st.markdown("A simulated user (the generator model) plays each persona against the mocked agent; violations are mined back into the dataset as pending cases.")
    with st.container(horizontal=True):
        st.metric("Scenarios", len(results), border=True)
        st.metric("Succeeded", sum(1 for r in results if r.get("stop_reason") == "success"), border=True)
        st.metric("With violations", sum(1 for r in results if r.get("violations")), border=True)
        st.metric("Mined pending cases", details.get("mined_pending_cases", "—"), border=True)

    with st.container(border=True):
        st.subheader("Outcomes", anchor="simulation-outcomes")
        table([{"scenario": r.get("scenario_id"), "stop reason": r.get("stop_reason"), "turns": r.get("turns"),
                "violations": len(r.get("violations") or []), "goal": (scenarios.get(r.get("scenario_id")) or {}).get("goal", "")} for r in results])

    for r in results:
        scen = scenarios.get(r.get("scenario_id")) or {}
        ok = r.get("stop_reason") == "success" and not r.get("violations")
        with st.container(border=True):
            st.subheader(f"{r.get('scenario_id')}", anchor=f"sim-{r.get('scenario_id')}")
            st.markdown(
                (":green-badge[ok]" if ok else f":red-badge[{r.get('stop_reason')}]") + f" turns {r.get('turns')}"
                + (f" · persona: *{scen.get('persona')}*" if scen.get("persona") else "")
                + (f" · goal: {scen.get('goal')}" if scen.get("goal") else "")
            )
            for v in r.get("violations") or []:
                st.markdown(f"- :red-badge[violation] {v}")
            if scen.get("expect") or scen.get("success_contains"):
                st.caption(f"success_contains: `{scen.get('success_contains')}` · expect: `{scen.get('expect')}`")
            with st.expander("Transcript", expanded=not ok):
                transcript(r.get("transcript") or [], key=f"sim-{r.get('scenario_id')}")
