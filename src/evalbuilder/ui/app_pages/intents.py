"""Intents & scenarios — hypotheses with evidence, failure scenarios, applicable failure types."""

from __future__ import annotations

import streamlit as st

from evalbuilder.ui.common import evidence_md, get_bundle, require, table


def render() -> None:
    bundle = get_bundle()
    if not require(bundle, "agent_map"):
        return
    amap = bundle.agent_map
    intents = amap.get("intents") or []
    scenarios = amap.get("scenarios") or []
    failures = amap.get("failure_scenarios") or []
    applicable = bundle.get("applicable_failures") or {}
    cases = (bundle.dataset or {}).get("cases") or []

    st.header("Intents & scenarios", anchor="intents")
    with st.container(horizontal=True):
        st.metric("Intents", len(intents), border=True)
        st.metric("Scenarios", len(scenarios), border=True)
        st.metric("Happy / failure", f"{sum(1 for s in scenarios if s.get('kind') == 'happy')} / "
                  f"{sum(1 for s in scenarios if s.get('kind') != 'happy')}", border=True)
        st.metric("Failure types", len(failures), border=True)
        st.metric("Applicable types", len(applicable), border=True)

    cases_by_intent: dict[str, int] = {}
    cases_by_scenario: dict[str, int] = {}
    for c in cases:
        md = c.get("metadata") or {}
        cases_by_intent[md.get("intent", "unspecified")] = cases_by_intent.get(md.get("intent", "unspecified"), 0) + 1
        cases_by_scenario[md.get("scenario", "unspecified")] = cases_by_scenario.get(md.get("scenario", "unspecified"), 0) + 1

    with st.container(border=True):
        st.subheader("Intents", anchor="intents-table")
        table(
            [
                {"intent": i["id"], "description": i.get("description", ""), "status": i.get("status", ""),
                 "scenarios": sum(1 for s in scenarios if s.get("intent") == i["id"]),
                 "cases": cases_by_intent.get(i["id"], 0), "evidence": ", ".join(i.get("evidence") or [])}
                for i in intents
            ]
        )

    with st.container(border=True):
        st.subheader("Scenarios by intent", anchor="scenarios")
        filt = st.pills("Kind", ["happy", "failure"], selection_mode="multi", default=["happy", "failure"], key="scenario_kind")
        for intent in intents + [{"id": "cross-cutting", "description": "scenarios without an intent"}]:
            rows = [s for s in scenarios if (s.get("intent") or "cross-cutting") == intent["id"] and (s.get("kind") or "happy") in (filt or [])]
            if not rows:
                continue
            st.markdown(f"**{intent['id']}** — {intent.get('description', '')}")
            for s in rows:
                badge = ":green-badge[happy]" if s.get("kind") == "happy" else f":red-badge[{s.get('failure_mode') or 'failure'}]"
                with st.expander(f"{s['id']}  ·  cases: {cases_by_scenario.get(s['id'], 0)}"):
                    st.markdown(f"{badge} {s.get('description', '')}")
                    if s.get("expected_behavior"):
                        st.markdown(f"**Expected behavior:** {s['expected_behavior']}")
                    st.caption(f"evidence: {evidence_md(s.get('evidence'))} · status {s.get('status', '')}")

    with st.container(border=True):
        st.subheader("Failure scenarios", anchor="failure-scenarios")
        for f in failures:
            with st.expander(f"{f['failure_type']}"):
                st.markdown(f.get("rationale", ""))
                st.caption(f"evidence: {evidence_md(f.get('evidence'))}")
        if not failures:
            st.caption("none")

    with st.container(border=True):
        st.subheader("Structurally applicable failure types", anchor="applicable-failures")
        st.caption("Gated by the taxonomy from what the agent actually has (tools, branches, constraints, multi-turn).")
        table([{"failure type": k, "gated by": ", ".join(v)} for k, v in applicable.items()])
