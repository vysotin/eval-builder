"""Dataset & mocks — case table with filters, case detail, review state, tool fixtures."""

from __future__ import annotations

import json

import streamlit as st

from evalbuilder.ui.common import case_input, code_json, evidence_md, get_bundle, require, table


def _case_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        md = c.get("metadata") or {}
        ref = c.get("reference_outputs") or {}
        rows.append(
            {
                "id": c.get("id"),
                "status": (c.get("review") or {}).get("status"),
                "intent": md.get("intent"),
                "scenario": md.get("scenario"),
                "failure_mode": md.get("failure_mode"),
                "variant": md.get("variant"),
                "topic": md.get("topic"),
                "source": md.get("source"),
                "turns": 1 + len(md.get("user_turns") or []),
                "expected tools": ", ".join(t.get("name", "?") for t in ref.get("expected_tools") or []),
                "input": case_input(c),
            }
        )
    return rows


def render() -> None:
    bundle = get_bundle()
    if not require(bundle, "dataset"):
        return
    ds = bundle.dataset
    cases = ds.get("cases") or []
    statuses = {"pending": 0, "approved": 0, "rejected": 0}
    for c in cases:
        statuses[(c.get("review") or {}).get("status", "pending")] = statuses.get((c.get("review") or {}).get("status", "pending"), 0) + 1

    st.header("Dataset & mocks", anchor="dataset")
    st.markdown(
        f"**{ds.get('name')}** · schema `{ds.get('schema')}` · type `{ds.get('dataset_type')}` · "
        f"target `{(ds.get('target') or {}).get('module')}` · mock miss policy `{(ds.get('mocks') or {}).get('on_miss', 'real')}`"
    )
    with st.container(horizontal=True):
        st.metric("Cases", len(cases), border=True)
        st.metric("Approved", statuses.get("approved", 0), border=True)
        st.metric("Pending", statuses.get("pending", 0), border=True)
        st.metric("Rejected", statuses.get("rejected", 0), border=True)
        st.metric("Mocked tools", len((ds.get("mocks") or {}).get("tools") or {}), border=True)
        st.metric("Multi-turn", sum(1 for c in cases if (c.get("metadata") or {}).get("user_turns")), border=True)

    rows = _case_rows(cases)
    with st.container(border=True):
        st.subheader("Cases", anchor="cases")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            status_f = st.multiselect("Review status", sorted({r["status"] for r in rows if r["status"]}), key="ds_status")
        with f2:
            intent_f = st.multiselect("Intent", sorted({r["intent"] for r in rows if r["intent"]}), key="ds_intent")
        with f3:
            fm_f = st.multiselect("Failure mode", sorted({r["failure_mode"] for r in rows if r["failure_mode"]}), key="ds_failure")
        with f4:
            variant_f = st.multiselect("Variant", sorted({r["variant"] for r in rows if r["variant"]}), key="ds_variant")
        shown = [
            r for r in rows
            if (not status_f or r["status"] in status_f)
            and (not intent_f or r["intent"] in intent_f)
            and (not fm_f or r["failure_mode"] in fm_f)
            and (not variant_f or r["variant"] in variant_f)
        ]
        st.caption(f"{len(shown)} of {len(rows)} cases")
        table(shown, column_config={"id": st.column_config.TextColumn(pinned=True)})

    with st.container(border=True):
        st.subheader("Case detail", anchor="case-detail")
        ids = [r["id"] for r in shown] or [r["id"] for r in rows]
        if not ids:
            st.caption("no cases")
        else:
            selected = st.selectbox("Case", ids, key="ds_case", format_func=lambda i: f"{i} — {next((r['input'] for r in rows if r['id'] == i), '')[:70]}")
            case = next(c for c in cases if c.get("id") == selected)
            md = case.get("metadata") or {}
            review = case.get("review") or {}
            st.markdown(
                f":{ {'approved': 'green', 'rejected': 'red'}.get(review.get('status'), 'orange') }-badge[{review.get('status')}] "
                f"intent `{md.get('intent')}` · scenario `{md.get('scenario')}` · failure mode `{md.get('failure_mode')}` · "
                f"variant `{md.get('variant')}` · topic `{md.get('topic')}` · source `{md.get('source')}`"
            )
            if review.get("note"):
                st.caption(f"review note: {review['note']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Inputs**")
                for m in (case.get("inputs") or {}).get("messages") or []:
                    with st.chat_message("user" if m.get("role") == "user" else "assistant"):
                        st.markdown(str(m.get("content", "")))
                for turn in md.get("user_turns") or []:
                    with st.chat_message("user"):
                        st.markdown(f"*(follow-up)* {turn}")
            with c2:
                st.markdown("**Reference outputs**")
                ref = case.get("reference_outputs") or {}
                for key, value in ref.items():
                    if isinstance(value, (dict, list)):
                        st.markdown(f"`{key}`")
                        code_json(value)
                    else:
                        st.markdown(f"`{key}`: {value}")
            if md.get("evidence"):
                st.caption(f"evidence: {evidence_md(md.get('evidence'))}")
            case_mocks = (md.get("mocks") or {}).get("tools") or {}
            if case_mocks:
                st.markdown("**Per-case mock overrides**")
                for tool, rules in case_mocks.items():
                    with st.expander(f"{tool} — {len(rules)} rule(s)"):
                        code_json(rules)
            if case.get("publication", {}).get("langsmith_example_id"):
                st.caption(f"LangSmith example: `{case['publication']['langsmith_example_id']}`")

    with st.container(border=True):
        st.subheader("Tool mock fixtures", anchor="mock-rules")
        rules = bundle.get("mock_rules") or (ds.get("mocks") or {}).get("tools") or {}
        st.caption("Ordered per-tool rules, first match wins; `matchArgs` is a subset match, `{}` is a wildcard.")
        table([{"tool": t, "rules": len(r), "wildcard": any(not (x.get("matchArgs") or {}) for x in r)} for t, r in rules.items()],
              column_config={"wildcard": st.column_config.CheckboxColumn()})
        for tool, tool_rules in rules.items():
            with st.expander(f"{tool} — {len(tool_rules)} rule(s)"):
                table(
                    [{"matchArgs": json.dumps(r.get("matchArgs") or {}, ensure_ascii=False),
                      "response": json.dumps(r.get("response"), ensure_ascii=False)[:300]} for r in tool_rules]
                )
