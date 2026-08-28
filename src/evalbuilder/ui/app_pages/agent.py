"""Agent — graph (AST + live), node prompts, tools with schemas, constraints."""

from __future__ import annotations

import json

import streamlit as st

from evalbuilder.ui.common import evidence_md, get_bundle, require, table

NODE_STYLE = {
    "llm": 'shape=box, style="rounded,filled", fillcolor="#cde2fb", color="#2a78d6"',
    "tool": 'shape=box, style="filled", fillcolor="#fde3d6", color="#eb6834"',
    "router": 'shape=diamond, style="filled", fillcolor="#d5f3e8", color="#1baf7a"',
    "function": 'shape=box, style="filled", fillcolor="#f0efec", color="#9a9892"',
}


def _q(s: str) -> str:
    return '"' + str(s).replace('"', '\\"') + '"'


def merged_nodes(graph: dict) -> list[dict]:
    """One entry per node id: discovery lists LLM nodes (kind `llm`, with prompt/tools) and
    graph nodes (kind `graph-node`) separately; the richer entry wins, fields are merged."""
    merged: dict[str, dict] = {}
    for n in graph.get("nodes", []):
        cur = merged.setdefault(n["id"], {"id": n["id"], "kind": "graph-node", "evidence": []})
        if n.get("kind") and n["kind"] != "graph-node":
            cur["kind"] = n["kind"]
        for key in ("prompt", "tools", "tools_source"):
            if n.get(key):
                cur[key] = n[key]
        for e in n.get("evidence") or []:
            if e not in cur["evidence"]:
                cur["evidence"].append(e)
    return list(merged.values())


def graph_dot(graph: dict, tools: list[dict], *, live: bool, show_tools: bool) -> str:
    """DOT for the agent graph. `live=True` draws the compiled graph's edges instead."""
    nodes = {n["id"]: n for n in merged_nodes(graph)}
    lines = [
        "digraph agent {",
        "  rankdir=LR; bgcolor=transparent; pad=0.2; nodesep=0.35; ranksep=0.6;",
        '  node [fontname="Helvetica", fontsize=11]; edge [fontname="Helvetica", fontsize=9, color="#52514e"];',
        '  START [shape=circle, label="", width=0.25, style=filled, fillcolor="#0b0b0b"];',
        '  END [shape=doublecircle, label="", width=0.2, style=filled, fillcolor="#0b0b0b"];',
    ]
    live_graph = graph.get("live") or {}
    if live and live_graph.get("nodes"):
        for n in live_graph["nodes"]:
            if n in ("__start__", "__end__"):
                continue
            kind = nodes.get(n, {}).get("kind", "function")
            lines.append(f"  {_q(n)} [{NODE_STYLE.get(kind, NODE_STYLE['function'])}];")
        for src, dst in live_graph.get("edges", []):
            s = "START" if src == "__start__" else _q(src)
            d = "END" if dst == "__end__" else _q(dst)
            lines.append(f"  {s} -> {d};")
    else:
        for n in nodes.values():
            kind = n.get("kind", "function")
            label = n["id"] if not n.get("tools") else f"{n['id']}\\n({len(n['tools'])} tools)"
            lines.append(f"  {_q(n['id'])} [label={_q(label)}, {NODE_STYLE.get(kind, NODE_STYLE['function'])}];")
        for edge in graph.get("edges", []):
            src, dst = edge[0], edge[1]
            lines.append(f"  {_q(src)} -> {_q(dst)};")
        for ce in graph.get("conditional_edges", []):
            for target in ce.get("targets", []):
                lines.append(f"  {_q(ce['source'])} -> {_q(target)} [style=dashed, label=\"?\"];")
    if show_tools:
        for t in tools:
            lines.append(f"  {_q('tool:' + t['name'])} [label={_q(t['name'])}, {NODE_STYLE['tool']}];")
            for user in t.get("used_by") or []:
                lines.append(f"  {_q(user)} -> {_q('tool:' + t['name'])} [style=dotted, arrowhead=open, color=\"#eb6834\"];")
    lines.append("}")
    return "\n".join(lines)


def render() -> None:
    bundle = get_bundle()
    if not require(bundle, "agent_map"):
        return
    amap = bundle.agent_map
    app = amap.get("app") or {}
    graph = amap.get("graph") or {}
    tools = amap.get("tools") or []

    st.header("Agent", anchor="agent")
    st.markdown(
        f"`{app.get('module', '?')}` · factory `{app.get('factory', 'build_agent')}` · source `{app.get('source', '?')}` · "
        f"framework **{amap.get('framework', '?')}** · sha256 `{(amap.get('source_sha256') or '')[:12]}`"
    )
    with st.container(horizontal=True):
        st.metric("Nodes", len(merged_nodes(graph)), border=True)
        st.metric("Edges", len(graph.get("edges", [])) + sum(len(c.get("targets", [])) for c in graph.get("conditional_edges", [])), border=True)
        st.metric("Tools", len(tools), border=True)
        st.metric("Constraints", len(amap.get("constraints", [])), border=True)
        st.metric("Decisions needed", len(amap.get("decisions_needed", [])), border=True)

    with st.container(border=True):
        st.subheader("Graph", anchor="agent-graph")
        c1, c2 = st.columns([2, 1])
        with c1:
            source = st.segmented_control(
                "Edges from", ["source (AST)", "live (compiled)"], default="source (AST)", key="agent_graph_source",
                disabled=not (graph.get("live") or {}).get("edges"),
            )
        with c2:
            show_tools = st.toggle("Show tool links", value=True, key="agent_graph_tools")
        st.graphviz_chart(graph_dot(graph, tools, live=source == "live (compiled)", show_tools=show_tools), width="stretch")
        st.caption("LLM nodes blue, tools orange, conditional edges dashed, prompt→tool links dotted.")

    with st.container(border=True):
        st.subheader("Nodes and prompts", anchor="agent-nodes")
        for n in merged_nodes(graph):
            with st.expander(f"{n['id']} · {n.get('kind', '?')}" + (f" · tools: {', '.join(n['tools'])}" if n.get("tools") else "")):
                if n.get("prompt"):
                    st.markdown(f"> {n['prompt']}")
                st.caption(f"evidence: {evidence_md(n.get('evidence'))}" + (f" · tools from {n.get('tools_source')}" if n.get("tools_source") else ""))
        if graph.get("conditional_edges"):
            st.markdown("**Conditional edges**")
            table(
                [{"source": c["source"], "targets": ", ".join(c.get("targets", [])), "evidence": ", ".join(c.get("evidence", []))}
                 for c in graph["conditional_edges"]]
            )

    with st.container(border=True):
        st.subheader("Tools", anchor="agent-tools")
        table(
            [
                {"tool": t["name"], "description": t.get("description", ""), "used by": ", ".join(t.get("used_by") or []),
                 "live": bool(t.get("live")), "args": ", ".join((t.get("args_schema") or {}).get("properties", {}).keys()),
                 "schema source": t.get("schema_source") or "—", "models": ", ".join(t.get("models") or []),
                 "output schema": bool(t.get("output_schema")), "side-effecting": bool(t.get("side_effecting")),
                 "edge cases": len(t.get("edge_cases") or [])}
                for t in tools
            ],
            column_config={
                "live": st.column_config.CheckboxColumn(help="introspected from the imported module"),
                "output schema": st.column_config.CheckboxColumn(help="return annotation captured as JSON schema"),
                "side-effecting": st.column_config.CheckboxColumn(help="docstring declares a side effect / confirmation gate"),
            },
        )
        for t in tools:
            with st.expander(f"{t['name']} — argument schema"):
                st.code(json.dumps(t.get("args_schema") or {}, indent=2), language="json")
                if t.get("output_schema"):
                    st.markdown("**Output schema**" + (f" · models: {', '.join(f'`{m}`' for m in t.get('models') or [])}" if t.get("models") else ""))
                    st.code(json.dumps(t["output_schema"], indent=2), language="json")
                if t.get("edge_cases"):
                    st.markdown("**Schema edge cases** (deterministic, from the schemas above)")
                    table([
                        {"kind": e.get("kind"), "field": e.get("field") or "—", "failure mode": e.get("failure_mode"),
                         "detail": e.get("detail"), "expected behavior": e.get("expected_behavior")}
                        for e in t["edge_cases"]
                    ])

    with st.container(border=True):
        st.subheader("Constraints", anchor="agent-constraints")
        for c in amap.get("constraints", []):
            st.markdown(f"- {c}")
        if not amap.get("constraints"):
            st.caption("none")
        if amap.get("decisions_needed"):
            st.markdown("**Decisions needed**")
            for d in amap["decisions_needed"]:
                st.markdown(f"- :orange-badge[open] {d}")
    topics = (amap.get("data_domains") or {}).get("topics") or []
    if topics:
        st.markdown("**Data-domain topics:** " + " ".join(f":blue-badge[{t}]" for t in topics))
