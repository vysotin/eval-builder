"""Pipeline setup — pick a target, preview its structure, build/edit the config, launch a run."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from evalbuilder.pipeline import jobs
from evalbuilder.pipeline import setup as setup_mod
from evalbuilder.pipeline.config import DEFAULT_MODEL
from evalbuilder.ui.common import code_json, table

CUSTOM = "custom target…"


def _targets() -> list[dict]:
    if "setup_targets" not in st.session_state:
        st.session_state["setup_targets"] = setup_mod.discover_targets()
    return st.session_state["setup_targets"]


def _apply_target() -> None:
    """Selecting an example fills source/module/name/output/config path."""
    choice = st.session_state.get("setup_target")
    target = next((t for t in _targets() if t["source"] == choice), None)
    if target:
        form = setup_mod.default_form(target)
        for key in ("source", "module", "name", "output_dir", "config_path"):
            st.session_state[f"setup_{key}"] = form[key]
        st.session_state["setup_preview"] = None


def _init_form() -> None:
    defaults = setup_mod.default_form(_targets()[0] if _targets() else None)
    for key, value in defaults.items():
        st.session_state.setdefault(f"setup_{key}", value)
    st.session_state.setdefault("setup_target", defaults["source"] or CUSTOM)
    st.session_state.setdefault("setup_preview", None)
    st.session_state.setdefault("setup_yaml", "")
    st.session_state.setdefault("setup_notice", None)


def _form_values() -> dict:
    return {key: st.session_state.get(f"setup_{key}") for key in setup_mod.default_form()}


def _render_preview(preview: dict) -> None:
    for p in preview.get("problems") or []:
        st.warning(p, icon=":material/warning:")
    if not preview.get("ok"):
        return
    with st.container(horizontal=True):
        st.metric("Graph nodes", len(preview["nodes"]), border=True)
        st.metric("Tools", len(preview["tools"]), border=True)
        st.metric("Pydantic models", len(preview["models"]), border=True)
        st.metric("Schema edge cases", preview.get("edge_case_count", 0), border=True)
    st.caption(
        "nodes: " + ", ".join(f"`{n['id']}`" for n in preview["nodes"]) + (
            " · live: " + ", ".join(f"`{n}`" for n in preview["live"].get("nodes", [])) if preview.get("live") and "nodes" in preview["live"] else ""
        )
    )
    st.markdown("**Tools and schemas**")
    table([
        {
            "tool": t["name"], "schema source": t.get("schema_source"), "models": ", ".join(t.get("models") or []),
            "output schema": "yes" if t.get("output_schema") else "—", "side-effecting": bool(t.get("side_effecting")),
            "edge cases": len(t.get("edge_cases") or []), "used by": ", ".join(t.get("used_by") or []),
        }
        for t in preview["tools"]
    ], column_config={"side-effecting": st.column_config.CheckboxColumn()})
    for t in preview["tools"]:
        with st.expander(f"{t['name']} — schemas & edge cases"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("`args_schema`")
                code_json(t.get("args_schema") or {})
            with c2:
                st.markdown("`output_schema`")
                code_json(t.get("output_schema") or {})
            edges = t.get("edge_cases") or []
            st.markdown("**Edge cases derived from the schemas**" if edges else "_no schema edge cases (no constraints, enums or required fields)_")
            for e in edges:
                field = f" `{e['field']}`" if e.get("field") else ""
                st.markdown(f"- **{e['kind']}**{field} → `{e['failure_mode']}`: {e['expected_behavior']}")


def render() -> None:
    _init_form()
    st.header("Pipeline setup", anchor="setup")
    st.caption(
        "Pick a target, discover its structure (no LLM), edit the config — constraints and free-text "
        "instructions reach every generation prompt — then run everything or only the dataset + mocks."
    )
    if st.session_state.get("setup_notice"):
        kind, text = st.session_state["setup_notice"]
        getattr(st, kind)(text)

    # ── target ────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Target agent", anchor="target")
        options = [t["source"] for t in _targets()] + [CUSTOM]
        st.selectbox(
            "Example agent", options, key="setup_target", on_change=_apply_target,
            format_func=lambda s: s if s == CUSTOM else f"{next(t['name'] for t in _targets() if t['source'] == s)} ({s})",
        )
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.text_input("Agent source file", key="setup_source", placeholder="path/to/agent.py")
        with c2:
            st.text_input("Importable module", key="setup_module", placeholder="pkg.agent")
        with c3:
            st.text_input("Factory", key="setup_factory")
        if st.button("Discover structure", key="setup_discover", icon=":material/search:", type="secondary"):
            with st.spinner("Introspecting the agent…"):
                st.session_state["setup_preview"] = setup_mod.preview_target(
                    st.session_state["setup_source"], st.session_state["setup_module"], st.session_state["setup_factory"]
                )
        if st.session_state.get("setup_preview"):
            _render_preview(st.session_state["setup_preview"])

    # ── config form ───────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Config", anchor="config")
        with st.form("setup_form", border=False):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.text_input("Pipeline name", key="setup_name")
            with c2:
                st.text_input("Config file path", key="setup_config_path")
            st.markdown("**Models** — `provider:model[@effort]`; default is Claude Sonnet 5 through the Claude Code CLI")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.text_input("Agent model (empty = target default)", key="setup_agent_model", help=f"e.g. {DEFAULT_MODEL} or scripted:module:factory")
            with m2:
                st.text_input("Judge model", key="setup_judge_model")
            with m3:
                st.text_input("Generator model", key="setup_generator_model")
            st.text_area("Constraints (one per line)", key="setup_constraints", height=100,
                         placeholder="Never call issue_refund before the customer explicitly confirms.")
            st.text_area("General rules & instructions for the generator (free text)", key="setup_instructions", height=120,
                         placeholder="Domain notes, what to emphasise, what to avoid, tone of the users…")
            st.markdown("**Coverage**")
            k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
            with k1:
                st.number_input("Total cases", 1, 500, key="setup_total_cases")
            with k2:
                st.number_input("Happy / intent", 0, 20, key="setup_happy")
            with k3:
                st.number_input("Failure / intent", 0, 20, key="setup_failure")
            with k4:
                st.number_input("Per failure type", 0, 20, key="setup_per_failure_category")
            with k5:
                st.number_input("Out of intent", 0, 20, key="setup_out_of_intent")
            with k6:
                st.number_input("Multi-turn share", 0.0, 1.0, step=0.05, key="setup_multi_turn_share")
            with k7:
                st.number_input("Edge cases / tool", 0, 10, key="setup_per_tool_edge_cases",
                                help="schema-derived: missing / wrong / out-of-range input, malformed tool output")
            e1, e2 = st.columns([2, 1])
            with e1:
                st.multiselect("Evaluators", list(setup_mod.EVALUATOR_CHOICES), key="setup_evaluators")
            with e2:
                st.selectbox("Mock miss policy", ["strict", "fallback", "real"], key="setup_on_miss")
            t1, t2, t3, t4, t5 = st.columns(5)
            with t1:
                st.number_input("Metric threshold", 0.0, 1.0, step=0.05, key="setup_threshold_default")
            with t2:
                st.number_input("Slice minimum", 0.0, 1.0, step=0.05, key="setup_slice_min")
            with t3:
                st.number_input("Overall pass", 0.0, 1.0, step=0.05, key="setup_overall_pass")
            with t4:
                st.number_input("Repeats", 1, 10, key="setup_repeats")
            with t5:
                st.checkbox("Simulate multi-turn", key="setup_simulate")
            r1, r2, r3 = st.columns([1, 1, 2])
            with r1:
                st.checkbox("Auto-approve generated cases", key="setup_auto_approve",
                            help="Your explicit authorization for the pipeline to approve cases (needs a name).")
            with r2:
                st.text_input("Approved by", key="setup_approved_by")
            with r3:
                st.text_input("Output directory", key="setup_output_dir")
            generated = st.form_submit_button("Generate YAML", icon=":material/description:", type="primary")
        if generated:
            try:
                cfg = setup_mod.build_config(_form_values())
                st.session_state["setup_yaml"] = cfg.to_yaml()
                st.session_state["setup_notice"] = ("success", f"Config generated for **{cfg.name}** — review the YAML below, then save or run.")
            except Exception as e:  # noqa: BLE001 - show the validation error inline
                st.session_state["setup_notice"] = ("error", f"Config invalid: {e}")
            st.rerun()

    # ── yaml editor ───────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Config YAML", anchor="yaml")
        st.text_area("pipeline config (editable)", key="setup_yaml", height=420, label_visibility="collapsed")
        b1, b2, b3, b4 = st.columns([1, 1, 1.5, 2])
        with b1:
            validate = st.button("Validate", key="setup_validate", icon=":material/rule:")
        with b2:
            save = st.button("Save config", key="setup_save", icon=":material/save:")
        with b3:
            run_dataset = st.button("Generate dataset & mocks only", key="setup_run_dataset", icon=":material/dataset:")
        with b4:
            run_full = st.button("Run full pipeline", key="setup_run_full", icon=":material/play_arrow:", type="primary")

        if validate or save or run_dataset or run_full:
            cfg, problems = setup_mod.validate_text(st.session_state["setup_yaml"])
            if cfg is None or problems:
                st.error("Config problems:\n\n" + "\n".join(f"- {p}" for p in problems), icon=":material/error:")
                return
            st.success(f"Config **{cfg.name}** is valid — target `{cfg.target.module}`, output `{cfg.output_dir}`.", icon=":material/check_circle:")
            if save or run_dataset or run_full:
                path = Path(st.session_state["setup_config_path"] or f"eval/pipeline/{cfg.name}.yaml")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(st.session_state["setup_yaml"])
                st.session_state["setup_saved_path"] = str(path)
                st.info(f"Saved to `{path}`.", icon=":material/save:")
            if run_dataset or run_full:
                try:
                    job = jobs.start_job(path, cfg.output_dir, mode="dataset" if run_dataset else "full")
                except RuntimeError as e:
                    st.error(str(e))
                    return
                st.session_state["job_dir"] = str(cfg.output_dir)
                st.session_state["run_dir_pending"] = str(cfg.output_dir)
                st.session_state["setup_notice"] = ("info", f"Started `{job['mode']}` job (pid {job['pid']}) for `{cfg.name}` → `{cfg.output_dir}`.")
                pages = st.session_state.get("pages") or {}
                if "run" in pages:
                    st.switch_page(pages["run"])
                st.json(job)
