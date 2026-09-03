"""Pipeline setup — choose the project (an existing output folder or a new one from a
target agent), preview the target's structure, build/edit the config, launch a run.

Everything on this page persists across navigation (see `ui.project`): widgets are
keyed `setup_<field>`, seeded from the persistent form before they are created and
captured back at the start of every run. Buttons act through callbacks so the form
is updated before any widget of the next run exists.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from evalbuilder.pipeline import jobs
from evalbuilder.pipeline import setup as setup_mod
from evalbuilder.pipeline.config import DEFAULT_MODEL
from evalbuilder.ui import loader, project
from evalbuilder.ui.common import code_json, table

CUSTOM = project.CUSTOM
NEW_PROJECT = "new project…"


# ── callbacks (run before the widgets of the next run are created) ──


def _apply_target() -> None:
    """Selecting an example fills source/module/name/output/config path; CUSTOM clears them."""
    choice = st.session_state.get("setup_target")
    target = next((t for t in project.targets() if t["source"] == choice), None)
    if target:
        values = setup_mod.default_form(target)
        project.set_form({key: values[key] for key in ("source", "module", "name", "output_dir", "config_path")})
    elif choice == CUSTOM:
        project.set_form({"source": "", "module": "", "name": "my-agent", "output_dir": "", "config_path": ""})
    project.form()["target"] = choice
    st.session_state.pop("setup_preview", None)
    st.session_state.pop("setup_notice", None)


def _apply_project() -> None:
    choice = st.session_state.get("setup_project")
    if choice == NEW_PROJECT:
        project.clear()
    elif choice:
        project.open_dir(choice)


def _open_path() -> None:
    path = (st.session_state.get("setup_project_path") or "").strip()
    if path:
        project.open_dir(path)
        st.session_state["setup_project_path"] = ""
    else:
        st.session_state["setup_result"] = [("error", "Type a folder path first.")]


def _discover() -> None:
    project.capture_widgets()
    f = project.form()
    with st.spinner("Introspecting the agent…"):
        st.session_state["setup_preview"] = setup_mod.preview_target(f["source"], f["module"], f["factory"])


def _generate_yaml() -> None:
    project.capture_widgets()
    try:
        cfg = setup_mod.build_config(project.form_values())
    except Exception as e:  # noqa: BLE001 - show the validation error inline
        st.session_state["setup_notice"] = ("error", f"Config invalid: {e}")
        return
    project.set_form({"yaml": cfg.to_yaml(), "output_dir": str(cfg.output_dir), "name": cfg.name})
    st.session_state["setup_notice"] = ("success", f"Config generated for **{cfg.name}** — review the YAML below, then save or run.")


def _yaml_action(action: str) -> None:
    """validate | save | dataset | full — on the YAML editor text."""
    project.capture_widgets()
    text = st.session_state.get("setup_yaml") or project.form().get("yaml") or ""
    project.form()["yaml"] = text
    cfg, problems = setup_mod.validate_text(text)
    if cfg is None or problems:
        st.session_state["setup_result"] = [("error", "Config problems:\n\n" + "\n".join(f"- {p}" for p in problems))]
        return
    result = [("success", f"Config **{cfg.name}** is valid — target `{cfg.target.module}`, output `{cfg.output_dir}`.")]
    # the project follows the YAML: folder, name and target come from what will be saved
    project.set_form({"output_dir": str(cfg.output_dir), "name": cfg.name, "source": cfg.target.source, "module": cfg.target.module,
                      "target": project.target_for_source(cfg.target.source)})
    if action != "validate":
        path = Path((project.form().get("config_path") or "").strip() or f"eval/pipeline/{cfg.name}.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        project.set_form({"config_path": str(path)})
        project.note_config_file(str(path))
        result.append(("info", f"Saved to `{path}`."))
    if action in ("dataset", "full"):
        try:
            job = jobs.start_job(path, cfg.output_dir, mode=action)
        except RuntimeError as e:
            result.append(("error", str(e)))
            st.session_state["setup_result"] = result
            return
        st.session_state["setup_notice"] = ("info", f"Started `{job['mode']}` job (pid {job['pid']}) for `{cfg.name}` → `{cfg.output_dir}`.")
        st.session_state["setup_goto"] = "run"
    st.session_state["setup_result"] = result


# ── rendering ────────────────────────────────────────────────────


def _render_preview(preview: dict) -> None:
    for p in preview.get("problems") or []:
        st.warning(p, icon=":material/warning:")
    if not preview.get("ok"):
        return
    with st.container(horizontal=True):
        st.metric("Graph nodes", len(preview["nodes"]), border=True)
        st.metric("Tools", len(preview["tools"]), border=True)
        st.metric("Skills", len(preview.get("skills") or []), border=True)
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
            "tool": t["name"], "kind": t.get("kind", "tool"), "schema source": t.get("schema_source"), "models": ", ".join(t.get("models") or []),
            "output schema": "yes" if t.get("output_schema") else "—", "side-effecting": bool(t.get("side_effecting")),
            "mocked": bool(t.get("mockable", True)), "edge cases": len(t.get("edge_cases") or []), "used by": ", ".join(t.get("used_by") or []),
        }
        for t in preview["tools"]
    ], column_config={"side-effecting": st.column_config.CheckboxColumn(),
                      "mocked": st.column_config.CheckboxColumn(help="skill loaders are local and never mocked")})
    if preview.get("skills"):
        st.markdown(f"**Agent skills** — `{preview.get('skills_dir')}`")
        table([
            {"skill": sk["name"], "description": sk.get("description", ""), "used by": ", ".join(sk.get("used_by") or []),
             "tools": ", ".join(sk.get("tools") or []), "allowed tools": ", ".join(sk.get("allowed_tools") or []),
             "summarized": bool(sk.get("summarized")), "references": len(sk.get("references") or []),
             "chars": len(sk.get("prompt") or "")}
            for sk in preview["skills"]
        ])
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


def _project_label(dir_: str, projects: dict[str, dict]) -> str:
    if dir_ == NEW_PROJECT:
        return NEW_PROJECT
    p = projects.get(dir_)
    if p is None:
        return dir_
    return f"{p['name']} ({dir_})" + ("" if p["has_artifacts"] else " — not run yet")


def _render_project() -> None:
    st.subheader("Project", anchor="project")
    st.caption("Every page follows this choice: an output folder (existing results, or where a new run will write) "
               "or uploaded artifacts. Clear it and every page returns to its empty state.")
    projects = {p["dir"]: p for p in project.projects()}
    current_dir = project.project_dir()
    options = [NEW_PROJECT] + list(projects)
    if current_dir and current_dir not in options:
        options.append(current_dir)
    st.session_state["setup_project"] = current_dir if current_dir else None
    c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment="bottom")
    with c1:
        st.selectbox("Project folder", options, key="setup_project", on_change=_apply_project, placeholder="no project selected",
                     format_func=lambda d: _project_label(d, projects))
    with c2:
        st.text_input("…or any output folder", key="setup_project_path", placeholder="path/to/eval/pipeline/<name>")
    with c3:
        st.button("Open", key="setup_open_path", icon=":material/folder_open:", on_click=_open_path)
    proj = project.current()
    if proj is None:
        st.info("No project yet — open a folder above or pick a target agent below.", icon=":material/folder_off:")
    elif proj["mode"] == "uploads":
        st.markdown(f":violet-badge[uploaded artifacts] **{project.name()}** — read-only (jobs need a folder)")
    else:
        status = jobs.job_status(Path(proj["dir"]))
        badge = {"running": ":blue-badge[job running]", "finished": ":green-badge[job finished]", "lost": ":orange-badge[job lost]"}.get(status["status"], ":grey-badge[not run yet]")
        st.markdown(f"{badge} **{project.name()}** · folder `{proj['dir']}`" + (f" · config `{project.config_path()}`" if project.config_path() else ""))
    with st.expander("Upload artifacts instead (read-only)"):
        uploads = st.file_uploader("Artifact files (JSON / YAML)", type=["json", "yaml", "yml"], accept_multiple_files=True, key="setup_uploads",
                                   help="Files are identified by their embedded `schema` id, then by file name.")
        if uploads:
            bundle = loader.load_files([(u.name, u.getvalue()) for u in uploads])
            proj = project.current()
            if not (proj and proj["mode"] == "uploads" and proj["bundle"].files == bundle.files):
                project.set_uploads(bundle)
                st.rerun()


def render() -> None:
    project.seed_widgets()
    st.header("Pipeline setup", anchor="setup")
    st.caption(
        "Choose the project, pick a target, discover its structure (no LLM), edit the config — constraints and free-text "
        "instructions reach every generation prompt — then run everything or only the dataset + mocks. "
        "The page keeps its state while you visit other pages."
    )
    goto = st.session_state.pop("setup_goto", None)
    if goto:
        pages = st.session_state.get("pages") or {}
        if goto in pages:
            st.switch_page(pages[goto])
    if st.session_state.get("setup_notice"):
        kind, text = st.session_state["setup_notice"]
        getattr(st, kind)(text)

    with st.container(border=True):
        _render_project()

    # ── target ────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Target agent", anchor="target")
        options = [t["source"] for t in project.targets()] + [CUSTOM]
        if st.session_state.get("setup_target") not in options:
            st.session_state["setup_target"] = None
        st.selectbox(
            "Example agent", options, key="setup_target", on_change=_apply_target, placeholder="choose a target agent…",
            format_func=lambda s: s if s == CUSTOM else f"{next(t['name'] for t in project.targets() if t['source'] == s)} ({s})",
        )
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.text_input("Agent source file", key="setup_source", placeholder="path/to/agent.py")
        with c2:
            st.text_input("Importable module", key="setup_module", placeholder="pkg.agent")
        with c3:
            st.text_input("Factory", key="setup_factory")
        st.button("Discover structure", key="setup_discover", icon=":material/search:", type="secondary", on_click=_discover,
                  disabled=not (st.session_state.get("setup_source") or "").strip())
        if st.session_state.get("setup_preview"):
            _render_preview(st.session_state["setup_preview"])

    # ── config form ───────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Config", anchor="config")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.text_input("Pipeline name", key="setup_name")
        with c2:
            st.text_input("Config file path", key="setup_config_path", placeholder="eval/pipeline/<name>.yaml")
        st.markdown("**Models** — `provider:model[@effort]`; default is Claude Sonnet 5 through the Claude Code CLI")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.text_input("Agent model (empty = target default)", key="setup_agent_model", help=f"e.g. {DEFAULT_MODEL} or scripted:module:factory")
        with m2:
            st.text_input("Judge model", key="setup_judge_model")
        with m3:
            st.text_input("Generator model", key="setup_generator_model")
        with m4:
            st.text_input("Mock model (empty = generator)", key="setup_mock_model",
                          help="drives the LLM mock engine (layer 2) when the miss policy is llm")
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
        e1, e2, e3, e4 = st.columns([2, 1, 1, 1])
        with e1:
            st.multiselect("Evaluators", list(setup_mod.EVALUATOR_CHOICES), key="setup_evaluators")
        with e2:
            st.selectbox("Mock miss policy", ["strict", "llm", "fallback", "real"], key="setup_on_miss",
                         help="layer 1 = deterministic rules; llm = an LLM mock engine answers calls no rule covers, from pre-generated strategies")
        with e3:
            st.selectbox("On invalid mock", ["fallback", "strict"], key="setup_on_invalid",
                         help="an engine answer still violating the tool's output schema after one repair: fallback (schema sample) or error")
        with e4:
            st.checkbox("Generate strategies", key="setup_strategies", help="pre-generate the LLM mock engine's strategies in the mocks stage (stored in dataset.mocks.strategies)")
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
            st.text_input("Output directory", key="setup_output_dir", placeholder="eval/pipeline/<name>",
                          help="The project folder: every page follows it.")
        st.button("Generate YAML", key="setup_generate", icon=":material/description:", type="primary", on_click=_generate_yaml)

    # ── yaml editor ───────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Config YAML", anchor="yaml")
        st.text_area("pipeline config (editable)", key="setup_yaml", height=420, label_visibility="collapsed")
        b1, b2, b3, b4 = st.columns([1, 1, 1.5, 2])
        with b1:
            st.button("Validate", key="setup_validate", icon=":material/rule:", on_click=_yaml_action, args=("validate",))
        with b2:
            st.button("Save config", key="setup_save", icon=":material/save:", on_click=_yaml_action, args=("save",))
        with b3:
            st.button("Generate dataset & mocks only", key="setup_run_dataset", icon=":material/dataset:", on_click=_yaml_action, args=("dataset",))
        with b4:
            st.button("Run full pipeline", key="setup_run_full", icon=":material/play_arrow:", type="primary", on_click=_yaml_action, args=("full",))
        for kind, text in st.session_state.get("setup_result") or []:
            getattr(st, kind)(text, icon={"error": ":material/error:", "success": ":material/check_circle:", "info": ":material/save:"}.get(kind))
    project.capture_widgets()
