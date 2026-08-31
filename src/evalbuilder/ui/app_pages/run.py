"""Run & review — live status of the project's background pipeline job, the generation
review loop (feedback → regenerate) and the hand-off to evaluation and the report pages.
The page follows the project chosen on Pipeline setup; it has no selector of its own."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from evalbuilder.pipeline import jobs
from evalbuilder.pipeline import setup as setup_mod
from evalbuilder.ui import loader, project
from evalbuilder.ui.common import job_badge_line, job_progress, status_md, table, verdict_badge


def _status_block(out_dir: Path, live: bool) -> dict:
    """Job status + progress bars + stop button + stage table; re-polls every 2 s
    while the job is running."""

    @st.fragment(run_every="2s" if live else None)
    def block() -> None:
        status = jobs.job_status(out_dir)
        job = status.get("job") or {}
        line = job_badge_line(status)
        if job:
            line += f" · mode `{job.get('mode')}`" + (f" from `{job.get('from_stage')}`" if job.get("from_stage") else "")
            line += f" · started {str(job.get('started_at', ''))[:19].replace('T', ' ')}"
            if status["status"] in ("finished", "stopped"):
                line += f" · exit code `{status.get('exit_code')}`"
        if status.get("stopped_after"):
            line += f" · stopped after **{status['stopped_after']}**"
        st.markdown(line)
        if status["status"] == "running":
            if st.button("Stop job", key="run_stop", icon=":material/stop_circle:",
                         help="Terminate the background pipeline process. Completed stages keep "
                              "their artifacts; resume later from Pipeline setup or the review loop."):
                jobs.stop_job(out_dir)
                st.rerun(scope="app")
        job_progress(status)
        rp = status.get("run_progress")
        if rp and (rp.get("current") or []) and (
            (status.get("progress") or {}).get("stage") == "run" or status.get("interrupted_stage") == "run"
        ):
            st.markdown("**Partial results** — cases completed in the current repeat")
            table(rp["current"], columns=["case_id", "intent", "error_class", "error"])
        if job.get("config_path"):
            st.caption(f"config `{job['config_path']}` · log `{job.get('log')}`")
        rows = [
            {"stage": name, "status": rec.get("status"), "seconds": round(rec.get("seconds") or 0, 1),
             "attempts": rec.get("attempts"), "note": (rec.get("reason") or rec.get("error") or "")[:140]}
            for name, rec in status["stages"].items()
        ]
        if rows:
            st.markdown("**Stages** — " + " ".join(f"{r['stage']} {status_md(r['status'])}" for r in rows))
            table(rows)
        else:
            st.caption("no stage has run yet")
        with st.expander("Pipeline log", expanded=status["status"] == "running"):
            st.code(jobs.tail_log(out_dir, 60) or "(empty)", language="text")
        if live and status["status"] != "running":
            st.rerun(scope="app")

    block()
    return jobs.job_status(out_dir)


def _generation_done(status: dict) -> bool:
    stages = status.get("stages") or {}
    return stages.get("dataset", {}).get("status") in ("ok", "recovered") and stages.get("run", {}).get("status") not in ("ok", "recovered")


def _review_section(out_dir: Path, status: dict, config_path: str | None) -> None:
    summary = setup_mod.dataset_summary(out_dir)
    st.subheader("Review & iterate", anchor="review")
    if summary is None:
        st.caption("no dataset yet")
        return
    with st.container(horizontal=True):
        st.metric("Cases", summary["cases"], border=True)
        for s in ("pending", "approved", "rejected"):
            st.metric(s.capitalize(), summary["by_status"].get(s, 0), border=True)
        st.metric("Schema-edge cases", len(summary["schema_edge_cases"]), border=True)
        st.metric("Mocked tools", len(summary["mocked_tools"]), border=True)
    st.caption("failure modes: " + ", ".join(f"`{k}` {v}" for k, v in sorted(summary["by_failure_mode"].items())))
    if summary["schema_edge_cases"]:
        st.markdown("**Schema edge cases** (from tool input/output schemas)")
        table(summary["schema_edge_cases"], column_config={"mock_override": st.column_config.CheckboxColumn("mock override")})
    st.caption("Open **Dataset & mocks** in the sidebar for every case and fixture.")

    if not config_path:
        st.warning("No config file is known for this folder; save one from the Setup page first.")
        project.setup_link()
        return

    st.markdown("**Feedback for the generator** — appended to the config (`feedback`) and read by every generation prompt on rerun.")
    st.text_area("Comment / new instruction", key="review_feedback", height=100,
                 placeholder="e.g. Add more multi-turn refund cases; use European order ids; avoid duplicate topics…")
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        st.selectbox("Regenerate from", list(jobs.GENERATION_STAGES), index=2, key="review_from")
    with f2:
        save_only = st.button("Save feedback", key="review_save", icon=":material/comment:")
    with f3:
        regenerate = st.button("Save feedback & regenerate", key="review_regen", icon=":material/autorenew:", type="primary")
    if save_only or regenerate:
        note = (st.session_state.get("review_feedback") or "").strip()
        if not note:
            st.error("Write a comment first.")
        else:
            cfg = setup_mod.add_feedback(Path(config_path), note, st.session_state["review_from"])
            st.success(f"Feedback saved ({len(cfg.feedback)} entries in `{config_path}`).", icon=":material/check_circle:")
            if regenerate:
                try:
                    jobs.start_job(Path(config_path), out_dir, mode="regenerate", from_stage=st.session_state["review_from"])
                except RuntimeError as e:
                    st.error(str(e))
                    return
                st.rerun()

    st.subheader("Proceed with evaluation", anchor="proceed")
    st.caption("Approving is your decision: the name below is recorded as `review.approved_by`; selected cases are rejected first.")
    p1, p2 = st.columns([1, 2])
    with p1:
        st.text_input("Approved by", key="proceed_by", placeholder="your name")
    with p2:
        st.multiselect("Reject these cases", summary["case_ids"], key="proceed_reject",
                       format_func=lambda i: f"{i} — {summary['inputs'].get(i, '')}")
    if st.button("Approve remaining cases & run evaluation", key="proceed_run", icon=":material/play_arrow:", type="primary"):
        by = (st.session_state.get("proceed_by") or "").strip()
        if not by:
            st.error("Enter the approver's name first.")
            return
        rejected = setup_mod.reject_cases(out_dir, list(st.session_state.get("proceed_reject") or []), "rejected before evaluation", by)
        setup_mod.approve_in_config(Path(config_path), by, note="approved in the UI")
        try:
            jobs.start_job(Path(config_path), out_dir, mode="resume")
        except RuntimeError as e:
            st.error(str(e))
            return
        st.session_state["proceed_notice"] = f"Rejected {rejected} case(s); evaluation started for {by}."
        st.rerun()


def _results_section(out_dir: Path) -> None:
    bundle = loader.load_dir(str(out_dir))
    st.subheader("Results", anchor="results")
    report = bundle.report or {}
    c1, c2 = st.columns([1, 4], vertical_alignment="center")
    with c1:
        verdict_badge(bundle.verdict)
    with c2:
        st.markdown(
            f"overall score **{report.get('overall_score', '—')}** · cases {len((bundle.dataset or {}).get('cases', []))} · "
            f"runs {len(bundle.run_ids)} · artifacts {len(bundle.summary()['artifacts'])}"
        )
    for reason in (report.get("verdict_reasons") or [])[:5]:
        st.markdown(f"- {reason}")
    if st.button("Open results in the report pages", key="run_open_results", icon=":material/analytics:", type="primary"):
        pages = st.session_state.get("pages") or {}
        if "summary" in pages:
            st.switch_page(pages["summary"])
        st.rerun()


def render() -> None:
    st.header("Run & review", anchor="run")
    proj = project.current()
    if proj is None:
        project.no_project_hint()
        return
    if proj["mode"] == "uploads":
        st.info("Uploaded artifacts are read-only — jobs need a project folder. Choose or create one on **Pipeline setup**.", icon=":material/folder_off:")
        project.setup_link()
        return
    out_dir = Path(proj["dir"])
    config_path = project.config_path()
    st.caption(f"project **{project.name()}** · folder `{out_dir}`" + (f" · config `{config_path}`" if config_path else ""))
    if st.session_state.get("proceed_notice"):
        st.success(st.session_state.pop("proceed_notice"))

    initial = jobs.job_status(out_dir)
    if initial["status"] == "none" and not initial["stages"]:
        st.info(f"Nothing has run in `{out_dir}` yet — start a job from **Pipeline setup** (*Generate dataset & mocks only* or *Run full pipeline*).",
                icon=":material/hourglass_empty:")
        project.setup_link()
        return
    with st.container(border=True):
        st.subheader("Job status", anchor="status")
        status = _status_block(out_dir, live=initial["status"] == "running")
    if status["status"] == "running":
        st.caption("The page refreshes every 2 seconds while the job runs.")
        return

    job = status.get("job") or {}
    recorded = job.get("config_path")
    if not recorded:
        report_path = out_dir / "report.json"
        if report_path.exists():
            try:
                recorded = json.loads(report_path.read_text()).get("config_path")
            except ValueError:
                recorded = None
    if recorded and Path(recorded).exists():
        config_path = recorded
    elif config_path and not Path(config_path).exists():
        config_path = None
    if _generation_done(status):
        with st.container(border=True):
            _review_section(out_dir, status, config_path)
    if (out_dir / "report.json").exists():
        with st.container(border=True):
            _results_section(out_dir)
