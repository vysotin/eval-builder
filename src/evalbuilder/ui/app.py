"""Streamlit entry point: `evalbuilder ui [DIR]` or `streamlit run src/evalbuilder/ui/app.py -- --dir DIR`.

Every page depends on one choice — the *project* made on the Pipeline setup page (an
output folder, or uploaded artifacts). `project.begin_run` derives the project and its
`Bundle` (`st.session_state["bundle"]`) before any page renders; the sidebar only shows
that project and lets the user clear it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

from evalbuilder.pipeline import jobs
from evalbuilder.pipeline.layout import ARTIFACTS
from evalbuilder.ui import project
from evalbuilder.ui.common import job_progress
from evalbuilder.ui.app_pages import (
    agent,
    analysis,
    coverage,
    dataset,
    intents,
    results,
    run,
    setup,
    simulation,
    stability,
    stages,
    summary,
)

st.set_page_config(page_title="evalbuilder", page_icon=":material/analytics:", layout="wide")


def _initial_dir() -> str | None:
    """`--dir` after `--` on the streamlit command line, else EVALBUILDER_UI_DIR, else a query param."""
    argv = sys.argv
    if "--dir" in argv:
        i = argv.index("--dir")
        if i + 1 < len(argv):
            return argv[i + 1]
    env = os.environ.get("EVALBUILDER_UI_DIR")
    if env:
        return env
    qp = st.query_params.get("dir")
    return qp or None


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### Project")
        proj = project.current()
        if proj is None:
            st.info("No project selected.", icon=":material/folder_off:")
            project.setup_link("Choose one on Pipeline setup")
        else:
            bundle = st.session_state.get("bundle")
            if proj["mode"] == "uploads":
                st.markdown(f"**{project.name()}** · uploaded artifacts (read-only)")
            else:
                out = proj["dir"]
                st.markdown(f"**{project.name()}**")
                st.caption(f"folder `{out}`" + (f" · config `{project.config_path()}`" if project.config_path() else ""))
                status = jobs.job_status(Path(out))
                if status["status"] != "none":
                    live = status["status"] == "running"

                    @st.fragment(run_every="2s" if live else None)
                    def _job_box() -> None:
                        s = jobs.job_status(Path(out))
                        job_progress(s, compact=True)  # badge + progress bars, refreshed live
                        if live and s["status"] != "running":
                            st.rerun(scope="app")

                    _job_box()
            if bundle is not None:
                s = bundle.summary()
                if s["artifacts"]:
                    st.markdown(f"{len(s['artifacts'])}/{len(ARTIFACTS)} artifact kinds")
                    st.caption(f"cases {s['cases']} · intents {s['intents']} · scenarios {s['scenarios']} · tools {s['tools']}"
                               + (f" · skills {s['skills']}" if s.get("skills") else "") + f" · runs {s['runs']}")
                else:
                    st.caption("no artifacts yet")
                for p in bundle.problems:
                    st.warning(p, icon=":material/warning:")
            st.button("Clear project", key="sidebar_clear", icon=":material/close:", on_click=project.clear,
                      help="Forget the project: every page returns to its empty state.")
        with st.expander("Naming convention"):
            st.markdown("\n".join(f"- `{a.file}` — {a.kind}" for a in ARTIFACTS.values()))


def main() -> None:
    setup_page = st.Page(setup.render, title="Pipeline setup", icon=":material/tune:", default=True, url_path="setup")
    run_page = st.Page(run.render, title="Run & review", icon=":material/play_circle:", url_path="run")
    summary_page = st.Page(summary.render, title="Summary", icon=":material/dashboard:", url_path="summary")
    st.session_state["pages"] = {"setup": setup_page, "run": run_page, "summary": summary_page}
    pages = st.navigation(
        {
            "Pipeline": [setup_page, run_page],
            "Agent": [
                st.Page(agent.render, title="Agent graph & tools", icon=":material/account_tree:", url_path="agent"),
                st.Page(intents.render, title="Intents & scenarios", icon=":material/psychology:", url_path="intents"),
            ],
            "Data": [
                st.Page(dataset.render, title="Dataset & mocks", icon=":material/dataset:", url_path="dataset"),
                st.Page(coverage.render, title="Coverage", icon=":material/grid_on:", url_path="coverage"),
            ],
            "Evaluation": [
                st.Page(results.render, title="Eval results", icon=":material/query_stats:", url_path="results"),
                st.Page(stability.render, title="Stability", icon=":material/vibration:", url_path="stability"),
                st.Page(simulation.render, title="Simulation", icon=":material/forum:", url_path="simulation"),
                st.Page(analysis.render, title="Analysis", icon=":material/lightbulb:", url_path="analysis"),
                summary_page,
            ],
            "Run": [st.Page(stages.render, title="Stages & problems", icon=":material/timeline:", url_path="stages")],
        },
        position="sidebar",
    )
    project.begin_run(_initial_dir())
    sidebar()
    pages.run()


main()
