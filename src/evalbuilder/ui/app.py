"""Streamlit entry point: `evalbuilder ui [DIR]` or `streamlit run src/evalbuilder/ui/app.py -- --dir DIR`.

The sidebar picks an artifact source (a pipeline output directory discovered under
`eval/pipeline/` and `docs/examples/`, any path, or uploaded files); the loaded
`Bundle` lives in `st.session_state["bundle"]` and every page reads from it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

from evalbuilder.pipeline.layout import ARTIFACTS
from evalbuilder.ui import loader
from evalbuilder.ui.app_pages import (
    agent,
    analysis,
    coverage,
    dataset,
    intents,
    overview,
    results,
    run,
    setup,
    simulation,
    stability,
    stages,
)

st.set_page_config(page_title="evalbuilder report", page_icon=":material/analytics:", layout="wide")


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


@st.cache_data(show_spinner="Loading artifacts…")
def _load_dir_cached(path: str, fingerprint: tuple) -> loader.Bundle:  # fingerprint busts the cache on file changes
    return loader.load_dir(path)


def _fingerprint(path: str) -> tuple:
    p = Path(path)
    if not p.is_dir():
        return ()
    return tuple(sorted((str(f.relative_to(p)), f.stat().st_mtime_ns) for f in p.rglob("*") if f.is_file()))


def load_bundle_from_dir(path: str) -> loader.Bundle:
    return _load_dir_cached(path, _fingerprint(path))


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### Artifact source")
        discovered = loader.discover_dirs()
        initial = _initial_dir()
        opened = st.session_state.pop("open_dir", None)  # hand-off from the Run & review page
        if opened:
            st.session_state["source_mode"] = "directory"
            st.session_state["source_custom"] = opened
            initial = opened
        if initial and initial not in discovered:
            discovered.insert(0, initial)
        mode = st.radio("Load from", ["directory", "uploaded files"], key="source_mode", horizontal=True)
        if mode == "directory":
            options = discovered or []
            default_index = options.index(initial) if initial in options else 0
            chosen = st.selectbox("Pipeline output directory", options, index=default_index if options else None,
                                  key="source_dir", placeholder="no pipeline outputs found")
            custom = st.text_input("…or any path", key="source_custom", placeholder="path/to/eval/pipeline/<name>")
            path = custom.strip() or chosen
            if path:
                bundle = load_bundle_from_dir(path)
                st.session_state["bundle"] = bundle
        else:
            uploads = st.file_uploader(
                "Artifact files (JSON / YAML)", type=["json", "yaml", "yml"], accept_multiple_files=True, key="source_uploads",
                help="Files are identified by their embedded `schema` id, then by file name.",
            )
            if uploads:
                st.session_state["bundle"] = loader.load_files([(u.name, u.getvalue()) for u in uploads])
            elif st.session_state.get("bundle") is not None and st.session_state["bundle"].source == "uploads":
                pass
            else:
                st.session_state["bundle"] = None

        bundle = st.session_state.get("bundle")
        if bundle is not None:
            s = bundle.summary()
            st.markdown(f"**{s['name']}** · {len(s['artifacts'])}/{len(ARTIFACTS)} artifact kinds")
            st.caption(
                f"cases {s['cases']} · intents {s['intents']} · scenarios {s['scenarios']} · tools {s['tools']} · runs {s['runs']}"
            )
            for p in bundle.problems:
                st.warning(p, icon=":material/warning:")
        with st.expander("Naming convention"):
            st.markdown("\n".join(f"- `{a.file}` — {a.kind}" for a in ARTIFACTS.values()))


def main() -> None:
    st.session_state.setdefault("bundle", None)
    overview_page = st.Page(overview.render, title="Overview", icon=":material/dashboard:", default=True, url_path="overview")
    setup_page = st.Page(setup.render, title="Pipeline setup", icon=":material/tune:", url_path="setup")
    run_page = st.Page(run.render, title="Run & review", icon=":material/play_circle:", url_path="run")
    st.session_state["pages"] = {"overview": overview_page, "setup": setup_page, "run": run_page}
    pages = st.navigation(
        {
            "": [overview_page],
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
            ],
            "Run": [st.Page(stages.render, title="Stages & problems", icon=":material/timeline:", url_path="stages")],
        },
        position="sidebar",
    )
    sidebar()
    pages.run()


main()
