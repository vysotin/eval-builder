"""The project — the one choice every UI page depends on.

`st.session_state["project"]` is `None`, `{"mode": "dir", "dir": <output folder>}` or
`{"mode": "uploads", "bundle": Bundle}`. Only the *Pipeline setup* page (and the
`--dir` start-up argument) changes it; the sidebar, *Run & review* and every report
page derive what they show from it, and clearing it clears them all.

The setup form lives in `st.session_state["setup_form"]` — a plain dict, because
Streamlit drops a keyed widget's value as soon as the widget is not rendered (i.e. on
navigating to another page). Widgets keyed `setup_<field>` are seeded from the dict
before they are created and captured back into it at the start of every run, so the
page can be left and resumed. In folder mode the project folder *is* the form's
output directory: pick a target → `eval/pipeline/<name>`; open an existing folder →
its config is loaded back into the form.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from evalbuilder.pipeline import setup as setup_mod
from evalbuilder.ui import loader

CUSTOM = "custom target…"
FORM_KEY = "setup_form"
WIDGET_PREFIX = "setup_"
# form fields without a widget (never seeded into / captured from session state)
INTERNAL_FIELDS = ("config_mtime",)
# setup-page state (besides the form) that a new project / clear must drop
SETUP_STATE_KEYS = ("setup_preview", "setup_notice", "setup_result", "setup_goto")
# run-page state that must not survive a change of folder
RUN_STATE_KEYS = ("review_feedback", "review_from", "proceed_by", "proceed_reject", "proceed_notice")


# ── form (persistent copy of the setup widgets) ─────────────────


def empty_form() -> dict:
    """Form defaults plus the UI-only fields: no target, no YAML, no project."""
    return {**setup_mod.default_form(), "target": None, "yaml": "", "config_mtime": None}


def form() -> dict:
    """The persistent setup form (created with empty defaults: no target, no project)."""
    if FORM_KEY not in st.session_state:
        st.session_state[FORM_KEY] = empty_form()
    return st.session_state[FORM_KEY]


def set_form(values: dict, replace: bool = False) -> None:
    """Update the persistent form and drop the widget copies so they re-seed.

    Call this from callbacks or start-up code only — never after the setup widgets
    were created in the current run."""
    current = empty_form() if replace else form()
    current.update(values)
    st.session_state[FORM_KEY] = current
    for key in list(values) if not replace else list(current):
        st.session_state.pop(f"{WIDGET_PREFIX}{key}", None)


def seed_widgets() -> None:
    """Before the setup widgets are created: give every missing widget key its form value."""
    for key, value in form().items():
        if key not in INTERNAL_FIELDS:
            st.session_state.setdefault(f"{WIDGET_PREFIX}{key}", value)


def capture_widgets() -> None:
    """At the start of a run: copy whatever the setup widgets hold back into the form."""
    current = form()
    for key in current:
        widget_key = f"{WIDGET_PREFIX}{key}"
        if key not in INTERNAL_FIELDS and widget_key in st.session_state:
            current[key] = st.session_state[widget_key]


def _mtime(path: str | None) -> int | None:
    try:
        return Path(path).stat().st_mtime_ns if path else None
    except OSError:
        return None


def note_config_file(path: str | None) -> None:
    """Remember the config file's timestamp (after opening or saving it) so a later
    change on disk — review feedback, an approval, a job — is noticed."""
    form()["config_mtime"] = _mtime(path)


def refresh_from_disk() -> None:
    """Reload the form + YAML when the project's config file changed on disk since it
    was opened or saved (the Run & review page and jobs write to it)."""
    f = form()
    path = (f.get("config_path") or "").strip()
    mtime = _mtime(path)
    if mtime is None or f.get("config_mtime") == mtime:
        return
    if f.get("config_mtime") is None:  # first sighting of a path typed by the user: just track it
        f["config_mtime"] = mtime
        return
    cfg = setup_mod._try_load(path)
    if cfg is None:
        return
    values = setup_mod.form_from_config(cfg, output_dir=(f.get("output_dir") or "").strip() or None, config_path=path)
    values.update(target=target_for_source(values["source"]), yaml=cfg.to_yaml(), config_mtime=mtime)
    set_form(values)
    st.session_state["setup_notice"] = ("info", f"Config reloaded from `{path}` — it changed on disk (review feedback, approval or a job).")


def form_values() -> dict:
    """The form fields `setup.build_config` expects (no UI-only keys)."""
    return {key: form().get(key) for key in setup_mod.default_form()}


# ── targets / projects discovered on disk ───────────────────────


def targets() -> list[dict]:
    if "setup_targets" not in st.session_state:
        st.session_state["setup_targets"] = setup_mod.discover_targets()
    return st.session_state["setup_targets"]


def target_for_source(source: str | None) -> str | None:
    """The target selectbox value for a source path: the path when it is a discovered
    example, CUSTOM for any other source, None when there is no target."""
    if not (source or "").strip():
        return None
    return source if any(t["source"] == source for t in targets()) else CUSTOM


def projects() -> list[dict]:
    return setup_mod.discover_projects()


# ── the project ─────────────────────────────────────────────────


def current() -> dict | None:
    return st.session_state.get("project")


def project_dir() -> str | None:
    proj = current()
    return proj["dir"] if proj and proj["mode"] == "dir" else None


def config_path() -> str | None:
    """The config file the project folder is tied to (form value, else what the folder records)."""
    path = (form().get("config_path") or "").strip()
    if path:
        return path
    out = project_dir()
    if out:
        return setup_mod.project_config(Path(out))[1]
    return None


def name() -> str:
    proj = current()
    if proj is None:
        return ""
    if proj["mode"] == "uploads":
        return proj["bundle"].name or "uploads"
    return (form().get("name") or "").strip() or Path(proj["dir"]).name


def sync() -> None:
    """Derive the project from the form: a non-empty output directory selects that
    folder; an empty one clears a folder project (uploads are left alone)."""
    out = (form().get("output_dir") or "").strip()
    proj = current()
    if out:
        if not (proj and proj["mode"] == "dir" and proj["dir"] == out):
            st.session_state["project"] = {"mode": "dir", "dir": out}
            _reset_run_state()
    elif proj and proj["mode"] == "dir":
        st.session_state["project"] = None
        _reset_run_state()


def open_dir(path: str) -> dict:
    """Make `path` the project: load its config into the form (target, models, coverage,
    YAML …) when one can be found, else an empty form pointed at the folder.
    Returns {"config": PipelineConfig | None, "config_path": str | None}."""
    path = str(path).strip()
    cfg, cfg_path = setup_mod.project_config(Path(path))
    if cfg is not None:
        values = setup_mod.form_from_config(cfg, output_dir=path, config_path=cfg_path)
        values["target"] = target_for_source(values["source"])
        values["yaml"] = cfg.to_yaml()
        values["config_mtime"] = _mtime(cfg_path)
        set_form(values, replace=True)
        note = f"Opened **{cfg.name}** from `{path}`" + (f" (config `{cfg_path}`)." if cfg_path else " (config recovered from report.json — save it to a file before running).")
    else:
        set_form({"name": Path(path).name, "output_dir": path, "config_path": ""}, replace=True)
        note = f"Opened `{path}` — no pipeline config found next to it; pick a target agent to create one."
    st.session_state["project"] = {"mode": "dir", "dir": path}
    _reset_setup_state()
    _reset_run_state()
    st.session_state["setup_notice"] = ("info", note)
    return {"config": cfg, "config_path": cfg_path}


def set_uploads(bundle: loader.Bundle) -> None:
    st.session_state["project"] = {"mode": "uploads", "bundle": bundle}
    set_form({}, replace=True)
    _reset_setup_state()
    _reset_run_state()


def clear() -> None:
    """No project: empty form, no bundle, every page back to its empty state."""
    st.session_state["project"] = None
    set_form({}, replace=True)
    _reset_setup_state()
    _reset_run_state()
    st.session_state["bundle"] = None


def _reset_setup_state() -> None:
    for key in SETUP_STATE_KEYS:
        st.session_state.pop(key, None)


def _reset_run_state() -> None:
    for key in RUN_STATE_KEYS:
        st.session_state.pop(key, None)


# ── bundle ──────────────────────────────────────────────────────


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


def bundle() -> loader.Bundle | None:
    """The artifacts of the current project (None without a project; an empty bundle
    for a folder that has not run yet)."""
    proj = current()
    if proj is None:
        return None
    if proj["mode"] == "uploads":
        return proj["bundle"]
    path = proj["dir"]
    if not Path(path).is_dir():
        return loader.Bundle(name=name() or Path(path).name, source=path)
    b = load_bundle_from_dir(path)
    if not any(b.has(k) for k in b.artifacts):
        b = loader.Bundle(name=name() or Path(path).name, source=path)
    return b


def begin_run(initial_dir: str | None = None) -> None:
    """Per-run bookkeeping the entry point does before any page renders: open the
    start-up folder once, capture the setup widgets, derive the project, load the bundle."""
    st.session_state.setdefault("project", None)
    if not st.session_state.get("project_started"):
        st.session_state["project_started"] = True
        proj = current()
        if initial_dir:
            open_dir(initial_dir)
        elif proj and proj["mode"] == "dir" and not (form().get("output_dir") or "").strip():
            open_dir(proj["dir"])  # a project injected from outside (tests, embedding): adopt it into the form
    capture_widgets()
    refresh_from_disk()
    sync()
    st.session_state["bundle"] = bundle()


def setup_link(label: str = "Open Pipeline setup") -> None:
    pages = st.session_state.get("pages") or {}
    if "setup" in pages:
        st.page_link(pages["setup"], label=label, icon=":material/tune:")


def no_project_hint() -> None:
    st.info("No project selected — choose an existing output folder or a target agent on **Pipeline setup**.", icon=":material/folder_off:")
    setup_link()
