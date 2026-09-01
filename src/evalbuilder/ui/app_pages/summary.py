"""Summary — verdict, headline stats, metric pass rates, coverage, stability, problems of the project."""

from __future__ import annotations

import streamlit as st

from evalbuilder.ui.common import (
    duration_chart,
    get_bundle,
    num,
    pass_rate_chart,
    pct,
    require,
    status_md,
    table,
    verdict_badge,
)


def render() -> None:
    bundle = get_bundle()
    if not require(bundle):
        return
    report = bundle.report or {}
    coverage = bundle.get("coverage") or report.get("coverage") or {}
    stability = (bundle.aggregate or {}).get("stability") or report.get("stability") or {}
    thresholds = (bundle.config.get("thresholds") or {})

    st.header("Summary", anchor="summary")
    left, right = st.columns([1, 3], vertical_alignment="center")
    with left:
        verdict_badge(bundle.verdict)
    with right:
        st.markdown(
            f"**{bundle.name}** · target `{(bundle.config.get('target') or {}).get('module', '?')}` · "
            f"generated {report.get('generated_at', '—')[:19].replace('T', ' ')}"
        )
    reasons = report.get("verdict_reasons") or (bundle.aggregate or {}).get("verdict_reasons") or []
    if len(reasons) <= 3:
        for reason in reasons:
            st.markdown(f"- {reason}")
    elif reasons:
        with st.expander(f"{len(reasons)} verdict reasons", expanded=False):
            for reason in reasons:
                st.markdown(f"- {reason}")

    # headline numbers
    metrics = (bundle.aggregate or {}).get("metrics") or report.get("metrics") or {}
    ds = bundle.dataset or {}
    cases = ds.get("cases", [])
    approved = sum(1 for c in cases if (c.get("review") or {}).get("status") == "approved")
    with st.container(horizontal=True):
        overall = (bundle.aggregate or {}).get("overall_score", report.get("overall_score"))
        st.metric(
            "Overall score", num(overall), border=True,
            help=f"mean of metric pass rates; target overall_pass = {thresholds.get('overall_pass', 0.8)}",
        )
        st.metric("Metrics passing", f"{sum(1 for v in metrics.values() if v.get('passed'))}/{len(metrics)}", border=True)
        st.metric("Cases (approved)", f"{len(cases)} ({approved})", border=True)
        st.metric("Coverage", f"{coverage.get('coverage_pct', '—')}%", border=True,
                  help="covered / planned cells")
        st.metric("Repeats", str(stability.get("repeats", len(bundle.run_ids))), border=True)
        st.metric("Stable cases", pct(stability.get("stable_case_fraction")), border=True,
                  help="cases whose tool trajectory was identical across repeats")

    col_metrics, col_stages = st.columns(2)
    with col_metrics:
        with st.container(border=True):
            st.subheader("Metric pass rates vs thresholds", anchor="metric-pass-rates")
            if metrics:
                rows = [
                    {"metric": m, "pass_rate": v.get("pass_rate") or 0.0, "threshold": v.get("threshold", 0),
                     "passed": bool(v.get("passed"))}
                    for m, v in metrics.items()
                ]
                st.altair_chart(pass_rate_chart(rows), width="stretch")
                table(
                    [
                        {"metric": m, "pass rate": v.get("pass_rate"), "min over repeats": v.get("min_over_repeats"),
                         "threshold": v.get("threshold"), "cases": v.get("n_cases"), "scores": v.get("n_scores"),
                         "errors": v.get("errors"), "passed": bool(v.get("passed"))}
                        for m, v in metrics.items()
                    ],
                    column_config={
                        "pass rate": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                        "passed": st.column_config.CheckboxColumn(),
                    },
                )
            else:
                st.caption("no metrics yet — the score/aggregate stages have not run")
    with col_stages:
        with st.container(border=True):
            st.subheader("Stages", anchor="stages-summary")
            stages = report.get("stages") or (bundle.state or {}).get("stages") or {}
            if stages:
                st.markdown(" ".join(f"{status_md(rec.get('status'))} {name}" for name, rec in stages.items()))
                rows = [
                    {"stage": name, "status": rec.get("status"), "seconds": rec.get("seconds") or 0.0,
                     "attempts": rec.get("attempts", 0)}
                    for name, rec in stages.items()
                ]
                st.altair_chart(duration_chart(rows), width="stretch")
            else:
                st.caption("no stage state (state.json / report.json missing)")

    col_cov, col_stab = st.columns(2)
    with col_cov:
        with st.container(border=True):
            st.subheader("Coverage", anchor="coverage-summary")
            if coverage:
                st.markdown(
                    f"**{coverage.get('covered')} / {coverage.get('planned')}** planned cells covered "
                    f"({coverage.get('coverage_pct')}%) · multi-turn {coverage.get('multi_turn', {}).get('have', '—')}"
                    f"/{coverage.get('multi_turn', {}).get('planned', '—')} · gaps: {len(coverage.get('gaps') or [])}"
                )
                table([{"kind": k, **v} for k, v in (coverage.get("by_kind") or {}).items()])
            else:
                st.caption("no coverage artifact")
    with col_stab:
        with st.container(border=True):
            st.subheader("Stability", anchor="stability-summary")
            if stability:
                st.markdown(
                    f"- unstable **cases** (trajectory differs): {len(stability.get('unstable_cases') or [])}\n"
                    f"- unstable **outputs** (deterministic metric flips): {len(stability.get('unstable_outputs') or [])}\n"
                    f"- unstable **evaluators** (judge flips): {len(stability.get('unstable_evaluators') or [])}\n"
                    f"- wording varies: {stability.get('text_varies', 0)} · suspect judge comments: "
                    f"{len(stability.get('suspect_judge_comments') or [])}"
                )
            else:
                st.caption("no stability data (needs ≥ 1 scored run)")

    mocking = report.get("mocking") or {}
    if mocking:
        with st.container(border=True):
            st.subheader("Mocking", anchor="mocking-summary")
            calls = mocking.get("calls") or {}
            layers = " → ".join(mocking.get("layers") or ["rules"])
            line = f"layers **{layers}** · miss policy `{mocking.get('on_miss')}`"
            if mocking.get("model"):
                line += f" · mock model `{mocking['model']}` · strategy `{mocking.get('strategy')}` · on invalid `{mocking.get('on_invalid')}`"
            if mocking.get("strategies"):
                line += " · strategies " + " ".join(f":blue-badge[{sid}]" for sid in mocking["strategies"])
            st.markdown(line)
            if calls:
                st.caption("tool calls answered by layer: " + " · ".join(f"`{k}` {v}" for k, v in calls.items()))
            llm_unstable = (stability or {}).get("llm_mocked_unstable") or []
            if llm_unstable:
                st.warning(f"{len(llm_unstable)} unstable case(s) had LLM-mocked tool answers — attribute the instability to the mock layer before the agent.", icon=":material/warning:")

    analysis = bundle.get("analysis") or report.get("analysis")
    if analysis:
        with st.container(border=True):
            st.subheader("Analysis summary", anchor="analysis-summary")
            st.caption(f"source: {analysis.get('source', '?')}")
            st.markdown(analysis.get("summary", ""))

    problems = report.get("problems") or ((bundle.state or {}).get("data") or {}).get("problems") or []
    if problems or bundle.problems:
        with st.container(border=True):
            st.subheader(f"Problems ({len(problems) + len(bundle.problems)})", anchor="problems-summary")
            for p in bundle.problems:
                st.markdown(f":red-badge[loader] {p}")
            table(problems, ["severity", "stage", "message"])
