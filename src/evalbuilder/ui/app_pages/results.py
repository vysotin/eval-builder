"""Eval results — per-run metrics, aggregate vs thresholds, slices, per-case matrix, drill-down."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from evalbuilder.ui.common import (
    case_input,
    code_json,
    get_bundle,
    heatmap,
    num,
    pass_rate_chart,
    require,
    table,
    tool_calls_inline,
    transcript,
    verdict_badge,
)


def _score_matrix(bundle) -> pd.DataFrame:
    """Rows: case × run, columns: metric scores (None when missing)."""
    rows = []
    for run_id in bundle.run_ids:
        rep = bundle.score_reports.get(run_id) or {}
        for row in rep.get("cases", []):
            entry = {"case": row["case_id"], "run": run_id}
            for metric, value in (row.get("scores") or {}).items():
                entry[metric] = value.get("score")
            for metric, err in (row.get("errors") or {}).items():
                entry[metric] = None
                entry[f"{metric}·error"] = str(err)[:80]
            rows.append(entry)
    return pd.DataFrame(rows)


def render() -> None:
    bundle = get_bundle()
    if not require(bundle):
        return
    agg = bundle.aggregate or {}
    reports = bundle.score_reports
    runs = bundle.runs
    if not agg and not reports:
        require(bundle, "score_report", "aggregate")
        return
    ds_cases = {c.get("id"): c for c in (bundle.dataset or {}).get("cases") or []}
    thresholds = bundle.config.get("thresholds") or {}

    st.header("Eval results", anchor="results")
    c1, c2 = st.columns([1, 5], vertical_alignment="center")
    with c1:
        verdict_badge(agg.get("verdict") or bundle.verdict)
    with c2:
        st.markdown(
            f"overall **{num(agg.get('overall_score'))}** vs `overall_pass` {agg.get('overall_pass', thresholds.get('overall_pass', '—'))} · "
            f"{agg.get('repeats', len(bundle.run_ids))} repeat(s) · slice_min {thresholds.get('slice_min', '—')}"
        )

    # ── aggregate metrics ────────────────────────────────────────
    metrics = agg.get("metrics") or {}
    if metrics:
        with st.container(border=True):
            st.subheader("Aggregate metrics", anchor="aggregate-metrics")
            rows = [
                {"metric": m, "pass_rate": v.get("pass_rate") or 0.0, "threshold": v.get("threshold", 0), "passed": bool(v.get("passed"))}
                for m, v in metrics.items()
            ]
            st.altair_chart(pass_rate_chart(rows), width="stretch")
            table(
                [{"metric": m, "pass rate": v.get("pass_rate"), "min over repeats": v.get("min_over_repeats"),
                  "threshold": v.get("threshold"), "cases": v.get("n_cases"), "scores": v.get("n_scores"),
                  "errors": v.get("errors"), "passed": bool(v.get("passed"))} for m, v in metrics.items()],
                column_config={"pass rate": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                               "passed": st.column_config.CheckboxColumn()},
            )

    # ── per-run metrics ──────────────────────────────────────────
    if reports:
        with st.container(border=True):
            st.subheader("Per-run metrics", anchor="per-run-metrics")
            rows = []
            for run_id in bundle.run_ids:
                rep = reports.get(run_id) or {}
                run = runs.get(run_id) or {}
                for m, v in (rep.get("metrics") or {}).items():
                    rows.append({"run": run_id, "metric": m, "avg": v.get("avg"), "min": v.get("min"), "max": v.get("max"),
                                 "n": v.get("n"), "errors": v.get("errors"), "skipped": v.get("skipped", 0),
                                 "agent model": run.get("agent_model"), "timestamp": (run.get("timestamp") or "")[:19]})
            table(rows, column_config={"avg": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
            if len(bundle.run_ids) > 1 and rows:
                df = pd.DataFrame(rows)
                df["avg"] = df["avg"].fillna(0.0)
                st.altair_chart(heatmap(df, "run", "metric", "avg", "metric average per run"), width="stretch")

    # ── slices ───────────────────────────────────────────────────
    slices = agg.get("slices") or {}
    if slices:
        with st.container(border=True):
            st.subheader("Slices", anchor="slices")
            weak = set(agg.get("weak_slices") or [])
            if weak:
                st.markdown("Weak slices (below `slice_min`): " + " ".join(f":red-badge[{w}]" for w in sorted(weak)))
            tabs = st.tabs(list(slices))
            for tab, dim in zip(tabs, slices):
                with tab:
                    rows = []
                    for value, entry in slices[dim].items():
                        for m, rate in (entry.get("metrics") or {}).items():
                            rows.append({dim: f"{value} (n={entry.get('n')})", "metric": m, "pass_rate": rate})
                    if rows:
                        st.altair_chart(heatmap(pd.DataFrame(rows), "metric", dim, "pass_rate"), width="stretch")
                    table([{dim: value, "n": e.get("n"), "passed": bool(e.get("passed")), **(e.get("metrics") or {})}
                           for value, e in slices[dim].items()],
                          column_config={"passed": st.column_config.CheckboxColumn()})

    # ── per-case matrix ──────────────────────────────────────────
    matrix = _score_matrix(bundle)
    if not matrix.empty:
        with st.container(border=True):
            st.subheader("Per-case scores across runs", anchor="case-matrix")
            metric_cols = [c for c in matrix.columns if c not in ("case", "run") and "·error" not in c]
            long = matrix.melt(id_vars=["case", "run"], value_vars=metric_cols, var_name="metric", value_name="score").dropna()
            if not long.empty:
                long["metric·run"] = long["metric"] + " @" + long["run"]
                st.altair_chart(heatmap(long, "metric·run", "case", "score", fmt=".1f", label_angle=-25), width="stretch")
            table(matrix.to_dict("records"))

    # ── failing cases ────────────────────────────────────────────
    failing = agg.get("failing_cases") or []
    with st.container(border=True):
        st.subheader(f"Failing cases ({len(failing)})", anchor="failing-cases")
        if not failing:
            st.success("No failing cases.", icon=":material/check_circle:")
        for f in failing:
            metrics_md = ", ".join(f"`{m}`={num(v)}" for m, v in (f.get("failing_metrics") or {}).items())
            with st.expander(f"{f['id']} · {f.get('intent')} · {f.get('failure_mode')} · {metrics_md}"):
                st.markdown(f"> {f.get('input', '')}")
                for m, comments in (f.get("comments") or {}).items():
                    for c in comments:
                        st.markdown(f"- **{m}**: {c}")
                for err in f.get("agent_errors") or []:
                    st.markdown(f"- :red-badge[agent error] {err}")

    # ── drill-down ───────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Case drill-down", anchor="case-drilldown")
        case_ids = sorted({row["case_id"] for rep in reports.values() for row in rep.get("cases", [])} | set(ds_cases))
        if not case_ids:
            st.caption("no cases")
            return
        selected = st.selectbox(
            "Case", case_ids, key="results_case",
            format_func=lambda i: f"{i} — {case_input(ds_cases[i])[:70]}" if i in ds_cases else i,
        )
        case = ds_cases.get(selected) or {}
        if case:
            md = case.get("metadata") or {}
            st.markdown(f"intent `{md.get('intent')}` · scenario `{md.get('scenario')}` · failure mode `{md.get('failure_mode')}` · variant `{md.get('variant')}`")
            with st.expander("Reference outputs"):
                code_json(case.get("reference_outputs") or {})
        agg_row = next((r for r in agg.get("cases") or [] if r.get("id") == selected), None)
        if agg_row:
            st.markdown(
                ("✅ stable trajectory" if agg_row.get("stable") else ":red-badge[unstable trajectory]")
                + " · means: " + ", ".join(f"`{m}`={num(v)}" for m, v in (agg_row.get("means") or {}).items())
                + (f" · :blue-badge[{agg_row['llm_mock_calls']} LLM-mocked call(s)]" if agg_row.get("llm_mock_calls") else "")
            )
        run_tabs = st.tabs([f"run {r}" for r in bundle.run_ids] or ["runs"])
        for tab, run_id in zip(run_tabs, bundle.run_ids):
            with tab:
                rep = reports.get(run_id) or {}
                score_row = next((r for r in rep.get("cases", []) if r.get("case_id") == selected), None)
                run = runs.get(run_id) or {}
                cr = next((r for r in run.get("case_runs", []) if r.get("case_id") == selected), None)
                left, right = st.columns([1, 1])
                with left:
                    st.markdown("**Scores**")
                    if score_row:
                        for m, v in (score_row.get("scores") or {}).items():
                            ok = (v.get("score") or 0) >= 1
                            st.markdown(f"{':green-badge[' if ok else ':red-badge['}{m} {num(v.get('score'))}] {v.get('comment') or ''}")
                        for m, e in (score_row.get("errors") or {}).items():
                            st.markdown(f":orange-badge[{m} error] {e}")
                        for m, why in (score_row.get("skipped") or {}).items():
                            st.caption(f"{m} skipped: {why}")
                    else:
                        st.caption("not scored in this run")
                with right:
                    st.markdown("**Behavior**")
                    if cr:
                        st.markdown(f"tool calls: {tool_calls_inline(cr.get('tool_calls') or [])}")
                        st.markdown(f"node path: `{' → '.join(cr.get('node_path') or []) or '—'}`")
                        if cr.get("error"):
                            st.error(f"{cr.get('error_class')}: {cr['error']}")
                        st.markdown("**Response**")
                        st.markdown(str((cr.get("outputs") or {}).get("response", "")))
                    else:
                        st.caption("no run record for this case")
                if cr and cr.get("mock_calls"):
                    with st.expander(f"Mock calls ({len(cr['mock_calls'])}) — which layer answered each tool call"):
                        table([
                            {"tool": m.get("tool"), "layer": m.get("layer"), "strategy": m.get("strategy"), "rule": m.get("rule"),
                             "valid": m.get("valid"), "repairs": m.get("repairs"), "fallback": m.get("fallback"),
                             "args": json.dumps(m.get("args"), ensure_ascii=False)[:120],
                             "response": json.dumps(m.get("response"), ensure_ascii=False, default=str)[:160],
                             "error": m.get("error") or ", ".join(m.get("problems") or [])[:120]}
                            for m in cr["mock_calls"]
                        ])
                if cr and cr.get("trajectory"):
                    with st.expander("Trajectory transcript"):
                        transcript(cr["trajectory"], key=f"{run_id}-{selected}")
