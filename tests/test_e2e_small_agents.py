"""End-to-end pipeline runs on the smallest example agents with tiny datasets — every
stage offline (scripted agent + offline generator), through the same entry points the
UI jobs and the CLI use: a full autonomous run, the stop-after-dataset review loop
(feedback → regenerate → approve → resume) and the CLI job wrapper."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from evalbuilder.cli import app
from evalbuilder.config import Settings
from evalbuilder.pipeline import jobs
from evalbuilder.pipeline import setup as setup_mod
from evalbuilder.pipeline.config import load_config
from evalbuilder.pipeline.report import run_pipeline

WEATHER_AGENT = "scripted:examples.weather_bot.agent:default_scripted_model"
WEATHER_GENERATOR = "scripted:examples.weather_bot.offline:generator_model"


def _weather_config(tmp_path: Path, **overrides) -> Path:
    form = setup_mod.default_form({"name": "weather-small", "source": "examples/weather_bot/agent.py", "module": "examples.weather_bot.agent"})
    form.update(agent_model=WEATHER_AGENT, judge_model=WEATHER_AGENT, generator_model=WEATHER_GENERATOR,
                evaluators=["expected_tools", "contains"], total_cases=4, happy=1, failure=1, per_failure_category=1, out_of_intent=1,
                multi_turn_share=0.2, per_tool_edge_cases=1, repeats=2, simulate=True, auto_approve=True, approved_by="tester",
                constraints="Always name the city in the answer.", instructions="Tiny smoke dataset.",
                output_dir=str(tmp_path / "out"), config_path=str(tmp_path / "weather.yaml"))
    form.update(overrides)
    cfg = setup_mod.build_config(form)
    cfg.stages.publish = "never"
    return cfg.save(Path(form["config_path"]))


def test_weather_bot_full_offline_run(tmp_path):
    cfg_path = _weather_config(tmp_path)
    state, report = run_pipeline(cfg_path, settings=Settings())
    statuses = {k: v["status"] for k, v in report["stages"].items()}
    assert statuses["report"] == "ok" and statuses["publish"] == "skipped"
    assert all(s == "ok" for k, s in statuses.items() if k != "publish"), statuses
    assert report["verdict"] == "pass" and report["overall_score"] == 1.0, report["verdict_reasons"]
    assert report["agent"]["tools"] == ["get_weather", "get_alerts"]
    assert report["coverage"]["coverage_pct"] == 100.0
    by_kind = report["coverage"]["by_kind"]
    assert by_kind["schema-edge"]["covered"] == 2 and by_kind["out-of-intent"]["covered"] >= 1
    assert report["coverage"]["multi_turn"]["have"] == report["coverage"]["multi_turn"]["planned"] >= 1
    assert set(report["metrics"]) == {"expected_tools", "contains"} and all(v["passed"] for v in report["metrics"].values())
    assert report["stability"]["repeats"] == 2 and report["stability"]["unstable_cases"] == []
    assert report["simulation"]["details"]["stop_reasons"] == {"weather-then-alerts": "success"}
    assert report["stages"]["review"]["details"]["approved_by"] == "tester"
    ds = json.loads((tmp_path / "out" / "dataset.json").read_text())
    assert 4 <= len(ds["cases"]) <= 9 and all(c["review"]["status"] == "approved" for c in ds["cases"])
    edges = [c for c in ds["cases"] if c["metadata"].get("edge")]
    assert {c["metadata"]["edge"]["kind"] for c in edges} == {"missing_required"} and {c["metadata"]["tool"] for c in edges} == {"get_weather", "get_alerts"}
    multi = [c for c in ds["cases"] if c["metadata"].get("user_turns")]
    assert multi and multi[0]["metadata"]["variant"] == "multi-turn" and multi[0]["reference_outputs"]["expected_tools"][-1]["name"] == "get_alerts"
    error_case = next(c for c in ds["cases"] if c["metadata"]["failure_mode"] == "tool_error_handling")
    assert next(iter(error_case["metadata"]["mocks"]["tools"].values()))[0]["response"] == {"error": "service unavailable"}
    result = CliRunner().invoke(app, ["pipeline", "report", str(tmp_path / "out")])
    assert result.exit_code == 0 and "verdict=pass" in result.stdout


def test_weather_bot_review_loop_until_dataset_feedback_regenerate_approve(tmp_path):
    cfg_path = _weather_config(tmp_path, auto_approve=False, approved_by="")
    out = tmp_path / "out"
    _, report = run_pipeline(cfg_path, until="dataset", settings=Settings())
    stages = report["stages"]
    assert stages["dataset"]["status"] == "ok" and stages["review"]["status"] == "skipped" and stages["run"]["status"] == "skipped"
    assert json.loads((out / "state.json").read_text())["data"]["stopped_after"] == "dataset"
    ds = json.loads((out / "dataset.json").read_text())
    assert ds["cases"] and all(c["review"]["status"] == "pending" for c in ds["cases"])
    summary = setup_mod.dataset_summary(out)
    assert summary["by_status"] == {"pending": len(ds["cases"])} and summary["mocked_tools"] == ["get_alerts", "get_weather"]
    assert len(summary["schema_edge_cases"]) == 2

    # reviewer feedback → regenerate the dataset only (map/mocks cached)
    setup_mod.add_feedback(cfg_path, "Use Lyon instead of Paris in one case.", "dataset")
    assert load_config(cfg_path).feedback[0].from_stage == "dataset"
    log: list[str] = []
    _, report2 = run_pipeline(cfg_path, resume=True, invalidate_from="dataset", until="dataset", log=log.append, settings=Settings())
    assert report2["stages"]["map"]["status"] == "ok" and report2["stages"]["dataset"]["status"] == "ok"
    assert any("map: cached" in line for line in log) and not any("map: cached" not in line and "generator: agent_map" in line for line in log)
    assert "REVIEWER FEEDBACK" in load_config(cfg_path).guidance()

    # reject one case, approve the rest, resume to the end
    first = ds["cases"][0]["id"]
    assert setup_mod.reject_cases(out, [first], "not needed", "tester") == 1
    setup_mod.approve_in_config(cfg_path, "tester", note="approved in the test")
    _, report3 = run_pipeline(cfg_path, resume=True, settings=Settings())
    assert report3["verdict"] == "pass", report3["verdict_reasons"]
    assert report3["stages"]["review"]["details"]["approved_by"] == "tester"
    final = json.loads((out / "dataset.json").read_text())
    statuses = {c["id"]: c["review"]["status"] for c in final["cases"]}
    assert statuses[first] == "rejected" and set(statuses.values()) == {"approved", "rejected"}
    assert len(report3["cases"]) == len(final["cases"]) - 1


def _wait_job(out: Path, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = jobs.job_status(out)
        if status["status"] == "finished":
            return status
        time.sleep(0.2)
    raise AssertionError(f"job did not finish: {jobs.job_status(out)}\n{jobs.tail_log(out)}")


@pytest.mark.slow
def test_weather_bot_background_jobs_dataset_then_resume(tmp_path):
    """The UI's job modes as real subprocesses (`python -m evalbuilder.cli pipeline run …`)."""
    cfg_path = _weather_config(tmp_path, auto_approve=False, approved_by="", repeats=1, simulate=False)
    out = tmp_path / "out"
    job = jobs.start_job(cfg_path, out, mode="dataset")
    status = _wait_job(out)
    assert status["exit_code"] == 0, jobs.tail_log(out)
    assert status["stopped_after"] == "dataset" and status["stages"]["run"]["status"] == "skipped"
    assert job["until"] == "dataset" and "--until dataset" in jobs.tail_log(out)
    setup_mod.approve_in_config(cfg_path, "job tester")
    jobs.start_job(cfg_path, out, mode="resume")
    status = _wait_job(out)
    assert status["exit_code"] == 0, jobs.tail_log(out)
    report = json.loads((out / "report.json").read_text())
    assert report["verdict"] == "pass" and report["stages"]["simulate"]["status"] == "skipped"
    assert yaml.safe_load(cfg_path.read_text())["review"]["approved_by"] == "job tester"
