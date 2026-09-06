"""The phase commands: `evalbuilder infer` (in-process, against an endpoint, with
scenarios and repeats), the `run` alias, and `evalbuilder eval` (score + aggregate)."""

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from evalbuilder.agent_client import LocalAgent
from evalbuilder.artifacts import add_case, save_json, set_review
from evalbuilder.cli import app
from evalbuilder.schemas import Dataset, Target
from evalbuilder.serve import make_server

WEATHER = "examples.weather_bot.agent"
SCENARIO = {"id": "weather-then-alerts", "opening": "What is the weather in Paris?", "followups": ["Any alerts for Paris?"],
            "max_turns": 3, "success_contains": "heat advisory"}


def _ok(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _dataset(tmp_path: Path, n: int = 3) -> Path:
    ds = Dataset(name="w", dataset_type="final_response", target=Target(module=WEATHER),
                 mocks={"tools": {"get_weather": [{"matchArgs": {}, "response": {"temp": 5, "condition": "mocked", "city": "Paris"}}],
                                  "get_alerts": [{"matchArgs": {}, "response": {"alerts": ["heat advisory"], "city": "Paris"}}]}, "on_miss": "strict"})
    for i in range(n):
        add_case(ds, {"inputs": {"messages": [{"role": "user", "content": f"What is the weather in Paris? ({i})"}]},
                      "reference_outputs": {"contains": "Paris", "expected_tools": [{"name": "get_weather", "args": {"city": "Paris"}}]},
                      "metadata": {"intent": f"intent.{i % 2}", "variant": "happy"}})
    set_review(ds, [c.id for c in ds.cases], "approved", "t")
    p = tmp_path / "ds.json"
    save_json(p, ds)
    return p


def _evaluators(tmp_path: Path) -> Path:
    p = tmp_path / "evaluators.yaml"
    p.write_text(yaml.safe_dump({"evaluators": [{"type": "expected_tools"}, {"type": "contains"}]}))
    return p


def test_infer_in_process_with_scenarios_and_repeats(tmp_path):
    ds_path = _dataset(tmp_path)
    scen = tmp_path / "scenarios.yaml"
    scen.write_text(yaml.safe_dump({"scenarios": [SCENARIO]}))
    out = tmp_path / "results"
    summary = _ok(CliRunner().invoke(app, ["infer", str(ds_path), "--scenarios", str(scen), "--repeats", "2", "--workers", "2", "--out", str(out)]))
    assert summary["mode"] == "local" and summary["endpoint"] is None and summary["repeats"] == 2 and summary["workers"] == 2
    assert len(summary["runs"]) == 2 and all(r["cases"] == 3 and r["errors"] == 0 for r in summary["runs"])
    assert summary["path"] == summary["runs"][0]["path"] and summary["run_id"] == summary["runs"][0]["run_id"]  # `run` contract kept
    assert summary["execution"]["backend"] == "threads" and summary["mocking"]["on_miss"] == "strict"
    assert summary["simulation"]["scenarios"] == 1 and summary["simulation"]["stop_reasons"] == {"weather-then-alerts": "success"}
    assert summary["simulation"]["mined"] == 0 and summary["simulation"]["mock_calls"]["weather-then-alerts"]["rule"] == 2
    assert len(list(out.glob("run-*.json"))) == 2 and len(list(out.glob("simulation-*.json"))) == 1
    run = json.loads(Path(summary["path"]).read_text())
    assert run["execution"]["mode"] == "local" and run["case_runs"][0]["log"][0]["turn"] == 0


def test_run_alias_and_bad_backend(tmp_path):
    ds_path = _dataset(tmp_path, n=1)
    summary = _ok(CliRunner().invoke(app, ["run", str(ds_path), "--mock", "--out", str(tmp_path)]))
    assert summary["cases"] == 1 and summary["errors"] == 0 and Path(summary["path"]).exists()
    result = CliRunner().invoke(app, ["infer", str(ds_path), "--backend", "gpu", "--out", str(tmp_path)])
    assert result.exit_code == 1 and "--backend" in result.output


def test_infer_against_an_endpoint(tmp_path):
    ds_path = _dataset(tmp_path, n=2)
    srv = make_server(LocalAgent(WEATHER), "127.0.0.1", 0)
    srv.serve_in_thread()
    try:
        summary = _ok(CliRunner().invoke(app, ["infer", str(ds_path), "--endpoint", srv.endpoint, "--out", str(tmp_path), "--model", "scripted:ignored:x"]))
    finally:
        srv.shutdown()
        srv.server_close()
    assert summary["mode"] == "remote" and summary["endpoint"] == srv.endpoint and summary["errors"] == 0
    dead = CliRunner().invoke(app, ["infer", str(ds_path), "--endpoint", "http://127.0.0.1:9", "--out", str(tmp_path), "--model", "x:y"])
    assert dead.exit_code == 1 and "not healthy" in dead.output and "ignored against a deployed agent" in dead.output


def test_eval_scores_runs_and_aggregates(tmp_path):
    ds_path = _dataset(tmp_path)
    out = tmp_path / "results"
    summary = _ok(CliRunner().invoke(app, ["infer", str(ds_path), "--repeats", "2", "--out", str(out)]))
    run_paths = [r["path"] for r in summary["runs"]]
    ev = _evaluators(tmp_path)
    result = _ok(CliRunner().invoke(app, ["eval", *run_paths, "--dataset", str(ds_path), "--evaluators", str(ev), "--out", str(out), "--workers", "2"]))
    assert [r["run_id"] for r in result["reports"]] == [r["run_id"] for r in summary["runs"]]
    assert all(r["metrics"] == {"expected_tools": 1.0, "contains": 1.0} for r in result["reports"])
    assert result["aggregate"]["verdict"] == "pass" and result["aggregate"]["overall_score"] == 1.0 and result["aggregate"]["unstable_cases"] == 0
    agg = json.loads(Path(result["aggregate"]["path"]).read_text())
    assert agg["schema"] == "evalbuilder/aggregate/v1" and agg["repeats"] == 2
    assert len(list(out.glob("score-report-*.json"))) == 2
    single = _ok(CliRunner().invoke(app, ["eval", run_paths[0], "--dataset", str(ds_path), "--evaluators", str(ev), "--out", str(tmp_path / "one")]))
    assert single["aggregate"] is None and len(single["reports"]) == 1
    forced = _ok(CliRunner().invoke(app, ["eval", run_paths[0], "--dataset", str(ds_path), "--evaluators", str(ev), "--out", str(tmp_path / "one"), "--aggregate"]))
    assert forced["aggregate"]["verdict"] == "pass"


def test_eval_uses_the_config_thresholds(tmp_path):
    ds_path = _dataset(tmp_path)
    out = tmp_path / "results"
    summary = _ok(CliRunner().invoke(app, ["infer", str(ds_path), "--out", str(out)]))
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(yaml.safe_dump({"schema": "evalbuilder/pipeline-config/v1", "name": "w",
                                   "target": {"source": "examples/weather_bot/agent.py", "module": WEATHER},
                                   "thresholds": {"default": 1.0, "metrics": {"contains": 1.1}, "overall_pass": 1.0}}))
    result = _ok(CliRunner().invoke(app, ["eval", summary["path"], "--dataset", str(ds_path), "--evaluators", str(_evaluators(tmp_path)),
                                          "--out", str(out), "--aggregate", "--config", str(cfg)]))
    assert result["aggregate"]["verdict"] == "fail"  # contains can never reach 1.1
