import json

from typer.testing import CliRunner

from evalbuilder.artifacts import add_case, save_json
from evalbuilder.cli import app
from evalbuilder.coverage import coverage_gaps, required_cells
from evalbuilder.schemas import AgentMap, Dataset, Target


def _map():
    return AgentMap(
        intents=[{"id": "intent.a", "status": "hypothesis", "evidence": ["prompt:n1"]}],
        scenarios=[
            {
                "id": "scenario.a.happy",
                "intent": "intent.a",
                "status": "hypothesis",
                "evidence": ["prompt:n1"],
            }
        ],
        failure_scenarios=[
            {
                "failure_type": "input_validation",
                "rationale": "r",
                "evidence": ["app:always"],
            }
        ],
        data_domains={"topics": ["billing", "shipping"], "sources": []},
    )


def test_required_cells_expand_topics_and_failures():
    cells = required_cells(_map())
    assert {
        "intent": "intent.a",
        "topic": "billing",
        "scenario": "scenario.a.happy",
        "failure_mode": "none",
    } in cells
    assert {
        "intent": "cross-cutting",
        "topic": "unspecified",
        "scenario": "failure",
        "failure_mode": "input_validation",
    } in cells
    assert len(cells) == 3  # 2 topics x 1 scenario + 1 failure


def test_gaps_count_missing():
    ds = Dataset(name="d", dataset_type="final_response", target=Target(module="m"))
    add_case(
        ds,
        {
            "inputs": {"q": 1},
            "metadata": {
                "intent": "intent.a",
                "topic": "billing",
                "scenario": "scenario.a.happy",
            },
        },
    )
    report = coverage_gaps(ds, _map(), target_per_cell=1)
    assert report["required_cells"] == 3 and report["covered_cells"] == 1
    assert len(report["gaps"]) == 2 and report["gaps"][0]["missing"] == 1


def test_cli_agent_map_update_and_gaps(tmp_path):
    runner = CliRunner()
    map_path = tmp_path / "agent-map.json"
    save_json(map_path, AgentMap(data_domains={"topics": ["billing"], "sources": []}))

    intents = tmp_path / "intents.json"
    intents.write_text(
        json.dumps([{"id": "intent.a", "status": "hypothesis", "evidence": ["prompt:n1"]}])
    )
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        json.dumps(
            [
                {
                    "id": "scenario.a.happy",
                    "intent": "intent.a",
                    "status": "hypothesis",
                    "evidence": ["prompt:n1"],
                }
            ]
        )
    )
    r = runner.invoke(
        app,
        ["agent-map", "update", str(map_path), "--intents", f"@{intents}",
         "--scenarios", f"@{scenarios}"],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(map_path.read_text())
    assert data["intents"][0]["id"] == "intent.a"

    ds_path = tmp_path / "ds.json"
    runner.invoke(
        app,
        ["dataset", "init", str(ds_path), "--name", "d", "--type", "final_response",
         "--target", "m"],
    )
    r = runner.invoke(app, ["dataset", "gaps", str(ds_path), "--agent-map", str(map_path)])
    assert r.exit_code == 0, r.output
    report = json.loads(r.stdout)
    assert report["required_cells"] == 1 and report["gaps"]


def test_cli_agent_map_update_rejects_missing_evidence(tmp_path):
    runner = CliRunner()
    map_path = tmp_path / "agent-map.json"
    save_json(map_path, AgentMap())
    r = runner.invoke(
        app,
        ["agent-map", "update", str(map_path), "--intents",
         json.dumps([{"id": "intent.a"}])],
    )
    assert r.exit_code == 1
    assert "evidence" in r.output
