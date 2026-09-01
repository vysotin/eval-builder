import json

import pytest
from typer.testing import CliRunner

from evalbuilder.artifacts import (
    add_case,
    import_cases,
    load_dataset,
    normalize_case,
    save_json,
    set_review,
    validate_dataset,
)
from evalbuilder.cli import app
from evalbuilder.schemas import Dataset, Target


def _ds():
    return Dataset(
        name="d",
        dataset_type="final_response",
        target=Target(module="examples.weather_bot.agent"),
    )


def test_normalize_forces_pending_and_hash_id():
    raw = {
        "inputs": {"messages": [{"role": "user", "content": "hi"}]},
        "metadata": {"intent": "intent.x"},
        "review": {"status": "approved", "note": "sneaky"},
    }
    case = normalize_case(raw)
    assert case.review.status == "pending"
    assert case.id.startswith("case-") and len(case.id) == 15
    assert case.metadata["failure_mode"] == "none"
    assert case.metadata["topic"] == "unspecified"


def test_same_inputs_different_cell_is_different_case():
    a = normalize_case({"inputs": {"q": 1}, "metadata": {"intent": "a"}})
    b = normalize_case({"inputs": {"q": 1}, "metadata": {"intent": "b"}})
    assert a.id != b.id


def test_add_case_rejects_duplicates():
    ds = _ds()
    add_case(ds, {"inputs": {"q": 1}})
    with pytest.raises(ValueError, match="duplicate"):
        add_case(ds, {"inputs": {"q": 1}})


def test_validate_reports_errors():
    ds = _ds()
    add_case(ds, {"inputs": {"q": 1}})
    ds.cases[0].review.status = "bogus"
    errs = validate_dataset(ds)
    assert any("review" in e for e in errs)


def test_validate_checks_mock_rule_shape():
    ds = _ds()
    c = add_case(ds, {"inputs": {"q": 1}})
    c.metadata["mocks"] = {"tools": {"t": [{"matchArgs": {}}]}}  # no response
    errs = validate_dataset(ds)
    assert any("response" in e for e in errs)


def test_roundtrip_serializes_schema_alias(tmp_path):
    ds = _ds()
    add_case(ds, {"inputs": {"q": 1}})
    p = tmp_path / "ds.json"
    save_json(p, ds)
    data = json.loads(p.read_text())
    assert data["schema"] == "evalbuilder/dataset/v1"
    ds2 = load_dataset(p)
    assert ds2.cases[0].id == ds.cases[0].id


def test_import_maps_foreign_fields():
    ds = _ds()
    n = import_cases(
        ds,
        {
            "examples": [
                {
                    "inputs": {"q": 2},
                    "outputs": {"a": 3},
                    "additional_metadata": {"intent": "i"},
                }
            ]
        },
        "f.json",
    )
    assert n == 1
    c = ds.cases[0]
    assert c.reference_outputs == {"a": 3}
    assert c.metadata["source"] == "import" and c.metadata["intent"] == "i"
    assert c.review.status == "pending"


def test_set_review():
    ds = _ds()
    c = add_case(ds, {"inputs": {"q": 1}})
    assert set_review(ds, [c.id], "approved", "ok by user") == 1
    assert ds.cases[0].review.status == "approved"


def test_set_review_unknown_id_raises():
    ds = _ds()
    with pytest.raises(ValueError, match="unknown"):
        set_review(ds, ["case-nope"], "approved", "")


def test_cli_dataset_flow(tmp_path):
    runner = CliRunner()
    ds_path = tmp_path / "ds.json"
    r = runner.invoke(
        app,
        ["dataset", "init", str(ds_path), "--name", "demo", "--type", "final_response",
         "--target", "examples.weather_bot.agent:build_agent"],
    )
    assert r.exit_code == 0, r.output

    case_file = tmp_path / "case.json"
    case_file.write_text(json.dumps({"inputs": {"q": 1}, "metadata": {"intent": "i.a"}}))
    r = runner.invoke(app, ["dataset", "add", str(ds_path), "--case", f"@{case_file}"])
    assert r.exit_code == 0, r.output
    case_id = json.loads(r.stdout)["id"]

    r = runner.invoke(app, ["dataset", "validate", str(ds_path)])
    assert r.exit_code == 0, r.output

    r = runner.invoke(app, ["review", str(ds_path), "--approve", case_id, "--note", "user said yes"])
    assert r.exit_code == 0, r.output

    r = runner.invoke(app, ["dataset", "list", str(ds_path), "--status", "approved"])
    assert r.exit_code == 0
    assert case_id in r.stdout

    r = runner.invoke(app, ["dataset", "list", str(ds_path), "--status", "pending"])
    assert case_id not in r.stdout


def test_cli_dataset_add_refuses_invalid(tmp_path):
    runner = CliRunner()
    ds_path = tmp_path / "ds.json"
    runner.invoke(
        app,
        ["dataset", "init", str(ds_path), "--name", "demo", "--type", "final_response",
         "--target", "m"],
    )
    r = runner.invoke(app, ["dataset", "add", str(ds_path), "--case", json.dumps({"inputs": {}})])
    assert r.exit_code == 1
    assert "inputs" in r.output
    assert json.loads(ds_path.read_text())["cases"] == []


def test_validate_checks_the_llm_mock_policy_and_strategy_references():
    ds = _ds()
    c = add_case(ds, {"inputs": {"q": 1}})
    ds.mocks = {"tools": {}, "on_miss": "guess"}
    assert any("on_miss" in e for e in validate_dataset(ds))
    ds.mocks = {"tools": {}, "on_miss": "llm"}
    assert any("mocks.llm.model" in e for e in validate_dataset(ds))
    ds.mocks = {"tools": {}, "on_miss": "llm", "llm": {"model": "nocolon"}}
    assert any("provider:model" in e for e in validate_dataset(ds))
    ds.mocks = {
        "tools": {}, "on_miss": "llm", "strategy": "default",
        "llm": {"model": "scripted:x:y", "on_invalid": "fallback", "max_repairs": 1},
        "strategies": {"world": "w", "strategies": {"default": {"description": "d", "tools": {}}}},
    }
    assert validate_dataset(ds) == []
    c.metadata["mocks"] = {"strategy": "degraded"}
    assert any("unknown mock strategy 'degraded'" in e for e in validate_dataset(ds))
    ds.mocks["strategy"] = "nope"
    assert any("dataset: unknown mock strategy 'nope'" in e for e in validate_dataset(ds))
    ds.mocks["llm"]["on_invalid"] = "shrug"
    assert any("on_invalid" in e for e in validate_dataset(ds))
