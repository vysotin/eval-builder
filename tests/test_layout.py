"""Artifact naming convention: unique names/schemas, identification, wrap/unwrap."""

import json

from evalbuilder.pipeline.layout import (
    ARTIFACTS,
    artifact_index,
    convention_table,
    identify,
    kind_of_file,
    kind_of_schema,
    path_for,
    stamp,
    unwrap,
)
from evalbuilder.schemas import AGENT_MAP_SCHEMA, DATASET_SCHEMA, REPORT_SCHEMA, RUN_SCHEMA


def test_kinds_files_and_schemas_are_unique():
    files = [a.file for a in ARTIFACTS.values()]
    schemas = [a.schema for a in ARTIFACTS.values() if a.schema]
    assert len(files) == len(set(files))
    assert len(schemas) == len(set(schemas))
    for a in ARTIFACTS.values():
        if a.schema:
            assert a.schema == f"evalbuilder/{a.file.split('/')[-1].split('.')[0].replace('-{run_id}', '')}/v1" or a.kind in ("pipeline_state", "pipeline_report", "pipeline_job"), a.kind
        stem = a.file.split("/")[-1].replace("-{run_id}", "").rsplit(".", 1)[0]
        assert stem == a.kind.replace("_", "-") or a.kind in ("pipeline_state", "pipeline_report", "pipeline_job"), a.kind


def test_schema_ids_match_pydantic_models():
    assert ARTIFACTS["agent_map"].schema == AGENT_MAP_SCHEMA
    assert ARTIFACTS["dataset"].schema == DATASET_SCHEMA
    assert ARTIFACTS["run"].schema == RUN_SCHEMA
    assert ARTIFACTS["score_report"].schema == REPORT_SCHEMA
    assert kind_of_schema("evalbuilder/report/v1").kind == "score_report"  # legacy id still resolves


def test_path_for_and_kind_of_file(tmp_path):
    assert path_for(tmp_path, "run", "abc123").name == "run-abc123.json"
    assert path_for(tmp_path, "score_report", "abc123") == tmp_path / "results" / "score-report-abc123.json"
    assert kind_of_file("results/run-abc123.json") == (ARTIFACTS["run"], "abc123")
    assert kind_of_file("score-report-abc123.json") == (ARTIFACTS["score_report"], "abc123")
    assert kind_of_file("results/report-abc123.json") == (ARTIFACTS["score_report"], "abc123")  # legacy
    assert kind_of_file("/abs/dir/mocks.json") == (ARTIFACTS["mock_rules"], "")  # legacy
    assert kind_of_file("coverage-plan.json")[0].kind == "coverage_plan"
    assert kind_of_file("report.json")[0].kind == "pipeline_report"
    assert kind_of_file("unknown.json") is None


def test_identify_prefers_schema_over_name():
    data = {"schema": "evalbuilder/aggregate/v1", "verdict": "pass"}
    assert identify(data, "whatever.json")[0].kind == "aggregate"
    assert identify({"no": "schema"}, "simulation.json")[0].kind == "simulation"
    assert identify({"schema": "evalbuilder/run/v1", "run_id": "r1"})[1] == "r1"
    assert identify({"schema": "evalbuilder/report/v1"}, "results/report-r9.json") == (ARTIFACTS["score_report"], "r9")


def test_stamp_and_unwrap_roundtrip_and_legacy_shape():
    rules = {"lookup_order": [{"matchArgs": {}, "response": "x"}]}
    stamped = stamp("mock_rules", rules)
    assert stamped == {"schema": "evalbuilder/mock-rules/v1", "tools": rules}
    assert unwrap("mock_rules", stamped) == rules
    assert unwrap("mock_rules", rules) == rules  # pre-convention file
    agg = stamp("aggregate", {"verdict": "pass"})
    assert agg["schema"] == "evalbuilder/aggregate/v1" and unwrap("aggregate", agg) == agg
    applicable = {"out_of_scope": ["app:always"]}
    assert unwrap("applicable_failures", stamp("applicable_failures", applicable)) == applicable


def test_artifact_index_finds_current_and_legacy_files(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "agent-map.json").write_text("{}")
    (tmp_path / "mocks.json").write_text("{}")
    (tmp_path / "results" / "run-r1.json").write_text("{}")
    (tmp_path / "results" / "report-r1.json").write_text("{}")
    (tmp_path / "results" / "score-report-r2.json").write_text("{}")
    idx = artifact_index(tmp_path)
    assert idx["agent_map"].endswith("agent-map.json")
    assert idx["mock_rules"].endswith("mocks.json")
    assert set(idx["run"]) == {"r1"} and set(idx["score_report"]) == {"r1", "r2"}
    assert "dataset" not in idx


def test_convention_table_lists_every_kind():
    rows = convention_table()
    assert {r["kind"] for r in rows} == set(ARTIFACTS)
    assert all(r["description"] for r in rows)
    json.dumps(rows)
