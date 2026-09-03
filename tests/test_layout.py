"""Artifact naming convention: unique names/schemas, identification, wrap/unwrap."""

import json

import pytest

from evalbuilder.pipeline.layout import (
    ARTIFACTS,
    FOLDED,
    WORK_DIR,
    artifact_index,
    compact_dir,
    convention_table,
    existing_path,
    fold_from,
    identify,
    kind_of_file,
    kind_of_schema,
    kinds_of_tier,
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


def test_every_kind_declares_a_known_tier():
    assert {a.tier for a in ARTIFACTS.values()} <= {"final", "work", "derived"}
    for a in ARTIFACTS.values():
        if a.tier == "derived":
            assert a.folded_into, a.kind  # derived == it lives inside another artifact
        if a.tier == "work":
            assert a.file.startswith(f"{WORK_DIR}/"), a.kind
        if a.tier == "final":
            assert not a.file.startswith(f"{WORK_DIR}/"), a.kind
            assert not a.folded_into, a.kind  # a deliverable is never a copy of another one


def test_work_tier_holds_only_scratch_and_final_tier_only_deliverables():
    assert set(kinds_of_tier("work")) == {
        "mock_rules", "mock_strategies", "run_progress", "pipeline_state", "pipeline_job",
    }
    assert set(kinds_of_tier("derived")) == {"applicable_failures", "coverage_plan", "coverage"}
    # every non-final kind that carries evaluation data names the deliverable holding it,
    # so nothing in work/ is data you would lose by deleting the directory
    assert FOLDED == {
        "applicable_failures": "agent_map.applicable_failures",
        "mock_rules": "dataset.mocks.tools",
        "mock_strategies": "dataset.mocks.strategies",
        "coverage_plan": "dataset.coverage.plan",
        "coverage": "dataset.coverage.achieved",
    }
    for parent in {v.split(".")[0] for v in FOLDED.values()}:
        assert ARTIFACTS[parent].tier == "final"
    assert set(kinds_of_tier("work")) - set(FOLDED) == {"run_progress", "pipeline_state", "pipeline_job"}


def test_derived_kinds_have_no_path_of_their_own(tmp_path):
    for kind in kinds_of_tier("derived"):
        with pytest.raises(ValueError, match="derived"):
            path_for(tmp_path, kind)


def test_fold_from_reads_a_derived_kind_out_of_its_parent():
    parents = {
        "agent_map": {"applicable_failures": {"out_of_scope": ["app:always"]}},
        "dataset": {"coverage": {"plan": {"cells": [1]}, "achieved": {"planned": 3}}},
    }
    assert fold_from("applicable_failures", parents) == {"out_of_scope": ["app:always"]}
    assert fold_from("coverage_plan", parents) == {"cells": [1]}
    assert fold_from("coverage", parents) == {"planned": 3}
    assert fold_from("coverage", {}) is None  # parent absent
    assert fold_from("coverage", {"dataset": {}}) is None  # parent carries nothing yet
    assert fold_from("dataset", parents) is None  # not a derived kind


def test_existing_path_prefers_the_current_name_then_legacy(tmp_path):
    assert existing_path(tmp_path, "pipeline_state") is None
    (tmp_path / "state.json").write_text("{}")
    assert existing_path(tmp_path, "pipeline_state").name == "state.json"  # pre-move dir
    (tmp_path / WORK_DIR).mkdir()
    (tmp_path / WORK_DIR / "state.json").write_text("{}")
    assert existing_path(tmp_path, "pipeline_state").parent.name == WORK_DIR
    assert existing_path(tmp_path, "score_report", "r1") is None


def test_path_for_and_kind_of_file(tmp_path):
    assert path_for(tmp_path, "run", "abc123").name == "run-abc123.json"
    assert path_for(tmp_path, "score_report", "abc123") == tmp_path / "results" / "score-report-abc123.json"
    assert kind_of_file("results/run-abc123.json") == (ARTIFACTS["run"], "abc123")
    assert kind_of_file("score-report-abc123.json") == (ARTIFACTS["score_report"], "abc123")
    assert kind_of_file("results/report-abc123.json") == (ARTIFACTS["score_report"], "abc123")  # legacy
    assert kind_of_file("/abs/dir/mocks.json") == (ARTIFACTS["mock_rules"], "")  # legacy
    assert kind_of_file("coverage-plan.json")[0].kind == "coverage_plan"  # folded, still resolves
    assert kind_of_file("report.json")[0].kind == "pipeline_report"
    assert kind_of_file("unknown.json") is None
    # work-tier kinds resolve under work/, by their bare name, and by their pre-move name
    assert path_for(tmp_path, "pipeline_state") == tmp_path / "work" / "state.json"
    assert kind_of_file("work/state.json")[0].kind == "pipeline_state"
    assert kind_of_file("state.json")[0].kind == "pipeline_state"
    assert kind_of_file("mock-rules.json")[0].kind == "mock_rules"


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


def test_convention_table_lists_every_kind_with_its_tier():
    rows = convention_table()
    assert {r["kind"] for r in rows} == set(ARTIFACTS)
    assert all(r["description"] and r["tier"] for r in rows)
    by_kind = {r["kind"]: r for r in rows}
    assert by_kind["coverage"] | {"file": "—", "folded_into": "dataset.coverage.achieved"} == by_kind["coverage"]
    assert by_kind["pipeline_state"]["file"] == "work/state.json"
    assert by_kind["mock_rules"]["folded_into"] == "dataset.mocks.tools"
    json.dumps(rows)


def test_mock_strategies_artifact_kind():
    a = ARTIFACTS["mock_strategies"]
    assert a.file == "work/mock-strategies.json" and a.schema == "evalbuilder/mock-strategies/v1"
    assert a.stage == "mocks" and a.tier == "work"
    payload = {"world": "w", "strategies": {"default": {"description": "d", "tools": {}}}}
    stamped = stamp("mock_strategies", payload)
    assert stamped["schema"] == a.schema and unwrap("mock_strategies", stamped) == {"schema": a.schema, **payload}
    assert kind_of_file("mock-strategies.json")[0].kind == "mock_strategies"
    assert kind_of_schema("evalbuilder/mock-strategies/v1").kind == "mock_strategies"


def test_compact_dir_folds_copies_moves_scratch_and_is_idempotent(tmp_path):
    """A directory in the pre-fold layout is brought onto the current one, losslessly."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "agent-map.json").write_text(json.dumps({"schema": AGENT_MAP_SCHEMA, "tools": []}))
    (out / "applicable-failures.json").write_text(json.dumps(
        {"schema": "evalbuilder/applicable-failures/v1", "failure_types": {"out_of_scope": ["app:always"]}}))
    (out / "dataset.json").write_text(json.dumps(
        {"schema": DATASET_SCHEMA, "name": "d", "mocks": {"tools": {"t": []}}, "cases": []}))
    (out / "plan.json").write_text(json.dumps({"cells": [{"intent": "i"}], "summary": {"cells": 1}}))  # legacy name
    (out / "coverage.json").write_text(json.dumps({"schema": "evalbuilder/coverage/v1", "planned": 1, "covered": 1}))
    (out / "mocks.json").write_text(json.dumps({"t": []}))  # legacy name, unstamped
    (out / "state.json").write_text(json.dumps({"schema": "evalbuilder/pipeline-state/v1", "name": "d", "stages": {}}))
    (out / "report.json").write_text(json.dumps(
        {"schema": "evalbuilder/pipeline-report/v1", "output_dir": "eval/pipeline/d", "artifacts": {}}))

    result = compact_dir(out)
    assert sorted(p.name for p in out.iterdir()) == [
        "agent-map.json", "dataset.json", "report.json", "work",
    ]
    assert sorted(p.name for p in (out / WORK_DIR).iterdir()) == ["mock-rules.json", "state.json"]
    assert len(result["folded"]) == 3 and len(result["removed"]) == 3 and len(result["moved"]) == 2

    amap = json.loads((out / "agent-map.json").read_text())
    ds = json.loads((out / "dataset.json").read_text())
    assert amap["applicable_failures"] == {"out_of_scope": ["app:always"]}
    assert ds["coverage"] == {"plan": {"cells": [{"intent": "i"}], "summary": {"cells": 1}},
                              "achieved": {"planned": 1, "covered": 1}}  # the schema stamp is dropped
    assert ds["mocks"]["tools"] == {"t": []}  # already folded, so the copy was moved, not re-read
    # the report's paths are provenance: re-indexed, but still under the dir the run wrote
    index = json.loads((out / "report.json").read_text())["artifacts"]
    assert set(index) >= {"agent_map", "dataset", "mock_rules", "pipeline_state"}
    assert index["mock_rules"] == "eval/pipeline/d/work/mock-rules.json"
    assert index["dataset"] == "eval/pipeline/d/dataset.json"
    assert "coverage" not in index and "applicable_failures" not in index

    assert compact_dir(out) == {"folded": [], "moved": [], "removed": []}


def test_compact_dir_dry_run_changes_nothing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "agent-map.json").write_text(json.dumps({"schema": AGENT_MAP_SCHEMA}))
    (out / "applicable-failures.json").write_text(json.dumps({"failure_types": {"a": ["b"]}}))
    before = {p.name: p.read_text() for p in out.iterdir()}
    assert compact_dir(out, dry_run=True)["folded"] == ["applicable-failures.json → agent_map.applicable_failures"]
    assert {p.name: p.read_text() for p in out.iterdir()} == before


def test_compact_dir_leaves_a_copy_alone_when_its_parent_is_missing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "coverage.json").write_text(json.dumps({"planned": 1}))
    assert compact_dir(out) == {"folded": [], "moved": [], "removed": []}
    assert (out / "coverage.json").exists()
