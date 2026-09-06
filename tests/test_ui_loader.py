"""Report-UI loader: directory and upload loading, legacy names, discovery — no Streamlit."""

import json
import shutil
from pathlib import Path

import pytest

from evalbuilder.pipeline.layout import ARTIFACTS
from evalbuilder.ui import loader

EXAMPLE = Path("docs/examples/support-bot")


@pytest.fixture(scope="module")
def example_bundle() -> loader.Bundle:
    return loader.load_dir(EXAMPLE)


def test_load_dir_reads_every_artifact_kind(example_bundle):
    b = example_bundle
    assert b.problems == []
    assert b.name == "support-bot" and b.verdict == "fail"
    present = {k for k in ARTIFACTS if b.has(k)}
    transient = {"pipeline_job", "run_progress", "mock_strategies", "deployment"}  # job-time files, and kinds newer than the committed example
    assert present == set(ARTIFACTS) - transient, f"missing kinds: {set(ARTIFACTS) - present}"
    assert set(b.run_ids) == set(b.runs) == set(b.score_reports) and len(b.run_ids) == 2
    assert b.run_ids == [e["run_id"] for e in b.state["stages"]["run"]["artifacts"]["runs"]]
    # dict-shaped artifacts come back unwrapped
    assert set(b.get("mock_rules")) == {"lookup_order", "check_refund_policy", "issue_refund", "search_kb"}
    assert "schema" not in b.get("mock_rules") and "schema" not in b.get("applicable_failures")
    assert b.config["name"] == "support-bot"
    summary = b.summary()
    assert summary["cases"] == 21 and summary["tools"] == 4 and summary["runs"] == 2


def _unfold(example: Path, dest: Path) -> dict:
    """Rewrite a current output dir the way older versions wrote it: stand-alone copies
    of the folded artifacts at the root, scratch beside them, pre-convention names."""
    shutil.copytree(example, dest)
    amap = json.loads((dest / "agent-map.json").read_text())
    ds = json.loads((dest / "dataset.json").read_text())
    (dest / "applicable-failures.json").write_text(json.dumps(
        {"schema": "evalbuilder/applicable-failures/v1", "failure_types": amap.pop("applicable_failures")}))
    plan = ds["coverage"].pop("plan")
    (dest / "plan.json").write_text(json.dumps(plan))  # pre-convention name
    (dest / "coverage.json").write_text(json.dumps(
        {"schema": "evalbuilder/coverage/v1", **ds["coverage"].pop("achieved")}))
    (dest / "agent-map.json").write_text(json.dumps(amap))
    (dest / "dataset.json").write_text(json.dumps(ds))
    rules = json.loads((dest / "work" / "mock-rules.json").read_text())["tools"]
    (dest / "mocks.json").write_text(json.dumps(rules))  # pre-convention name, unwrapped, unstamped
    (dest / "work" / "state.json").rename(dest / "state.json")
    shutil.rmtree(dest / "work")
    for p in (dest / "results").glob("score-report-*.json"):
        data = json.loads(p.read_text())
        data["schema"] = "evalbuilder/report/v1"
        p.with_name(p.name.replace("score-report-", "report-")).write_text(json.dumps(data))
        p.unlink()
    return {"rules": rules, "plan": plan}


def test_load_dir_accepts_legacy_file_names_and_shapes(tmp_path):
    """Output directories written before the naming convention — and before the fold —
    still open, and their stand-alone files are what the pages read."""
    legacy = tmp_path / "legacy"
    old = _unfold(EXAMPLE, legacy)
    b = loader.load_dir(legacy)
    assert b.problems == []
    assert set(b.get("mock_rules")) == set(old["rules"])
    assert b.get("coverage_plan")["summary"]["cells"] == 16
    assert b.get("coverage")["planned"] and b.get("applicable_failures")
    assert b.state["stages"]  # state.json still found beside the deliverables
    assert len(b.score_reports) == 2 and all(r["schema"] == "evalbuilder/report/v1" for r in b.score_reports.values())


def test_folded_kinds_are_derived_from_their_parent_when_there_is_no_file(example_bundle, tmp_path):
    """The bundle presents the same kinds either way, so pages never learn about the fold."""
    folded = ("applicable_failures", "coverage_plan", "coverage", "mock_rules")
    current = {k: example_bundle.get(k) for k in folded}
    assert all(v for v in current.values())
    assert "coverage.json" not in {Path(f).name for f in example_bundle.files.values() if isinstance(f, str)}

    legacy = tmp_path / "legacy"
    _unfold(EXAMPLE, legacy)
    assert {k: loader.load_dir(legacy).get(k) for k in folded} == current

    # and with work/ deleted the rules still come back — from dataset.mocks.tools
    trimmed = tmp_path / "trimmed"
    shutil.copytree(EXAMPLE, trimmed)
    shutil.rmtree(trimmed / "work")
    b = loader.load_dir(trimmed)
    assert b.problems == [] and {k: b.get(k) for k in folded} == current


def test_load_dir_reports_missing_and_broken_files(tmp_path):
    b = loader.load_dir(tmp_path / "nope")
    assert b.problems and "not a directory" in b.problems[0]
    (tmp_path / "aggregate.json").write_text("{not json")
    (tmp_path / "agent-map.json").write_text(json.dumps({"schema": "evalbuilder/agent-map/v1", "intents": []}))
    b = loader.load_dir(tmp_path)
    assert b.has("agent_map") and not b.has("aggregate")
    assert any("aggregate.json" in p for p in b.problems)


def test_load_files_identifies_by_schema_then_name():
    files = []
    for p in sorted(EXAMPLE.rglob("*")):
        if p.is_file():
            # strip the informative file names: identification must come from the schema id
            name = "upload-%d.json" % len(files) if p.suffix == ".json" else p.name
            files.append((name, p.read_bytes()))
    b = loader.load_files(files)
    assert b.problems == []
    assert b.name == "support-bot"
    assert {k for k in ARTIFACTS if b.has(k)} == set(ARTIFACTS) - {"pipeline_job", "run_progress", "mock_strategies", "deployment"}
    assert set(b.run_ids) == {"093d879e", "8a6166d8"}  # run ids recovered from the payloads
    assert set(b.get("mock_rules")) == {"lookup_order", "check_refund_policy", "issue_refund", "search_kb"}


def test_load_files_flags_unknown_and_broken_uploads():
    b = loader.load_files([("random.json", b'{"hello": 1}'), ("broken.json", b"{"), ("simulation.json", b'{"results": []}')])
    assert any("random.json" in p and "unrecognised" in p for p in b.problems)
    assert any("broken.json" in p for p in b.problems)
    assert b.get("simulation") == {"results": []}  # identified by file name


def test_discover_dirs_finds_example_outputs(tmp_path):
    assert str(EXAMPLE) in loader.discover_dirs()
    root = tmp_path / "eval" / "pipeline"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "a" / "report.json").write_text("{}")
    assert loader.discover_dirs(cwd=tmp_path) == [str(root / "a")]


def test_load_dir_reads_run_progress(tmp_path):
    (tmp_path / "run-progress.json").write_text(json.dumps({
        "schema": "evalbuilder/run-progress/v1", "repeat": 1, "repeats": 3,
        "cases_total": 4, "cases_done": 2, "overall_total": 12, "overall_done": 2,
        "by_intent": {"intent.a": {"done": 2, "total": 4, "errors": 0}}, "runs": [], "current": [],
    }))
    b = loader.load_dir(tmp_path)
    assert b.has("run_progress")
    assert b.get("run_progress")["cases_done"] == 2
