import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from evalbuilder.artifacts import add_case, load_dataset, save_json, set_review
from evalbuilder.cli import app
from evalbuilder.config import Settings
from evalbuilder.langsmith_io import publish_approved
from evalbuilder.schemas import Dataset, Target


class FakeClient:
    def __init__(self, drop_readback=False):
        self.drop_readback = drop_readback
        self.dataset = None
        self.rows = []

    def read_dataset(self, dataset_name):
        if self.dataset is None:
            raise LookupError("not found")
        return self.dataset

    def create_dataset(self, dataset_name, description=""):
        self.dataset = SimpleNamespace(id=uuid4(), name=dataset_name)
        return self.dataset

    def create_examples(self, dataset_id, inputs, outputs, metadata):
        for i, o, m in zip(inputs, outputs, metadata):
            self.rows.append(
                SimpleNamespace(id=uuid4(), inputs=i, outputs=o, metadata=m)
            )

    def list_examples(self, dataset_id):
        rows = self.rows[1:] if self.drop_readback and self.rows else self.rows
        yield from rows


def _settings():
    return Settings(langsmith_api_key="test-key")


def _ds_with_two_cases(tmp_path):
    ds = Dataset(name="d", dataset_type="final_response", target=Target(module="m"))
    a = add_case(ds, {"inputs": {"q": 1}, "reference_outputs": {"a": 1}})
    add_case(ds, {"inputs": {"q": 2}})
    set_review(ds, [a.id], "approved", "user approved")
    path = tmp_path / "ds.json"
    save_json(path, ds)
    return ds, path


def test_publish_uploads_approved_only_and_records_ids(tmp_path):
    ds, path = _ds_with_two_cases(tmp_path)
    fake = FakeClient()
    out = publish_approved(ds, path, _settings(), client=fake)
    assert out["uploaded"] == 1 and out["verified"] == 1
    assert len(fake.rows) == 1
    assert fake.rows[0].metadata["local_case_id"] == ds.cases[0].id
    approved = [c for c in ds.cases if c.review.status == "approved"][0]
    assert approved.publication.langsmith_example_id

    again = publish_approved(load_dataset(path), path, _settings(), client=fake)
    assert again["uploaded"] == 0 and again["verified"] == 1


def test_publish_raises_on_readback_mismatch(tmp_path):
    ds, path = _ds_with_two_cases(tmp_path)
    fake = FakeClient(drop_readback=True)
    with pytest.raises(RuntimeError, match="read-back"):
        publish_approved(ds, path, _settings(), client=fake)
    # local publication state untouched
    assert json.loads(path.read_text())["cases"][0]["publication"][
        "langsmith_example_id"
    ] is None


def test_publish_refuses_without_approved(tmp_path):
    ds = Dataset(name="d", dataset_type="final_response", target=Target(module="m"))
    add_case(ds, {"inputs": {"q": 1}})
    with pytest.raises(ValueError, match="approved"):
        publish_approved(ds, tmp_path / "ds.json", _settings(), client=FakeClient())


def test_cli_publish_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    ds, path = _ds_with_two_cases(tmp_path)
    r = CliRunner().invoke(app, ["publish", str(path)])
    assert r.exit_code == 1
    assert "LANGSMITH_API_KEY" in r.output
