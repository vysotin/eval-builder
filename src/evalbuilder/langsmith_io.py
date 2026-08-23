"""Optional LangSmith backend: idempotent dataset publication with read-back checks."""

from __future__ import annotations

from pathlib import Path

from evalbuilder.artifacts import save_json
from evalbuilder.config import Settings
from evalbuilder.schemas import Dataset


def _default_client(settings: Settings):
    from langsmith import Client

    kwargs = {"api_key": settings.langsmith_api_key}
    if settings.langsmith_endpoint:
        kwargs["api_url"] = settings.langsmith_endpoint
    return Client(**kwargs)


def publish_approved(
    ds: Dataset,
    dataset_path: Path,
    settings: Settings,
    client=None,
    dataset_name: str | None = None,
) -> dict:
    approved = [c for c in ds.cases if c.review.status == "approved"]
    if not approved:
        raise ValueError("dataset has no approved cases to publish")

    if client is None:
        if not settings.langsmith_api_key:
            raise ValueError("LANGSMITH_API_KEY is not configured")
        client = _default_client(settings)

    name = dataset_name or ds.langsmith.get("dataset_name") or ds.name
    try:
        remote = client.read_dataset(dataset_name=name)
    except Exception:  # noqa: BLE001 - missing dataset is the expected miss
        remote = client.create_dataset(
            dataset_name=name, description="Approved evalbuilder test cases"
        )
    dataset_id = str(remote.id)

    to_upload = [c for c in approved if not c.publication.langsmith_example_id]
    if to_upload:
        client.create_examples(
            dataset_id=dataset_id,
            inputs=[c.inputs for c in to_upload],
            outputs=[c.reference_outputs or None for c in to_upload],
            metadata=[{**c.metadata, "local_case_id": c.id} for c in to_upload],
        )

    # Read-back: bulk-create response ordering is not trusted; match by metadata.
    remote_by_case_id: dict[str, str] = {}
    for example in client.list_examples(dataset_id=dataset_id):
        metadata = getattr(example, "metadata", None) or {}
        local_id = metadata.get("local_case_id")
        if local_id:
            remote_by_case_id[local_id] = str(example.id)

    verified = 0
    missing: list[str] = []
    for case in approved:
        example_id = remote_by_case_id.get(case.id)
        if example_id is None:
            if case in to_upload or not case.publication.langsmith_example_id:
                missing.append(case.id)
            continue
        case.publication.langsmith_example_id = example_id
        verified += 1
    if missing:
        raise RuntimeError(
            f"LangSmith read-back is missing uploaded cases: {', '.join(missing)}; "
            "local publication state was not updated"
        )

    ds.langsmith["dataset_id"] = dataset_id
    ds.langsmith["dataset_name"] = name
    save_json(dataset_path, ds)
    return {
        "dataset_id": dataset_id,
        "dataset_name": name,
        "uploaded": len(to_upload),
        "verified": verified,
    }
