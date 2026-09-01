"""Load/save/validate/normalize dataset artifacts. The only writers of artifact JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evalbuilder.schemas import DATASET_SCHEMA, Case, Dataset

SELF_APPROVAL_NOTE = "New/generated/imported content can never grant itself approval."
REVIEW_STATUSES = {"pending", "approved", "rejected"}
ON_MISS_POLICIES = ("real", "fallback", "strict", "llm")
ON_INVALID_POLICIES = ("fallback", "strict")


def save_json(path: Path, obj) -> None:
    data = obj.model_dump(by_alias=True) if hasattr(obj, "model_dump") else obj
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_dataset(path: Path) -> Dataset:
    return Dataset.model_validate(json.loads(Path(path).read_text()))


def case_id(raw: dict) -> str:
    md = raw.get("metadata") or {}
    identity = {
        "inputs": raw.get("inputs"),
        "intent": md.get("intent", "unspecified"),
        "topic": md.get("topic", "unspecified"),
        "scenario": md.get("scenario", "unspecified"),
        "failure_mode": md.get("failure_mode", "none"),
    }
    material = json.dumps(identity, sort_keys=True, ensure_ascii=False)
    return "case-" + hashlib.sha256(material.encode()).hexdigest()[:10]


def normalize_case(raw: dict) -> Case:
    raw = dict(raw)
    md = dict(raw.get("metadata") or {})
    md.setdefault("intent", "unspecified")
    md.setdefault("topic", "unspecified")
    md.setdefault("scenario", "unspecified")
    md.setdefault("failure_mode", "none")
    md.setdefault("source", "synthetic")
    raw["metadata"] = md
    raw["id"] = raw.get("id") or case_id(raw)
    raw["review"] = {"status": "pending", "note": SELF_APPROVAL_NOTE}
    raw.setdefault("publication", {})
    return Case.model_validate(raw)


def add_case(ds: Dataset, raw: dict) -> Case:
    case = normalize_case(raw)
    if any(c.id == case.id for c in ds.cases):
        raise ValueError(f"duplicate case id {case.id}")
    ds.cases.append(case)
    return case


def strategy_ids(mocks) -> list[str]:
    """Strategy ids declared in a dataset's `mocks.strategies` block."""
    if not isinstance(mocks, dict):
        return []
    table = (mocks.get("strategies") or {}).get("strategies") if isinstance(mocks.get("strategies"), dict) else None
    return list(table) if isinstance(table, dict) else []


def _validate_mock_policy(errors: list[str], mocks: dict) -> None:
    """Dataset-level policy keys: on_miss, the llm block, the default strategy."""
    policy = mocks.get("on_miss", "real")
    if policy not in ON_MISS_POLICIES:
        errors.append(f"dataset: mocks.on_miss must be one of {ON_MISS_POLICIES}, got {policy!r}")
    llm = mocks.get("llm")
    if policy == "llm":
        model = (llm or {}).get("model") if isinstance(llm, dict) else None
        if not model:
            errors.append("dataset: mocks.llm.model is required when mocks.on_miss is llm")
        elif ":" not in str(model):
            errors.append(f"dataset: mocks.llm.model must look like provider:model, got {model!r}")
    if isinstance(llm, dict):
        if llm.get("on_invalid") not in (None, *ON_INVALID_POLICIES):
            errors.append(f"dataset: mocks.llm.on_invalid must be one of {ON_INVALID_POLICIES}")
        if "max_repairs" in llm and (not isinstance(llm["max_repairs"], int) or llm["max_repairs"] < 0):
            errors.append("dataset: mocks.llm.max_repairs must be an integer >= 0")
    ids = strategy_ids(mocks)
    chosen = mocks.get("strategy")
    if ids and chosen and chosen not in ids:
        errors.append(f"dataset: unknown mock strategy {chosen!r} (declared: {ids})")


def _validate_mock_block(errors: list[str], where: str, mocks, ids: list[str] | None = None) -> None:
    if not isinstance(mocks, dict):
        errors.append(f"{where}: mocks must be an object")
        return
    chosen = mocks.get("strategy")
    if chosen is not None and ids is not None and chosen not in ids:
        errors.append(f"{where}: unknown mock strategy {chosen!r} (declared: {ids})")
    tools = mocks.get("tools", {})
    if not isinstance(tools, dict):
        errors.append(f"{where}: mocks.tools must be an object")
        return
    for tool_name, rules in tools.items():
        if not isinstance(rules, list):
            errors.append(f"{where}: mocks.tools.{tool_name} must be a list of rules")
            continue
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict) or not isinstance(
                rule.get("matchArgs", {}), dict
            ):
                errors.append(
                    f"{where}: mocks.tools.{tool_name}[{i}] needs an object matchArgs"
                )
            elif "response" not in rule:
                errors.append(
                    f"{where}: mocks.tools.{tool_name}[{i}] needs a response key"
                )


def validate_dataset(ds: Dataset) -> list[str]:
    errors: list[str] = []
    if ds.schema_ != DATASET_SCHEMA:
        errors.append(f"schema must be {DATASET_SCHEMA}")
    seen: set[str] = set()
    _validate_mock_block(errors, "dataset", ds.mocks)
    if isinstance(ds.mocks, dict):
        _validate_mock_policy(errors, ds.mocks)
    ids = strategy_ids(ds.mocks)
    for c in ds.cases:
        where = c.id or "<missing id>"
        if not c.id:
            errors.append("case with missing id")
        elif c.id in seen:
            errors.append(f"duplicate case id {c.id}")
        seen.add(c.id)
        if not isinstance(c.inputs, dict) or not c.inputs:
            errors.append(f"{where}: inputs must be a non-empty object")
        if c.review.status not in REVIEW_STATUSES:
            errors.append(f"{where}: invalid review status {c.review.status!r}")
        if "mocks" in c.metadata:
            _validate_mock_block(errors, where, c.metadata["mocks"], ids)
    return errors


def import_cases(ds: Dataset, payload, source_path: str) -> int:
    if isinstance(payload, dict):
        rows = (
            payload.get("cases")
            or payload.get("examples")
            or payload.get("goldens")
            or []
        )
    else:
        rows = list(payload)
    count = 0
    for row in rows:
        raw = dict(row)
        md = dict(raw.pop("metadata", None) or raw.pop("additional_metadata", None) or {})
        if "reference_outputs" not in raw and "outputs" in raw:
            raw["reference_outputs"] = raw.pop("outputs")
        md["source"] = "import"
        md["source_path"] = source_path
        raw["metadata"] = md
        raw.pop("id", None)
        raw.pop("review", None)
        raw.pop("publication", None)
        add_case(ds, raw)
        count += 1
    return count


def set_review(ds: Dataset, ids: list[str], status: str, note: str) -> int:
    if status not in REVIEW_STATUSES:
        raise ValueError(f"invalid review status {status!r}")
    by_id = {c.id: c for c in ds.cases}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise ValueError(f"unknown case ids: {', '.join(unknown)}")
    for i in ids:
        by_id[i].review.status = status
        by_id[i].review.note = note
    return len(ids)
