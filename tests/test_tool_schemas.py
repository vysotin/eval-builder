"""Tool schema capture, the JSON-schema-subset validator, sample/corrupt payloads, edge cases."""

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from evalbuilder import tool_schemas as ts


class RunbookQuery(BaseModel):
    """Runbook search query."""

    query: str = Field(description="free text")
    severity: Literal["low", "high"] = "low"
    limit: int = Field(3, ge=1, le=10)


class Ticket(BaseModel):
    title: str = Field(min_length=3)
    priority: Literal["p1", "p2", "p3"]
    tags: list[str] = []


class Receipt(BaseModel):
    ticket_id: str
    url: str | None = None


@tool(args_schema=RunbookQuery)
def search_runbooks(query: str, severity: str = "low", limit: int = 3) -> dict:
    """Search runbooks (external)."""
    return {}


@tool
def create_ticket(ticket: Ticket, note: str | None = None) -> Receipt:
    """Create a ticket. Side-effecting: only call after the user confirms."""
    return Receipt(ticket_id="T1")


@tool
def ping(host: str) -> str:
    """Ping a host."""
    return "ok"


def test_resolve_args_schema_distinguishes_explicit_class_from_annotations():
    schema, source = ts.resolve_args_schema(search_runbooks)
    assert source == "args_schema"
    assert schema["properties"]["severity"]["enum"] == ["low", "high"]
    assert schema["properties"]["limit"]["maximum"] == 10 and schema["required"] == ["query"]
    schema2, source2 = ts.resolve_args_schema(create_ticket)
    assert source2 == "annotations"
    assert "Ticket" in schema2["$defs"] and schema2["properties"]["ticket"]["$ref"].endswith("Ticket")


def test_resolve_output_schema_from_return_annotation():
    out = ts.resolve_output_schema(create_ticket)
    assert out["required"] == ["ticket_id"] and "url" in out["properties"]
    assert ts.resolve_output_schema(ping) == {"type": "string"}
    assert ts.resolve_output_schema(search_runbooks)["type"] == "object"
    assert out["title"] == "Receipt"


def test_describe_tool_records_models_side_effects_and_edges():
    entry = ts.describe_tool(create_ticket)
    assert entry["schema_source"] == "annotations" and entry["side_effecting"] is True
    assert entry["models"] == ["Ticket", "Receipt"]
    kinds = {e["kind"] for e in entry["edge_cases"]}
    assert {"missing_required", "wrong_type", "malformed_output"} <= kinds
    assert ts.describe_tool(ping)["side_effecting"] is False


def test_validate_subset():
    schema = ts.resolve_args_schema(search_runbooks)[0]
    assert ts.validate({"query": "disk full", "severity": "high", "limit": 2}, schema) == []
    errors = ts.validate({"severity": "urgent", "limit": 50}, schema)
    assert any("missing required 'query'" in e for e in errors)
    assert any("not in enum" in e for e in errors) and any("above maximum" in e for e in errors)
    assert ts.validate({"query": 5}, schema) == ["$.query: expected string, got int"]
    nested = ts.resolve_args_schema(create_ticket)[0]
    assert ts.validate({"ticket": {"title": "abc", "priority": "p1"}, "note": None}, nested) == []
    errors = ts.validate({"ticket": {"title": "ab", "priority": "p9"}}, nested)
    assert any("shorter than minLength" in e for e in errors) and any("$.ticket.priority" in e for e in errors)
    assert ts.validate({"ticket": "text"}, nested) == ["$.ticket: expected object, got str"]
    assert ts.validate(True, {"type": "integer"}) and ts.validate(3, {"type": "integer"}) == []
    assert ts.validate("anything", None) == [] and ts.validate({"x": 1}, {"type": "object", "additionalProperties": False}) != []


def test_example_conforms_to_schema():
    for t in (search_runbooks, create_ticket):
        schema = ts.resolve_args_schema(t)[0]
        sample = ts.example(schema)
        assert ts.validate(sample, schema) == [], (sample, ts.validate(sample, schema))
    out = ts.resolve_output_schema(create_ticket)
    assert ts.validate(ts.example(out), out) == []
    assert ts.example({"type": "string", "pattern": r"^\d{4}$"}) == "1111"
    assert ts.example({"type": "integer", "minimum": 5, "maximum": 6}) == 5
    assert ts.example({"anyOf": [{"type": "null"}, {"type": "boolean"}]}) is True
    assert ts.example({}) == {}


def test_corrupt_violates_schema():
    out = ts.resolve_output_schema(create_ticket)
    missing = ts.corrupt(out, "missing_required")
    assert "ticket_id" not in missing and ts.validate(missing, out)
    wrong = ts.corrupt(out, "wrong_type")
    assert wrong["ticket_id"] == 12345 and ts.validate(wrong, out)
    assert ts.corrupt({}) == {"error": "malformed tool output"}
    assert ts.corrupt({"type": "string"}).startswith("malformed")


def test_edge_cases_round_robin_and_details():
    entry = ts.describe_tool(search_runbooks)
    edges = entry["edge_cases"]
    assert [e["kind"] for e in edges[:4]] == ["missing_required", "wrong_type", "out_of_enum", "boundary"]
    missing = edges[0]
    assert missing["field"] == "query" and missing["failure_mode"] == "input_validation"
    assert missing["id"] == "edge.search_runbooks.missing_required.query"
    assert missing["evidence"] == ["schema:search_runbooks.query"]
    enum_edge = next(e for e in edges if e["kind"] == "out_of_enum")
    assert enum_edge["field"] == "severity" and "['low', 'high']" in enum_edge["detail"]
    boundary = next(e for e in edges if e["kind"] == "boundary")
    assert boundary["field"] == "limit" and "max 10" in boundary["detail"]
    assert not [e for e in edges if e["kind"] == "malformed_output"]  # dict output has no required fields
    ticket_edges = ts.describe_tool(create_ticket)["edge_cases"]
    malformed = next(e for e in ticket_edges if e["kind"] == "malformed_output")
    assert malformed["failure_mode"] == "tool_error_handling" and malformed["field"] is None
    assert ts.edge_cases({"name": "bare"}) == []
    assert ts.tool_edge_cases([{"name": "ping", "args_schema": {"type": "object", "properties": {"host": {"type": "string"}}, "required": ["host"]}}])["ping"][0]["kind"] == "missing_required"


def test_partial_validation_skips_required_for_matchers():
    nested = ts.resolve_args_schema(create_ticket)[0]
    partial = {"ticket": {"priority": "p1"}}
    assert ts.validate(partial, nested) and ts.validate(partial, nested, partial=True) == []
    assert ts.validate({"ticket": {"priority": "p9"}}, nested, partial=True) == ["$.ticket.priority: 'p9' not in enum ['p1', 'p2', 'p3']"]
