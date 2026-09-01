"""Tool input/output schemas: live resolution, a JSON-schema-subset validator, sample and
corrupt payload generators, and deterministic edge cases derived from the schemas.

Everything here is pure Python (no LLM). The discover stage stores the results on each
`tools[]` entry of the agent map; the mocks stage validates fixtures against
`output_schema`; the planner turns `edge_cases` into coverage cells.
"""

from __future__ import annotations

import re
import typing
from typing import Any

EDGE_KINDS = ("missing_required", "wrong_type", "out_of_enum", "boundary", "malformed_output")
INPUT_EDGE_FAILURE = "input_validation"
OUTPUT_EDGE_FAILURE = "tool_error_handling"
SIDE_EFFECT_RX = re.compile(r"side[- ]effect|only (call|use|invoke) (this )?after|after (the )?(user|customer) (explicitly )?confirm", re.I)

_INFERRED_MODULES = ("langchain_core.utils.pydantic", "langchain_core.tools")


# ── live resolution ────────────────────────────────────────────


def _clean(schema: dict) -> dict:
    schema = dict(schema)
    schema.pop("title", None)
    return schema


def resolve_args_schema(tool) -> tuple[dict, str]:
    """(JSON schema of the tool's arguments, source) for a LangChain tool.

    source: `args_schema` when an explicit pydantic class was passed, else `annotations`.
    """
    try:
        schema = _clean(tool.tool_call_schema.model_json_schema())
    except Exception:  # noqa: BLE001 - fall back to the loose `args` dict
        schema = {"type": "object", "properties": dict(getattr(tool, "args", {}) or {})}
    cls = getattr(tool, "args_schema", None)
    module = getattr(cls, "__module__", "") or ""
    explicit = cls is not None and isinstance(cls, type) and not module.startswith(_INFERRED_MODULES)
    return schema, "args_schema" if explicit else "annotations"


def resolve_output_schema(tool) -> dict:
    """JSON schema of the tool function's return annotation ({} when unknown)."""
    fn = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    if fn is None:
        return {}
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001
        hints = getattr(fn, "__annotations__", {}) or {}
    ret = hints.get("return")
    if ret is None or ret is type(None):
        return {}
    try:
        from pydantic import TypeAdapter

        return dict(TypeAdapter(ret).json_schema())  # keeps the root title = model name
    except Exception:  # noqa: BLE001 - unknown annotation shapes are not an error
        return {}


def tool_kind(tool) -> str:
    """`skill_loader` for Agent-Skill loaders (local, deterministic, never mocked), else `tool`."""
    from evalbuilder.skills import SKILL_LOADER_KIND, is_skill_loader

    return SKILL_LOADER_KIND if is_skill_loader(tool) else "tool"


def describe_tool(tool) -> dict:
    """Everything the agent map records about a live tool (schemas, source, kind, edges)."""
    args_schema, source = resolve_args_schema(tool)
    output_schema = resolve_output_schema(tool)
    kind = tool_kind(tool)
    entry = {
        "name": tool.name,
        "description": tool.description or "",
        "args_schema": args_schema,
        "output_schema": output_schema,
        "schema_source": source,
        "models": list(dict.fromkeys(model_names(args_schema) + model_names(output_schema))),
        "side_effecting": bool(SIDE_EFFECT_RX.search(tool.description or "")),
        "kind": kind,
        "mockable": kind == "tool",
    }
    entry["edge_cases"] = edge_cases(entry) if entry["mockable"] else []
    return entry


def model_names(schema: dict) -> list[str]:
    """Names of pydantic models referenced by a schema (its `$defs` plus a titled root)."""
    names = list((schema or {}).get("$defs", {}).keys())
    title = (schema or {}).get("title")
    if title and (schema or {}).get("type") == "object" and title not in names:
        names.insert(0, title)
    return names


# ── validation (JSON-schema subset) ────────────────────────────

_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def _deref(schema: dict, defs: dict) -> dict:
    ref = schema.get("$ref") if isinstance(schema, dict) else None
    if ref and ref.startswith("#/$defs/"):
        target = defs.get(ref.split("/")[-1], {})
        merged = {k: v for k, v in schema.items() if k != "$ref"}
        merged.update(target)
        return merged
    return schema


def _type_ok(value: Any, type_name: str) -> bool:
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    return isinstance(value, _TYPES.get(type_name, (object,)))


def validate(value: Any, schema: dict | None, defs: dict | None = None, path: str = "$", partial: bool = False) -> list[str]:
    """Problems of `value` against `schema` (empty list = conforms). Unknown keywords are ignored.
    `partial=True` skips `required` at every level (for subset matchers such as mock `matchArgs`
    and `expected_tools[].args`)."""
    if not schema:
        return []
    defs = defs if defs is not None else dict(schema.get("$defs", {}))
    schema = _deref(schema, defs)
    errors: list[str] = []

    if "const" in schema and value != schema["const"]:
        return [f"{path}: expected const {schema['const']!r}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: {value!r} not in enum {schema['enum']!r}"]
    for key in ("anyOf", "oneOf"):
        if key in schema:
            branches = schema[key]
            results = [validate(value, b, defs, path, partial) for b in branches]
            if not any(not r for r in results):
                return [f"{path}: matches no {key} branch ({'; '.join(r[0] for r in results if r)})"]
            return []
    if "allOf" in schema:
        for b in schema["allOf"]:
            errors += validate(value, b, defs, path, partial)

    types = schema.get("type")
    if types is not None:
        allowed = types if isinstance(types, list) else [types]
        if not any(_type_ok(value, t) for t in allowed):
            return [f"{path}: expected {'/'.join(allowed)}, got {type(value).__name__}"]

    if isinstance(value, dict) and (schema.get("type") == "object" or "properties" in schema or "required" in schema):
        props = schema.get("properties", {}) or {}
        for req in ([] if partial else schema.get("required", []) or []):
            if req not in value:
                errors.append(f"{path}: missing required {req!r}")
        for name, sub in props.items():
            if name in value:
                errors += validate(value[name], sub, defs, f"{path}.{name}", partial)
        if schema.get("additionalProperties") is False:
            for extra in value:
                if extra not in props:
                    errors.append(f"{path}: unexpected property {extra!r}")
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                errors += validate(item, items, defs, f"{path}[{i}]", partial)
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: not above exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: not below exclusiveMaximum {schema['exclusiveMaximum']}")
    return errors


# ── sample payloads ────────────────────────────────────────────

_DIGITS_RX = re.compile(r"^\^?\\d\{(\d+)\}\$?$|^\^?\[0-9\]\{(\d+)\}\$?$")


def _sample_string(schema: dict, name: str) -> str:
    if schema.get("format") == "date":
        return "2026-01-15"
    if schema.get("format") == "date-time":
        return "2026-01-15T10:00:00Z"
    if schema.get("format") == "email":
        return "user@example.com"
    pattern = schema.get("pattern")
    if pattern:
        m = _DIGITS_RX.match(pattern)
        if m:
            return "1" * int(m.group(1) or m.group(2))
    text = name or "example"
    min_len = schema.get("minLength", 0) or 0
    if len(text) < min_len:
        text = (text + "-example") * (min_len // max(len(text), 1) + 1)
        text = text[: max(min_len, len(text))]
    max_len = schema.get("maxLength")
    if max_len is not None and len(text) > max_len:
        text = text[:max_len]
    return text


def example(schema: dict | None, defs: dict | None = None, name: str = "") -> Any:
    """A payload that conforms to `schema` (every property filled, first enum/branch)."""
    if not schema:
        return {}
    defs = defs if defs is not None else dict(schema.get("$defs", {}))
    schema = _deref(schema, defs)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if "examples" in schema and schema["examples"]:
        return schema["examples"][0]
    for key in ("anyOf", "oneOf"):
        if key in schema:
            branches = [b for b in schema[key] if _deref(b, defs).get("type") != "null"] or schema[key]
            return example(branches[0], defs, name)
    if "allOf" in schema:
        merged: dict = {}
        for b in schema["allOf"]:
            merged.update(_deref(b, defs))
        return example(merged, defs, name)
    types = schema.get("type")
    if isinstance(types, list):
        types = next((t for t in types if t != "null"), "null")
    if types is None:
        types = "object" if "properties" in schema else ("array" if "items" in schema else "string")
    if types == "object":
        out = {}
        for prop, sub in (schema.get("properties") or {}).items():
            out[prop] = example(sub, defs, prop)
        return out
    if types == "array":
        item = example(schema.get("items") or {"type": "string"}, defs, name)
        n = max(1, schema.get("minItems", 1) or 1)
        return [item] * n
    if types == "string":
        return _sample_string(schema, name)
    if types in ("integer", "number"):
        if "default" in schema and isinstance(schema["default"], (int, float)):
            value = schema["default"]
        else:
            value = schema.get("minimum", schema.get("exclusiveMinimum", 0) + 1 if "exclusiveMinimum" in schema else 1)
        if "maximum" in schema and value > schema["maximum"]:
            value = schema["maximum"]
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            value = schema["exclusiveMaximum"] - 1
        return value if types == "integer" else float(value)
    if types == "boolean":
        return True
    return None


def _wrong_value(schema: dict, defs: dict) -> Any:
    schema = _deref(schema, defs)
    types = schema.get("type")
    if isinstance(types, list):
        types = next((t for t in types if t != "null"), "string")
    if types in ("integer", "number"):
        return "not-a-number"
    if types == "boolean":
        return "yes"
    if types in ("object", "array"):
        return "free text instead of structured data"
    return 12345


def corrupt(schema: dict | None, kind: str = "missing_required") -> Any:
    """A payload that violates `schema`: `missing_required` drops the first required
    property (or the first property), `wrong_type` mistypes the first property."""
    if not schema:
        return {"error": "malformed tool output"}
    defs = dict(schema.get("$defs", {}))
    root = _deref(schema, defs)
    sample = example(schema, defs)
    if not isinstance(sample, dict) or not sample:
        return "malformed: " + str(sample)
    props = root.get("properties") or {}
    targets = [r for r in root.get("required", []) or [] if r in sample] or list(sample)
    field = targets[0]
    if kind == "wrong_type":
        sample[field] = _wrong_value(props.get(field, {}), defs)
    else:
        sample.pop(field)
    return sample


# ── edge cases ─────────────────────────────────────────────────


def _limit_text(schema: dict) -> str | None:
    parts = []
    for key, label in (
        ("minimum", "min"), ("maximum", "max"), ("exclusiveMinimum", ">"), ("exclusiveMaximum", "<"),
        ("minLength", "minLength"), ("maxLength", "maxLength"), ("minItems", "minItems"), ("maxItems", "maxItems"),
    ):
        if key in schema:
            parts.append(f"{label} {schema[key]}")
    return ", ".join(parts) or None


def _type_text(schema: dict) -> str:
    types = schema.get("type")
    if isinstance(types, list):
        types = "/".join(t for t in types if t != "null")
    if types:
        return str(types)
    if "$ref" in schema:
        return schema["$ref"].split("/")[-1]
    if "anyOf" in schema:
        return "/".join(_type_text(b) for b in schema["anyOf"] if b.get("type") != "null")
    return "value"


def edge_cases(tool: dict) -> list[dict]:
    """Deterministic edge cases from a tool's `args_schema` / `output_schema`.

    Order is round-robin over kinds so the first N are diverse:
    missing_required → wrong_type → out_of_enum → boundary → malformed_output → …
    """
    name = tool.get("name", "tool")
    args = tool.get("args_schema") or {}
    defs = dict(args.get("$defs", {}))
    props = args.get("properties") or {}
    required = list(args.get("required") or [])
    by_kind: dict[str, list[dict]] = {k: [] for k in EDGE_KINDS}

    def edge(kind: str, field: str | None, detail: str, expected: str, failure_mode: str) -> dict:
        return {
            "id": f"edge.{name}.{kind}" + (f".{field}" if field else ""),
            "tool": name,
            "kind": kind,
            "field": field,
            "detail": detail,
            "failure_mode": failure_mode,
            "expected_behavior": expected,
            "evidence": [f"schema:{name}" + (f".{field}" if field else "")],
        }

    for field in required:
        sub = _deref(props.get(field, {}), defs)
        by_kind["missing_required"].append(edge(
            "missing_required", field,
            f"`{field}` ({_type_text(sub)}) is required by {name} but the user does not provide it",
            f"Ask the user for `{field}` instead of calling {name} with a guessed or placeholder value.",
            INPUT_EDGE_FAILURE,
        ))
    for field, raw in props.items():
        sub = _deref(raw, defs)
        types = sub.get("type")
        if isinstance(types, list):
            types = next((t for t in types if t != "null"), None)
        if "enum" in sub or "const" in sub:
            allowed = sub.get("enum") or [sub.get("const")]
            by_kind["out_of_enum"].append(edge(
                "out_of_enum", field,
                f"`{field}` must be one of {allowed}; the user asks for a value outside that set",
                f"Offer the allowed values {allowed} and do not call {name} with an invalid `{field}`.",
                INPUT_EDGE_FAILURE,
            ))
        elif types in ("integer", "number", "boolean", "object", "array") or "$ref" in raw or sub.get("pattern") or sub.get("format"):
            expected_type = _type_text(raw) if "$ref" in raw else (types or "string")
            fmt = sub.get("pattern") or sub.get("format")
            by_kind["wrong_type"].append(edge(
                "wrong_type", field,
                f"`{field}` expects {expected_type}" + (f" matching {fmt!r}" if fmt else "") + "; the user gives a non-conforming value",
                f"Do not pass the malformed `{field}` to {name}; ask for a valid {expected_type} or explain the requirement.",
                INPUT_EDGE_FAILURE,
            ))
        limit = _limit_text(sub)
        if limit:
            by_kind["boundary"].append(edge(
                "boundary", field,
                f"`{field}` has limits ({limit}); the user's value violates them",
                f"Refuse or adjust the out-of-range `{field}` and explain the limit ({limit}) before calling {name}.",
                INPUT_EDGE_FAILURE,
            ))
    output = tool.get("output_schema") or {}
    out_root = _deref(output, dict(output.get("$defs", {}))) if output else {}
    if out_root.get("required"):
        by_kind["malformed_output"].append(edge(
            "malformed_output", None,
            f"{name} returns a payload missing required fields {out_root['required'][:3]} of its output schema",
            f"Do not present fabricated values; tell the user the {name} result was incomplete and offer to retry.",
            OUTPUT_EDGE_FAILURE,
        ))

    ordered: list[dict] = []
    queues = [list(v) for v in by_kind.values()]
    while any(queues):
        for q in queues:
            if q:
                ordered.append(q.pop(0))
    return ordered


def tool_edge_cases(tools: list[dict]) -> dict[str, list[dict]]:
    return {t["name"]: (t.get("edge_cases") or edge_cases(t)) for t in tools}
