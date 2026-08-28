"""Discover a LangGraph agent's structure: AST parse (no import) + live introspection."""

from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path

from evalbuilder import tool_schemas
from evalbuilder.schemas import AgentMap

DEFAULT_DECISIONS = [
    "Confirm inferred intents and scenarios",
    "Provide sample documents or describe data-domain topics",
    "State constraints the code cannot express",
]

_AGENT_FACTORIES = {"create_agent", "create_react_agent"}
_PROMPT_KWARGS = ("system_prompt", "prompt", "state_modifier", "instruction")

_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "List": "array",
    "dict": "object",
    "Dict": "object",
    "Any": None,
}

_MODEL_BASES = {"BaseModel", "TypedDict"}
_SKIP_PARAMS = {"self", "ctx", "config", "state", "tool_context", "context"}
_FIELD_KEYWORDS = {
    "description": "description", "ge": "minimum", "le": "maximum", "gt": "exclusiveMinimum",
    "lt": "exclusiveMaximum", "min_length": "minLength", "max_length": "maxLength", "pattern": "pattern",
}


def _literal_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    return ast.unparse(node)


def _annotation_schema(node, models: dict[str, dict]) -> dict:
    """JSON-schema fragment for a type annotation; model names become `$ref`s."""
    if node is None:
        return {}
    if isinstance(node, ast.Constant) and node.value is None:
        return {"type": "null"}
    if isinstance(node, ast.Name):
        if node.id in models:
            return {"$ref": f"#/$defs/{node.id}"}
        mapped = _TYPE_MAP.get(node.id, "?")
        return {} if mapped in (None, "?") else {"type": mapped}
    if isinstance(node, ast.Attribute):
        return _annotation_schema(ast.Name(id=node.attr), models)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        branches = [_annotation_schema(node.left, models), _annotation_schema(node.right, models)]
        return {"anyOf": branches}
    if isinstance(node, ast.Subscript):
        base = node.value.id if isinstance(node.value, ast.Name) else (node.value.attr if isinstance(node.value, ast.Attribute) else "")
        args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if base == "Literal":
            return {"enum": [_literal_value(a) for a in args]}
        if base == "Optional":
            return {"anyOf": [_annotation_schema(args[0], models), {"type": "null"}]}
        if base in ("list", "List", "Sequence", "Iterable"):
            return {"type": "array", "items": _annotation_schema(args[0], models)}
        if base in ("dict", "Dict", "Mapping"):
            return {"type": "object"}
        if base in ("Annotated",):
            return _annotation_schema(args[0], models)
        return _annotation_schema(node.value, models)
    return {}


def _field_schema(value, schema: dict) -> tuple[dict, bool]:
    """Apply a `Field(...)` default/keywords; returns (schema, required)."""
    schema = dict(schema)
    if value is None:
        return schema, True
    if isinstance(value, ast.Call) and _call_name(value) == "Field":
        default = value.args[0] if value.args else None
        for kw in value.keywords:
            if kw.arg == "default":
                default = kw.value
            elif kw.arg in _FIELD_KEYWORDS and isinstance(kw.value, ast.Constant):
                schema[_FIELD_KEYWORDS[kw.arg]] = kw.value.value
        if default is None or (isinstance(default, ast.Constant) and default.value is Ellipsis):
            return schema, True
        if isinstance(default, ast.Constant):
            schema["default"] = default.value
        return schema, False
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            return schema, True
        schema["default"] = value.value
    elif isinstance(value, (ast.List, ast.Dict)):
        try:
            schema["default"] = ast.literal_eval(value)
        except ValueError:
            pass
    return schema, False


def _is_model_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else "")
        if name in _MODEL_BASES:
            return True
    return False


def _pydantic_models(tree: ast.Module) -> dict[str, dict]:
    """JSON schemas of the pydantic / TypedDict classes defined at module level."""
    models: dict[str, dict] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_model_class(node):
            models[node.name] = {}  # placeholder so fields can reference each other
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name in models):
            continue
        props: dict = {}
        required: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                schema, is_required = _field_schema(stmt.value, _annotation_schema(stmt.annotation, models))
                props[stmt.target.id] = schema
                if is_required:
                    required.append(stmt.target.id)
        entry = {"type": "object", "title": node.name, "properties": props, "required": required}
        doc = ast.get_docstring(node)
        if doc:
            entry["description"] = doc
        models[node.name] = entry
    return models


def _refs_in(schema, out: set[str]) -> None:
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            out.add(ref.split("/")[-1])
        for v in schema.values():
            _refs_in(v, out)
    elif isinstance(schema, list):
        for v in schema:
            _refs_in(v, out)


def _with_defs(schema: dict, models: dict[str, dict]) -> dict:
    """Attach `$defs` for every model the schema references (transitively)."""
    needed: set[str] = set()
    _refs_in(schema, needed)
    done: set[str] = set()
    while needed - done:
        name = (needed - done).pop()
        done.add(name)
        _refs_in(models.get(name, {}), needed)
    if done:
        schema = dict(schema)
        schema["$defs"] = {n: {k: v for k, v in models[n].items()} for n in sorted(done) if n in models}
    return schema


def _function_schema(fn: ast.FunctionDef, models: dict[str, dict] | None = None) -> dict:
    models = models or {}
    properties: dict = {}
    required: list[str] = []
    args = fn.args.args
    defaults_offset = len(args) - len(fn.args.defaults)
    for i, arg in enumerate(args):
        if arg.arg in _SKIP_PARAMS:
            continue
        name = ast.unparse(arg.annotation) if arg.annotation is not None else ""
        if "Context" in name or "RunnableConfig" in name:
            continue
        properties[arg.arg] = _annotation_schema(arg.annotation, models)
        if i < defaults_offset:
            required.append(arg.arg)
        else:
            default = fn.args.defaults[i - defaults_offset]
            if isinstance(default, ast.Constant):
                properties[arg.arg]["default"] = default.value
    return _with_defs({"type": "object", "properties": properties, "required": required}, models)


def _is_tool_decorator(dec) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id == "tool"
    if isinstance(target, ast.Attribute):
        return target.attr == "tool"
    return False


def _decorator_args_schema(fn: ast.FunctionDef, models: dict[str, dict]) -> dict | None:
    """`@tool(args_schema=Model)` → that model's schema (with `$defs`)."""
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call) and _is_tool_decorator(dec):
            for kw in dec.keywords:
                if kw.arg == "args_schema" and isinstance(kw.value, ast.Name) and kw.value.id in models:
                    schema = {k: v for k, v in models[kw.value.id].items() if k != "title"}
                    return _with_defs(schema, models)
    return None


def _return_schema(fn: ast.FunctionDef, models: dict[str, dict]) -> dict:
    schema = _annotation_schema(fn.returns, models)
    if schema.get("$ref"):
        name = schema["$ref"].split("/")[-1]
        return _with_defs(dict(models.get(name, {})), models) if name in models else {}
    return _with_defs(schema, models)


def _const_str(node, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def discover_from_source(source_path: Path) -> AgentMap:
    source = Path(source_path).read_text()
    tree = ast.parse(source)

    constants: dict[str, str] = {}
    list_constants: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
            if not isinstance(target, ast.Name):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                constants[target.id] = value.value
            elif isinstance(value, ast.List):
                list_constants[target.id] = [
                    e.id for e in value.elts if isinstance(e, ast.Name)
                ]

    models = _pydantic_models(tree)
    tools: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            _is_tool_decorator(d) for d in node.decorator_list
        ):
            explicit = _decorator_args_schema(node, models)
            args_schema = explicit if explicit is not None else _function_schema(node, models)
            output_schema = _return_schema(node, models)
            doc = ast.get_docstring(node) or ""
            entry = {
                "name": node.name,
                "description": doc,
                "args_schema": args_schema,
                "output_schema": output_schema,
                "schema_source": "args_schema" if explicit is not None else "ast",
                "models": list(dict.fromkeys(tool_schemas.model_names(args_schema) + tool_schemas.model_names(output_schema))),
                "side_effecting": bool(tool_schemas.SIDE_EFFECT_RX.search(doc)),
                "used_by": [],
            }
            entry["edge_cases"] = tool_schemas.edge_cases(entry)
            tools.append(entry)

    nodes: list[dict] = []
    edges: list[list[str]] = []
    conditional_edges: list[dict] = []

    def visit(node, enclosing: str, assigned: str | None = None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            enclosing = node.name
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                visit(node.value, enclosing, assigned=target.id)
                return
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _AGENT_FACTORIES and enclosing in _AGENT_FACTORIES:
                pass  # compatibility shim wrapping the factory, not an agent node
            elif name in _AGENT_FACTORIES:
                prompt = None
                tool_names: list[str] = []
                for kw in node.keywords:
                    if kw.arg in _PROMPT_KWARGS and prompt is None:
                        prompt = _const_str(kw.value, constants)
                def _resolve_tools(candidate) -> list[str]:
                    if isinstance(candidate, ast.List):
                        return [e.id for e in candidate.elts if isinstance(e, ast.Name)]
                    if isinstance(candidate, ast.Name):
                        return list_constants.get(candidate.id, [])
                    if isinstance(candidate, ast.IfExp):
                        return _resolve_tools(candidate.body) or _resolve_tools(
                            candidate.orelse
                        )
                    return []

                for candidate in list(node.args) + [
                    kw.value for kw in node.keywords if kw.arg in (None, "tools")
                ]:
                    resolved = _resolve_tools(candidate)
                    if resolved:
                        tool_names = resolved
                if prompt is None:
                    for arg in node.args:
                        value = _const_str(arg, constants)
                        if value and len(value) > 20:
                            prompt = value
                nodes.append(
                    {
                        "id": assigned or enclosing or "agent",
                        "kind": "llm",
                        "prompt": prompt,
                        "tools": tool_names,
                        "evidence": [f"source:{source_path}:{node.lineno}"],
                    }
                )
            elif name == "add_node" and node.args:
                node_name = _const_str(node.args[0], constants) or ast.unparse(
                    node.args[0]
                )
                nodes.append(
                    {
                        "id": node_name,
                        "kind": "graph-node",
                        "prompt": None,
                        "tools": [],
                        "evidence": [f"source:{source_path}:{node.lineno}"],
                    }
                )
            elif name == "add_edge" and len(node.args) >= 2:
                edges.append(
                    [ast.unparse(a).strip("'\"") for a in node.args[:2]]
                )
            elif name == "add_conditional_edges" and node.args:
                targets: list[str] = []
                for arg in node.args[1:]:
                    if isinstance(arg, ast.Dict):
                        targets = [
                            ast.unparse(v).strip("'\"") for v in arg.values
                        ]
                conditional_edges.append(
                    {
                        "source": ast.unparse(node.args[0]).strip("'\""),
                        "targets": targets,
                        "evidence": [f"source:{source_path}:{node.lineno}"],
                    }
                )
        for child in ast.iter_child_nodes(node):
            visit(child, enclosing)

    visit(tree, "")

    for n in nodes:
        if not n.get("tools") and n.get("prompt"):
            # Dynamically built tool lists are opaque to the AST; the prompt usually
            # names the tools the node is meant to use.
            mentioned = [t["name"] for t in tools if t["name"] in n["prompt"]]
            if mentioned:
                n["tools"] = mentioned
                n["tools_source"] = "prompt"
        for t in tools:
            if t["name"] in n.get("tools", []):
                t["used_by"].append(n["id"])

    return AgentMap(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        app={"source": str(source_path), "models": sorted(models)},
        graph={
            "nodes": nodes,
            "edges": edges,
            "conditional_edges": conditional_edges,
        },
        tools=tools,
        decisions_needed=list(DEFAULT_DECISIONS),
    )


def discover_live(module: str, factory: str = "build_agent") -> dict:
    try:
        mod = importlib.import_module(module)
        graph = getattr(mod, factory)()
        drawable = graph.get_graph()
        return {
            "nodes": list(drawable.nodes),
            "edges": [[e.source, e.target] for e in drawable.edges],
        }
    except Exception as e:  # noqa: BLE001 - report, never crash discovery
        return {"error": f"{type(e).__name__}: {e}"}
