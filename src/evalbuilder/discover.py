"""Discover a LangGraph agent's structure: AST parse (no import) + live introspection."""

from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path

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
    "dict": "object",
}

_SKIP_PARAMS = {"self", "ctx", "config", "state", "tool_context", "context"}


def _annotation_to_type(node) -> dict:
    if node is None:
        return {}
    name = ast.unparse(node)
    if "Context" in name:
        return {"skip": True}
    base = name.split("[", 1)[0]
    if base in _TYPE_MAP:
        return {"type": _TYPE_MAP[base]}
    return {}


def _is_tool_decorator(dec) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id == "tool"
    if isinstance(target, ast.Attribute):
        return target.attr == "tool"
    return False


def _function_schema(fn: ast.FunctionDef) -> dict:
    properties: dict = {}
    required: list[str] = []
    args = fn.args.args
    defaults_offset = len(args) - len(fn.args.defaults)
    for i, arg in enumerate(args):
        if arg.arg in _SKIP_PARAMS:
            continue
        info = _annotation_to_type(arg.annotation)
        if info.get("skip"):
            continue
        properties[arg.arg] = {k: v for k, v in info.items() if k != "skip"}
        if i < defaults_offset:
            required.append(arg.arg)
    return {"type": "object", "properties": properties, "required": required}


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

    tools: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            _is_tool_decorator(d) for d in node.decorator_list
        ):
            tools.append(
                {
                    "name": node.name,
                    "description": ast.get_docstring(node) or "",
                    "args_schema": _function_schema(node),
                    "used_by": [],
                }
            )

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
        app={"source": str(source_path)},
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
