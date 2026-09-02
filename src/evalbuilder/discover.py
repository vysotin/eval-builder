"""Discover a LangGraph agent's structure: AST parse (no import) + live introspection."""

from __future__ import annotations

import ast
import hashlib
import importlib
import re
from pathlib import Path

from evalbuilder import skills as skills_mod
from evalbuilder import tool_schemas
from evalbuilder.schemas import AgentMap

DEFAULT_DECISIONS = [
    "Confirm inferred intents and scenarios",
    "Provide sample documents or describe data-domain topics",
    "State constraints the code cannot express",
]

_AGENT_FACTORIES = {"create_agent", "create_react_agent"}
_SKILL_FACTORIES = _AGENT_FACTORIES | {"create_deep_agent"}
_PROMPT_KWARGS = ("system_prompt", "prompt", "state_modifier", "instruction")
_SKILLS_VAR = re.compile(r"SKILLS?_(DIRS?|PATHS?|ROOTS?|FOLDERS?)", re.I)
_STRING_METHODS = {"format", "strip", "rstrip", "lstrip", "replace", "removesuffix", "removeprefix"}
_SKILL_PROMPT_CALLS = {"skills_prompt", "skills_inline_prompt"}

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



# ── skills: path expressions and composed prompts ──────────────


def _resolve_rel(value: str, source_dir: Path) -> Path:
    """A path literal relative to the agent source first, then to the cwd."""
    p = Path(value)
    if p.is_absolute():
        return p
    beside = source_dir / p
    if beside.exists():
        return beside
    here = Path.cwd() / p
    return here if here.exists() else beside


def _path_expr(node, source_path: Path, constants: dict[str, str], paths: dict[str, Path]) -> Path | None:
    """Resolve `"skills"`, `Path(__file__).parent / "skills"`, `Path(__file__).with_name(..)`,
    `os.path.join(os.path.dirname(__file__), "skills")`, `str(...)` and names bound to any of those."""
    source_dir = source_path.resolve().parent
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _resolve_rel(node.value, source_dir)
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return source_path.resolve()
        if node.id in paths:
            return paths[node.id]
        if node.id in constants:
            return _resolve_rel(constants[node.id], source_dir)
        return None
    if isinstance(node, ast.Attribute):
        base = _path_expr(node.value, source_path, constants, paths)
        if base is not None and node.attr == "parent":
            return base.parent
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _path_expr(node.left, source_path, constants, paths)
        if left is None:
            return None
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            return left / node.right.value
        right = _path_expr(node.right, source_path, constants, paths)
        return left / right if right is not None else None
    if isinstance(node, ast.Call):
        name = _call_name(node)
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        if name in ("Path", "str", "resolve", "absolute", "abspath"):
            target = node.args[0] if node.args else receiver
            return _path_expr(target, source_path, constants, paths) if target is not None else None
        if name == "dirname" and node.args:
            base = _path_expr(node.args[0], source_path, constants, paths)
            return base.parent if base is not None else None
        if name in ("join", "joinpath"):
            base = _path_expr(node.args[0], source_path, constants, paths) if name == "join" and node.args else (
                _path_expr(receiver, source_path, constants, paths) if receiver is not None else None
            )
            rest = node.args[1:] if name == "join" else node.args
            if base is None or not all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in rest):
                return None
            return base.joinpath(*[a.value for a in rest])
        if name == "with_name" and receiver is not None and node.args and isinstance(node.args[0], ast.Constant):
            base = _path_expr(receiver, source_path, constants, paths)
            return base.with_name(str(node.args[0].value)) if base is not None else None
    return None


def _path_exprs(node, source_path: Path, constants: dict[str, str], paths: dict[str, Path]) -> list[Path]:
    if isinstance(node, (ast.List, ast.Tuple)):
        return [p for e in node.elts for p in _path_exprs(e, source_path, constants, paths)]
    p = _path_expr(node, source_path, constants, paths)
    return [p] if p is not None else []


def _render(node, constants: dict[str, str], constant_info: dict[str, dict], skills, loader: str) -> tuple[str | None, dict]:
    """Best-effort text of a prompt expression: constants, `+`, f-strings, `x if c else y`
    (the true branch), string methods, and the skill helpers rendered from the loaded
    skills. Returns (text, {"skills": [...], "source": inline|listing|None})."""
    info: dict = {"skills": [], "source": None}

    def merge(sub: dict) -> None:
        for name in sub.get("skills", []):
            if name not in info["skills"]:
                info["skills"].append(name)
        info["source"] = info["source"] or sub.get("source")

    if isinstance(node, ast.Constant):
        return (node.value if isinstance(node.value, str) else None), info
    if isinstance(node, ast.Name):
        if node.id in constants:
            merge(constant_info.get(node.id, {}))
            return constants[node.id], info
        return None, info
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, li = _render(node.left, constants, constant_info, skills, loader)
        right, ri = _render(node.right, constants, constant_info, skills, loader)
        if left is None and right is None:
            return None, info
        merge(li)
        merge(ri)
        return (left or "") + (right or ""), info
    if isinstance(node, ast.IfExp):
        text, sub = _render(node.body, constants, constant_info, skills, loader)
        if text is None:
            text, sub = _render(node.orelse, constants, constant_info, skills, loader)
        merge(sub)
        return text, info
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                text, sub = _render(value.value, constants, constant_info, skills, loader)
                merge(sub)
                parts.append(text if text is not None else "{" + ast.unparse(value.value) + "}")
        return "".join(parts), info
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name in _SKILL_PROMPT_CALLS:
            if not skills:
                return "", info
            if name == "skills_prompt":
                loader_kw = next((kw.value.value for kw in node.keywords if kw.arg == "loader" and isinstance(kw.value, ast.Constant)), loader)
                info["skills"] = [s.name for s in skills]
                info["source"] = "listing"
                return skills_mod.skills_prompt(skills, loader=str(loader_kw)), info
            names = [a.value for a in node.args[1:] if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            known = {s.name for s in skills}
            names = [n for n in names if n in known] or ([] if len(node.args) > 1 else [s.name for s in skills])
            if not names:
                return "", info
            info["skills"] = list(names)
            info["source"] = "inline"
            return skills_mod.skills_inline_prompt(skills, *names), info
        if name == "str" and node.args:
            return _render(node.args[0], constants, constant_info, skills, loader)
        if name in _STRING_METHODS and isinstance(node.func, ast.Attribute):
            return _render(node.func.value, constants, constant_info, skills, loader)
    return None, info


def _skill_declarations(tree: ast.Module, source_path: Path, constants: dict[str, str]) -> tuple[list[Path], dict[str, str], dict[str, Path]]:
    """(skill paths, loader tools by variable, path constants) declared at module level or as
    `skills=` keywords on agent factories."""
    skill_paths: list[Path] = []
    loaders: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name, value = node.targets[0].id, node.value
        if isinstance(value, ast.Call) and _call_name(value) == "load_skills":
            for arg in list(value.args) + [kw.value for kw in value.keywords if kw.arg in ("paths", "path")]:
                skill_paths += _path_exprs(arg, source_path, constants, paths)
        elif isinstance(value, ast.Call) and _call_name(value) == "skill_loader_tool":
            tool_name = next((kw.value.value for kw in value.keywords if kw.arg == "name" and isinstance(kw.value, ast.Constant)), None)
            if tool_name is None and len(value.args) > 1 and isinstance(value.args[1], ast.Constant):
                tool_name = value.args[1].value
            loaders[name] = str(tool_name or "load_skill")
        else:
            resolved = _path_expr(value, source_path, constants, paths) if not isinstance(value, ast.Constant) else None
            if resolved is not None:
                paths[name] = resolved
            if _SKILLS_VAR.fullmatch(name):
                skill_paths += _path_exprs(value, source_path, constants, paths)
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and _call_name(call) in _SKILL_FACTORIES:
            for kw in call.keywords:
                if kw.arg == "skills":
                    skill_paths += _path_exprs(kw.value, source_path, constants, paths)
    unique: list[Path] = []
    for p in skill_paths:
        if p.is_dir() and p not in unique:
            unique.append(p)
    return unique, loaders, paths


def _loader_entry(name: str, skill_names: list[str]) -> dict:
    available = ", ".join(skill_names) or "none"
    return {
        "name": name,
        "description": f"Read the full instructions of a skill by name before using it. Available skills: {available}.",
        "args_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        "output_schema": {"type": "string"},
        "schema_source": "ast",
        "models": [],
        "side_effecting": False,
        "kind": skills_mod.SKILL_LOADER_KIND,
        "mockable": False,
        "used_by": [],
        "edge_cases": [],
    }


def discover_from_source(source_path: Path) -> AgentMap:
    source_path = Path(source_path)
    source = source_path.read_text()
    tree = ast.parse(source)

    constants: dict[str, str] = {}
    constant_info: dict[str, dict] = {}
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

    # Agent Skills: declared folders (loaded from disk, never executed) and loader tools.
    skill_paths, loader_tools, _ = _skill_declarations(tree, source_path, constants)
    loaded_skills = skills_mod.load_skills(skill_paths)
    loader_name = next(iter(loader_tools.values()), "load_skill")
    # Composed prompts (`BASE + skills_inline_prompt(...)`, f-strings) become constants too.
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target in constants or target in list_constants or isinstance(node.value, ast.Constant):
                continue
            text, info = _render(node.value, constants, constant_info, loaded_skills, loader_name)
            if text and (info["skills"] or len(text) > 20):
                constants[target] = text
                constant_info[target] = info

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
            loader = node.name in skills_mod.SKILL_LOADER_NAMES
            entry = {
                "name": node.name,
                "description": doc,
                "args_schema": args_schema,
                "output_schema": output_schema,
                "schema_source": "args_schema" if explicit is not None else "ast",
                "models": list(dict.fromkeys(tool_schemas.model_names(args_schema) + tool_schemas.model_names(output_schema))),
                "side_effecting": bool(tool_schemas.SIDE_EFFECT_RX.search(doc)),
                "kind": skills_mod.SKILL_LOADER_KIND if loader else "tool",
                "mockable": not loader,
                "used_by": [],
            }
            entry["edge_cases"] = tool_schemas.edge_cases(entry) if entry["mockable"] else []
            tools.append(entry)
    skill_names = [s.name for s in loaded_skills]
    for tool_name in loader_tools.values():
        if tool_name not in {t["name"] for t in tools}:
            tools.append(_loader_entry(tool_name, skill_names))
    if "TOOLS" in list_constants:  # keep the module's own order: TOOLS first, then the rest
        order = {n: i for i, n in enumerate(list_constants["TOOLS"])}
        tools.sort(key=lambda t: order.get(t["name"], len(order)))

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
            if isinstance(target, ast.Name) and enclosing and not isinstance(node.value, (ast.Constant, ast.List, ast.Dict)):
                # a prompt composed inside the factory (`prompt = BASE + skills_prompt(...)`)
                text, info = _render(node.value, constants, constant_info, loaded_skills, loader_name)
                if text and (info["skills"] or len(text) > 20):
                    constants[target.id] = text
                    constant_info[target.id] = info
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _AGENT_FACTORIES and enclosing in _AGENT_FACTORIES:
                pass  # compatibility shim wrapping the factory, not an agent node
            elif name in _AGENT_FACTORIES:
                prompt = None
                prompt_info: dict = {}
                tool_names: list[str] = []
                for kw in node.keywords:
                    if kw.arg in _PROMPT_KWARGS and prompt is None:
                        prompt, prompt_info = _render(kw.value, constants, constant_info, loaded_skills, loader_name)
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
                        value, value_info = _render(arg, constants, constant_info, loaded_skills, loader_name)
                        if value and len(value) > 20:
                            prompt, prompt_info = value, value_info
                entry = {
                    "id": assigned or enclosing or "agent",
                    "kind": "llm",
                    "prompt": prompt,
                    "tools": tool_names,
                    "evidence": [f"source:{source_path}:{node.lineno}"],
                }
                entry["_skills"] = prompt_info
                nodes.append(entry)
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

    skill_entries = skills_mod.describe_skills(loaded_skills, [t["name"] for t in tools])
    entry_by_name = {e["name"]: e for e in skill_entries}
    loader_names = {t["name"] for t in tools if t.get("kind") == skills_mod.SKILL_LOADER_KIND}
    for n in nodes:
        info = n.pop("_skills", None) or {}
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
        if n.get("kind") == "llm" and skill_names:
            linked = list(info.get("skills") or [])
            disclosure = info.get("source")
            for name in skill_names:  # a prompt that names a skill uses it, whatever the disclosure
                if name not in linked and name in (n.get("prompt") or ""):
                    linked.append(name)
                    disclosure = disclosure or "prompt"
            if not linked and loader_names & set(n.get("tools") or []):
                # holding the loader tool means the node can read — so might use — any skill
                linked = list(skill_names)
                disclosure = "loader"
            if linked:
                n["skills"] = linked
                n["skills_source"] = disclosure or "prompt"
                n["capabilities"] = [
                    skills_mod.capability_line(entry_by_name[name], n["skills_source"], n.get("tools"))
                    for name in linked
                    if name in entry_by_name
                ]
    for entry in skill_entries:
        entry["used_by"] = [n["id"] for n in nodes if entry["name"] in (n.get("skills") or [])]

    app = {"source": str(source_path), "models": sorted(models)}
    if skill_paths:
        app["skills_dir"] = str(skill_paths[0])
        if len(skill_paths) > 1:
            app["skills_dirs"] = [str(p) for p in skill_paths]
    return AgentMap(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        app=app,
        graph={
            "nodes": nodes,
            "edges": edges,
            "conditional_edges": conditional_edges,
        },
        tools=tools,
        skills=skill_entries,
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
