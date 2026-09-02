"""Agent Skills — `SKILL.md` folders (YAML frontmatter + markdown instructions, optional
`references/` and `scripts/`) as the target agent uses them and as the agent map records them.

Two disclosure styles are supported and both are pure data for evalbuilder:

- **inline**: `skills_inline_prompt(skills, …)` embeds skill bodies into a node's system
  prompt;
- **on demand**: `skills_prompt(skills)` lists name + description in the prompt and
  `skill_loader_tool(skills)` gives the model a `load_skill(name)` tool that returns the
  body. The loader is a *local, deterministic* tool: discovery marks it `kind:
  skill_loader` and the mock layers never wrap it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml
from langchain_core.tools import BaseTool, StructuredTool

SKILL_FILE = "SKILL.md"
SKILL_LOADER_KIND = "skill_loader"
SKILL_LOADER_NAMES = ("load_skill", "read_skill", "get_skill", "use_skill")
EXCERPT_CHARS = 300
SKILL_INSTRUCTION_CHARS = 1500  # bodies above this are summarized into `instruction`

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---[ \t]*\n?", re.S)
_HEADING = re.compile(r"^#{1,6}[ \t]*(.+?)[ \t]*$", re.M)
_RULE_RX = re.compile(r"\b(?:never|must|always|only|do not|don'?t)\b", re.I)
_LINE_PREFIX_RX = re.compile(r"^[\s>*+-]*(?:\d+\.\s*)?")


@dataclass
class Skill:
    name: str
    description: str
    path: str  # SKILL.md
    dir: str
    body: str
    metadata: dict = field(default_factory=dict)  # frontmatter minus name/description
    references: list[dict] = field(default_factory=list)  # {path, title, chars, excerpt}
    scripts: list[str] = field(default_factory=list)


# ── loading ────────────────────────────────────────────────────


def parse_skill_file(text: str) -> tuple[dict, str]:
    """(frontmatter, body) of a SKILL.md; a file without frontmatter is all body."""
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text.strip()
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data, text[m.end():].strip()


def _excerpt(text: str) -> str:
    plain = _HEADING.sub("", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain[:EXCERPT_CHARS]


def _references(skill_dir: Path) -> list[dict]:
    refs_dir = skill_dir / "references"
    if not refs_dir.is_dir():
        return []
    out = []
    for path in sorted(p for p in refs_dir.rglob("*") if p.is_file()):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        heading = _HEADING.search(text)
        out.append({
            "path": str(path.relative_to(skill_dir)).replace("\\", "/"),
            "title": heading.group(1).strip() if heading else path.stem,
            "chars": len(text),
            "excerpt": _excerpt(text),
        })
    return out


def _scripts(skill_dir: Path) -> list[str]:
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir():
        return []
    return [str(p.relative_to(skill_dir)).replace("\\", "/") for p in sorted(scripts_dir.rglob("*")) if p.is_file()]


def load_skill(path: Path) -> Skill:
    """One skill from its folder or its SKILL.md path."""
    path = Path(path)
    file = path if path.is_file() else path / SKILL_FILE
    skill_dir = file.parent
    frontmatter, body = parse_skill_file(file.read_text(errors="ignore"))
    name = str(frontmatter.get("name") or skill_dir.name).strip()
    description = str(frontmatter.get("description") or "").strip()
    metadata = {k: v for k, v in frontmatter.items() if k not in ("name", "description")}
    return Skill(
        name=name, description=description, path=str(file), dir=str(skill_dir), body=body,
        metadata=metadata, references=_references(skill_dir), scripts=_scripts(skill_dir),
    )


def load_skills(paths: str | Path | Iterable[str | Path] | None) -> list[Skill]:
    """Skills under `paths`: a skills root (every child folder holding SKILL.md, sorted),
    one skill folder, one SKILL.md file, or a list of those. Missing paths are ignored."""
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[Skill] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.name == SKILL_FILE:
            out.append(load_skill(p))
        elif p.is_dir() and (p / SKILL_FILE).exists():
            out.append(load_skill(p))
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_dir() and (child / SKILL_FILE).exists():
                    out.append(load_skill(child))
    return out


# ── prompts ────────────────────────────────────────────────────


def skills_prompt(skills: list[Skill], loader: str = "load_skill") -> str:
    """Progressive disclosure: names + descriptions, bodies behind the loader tool."""
    if not skills:
        return ""
    lines = [f"Skills available on demand — call {loader}(name) to read a skill's full instructions before using it:"]
    lines += [f"- {s.name}: {s.description}".rstrip(": ") for s in skills]
    return "\n".join(lines)


def skills_inline_prompt(skills: list[Skill], *names: str) -> str:
    """Full bodies of every skill (or only the named ones) as prompt sections."""
    if not skills:
        return ""
    by_name = {s.name: s for s in skills}
    chosen = [by_name[n] for n in names] if names else list(skills)  # KeyError for unknown names
    return "\n\n".join(f"## Skill: {s.name}\n{s.body}" for s in chosen)


# ── the loader tool ────────────────────────────────────────────


def skill_text(skill: Skill) -> str:
    """What the loader tool returns: the body plus the reference list."""
    text = skill.body
    if skill.references:
        text += "\n\nReferences:\n" + "\n".join(f"- {r['path']} ({r['title']})" for r in skill.references)
    return text


def skill_loader_tool(skills: list[Skill], name: str = "load_skill") -> BaseTool:
    by_name = {s.name: s for s in skills}
    available = ", ".join(by_name) or "none"

    def _load(name: str) -> str:
        skill = by_name.get(name)
        if skill is None:
            return f"unknown skill {name!r}; available skills: {available}"
        return skill_text(skill)

    return StructuredTool.from_function(
        func=_load,
        name=name,
        description=f"Read the full instructions of a skill by name before using it. Available skills: {available}.",
        metadata={"kind": SKILL_LOADER_KIND},
    )


def is_skill_loader(tool: Any) -> bool:
    """A skill loader: tagged by `skill_loader_tool`, or named like one."""
    metadata = getattr(tool, "metadata", None) or {}
    if isinstance(metadata, dict) and metadata.get("kind") == SKILL_LOADER_KIND:
        return True
    return getattr(tool, "name", "") in SKILL_LOADER_NAMES


# ── agent-map entry ────────────────────────────────────────────


def _allowed_tools(value: Any) -> list[str]:
    if isinstance(value, str):
        return [v for v in re.split(r"[,\s]+", value) if v]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _logical_lines(body: str) -> list[str]:
    """Body lines with wrapped continuations joined back to their bullet or sentence."""
    units: list[str] = [""]
    for raw in body.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            units.append("")
            continue
        if re.match(r"^\s*(?:[-*+]|\d+\.)\s+", raw) or not units[-1]:
            units.append(_LINE_PREFIX_RX.sub("", raw).strip())
        else:
            units[-1] += " " + stripped
    return [u for u in units if u]


def skill_rules(body: str) -> list[str]:
    """Imperative lines of the body (never/must/always/only/don't) — deterministic
    seeds for constraint and failure-case analysis."""
    return [line for line in _logical_lines(body) if len(line) > 15 and _RULE_RX.search(line)]


def skill_tools(skill: Skill, tool_names: Iterable[str]) -> tuple[list[str], list[str]]:
    """(resolved, unknown): frontmatter `allowed-tools` resolved against the agent's
    tools, then body-mentioned tools; unknown = allowed names matching no agent tool."""
    names = list(tool_names)
    allowed = _allowed_tools(skill.metadata.get("allowed-tools"))
    resolved = [n for n in allowed if n in names]
    unknown = [n for n in allowed if n not in names]
    for n in names:
        if n not in resolved and re.search(rf"\b{re.escape(n)}\b", skill.body):
            resolved.append(n)
    return resolved, unknown


def summarize_skill(skill: Skill, limit: int = SKILL_INSTRUCTION_CHARS) -> str:
    """Deterministic summary of a long body: the description, each section heading with
    its first content line, then the rule lines — capped at `limit`."""
    parts: list[str] = [skill.description] if skill.description else []
    section: str | None = None
    for raw in skill.body.splitlines():
        heading = _HEADING.match(raw)
        if heading:
            section = heading.group(1).strip()
            continue
        line = _LINE_PREFIX_RX.sub("", raw).strip()
        if section is not None and line:
            parts.append(f"{section}: {line}")
            section = None
    joined = " ".join(parts)
    parts += [f"rule: {r}" for r in skill_rules(skill.body) if r not in joined]
    text = ""
    for p in parts:
        if text and len(text) + len(p) + 1 > limit:
            break
        text += ("\n" if text else "") + p
    return text[:limit] if text else skill.body[:limit]


def capability_line(entry: dict, disclosure: str, node_tools: Iterable[str] | None = None) -> str:
    """One human line for an LLM node: which skill it can call, which tools that
    reaches (scoped to the node's own tools) and the result it achieves."""
    tools = list(entry.get("tools") or [])
    if node_tools is not None and tools:
        wired = [t for t in tools if t in set(node_tools)]
        via = (f" → tools {', '.join(wired)}" if wired
               else f" → tools {', '.join(tools)} (not wired on this node)")
    elif tools:
        via = f" → tools {', '.join(tools)}"
    else:
        via = ""
    outcome = entry.get("description") or ""
    return f"skill {entry.get('name')} ({disclosure}){via}" + (f" — {outcome}" if outcome else "")


def describe_skill(skill: Skill, tool_names: Iterable[str] = ()) -> dict:
    """The `skills[]` entry of the agent map (links `used_by` are filled by discovery)."""
    names = list(tool_names)
    mentioned = [n for n in names if re.search(rf"\b{re.escape(n)}\b", skill.body)]
    resolved, unknown = skill_tools(skill, names)
    chars = len(skill.body)
    summarized = chars > SKILL_INSTRUCTION_CHARS
    return {
        "name": skill.name,
        "description": skill.description,
        "path": skill.path,
        "dir": skill.dir,
        "prompt": skill.body,
        "chars": chars,
        "instruction": summarize_skill(skill) if summarized else skill.body,
        "summarized": summarized,
        "allowed_tools": _allowed_tools(skill.metadata.get("allowed-tools")),
        "tools": resolved,
        "unknown_tools": unknown,
        "rules": skill_rules(skill.body),
        "metadata": dict(skill.metadata.get("metadata") or {}),
        "references": [dict(r) for r in skill.references],
        "scripts": list(skill.scripts),
        "tools_mentioned": mentioned,
        "used_by": [],
        "evidence": [f"skill:{skill.name}"],
    }


def describe_skills(skills: list[Skill], tool_names: Iterable[str] = ()) -> list[dict]:
    names = list(tool_names)
    return [describe_skill(s, names) for s in skills]
