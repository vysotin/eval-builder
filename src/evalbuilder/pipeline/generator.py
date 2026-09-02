"""The generator: LLM authoring of map, mocks, cases, reviews, scenarios, analysis.

Every LLM answer is structured (JSON schema) and then validated by the same code the
CLI uses; validation errors are fed back once for repair, and whatever is still
invalid is dropped and reported — the LLM never writes an artifact directly.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from evalbuilder import tool_schemas
from evalbuilder.pipeline.planning import Cell
from evalbuilder.pipeline.taxonomy import FAILURE_TYPES
from evalbuilder.schemas import AgentMap

SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
VARIANTS = ("happy", "boundary", "adversarial", "linguistic", "multi-turn")
SKILLS_DIR = Path(__file__).resolve().parents[3] / "skills"


def _skill_reference(*parts: str) -> str:
    path = SKILLS_DIR.joinpath(*parts)
    return path.read_text() if path.exists() else ""


# ── LLM wrapper ────────────────────────────────────────────────


class Generator:
    """Structured-output wrapper around a chat model; records every call."""

    def __init__(self, model, log: Callable[[str], None] | None = None):
        self.model = model
        self.log = log or (lambda msg: None)
        self.calls: list[dict] = []

    def ask(self, system: str, user: str, schema: dict) -> dict:
        started = time.time()
        title = schema.get("title", "?")
        self.log(f"generator: {title}")
        runnable = self.model.with_structured_output(schema)
        try:
            result = runnable.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        finally:
            self.calls.append({"schema": title, "seconds": round(time.time() - started, 2)})
        if not isinstance(result, dict):
            raise ValueError(f"generator returned non-object for {title}: {type(result).__name__}")
        return result


class FakeGenerator(Generator):
    """Canned answers keyed by schema title (each a queue); for offline tests."""

    def __init__(self, responses: dict[str, list[dict]]):
        super().__init__(model=None)
        self.responses = {k: list(v) for k, v in responses.items()}
        self.prompts: list[tuple[str, str, str]] = []

    def ask(self, system: str, user: str, schema: dict) -> dict:
        title = schema.get("title", "?")
        self.prompts.append((title, system, user))
        self.calls.append({"schema": title, "seconds": 0.0})
        queue = self.responses.get(title)
        if not queue:
            raise ValueError(f"FakeGenerator has no response for {title}")
        return queue.pop(0)


# ── shared context ─────────────────────────────────────────────


def agent_brief(agent_map: AgentMap, constraints: list[str], source_text: str = "") -> str:
    nodes = [
        {k: v for k, v in n.items() if k in ("id", "kind", "prompt", "tools", "skills", "skills_source")}
        for n in agent_map.graph.get("nodes", [])
    ]
    brief = {
        "module": agent_map.app.get("module"),
        "nodes": nodes,
        "edges": agent_map.graph.get("edges", []),
        "conditional_edges": agent_map.graph.get("conditional_edges", []),
        "live_graph": agent_map.graph.get("live", {}),
        "tools": [tool_brief(t) for t in agent_map.tools],
        "constraints": constraints,
    }
    skills = getattr(agent_map, "skills", None) or []
    if skills:
        brief["skills"] = [skill_brief(s) for s in skills]
    text = "AGENT STRUCTURE (from code introspection):\n" + json.dumps(brief, indent=1, ensure_ascii=False)
    if source_text:
        text += f"\n\nAGENT SOURCE:\n```python\n{source_text[:12000]}\n```"
    return text


def tool_brief(t: dict) -> dict:
    """A tool as the prompts see it: schemas and side effects, without the edge-case list."""
    out = {k: t.get(k) for k in ("name", "description", "args_schema", "used_by")}
    if t.get("output_schema"):
        out["output_schema"] = t["output_schema"]
    if t.get("schema_source"):
        out["schema_source"] = t["schema_source"]
    if t.get("side_effecting"):
        out["side_effecting"] = True
    if t.get("kind") and t["kind"] != "tool":
        out["kind"] = t["kind"]  # e.g. skill_loader: local and deterministic, never mocked
    return out


SKILL_PROMPT_CHARS = 3000


def skill_brief(s: dict) -> dict:
    """An Agent Skill as the prompts see it: description, instructions, references, links."""
    out = {
        "name": s.get("name"),
        "description": s.get("description"),
        "instructions": (s.get("instruction") or s.get("prompt") or "")[:SKILL_PROMPT_CHARS],
        "used_by": s.get("used_by") or [],
    }
    if s.get("summarized"):
        out["instructions_summarized"] = True
    if s.get("tools"):
        out["tools"] = s["tools"]
    if s.get("rules"):
        out["rules"] = s["rules"]
    if s.get("allowed_tools"):
        out["allowed_tools"] = s["allowed_tools"]
    if s.get("references"):
        out["references"] = [{"path": r.get("path"), "title": r.get("title"), "excerpt": r.get("excerpt")} for r in s["references"]]
    return out


def with_guidance(user: str, guidance: str) -> str:
    """Append the user's free-text instructions / reviewer feedback to a prompt."""
    return user + ("\n\n" + guidance.strip() if guidance and guidance.strip() else "")


# ── map authoring ──────────────────────────────────────────────

MAP_SCHEMA = {
    "title": "agent_map",
    "type": "object",
    "properties": {
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "slug like intent.order-status"},
                    "description": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "description", "evidence"],
            },
        },
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "slug like scenario.order-status.happy"},
                    "intent": {"type": "string"},
                    "kind": {"type": "string", "enum": ["happy", "failure"]},
                    "failure_mode": {"type": "string", "description": "for kind=failure: a taxonomy type"},
                    "description": {"type": "string"},
                    "expected_behavior": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "names of the agent's skills this scenario exercises (empty when the agent has no skills)",
                    },
                },
                "required": ["id", "intent", "kind", "description", "expected_behavior", "evidence"],
            },
        },
        "failure_scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "failure_type": {"type": "string"},
                    "rationale": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["failure_type", "rationale", "evidence"],
            },
        },
        "skill_failures": {
            "type": "array",
            "description": "per-skill failure cases beyond tool failure (a tool failing belongs in tool_failures)",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "failure_cases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "failure_mode": {"type": "string", "description": "a taxonomy type from the applicable list (usually skill_misuse or constraint_violation)"},
                                "expected_behavior": {"type": "string"},
                                "evidence": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["description", "expected_behavior", "evidence"],
                        },
                    },
                },
                "required": ["skill", "failure_cases"],
            },
        },
        "tool_failures": {
            "type": "array",
            "description": "per-tool failure scenarios the agent must survive (error payloads, timeouts, empty or malformed results, misuse)",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "scenarios": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "failure_mode": {"type": "string", "description": "applicable taxonomy type; defaults to tool_error_handling"},
                                "expected_behavior": {"type": "string"},
                                "evidence": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["description", "expected_behavior", "evidence"],
                        },
                    },
                },
                "required": ["tool", "scenarios"],
            },
        },
        "topics": {"type": "array", "items": {"type": "string"}},
        "derived_constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "behavioral rules stated in the agent's prompts",
        },
    },
    "required": ["intents", "scenarios", "failure_scenarios", "skill_failures", "tool_failures", "topics", "derived_constraints"],
}

MAP_SYSTEM = """You map a LangGraph agent's test surface for evaluation. Work only from the
evidence given (prompts, tools, edges, skills, source). Every intent, scenario and failure
scenario must cite evidence tokens: source:<file>:<line>, prompt:<node>, tool:<name>,
edge:<a>-><b>, constraint:<text>, skill:<name>, app:always.

Rules:
- Intents are the user goals the agent is built to serve (typically 2-6). Each intent
  needs at least one happy scenario and at least one failure scenario (kind=failure,
  with failure_mode from the applicable taxonomy list).
- Failure scenarios (cross-cutting) may only use failure types from the APPLICABLE
  list; each must explain why the graph can fail that way.
- topics: only when the agent has an explicit data domain visible in the evidence
  (e.g. product lines named in prompts); otherwise return an empty list.
- derived_constraints: behavioral rules literally stated in prompts (e.g. "never
  confirm without explicit yes") — including rules stated inside the agent's skills.
- SKILLS (when the agent has any): every skill must be exercised by at least one
  scenario that lists it in `skills` and cites `skill:<name>`; add failure scenarios
  with failure_mode `skill_misuse` (when applicable) where the user pushes the agent to
  skip or bend a step the skill mandates. Tools of kind `skill_loader` are how the
  agent reads a skill on demand — a correct agent calls them before acting on that skill.
- skill_failures: for EVERY skill, the failure cases that are NOT caused by a tool
  failing — skipping or reordering steps the skill mandates, misapplying its policy,
  acting without reading it first (on-demand skills), exceeding its allowed tools,
  answering from stale skill knowledge. Cite skill:<name> plus the violated rule line.
  failure_mode, when set, must come from the applicable list (usually skill_misuse or
  constraint_violation). A case whose only cause is a tool failure belongs in
  tool_failures instead.
- tool_failures: for EVERY tool, the failure scenarios the agent must survive — error
  payloads, timeouts, empty results, malformed output, calling it with guessed args.
  failure_mode defaults to tool_error_handling and must be applicable when set.
- Use stable slugs: intent.<name>, scenario.<intent-name>.<short-name>."""


def author_map(
    gen: Generator,
    agent_map: AgentMap,
    source_text: str,
    constraints: list[str],
    applicable: dict[str, list[str]],
    guidance: str = "",
) -> tuple[dict, list[str]]:
    """Returns (validated map sections, problems)."""
    taxonomy = _skill_reference("agent-eval-discover", "references", "failure-taxonomy.md")
    user = with_guidance(
        agent_brief(agent_map, constraints, source_text)
        + "\n\nAPPLICABLE FAILURE TYPES (precondition evidence):\n"
        + json.dumps(applicable, indent=1)
        + ("\n\nFAILURE TAXONOMY:\n" + taxonomy if taxonomy else ""),
        guidance,
    )
    skill_names = [s.get("name") for s in (getattr(agent_map, "skills", None) or []) if s.get("name")]
    tool_names = [t.get("name") for t in agent_map.tools if t.get("name")]
    result = gen.ask(MAP_SYSTEM, user, MAP_SCHEMA)
    cleaned, errors = _validate_map(result, applicable, skill_names, tool_names)
    if errors:
        repair = (
            user
            + "\n\nYOUR PREVIOUS ANSWER HAD THESE PROBLEMS — return the complete corrected map:\n- "
            + "\n- ".join(errors)
            + "\n\nPREVIOUS ANSWER:\n"
            + json.dumps(result, ensure_ascii=False)[:8000]
        )
        result = gen.ask(MAP_SYSTEM, repair, MAP_SCHEMA)
        cleaned, errors = _validate_map(result, applicable, skill_names, tool_names)
    # An unexercised skill is a coverage gap, reported (also as coverage.uncovered_skills), not repaired.
    for name in skill_names:
        if not any(name in (s.get("skills") or []) for s in cleaned["scenarios"]):
            errors.append(f"skill {name} is not exercised by any scenario (list it in `skills` and cite skill:{name})")
    return cleaned, errors


def _validate_map(
    result: dict,
    applicable: dict[str, list[str]],
    skill_names: list[str] | None = None,
    tool_names: list[str] | None = None,
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    skill_names = list(skill_names or [])
    tool_names = list(tool_names or [])
    intents = [i for i in result.get("intents", []) if isinstance(i, dict)]
    scenarios = [s for s in result.get("scenarios", []) if isinstance(s, dict)]
    failures = [f for f in result.get("failure_scenarios", []) if isinstance(f, dict)]

    good_intents = []
    for i in intents:
        iid = str(i.get("id", ""))
        if not SLUG.match(iid) or not iid.startswith("intent."):
            errors.append(f"intent id {iid!r} must be a slug starting with 'intent.'")
            continue
        if not i.get("evidence"):
            errors.append(f"intent {iid} has no evidence")
            continue
        good_intents.append({**i, "status": "hypothesis"})
    intent_ids = {i["id"] for i in good_intents}
    if not good_intents:
        errors.append("no valid intents")

    good_scenarios = []
    for s in scenarios:
        sid = str(s.get("id", ""))
        if not SLUG.match(sid):
            errors.append(f"scenario id {sid!r} must be a slug")
            continue
        if s.get("intent") not in intent_ids:
            errors.append(f"scenario {sid} references unknown intent {s.get('intent')!r}")
            continue
        if not s.get("evidence"):
            errors.append(f"scenario {sid} has no evidence")
            continue
        kind = s.get("kind", "happy")
        if kind == "failure":
            fm = s.get("failure_mode")
            if fm not in applicable:
                errors.append(
                    f"scenario {sid} uses failure_mode {fm!r} which is not applicable "
                    f"(choose from {sorted(applicable)})"
                )
                continue
        entry = {**s, "kind": kind, "status": "hypothesis"}
        cited = [str(e)[len("skill:"):] for e in (s.get("evidence") or []) if str(e).startswith("skill:")]
        listed = [str(x) for x in (s.get("skills") or []) if isinstance(x, str)]
        unknown = [x for x in listed + cited if x not in skill_names]
        if unknown:
            errors.append(f"scenario {sid} references unknown skill(s) {sorted(set(unknown))} (known: {skill_names})")
        linked = [x for x in dict.fromkeys(listed + cited) if x in skill_names]
        if linked:
            entry["skills"] = linked
        elif "skills" in entry:
            entry.pop("skills")
        good_scenarios.append(entry)
    for iid in intent_ids:
        kinds = {s["kind"] for s in good_scenarios if s["intent"] == iid}
        if "happy" not in kinds:
            errors.append(f"intent {iid} has no happy scenario")
        if "failure" not in kinds:
            errors.append(f"intent {iid} has no failure scenario")

    good_failures = []
    seen = set()
    for f in failures:
        ft = f.get("failure_type")
        if ft not in applicable:
            errors.append(f"failure_scenarios uses non-applicable type {ft!r}")
            continue
        if ft in seen:
            continue
        seen.add(ft)
        good_failures.append({**f, "evidence": f.get("evidence") or applicable[ft]})

    good_skill_failures = []
    for sf in [x for x in result.get("skill_failures", []) or [] if isinstance(x, dict)]:
        name = sf.get("skill")
        if name not in skill_names:
            errors.append(f"skill_failures references unknown skill {name!r} (known: {skill_names})")
            continue
        cases = []
        for c in [c for c in sf.get("failure_cases", []) or [] if isinstance(c, dict)]:
            if not c.get("description"):
                continue
            fm = c.get("failure_mode")
            if fm and fm not in applicable:
                errors.append(f"skill_failures[{name}] uses non-applicable failure_mode {fm!r}")
                continue
            cases.append({**c, "evidence": c.get("evidence") or [f"skill:{name}"]})
        if cases:
            good_skill_failures.append({"skill": name, "failure_cases": cases})

    good_tool_failures = []
    for tf in [x for x in result.get("tool_failures", []) or [] if isinstance(x, dict)]:
        name = tf.get("tool")
        if name not in tool_names:
            errors.append(f"tool_failures references unknown tool {name!r}")
            continue
        scenarios = []
        for c in [c for c in tf.get("scenarios", []) or [] if isinstance(c, dict)]:
            if not c.get("description"):
                continue
            fm = c.get("failure_mode") or ("tool_error_handling" if "tool_error_handling" in applicable else None)
            if fm and fm not in applicable:
                errors.append(f"tool_failures[{name}] uses non-applicable failure_mode {fm!r}")
                continue
            entry = {**c, "evidence": c.get("evidence") or [f"tool:{name}"]}
            if fm:
                entry["failure_mode"] = fm
            scenarios.append(entry)
        if scenarios:
            good_tool_failures.append({"tool": name, "scenarios": scenarios})

    topics = [t for t in result.get("topics", []) if isinstance(t, str) and t.strip()]
    derived = [c for c in result.get("derived_constraints", []) if isinstance(c, str) and c.strip()]
    return (
        {
            "intents": good_intents,
            "scenarios": good_scenarios,
            "failure_scenarios": good_failures,
            "skill_failures": good_skill_failures,
            "tool_failures": good_tool_failures,
            "topics": topics,
            "derived_constraints": derived,
        },
        errors,
    )


# ── mock authoring ─────────────────────────────────────────────

MOCK_SCHEMA = {
    "title": "mock_fixtures",
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "default_response": {
                        "type": "string",
                        "description": "JSON text of a realistic successful response",
                    },
                    "variants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "match_args": {"type": "string", "description": "JSON object of args to match (subset)"},
                                "response": {"type": "string", "description": "JSON text"},
                                "purpose": {"type": "string"},
                            },
                            "required": ["match_args", "response", "purpose"],
                        },
                    },
                },
                "required": ["name", "default_response", "variants"],
            },
        }
    },
    "required": ["tools"],
}

MOCK_SYSTEM = """You write deterministic mock fixtures for an agent's tools. For every tool
return a realistic default successful response (JSON text) consistent with the tool's
description and args schema, plus 1-3 variants keyed by specific args (JSON text) that
later test cases can rely on — e.g. a known order id, a known query. Fixtures must be
self-consistent across tools (ids, categories, amounts referenced by one tool must exist
in the others). Do not include error responses here; error injections are declared
per case. Every response must be valid JSON text. When a tool declares an
`output_schema`, every response for it must conform to that schema exactly (all
required fields, right types, enum values); `match_args` must conform to `args_schema`
(pydantic models appear as nested objects)."""


def generic_fixture(tool: dict) -> dict:
    return {"ok": True, "tool": tool["name"], "note": "generic fixture (generator fallback)"}


def _schema_fixture(tool: dict) -> dict:
    """A fixture that conforms to the tool's output schema, else the generic one."""
    out = tool.get("output_schema") or {}
    if out.get("type") == "object" and out.get("properties"):
        sample = tool_schemas.example(out)
        if isinstance(sample, dict):
            return sample
    return generic_fixture(tool)


def _present_arg_problems(args: dict, tool: dict) -> list[str]:
    """Type problems of the keys present in `args` against the tool's args schema
    (subset semantics: missing keys are fine)."""
    schema = tool.get("args_schema") or {}
    props = schema.get("properties") or {}
    defs = dict(schema.get("$defs", {}))
    problems: list[str] = []
    for key, value in (args or {}).items():
        if key in props:
            problems += tool_schemas.validate(value, props[key], defs, f"$.{key}", partial=True)
    return problems


def author_mocks(gen: Generator, tools: list[dict], guidance: str = "", wildcard: bool = True) -> tuple[dict[str, list[dict]], list[str]]:
    """Returns (rules by tool, problems). Every tool ends with a wildcard rule (unless
    `wildcard=False`: under `on_miss: llm` only the keyed variants stay and the long tail
    goes to the mock engine); responses are validated against `output_schema`
    (non-conforming defaults are replaced with a schema-conformant sample,
    non-conforming variants are dropped)."""
    problems: list[str] = []
    rules: dict[str, list[dict]] = {}
    try:
        result = gen.ask(
            MOCK_SYSTEM,
            with_guidance("TOOLS:\n" + json.dumps([tool_brief(t) for t in tools], indent=1, ensure_ascii=False), guidance),
            MOCK_SCHEMA,
        )
        by_name = {t.get("name"): t for t in result.get("tools", []) if isinstance(t, dict)}
    except Exception as e:  # noqa: BLE001 - fall back to generic fixtures
        problems.append(f"mock generation failed, using generic fixtures: {type(e).__name__}: {e}")
        by_name = {}

    for tool in tools:
        name = tool["name"]
        entry = by_name.get(name)
        tool_rules: list[dict] = []
        output_schema = tool.get("output_schema") or {}
        if entry:
            for variant in entry.get("variants", []) or []:
                match_args = _parse_json(variant.get("match_args"))
                response = _parse_json(variant.get("response"), allow_scalar=True)
                if not (isinstance(match_args, dict) and match_args and response is not None):
                    problems.append(f"mock variant for {name} ignored (invalid JSON)")
                    continue
                bad = _present_arg_problems(match_args, tool) + tool_schemas.validate(response, output_schema)
                if bad:
                    problems.append(f"mock variant for {name} dropped (schema): {'; '.join(bad[:3])}")
                    continue
                tool_rules.append({"matchArgs": match_args, "response": response})
            default = _parse_json(entry.get("default_response"), allow_scalar=True)
            if default is None:
                problems.append(f"default fixture for {name} was not valid JSON; using schema/generic sample")
                default = _schema_fixture(tool)
            else:
                bad = tool_schemas.validate(default, output_schema)
                if bad:
                    problems.append(f"default fixture for {name} violated output_schema ({'; '.join(bad[:3])}); replaced with a conformant sample")
                    default = _schema_fixture(tool)
        else:
            if by_name:
                problems.append(f"generator returned no fixture for {name}; using schema/generic sample")
            default = _schema_fixture(tool)
        if wildcard:
            tool_rules.append({"matchArgs": {}, "response": default})
        rules[name] = tool_rules
    return rules, problems


# ── mock strategies (layer 2) ──────────────────────────────────

STRATEGIES_SCHEMA = {
    "title": "mock_strategies",
    "type": "object",
    "properties": {
        "world": {"type": "string", "description": "one paragraph: the simulated backend every tool shares (entities, ids, invariants)"},
        "strategies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "slug: default, degraded, empty, ..."},
                    "description": {"type": "string"},
                    "tools": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "behavior": {"type": "string", "description": "concrete rules an LLM follows to answer this tool under this strategy"},
                                "fallback_response": {"type": "string", "description": "JSON text: a valid response used when a generated one fails validation"},
                                "examples": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {"args": {"type": "string", "description": "JSON object"}, "response": {"type": "string", "description": "JSON text"}},
                                        "required": ["args", "response"],
                                    },
                                },
                            },
                            "required": ["name", "behavior", "fallback_response", "examples"],
                        },
                    },
                },
                "required": ["id", "description", "tools"],
            },
        },
    },
    "required": ["world", "strategies"],
}

STRATEGIES_SYSTEM = """You design the behaviour of a simulated backend for an agent's tools. At run time an
LLM plays each tool for calls that no deterministic fixture answers; your strategies are
its script. Write:
- world: one paragraph describing the shared backend every tool draws from — the
  entities and ids that exist (consistent with the MOCK FIXTURES), invariants, ranges.
- a `default` strategy (healthy backend) with a behaviour entry for EVERY tool: concrete
  rules the LLM can follow deterministically (which ids resolve to what, how unknown ids
  are answered, value ranges, ordering), 1-2 examples (args → response), and a
  fallback_response that conforms to the tool's output_schema.
- 1-2 alternate strategies named for a failure mode the cases may select (e.g. `degraded`:
  slow/stale/erroring backend as value-level error payloads; `empty`: no results). Only
  describe the tools whose behaviour changes.
Every response and fallback_response must be valid JSON text conforming to the tool's
output_schema; example args must conform to args_schema. No prose outside the fields."""


def _generic_strategy_entry(tool: dict) -> dict:
    from evalbuilder.mock_engine import generic_behavior

    return {"behavior": generic_behavior(tool), "examples": [], "fallback_response": _schema_fixture(tool)}


def author_strategies(gen: Generator, tools: list[dict], rules: dict[str, list[dict]] | None = None, guidance: str = "") -> tuple[dict, list[str]]:
    """Returns (`{world, strategies: {id: {description, tools: {name: {behavior, examples,
    fallback_response}}}}}`, problems). The `default` strategy always exists and covers every
    tool; fallbacks are validated against `output_schema` (replaced by a schema sample when
    invalid), examples too (dropped when invalid), unknown tools and id-less strategies dropped."""
    problems: list[str] = []
    by_name = {t["name"]: t for t in tools}
    try:
        result = gen.ask(
            STRATEGIES_SYSTEM,
            with_guidance(
                "TOOLS:\n" + json.dumps([tool_brief(t) for t in tools], indent=1, ensure_ascii=False)
                + "\n\nMOCK FIXTURES (deterministic rules the strategies must stay consistent with):\n"
                + json.dumps(rules or {}, ensure_ascii=False)[:6000],
                guidance,
            ),
            STRATEGIES_SCHEMA,
        )
        raw_strategies = [x for x in result.get("strategies", []) if isinstance(x, dict)]
        world = str(result.get("world") or "").strip()
    except Exception as e:  # noqa: BLE001 - deterministic default strategy
        problems.append(f"strategy generation failed, using a generic default strategy: {type(e).__name__}: {e}")
        raw_strategies, world = [], ""

    strategies: dict[str, dict] = {}
    for raw in raw_strategies:
        sid = str(raw.get("id") or "").strip()
        if not sid:
            problems.append("mock strategy without an id dropped")
            continue
        entry = {"description": str(raw.get("description") or "").strip(), "tools": {}}
        for raw_tool in raw.get("tools") or []:
            if not isinstance(raw_tool, dict):
                continue
            name = raw_tool.get("name")
            tool = by_name.get(name)
            if tool is None:
                problems.append(f"strategy {sid}: unknown tool {name!r} dropped")
                continue
            output_schema = tool.get("output_schema") or {}
            fallback = _parse_json(raw_tool.get("fallback_response"), allow_scalar=True)
            bad = tool_schemas.validate(fallback, output_schema) if fallback is not None else ["missing or invalid JSON"]
            if bad:
                problems.append(f"strategy {sid}: fallback_response for {name} replaced by a schema sample ({'; '.join(bad[:2])})")
                fallback = _schema_fixture(tool)
            examples = []
            for ex in raw_tool.get("examples") or []:
                if not isinstance(ex, dict):
                    continue
                args = _parse_json(ex.get("args"))
                response = _parse_json(ex.get("response"), allow_scalar=True)
                if not isinstance(args, dict) or response is None:
                    problems.append(f"strategy {sid}: example for {name} dropped (invalid JSON)")
                    continue
                bad = _present_arg_problems(args, tool) + tool_schemas.validate(response, output_schema)
                if bad:
                    problems.append(f"strategy {sid}: example for {name} dropped (schema): {'; '.join(bad[:2])}")
                    continue
                examples.append({"args": args, "response": response})
            entry["tools"][name] = {
                "behavior": str(raw_tool.get("behavior") or "").strip() or _generic_strategy_entry(tool)["behavior"],
                "examples": examples,
                "fallback_response": fallback,
            }
        strategies[sid] = entry

    default = strategies.setdefault("default", {"description": "healthy backend (generic)", "tools": {}})
    for tool in tools:
        if tool["name"] not in default["tools"]:
            if raw_strategies:
                problems.append(f"default strategy has no behaviour for {tool['name']}; using the tool description")
            default["tools"][tool["name"]] = _generic_strategy_entry(tool)
    ordered = {"default": default, **{k: v for k, v in strategies.items() if k != "default"}}
    return {"world": world, "strategies": ordered}, problems


def strategy_brief(strategies: dict | None) -> str:
    """The strategies as the case/scenario prompts see them (ids, descriptions, behaviours)."""
    if not strategies or not strategies.get("strategies"):
        return ""
    rows = {
        sid: {"description": s.get("description"), "tools": {n: t.get("behavior") for n, t in (s.get("tools") or {}).items()}}
        for sid, s in strategies["strategies"].items()
    }
    return (
        "\n\nMOCK STRATEGIES (backend behaviours the LLM mock engine follows for calls no fixture answers; "
        "world: " + str(strategies.get("world") or "")[:600] + "). Select one per case/scenario with "
        "`mock_strategy` ONLY when its failure mode needs it; '' keeps the dataset default:\n"
        + json.dumps(rows, indent=1, ensure_ascii=False)[:6000]
    )


def _parse_json(text: Any, allow_scalar: bool = False):
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return text if allow_scalar and text.strip() else None
    if isinstance(value, (dict, list)) or allow_scalar:
        return value
    return None


# ── case authoring ─────────────────────────────────────────────

CASES_SCHEMA = {
    "title": "cases",
    "type": "object",
    "properties": {
        "cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cell": {"type": "integer", "description": "index of the cell this case fills"},
                    "user_message": {"type": "string"},
                    "user_turns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "follow-up user messages for multi-turn cases (empty otherwise)",
                    },
                    "variant": {"type": "string", "enum": list(VARIANTS)},
                    "expected_tools": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "args": {"type": "string", "description": "JSON object of args that must match (subset); '{}' for any"},
                            },
                            "required": ["name", "args"],
                        },
                    },
                    "forbidden_tools": {"type": "array", "items": {"type": "string"}},
                    "contains": {
                        "type": "string",
                        "description": "a short literal the final answer must contain (only if guaranteed by fixtures); empty if none",
                    },
                    "contract": {"type": "string", "description": "one testable sentence a judge can verify"},
                    "expected_response": {"type": "string", "description": "what a correct final answer says"},
                    "mock_overrides": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {"type": "string"},
                                "match_args": {"type": "string", "description": "JSON object; '{}' for any"},
                                "response": {"type": "string", "description": "JSON text, e.g. an error payload"},
                            },
                            "required": ["tool", "match_args", "response"],
                        },
                        "description": "per-case mock rules (error injections, special fixtures)",
                    },
                    "mock_strategy": {"type": "string", "description": "id of the mock strategy for this case ('' = dataset default)"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "cell", "user_message", "user_turns", "variant", "expected_tools",
                    "forbidden_tools", "contains", "contract", "expected_response",
                    "mock_overrides", "evidence",
                ],
            },
        }
    },
    "required": ["cases"],
}

CASES_SYSTEM = """You author golden test cases for a LangGraph agent. Each case must be
answerable and testable given the MOCK FIXTURES (tools return exactly those payloads).

Rules:
- Fill every cell with exactly the requested number of cases (cell index given).
- inputs are realistic user messages; apply evolutions (concretizing, constraints,
  comparative, multicontext) so happy cases are not trivial. Use entity values that
  exist in the fixtures (ids, names) unless the cell tests unknown/invalid input.
- expected_tools: the ordered tool calls a correct agent makes (args as JSON subset).
  forbidden_tools: tools a correct agent must NOT call (e.g. side-effecting tools
  before confirmation).
- contains: only a literal guaranteed by fixtures (an id, a title); otherwise empty.
- contract: one sentence a judge can check against the final answer.
- Failure-mode cells: craft the input (or mock_overrides, e.g. an error payload for
  tool_error_handling / provider_error, injected instructions inside a fixture for
  prompt_injection) so the agent must handle the failure; the contract states the
  graceful behavior. Out-of-scope cells: requests the agent must decline politely.
- Multi-turn cases: first message in user_message, later ones in user_turns.
- Schema-edge cells (kind=schema-edge) probe one tool's declared input/output schema
  (`tool`, `edge`): missing_required → the user leaves out the named field; wrong_type →
  the user gives it in the wrong form (text for a number, a free-form date, a sentence
  for a boolean…); out_of_enum → a value outside the allowed set; boundary → a value
  past the declared limit; malformed_output → a normal, valid request (the tool's
  fixture is corrupted by the pipeline). The contract states the graceful behavior from
  `edge.expected_behavior`; a correct agent never passes non-conforming values to the
  tool or fabricates the missing data, so list that tool in forbidden_tools for
  missing/wrong/enum/boundary edges unless the agent can legitimately still call it.
- Skills: when the agent has skills, cases for scenarios that list them cite
  `skill:<name>`; with a `skill_loader` tool a correct agent reads the skill first, so
  put that call (e.g. load_skill with the skill name) at the start of expected_tools.
- evidence: cite the agent-map scenario id, failure type, skill:<name>, or schema:<tool>.<field>."""


def _cell_prompt(cells: list[Cell]) -> str:
    rows = []
    for idx, c in enumerate(cells):
        rows.append(
            {
                "cell": idx,
                "intent": c.intent,
                "related_intent": c.related_intent,
                "scenario": c.scenario,
                "failure_mode": c.failure_mode,
                "kind": c.kind,
                "topic": c.topic,
                "count": c.count,
                "of_which_multi_turn": c.multi_turn,
                "variants": c.variants,
            }
        )
        if c.tool or c.edge:
            rows[-1]["tool"] = c.tool
            rows[-1]["edge"] = {k: c.edge.get(k) for k in ("kind", "field", "detail", "expected_behavior")} if c.edge else None
    return json.dumps(rows, indent=1)


def author_cases(
    gen: Generator,
    cells: list[Cell],
    agent_map: AgentMap,
    constraints: list[str],
    mock_rules: dict[str, list[dict]],
    scenario_index: dict[str, dict],
    batch_size: int = 6,
    guidance: str = "",
    strategies: dict | None = None,
) -> tuple[list[dict], list[str]]:
    """Returns (case dicts ready for artifacts.add_case, problems)."""
    guide = _skill_reference("agent-eval-dataset", "references", "generation-guide.md")
    tools_by_name = {t["name"]: t for t in agent_map.tools}
    strategy_ids = list((strategies or {}).get("strategies") or {})
    context = with_guidance(
        agent_brief(agent_map, constraints)
        + "\n\nSCENARIOS:\n"
        + json.dumps(list(scenario_index.values()), indent=1, ensure_ascii=False)
        + "\n\nFAILURE TAXONOMY:\n"
        + json.dumps(FAILURE_TYPES, indent=1)
        + "\n\nMOCK FIXTURES (what tools will return):\n"
        + json.dumps(mock_rules, indent=1, ensure_ascii=False)[:8000]
        + strategy_brief(strategies)
        + ("\n\nGENERATION GUIDE:\n" + guide if guide else ""),
        guidance,
    )
    cases: list[dict] = []
    problems: list[str] = []
    pending = [c for c in cells if c.count > 0]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        got, errs = _author_batch(gen, batch, context, tools_by_name, strategy_ids=strategy_ids)
        problems += errs
        cases += got
        missing = _missing_in(batch, got)
        if missing:
            got2, errs2 = _author_batch(
                gen, missing, context, tools_by_name,
                note="These cells were missing or invalid in your previous answer; return only cases for them.",
                strategy_ids=strategy_ids,
            )
            problems += errs2
            cases += got2
            still = _missing_in(missing, got2)
            for c in still:
                problems.append(f"cell {c.key} short by {c.count} case(s) after repair")
    return cases, problems


def _missing_in(cells: list[Cell], cases: list[dict]) -> list[Cell]:
    out = []
    for c in cells:
        have = sum(1 for case in cases if case["metadata"]["_cell"] == c.key)
        if have < c.count:
            out.append(
                Cell(
                    intent=c.intent, scenario=c.scenario, failure_mode=c.failure_mode,
                    kind=c.kind, count=c.count - have, topic=c.topic, variants=c.variants,
                    multi_turn=max(0, c.multi_turn - sum(
                        1 for case in cases
                        if case["metadata"]["_cell"] == c.key and case["metadata"].get("user_turns")
                    )),
                    related_intent=c.related_intent, tool=c.tool, edge=c.edge,
                )
            )
    return out


def _author_batch(
    gen: Generator, cells: list[Cell], context: str, tools_by_name: dict[str, dict] | set, note: str = "",
    strategy_ids: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    if not isinstance(tools_by_name, dict):  # bare names (older callers/tests)
        tools_by_name = {n: {"name": n} for n in tools_by_name}
    tool_names = set(tools_by_name)
    user = context + "\n\nCELLS TO FILL:\n" + _cell_prompt(cells) + ("\n\n" + note if note else "")
    result = gen.ask(CASES_SYSTEM, user, CASES_SCHEMA)
    cases: list[dict] = []
    errors: list[str] = []
    for raw in result.get("cases", []):
        if not isinstance(raw, dict):
            continue
        idx = raw.get("cell")
        if not isinstance(idx, int) or not 0 <= idx < len(cells):
            errors.append(f"case with unknown cell index {idx!r} dropped")
            continue
        cell = cells[idx]
        message = str(raw.get("user_message") or "").strip()
        if not message:
            errors.append(f"empty user_message for cell {cell.key} dropped")
            continue
        expected_tools = []
        bad = False
        for et in raw.get("expected_tools") or []:
            name = et.get("name") if isinstance(et, dict) else None
            if name not in tool_names:
                errors.append(f"case for {cell.key} expects unknown tool {name!r}; dropped")
                bad = True
                break
            args = _parse_json(et.get("args"))
            args = args if isinstance(args, dict) else {}
            for problem in _present_arg_problems(args, tools_by_name[name]):
                errors.append(f"case for {cell.key}: expected_tools {name} args violate args_schema: {problem}")
            expected_tools.append({"name": name, "args": args})
        if bad:
            continue
        forbidden = [t for t in raw.get("forbidden_tools") or [] if t in tool_names]
        mocks: dict[str, list[dict]] = {}
        for ov in raw.get("mock_overrides") or []:
            if not isinstance(ov, dict) or ov.get("tool") not in tool_names:
                errors.append(f"mock override for unknown tool in {cell.key} ignored")
                continue
            match_args = _parse_json(ov.get("match_args"))
            response = _parse_json(ov.get("response"), allow_scalar=True)
            if response is None:
                errors.append(f"mock override for {ov['tool']} in {cell.key} ignored (invalid JSON)")
                continue
            mocks.setdefault(ov["tool"], []).append(
                {"matchArgs": match_args if isinstance(match_args, dict) else {}, "response": response}
            )
        if cell.edge and cell.edge.get("kind") == "malformed_output" and cell.tool:
            # The pipeline corrupts the fixture itself so the edge is always exercised.
            output_schema = tools_by_name.get(cell.tool, {}).get("output_schema") or {}
            mocks[cell.tool] = [{"matchArgs": {}, "response": tool_schemas.corrupt(output_schema, "missing_required")}]
        variant = raw.get("variant") if raw.get("variant") in VARIANTS else cell.variants[0]
        user_turns = [str(t) for t in raw.get("user_turns") or [] if str(t).strip()]
        if user_turns:
            variant = "multi-turn"
        reference: dict[str, Any] = {
            "response": str(raw.get("expected_response") or ""),
            "expected_tools": expected_tools,
        }
        if forbidden:
            reference["forbidden_tools"] = forbidden
        if str(raw.get("contains") or "").strip():
            reference["contains"] = str(raw["contains"]).strip()
        if str(raw.get("contract") or "").strip():
            reference["contract"] = str(raw["contract"]).strip()
        metadata: dict[str, Any] = {
            "intent": cell.intent,
            "topic": cell.topic,
            "scenario": cell.scenario,
            "failure_mode": cell.failure_mode,
            "variant": variant,
            "source": "synthetic",
            "evidence": [str(e) for e in raw.get("evidence") or []] or [cell.scenario],
            "_cell": cell.key,
        }
        if cell.related_intent:
            metadata["related_intent"] = cell.related_intent
        if cell.tool:
            metadata["tool"] = cell.tool
        if cell.edge:
            metadata["edge"] = {k: cell.edge.get(k) for k in ("id", "kind", "field")}
        if user_turns:
            metadata["user_turns"] = user_turns
        if mocks:
            metadata["mocks"] = {"tools": mocks}
        chosen = str(raw.get("mock_strategy") or "").strip()
        if chosen:
            if chosen in (strategy_ids or []):
                metadata.setdefault("mocks", {})["strategy"] = chosen
            else:
                errors.append(f"case for {cell.key}: unknown mock strategy {chosen!r} ignored (known: {strategy_ids or []})")
        cases.append(
            {
                "inputs": {"messages": [{"role": "user", "content": message}]},
                "reference_outputs": reference,
                "metadata": metadata,
            }
        )
    # Trim overflow per cell so the plan, not the model, decides the size.
    kept: list[dict] = []
    per_cell: dict[str, int] = {}
    limits = {c.key: c.count for c in cells}
    for case in cases:
        key = case["metadata"]["_cell"]
        if per_cell.get(key, 0) >= limits.get(key, 0):
            continue
        per_cell[key] = per_cell.get(key, 0) + 1
        kept.append(case)
    return kept, errors


# ── self review ────────────────────────────────────────────────

REVIEW_SCHEMA = {
    "title": "case_review",
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["approve", "reject"]},
                    "reason": {"type": "string"},
                },
                "required": ["id", "verdict", "reason"],
            },
        }
    },
    "required": ["reviews"],
}

REVIEW_SYSTEM = """You are a strict reviewer of generated eval cases. Reject a case when:
the input is ambiguous or unanswerable with the fixtures; the expected tools or contract
contradict the agent's prompts; the coverage cell (intent / scenario / failure_mode)
does not match what the input actually tests; or the `contains` literal is not
guaranteed by the fixtures. Otherwise approve. Give a one-line reason each."""


def self_review(
    gen: Generator, cases: list[dict], agent_map: AgentMap, mock_rules: dict, constraints: list[str], guidance: str = ""
) -> dict[str, dict]:
    """case id -> {verdict, reason}. Cases without a verdict are treated as approved."""
    if not cases:
        return {}
    payload = [
        {
            "id": c["id"],
            "input": c["inputs"],
            "user_turns": c["metadata"].get("user_turns", []),
            "reference_outputs": c["reference_outputs"],
            "cell": {k: c["metadata"].get(k) for k in ("intent", "scenario", "failure_mode", "variant")},
            "mocks": c["metadata"].get("mocks"),
        }
        for c in cases
    ]
    user = (
        agent_brief(agent_map, constraints)
        + "\n\nMOCK FIXTURES:\n"
        + json.dumps(mock_rules, ensure_ascii=False)[:6000]
        + "\n\nCASES:\n"
        + json.dumps(payload, indent=1, ensure_ascii=False)
    )
    result = gen.ask(REVIEW_SYSTEM, with_guidance(user, guidance), REVIEW_SCHEMA)
    out: dict[str, dict] = {}
    for r in result.get("reviews", []):
        if isinstance(r, dict) and r.get("id"):
            out[r["id"]] = {"verdict": r.get("verdict", "approve"), "reason": r.get("reason", "")}
    return out


# ── simulation scenarios ───────────────────────────────────────

SCENARIOS_SCHEMA = {
    "title": "simulation_scenarios",
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "intent": {"type": "string"},
                    "persona": {"type": "string"},
                    "goal": {"type": "string"},
                    "opening": {"type": "string"},
                    "followups": {"type": "array", "items": {"type": "string"}},
                    "max_turns": {"type": "integer"},
                    "success_contains": {"type": "string", "description": "literal that marks success; empty if none"},
                    "expect_contains": {"type": "string"},
                    "expect_not_contains": {"type": "string"},
                    "mock_strategy": {"type": "string", "description": "id of the mock strategy for this scenario ('' = dataset default)"},
                },
                "required": ["id", "intent", "persona", "goal", "opening", "followups", "max_turns",
                             "success_contains", "expect_contains", "expect_not_contains"],
            },
        }
    },
    "required": ["scenarios"],
}

SCENARIOS_SYSTEM = """Write one multi-turn conversation scenario per intent for simulation:
a persona, a goal, an opening message, 1-3 scripted follow-ups (including a correction
or a confirmation where the agent must ask for one), max_turns 3-5, and expectations:
success_contains (a literal in a successful final reply, using fixture values) and/or
expect_not_contains (text that would indicate a violated constraint). Use fixture ids."""


def author_scenarios(
    gen: Generator, agent_map: AgentMap, constraints: list[str], mock_rules: dict, guidance: str = "",
    strategies: dict | None = None,
) -> tuple[list[dict], list[str]]:
    strategy_ids = list((strategies or {}).get("strategies") or {})
    user = (
        agent_brief(agent_map, constraints)
        + "\n\nINTENTS:\n"
        + json.dumps(agent_map.intents, ensure_ascii=False)
        + "\n\nMOCK FIXTURES:\n"
        + json.dumps(mock_rules, ensure_ascii=False)[:6000]
        + strategy_brief(strategies)
    )
    result = gen.ask(SCENARIOS_SYSTEM, with_guidance(user, guidance), SCENARIOS_SCHEMA)
    scenarios: list[dict] = []
    problems: list[str] = []
    for raw in result.get("scenarios", []):
        if not isinstance(raw, dict) or not raw.get("id") or not raw.get("opening"):
            problems.append("simulation scenario without id/opening dropped")
            continue
        expect = {}
        if raw.get("expect_contains"):
            expect["contains"] = raw["expect_contains"]
        if raw.get("expect_not_contains"):
            expect["not_contains"] = raw["expect_not_contains"]
        if not raw.get("success_contains") and not expect:
            problems.append(f"scenario {raw['id']} has no success/expect condition; dropped")
            continue
        scenario = {
            "id": re.sub(r"[^a-z0-9._-]+", "-", str(raw["id"]).lower()),
            "intent": raw.get("intent"),
            "persona": raw.get("persona", "a user"),
            "goal": raw.get("goal", ""),
            "opening": raw["opening"],
            "followups": [str(f) for f in raw.get("followups") or []],
            "max_turns": max(2, min(int(raw.get("max_turns") or 4), 6)),
        }
        if raw.get("success_contains"):
            scenario["success_contains"] = raw["success_contains"]
        if expect:
            scenario["expect"] = expect
        chosen = str(raw.get("mock_strategy") or "").strip()
        if chosen:
            if chosen in strategy_ids:
                scenario["mock_strategy"] = chosen
            else:
                problems.append(f"scenario {scenario['id']}: unknown mock strategy {chosen!r} ignored (known: {strategy_ids})")
        scenarios.append(scenario)
    return scenarios, problems


# ── analysis ───────────────────────────────────────────────────

ANALYSIS_SCHEMA = {
    "title": "analysis",
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "verdict_explanation": {"type": "string"},
        "failure_patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "affected_cases": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["pattern", "affected_cases", "evidence", "severity"],
            },
        },
        "weak_slices": {"type": "array", "items": {"type": "string"}},
        "stability_notes": {"type": "string"},
        "evaluator_issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "problems that are evaluator/mock/pipeline issues, not agent defects",
        },
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "verdict_explanation", "failure_patterns", "weak_slices",
                 "stability_notes", "evaluator_issues", "recommendations"],
}

ANALYSIS_SYSTEM = """You analyze an agent evaluation. Separate three kinds of facts: agent
defects (wrong behavior on good cases), evaluator/mock problems (bad references, judge
errors, fixture gaps), and instability (differs across repeats). Cite case ids. Be
concrete and short; recommendations must be actionable changes to prompts, tools,
routing, or the eval itself."""


def author_analysis(gen: Generator, summary: dict) -> dict:
    user = "EVALUATION SUMMARY (JSON):\n" + json.dumps(summary, indent=1, ensure_ascii=False)[:60000]
    return gen.ask(ANALYSIS_SYSTEM, user, ANALYSIS_SCHEMA)


def deterministic_analysis(summary: dict, reason: str) -> dict:
    """Fallback when the generator is unavailable: facts only, no interpretation."""
    failing = summary.get("failing_cases", [])
    return {
        "summary": f"Automated analysis unavailable ({reason}); listing facts only.",
        "verdict_explanation": f"verdict={summary.get('verdict')} overall_score={summary.get('overall_score')}",
        "failure_patterns": [
            {
                "pattern": f"{len(failing)} case(s) below threshold",
                "affected_cases": [c.get("id") for c in failing][:50],
                "evidence": "see cases[].scores",
                "severity": "high" if failing else "low",
            }
        ] if failing else [],
        "weak_slices": summary.get("weak_slices", []),
        "stability_notes": f"unstable cases: {len(summary.get('unstable_cases', []))}",
        "evaluator_issues": summary.get("evaluator_errors", []),
        "recommendations": [],
    }
