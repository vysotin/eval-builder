"""Coverage plan: config × agent map → the cells the dataset must fill."""

from __future__ import annotations

from dataclasses import dataclass, field

from evalbuilder.pipeline.config import CoverageConfig
from evalbuilder.schemas import AgentMap

OUT_OF_SCOPE = "out_of_scope"
CROSS_CUTTING = "cross-cutting"
SCHEMA_EDGE = "schema-edge"


def _intent_for_tool(agent_map: AgentMap, tool: dict, intents: list[str], index: int) -> str | None:
    """The intent whose scenarios cite the tool (evidence `tool:<name>`), else rotate."""
    token = f"tool:{tool.get('name')}"
    for s in agent_map.scenarios:
        if token in (s.get("evidence") or []) and s.get("intent") in intents:
            return s["intent"]
    for i in agent_map.intents:
        if token in (i.get("evidence") or []):
            return i["id"]
    return intents[index % len(intents)] if intents else None


@dataclass
class Cell:
    intent: str
    scenario: str
    failure_mode: str
    kind: str  # happy | failure | category | out-of-intent
    count: int
    topic: str = "unspecified"
    variants: list[str] = field(default_factory=list)
    multi_turn: int = 0
    related_intent: str | None = None
    tool: str | None = None  # schema-edge cells: the tool whose schema is probed
    edge: dict | None = None  # schema-edge cells: {kind, field, detail, expected_behavior, …}

    @property
    def key(self) -> str:
        return f"{self.intent}/{self.topic}/{self.scenario}/{self.failure_mode}"

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "topic": self.topic,
            "scenario": self.scenario,
            "failure_mode": self.failure_mode,
            "kind": self.kind,
            "count": self.count,
            "variants": list(self.variants),
            "multi_turn": self.multi_turn,
            "related_intent": self.related_intent,
            "tool": self.tool,
            "edge": dict(self.edge) if self.edge else None,
        }


def _spread(total: int, buckets: int) -> list[int]:
    if buckets <= 0:
        return []
    base, extra = divmod(max(total, buckets), buckets)
    return [base + (1 if i < extra else 0) for i in range(buckets)]


def plan_cells(
    cfg: CoverageConfig, agent_map: AgentMap, failure_types: list[str]
) -> list[Cell]:
    topics = agent_map.data_domains.get("topics") or ["unspecified"]
    cells: list[Cell] = []
    topic_cursor = 0  # topics rotate across cells; they never multiply the case count

    for intent in agent_map.intents:
        iid = intent["id"]
        for kind, per in (("happy", cfg.per_intent.happy), ("failure", cfg.per_intent.failure)):
            scenarios = [
                s for s in agent_map.scenarios
                if s.get("intent") == iid and s.get("kind", "happy") == kind
            ]
            if not scenarios:
                continue
            for scenario, n in zip(scenarios, _spread(per, len(scenarios))):
                cells.append(
                    Cell(
                        intent=iid,
                        scenario=scenario["id"],
                        failure_mode=scenario.get("failure_mode", "none") if kind == "failure" else "none",
                        kind=kind,
                        count=n,
                        topic=topics[topic_cursor % len(topics)],
                        variants=["happy"] if kind == "happy" else ["boundary", "adversarial"],
                    )
                )
                topic_cursor += 1

    if cfg.out_of_intent > 0 and OUT_OF_SCOPE not in failure_types:
        failure_types = [*failure_types, OUT_OF_SCOPE]  # always applicable
    intents = [i["id"] for i in agent_map.intents]
    for i, ftype in enumerate(failure_types):
        if ftype == OUT_OF_SCOPE:
            count = max(cfg.per_failure_category, cfg.out_of_intent)
        else:
            count = cfg.per_failure_category
        if count <= 0:
            continue
        cells.append(
            Cell(
                intent=CROSS_CUTTING,
                scenario="failure",
                failure_mode=ftype,
                kind="out-of-intent" if ftype == OUT_OF_SCOPE else "category",
                count=count,
                variants=["adversarial", "boundary"],
                related_intent=intents[i % len(intents)] if intents else None,
            )
        )

    # Schema edge cells: per tool, the first N deterministic edge cases (missing / wrong /
    # out-of-range input, malformed output) derived from its args/output schemas.
    per_tool = int(getattr(cfg, "per_tool_edge_cases", 0) or 0)
    if per_tool > 0:
        for j, tool in enumerate(agent_map.tools):
            for edge in (tool.get("edge_cases") or [])[:per_tool]:
                cells.append(
                    Cell(
                        intent=CROSS_CUTTING,
                        scenario=edge["id"],
                        failure_mode=edge.get("failure_mode", "input_validation"),
                        kind=SCHEMA_EDGE,
                        count=1,
                        variants=["boundary", "adversarial"],
                        related_intent=_intent_for_tool(agent_map, tool, intents, j),
                        tool=tool["name"],
                        edge=edge,
                    )
                )

    # Floor on total size: extra happy cases spread over intent happy cells.
    planned = sum(c.count for c in cells)
    happy_cells = [c for c in cells if c.kind == "happy"]
    if planned < cfg.total_cases and happy_cells:
        for c, extra in zip(happy_cells, _spread(cfg.total_cases - planned, len(happy_cells))):
            c.count += extra
            c.variants = ["happy", "linguistic", "boundary"]

    # Multi-turn share: flag a slice of intent cases as multi-turn conversations.
    total = sum(c.count for c in cells)
    wanted = round(total * cfg.multi_turn_share)
    for c in sorted((c for c in cells if c.kind in ("happy", "failure")), key=lambda c: -c.count):
        if wanted <= 0:
            break
        take = min(c.count, max(1, wanted // 2 or 1))
        c.multi_turn = take
        wanted -= take
    return cells


def summarize_plan(cells: list[Cell]) -> dict:
    by_kind: dict[str, int] = {}
    for c in cells:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + c.count
    return {
        "cells": len(cells),
        "cases": sum(c.count for c in cells),
        "multi_turn": sum(c.multi_turn for c in cells),
        "by_kind": by_kind,
    }


def skills_of_case(case: dict, scenario_skills: dict[str, list[str]], skill_names: list[str]) -> list[str]:
    """Skills a case exercises: its scenario lists them, its evidence cites `skill:<name>`,
    or an expected tool call loads the skill by name."""
    md = case.get("metadata", {}) or {}
    found = list(scenario_skills.get(str(md.get("scenario")), []))
    for token in md.get("evidence") or []:
        if isinstance(token, str) and token.startswith("skill:") and token[6:] in skill_names:
            found.append(token[6:])
    for call in (case.get("reference_outputs") or {}).get("expected_tools") or []:
        name = (call.get("args") or {}).get("name") if isinstance(call, dict) else None
        if isinstance(name, str) and name in skill_names and str(call.get("name", "")).endswith("skill"):
            found.append(name)
    return list(dict.fromkeys(found))


def achieved(cells: list[Cell], cases: list[dict], skills: list[dict] | None = None, scenarios: list[dict] | None = None) -> dict:
    """Compare planned counts with what the dataset actually holds (approved only).
    With `skills` (agent-map entries) the result also counts the cases per skill."""
    have: dict[str, int] = {}
    multi = 0
    skill_names = [s.get("name") for s in (skills or []) if s.get("name")]
    scenario_skills = {s.get("id"): list(s.get("skills") or []) for s in (scenarios or [])}
    per_skill = {name: 0 for name in skill_names}
    for case in cases:
        md = case.get("metadata", {})
        key = "/".join(
            str(md.get(k, "unspecified" if k != "failure_mode" else "none"))
            for k in ("intent", "topic", "scenario", "failure_mode")
        )
        have[key] = have.get(key, 0) + 1
        if md.get("user_turns"):
            multi += 1
        for name in skills_of_case(case, scenario_skills, skill_names):
            per_skill[name] += 1
    gaps = []
    by_kind_planned: dict[str, int] = {}
    by_kind_have: dict[str, int] = {}
    for c in cells:
        count = have.get(c.key, 0)
        by_kind_planned[c.kind] = by_kind_planned.get(c.kind, 0) + c.count
        by_kind_have[c.kind] = by_kind_have.get(c.kind, 0) + min(count, c.count)
        if count < c.count:
            gaps.append({**c.to_dict(), "have": count, "missing": c.count - count})
    planned_total = sum(c.count for c in cells)
    covered_total = sum(by_kind_have.values())
    out = {
        "planned": planned_total,
        "covered": covered_total,
        "coverage_pct": round(100.0 * covered_total / planned_total, 1) if planned_total else 100.0,
        "by_kind": {
            k: {"planned": by_kind_planned[k], "covered": by_kind_have.get(k, 0)}
            for k in by_kind_planned
        },
        "multi_turn": {"planned": sum(c.multi_turn for c in cells), "have": multi},
        "gaps": gaps,
    }
    if skill_names:
        out["skills"] = {name: {"cases": n} for name, n in per_skill.items()}
        out["uncovered_skills"] = [name for name, n in per_skill.items() if n == 0]
    return out
