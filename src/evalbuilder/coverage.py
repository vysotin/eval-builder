"""Coverage grid (intent x topic x scenario x failure_mode) and gap math."""

from __future__ import annotations

from evalbuilder.schemas import AgentMap, Dataset

CELL_KEYS = ("intent", "topic", "scenario", "failure_mode")


def required_cells(agent_map: AgentMap) -> list[dict]:
    topics = agent_map.data_domains.get("topics") or ["unspecified"]
    cells: list[dict] = []
    for scenario in agent_map.scenarios:
        for topic in topics:
            cells.append(
                {
                    "intent": scenario.get("intent", "unspecified"),
                    "topic": topic,
                    "scenario": scenario["id"],
                    "failure_mode": "none",
                }
            )
    for failure in agent_map.failure_scenarios:
        cells.append(
            {
                "intent": "cross-cutting",
                "topic": "unspecified",
                "scenario": "failure",
                "failure_mode": failure["failure_type"],
            }
        )
    return cells


def _cell_key(cell: dict) -> str:
    return "/".join(str(cell.get(k, "unspecified")) for k in CELL_KEYS)


def coverage_gaps(ds: Dataset, agent_map: AgentMap, target_per_cell: int = 1) -> dict:
    from evalbuilder.pipeline.planning import skills_of_case

    required = required_cells(agent_map)
    have: dict[str, int] = {}
    skill_names = [s.get("name") for s in (getattr(agent_map, "skills", None) or []) if s.get("name")]
    scenario_skills = {s.get("id"): list(s.get("skills") or []) for s in agent_map.scenarios}
    per_skill = {name: 0 for name in skill_names}
    for case in ds.cases:
        key = _cell_key(case.metadata)
        have[key] = have.get(key, 0) + 1
        for name in skills_of_case(case.model_dump(), scenario_skills, skill_names):
            per_skill[name] += 1

    gaps: list[dict] = []
    covered = 0
    for cell in required:
        count = have.get(_cell_key(cell), 0)
        if count >= target_per_cell:
            covered += 1
        else:
            gaps.append({**cell, "missing": target_per_cell - count})

    out = {
        "required_cells": len(required),
        "covered_cells": covered,
        "target_per_cell": target_per_cell,
        "gaps": gaps,
    }
    if skill_names:
        out["skills"] = {name: {"cases": n} for name, n in per_skill.items()}
        out["uncovered_skills"] = [name for name, n in per_skill.items() if n == 0]
    return out
