"""Coverage plan: config × agent map → the cells the dataset must fill."""

from __future__ import annotations

from dataclasses import dataclass, field

from evalbuilder.pipeline.config import CoverageConfig
from evalbuilder.schemas import AgentMap

OUT_OF_SCOPE = "out_of_scope"
CROSS_CUTTING = "cross-cutting"


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
                for topic in topics:
                    cells.append(
                        Cell(
                            intent=iid,
                            scenario=scenario["id"],
                            failure_mode=scenario.get("failure_mode", "none") if kind == "failure" else "none",
                            kind=kind,
                            count=n,
                            topic=topic,
                            variants=["happy"] if kind == "happy" else ["boundary", "adversarial"],
                        )
                    )

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


def achieved(cells: list[Cell], cases: list[dict]) -> dict:
    """Compare planned counts with what the dataset actually holds (approved only)."""
    have: dict[str, int] = {}
    multi = 0
    for case in cases:
        md = case.get("metadata", {})
        key = "/".join(
            str(md.get(k, "unspecified" if k != "failure_mode" else "none"))
            for k in ("intent", "topic", "scenario", "failure_mode")
        )
        have[key] = have.get(key, 0) + 1
        if md.get("user_turns"):
            multi += 1
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
    return {
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
