"""Tool mocking, layer 1: ordered rules, subset matchArgs, wildcard, miss policies.

A wrapped tool answers from its rules first (first match wins, `{}` is a wildcard). On a
miss the policy decides: `real` (call the tool), `fallback` (a canned value), `strict`
(raise `MockMissError`), or `llm` — hand the call to an `LLMMockEngine`
(`mock_engine.py`), the second layer. Every answered call can be appended to a ledger
(`layer: rule | llm | real | fallback | error`). Skill loaders are never wrapped.
"""

from __future__ import annotations

import copy

from langchain_core.tools import BaseTool, StructuredTool

from evalbuilder.skills import is_skill_loader

ON_MISS_POLICIES = ("real", "fallback", "strict", "llm")


def mockable(tool: BaseTool) -> bool:
    """Whether the mock layers may answer for a tool: skill loaders (local, deterministic
    reads of SKILL.md) are never wrapped."""
    return not is_skill_loader(tool)


class MockMissError(Exception):
    def __init__(self, tool_name: str, args: dict, rules: list[dict]):
        self.tool_name = tool_name
        self.args = args
        super().__init__(
            f"no mock rule matched tool {tool_name!r} with args {args!r} "
            f"(strict mode; {len(rules)} rule(s) declared)"
        )


def args_subset(expected: dict, actual: dict) -> bool:
    """`expected` ⊆ `actual`, recursively for nested objects (pydantic-model arguments):
    every expected key must be present with an equal value, or a nested subset."""
    if not isinstance(actual, dict):
        return False
    for key, value in (expected or {}).items():
        if key not in actual:
            return False
        if isinstance(value, dict) and isinstance(actual[key], dict):
            if not args_subset(value, actual[key]):
                return False
        elif actual[key] != value:
            return False
    return True


def match_index(rules: list[dict], args: dict) -> int | None:
    """Index of the first rule whose matchArgs is a (recursive) subset of args; {} matches anything."""
    for index, rule in enumerate(rules):
        match_args = rule.get("matchArgs", {})
        if not match_args or args_subset(match_args, args):
            return index
    return None


def match_rule(rules: list[dict], args: dict) -> dict | None:
    """First rule whose matchArgs is a (recursive) subset of args; {} matches anything."""
    index = match_index(rules, args)
    return rules[index] if index is not None else None


def wrap_tool(
    tool: BaseTool,
    rules: list[dict],
    on_miss: str = "real",
    fallback=None,
    *,
    engine=None,
    ledger: list[dict] | None = None,
) -> StructuredTool:
    """Wrap a tool so matching mock rules answer instead of the real function.

    The wrapped tool keeps the original name/description/args_schema so the model's
    tool-selection behavior is unchanged. `on_miss="llm"` needs an `engine`
    (`mock_engine.LLMMockEngine`); `ledger` collects one entry per call.
    """
    if on_miss not in ON_MISS_POLICIES:
        raise ValueError(f"invalid on_miss policy {on_miss!r}; use one of {ON_MISS_POLICIES}")
    if on_miss == "llm" and engine is None:
        raise ValueError("on_miss='llm' needs a mock engine (mock_engine.LLMMockEngine)")

    def _record(entry: dict) -> None:
        if ledger is not None:
            ledger.append(entry)

    def _mocked(**kwargs):
        index = match_index(rules, kwargs)
        if index is not None:
            response = copy.deepcopy(rules[index]["response"])
            _record({"tool": tool.name, "args": kwargs, "layer": "rule", "rule": index, "response": copy.deepcopy(response)})
            return response
        if on_miss == "llm":
            try:
                response = engine.respond(tool.name, kwargs)
            except Exception as e:  # noqa: BLE001 - the engine already logged the failed entry
                if engine.ledger:
                    _record(dict(engine.ledger[-1]))
                else:
                    _record({"tool": tool.name, "args": kwargs, "layer": "error", "error": f"{type(e).__name__}: {e}"})
                raise
            _record(dict(engine.ledger[-1]))
            return response
        if on_miss == "real":
            response = tool.invoke(kwargs)
            _record({"tool": tool.name, "args": kwargs, "layer": "real", "response": response})
            return response
        if on_miss == "strict":
            error = MockMissError(tool.name, kwargs, rules)
            _record({"tool": tool.name, "args": kwargs, "layer": "error", "error": str(error)})
            raise error
        _record({"tool": tool.name, "args": kwargs, "layer": "fallback", "response": fallback})
        return copy.deepcopy(fallback)

    return StructuredTool.from_function(
        func=_mocked,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def merge_mock_rules(dataset_mocks: dict, case_mocks: dict) -> dict[str, list[dict]]:
    """Per tool name, case-level rules replace dataset-level rules."""
    merged = {name: list(rules) for name, rules in (dataset_mocks or {}).items()}
    for name, rules in (case_mocks or {}).items():
        merged[name] = list(rules)
    return merged


def wrap_tools(
    tools: list[BaseTool],
    rules_by_tool: dict[str, list[dict]],
    on_miss: str = "real",
    fallback=None,
    *,
    engine=None,
    ledger: list[dict] | None = None,
) -> list[BaseTool]:
    """Wrap the tools that have rules — and, when an engine is given, every mockable
    tool (the engine answers calls no rule covers). Skill loaders pass through untouched."""
    if on_miss == "llm" and engine is None:
        raise ValueError("on_miss='llm' needs a mock engine (mock_engine.LLMMockEngine)")
    out: list[BaseTool] = []
    for t in tools:
        if mockable(t) and (t.name in rules_by_tool or engine is not None):
            out.append(wrap_tool(t, rules_by_tool.get(t.name, []), on_miss=on_miss, fallback=fallback, engine=engine, ledger=ledger))
        else:
            out.append(t)
    return out


def with_fallback(case_rules: list[dict], dataset_rules: list[dict]) -> list[dict]:
    """Case-level rules first, then the dataset-level rules unless the case already
    ends with a wildcard — so a per-case error injection for one arg set keeps the
    tool answerable for every other call."""
    rules = list(case_rules)
    if rules and not rules[-1].get("matchArgs"):
        return rules
    return rules + [r for r in dataset_rules if r not in rules]


LEDGER_LAYERS = ("rule", "llm", "real", "fallback", "error")


def ledger_totals(entries: list[dict]) -> dict[str, int]:
    """Counts per layer plus `invalid` (LLM answers that failed validation, fallback or error)."""
    totals = {layer: 0 for layer in LEDGER_LAYERS}
    totals["invalid"] = 0
    for e in entries or []:
        layer = e.get("layer", "rule")
        if layer in totals:
            totals[layer] += 1
        if e.get("fallback") and layer != "fallback":
            totals["fallback"] += 1
        if e.get("error") and layer != "error":
            totals["error"] += 1
        if layer == "llm" and not e.get("valid", True):
            totals["invalid"] += 1
    return totals


def verify_dataset(ds, only_approved: bool = False) -> list[dict]:
    """Expected tool calls in mocked cases that no rule answers: [{case, tool, args}]."""
    misses: list[dict] = []
    for c in ds.cases:
        if only_approved and c.review.status != "approved":
            continue
        merged = merge_mock_rules(
            ds.mocks.get("tools", {}), c.metadata.get("mocks", {}).get("tools", {})
        )
        if not merged:
            continue
        for expected in c.reference_outputs.get("expected_tools", []):
            name = expected.get("name")
            if name in merged and match_rule(merged[name], expected.get("args", {})) is None:
                misses.append({"case": c.id, "tool": name, "args": expected.get("args", {})})
    return misses


def verify_summary(ds, only_approved: bool = False) -> dict:
    """`verify_dataset` read through the dataset's policy: under `on_miss: llm` a miss is
    answered by the mock engine (informational), under any other policy it is a defect."""
    policy = (ds.mocks or {}).get("on_miss", "real")
    misses = verify_dataset(ds, only_approved=only_approved)
    return {
        "policy": policy,
        "misses": [] if policy == "llm" else misses,
        "llm_answered": misses if policy == "llm" else [],
        "ok": policy == "llm" or not misses,
    }
