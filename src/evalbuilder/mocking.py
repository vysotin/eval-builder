"""ADK-style tool mocking: ordered rules, subset matchArgs, wildcard, miss policies."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool


class MockMissError(Exception):
    def __init__(self, tool_name: str, args: dict, rules: list[dict]):
        self.tool_name = tool_name
        self.args = args
        super().__init__(
            f"no mock rule matched tool {tool_name!r} with args {args!r} "
            f"(strict mode; {len(rules)} rule(s) declared)"
        )


def match_rule(rules: list[dict], args: dict) -> dict | None:
    """First rule whose matchArgs is a subset of args; {} matches anything."""
    for rule in rules:
        match_args = rule.get("matchArgs", {})
        if not match_args:
            return rule
        if all(args.get(k) == v for k, v in match_args.items()):
            return rule
    return None


def wrap_tool(
    tool: BaseTool,
    rules: list[dict],
    on_miss: str = "real",
    fallback=None,
) -> StructuredTool:
    """Wrap a tool so matching mock rules answer instead of the real function.

    The wrapped tool keeps the original name/description/args_schema so the
    model's tool-selection behavior is unchanged.
    """
    if on_miss not in ("real", "fallback", "strict"):
        raise ValueError(f"invalid on_miss policy {on_miss!r}")

    def _mocked(**kwargs):
        rule = match_rule(rules, kwargs)
        if rule is not None:
            return rule["response"]
        if on_miss == "real":
            return tool.invoke(kwargs)
        if on_miss == "strict":
            raise MockMissError(tool.name, kwargs, rules)
        return fallback

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
) -> list[BaseTool]:
    """Wrap only the tools that have rules; others pass through untouched."""
    return [
        wrap_tool(t, rules_by_tool[t.name], on_miss=on_miss, fallback=fallback)
        if t.name in rules_by_tool
        else t
        for t in tools
    ]


def with_fallback(case_rules: list[dict], dataset_rules: list[dict]) -> list[dict]:
    """Case-level rules first, then the dataset-level rules unless the case already
    ends with a wildcard — so a per-case error injection for one arg set keeps the
    tool answerable for every other call."""
    rules = list(case_rules)
    if rules and not rules[-1].get("matchArgs"):
        return rules
    return rules + [r for r in dataset_rules if r not in rules]


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
