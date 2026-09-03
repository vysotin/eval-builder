"""LLM mock engine — the second mocking layer.

Layer 1 (`mocking.py`) answers tool calls from deterministic rules. When no rule matches
and the dataset's policy is `on_miss: llm`, the wrapped tool hands the call to an
`LLMMockEngine`, which plays the backend: it prompts a model with the shared *world*, the
selected *strategy*'s behaviour for that tool, the tool definition (description, args and
output schemas), the previous calls of the same conversation and the call itself, asks for
a structured answer, validates it against the tool's `output_schema`
(`tool_schemas.validate`), re-asks once with the problems listed, and finally either
returns the strategy's `fallback_response` or raises `MockEngineError`
(`on_invalid: fallback | strict`). Every call lands in `engine.ledger`.

Strategies are pre-generated in the pipeline's `mocks` stage (into `dataset.mocks.strategies`) and
embedded in the dataset; a case or a simulation scenario may select a strategy by id.
The shape mirrors ADK's user simulator (a described plan + a model named in config), applied
to tools instead of users.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from evalbuilder import tool_schemas

MOCK_RESPONSE_TITLE = "mock_response"
CALL_MARKER = "TOOL CALL:\n"
HISTORY_LIMIT = 8
RESPONSE_PREVIEW_CHARS = 800
DEFAULT_STRATEGY = "default"
ON_INVALID_POLICIES = ("fallback", "strict")

MOCK_ENGINE_SYSTEM = """You simulate the backend behind ONE tool of an agent under evaluation. You are not
the agent: you answer exactly like the real tool would, as data only.

Rules:
- Follow the WORLD and the TOOL BEHAVIOUR strictly; when they name ids, ranges, states
  or outcomes, honour them. Where they are silent, invent realistic, specific values.
- Stay consistent with PREVIOUS CALLS in this conversation (the same id resolves to the
  same entity; state only changes when a call changes it).
- The answer must conform to the OUTPUT SCHEMA exactly (every required field, right
  types, enum values). Without a schema, answer with `response_json`: the JSON text the
  tool would return.
- No explanations, no markdown, no fields the schema does not allow."""


class MockEngineError(Exception):
    """The engine could not produce a schema-conformant response (strict policy)."""

    def __init__(self, tool_name: str, args: dict, problems: list[str]):
        self.tool_name = tool_name
        self.args = args
        self.problems = list(problems)
        super().__init__(
            f"LLM mock for tool {tool_name!r} with args {args!r} is invalid after repair: "
            + "; ".join(self.problems[:4])
        )


# ── strategies ─────────────────────────────────────────────────


def generic_behavior(spec: dict) -> str:
    """Behaviour text when no strategy describes the tool: the tool's own contract."""
    description = (spec.get("description") or "").strip() or "no description"
    return (
        f"Answer like the real `{spec.get('name', 'tool')}` would — {description}. Return a realistic, "
        "specific, internally consistent successful response for these arguments; stay consistent "
        "with previous calls; conform to the output schema."
    )


def tool_strategy_entry(strategies: dict | None, strategy_id: str, tool_name: str, spec: dict) -> dict:
    """The behaviour/examples/fallback for a tool under a strategy: the strategy's own entry,
    else the default strategy's, else a generic behaviour derived from the tool definition."""
    table = (strategies or {}).get("strategies") or {}
    for sid in (strategy_id, DEFAULT_STRATEGY):
        entry = ((table.get(sid) or {}).get("tools") or {}).get(tool_name)
        if isinstance(entry, dict):
            return {
                "behavior": str(entry.get("behavior") or generic_behavior(spec)),
                "examples": [e for e in (entry.get("examples") or []) if isinstance(e, dict)],
                "fallback_response": entry.get("fallback_response"),
            }
    return {"behavior": generic_behavior(spec), "examples": [], "fallback_response": None}


def fallback_response(spec: dict, entry: dict) -> Any:
    """What a still-invalid answer is replaced with: the strategy's fallback, else a
    schema-conformant sample, else a generic payload."""
    if entry.get("fallback_response") is not None:
        return copy.deepcopy(entry["fallback_response"])
    output = spec.get("output_schema") or {}
    if output.get("type") == "object" and output.get("properties"):
        sample = tool_schemas.example(output)
        if isinstance(sample, dict):
            return sample
    return {"ok": True, "tool": spec.get("name"), "note": "generic fixture (mock engine fallback)"}


def validate_strategies(strategies: Any, tool_specs: dict[str, dict]) -> list[str]:
    """Problems of a strategies document (`{world, strategies: {id: {description, tools: {name:
    {behavior, examples, fallback_response}}}}}`) against the tools' schemas; [] = usable."""
    problems: list[str] = []
    if not isinstance(strategies, dict):
        return ["strategies must be an object with `world` and `strategies`"]
    if not isinstance(strategies.get("world", ""), str):
        problems.append("world must be a string")
    table = strategies.get("strategies")
    if not isinstance(table, dict) or not table:
        return problems + ["strategies must be a non-empty object keyed by strategy id"]
    if DEFAULT_STRATEGY not in table:
        problems.append(f"a {DEFAULT_STRATEGY!r} strategy is required")
    for sid, entry in table.items():
        if not isinstance(entry, dict):
            problems.append(f"strategy {sid}: must be an object")
            continue
        tools = entry.get("tools") or {}
        if not isinstance(tools, dict):
            problems.append(f"strategy {sid}: tools must be an object keyed by tool name")
            continue
        for name, t in tools.items():
            spec = tool_specs.get(name)
            if spec is None:
                problems.append(f"strategy {sid}: unknown tool {name!r}")
                continue
            if not isinstance(t, dict) or not str(t.get("behavior") or "").strip():
                problems.append(f"strategy {sid}: {name} needs a behavior text")
                continue
            output = spec.get("output_schema") or {}
            if "fallback_response" in t and t["fallback_response"] is not None:
                for err in tool_schemas.validate(t["fallback_response"], output):
                    problems.append(f"strategy {sid}: {name} fallback_response {err}")
            for i, ex in enumerate(t.get("examples") or []):
                if not isinstance(ex, dict) or not isinstance(ex.get("args"), dict) or "response" not in ex:
                    problems.append(f"strategy {sid}: {name} example {i} needs args (object) and response")
                    continue
                props = (spec.get("args_schema") or {}).get("properties") or {}
                defs = dict((spec.get("args_schema") or {}).get("$defs", {}))
                for key, value in ex["args"].items():
                    if key in props:
                        for err in tool_schemas.validate(value, props[key], defs, f"$.{key}", partial=True):
                            problems.append(f"strategy {sid}: {name} example {i} args {err}")
                for err in tool_schemas.validate(ex["response"], output):
                    problems.append(f"strategy {sid}: {name} example {i} response {err}")
    return problems


# ── prompt ─────────────────────────────────────────────────────


def _definition(spec: dict) -> dict:
    return {k: spec.get(k) for k in ("name", "description", "args_schema", "output_schema") if spec.get(k) is not None}


def _preview(value: Any) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(text) <= RESPONSE_PREVIEW_CHARS else text[:RESPONSE_PREVIEW_CHARS] + "…"


def build_prompt(
    tool_name: str, args: dict, spec: dict, world: str, strategy_id: str, strategy: dict,
    history: list[dict], problems: list[str],
) -> str:
    entry = tool_strategy_entry({"strategies": {strategy_id: strategy}}, strategy_id, tool_name, spec)
    parts = [
        "WORLD:\n" + (world or "(no shared world description)"),
        f"STRATEGY ({strategy_id}): " + str(strategy.get("description") or ""),
        "TOOL BEHAVIOUR:\n" + entry["behavior"],
        "EXAMPLES (args → response):\n" + json.dumps(entry["examples"], ensure_ascii=False, indent=1),
        "TOOL DEFINITION:\n" + json.dumps(_definition(spec), ensure_ascii=False, indent=1),
        "PREVIOUS CALLS IN THIS CONVERSATION:\n" + json.dumps(
            [{**h, "response": _preview(h.get("response"))} for h in history], ensure_ascii=False, indent=1
        ),
        CALL_MARKER + json.dumps({"tool": tool_name, "args": args, "strategy": strategy_id}, ensure_ascii=False, default=str),
    ]
    if problems:
        parts.append("PROBLEMS WITH YOUR PREVIOUS ANSWER (fix every one, return the complete answer):\n- " + "\n- ".join(problems))
    return "\n\n".join(parts)


def call_in_prompt(text: str) -> dict | None:
    """The `TOOL CALL` block of an engine prompt (for scripted mock models and tests)."""
    if CALL_MARKER not in text:
        return None
    line = text.rsplit(CALL_MARKER, 1)[1].split("\n", 1)[0]
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def response_schema(spec: dict) -> tuple[dict, bool]:
    """(structured-output schema, uses the `response_json` envelope?)."""
    output = spec.get("output_schema") or {}
    if output.get("type") == "object" and output.get("properties"):
        return {**{k: v for k, v in output.items() if k != "title"}, "title": MOCK_RESPONSE_TITLE}, False
    return {
        "title": MOCK_RESPONSE_TITLE,
        "type": "object",
        "properties": {"response_json": {"type": "string", "description": "the JSON text the tool returns"}},
        "required": ["response_json"],
    }, True


# ── the engine ─────────────────────────────────────────────────


class LLMMockEngine:
    """One engine per conversation (case or simulation scenario): it keeps the call history
    for consistency and a ledger of everything it answered."""

    def __init__(
        self,
        model,
        strategies: dict | None,
        tool_specs: dict[str, dict],
        *,
        strategy: str = DEFAULT_STRATEGY,
        on_invalid: str = "fallback",
        max_repairs: int = 1,
        history_limit: int = HISTORY_LIMIT,
        log: Callable[[str], None] | None = None,
    ):
        if on_invalid not in ON_INVALID_POLICIES:
            raise ValueError(f"invalid on_invalid policy {on_invalid!r}; use one of {ON_INVALID_POLICIES}")
        self.model = model
        self.strategies = strategies or {"world": "", "strategies": {}}
        self.tool_specs = tool_specs or {}
        self.on_invalid = on_invalid
        self.max_repairs = max(0, int(max_repairs))
        self.history_limit = history_limit
        self.log = log or (lambda msg: None)
        self.notes: list[str] = []
        table = self.strategies.get("strategies") or {}
        if table and strategy not in table:
            self.notes.append(f"unknown mock strategy {strategy!r}; using {DEFAULT_STRATEGY!r}")
            strategy = DEFAULT_STRATEGY
        self.strategy = strategy or DEFAULT_STRATEGY
        self.ledger: list[dict] = []
        self.history: list[dict] = []

    # ── strategy lookups ──
    def spec(self, tool_name: str) -> dict:
        return self.tool_specs.get(tool_name) or {"name": tool_name, "description": "", "args_schema": {}, "output_schema": {}}

    def tool_strategy(self, tool_name: str) -> dict:
        return tool_strategy_entry(self.strategies, self.strategy, tool_name, self.spec(tool_name))

    def fallback_for(self, tool_name: str) -> Any:
        return fallback_response(self.spec(tool_name), self.tool_strategy(tool_name))

    def stats(self) -> dict:
        return {
            "calls": len(self.ledger),
            "valid": sum(1 for e in self.ledger if e.get("valid")),
            "repaired": sum(1 for e in self.ledger if e.get("valid") and e.get("repairs")),
            "fallback": sum(1 for e in self.ledger if e.get("fallback")),
            "errors": sum(1 for e in self.ledger if e.get("error")),
        }

    # ── answering ──
    def _ask(self, runnable, prompt: str) -> tuple[Any, list[str]]:
        try:
            raw = runnable.invoke([SystemMessage(content=MOCK_ENGINE_SYSTEM), HumanMessage(content=prompt)])
        except Exception as e:  # noqa: BLE001 - a model failure is a validation failure of this attempt
            return None, [f"mock model error: {type(e).__name__}: {e}"]
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        return raw, []

    def _parse(self, raw: Any, envelope: bool, spec: dict) -> tuple[Any, list[str]]:
        if raw is None:
            return None, ["the mock model returned no answer"]
        if envelope:
            text = raw.get("response_json") if isinstance(raw, dict) else None
            if not isinstance(text, str):
                return None, ["response_json (JSON text) is missing"]
            try:
                response = json.loads(text)
            except ValueError as e:
                return None, [f"response_json is invalid JSON: {e}"]
        else:
            response = raw
        problems = tool_schemas.validate(response, spec.get("output_schema") or {})
        return response, problems

    def respond(self, tool_name: str, args: dict) -> Any:
        started = time.time()
        spec = self.spec(tool_name)
        schema, envelope = response_schema(spec)
        runnable = self.model.with_structured_output(schema)
        strategy = (self.strategies.get("strategies") or {}).get(self.strategy) or {}
        world = str(self.strategies.get("world") or "")
        self.log(f"mock engine: {tool_name}({json.dumps(args, default=str)[:80]}) [{self.strategy}]")
        problems: list[str] = []
        response: Any = None
        valid = False
        repairs = 0
        for attempt in range(self.max_repairs + 1):
            prompt = build_prompt(tool_name, args, spec, world, self.strategy, strategy, self.history[-self.history_limit:], problems)
            raw, problems = self._ask(runnable, prompt)
            if not problems:
                response, problems = self._parse(raw, envelope, spec)
            if not problems:
                valid = True
                break
            if attempt < self.max_repairs:
                repairs += 1
        entry = {
            "tool": tool_name, "args": args, "layer": "llm", "strategy": self.strategy, "response": response,
            "valid": valid, "repairs": repairs, "fallback": False, "seconds": round(time.time() - started, 3),
        }
        if not valid:
            entry["problems"] = list(problems)
            if self.on_invalid == "strict":
                entry["error"] = f"invalid mock response: {'; '.join(problems[:3])}"
                self.ledger.append(entry)
                raise MockEngineError(tool_name, args, problems)
            response = self.fallback_for(tool_name)
            entry["response"] = response
            entry["fallback"] = True
        self.ledger.append(entry)
        self.history.append({"tool": tool_name, "args": args, "response": response})
        return copy.deepcopy(response)
