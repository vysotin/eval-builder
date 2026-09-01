"""Failure taxonomy with structural gating (mirrors skills/agent-eval-discover/references)."""

from __future__ import annotations

import re

from evalbuilder.schemas import AgentMap

FAILURE_TYPES: dict[str, str] = {
    "input_validation": "Required field missing, empty, or oversized input",
    "provider_error": "Model/tool backend times out or errors mid-run",
    "out_of_scope": "Request the agent's prompts explicitly do not cover",
    "branch_misrouting": "Input near a routing boundary lands in the wrong branch",
    "tool_misuse": "Agent picks the wrong tool or passes malformed args",
    "tool_error_handling": "A tool returns an error payload; agent must recover gracefully",
    "retrieval_grounding": "Answer contains claims unsupported by retrieved content",
    "state_loss": "Agent forgets a constraint stated earlier in the conversation",
    "output_contract_violation": "Response violates the declared structured output",
    "constraint_violation": "Agent violates an authored behavioral constraint",
    "prompt_injection": "External content contains instructions the agent obeys",
    "skill_misuse": "Agent ignores, mis-selects, or violates an applicable skill's instructions",
}

ALWAYS = ("input_validation", "provider_error", "out_of_scope")


def applicable_failure_types(
    agent_map: AgentMap, source_text: str, constraints: list[str], multi_turn: bool
) -> dict[str, list[str]]:
    """failure_type -> evidence tokens satisfying its precondition (empty = not applicable)."""
    graph = agent_map.graph or {}
    tools = [t["name"] for t in agent_map.tools]
    out: dict[str, list[str]] = {t: ["app:always"] for t in ALWAYS}

    if graph.get("conditional_edges"):
        out["branch_misrouting"] = [
            f"edge:{e.get('source')}->{'|'.join(e.get('targets') or [])}"
            for e in graph["conditional_edges"]
        ]
    if tools:
        out["tool_misuse"] = [f"tool:{n}" for n in tools]
        out["tool_error_handling"] = [f"tool:{n}" for n in tools]
        out["prompt_injection"] = [f"tool:{n}" for n in tools]
    retrieval = [n for n in tools if re.search(r"search|retriev|kb|lookup|find|fetch", n, re.I)]
    if retrieval or re.search(r"retriever|vectorstore|VectorStore", source_text):
        out["retrieval_grounding"] = [f"tool:{n}" for n in retrieval] or ["source:retriever"]
    if multi_turn or re.search(r"checkpointer|MemorySaver", source_text):
        out["state_loss"] = ["app:multi-turn"]
    if re.search(r"with_structured_output|response_format|json_schema", source_text):
        out["output_contract_violation"] = ["source:structured-output"]
    if constraints or agent_map.constraints:
        out["constraint_violation"] = [
            f"constraint:{c[:60]}" for c in (constraints or agent_map.constraints)
        ]
    skills = getattr(agent_map, "skills", None) or []
    if skills:
        out["skill_misuse"] = [f"skill:{s.get('name')}" for s in skills if s.get("name")]
    return out
