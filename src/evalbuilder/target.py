"""Load the target LangGraph agent and execute one case with trajectory capture."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, convert_to_openai_messages

from evalbuilder.schemas import Case, CaseRun, Target


def load_target(target: Target):
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    return importlib.import_module(target.module)


def build_graph(module, target: Target, tools=None, model=None):
    """Call the target factory, passing only the overrides that were given.

    The contract is `build_agent(model=None, tools=None)`; omitting an argument lets
    the factory fall back to its own default (e.g. a scripted model for tests).
    """
    factory = getattr(module, target.factory)
    kwargs = {}
    if tools is not None:
        kwargs["tools"] = tools
    if model is not None:
        kwargs["model"] = model
    return factory(**kwargs)


def _extract(state) -> tuple[list[dict], list[dict], str]:
    messages = state["messages"]
    trajectory = convert_to_openai_messages(messages)
    tool_calls = [
        {"name": tc["name"], "args": tc["args"]}
        for m in messages
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    ]
    last = messages[-1].content
    response = last if isinstance(last, str) else str(last)
    return trajectory, tool_calls, response


def _invoke_with_path(graph, inputs: dict) -> tuple[dict, list[str]]:
    node_path: list[str] = []
    state = None
    for mode, chunk in graph.stream(inputs, stream_mode=["updates", "values"]):
        if mode == "updates":
            node_path.extend(k for k in chunk if k != "__interrupt__")
        else:
            state = chunk
    if state is None:
        raise RuntimeError("graph produced no output state")
    return state, node_path


def run_case(graph, case: Case) -> CaseRun:
    try:
        state, node_path = _invoke_with_path(graph, dict(case.inputs))
        for turn in case.metadata.get("user_turns", []) or []:
            messages = list(state["messages"])
            messages.append({"role": "user", "content": turn})
            state, more_path = _invoke_with_path(graph, {"messages": messages})
            node_path.extend(more_path)
        trajectory, tool_calls, response = _extract(state)
        return CaseRun(
            case_id=case.id,
            outputs={"response": response},
            trajectory=trajectory,
            tool_calls=tool_calls,
            node_path=node_path,
        )
    except Exception as e:  # noqa: BLE001 - the run artifact records the failure
        return CaseRun(
            case_id=case.id,
            error=f"{type(e).__name__}: {e}",
            error_class="infrastructure" if _is_mock_engine_error(e) else "agent",
        )


def _is_mock_engine_error(exc: BaseException) -> bool:
    """A mock engine failure (invalid LLM mock under `on_invalid: strict`) is not the agent's doing."""
    from evalbuilder.mock_engine import MockEngineError

    seen = 0
    while exc is not None and seen < 5:
        if isinstance(exc, MockEngineError):
            return True
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return False
