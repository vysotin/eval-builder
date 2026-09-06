"""Agent clients: one `invoke(messages, mocks)` contract, served in-process or over HTTP.

`LocalAgent` imports the target module and builds a graph per call with the requested
mocks installed — the code `evalbuilder serve` runs inside the container and what
`evalbuilder infer --local` uses. `RemoteAgent` sends the same request to a deployed
agent server (`serve.py`). Both return an `InvokeResult`; both pickle, so joblib's
process backend can hand them to workers.

The conversation is stateless: the caller sends the full history (OpenAI-format
messages, as `convert_to_openai_messages` renders them) and receives the full history
back, tool calls and tool results included, so the next turn is `messages + [user]`.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

from evalbuilder.mocking import merge_mock_rules

CLIENT_VERSION = "evalbuilder/agent-client/v1"


@dataclass
class InvokeResult:
    messages: list[dict] = field(default_factory=list)  # the full history after this turn (OpenAI format)
    response: str = ""  # the last assistant text
    tool_calls: list[dict] = field(default_factory=list)  # every tool call of the whole history
    node_path: list[str] = field(default_factory=list)  # graph nodes visited in this turn
    mock_calls: list[dict] = field(default_factory=list)  # mock ledger of this turn
    error: str | None = None
    error_class: str = "none"  # none | agent | infrastructure
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InvokeResult":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


def mocks_for_case(ds_mocks: dict | None, case_meta: dict | None, *, on_miss: str | None = None, strategy: str | None = None) -> dict:
    """The mock block one request carries: dataset rules with the case's overrides on
    top, the miss policy, the selected strategy and the LLM engine settings.

    `on_miss` / `strategy` override the dataset's defaults; a case's own
    `metadata.mocks.strategy` always wins over the run-level `strategy`.
    """
    ds_mocks = ds_mocks or {}
    case_mocks = (case_meta or {}).get("mocks") or {}
    block: dict[str, Any] = {
        "tools": merge_mock_rules(ds_mocks.get("tools", {}), case_mocks.get("tools", {})),
        "on_miss": on_miss or ds_mocks.get("on_miss") or "real",
        "strategy": case_mocks.get("strategy") or strategy or ds_mocks.get("strategy"),
    }
    if ds_mocks.get("strategies"):
        block["strategies"] = ds_mocks["strategies"]
    if ds_mocks.get("llm"):
        block["llm"] = dict(ds_mocks["llm"])
    return block


def _messages_in(messages: list[dict]):
    from langchain_core.messages import convert_to_messages

    return convert_to_messages([dict(m) for m in messages])


class LocalAgent:
    """The target agent in this process: a fresh graph (and fresh tool wrappers) per call."""

    mode = "local"
    endpoint = None

    def __init__(
        self,
        module: str,
        factory: str = "build_agent",
        *,
        agent_model_spec: str | None = None,
        mock_model_spec: str | None = None,
        agent_model=None,
        mock_model=None,
    ):
        self.module = module
        self.factory = factory
        self.agent_model_spec = agent_model_spec
        self.mock_model_spec = mock_model_spec
        self._agent_model = agent_model
        self._mock_models: dict[str, Any] = {}
        if mock_model is not None:
            self._mock_models[""] = mock_model
        self._module = None
        self.default_fallback = None  # canned answer for `on_miss: fallback` when the request names none

    # objects built lazily are not pickled: workers rebuild them from the specs
    def __getstate__(self):
        state = dict(self.__dict__)
        state["_module"] = None
        state["_agent_model"] = None if self.agent_model_spec else self._agent_model
        state["_mock_models"] = {} if self.mock_model_spec else dict(self._mock_models)
        return state

    def _target(self):
        from evalbuilder import target as target_mod
        from evalbuilder.schemas import Target

        if self._module is None:
            self._module = target_mod.load_target(Target(module=self.module, factory=self.factory))
        return self._module

    def _agent(self):
        if self._agent_model is None and self.agent_model_spec:
            from evalbuilder.claude_cli import model_from_spec

            self._agent_model = model_from_spec(self.agent_model_spec)
        return self._agent_model

    def _mock_model(self, spec: str | None):
        """The engine's model: this client's own (an object, or `mock_model_spec` — what
        the deployment configured) wins; the request's `llm.model` spec is the fallback."""
        if "" in self._mock_models:
            return self._mock_models[""]
        key = "" if self.mock_model_spec else (spec or "")
        if key not in self._mock_models:
            chosen = self.mock_model_spec or spec
            if not chosen:
                raise ValueError("on_miss='llm' needs a mock model: mocks.llm.model in the request or mock_model_spec on the client")
            from evalbuilder.claude_cli import model_from_spec

            self._mock_models[key] = model_from_spec(chosen)
        return self._mock_models[key]

    def health(self) -> dict:
        module = self._target()
        return {
            "ok": True, "mode": self.mode, "module": self.module, "factory": self.factory,
            "tools": [t.name for t in getattr(module, "TOOLS", []) or []],
            "agent_model": self.agent_model_spec, "mock_model": self.mock_model_spec, "client": CLIENT_VERSION,
        }

    def invoke(self, messages: list[dict], mocks: dict | None = None, *, mocked: bool = True) -> InvokeResult:
        from evalbuilder import target as target_mod
        from evalbuilder.mock_engine import DEFAULT_STRATEGY, LLMMockEngine
        from evalbuilder.mocking import wrap_tools
        from evalbuilder.runner import tool_specs_of
        from evalbuilder.schemas import Target

        started = time.time()
        ledger: list[dict] = []
        try:
            module = self._target()
            target = Target(module=self.module, factory=self.factory)
            if mocked and mocks is not None:
                on_miss = mocks.get("on_miss") or "real"
                engine = None
                if on_miss == "llm":
                    llm = mocks.get("llm") or {}
                    engine = LLMMockEngine(
                        self._mock_model(llm.get("model")), mocks.get("strategies"), tool_specs_of(module),
                        strategy=mocks.get("strategy") or DEFAULT_STRATEGY,
                        on_invalid=llm.get("on_invalid", "fallback"), max_repairs=int(llm.get("max_repairs", 1)),
                    )
                    # earlier turns of a stateless conversation: what the engine already answered
                    engine.history.extend(h for h in (mocks.get("history") or []) if isinstance(h, dict))
                fallback = mocks["fallback"] if "fallback" in mocks else self.default_fallback
                tools = wrap_tools(list(getattr(module, "TOOLS")), mocks.get("tools") or {}, on_miss=on_miss,
                                   fallback=fallback, engine=engine, ledger=ledger)
                graph = target_mod.build_graph(module, target, tools=tools, model=self._agent())
            else:
                graph = target_mod.build_graph(module, target, model=self._agent())
        except Exception as e:  # noqa: BLE001 - building the agent is infrastructure
            return InvokeResult(messages=list(messages), error=f"{type(e).__name__}: {e}", error_class="infrastructure",
                                mock_calls=ledger, seconds=round(time.time() - started, 3))
        try:
            state, node_path = target_mod.invoke_messages(graph, _messages_in(messages))
            history, tool_calls, response = target_mod.extract(state)
            return InvokeResult(messages=history, response=response, tool_calls=tool_calls, node_path=node_path,
                                mock_calls=ledger, seconds=round(time.time() - started, 3))
        except Exception as e:  # noqa: BLE001 - the result records the failure
            return InvokeResult(
                messages=list(messages), error=f"{type(e).__name__}: {e}",
                error_class="infrastructure" if target_mod.is_mock_engine_error(e) else "agent",
                mock_calls=ledger, seconds=round(time.time() - started, 3),
            )


class RemoteAgent:
    """A deployed agent server (`evalbuilder serve`) behind `endpoint`."""

    mode = "remote"

    def __init__(self, endpoint: str, timeout: float = 120.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.endpoint}{path}", data=data, method="POST" if data is not None else "GET",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode() or "{}")

    def health(self) -> dict:
        try:
            return {**self._request("/health"), "mode": self.mode}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            return {"ok": False, "mode": self.mode, "error": f"{type(e).__name__}: {e}"}

    def invoke(self, messages: list[dict], mocks: dict | None = None, *, mocked: bool = True) -> InvokeResult:
        started = time.time()
        try:
            data = self._request("/invoke", {"messages": messages, "mocks": mocks, "mocked": mocked})
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            return InvokeResult(messages=list(messages), error=f"HTTP {e.code} from {self.endpoint}/invoke: {body}",
                                error_class="infrastructure", seconds=round(time.time() - started, 3))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            return InvokeResult(messages=list(messages), error=f"{type(e).__name__}: {e} ({self.endpoint})",
                                error_class="infrastructure", seconds=round(time.time() - started, 3))
        result = InvokeResult.from_dict(data)
        result.seconds = round(time.time() - started, 3)
        return result


def agent_for(endpoint: str | None = None, *, module: str | None = None, factory: str = "build_agent",
              timeout: float = 120.0, **local_kwargs):
    """`RemoteAgent(endpoint)` when an endpoint is given, else a `LocalAgent(module)`."""
    if endpoint:
        return RemoteAgent(endpoint, timeout=timeout)
    if not module:
        raise ValueError("agent_for needs an endpoint or a target module")
    return LocalAgent(module, factory, **local_kwargs)
