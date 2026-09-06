"""The agent server: the target agent, with both mock layers, behind two JSON routes.

    GET  /health  → {ok, module, factory, tools, agent_model, mock_model, server}
    POST /invoke  → {messages, mocks?, mocked?}  ⇒  InvokeResult (see agent_client.py)

Stdlib only (`ThreadingHTTPServer`), so the container image needs no web framework.
The server is stateless: every request carries the full conversation and the mock
block it wants installed; a graph (and fresh tool wrappers) is built per request.
This is the entry point of the agent image (`evalbuilder serve`) and of the `local`
deployment target.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evalbuilder.agent_client import LocalAgent

SERVER_VERSION = "evalbuilder/serve/v1"
MAX_BODY = 64 * 1024 * 1024


class _Handler(BaseHTTPRequestHandler):
    agent: LocalAgent  # set on the server class per instance (see make_server)
    quiet = True

    def log_message(self, fmt, *args):  # noqa: D401 - silence the default access log
        if not self.quiet:
            super().log_message(fmt, *args)

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] in ("/health", "/"):
            try:
                self._send(200, {**self.server.agent.health(), "server": SERVER_VERSION})
            except Exception as e:  # noqa: BLE001 - a broken target is reported, not raised
                self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}", "server": SERVER_VERSION})
            return
        self._send(404, {"ok": False, "error": f"unknown route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/invoke":
            self._send(404, {"ok": False, "error": f"unknown route {self.path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._send(400, {"ok": False, "error": "a JSON body with 'messages' is required"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode())
        except ValueError as e:
            self._send(400, {"ok": False, "error": f"invalid JSON: {e}"})
            return
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list) or not messages:
            self._send(400, {"ok": False, "error": "'messages' must be a non-empty list of {role, content} messages"})
            return
        mocks = payload.get("mocks")
        if mocks is not None and not isinstance(mocks, dict):
            self._send(400, {"ok": False, "error": "'mocks' must be an object"})
            return
        result = self.server.agent.invoke(messages, mocks, mocked=bool(payload.get("mocked", True)))
        self._send(200, result.to_dict())


class AgentServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, agent: LocalAgent, quiet: bool = True):
        handler = type("Handler", (_Handler,), {"quiet": quiet})
        super().__init__(address, handler)
        self.agent = agent

    @property
    def port(self) -> int:
        return self.server_address[1]

    @property
    def endpoint(self) -> str:
        host = self.server_address[0]
        return f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{self.port}"

    def serve_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="evalbuilder-serve", daemon=True)
        thread.start()
        return thread


def make_server(agent: LocalAgent, host: str = "127.0.0.1", port: int = 0, quiet: bool = True) -> AgentServer:
    """Bind the server (port 0 = any free port); call `.serve_forever()` or `.serve_in_thread()`."""
    return AgentServer((host, port), agent, quiet=quiet)


def serve(
    module: str,
    factory: str = "build_agent",
    host: str = "0.0.0.0",
    port: int = 8080,
    agent_model: str | None = None,
    mock_model: str | None = None,
    quiet: bool = False,
) -> None:
    """Run the agent server until interrupted (the container / local-target entry point).
    Model specs default to `EVALBUILDER_AGENT_MODEL` / `EVALBUILDER_MOCK_MODEL`."""
    agent = LocalAgent(
        module, factory,
        agent_model_spec=agent_model or os.environ.get("EVALBUILDER_AGENT_MODEL") or None,
        mock_model_spec=mock_model or os.environ.get("EVALBUILDER_MOCK_MODEL") or None,
    )
    agent.health()  # import the target now so a broken module fails at start-up, not on the first request
    server = make_server(agent, host, port, quiet=quiet)
    print(f"evalbuilder serve: {module}:{factory} on {server.endpoint} (agent model {agent.agent_model_spec or 'target default'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
