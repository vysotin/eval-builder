"""ChatClaudeCLI: subprocess transport, transcript flattening, tools, structured output."""

import json
import subprocess

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from evalbuilder import claude_cli
from evalbuilder.claude_cli import ChatClaudeCLI, ClaudeCLIError, model_from_spec, parse_spec


class FakeRun:
    """Records every subprocess call and replays canned CLI JSON results."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": argv, **kwargs})
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, tuple):
            code, stdout = result
        else:
            code, stdout = 0, json.dumps(result)
        return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="")


def _ok(result="hello", structured=None, **extra):
    return {
        "type": "result",
        "is_error": False,
        "result": result,
        "structured_output": structured,
        "session_id": "sess-1",
        "duration_ms": 12,
        "num_turns": 1,
        "total_cost_usd": 0.01,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        **extra,
    }


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_plain_invoke_uses_stdin_prompt_and_system_flag(monkeypatch):
    fake = FakeRun([_ok("OK")])
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    llm = ChatClaudeCLI(model="sonnet")
    out = llm.invoke([SystemMessage(content="Be terse."), HumanMessage(content="Say OK")])
    assert isinstance(out, AIMessage) and out.content == "OK"
    assert out.response_metadata["session_id"] == "sess-1"
    call = fake.calls[0]
    argv = call["argv"]
    assert argv[:2] == ["claude", "-p"]
    assert _flag(argv, "--model") == "sonnet"
    assert _flag(argv, "--output-format") == "json"
    assert "--json-schema" not in argv
    assert "Be terse." in _flag(argv, "--system-prompt")
    assert "Say OK" in call["input"]
    # isolation flags
    for flag in ("--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
        assert flag in argv
    assert _flag(argv, "--tools") == ""
    assert _flag(argv, "--setting-sources") == ""


def test_api_key_is_scrubbed_from_child_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    fake = FakeRun([_ok()])
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    ChatClaudeCLI().invoke("hi")
    assert "ANTHROPIC_API_KEY" not in fake.calls[0]["env"]


def test_bind_tools_emits_schema_and_parses_tool_calls(monkeypatch):
    @tool
    def get_weather(city: str) -> dict:
        """Weather for a city."""
        return {}

    fake = FakeRun(
        [
            _ok(
                json.dumps({"content": "", "tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}]}),
                structured={"content": "", "tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}]},
            )
        ]
    )
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    llm = ChatClaudeCLI().bind_tools([get_weather])
    out = llm.invoke([HumanMessage(content="weather in Paris")])
    assert out.tool_calls and out.tool_calls[0]["name"] == "get_weather"
    assert out.tool_calls[0]["args"] == {"city": "Paris"}
    assert out.tool_calls[0]["id"].startswith("call_")
    argv = fake.calls[0]["argv"]
    schema = json.loads(_flag(argv, "--json-schema"))
    assert "tool_calls" in schema["properties"]
    assert "get_weather" in _flag(argv, "--system-prompt")


def test_tool_call_ids_are_unique(monkeypatch):
    structured = {
        "content": "",
        "tool_calls": [{"name": "a", "args": {}}, {"name": "a", "args": {}}],
    }
    fake = FakeRun([_ok("x", structured=structured)])
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    out = ChatClaudeCLI().bind_tools([{"name": "a", "description": "", "input_schema": {}}]).invoke("go")
    ids = [tc["id"] for tc in out.tool_calls]
    assert len(set(ids)) == 2


def test_transcript_flattens_tool_history(monkeypatch):
    fake = FakeRun([_ok("It is sunny.", structured={"content": "It is sunny.", "tool_calls": []})])
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    llm = ChatClaudeCLI().bind_tools([{"name": "get_weather", "description": "", "input_schema": {}}])
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="weather in Paris"),
        AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}]),
        ToolMessage(content='{"temp": 72}', tool_call_id="call_1", name="get_weather"),
    ]
    out = llm.invoke(messages)
    assert out.content == "It is sunny." and not out.tool_calls
    prompt = fake.calls[0]["input"]
    assert "get_weather" in prompt and '{"temp": 72}' in prompt and "weather in Paris" in prompt
    assert prompt.index("weather in Paris") < prompt.index('{"temp": 72}')


def test_with_structured_output_dict_and_pydantic(monkeypatch):
    class Score(BaseModel):
        score: bool
        reasoning: str

    fake = FakeRun(
        [
            _ok("{}", structured={"score": True, "reasoning": "fine"}),
            _ok("{}", structured={"score": False, "reasoning": "nope"}),
        ]
    )
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    llm = ChatClaudeCLI()
    schema = {"title": "score", "type": "object", "properties": {"score": {"type": "boolean"}, "reasoning": {"type": "string"}}}
    assert llm.with_structured_output(schema).invoke([{"role": "user", "content": "judge"}]) == {
        "score": True,
        "reasoning": "fine",
    }
    result = llm.with_structured_output(Score).invoke("judge again")
    assert isinstance(result, Score) and result.score is False
    sent = json.loads(_flag(fake.calls[1]["argv"], "--json-schema"))
    assert "score" in sent["properties"]


def test_cli_error_raises_after_retries(monkeypatch):
    fake = FakeRun([(1, "boom"), (1, "boom again")])
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    monkeypatch.setattr(claude_cli.time, "sleep", lambda s: None)
    with pytest.raises(ClaudeCLIError):
        ChatClaudeCLI(max_retries=1).invoke("hi")
    assert len(fake.calls) == 2


def test_is_error_result_raises(monkeypatch):
    fake = FakeRun([_ok("Not logged in", is_error=True)])
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    with pytest.raises(ClaudeCLIError, match="Not logged in"):
        ChatClaudeCLI(max_retries=0).invoke("hi")


def test_falls_back_to_parsing_result_text_when_structured_missing(monkeypatch):
    text = json.dumps({"content": "done", "tool_calls": []})
    fake = FakeRun([_ok(text, structured=None)])
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    out = ChatClaudeCLI().bind_tools([{"name": "a", "description": "", "input_schema": {}}]).invoke("go")
    assert out.content == "done" and not out.tool_calls


def test_parse_spec_and_model_from_spec():
    assert parse_spec("claude-cli:opus@high") == ("claude-cli", "opus", "high")
    assert parse_spec("claude-cli:sonnet") == ("claude-cli", "sonnet", None)
    assert parse_spec("anthropic:claude-sonnet-5") == ("anthropic", "claude-sonnet-5", None)
    llm = model_from_spec("claude-cli:opus@high")
    assert isinstance(llm, ChatClaudeCLI) and llm.model == "opus" and llm.effort == "high"


def test_model_from_spec_scripted_provider():
    llm = model_from_spec("scripted:examples.weather_bot.agent:default_scripted_model")
    assert type(llm).__name__ == "ScriptedChatModel"


def test_tool_call_json_inside_content_is_unwrapped(monkeypatch):
    inner = json.dumps({"tool_calls": [{"name": "lookup_order", "args": {"order_id": "A1"}}]})
    fake = FakeRun([_ok(inner, structured={"content": inner, "tool_calls": []})])
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    out = ChatClaudeCLI().bind_tools([{"name": "lookup_order", "description": "", "input_schema": {}}]).invoke("go")
    assert out.tool_calls and out.tool_calls[0]["name"] == "lookup_order" and out.content == ""
