"""LLM backend adapter tests (no network).

The Claude CLI adapter is tested against a fake `claude` executable that records
its argv/env and prints a canned --output-format json envelope. The OpenAI
adapter's tool loop is tested against an openai-shaped stub.
"""

import json
import stat
from types import SimpleNamespace

import pytest

from playtest.config import Settings
from playtest.errors import PlaytestError
from playtest.llm import LLMTool, create_llm_client
from playtest.llm.claude_cli import ClaudeCLIClient
from playtest.llm.openai_api import OpenAIClient

# ----------------------------------------------------------- claude_cli


def fake_claude(tmp_path, envelope: dict, exit_code: int = 0) -> tuple[str, str]:
    """Write a fake `claude` executable; returns (script_path, capture_path)."""
    capture = tmp_path / "capture.json"
    script = tmp_path / "claude"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"with open({str(capture)!r}, 'w') as f:\n"
        "    json.dump({'argv': sys.argv[1:], "
        "'token': os.environ.get('CLAUDE_CODE_OAUTH_TOKEN')}, f)\n"
        f"print(json.dumps({envelope!r}))\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script), str(capture)


def cli_settings(script: str) -> Settings:
    return Settings(
        _env_file=None,
        llm_backend="claude_cli",
        claude_cli_path=script,
        claude_code_oauth_token="tok-test-123",
        openai_api_key=None,
    )


SCHEMA = {"name": "t", "strict": True, "schema": {"type": "object", "properties": {}}}
MESSAGES = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hello"},
]


def test_claude_cli_invocation_flags_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    script, capture = fake_claude(
        tmp_path, {"type": "result", "is_error": False, "result": '{"ok": true}'}
    )
    client = ClaudeCLIClient(cli_settings(script))
    result = client.complete(MESSAGES, role="player", json_schema=SCHEMA)
    assert result == '{"ok": true}'

    recorded = json.loads(open(capture).read())
    argv = recorded["argv"]
    assert argv[0] == "-p" and argv[1] == "hello"
    assert ["--output-format", "json"] == argv[argv.index("--output-format") :][:2]
    assert ["--tools", ""] == argv[argv.index("--tools") :][:2]
    assert ["--model", "sonnet"] == argv[argv.index("--model") :][:2]
    assert ["--system-prompt", "be brief"] == argv[argv.index("--system-prompt") :][:2]
    assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA["schema"]
    assert recorded["token"] == "tok-test-123"


def test_claude_cli_no_schema_omits_flag_and_returns_text(tmp_path):
    script, capture = fake_claude(
        tmp_path, {"type": "result", "is_error": False, "result": "plain text answer"}
    )
    client = ClaudeCLIClient(cli_settings(script))
    assert client.complete(MESSAGES, role="codegen") == "plain text answer"
    argv = json.loads(open(capture).read())["argv"]
    assert "--json-schema" not in argv


def test_claude_cli_structured_output_field_wins(tmp_path):
    script, _ = fake_claude(
        tmp_path,
        {"type": "result", "is_error": False, "result": "ignored", "structured_output": {"a": 1}},
    )
    client = ClaudeCLIClient(cli_settings(script))
    assert json.loads(client.complete(MESSAGES, role="player", json_schema=SCHEMA)) == {"a": 1}


def test_claude_cli_multi_turn_messages_flatten(tmp_path):
    script, capture = fake_claude(tmp_path, {"is_error": False, "result": "x"})
    client = ClaudeCLIClient(cli_settings(script))
    client.complete(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "guess"},
            {"role": "user", "content": "try again"},
        ],
        role="player",
    )
    prompt = json.loads(open(capture).read())["argv"][1]
    assert "User:\nfirst" in prompt
    assert "Assistant:\nguess" in prompt
    assert "User:\ntry again" in prompt


def test_claude_cli_nonzero_exit_raises(tmp_path):
    script, _ = fake_claude(tmp_path, {"is_error": False, "result": "x"}, exit_code=3)
    client = ClaudeCLIClient(cli_settings(script))
    with pytest.raises(PlaytestError, match="exit 3"):
        client.complete(MESSAGES, role="player")


def test_claude_cli_error_envelope_raises(tmp_path):
    script, _ = fake_claude(
        tmp_path, {"is_error": True, "subtype": "error_during_execution", "result": "nope"}
    )
    client = ClaudeCLIClient(cli_settings(script))
    with pytest.raises(PlaytestError, match="error_during_execution"):
        client.complete(MESSAGES, role="player")


def test_claude_cli_rejects_tools(tmp_path):
    script, _ = fake_claude(tmp_path, {"is_error": False, "result": "x"})
    client = ClaudeCLIClient(cli_settings(script))
    tool = LLMTool(name="t", description="", parameters={}, handler=lambda a: "")
    assert not client.supports_tools
    with pytest.raises(ValueError, match="does not support tools"):
        client.complete(MESSAGES, role="player", tools=[tool])


# ----------------------------------------------------------- openai


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class OpenAIShapedStub:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self._responses.pop(0))])


def openai_client(responses) -> tuple[OpenAIClient, OpenAIShapedStub]:
    stub = OpenAIShapedStub(responses)
    settings = Settings(_env_file=None, openai_api_key="test-key")
    return OpenAIClient(settings, client=stub), stub


def test_openai_plain_completion_and_schema_passthrough():
    client, stub = openai_client([_message("hi")])
    assert client.complete(MESSAGES, role="player", json_schema=SCHEMA) == "hi"
    request = stub.requests[0]
    assert request["model"] == "gpt-5-mini"
    assert request["response_format"]["json_schema"] == SCHEMA


def test_openai_tool_loop_executes_handler():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="lookup", arguments=json.dumps({"q": "rule?"})),
    )
    client, stub = openai_client([_message(None, [tool_call]), _message("done")])
    seen = []
    tool = LLMTool(
        name="lookup",
        description="look things up",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        handler=lambda args: (seen.append(args), "the answer")[1],
    )
    assert client.complete(MESSAGES, role="player", tools=[tool]) == "done"
    assert seen == [{"q": "rule?"}]
    roles = [m["role"] for m in stub.requests[1]["messages"]]
    assert "tool" in roles


def test_openai_tool_handler_failure_is_reported_to_model_not_raised():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="lookup", arguments="{}"),
    )
    client, stub = openai_client([_message(None, [tool_call]), _message("done")])

    def broken_handler(args):
        raise RuntimeError("chromadb down")

    tool = LLMTool(
        name="lookup",
        description="look things up",
        parameters={"type": "object", "properties": {}},
        handler=broken_handler,
    )
    assert client.complete(MESSAGES, role="player", tools=[tool]) == "done"
    tool_message = next(m for m in stub.requests[1]["messages"] if m["role"] == "tool")
    assert "failed" in tool_message["content"]
    assert "chromadb down" in tool_message["content"]


def test_openai_tool_loop_is_capped():
    def tool_call(i):
        return SimpleNamespace(
            id=f"call_{i}",
            function=SimpleNamespace(name="lookup", arguments="{}"),
        )

    client, _ = openai_client([_message(None, [tool_call(i)]) for i in range(20)])
    tool = LLMTool(
        name="lookup",
        description="look things up",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "the answer",
    )
    with pytest.raises(PlaytestError, match="tool-call rounds"):
        client.complete(MESSAGES, role="player", tools=[tool])


def test_openai_malformed_tool_arguments_raise_playtest_error():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="lookup", arguments="not json"),
    )
    client, _ = openai_client([_message(None, [tool_call])])
    tool = LLMTool(
        name="lookup",
        description="look things up",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "",
    )
    with pytest.raises(PlaytestError, match="malformed arguments"):
        client.complete(MESSAGES, role="player", tools=[tool])


# ----------------------------------------------------------- factory


def test_factory_selects_backend(tmp_path):
    script, _ = fake_claude(tmp_path, {"is_error": False, "result": "x"})
    assert isinstance(create_llm_client(cli_settings(script)), ClaudeCLIClient)
    assert isinstance(create_llm_client(Settings(_env_file=None, openai_api_key="k")), OpenAIClient)


def test_factory_openai_requires_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_llm_client(Settings(_env_file=None, openai_api_key=None))


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown llm_backend"):
        create_llm_client(Settings(_env_file=None, llm_backend="bogus", openai_api_key="k"))
