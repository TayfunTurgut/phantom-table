"""LLM backend adapter tests (no network).

The Claude CLI adapter is tested against a fake `claude` executable that records
its argv/env and prints a canned --output-format json envelope.
"""

import json
import stat
import subprocess

import pytest

from playtest.config import Settings
from playtest.errors import PlaytestError
from playtest.llm import create_llm_client
from playtest.llm.claude_cli import ClaudeCLIClient

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


def cli_settings(script: str, llm_retry_attempts: int = 1) -> Settings:
    # Most cases here exercise a single `claude -p` invocation; default to no
    # retries so a deterministic single failure doesn't sleep for real.
    return Settings(
        _env_file=None,
        claude_cli_path=script,
        claude_code_oauth_token="tok-test-123",
        llm_retry_attempts=llm_retry_attempts,
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


# ----------------------------------------------------------- retry


class _FakeProc:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_complete_retries_transient_failure_then_succeeds(monkeypatch, caplog):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) < 3:
            return _FakeProc(1, "", "transient boom")
        return _FakeProc(0, json.dumps({"is_error": False, "result": "recovered"}))

    monkeypatch.setattr("playtest.llm.claude_cli.subprocess.run", fake_run)
    monkeypatch.setattr("playtest.llm.claude_cli.time.sleep", lambda _seconds: None)

    client = ClaudeCLIClient(cli_settings("unused", llm_retry_attempts=3))
    with caplog.at_level("WARNING", logger="playtest.llm.claude_cli"):
        result = client.complete(MESSAGES, role="player")

    assert result == "recovered"
    assert len(calls) == 3
    assert sum("retrying" in r.message for r in caplog.records) == 2


def test_complete_exhausts_retries_and_reraises(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(cmd=command, timeout=1)

    monkeypatch.setattr("playtest.llm.claude_cli.subprocess.run", fake_run)
    monkeypatch.setattr("playtest.llm.claude_cli.time.sleep", lambda _seconds: None)

    settings = cli_settings("unused", llm_retry_attempts=3)
    client = ClaudeCLIClient(settings)
    with pytest.raises(PlaytestError, match="timed out"):
        client.complete(MESSAGES, role="player")

    assert len(calls) == settings.llm_retry_attempts


# ----------------------------------------------------------- factory


def test_factory_returns_claude_cli_client(tmp_path):
    script, _ = fake_claude(tmp_path, {"is_error": False, "result": "x"})
    assert isinstance(create_llm_client(cli_settings(script)), ClaudeCLIClient)
