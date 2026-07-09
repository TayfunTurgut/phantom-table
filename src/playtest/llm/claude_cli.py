"""Claude CLI backend: headless ``claude -p`` per completion.

Runs against the user's Claude subscription (enterprise OAuth login or a
``claude setup-token`` token via CLAUDE_CODE_OAUTH_TOKEN) — the only
programmatic path OAuth credentials support. All built-in tools are disabled so
the CLI acts as a pure completion endpoint; JSON schemas are hard-enforced via
``--json-schema``. Costs ~1-2s process spawn per call.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

from playtest.errors import PlaytestError
from playtest.llm import LLMClient

if TYPE_CHECKING:
    from playtest.config import Settings


def _flatten(messages: list[dict]) -> tuple[str, str]:
    """Split messages into (system_prompt, prompt transcript)."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if len(rest) == 1:
        prompt = rest[0]["content"] or ""
    else:
        prompt = "\n\n".join(f"{m['role'].capitalize()}:\n{m['content'] or ''}" for m in rest)
    return "\n\n".join(system_parts), prompt


class ClaudeCLIClient(LLMClient):
    def __init__(self, settings: Settings) -> None:
        self.models = {
            "player": settings.claude_player_model,
            "digest": settings.claude_digest_model,
            "codegen": settings.claude_codegen_model,
        }
        self._cli_path = settings.claude_cli_path
        self._oauth_token = settings.claude_code_oauth_token
        self._timeout = settings.llm_timeout_seconds

    def complete(
        self,
        messages: list[dict],
        *,
        role: str,
        json_schema: dict | None = None,
    ) -> str:
        system, prompt = _flatten(messages)

        command = [
            self._cli_path,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--tools",
            "",
            "--model",
            self.models[role],
        ]
        if system:
            command += ["--system-prompt", system]
        if json_schema is not None:
            command += ["--json-schema", json.dumps(json_schema["schema"])]

        env = dict(os.environ)
        if self._oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self._oauth_token

        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=self._timeout, env=env
            )
        except subprocess.TimeoutExpired as exc:
            raise PlaytestError(
                f"claude -p timed out after {self._timeout}s (role={role})"
            ) from exc
        if proc.returncode != 0:
            raise PlaytestError(
                f"claude -p failed (role={role}, exit {proc.returncode}): "
                f"{proc.stderr.strip()[-2000:] or proc.stdout.strip()[-2000:]}"
            )

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise PlaytestError(
                f"claude -p returned a non-JSON envelope (role={role}): {proc.stdout.strip()[:500]}"
            ) from exc
        if envelope.get("is_error"):
            raise PlaytestError(
                f"claude -p reported an error (role={role}, "
                f"subtype={envelope.get('subtype')}): {str(envelope.get('result'))[:2000]}"
            )

        # Structured output may arrive as a dedicated field or as the result
        # payload itself; normalize to a string (JSON text when a schema was set).
        structured = envelope.get("structured_output")
        if json_schema is not None and structured is not None:
            return json.dumps(structured)
        result = envelope.get("result", "")
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return result or ""
