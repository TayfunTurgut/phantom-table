"""OpenAI API backend: structured outputs natively, tool loop in-process."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from playtest.errors import PlaytestError
from playtest.llm import LLMClient, LLMTool

if TYPE_CHECKING:
    from playtest.config import Settings

_log = logging.getLogger(__name__)

# Safety cap on consecutive tool-call rounds in one completion (cost ceiling).
_MAX_TOOL_ROUNDS = 8


class OpenAIClient(LLMClient):
    supports_tools: ClassVar[bool] = True

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self.models = {
            "player": settings.player_model,
            "digest": settings.digest_model,
            "codegen": settings.codegen_model,
        }
        if client is not None:
            self._client = client
            return
        if not settings.openai_api_key:
            raise ValueError(
                "llm_backend=openai requires OPENAI_API_KEY (or switch to LLM_BACKEND=claude_cli)"
            )
        from openai import OpenAI

        from playtest.config import maybe_wrap_openai

        self._client = maybe_wrap_openai(OpenAI(api_key=settings.openai_api_key))

    def complete(
        self,
        messages: list[dict],
        *,
        role: str,
        json_schema: dict | None = None,
        tools: list[LLMTool] | None = None,
    ) -> str:
        messages = list(messages)
        handlers = {tool.name: tool.handler for tool in (tools or [])}
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": True,
                },
            }
            for tool in (tools or [])
        ]

        for _ in range(_MAX_TOOL_ROUNDS):
            kwargs: dict = {"model": self.models[role], "messages": messages}
            if json_schema is not None:
                kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
            if tool_schemas:
                kwargs["tools"] = tool_schemas
            completion = self._client.chat.completions.create(**kwargs)
            message = completion.choices[0].message

            calls = getattr(message, "tool_calls", None)
            if not calls:
                return message.content or ""

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in calls
                    ],
                }
            )
            for call in calls:
                handler = handlers.get(call.function.name)
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise PlaytestError(
                        f"model returned malformed arguments for tool "
                        f"{call.function.name!r}: {(call.function.arguments or '')[:200]!r}"
                    ) from exc
                if handler is None:
                    result = f"unknown tool {call.function.name}"
                else:
                    # A failing tool (network, chromadb) must not crash the game:
                    # report the failure to the model and let it answer without it.
                    try:
                        result = handler(args)
                    except Exception as exc:
                        _log.warning(
                            "tool %s failed: %s: %s", call.function.name, type(exc).__name__, exc
                        )
                        result = f"tool {call.function.name} failed: {type(exc).__name__}: {exc}"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        raise PlaytestError(
            f"model exceeded {_MAX_TOOL_ROUNDS} tool-call rounds without answering (role={role})"
        )
