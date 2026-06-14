"""The shared LLM client construct: one interface, selectable backends.

Every completion in the app flows through ``LLMClient.complete`` — player
decisions and the digest use hard-enforced JSON schemas, codegen uses plain
text, and the (optional) rulebook lookup is an in-process ``LLMTool``. Models
are addressed by ROLE ("player", "digest", "codegen"); each adapter
maps roles to backend-appropriate model names from settings.

Backends:
- ``openai``      — the OpenAI API (per-token billing)
- ``claude_cli``  — headless ``claude -p`` subprocess (billed to the user's
                    Claude subscription; works with enterprise OAuth tokens)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from playtest.config import Settings

ROLES = ("player", "digest", "codegen")


@dataclass(frozen=True)
class LLMTool:
    """An in-process tool the model may call before answering."""

    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object
    handler: Callable[[dict], str]


class LLMClient(ABC):
    """One LLM completion interface for all backends."""

    supports_tools: ClassVar[bool] = False
    models: dict[str, str]

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        *,
        role: str,
        json_schema: dict | None = None,
        tools: list[LLMTool] | None = None,
    ) -> str:
        """Run one completion.

        ``messages`` is a list of {"role": "system"|"user"|"assistant",
        "content": str}. ``role`` selects the model via ``self.models``.
        ``json_schema`` ({"name", "strict", "schema"}) hard-enforces the output
        shape; the return value is then a JSON string. Backends that do not
        support tools raise ``ValueError`` when ``tools`` is passed — gate on
        ``supports_tools``.
        """


def create_llm_client(settings: Settings) -> LLMClient:
    """Build the configured backend's client."""
    if settings.llm_backend == "openai":
        from playtest.llm.openai_api import OpenAIClient

        return OpenAIClient(settings)
    if settings.llm_backend == "claude_cli":
        from playtest.llm.claude_cli import ClaudeCLIClient

        return ClaudeCLIClient(settings)
    raise ValueError(
        f"unknown llm_backend {settings.llm_backend!r}; expected 'openai' or 'claude_cli'"
    )
