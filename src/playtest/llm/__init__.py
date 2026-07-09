"""The shared LLM client construct: one completion interface, one backend.

Every completion in the app flows through ``LLMClient.complete`` — player
decisions and the digest use hard-enforced JSON schemas, codegen uses plain
text. Models are addressed by ROLE ("player", "digest", "codegen"); the adapter
maps roles to model names from settings.

Backend:
- ``claude_cli`` — headless ``claude -p`` subprocess (billed to the user's
                   Claude subscription; works with enterprise OAuth tokens).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playtest.config import Settings

ROLES = ("player", "digest", "codegen")


class LLMClient(ABC):
    """One LLM completion interface for all backends."""

    models: dict[str, str]

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        *,
        role: str,
        json_schema: dict | None = None,
    ) -> str:
        """Run one completion.

        ``messages`` is a list of {"role": "system"|"user"|"assistant",
        "content": str}. ``role`` selects the model via ``self.models``.
        ``json_schema`` ({"name", "strict", "schema"}) hard-enforces the output
        shape; the return value is then a JSON string.
        """


def create_llm_client(settings: Settings) -> LLMClient:
    """Build the LLM client (headless ``claude -p``)."""
    from playtest.llm.claude_cli import ClaudeCLIClient

    return ClaudeCLIClient(settings)
