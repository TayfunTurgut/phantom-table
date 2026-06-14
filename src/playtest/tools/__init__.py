"""Agent-facing tools. Only the rulebook Q&A tool remains: legality and state
mutation are the engine's job now, so agents need no other tools."""

from playtest.tools.rulebook import RulebookTool

__all__ = ["RulebookTool"]
