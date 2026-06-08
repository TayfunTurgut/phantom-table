"""Player action tool dispatch: capture intent, never execute game logic."""

_META_FIELDS = {"reasoning", "public_statement"}


class PlayerActionDispatch:
    """Capture player tool calls as structured proposed actions for the GM."""

    def __init__(self, tool_definitions: dict[str, dict]) -> None:
        self.tool_definitions = tool_definitions

    def get_tool_schemas(self, available_tools: list[str] | None = None) -> list[dict]:
        """Return OpenAI function schemas for the specified tools (all if None)."""
        if available_tools is None:
            return list(self.tool_definitions.values())
        schemas: list[dict] = []
        for name in available_tools:
            if name not in self.tool_definitions:
                raise ValueError(f"unknown action tool '{name}'")
            schemas.append(self.tool_definitions[name])
        return schemas

    def dispatch(self, tool_name: str, tool_args: dict) -> dict:
        """Capture a player's tool call as a proposed action.

        Does NOT validate legality — that is the GM's responsibility.
        """
        return {
            "action_type": tool_name,
            "parameters": {k: v for k, v in tool_args.items() if k not in _META_FIELDS},
            "reasoning": tool_args.get("reasoning", ""),
            "public_statement": tool_args.get("public_statement", ""),
        }
