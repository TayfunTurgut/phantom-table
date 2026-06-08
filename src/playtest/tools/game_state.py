"""Tools wrapping the GameStateManager: read (any caller) and write (GM only)."""

from playtest.state.manager import GameStateManager


class GetGameStateTool:
    """Return the caller-filtered game state."""

    def __init__(self, state_manager: GameStateManager) -> None:
        self.manager = state_manager

    def execute(self, caller_id: str) -> dict:
        """Return the filtered game state for the caller."""
        return self.manager.get_state(caller_id)

    def as_openai_schema(self) -> dict:
        """Return the OpenAI function schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": "get_game_state",
                "description": (
                    "Get the current game state. Returns your hand, all public "
                    "information (discards, tokens, who is eliminated/protected, "
                    "revealed cards, deck count), and turn information. You cannot see "
                    "other players' hands."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Why you want to check the game state",
                        }
                    },
                    "required": ["reasoning"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }


class SetGameStateTool:
    """Replace the game state (GM only). Structural validation lives in the manager."""

    def __init__(self, state_manager: GameStateManager) -> None:
        self.manager = state_manager

    def execute(self, new_state: dict) -> dict:
        """Replace the game state. GM only."""
        return self.manager.set_state(new_state)

    def as_openai_schema(self) -> dict:
        """Return the OpenAI function schema for this tool.

        ``new_state`` is the entire game state object. Its structure is game-specific,
        so it is typed as a generic JSON object and this tool is not marked ``strict``
        (OpenAI strict mode forbids open-ended objects). The manager validates
        structure on write.
        """
        return {
            "type": "function",
            "function": {
                "name": "set_game_state",
                "description": (
                    "Replace the full game state after resolving an action. Provide "
                    "the complete updated state object, including all players, the "
                    "deck, and turn information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Why you are updating the state this way.",
                        },
                        "new_state": {
                            "type": "object",
                            "description": "The complete new game state object.",
                        },
                    },
                    "required": ["reasoning", "new_state"],
                },
                "strict": False,
            },
        }
