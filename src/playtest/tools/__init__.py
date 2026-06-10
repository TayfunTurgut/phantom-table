"""Tool registry assembling the tool set for each agent type."""

from playtest.ingestion.schemas import GameConfig
from playtest.state.manager import GameStateManager
from playtest.tools.actions import PlayerActionDispatch
from playtest.tools.game_state import GetGameStateTool, SetGameStateTool
from playtest.tools.rulebook import RulebookTool


def _finish_resolution_schema(phases: list[str]) -> dict:
    """Terminal GM tool: report the structured outcome of a resolution in one call.

    Calling this ends the GM's tool loop, so the structured fields come back directly
    without a second LLM summary call. It performs no state mutation itself. The
    ``next_phase`` enum is the ingested game's phases — nothing here is game-specific.
    """
    return {
        "type": "function",
        "function": {
            "name": "finish_resolution",
            "description": (
                "Report the final outcome of resolving the action and end your turn. "
                "Call this exactly once, after you have validated the action and (if "
                "legal) committed the new state with set_game_state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "is_valid": {
                        "type": "boolean",
                        "description": "Was the proposed action legal per the rules?",
                    },
                    "error_message": {
                        "type": ["string", "null"],
                        "description": "If invalid, the player-facing reason; else null.",
                    },
                    "action_summary": {
                        "type": "string",
                        "description": "One-line factual summary of what happened.",
                    },
                    "narration": {
                        "type": "string",
                        "description": "Brief, flavorful description for the players.",
                    },
                    "turn_ended": {
                        "type": "boolean",
                        "description": "Did this action end the acting player's turn?",
                    },
                    "round_ended": {
                        "type": "boolean",
                        "description": "Did this action end the current round/hand?",
                    },
                    "game_ended": {
                        "type": "boolean",
                        "description": "Did this action end the whole game?",
                    },
                    "winner": {
                        "type": ["string", "null"],
                        "description": "If the game ended: the winning player id, or null.",
                    },
                    "winners": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "All winners (shared wins possible), or null.",
                    },
                    "next_player": {
                        "type": ["string", "null"],
                        "description": "Player id whose turn is next, or null.",
                    },
                    "next_phase": {
                        "type": "string",
                        "enum": phases,
                        "description": "Phase for the next turn.",
                    },
                    "private_info": {
                        "type": ["object", "null"],
                        "description": (
                            "Info visible only to the acting player (e.g. a peeked card)."
                        ),
                    },
                    "gm_reasoning": {
                        "type": "string",
                        "description": "Your internal reasoning, for logging.",
                    },
                },
                "required": ["is_valid", "narration"],
            },
        },
    }


class ToolRegistry:
    """Assemble and route tool calls for GM and player agents."""

    def __init__(self, game_config: GameConfig, state_manager: GameStateManager) -> None:
        self.game_config = game_config
        self.rulebook_tool = RulebookTool(game_config.config_dir, game_config.game_name)
        self.get_state_tool = GetGameStateTool(state_manager)
        self.set_state_tool = SetGameStateTool(state_manager)
        self.action_dispatch = PlayerActionDispatch(game_config.tool_definitions)

    def get_shared_tool_schemas(self) -> list[dict]:
        """Tools available to every agent: query_rulebook + get_game_state."""
        return [
            self.rulebook_tool.as_openai_schema(),
            self.get_state_tool.as_openai_schema(),
        ]

    def get_action_tool_schemas(self, action_names: list[str]) -> list[dict]:
        """Look up specific player-action tool schemas by name."""
        return self.action_dispatch.get_tool_schemas(action_names)

    def get_gm_tools(self) -> list[dict]:
        """GM schemas: query_rulebook, get_game_state, set_game_state, finish_resolution."""
        return self.get_shared_tool_schemas() + [
            self.set_state_tool.as_openai_schema(),
            _finish_resolution_schema(self.game_config.game_spec.turn.phases),
        ]

    def get_player_tools(self, available_actions: list[str] | None = None) -> list[dict]:
        """Player schemas: query_rulebook, get_game_state, + specified action tools."""
        return self.get_shared_tool_schemas() + self.action_dispatch.get_tool_schemas(
            available_actions
        )

    def execute_tool(self, tool_name: str, tool_args: dict, caller_id: str) -> dict | str:
        """Route a tool call to the right handler."""
        if tool_name == "query_rulebook":
            # n_results isn't exposed in the (strict) query_rulebook schema, so it can
            # never be supplied; rely on query()'s default.
            return self.rulebook_tool.query(tool_args["query"])
        if tool_name == "get_game_state":
            return self.get_state_tool.execute(caller_id)
        if tool_name == "set_game_state":
            if caller_id != "gm":
                raise ValueError("only the GM may set game state")
            return self.set_state_tool.execute(tool_args["new_state"])
        return self.action_dispatch.dispatch(tool_name, tool_args)

    def get_rulebook_query_log(self) -> list[str]:
        """All rulebook query strings made this game (for analytics)."""
        return self.rulebook_tool.get_query_log()
