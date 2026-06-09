"""Tool registry assembling the tool set for each agent type."""

from playtest.state.manager import GameStateManager
from playtest.tools.actions import PlayerActionDispatch
from playtest.tools.game_state import GetGameStateTool, SetGameStateTool
from playtest.tools.rulebook import RulebookTool


class ToolRegistry:
    """Assemble and route tool calls for GM and player agents."""

    def __init__(self, game_config: object, state_manager: GameStateManager) -> None:
        self.rulebook_tool = RulebookTool(
            game_config.config_dir,  # type: ignore[attr-defined]
            game_config.game_name,  # type: ignore[attr-defined]
        )
        self.get_state_tool = GetGameStateTool(state_manager)
        self.set_state_tool = SetGameStateTool(state_manager)
        self.action_dispatch = PlayerActionDispatch(
            game_config.tool_definitions  # type: ignore[attr-defined]
        )

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
        """GM schemas: query_rulebook, get_game_state, set_game_state."""
        return self.get_shared_tool_schemas() + [self.set_state_tool.as_openai_schema()]

    def get_player_tools(self, available_actions: list[str] | None = None) -> list[dict]:
        """Player schemas: query_rulebook, get_game_state, + specified action tools."""
        return self.get_shared_tool_schemas() + self.action_dispatch.get_tool_schemas(
            available_actions
        )

    def execute_tool(self, tool_name: str, tool_args: dict, caller_id: str) -> dict | str:
        """Route a tool call to the right handler."""
        if tool_name == "query_rulebook":
            return self.rulebook_tool.query(tool_args["query"], tool_args.get("n_results", 3))
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
