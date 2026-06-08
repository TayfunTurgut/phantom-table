import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry
from playtest.tools.actions import PlayerActionDispatch

from .test_state import DECK, HANDS, REMOVED, REVEALED, TEMPLATE

SAMPLE_TOOL_DEFINITIONS = {
    "draw_card": {
        "type": "function",
        "function": {
            "name": "draw_card",
            "description": "Draw a card.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "public_statement": {"type": "string"},
                },
                "required": ["reasoning", "public_statement"],
                "additionalProperties": False,
            },
        },
    },
    "play_guard": {
        "type": "function",
        "function": {
            "name": "play_guard",
            "description": "Play the Guard card.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "target_player": {"type": "string"},
                    "named_card": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "public_statement": {"type": "string"},
                },
                "required": ["target_player", "named_card", "reasoning", "public_statement"],
                "additionalProperties": False,
            },
        },
    },
    "play_prince": {
        "type": "function",
        "function": {"name": "play_prince", "description": "Play the Prince.", "parameters": {}},
    },
}


def _tool_names(schemas: list[dict]) -> set[str]:
    return {schema["function"]["name"] for schema in schemas}


def _registry(tmp_path: Path) -> ToolRegistry:
    game_config = SimpleNamespace(
        config_dir=str(tmp_path),
        game_name="Sample Game",
        tool_definitions=copy.deepcopy(SAMPLE_TOOL_DEFINITIONS),
    )
    manager = GameStateManager()
    manager.initialize(
        initial_state=copy.deepcopy(TEMPLATE),
        deck_cards=DECK,
        removed_card=REMOVED,
        revealed_cards=REVEALED,
        player_hands=HANDS,
    )
    return ToolRegistry(game_config, manager)


def test_dispatch_captures_intent_without_executing() -> None:
    dispatch = PlayerActionDispatch(copy.deepcopy(SAMPLE_TOOL_DEFINITIONS))
    proposal = dispatch.dispatch(
        "play_guard",
        {
            "target_player": "player_2",
            "named_card": "Priest",
            "reasoning": "private thought",
            "public_statement": "I accuse you!",
        },
    )
    assert proposal == {
        "action_type": "play_guard",
        "parameters": {"target_player": "player_2", "named_card": "Priest"},
        "reasoning": "private thought",
        "public_statement": "I accuse you!",
    }


def test_get_tool_schemas_subset_all_and_unknown() -> None:
    dispatch = PlayerActionDispatch(copy.deepcopy(SAMPLE_TOOL_DEFINITIONS))
    assert _tool_names(dispatch.get_tool_schemas()) == {"draw_card", "play_guard", "play_prince"}
    assert _tool_names(dispatch.get_tool_schemas(["play_guard"])) == {"play_guard"}
    with pytest.raises(ValueError, match="unknown action tool"):
        dispatch.get_tool_schemas(["nonexistent"])


def test_gm_tools_include_set_game_state(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    names = _tool_names(registry.get_gm_tools())
    assert names == {"query_rulebook", "get_game_state", "set_game_state"}


def test_player_tools_exclude_set_game_state(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    names = _tool_names(registry.get_player_tools(["play_guard", "draw_card"]))
    assert "set_game_state" not in names
    assert {"query_rulebook", "get_game_state", "play_guard", "draw_card"} <= names


def test_shared_and_action_accessors(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert _tool_names(registry.get_shared_tool_schemas()) == {"query_rulebook", "get_game_state"}
    assert _tool_names(registry.get_action_tool_schemas(["play_prince"])) == {"play_prince"}


def test_player_cannot_set_game_state(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    new_state = registry.get_state_tool.execute("gm")
    with pytest.raises(ValueError, match="only the GM"):
        registry.execute_tool("set_game_state", {"new_state": new_state}, caller_id="player_1")


def test_gm_can_set_game_state(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    new_state = registry.get_state_tool.execute("gm")
    new_state["round_number"] = 2
    result = registry.execute_tool("set_game_state", {"new_state": new_state}, caller_id="gm")
    assert isinstance(result, dict)
    assert result["round_number"] == 2


def test_execute_tool_routes_player_action(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    result = registry.execute_tool(
        "play_guard",
        {
            "target_player": "player_2",
            "named_card": "Priest",
            "reasoning": "r",
            "public_statement": "p",
        },
        caller_id="player_1",
    )
    assert isinstance(result, dict)
    assert result["action_type"] == "play_guard"
    assert result["parameters"] == {"target_player": "player_2", "named_card": "Priest"}


def test_execute_tool_get_game_state_filters_by_caller(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    player_view = registry.execute_tool("get_game_state", {"reasoning": "r"}, caller_id="player_1")
    assert isinstance(player_view, dict)
    assert player_view["players"]["player_2"]["hand"] == ["HIDDEN"]
    assert "deck" not in player_view


@pytest.mark.integration
def test_rulebook_query_returns_relevant_chunks(openai_client) -> None:
    from playtest.config import get_settings
    from playtest.ingestion.schemas import GameConfig
    from playtest.tools.rulebook import RulebookTool

    config_dir = Path(get_settings().game_configs_dir) / "love_letter_classic"
    if not config_dir.exists():
        pytest.skip("love_letter_classic config not present; run ingestion first")

    config = GameConfig.load(str(config_dir))
    tool = RulebookTool(config.config_dir, config.game_name)
    result = tool.query("What does the Guard do?")

    assert isinstance(result, str)
    assert result.strip()
    assert "guard" in result.lower()
