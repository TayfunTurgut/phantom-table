"""Tool registry tests: schema assembly, routing, caller filtering, rulebook cache."""

import copy

import pytest

from playtest.rules import GameRules
from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry
from playtest.tools.actions import PlayerActionDispatch
from playtest.tools.rulebook import RulebookTool

from .fixtures import TOOL_DEFINITIONS, sample_config


def _tool_names(schemas: list[dict]) -> set[str]:
    return {schema["function"]["name"] for schema in schemas}


def _registry(tmp_path) -> ToolRegistry:
    config = sample_config(config_dir=str(tmp_path))
    manager = GameStateManager()
    manager.initialize(GameRules(config).setup(2, seed=1), config.game_spec.visibility)
    return ToolRegistry(config, manager)


def test_dispatch_captures_intent_without_executing() -> None:
    dispatch = PlayerActionDispatch(copy.deepcopy(TOOL_DEFINITIONS))
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
    dispatch = PlayerActionDispatch(copy.deepcopy(TOOL_DEFINITIONS))
    assert _tool_names(dispatch.get_tool_schemas()) == set(TOOL_DEFINITIONS)
    assert _tool_names(dispatch.get_tool_schemas(["play_guard"])) == {"play_guard"}
    with pytest.raises(ValueError, match="unknown action tool"):
        dispatch.get_tool_schemas(["nonexistent"])


def test_gm_tools_include_set_game_state(tmp_path) -> None:
    registry = _registry(tmp_path)
    names = _tool_names(registry.get_gm_tools())
    assert names == {
        "query_rulebook",
        "get_game_state",
        "set_game_state",
        "finish_resolution",
    }


def test_finish_resolution_schema_uses_spec_phases_and_flags(tmp_path) -> None:
    registry = _registry(tmp_path)
    schema = next(
        t for t in registry.get_gm_tools() if t["function"]["name"] == "finish_resolution"
    )
    properties = schema["function"]["parameters"]["properties"]
    # next_phase enum comes from the ingested spec, not hardcoded game phases.
    assert properties["next_phase"]["enum"] == ["draw", "play"]
    for flag in ("turn_ended", "round_ended", "game_ended"):
        assert properties[flag]["type"] == "boolean"
    assert properties["winners"]["items"] == {"type": "string"}


def test_player_tools_exclude_set_game_state(tmp_path) -> None:
    registry = _registry(tmp_path)
    names = _tool_names(registry.get_player_tools(["play_guard", "draw_card"]))
    assert "set_game_state" not in names
    assert {"query_rulebook", "get_game_state", "play_guard", "draw_card"} <= names


def test_shared_and_action_accessors(tmp_path) -> None:
    registry = _registry(tmp_path)
    assert _tool_names(registry.get_shared_tool_schemas()) == {"query_rulebook", "get_game_state"}
    assert _tool_names(registry.get_action_tool_schemas(["play_prince"])) == {"play_prince"}


def test_player_cannot_set_game_state(tmp_path) -> None:
    registry = _registry(tmp_path)
    new_state = registry.get_state_tool.execute("gm")
    with pytest.raises(ValueError, match="only the GM"):
        registry.execute_tool("set_game_state", {"new_state": new_state}, caller_id="player_1")


def test_gm_can_set_game_state(tmp_path) -> None:
    registry = _registry(tmp_path)
    new_state = registry.get_state_tool.execute("gm")
    new_state["round_number"] = 2
    result = registry.execute_tool("set_game_state", {"new_state": new_state}, caller_id="gm")
    assert isinstance(result, dict)
    assert result["round_number"] == 2


def test_execute_tool_routes_player_action(tmp_path) -> None:
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


def test_execute_tool_get_game_state_filters_by_caller(tmp_path) -> None:
    registry = _registry(tmp_path)
    player_view = registry.execute_tool("get_game_state", {"reasoning": "r"}, caller_id="player_1")
    assert isinstance(player_view, dict)
    assert player_view["players"]["player_2"]["hand"] == ["HIDDEN"]
    assert "deck" not in player_view


# --- Rulebook query cache ------------------------------------------------------


def test_rulebook_query_cache_skips_embedding_on_repeat(monkeypatch, tmp_path) -> None:
    embeds: list[str] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        embeds.extend(texts)
        return [[0.0] for _ in texts]

    class _FakeCollection:
        def query(self, **_: object) -> dict:
            return {"documents": [["chunk text"]], "metadatas": [[{"section": "Rules"}]]}

    monkeypatch.setattr("playtest.tools.rulebook._embed_texts", fake_embed)
    tool = RulebookTool(str(tmp_path), "Sample Letters")
    tool._collection = _FakeCollection()

    first = tool.query("how does drawing work")
    again = tool.query("how does drawing work")
    other = tool.query("how does scoring work")

    assert first == again
    assert "chunk text" in other
    assert embeds == ["how does drawing work", "how does scoring work"]  # 2 embeds for 3 calls
    assert tool.get_query_log() == [
        "how does drawing work",
        "how does drawing work",
        "how does scoring work",
    ]
