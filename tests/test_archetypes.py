"""Tests for player archetypes and their application to the player system prompt.

NOTE: archetype *behavioral* divergence (e.g. aggressive players make more Guard plays,
cautious players play more Handmaids) is verified MANUALLY via bulk runs + analytics — it
is not asserted here, since it would need many paid games to be statistically meaningful.
"""

from pathlib import Path

import pytest

from playtest.agents.archetypes import ARCHETYPES, apply_archetype
from playtest.agents.player import PlayerAgent
from playtest.config import get_settings
from playtest.ingestion.schemas import GameConfig
from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry


def test_default_archetype_is_noop() -> None:
    base = "You are playing Love Letter."
    assert apply_archetype(base, "default") == base


def test_known_archetype_appends_overlay() -> None:
    base = "BASE PROMPT"
    result = apply_archetype(base, "aggressive")
    assert result.startswith(base)
    assert "aggressive player" in result
    assert len(result) > len(base)


def test_unknown_archetype_raises() -> None:
    with pytest.raises(ValueError, match="unknown archetype"):
        apply_archetype("base", "berserker")


def test_all_archetypes_apply_cleanly() -> None:
    base = "BASE"
    for name in ARCHETYPES:
        out = apply_archetype(base, name)
        assert out.startswith(base)


def test_player_agent_system_prompt_includes_overlay(settings) -> None:
    config_dir = Path(get_settings().game_configs_dir) / "love_letter_classic"
    if not config_dir.exists():
        pytest.skip("love_letter_classic config not present; run ingestion first")
    config = GameConfig.load(str(config_dir))
    registry = ToolRegistry(config, GameStateManager())

    agent = PlayerAgent("player_1", config, registry, object(), archetype="cautious")
    assert agent.archetype == "cautious"
    assert "cautious player" in agent.system_prompt
    assert "player_1" in agent.system_prompt
