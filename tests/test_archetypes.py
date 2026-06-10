"""Tests for player archetypes and their application to the player system prompt.

NOTE: archetype *behavioral* divergence (e.g. aggressive players attack more often) is
verified MANUALLY via bulk runs + analytics — it is not asserted here, since it would
need many paid games to be statistically meaningful.
"""

import pytest

from playtest.agents.archetypes import ARCHETYPES, apply_archetype
from playtest.agents.player import PlayerAgent
from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry

from .fixtures import sample_config


def test_default_archetype_is_noop() -> None:
    base = "You are playing Sample Letters."
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


def test_overlays_are_game_agnostic() -> None:
    """Archetypes must reference styles of play, never specific game components."""
    game_terms = ("Guard", "Baron", "Prince", "Handmaid", "Countess", "Princess", "token")
    for name, overlay in ARCHETYPES.items():
        for term in game_terms:
            assert term not in overlay, f"archetype {name!r} hardcodes {term!r}"


def test_player_agent_system_prompt_includes_overlay(settings) -> None:
    config = sample_config()
    registry = ToolRegistry(config, GameStateManager())

    agent = PlayerAgent("player_1", config, registry, object(), archetype="cautious")
    assert agent.archetype == "cautious"
    assert "cautious player" in agent.system_prompt
    assert "player_1" in agent.system_prompt
