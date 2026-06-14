"""Runner tests: engine resolution and run wiring (no network)."""

import pytest

from playtest.config import get_settings
from playtest.errors import PlaytestError
from playtest.runner import resolve_game, run_game


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_resolve_game_by_module_path():
    engine, config_dir = resolve_game("playtest.games.love_letter")
    assert engine.game_name == "Love Letter"
    assert config_dir is None


def test_resolve_game_unknown_raises_friendly_error():
    with pytest.raises(PlaytestError, match="could not resolve game"):
        resolve_game("no.such.module")


def test_run_game_rejects_bad_player_count(monkeypatch):
    with pytest.raises(ValueError, match="players"):
        run_game("playtest.games.love_letter", num_players=9)


def test_run_game_rejects_archetype_mismatch():
    with pytest.raises(ValueError, match="archetypes"):
        run_game(
            "playtest.games.love_letter",
            num_players=2,
            archetypes=["aggressive"],
        )
