"""Runner tests: engine resolution and run wiring (no network)."""

import json

import pytest

from playtest.checkpoint import Checkpoint
from playtest.config import get_settings
from playtest.errors import PlaytestError
from playtest.runner import _effective_max_steps, resolve_game, run_game


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
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


def test_effective_max_steps_prefers_meta_max_decisions(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"max_decisions": 77}), encoding="utf-8")
    assert _effective_max_steps(tmp_path, get_settings()) == 77


def test_effective_max_steps_no_config_dir_falls_back():
    settings = get_settings()
    assert _effective_max_steps(None, settings) == settings.max_steps


def test_effective_max_steps_missing_meta_json_falls_back(tmp_path):
    settings = get_settings()
    assert _effective_max_steps(tmp_path, settings) == settings.max_steps


def test_effective_max_steps_zero_max_decisions_falls_back(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"max_decisions": 0}), encoding="utf-8")
    settings = get_settings()
    assert _effective_max_steps(tmp_path, settings) == settings.max_steps


def test_effective_max_steps_old_meta_without_key_falls_back(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"game_name": "Old Game"}), encoding="utf-8")
    settings = get_settings()
    assert _effective_max_steps(tmp_path, settings) == settings.max_steps


def test_effective_max_steps_malformed_meta_json_falls_back(tmp_path):
    (tmp_path / "meta.json").write_text("{not json", encoding="utf-8")
    settings = get_settings()
    assert _effective_max_steps(tmp_path, settings) == settings.max_steps


def _spy_run_session(monkeypatch):
    """Patch playtest.runner.run_session, returning the kwargs it was called with."""
    from playtest import runner

    calls = []

    def fake_run_session(*args, **kwargs):
        calls.append(kwargs)
        return {"final_state": {}}

    monkeypatch.setattr(runner, "run_session", fake_run_session)
    return calls


def test_run_game_uses_meta_max_decisions_as_max_steps(monkeypatch, tmp_path):
    from playtest import runner

    engine, _ = resolve_game("playtest.games.love_letter")
    (tmp_path / "meta.json").write_text(json.dumps({"max_decisions": 77}), encoding="utf-8")
    monkeypatch.setattr(runner, "resolve_game", lambda game_ref: (engine, tmp_path))
    calls = _spy_run_session(monkeypatch)

    runner.run_game("playtest.games.love_letter", num_players=2)

    assert calls[0]["max_steps"] == 77


def test_run_game_falls_back_to_settings_max_steps_without_config_dir(monkeypatch):
    from playtest import runner

    calls = _spy_run_session(monkeypatch)

    runner.run_game("playtest.games.love_letter", num_players=2)

    assert calls[0]["max_steps"] == get_settings().max_steps


def test_resume_game_uses_meta_max_decisions_as_max_steps(monkeypatch, tmp_path):
    from playtest import runner

    engine, _ = resolve_game("playtest.games.love_letter")
    (tmp_path / "meta.json").write_text(json.dumps({"max_decisions": 77}), encoding="utf-8")
    monkeypatch.setattr(runner, "resolve_game", lambda game_ref: (engine, tmp_path))

    cp = Checkpoint(
        game_ref="playtest.games.love_letter",
        num_players=2,
        seed=1,
        archetypes=["default", "default"],
        session_id="s",
        step=0,
        state={},
        buffers={"player_1": [], "player_2": []},
        notebooks={"player_1": "", "player_2": ""},
    )
    monkeypatch.setattr(runner, "load_checkpoint", lambda path: cp)
    calls = _spy_run_session(monkeypatch)

    runner.resume_game("unused-checkpoint-path.json")

    assert calls[0]["max_steps"] == 77
