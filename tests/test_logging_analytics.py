"""Logger and analytics tests over the new event schema."""

import json

import pytest

from playtest.analytics import analyze_games
from playtest.ui.logger import GameLogger


def _play_fake_game(logger: GameLogger, winner: str = "player_1") -> None:
    logger.log_event(
        "game_start",
        {"session_id": "s", "seed": 1, "game_name": "Love Letter", "num_players": 2, "state": {}},
    )
    logger.log_event(
        "decision",
        {
            "step": 1,
            "seat": "player_1",
            "action": "play_guard",
            "args": {},
            "label": "Play Guard",
            "reasoning": "r",
            "table_talk": None,
            "confused": False,
            "num_legal_actions": 9,
        },
    )
    logger.log_event(
        "engine_event", {"step": 1, "text": "player_1 played Guard.", "visible_to": None}
    )
    logger.log_event(
        "decision",
        {
            "step": 2,
            "seat": "player_2",
            "action": "play_priest",
            "args": {},
            "label": "Play Priest",
            "reasoning": "r",
            "table_talk": None,
            "confused": True,
            "num_legal_actions": 4,
        },
    )
    logger.log_event("player_confusion", {"step": 2, "seat": "player_2"})
    logger.log_event(
        "game_end",
        {
            "winners": [winner],
            "scores": {"player_1": 7.0, "player_2": 3.0},
            "total_steps": 2,
            "final_state": {},
        },
    )


def test_logger_header_and_summary():
    logger = GameLogger()
    logger.set_run_metadata(archetypes=["default", "aggressive"])
    _play_fake_game(logger)

    assert logger.log["game_name"] == "Love Letter"
    assert logger.log["winners"] == ["player_1"]
    summary = logger.get_summary()
    assert summary["most_played_actions"] == {"play_guard": 1, "play_priest": 1}
    assert summary["player_confusions"]["count"] == 1
    assert summary["total_steps"] == 2


def test_logger_round_trips_through_file(tmp_path):
    logger = GameLogger()
    _play_fake_game(logger)
    path = tmp_path / "game.json"
    logger.save(str(path))
    loaded = GameLogger.from_file(str(path))
    assert loaded.log == logger.log


def test_analyze_games_aggregates_directory(tmp_path):
    for i, winner in enumerate(["player_1", "player_1", "player_2"]):
        logger = GameLogger()
        logger.set_run_metadata(archetypes=["aggressive", "default"])
        _play_fake_game(logger, winner=winner)
        logger.save(str(tmp_path / f"game_{i}.json"))
    # A partial/error log must be skipped, not crash aggregation.
    (tmp_path / "broken.json").write_text(json.dumps({"seed": 9, "error": "boom"}))

    analytics = analyze_games(str(tmp_path))
    assert analytics["games_played"] == 3
    assert analytics["win_rates"]["player_1"] == pytest.approx(2 / 3)
    assert analytics["action_frequency"] == {"play_guard": 3, "play_priest": 3}
    assert analytics["confusion_rate"] == pytest.approx(0.5)
    arch = analytics["archetype_performance"]
    assert arch["aggressive"]["games"] == 3
    assert arch["aggressive"]["wins"] == 2
