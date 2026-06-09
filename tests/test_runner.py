"""Tests for the multi-game runner (no API; run_game is monkeypatched)."""

import json
from pathlib import Path

import playtest.runner as runner
from playtest.runner import run_multiple_games


def _fake_log(winner: str) -> dict:
    return {
        "session_id": "s",
        "game_name": "Love Letter",
        "variant": "classic",
        "num_players": 2,
        "seed": 0,
        "start_time": None,
        "end_time": None,
        "winner": winner,
        "rounds_played": 1,
        "total_turns": 2,
        "archetypes": ["default", "default"],
        "rule_queries": [],
        "events": [
            {"type": "game_start", "state": {"players": {}}},
            {"type": "round_end", "round_number": 1, "winner": winner, "winning_card": "Guard"},
        ],
    }


def test_run_multiple_games_writes_logs_and_aggregates(tmp_path, monkeypatch) -> None:
    def fake_run_game(
        game_config_name, num_players=2, seed=None, log_file=None, verbose=False, archetypes=None
    ):
        Path(log_file).write_text(json.dumps(_fake_log("player_1")), encoding="utf-8")
        return {"final_state": {}, "summary": {}}

    monkeypatch.setattr(runner, "run_game", fake_run_game)

    analytics = run_multiple_games(
        "love_letter_classic", num_games=3, num_players=2, output_dir=str(tmp_path)
    )

    for name in ("game_001.json", "game_002.json", "game_003.json"):
        assert (tmp_path / name).exists()
    assert analytics["games_played"] == 3


def test_run_multiple_games_skips_and_continues_on_failure(tmp_path, monkeypatch) -> None:
    def fake_run_game(
        game_config_name, num_players=2, seed=None, log_file=None, verbose=False, archetypes=None
    ):
        if seed == 1:  # seed_start 0 -> seeds 0, 1, 2; fail the middle game
            raise RuntimeError("boom")
        Path(log_file).write_text(json.dumps(_fake_log("player_2")), encoding="utf-8")
        return {"final_state": {}, "summary": {}}

    monkeypatch.setattr(runner, "run_game", fake_run_game)

    analytics = run_multiple_games(
        "love_letter_classic", num_games=3, num_players=2, output_dir=str(tmp_path)
    )

    assert (tmp_path / "game_001.json").exists()
    assert (tmp_path / "game_003.json").exists()
    assert (tmp_path / "game_002_error.json").exists()  # failed game logged, batch continued
    assert analytics["games_played"] == 2
