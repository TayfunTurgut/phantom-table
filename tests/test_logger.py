import json

from playtest.ui.logger import GameLogger

STATE = {
    "game_name": "Love Letter",
    "variant": "classic",
    "num_players": 2,
    "tokens_to_win": 7,
    "round_number": 1,
    "players": {
        "player_1": {"tokens": 0, "is_eliminated": False},
        "player_2": {"tokens": 0, "is_eliminated": False},
    },
}


def _play_sample_game(logger: GameLogger) -> None:
    logger.log_event(
        "game_start",
        {"session_id": "sess-1", "seed": 42, "state": STATE, "narration": "A new game begins."},
    )
    logger.log_event("turn_start", {"player": "player_1", "turn_index": 1, "phase": "draw"})
    logger.log_event(
        "player_action",
        {"player": "player_1", "action_type": "draw_card", "parameters": {}},
    )
    logger.log_event("gm_validation", {"player": "player_1", "is_valid": True})
    logger.log_event(
        "player_action",
        {"player": "player_1", "action_type": "play_guard", "parameters": {"target": "player_2"}},
    )
    logger.log_event(
        "gm_validation",
        {"player": "player_1", "is_valid": False, "error_message": "player_2 is protected"},
    )
    logger.log_event(
        "player_action",
        {"player": "player_1", "action_type": "play_guard", "parameters": {"target": "player_2"}},
    )
    logger.log_event("gm_validation", {"player": "player_1", "is_valid": True})
    logger.log_event("round_end", {"round_number": 1, "winner": "player_1", "scores": {}})
    logger.log_event(
        "game_end", {"winner": "player_1", "total_turns": 4, "rounds_played": 1, "final_scores": {}}
    )


def test_every_event_has_type_and_timestamp() -> None:
    logger = GameLogger()
    _play_sample_game(logger)

    assert logger.log["events"], "events should be recorded"
    for event in logger.log["events"]:
        assert "type" in event
        assert isinstance(event.get("timestamp"), str) and event["timestamp"]


def test_header_fields_populated_from_events() -> None:
    logger = GameLogger()
    _play_sample_game(logger)

    assert logger.log["session_id"] == "sess-1"
    assert logger.log["game_name"] == "Love Letter"
    assert logger.log["variant"] == "classic"
    assert logger.log["num_players"] == 2
    assert logger.log["seed"] == 42
    assert logger.log["winner"] == "player_1"
    assert logger.log["start_time"]
    assert logger.log["end_time"]


def test_save_and_from_file_round_trips(tmp_path) -> None:
    logger = GameLogger()
    _play_sample_game(logger)
    path = tmp_path / "game.json"
    logger.save(str(path))

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == logger.log

    reloaded = GameLogger.from_file(str(path))
    assert reloaded.log == logger.log
    assert reloaded.get_summary() == logger.get_summary()


def test_summary_counts_rejections_and_actions() -> None:
    logger = GameLogger()
    _play_sample_game(logger)
    summary = logger.get_summary()

    assert summary["winner"] == "player_1"
    assert summary["rounds_played"] == 1
    assert summary["total_turns"] == 4
    assert summary["actions_rejected"]["count"] == 1
    assert len(summary["actions_rejected"]["details"]) == 1
    # Lossless raw action_type keys (play_ prefix not stripped in stored data).
    assert summary["most_played_actions"]["play_guard"] == 2
    assert summary["most_played_actions"]["draw_card"] == 1
    assert summary["average_turns_per_round"] == 4.0
