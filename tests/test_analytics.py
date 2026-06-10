"""Analytics tests driven by hand-built fixture log files (no API, no game knowledge)."""

import json
from pathlib import Path

from rich.console import Console

from playtest.analytics import analyze_games, print_analytics_report


def _players() -> dict:
    return {
        pid: {"hand_count": 1, "tokens": 0, "is_eliminated": False}
        for pid in ("player_1", "player_2")
    }


def _game_start() -> dict:
    return {"type": "game_start", "state": {"players": _players()}}


def _write(path: Path, log: dict) -> None:
    path.write_text(json.dumps(log), encoding="utf-8")


def _base(winner, rounds, turns, archetypes, rule_queries, events, **extra) -> dict:
    log = {
        "session_id": "s",
        "game_name": "Sample Letters",
        "variant": "classic",
        "num_players": 2,
        "seed": 1,
        "start_time": None,
        "end_time": None,
        "winner": winner,
        "rounds_played": rounds,
        "total_turns": turns,
        "archetypes": archetypes,
        "rule_queries": rule_queries,
        "events": events,
    }
    log.update(extra)
    return log


def _make_logs(d: Path) -> None:
    # Game 1: player_1 wins cleanly.
    _write(
        d / "game_001.json",
        _base(
            winner="player_1",
            rounds=1,
            turns=3,
            archetypes=["aggressive", "cautious"],
            rule_queries=["How does targeting work?", "Can I target a protected player?"],
            start_time="2026-06-09T10:00:00+00:00",
            end_time="2026-06-09T10:05:00+00:00",
            events=[
                _game_start(),
                {"type": "player_action", "player": "player_1", "action_type": "play_guard"},
                {
                    "type": "gm_validation",
                    "player": "player_1",
                    "is_valid": True,
                    "action_type": "play_guard",
                },
                {"type": "round_end", "round_number": 1, "winner": "player_1"},
            ],
        ),
    )
    # Game 2: player_2 wins; one rejected proposal among three validations.
    _write(
        d / "game_002.json",
        _base(
            winner="player_2",
            rounds=2,
            turns=5,
            archetypes=["aggressive", "cautious"],
            rule_queries=["How does targeting work?"],
            events=[
                _game_start(),
                {"type": "player_action", "player": "player_1", "action_type": "play_guard"},
                {
                    "type": "gm_validation",
                    "player": "player_1",
                    "is_valid": True,
                    "action_type": "play_guard",
                },
                {"type": "player_action", "player": "player_2", "action_type": "play_baron"},
                {
                    "type": "gm_validation",
                    "player": "player_2",
                    "is_valid": False,
                    "action_type": "play_baron",
                    "error_message": "Baron cannot target a protected player.",
                },
                {"type": "player_action", "player": "player_2", "action_type": "play_handmaid"},
                {
                    "type": "gm_validation",
                    "player": "player_2",
                    "is_valid": True,
                    "action_type": "play_handmaid",
                },
                {"type": "round_end", "round_number": 2, "winner": "player_2"},
            ],
        ),
    )
    # Game 3: shared multi-winner (comma-joined).
    _write(
        d / "game_003.json",
        _base(
            winner="player_1,player_2",
            rounds=1,
            turns=2,
            archetypes=["aggressive", "cautious"],
            rule_queries=[],
            events=[_game_start(), {"type": "round_end", "round_number": 1}],
        ),
    )
    # An error/partial log (no events) must be skipped.
    _write(d / "game_004_error.json", {"seed": 4, "error": "RuntimeError: boom"})


def test_analyze_games_aggregates(tmp_path) -> None:
    _make_logs(tmp_path)
    a = analyze_games(str(tmp_path))

    assert a["games_played"] == 3  # error log skipped

    assert a["avg_rounds_per_game"] == (1 + 2 + 1) / 3
    assert a["avg_turns_per_game"] == (3 + 5 + 2) / 3
    assert a["avg_turns_per_round"] == 10 / 4

    # Shared win credits both players.
    assert a["win_rates"]["player_1"] == 2 / 3
    assert a["win_rates"]["player_2"] == 2 / 3

    # Action counts use the raw ingested tool names (no game knowledge).
    assert a["action_frequency"]["play_guard"] == 2
    assert a["action_frequency"]["play_baron"] == 1

    assert a["action_rejection_rate"] == 1 / 4
    assert a["rejection_reasons"]["Baron cannot target a protected player."] == 1

    assert a["avg_game_length_minutes"] == 5.0

    assert a["archetype_performance"]["aggressive"]["games"] == 3
    assert a["archetype_performance"]["aggressive"]["wins"] == 2
    assert a["archetype_performance"]["cautious"]["wins"] == 2

    assert a["common_rule_queries"][0] == {"query": "How does targeting work?", "count": 2}


def test_empty_dir_is_safe(tmp_path) -> None:
    a = analyze_games(str(tmp_path))
    assert a["games_played"] == 0
    assert a["win_rates"] == {}
    assert a["avg_turns_per_round"] == 0.0


def test_print_report_runs(tmp_path) -> None:
    _make_logs(tmp_path)
    a = analyze_games(str(tmp_path))
    console = Console(record=True, width=120)
    print_analytics_report(a, console)
    out = console.export_text()
    assert "Playtest Analytics" in out
    assert "Win Rates" in out
    assert "Action Frequency" in out
