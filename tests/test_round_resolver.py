"""Unit tests for the deterministic round-end resolver (no API, pure arithmetic)."""

import pytest

from playtest.agents.gm import resolve_round


def _player(
    hand: list[str], discards: list[str] | None = None, tokens: int = 0, eliminated: bool = False
) -> dict:
    return {
        "hand": hand,
        "hand_count": len(hand),
        "discards": discards or [],
        "tokens": tokens,
        "is_eliminated": eliminated,
    }


def test_single_winner_by_rank() -> None:
    players = {
        "player_1": _player(["Princess"], tokens=2),
        "player_2": _player(["Guard"], tokens=1),
    }
    result = resolve_round(players)
    assert result["winners"] == ["player_1"]
    assert result["winning_card"] == "Princess"
    assert result["scores"] == {"player_1": 3, "player_2": 1}


def test_eliminated_players_excluded() -> None:
    players = {
        "player_1": _player(["Princess"], eliminated=True),  # highest card but out
        "player_2": _player(["Guard"]),
        "player_3": _player(["Baron"]),
    }
    result = resolve_round(players)
    assert result["winners"] == ["player_3"]  # Baron(3) beats Guard(1) among survivors
    assert result["winning_card"] == "Baron"


def test_tie_broken_by_discard_sum() -> None:
    players = {
        "player_1": _player(["King"], discards=["Guard", "Guard"]),  # discard sum 2
        "player_2": _player(["King"], discards=["Prince", "Baron"]),  # discard sum 8
    }
    result = resolve_round(players)
    assert result["winners"] == ["player_2"]
    assert result["scores"]["player_2"] == 1
    assert result["scores"]["player_1"] == 0


def test_full_tie_is_shared_each_gets_a_token() -> None:
    players = {
        "player_1": _player(["King"], discards=["Guard"], tokens=1),
        "player_2": _player(["King"], discards=["Guard"], tokens=1),
    }
    result = resolve_round(players)
    assert result["winners"] == ["player_1", "player_2"]
    assert result["scores"] == {"player_1": 2, "player_2": 2}


def test_no_survivors_raises() -> None:
    players = {"player_1": _player(["Guard"], eliminated=True)}
    with pytest.raises(ValueError, match="no surviving players"):
        resolve_round(players)


def test_survivor_with_wrong_hand_length_raises() -> None:
    players = {
        "player_1": _player(["Guard", "King"]),  # two cards at round end = corruption
        "player_2": _player(["Baron"]),
    }
    with pytest.raises(ValueError, match="exactly one card"):
        resolve_round(players)
