"""Rule-level acceptance tests for 6 nimmt! engines.

Parameterized over engine instances so the same suite can validate both the
hand-written reference engine and (later) a generated one. Crafted states use
the canonical 6 nimmt! state shape; resolution is exercised with hand-built
board positions rather than only random play.
"""

import pytest

from playtest.engine import Action
from playtest.games import six_nimmt

ENGINES = [six_nimmt.Game()]


@pytest.fixture(params=ENGINES, ids=lambda e: e.game_name)
def engine(request):
    return request.param


def make_state(
    hands: dict[str, list[int]],
    rows: list[list[int]],
    totals: dict[str, int] | None = None,
    piles: dict[str, list[int]] | None = None,
    round_number: int = 1,
) -> dict:
    seats = list(hands)
    return {
        "num_players": len(seats),
        "round_number": round_number,
        "phase": "commit",
        "hands": {seat: list(hands[seat]) for seat in seats},
        "rows": [list(row) for row in rows],
        "piles": {seat: list((piles or {}).get(seat, [])) for seat in seats},
        "committed": {},
        "pending": None,
        "totals": {seat: (totals or {}).get(seat, 0) for seat in seats},
        "rng_seed": 999,
        "game_over": False,
        "winners": [],
    }


def commit(engine, state, choices: dict[str, int]):
    """Apply one simultaneous commit turn from a {seat: card} mapping."""
    actions = []
    for seat in engine.to_act(state):
        legal = engine.legal_actions(state, seat)
        match = next(a for a in legal if a.name == "play_card" and a.args["card"] == choices[seat])
        actions.append(match)
    return engine.apply(state, actions)


def take(engine, state, row: int):
    """Resolve a pending choose_row decision by taking the given row index."""
    seat = engine.to_act(state)[0]
    legal = engine.legal_actions(state, seat)
    match = next(a for a in legal if a.name == "take_row" and a.args["row"] == row)
    return engine.apply(state, [match])


def all_cards(state) -> list[int]:
    cards = [c for hand in state["hands"].values() for c in hand]
    cards += [c for row in state["rows"] for c in row]
    return cards


# ------------------------------------------------------------- bull heads


@pytest.mark.parametrize(
    "card,points",
    [
        (55, 7),
        (11, 5),
        (22, 5),
        (44, 5),
        (99, 5),
        (10, 3),
        (20, 3),
        (100, 3),
        (5, 2),
        (15, 2),
        (45, 2),
        (95, 2),
        (1, 1),
        (2, 1),
        (7, 1),
        (103, 1),
        (104, 1),
    ],
)
def test_bull_heads_values(card, points):
    assert six_nimmt.bull_heads(card) == points


def test_bull_heads_covers_whole_deck():
    # Every card scores at least one bull head; the total is a fixed constant.
    assert all(six_nimmt.bull_heads(c) >= 1 for c in range(1, 105))
    # 7 (the 55) + 8×5 (other ×11) + 10×3 (×10) + 9×2 (other ×5) + 76×1 = 171.
    assert sum(six_nimmt.bull_heads(c) for c in range(1, 105)) == 171


# ------------------------------------------------------------------ setup


@pytest.mark.parametrize("num_players", [2, 10])
def test_setup_shape(engine, num_players):
    state, events = engine.setup(num_players, seed=7)
    assert all(len(hand) == 10 for hand in state["hands"].values())
    assert len(state["rows"]) == 4
    assert all(len(row) == 1 for row in state["rows"])
    # No duplicates: dealt cards are distinct and drawn from 1..104.
    cards = all_cards(state)
    assert len(cards) == len(set(cards)) == num_players * 10 + 4
    assert set(cards) <= set(range(1, 105))
    assert engine.to_act(state) == [f"player_{i}" for i in range(1, num_players + 1)]
    assert events


def test_ten_player_setup_uses_the_full_deck(engine):
    state, _ = engine.setup(10, seed=3)
    assert set(all_cards(state)) == set(range(1, 105))


def test_setup_is_seed_deterministic(engine):
    assert engine.setup(4, seed=9) == engine.setup(4, seed=9)
    assert engine.setup(4, seed=9) != engine.setup(4, seed=10)


def test_scores_reported_every_status_call(engine):
    state, _ = engine.setup(3, seed=1)
    status = engine.status(state)
    assert status.scores == {"player_1": 0.0, "player_2": 0.0, "player_3": 0.0}
    assert not status.over


# ------------------------------------------------------------- placement


def test_closest_lower_row_placement(engine):
    state = make_state(
        {"player_1": [55, 100], "player_2": [33, 101]},
        rows=[[10], [30], [50], [70]],
    )
    new_state, _ = commit(engine, state, {"player_1": 55, "player_2": 33})
    # 33 (resolved first) joins the row ending in 30; 55 joins the row ending in 50.
    assert new_state["rows"][1] == [30, 33]
    assert new_state["rows"][2] == [50, 55]
    assert new_state["phase"] == "commit"
    assert new_state["totals"] == {"player_1": 0, "player_2": 0}


def test_resolution_is_ascending_across_seats(engine):
    # Resolving by seat order (12 before 11) would strand 11; ascending does not.
    state = make_state(
        {"player_1": [12, 100], "player_2": [11, 101]},
        rows=[[10], [20], [30], [40]],
    )
    new_state, _ = commit(engine, state, {"player_1": 12, "player_2": 11})
    assert new_state["rows"][0] == [10, 11, 12]
    assert new_state["phase"] == "commit"  # nobody was forced to take


def test_sixth_card_forces_the_owner_to_take_the_row(engine):
    state = make_state(
        {"player_1": [6, 51], "player_2": [100, 52]},
        rows=[[1, 2, 3, 4, 5], [30], [50], [70]],
    )
    new_state, events = commit(engine, state, {"player_1": 6, "player_2": 100})
    # 1+1+1+1+2 = 6 bull heads scooped; the row restarts with the played card.
    assert new_state["piles"]["player_1"] == [1, 2, 3, 4, 5]
    assert new_state["totals"]["player_1"] == 6
    assert new_state["rows"][0] == [6]
    assert any("sixth" in e.text for e in events)


# ------------------------------------------------------------- choose_row


def test_too_low_card_opens_choose_row_phase(engine):
    state = make_state(
        {"player_1": [5, 100], "player_2": [60, 101]},
        rows=[[10], [30], [50], [70]],
    )
    mid, _ = commit(engine, state, {"player_1": 5, "player_2": 60})
    assert mid["phase"] == "choose_row"
    assert engine.to_act(mid) == ["player_1"]
    legal = engine.legal_actions(mid, "player_1")
    assert {a.name for a in legal} == {"take_row"}
    assert sorted(a.args["row"] for a in legal) == [0, 1, 2, 3]
    # Nobody else may act mid-resolution.
    assert engine.legal_actions(mid, "player_2") == []


def test_choose_row_resolves_then_resumes(engine):
    state = make_state(
        {"player_1": [5, 100], "player_2": [60, 101]},
        rows=[[10], [30], [50], [70]],
    )
    mid, _ = commit(engine, state, {"player_1": 5, "player_2": 60})
    done, _ = take(engine, mid, row=0)
    # player_1 scooped row 0 (a single 10 → 3 bull heads) and restarted it with 5.
    assert done["totals"]["player_1"] == 3
    assert done["rows"][0] == [5]
    # Resolution resumed: player_2's 60 landed on the row ending in 50.
    assert done["rows"][2] == [50, 60]
    assert done["phase"] == "commit"


# ------------------------------------------------------------- round flow


def test_round_redeals_when_nobody_reaches_threshold(engine):
    state = make_state(
        {"player_1": [90], "player_2": [91]},
        rows=[[50], [60], [70], [80]],
    )
    new_state, _ = commit(engine, state, {"player_1": 90, "player_2": 91})
    assert not new_state["game_over"]
    assert new_state["round_number"] == 2
    assert all(len(hand) == 10 for hand in new_state["hands"].values())
    assert all(len(row) == 1 for row in new_state["rows"])
    assert new_state["phase"] == "commit"


def test_game_ends_with_lowest_total_winning(engine):
    state = make_state(
        {"player_1": [45], "player_2": [103]},
        rows=[[55, 11, 22, 33, 44], [100], [101], [102]],
        totals={"player_1": 60, "player_2": 30},
    )
    new_state, _ = commit(engine, state, {"player_1": 45, "player_2": 103})
    # player_1's 45 was the sixth card: +27 bull heads → 87, crossing 66.
    status = engine.status(new_state)
    assert status.over
    assert status.winners == ("player_2",)
    assert status.scores == {"player_1": 87.0, "player_2": 30.0}
    assert engine.to_act(new_state) == []


def test_game_end_ties_share_the_win(engine):
    state = make_state(
        {"player_1": [41], "player_2": [42], "player_3": [43]},
        rows=[[10], [20], [30], [40]],
        totals={"player_1": 70, "player_2": 40, "player_3": 40},
    )
    new_state, _ = commit(engine, state, {"player_1": 41, "player_2": 42, "player_3": 43})
    status = engine.status(new_state)
    assert status.over
    assert status.winners == ("player_2", "player_3")


# ------------------------------------------------------------- hidden info


def test_observation_hides_other_hands_and_commitments(engine):
    state, _ = engine.setup(3, seed=4)
    view = engine.observe(state, "player_2")
    assert sorted(view["your_hand"]) == sorted(state["hands"]["player_2"])
    assert view["your_committed_card"] is None
    for other in ("player_1", "player_3"):
        assert "hand" not in view["players"][other]
        assert view["players"][other]["hand_count"] == 10
        assert view["players"][other]["committed"] is False
    # No zone in the view leaks another seat's concealed cards.
    assert "hands" not in view


def test_spectator_sees_everything(engine):
    state, _ = engine.setup(2, seed=5)
    view = engine.observe(state, "spectator")
    assert view["hands"] == state["hands"]
    assert view["rows"] == state["rows"]


def test_illegal_commit_is_rejected(engine):
    state = make_state(
        {"player_1": [5, 100], "player_2": [60, 101]},
        rows=[[10], [30], [50], [70]],
    )
    bogus = Action(seat="player_1", name="play_card", args={"card": 99})
    with pytest.raises(ValueError):
        # player_2 also commits legally; player_1's card is not in hand.
        legal2 = engine.legal_actions(state, "player_2")
        engine.apply(state, [bogus, legal2[0]])
