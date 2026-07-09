"""Rule-level acceptance tests for Love Letter engines.

Parameterized over engine instances so the same suite can validate both the
hand-written reference engine and (later) a generated one. Crafted states use
the canonical Love Letter state shape, which the ingestion digest pins for
generated engines.
"""

import pytest

from playtest.engine import Action
from playtest.games import love_letter

ENGINES = [love_letter.Game()]
SEATS_2P = ["player_1", "player_2"]


@pytest.fixture(params=ENGINES, ids=lambda e: e.game_name)
def engine(request):
    return request.param


def make_state(
    hands: dict[str, list[str]],
    deck: list[str],
    current: str = "player_1",
    discards: dict[str, list[str]] | None = None,
    protected: list[str] | None = None,
    eliminated: list[str] | None = None,
    tokens: dict[str, int] | None = None,
    tokens_to_win: int = 7,
    removed_card: str | None = "Princess",
) -> dict:
    seats = list(hands)
    return {
        "num_players": len(seats),
        "round_number": 1,
        "tokens_to_win": tokens_to_win,
        "current_player": current,
        "players": {
            seat: {
                "hand": list(hands[seat]),
                "discard": list((discards or {}).get(seat, [])),
                "tokens": (tokens or {}).get(seat, 0),
                "eliminated": seat in (eliminated or []),
                "protected": seat in (protected or []),
            }
            for seat in seats
        },
        "deck": list(deck),
        "removed_card": removed_card,
        "revealed_cards": [],
        "rng_seed": 1234,
        "game_over": False,
        "winners": [],
    }


def play(engine, state, name, **args):
    seat = state["current_player"]
    legal = engine.legal_actions(state, seat)
    matches = [
        a for a in legal if a.name == name and all(a.args.get(k) == v for k, v in args.items())
    ]
    assert matches, f"no legal action {name} {args}; legal: {[a.key() for a in legal]}"
    return engine.apply(state, [matches[0]])


def hand(state, seat):
    return state["players"][seat]["hand"]


# ----------------------------------------------------------------- setup


def test_setup_deals_correctly_for_two_players(engine):
    state, _ = engine.setup(2, seed=42)
    assert len(hand(state, "player_1")) == 2  # starter has drawn
    assert len(hand(state, "player_2")) == 1
    assert len(state["revealed_cards"]) == 3
    assert state["removed_card"] is not None
    # 16 cards conserved: 3 in hands, 3 revealed, 1 removed, 9 in deck
    assert len(state["deck"]) == 9
    assert engine.to_act(state) == ["player_1"]


def test_setup_is_seed_deterministic(engine):
    assert engine.setup(2, seed=9) == engine.setup(2, seed=9)
    assert engine.setup(2, seed=9) != engine.setup(2, seed=10)


def test_three_player_setup_reveals_nothing(engine):
    state, _ = engine.setup(3, seed=1)
    assert state["revealed_cards"] == []
    assert len(state["deck"]) == 16 - 1 - 3 - 1  # removed, three hands, starter draw


# ----------------------------------------------------------------- guard


def test_guard_correct_guess_eliminates_target(engine):
    # Three players so the elimination does not immediately end the round.
    state = make_state(
        {"player_1": ["Guard", "Priest"], "player_2": ["Baron"], "player_3": ["King"]},
        deck=["Handmaid", "Prince"],
    )
    new_state, events = play(engine, state, "play_guard", target="player_2", guess="Baron")
    assert new_state["players"]["player_2"]["eliminated"]
    # player_2's Baron is revealed into their discard pile (conservation).
    assert "Baron" in new_state["players"]["player_2"]["discard"]
    assert any("eliminated" in e.text for e in events)


def test_guard_wrong_guess_does_nothing(engine):
    state = make_state(
        {"player_1": ["Guard", "Priest"], "player_2": ["Baron"]},
        deck=["Handmaid", "Prince"],
    )
    new_state, _ = play(engine, state, "play_guard", target="player_2", guess="King")
    assert not new_state["players"]["player_2"]["eliminated"]
    # player_2 keeps the Baron (and has drawn for their own turn).
    assert "Baron" in hand(new_state, "player_2")


def test_guard_cannot_guess_guard(engine):
    state = make_state(
        {"player_1": ["Guard", "Priest"], "player_2": ["Guard"]},
        deck=["Handmaid", "Prince"],
    )
    legal = engine.legal_actions(state, "player_1")
    guesses = {a.args.get("guess") for a in legal if a.name == "play_guard"}
    assert "Guard" not in guesses
    assert guesses == set(love_letter.RANKS) - {"Guard"}


# ----------------------------------------------------------------- priest


def test_priest_reveals_target_hand_privately(engine):
    state = make_state(
        {"player_1": ["Priest", "King"], "player_2": ["Countess"]},
        deck=["Guard", "Guard"],
    )
    _, events = play(engine, state, "play_priest", target="player_2")
    private = [e for e in events if e.visible_to == ("player_1",)]
    assert any("Countess" in e.text for e in private)
    public = [e for e in events if e.visible_to is None]
    assert not any("Countess" in e.text for e in public)


# ----------------------------------------------------------------- baron


def test_baron_eliminates_lower_holder(engine):
    state = make_state(
        {"player_1": ["Baron", "Princess"], "player_2": ["Guard"], "player_3": ["King"]},
        deck=["Handmaid", "Prince"],
    )
    new_state, _ = play(engine, state, "play_baron", target="player_2")
    assert new_state["players"]["player_2"]["eliminated"]
    assert not new_state["players"]["player_1"]["eliminated"]


def test_baron_self_elimination_on_lower_card(engine):
    state = make_state(
        {"player_1": ["Baron", "Guard"], "player_2": ["Princess"], "player_3": ["King"]},
        deck=["Handmaid", "Prince"],
    )
    new_state, _ = play(engine, state, "play_baron", target="player_2")
    assert new_state["players"]["player_1"]["eliminated"]


def test_baron_tie_eliminates_nobody(engine):
    state = make_state(
        {"player_1": ["Baron", "Guard"], "player_2": ["Guard"]},
        deck=["Handmaid", "Prince"],
    )
    new_state, _ = play(engine, state, "play_baron", target="player_2")
    assert not new_state["players"]["player_1"]["eliminated"]
    assert not new_state["players"]["player_2"]["eliminated"]


# ----------------------------------------------------------------- handmaid


def test_handmaid_protects_until_next_turn(engine):
    state = make_state(
        {"player_1": ["Handmaid", "Guard"], "player_2": ["Guard"]},
        deck=["Prince", "Baron", "Priest"],
    )
    new_state, _ = play(engine, state, "play_handmaid")
    assert new_state["players"]["player_1"]["protected"]
    # player_2 (now holding Guard + drawn card) cannot target the protected player.
    legal = engine.legal_actions(new_state, "player_2")
    targeted = [a for a in legal if a.args.get("target") == "player_1"]
    assert targeted == []


def test_protection_expires_when_turn_returns(engine):
    state = make_state(
        {"player_1": ["Handmaid", "Guard"], "player_2": ["Countess"]},
        deck=["Prince", "Baron", "Priest", "King"],
    )
    after_handmaid, _ = play(engine, state, "play_handmaid")
    after_countess, _ = play(engine, after_handmaid, "play_countess")
    assert after_countess["current_player"] == "player_1"
    assert not after_countess["players"]["player_1"]["protected"]


# ----------------------------------------------------------------- prince


def test_prince_forces_discard_and_draw(engine):
    # Target player_3: the turn passes to player_2 next, so player_3's hand
    # stays observable (player_2 auto-draws for their own turn).
    state = make_state(
        {"player_1": ["Prince", "Guard"], "player_2": ["King"], "player_3": ["Countess"]},
        deck=["Baron", "Priest"],
    )
    new_state, _ = play(engine, state, "play_prince", target="player_3")
    assert "Countess" in new_state["players"]["player_3"]["discard"]
    assert len(hand(new_state, "player_3")) == 1
    assert hand(new_state, "player_3") != ["Countess"]


def test_prince_on_princess_eliminates(engine):
    state = make_state(
        {"player_1": ["Prince", "Guard"], "player_2": ["Princess"], "player_3": ["King"]},
        deck=["Baron", "Priest"],
    )
    new_state, _ = play(engine, state, "play_prince", target="player_2")
    assert new_state["players"]["player_2"]["eliminated"]


def test_prince_with_empty_deck_draws_removed_card(engine):
    state = make_state(
        {"player_1": ["Prince", "Guard"], "player_2": ["King"]},
        deck=[],
        removed_card="Countess",
    )
    new_state, _ = play(engine, state, "play_prince", target="player_2")
    # Deck was empty: player_2 takes the set-aside card, then the round ends
    # (deck still empty), showdown between Guard and Countess.
    assert new_state["players"]["player_2"]["tokens"] == 1


def test_prince_can_target_self(engine):
    state = make_state(
        {"player_1": ["Prince", "Guard"], "player_2": ["King"]},
        deck=["Baron", "Priest"],
    )
    new_state, _ = play(engine, state, "play_prince", target="player_1")
    assert "Guard" in new_state["players"]["player_1"]["discard"]


# ----------------------------------------------------------------- king


def test_king_swaps_hands(engine):
    state = make_state(
        {"player_1": ["King", "Guard"], "player_2": ["Princess"]},
        deck=["Baron", "Priest"],
    )
    new_state, _ = play(engine, state, "play_king", target="player_2")
    assert "Princess" in hand(new_state, "player_1")
    # player_2 got the Guard and then drew (their turn started).
    assert "Guard" in hand(new_state, "player_2")


# ----------------------------------------------------------------- countess


def test_countess_forced_with_king_or_prince(engine):
    for partner in ("King", "Prince"):
        state = make_state(
            {"player_1": ["Countess", partner], "player_2": ["Guard"]},
            deck=["Baron", "Priest"],
        )
        legal = engine.legal_actions(state, "player_1")
        assert {a.args["card"] for a in legal} == {"Countess"}


def test_countess_not_forced_with_other_cards(engine):
    state = make_state(
        {"player_1": ["Countess", "Guard"], "player_2": ["Baron"]},
        deck=["Priest", "Handmaid"],
    )
    legal = engine.legal_actions(state, "player_1")
    assert {a.args["card"] for a in legal} == {"Countess", "Guard"}


# ----------------------------------------------------------------- princess


def test_playing_princess_eliminates_self(engine):
    state = make_state(
        {"player_1": ["Princess", "Guard"], "player_2": ["Baron"], "player_3": ["King"]},
        deck=["Priest", "Handmaid"],
    )
    new_state, _ = play(engine, state, "play_princess")
    assert new_state["players"]["player_1"]["eliminated"]


# ----------------------------------------------------------------- round end


def test_round_ends_when_deck_empties_highest_card_wins(engine):
    state = make_state(
        {"player_1": ["Countess", "Guard"], "player_2": ["Princess"]},
        deck=[],
    )
    new_state, events = play(engine, state, "play_guard", target="player_2", guess="Baron")
    # Deck empty at end of turn: showdown, Princess(8) beats Countess(7).
    assert new_state["players"]["player_2"]["tokens"] == 1
    assert any("round ended" in e.text.lower() for e in events)


def test_showdown_tie_broken_by_discard_sum(engine):
    state = make_state(
        {"player_1": ["Handmaid", "Baron"], "player_2": ["Baron"]},
        deck=[],
        discards={"player_1": ["Prince"], "player_2": ["Guard"]},
    )
    new_state, _ = play(engine, state, "play_handmaid")
    # Both hold Baron(3); player_1's discards sum higher (Prince 5 + Handmaid 4).
    assert new_state["players"]["player_1"]["tokens"] == 1
    assert new_state["players"]["player_2"]["tokens"] == 0


def test_last_player_standing_wins_round(engine):
    state = make_state(
        {"player_1": ["Guard", "Priest"], "player_2": ["Baron"]},
        deck=["Handmaid", "Prince", "King"],
    )
    new_state, _ = play(engine, state, "play_guard", target="player_2", guess="Baron")
    assert new_state["players"]["player_1"]["tokens"] == 1
    # A new round started automatically: both players standing again.
    assert not new_state["players"]["player_2"]["eliminated"]
    assert new_state["round_number"] == 2
    assert new_state["current_player"] == "player_1"  # round winner starts


def test_game_ends_at_token_target(engine):
    state = make_state(
        {"player_1": ["Guard", "Priest"], "player_2": ["Baron"]},
        deck=["Handmaid", "Prince"],
        tokens={"player_1": 6},
        tokens_to_win=7,
    )
    new_state, _ = play(engine, state, "play_guard", target="player_2", guess="Baron")
    status = engine.status(new_state)
    assert status.over
    assert status.winners == ("player_1",)
    assert engine.to_act(new_state) == []


# ----------------------------------------------------------------- hidden info


def test_observation_hides_deck_and_other_hands(engine):
    state, _ = engine.setup(2, seed=3)
    view = engine.observe(state, "player_2")
    flat = repr(view)
    assert "your_hand" in view
    # player_1's actual hand contents must not be recoverable (only a count).
    assert "hand" not in view["players"]["player_1"]
    assert "deck" not in view or isinstance(view.get("deck"), int)
    assert state["removed_card"] not in flat or state["removed_card"] in str(
        state["revealed_cards"]
    ) + str(view.get("your_hand"))


def test_spectator_sees_everything(engine):
    state, _ = engine.setup(2, seed=3)
    view = engine.observe(state, "spectator")
    assert view["players"]["player_1"]["hand"] == state["players"]["player_1"]["hand"]
    assert view["deck"] == state["deck"]


def test_illegal_action_rejected(engine):
    state, _ = engine.setup(2, seed=0)
    bogus = Action(seat="player_1", name="play_princess", args={"card": "Princess"})
    if bogus.key() in {a.key() for a in engine.legal_actions(state, "player_1")}:
        pytest.skip("seed dealt player_1 the Princess")
    with pytest.raises(ValueError):
        engine.apply(state, [bogus])
