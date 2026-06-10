"""Offline tests for the GM state-integrity checker (pure function, no API)."""

from playtest.rules.love_letter import _DECK_COMPOSITION_COUNTS, check_state_invariants


def _clean_state() -> dict:
    """A structurally valid mid-game 2-player state (play phase), 16 cards conserved."""
    return {
        "turn_phase": "play",
        "current_turn": "player_1",
        "deck_count": 10,
        "deck": [
            "Guard",
            "Guard",
            "Priest",
            "Priest",
            "Baron",
            "Baron",
            "Handmaid",
            "Handmaid",
            "Prince",
            "Prince",
        ],
        "removed_card": "Princess",
        "revealed_cards": ["Guard", "Guard", "Guard"],
        "players": {
            "player_1": {
                "hand": ["King"],
                "hand_count": 1,
                "discards": [],
                "is_eliminated": False,
                "is_protected": False,
            },
            "player_2": {
                "hand": ["Countess"],
                "hand_count": 1,
                "discards": [],
                "is_eliminated": False,
                "is_protected": False,
            },
        },
    }


def test_clean_state_has_no_violations() -> None:
    assert check_state_invariants(_clean_state(), _DECK_COMPOSITION_COUNTS) == []


def test_clean_state_after_guard_play_passes_action_check() -> None:
    # player_1 drew a Guard (deck -1) and played it to discards.
    state = _clean_state()
    state["deck"] = state["deck"][1:]  # drop one Guard from deck
    state["deck_count"] = 9
    state["players"]["player_1"]["discards"] = ["Guard"]
    last_action = {"player_id": "player_1", "action_type": "play_guard", "parameters": {}}
    assert check_state_invariants(state, _DECK_COMPOSITION_COUNTS, last_action) == []


def test_card_conservation_violation() -> None:
    state = _clean_state()
    state["deck"].pop()  # lose a card without placing it anywhere
    state["deck_count"] = len(state["deck"])
    violations = check_state_invariants(state, _DECK_COMPOSITION_COUNTS)
    assert any("conservation" in v and "found 15" in v for v in violations)


def test_hand_count_mismatch_violation() -> None:
    state = _clean_state()
    state["players"]["player_1"]["hand_count"] = 2
    violations = check_state_invariants(state, _DECK_COMPOSITION_COUNTS)
    assert any("hand_count is 2 but hand has 1" in v for v in violations)


def test_deck_count_mismatch_violation() -> None:
    state = _clean_state()
    state["deck_count"] = 9
    violations = check_state_invariants(state, _DECK_COMPOSITION_COUNTS)
    assert any("deck_count is 9 but deck has 10" in v for v in violations)


def test_hand_too_large_violation() -> None:
    state = _clean_state()
    state["players"]["player_1"]["hand"] = ["King", "Guard", "Priest"]
    state["players"]["player_1"]["hand_count"] = 3
    violations = check_state_invariants(state, _DECK_COMPOSITION_COUNTS)
    assert any("holds 3 cards (max is 2)" in v for v in violations)


def test_eliminated_player_with_cards_violation() -> None:
    state = _clean_state()
    state["players"]["player_2"]["is_eliminated"] = True
    violations = check_state_invariants(state, _DECK_COMPOSITION_COUNTS)
    assert any("eliminated but still has cards" in v for v in violations)


def test_card_duplication_violation() -> None:
    state = _clean_state()
    state["revealed_cards"] = ["Guard", "Guard", "Guard", "Guard"]  # 4 revealed -> 6 Guards total
    violations = check_state_invariants(state, _DECK_COMPOSITION_COUNTS)
    assert any("'Guard' appears 6 times" in v for v in violations)


def test_played_card_not_in_discards_violation() -> None:
    # The seed-1 failure: a Guard was played but never moved to discards.
    state = _clean_state()
    state["players"]["player_1"]["hand"] = ["Guard", "Guard"]
    state["players"]["player_1"]["hand_count"] = 2
    state["players"]["player_1"]["discards"] = []
    last_action = {"player_id": "player_1", "action_type": "play_guard", "parameters": {}}
    violations = check_state_invariants(state, _DECK_COMPOSITION_COUNTS, last_action)
    assert any("played Guard but the last card in their discards" in v for v in violations)


def test_two_cards_after_play_violation() -> None:
    # The motivating bug: Prince played on self but the held card was never discarded,
    # so the actor ends the turn holding 2 cards.
    state = _clean_state()
    # Remove one Guard and one Prince from the deck to keep 16 cards conserved.
    state["deck"] = ["Priest", "Priest", "Baron", "Baron", "Handmaid", "Handmaid", "Prince"]
    state["deck"].insert(
        0, "Guard"
    )  # deck: Guard,Priest,Priest,Baron,Baron,Handmaid,Handmaid,Prince
    state["deck_count"] = 8
    state["players"]["player_1"]["hand"] = ["King", "Guard"]
    state["players"]["player_1"]["hand_count"] = 2
    state["players"]["player_1"]["discards"] = ["Prince"]
    last_action = {"player_id": "player_1", "action_type": "play_prince", "parameters": {}}
    violations = check_state_invariants(state, _DECK_COMPOSITION_COUNTS, last_action)
    assert any("ended their turn holding 2 card(s) after playing" in v for v in violations)


def test_one_card_after_play_has_no_turn_end_violation() -> None:
    state = _clean_state()
    state["deck"] = state["deck"][1:]
    state["deck_count"] = 9
    state["players"]["player_1"]["discards"] = ["Guard"]
    last_action = {"player_id": "player_1", "action_type": "play_guard", "parameters": {}}
    violations = check_state_invariants(state, _DECK_COMPOSITION_COUNTS, last_action)
    assert not any("after playing" in v for v in violations)


def test_eliminated_actor_after_play_has_no_turn_end_violation() -> None:
    # Playing the Princess eliminates the actor, who legitimately ends with an empty hand.
    state = _clean_state()
    state["removed_card"] = "King"
    state["players"]["player_1"]["hand"] = []
    state["players"]["player_1"]["hand_count"] = 0
    state["players"]["player_1"]["discards"] = ["Princess"]
    state["players"]["player_1"]["is_eliminated"] = True
    last_action = {"player_id": "player_1", "action_type": "play_princess", "parameters": {}}
    violations = check_state_invariants(state, _DECK_COMPOSITION_COUNTS, last_action)
    assert violations == []
