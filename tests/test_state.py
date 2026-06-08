import copy

import pytest

from playtest.state.manager import GameStateManager

TEMPLATE = {
    "game_name": "Love Letter",
    "variant": "classic",
    "num_players": 2,
    "tokens_to_win": 7,
    "round_number": 1,
    "current_turn": "player_1",
    "turn_phase": "draw",
    "deck_count": 0,
    "removed_card": "HIDDEN",
    "revealed_cards": [],
    "players": {
        "player_1": {
            "hand": [],
            "hand_count": 0,
            "discards": [],
            "tokens": 0,
            "is_eliminated": False,
            "is_protected": False,
        },
        "player_2": {
            "hand": [],
            "hand_count": 0,
            "discards": [],
            "tokens": 0,
            "is_eliminated": False,
            "is_protected": False,
        },
    },
}

DECK = ["Guard", "Priest", "Baron", "Prince", "King"]
REMOVED = "Princess"
REVEALED = ["Countess", "Handmaid"]
HANDS = {"player_1": ["King"], "player_2": ["Guard"]}


def _manager() -> GameStateManager:
    manager = GameStateManager()
    manager.initialize(
        initial_state=copy.deepcopy(TEMPLATE),
        deck_cards=DECK,
        removed_card=REMOVED,
        revealed_cards=REVEALED,
        player_hands=HANDS,
    )
    return manager


def test_initialize_sets_counts_hands_and_redacts_removed_card() -> None:
    manager = _manager()
    state = manager.get_state("gm")

    assert state["deck_count"] == len(DECK)
    assert state["revealed_cards"] == REVEALED
    assert state["players"]["player_1"]["hand"] == ["King"]
    assert state["players"]["player_1"]["hand_count"] == 1
    assert state["players"]["player_2"]["hand"] == ["Guard"]
    assert state["players"]["player_2"]["hand_count"] == 1


def test_gm_view_sees_everything() -> None:
    manager = _manager()
    state = manager.get_state("gm")

    assert state["players"]["player_1"]["hand"] == ["King"]
    assert state["players"]["player_2"]["hand"] == ["Guard"]
    assert state["removed_card"] == REMOVED
    assert state["deck"] == DECK


def test_player_view_hides_others_removed_card_and_deck() -> None:
    manager = _manager()
    state = manager.get_state("player_1")

    assert state["players"]["player_1"]["hand"] == ["King"]
    assert state["players"]["player_2"]["hand"] == ["HIDDEN"]
    assert state["players"]["player_2"]["hand_count"] == 1
    assert state["removed_card"] == "HIDDEN"
    assert "deck" not in state
    assert state["deck_count"] == len(DECK)
    # Public info remains visible.
    assert state["revealed_cards"] == REVEALED
    assert state["players"]["player_2"]["tokens"] == 0
    assert state["players"]["player_2"]["is_eliminated"] is False


def test_player_view_redaction_matches_hand_count() -> None:
    template = copy.deepcopy(TEMPLATE)
    manager = GameStateManager()
    manager.initialize(
        initial_state=template,
        deck_cards=DECK,
        removed_card=REMOVED,
        revealed_cards=REVEALED,
        player_hands={"player_1": ["King"], "player_2": ["Guard", "Priest"]},
    )
    state = manager.get_state("player_1")
    assert state["players"]["player_2"]["hand"] == ["HIDDEN", "HIDDEN"]
    assert state["players"]["player_2"]["hand_count"] == 2


def test_each_player_sees_own_hand_only() -> None:
    manager = _manager()
    p1 = manager.get_state("player_1")
    p2 = manager.get_state("player_2")

    assert p1["players"]["player_1"]["hand"] == ["King"]
    assert p1["players"]["player_2"]["hand"] == ["HIDDEN"]
    assert p2["players"]["player_2"]["hand"] == ["Guard"]
    assert p2["players"]["player_1"]["hand"] == ["HIDDEN"]


def test_set_state_full_replacement_round_trips() -> None:
    manager = _manager()
    new_state = manager.get_state("gm")
    new_state["round_number"] = 2
    new_state["players"]["player_1"]["tokens"] = 1

    returned = manager.set_state(new_state)
    assert returned["round_number"] == 2
    assert returned["players"]["player_1"]["tokens"] == 1
    # removed_card stays accessible to the GM and privatized internally.
    assert returned["removed_card"] == REMOVED


def test_set_state_rejects_missing_key() -> None:
    manager = _manager()
    new_state = manager.get_state("gm")
    del new_state["round_number"]
    with pytest.raises(ValueError, match="keys do not match"):
        manager.set_state(new_state)


def test_set_state_rejects_wrong_type() -> None:
    manager = _manager()
    new_state = manager.get_state("gm")
    new_state["deck_count"] = "five"
    with pytest.raises(ValueError, match="deck_count"):
        manager.set_state(new_state)


def test_set_state_rejects_wrong_player_set() -> None:
    manager = _manager()
    new_state = manager.get_state("gm")
    del new_state["players"]["player_2"]
    with pytest.raises(ValueError, match="players must be"):
        manager.set_state(new_state)


def test_set_state_rejects_non_dict() -> None:
    manager = _manager()
    with pytest.raises(ValueError, match="must be a JSON object"):
        manager.set_state(["not", "a", "dict"])  # type: ignore[arg-type]


def test_set_state_warns_when_removed_card_changes(capsys: pytest.CaptureFixture) -> None:
    manager = _manager()
    new_state = manager.get_state("gm")
    new_state["removed_card"] = "Baron"
    manager.set_state(new_state)
    out = capsys.readouterr().out
    assert "removed_card changed" in out
    assert manager.get_removed_card() == "Baron"


def test_operations_before_initialize_raise() -> None:
    manager = GameStateManager()
    with pytest.raises(ValueError, match="not initialized"):
        manager.get_state("gm")
    with pytest.raises(ValueError, match="not initialized"):
        manager.set_state(copy.deepcopy(TEMPLATE))


def test_get_deck_cards_and_removed_card() -> None:
    manager = _manager()
    assert manager.get_deck_cards() == DECK
    assert manager.get_removed_card() == REMOVED


def test_get_state_returns_deep_copy() -> None:
    manager = _manager()
    view = manager.get_state("gm")
    view["players"]["player_1"]["hand"].append("LEAK")
    view["deck"].append("LEAK")

    fresh = manager.get_state("gm")
    assert fresh["players"]["player_1"]["hand"] == ["King"]
    assert fresh["deck"] == DECK
