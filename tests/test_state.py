"""State manager tests: spec-driven visibility, structure validation, masking."""

import copy

import pytest

from playtest.ingestion.schemas import VisibilitySpec
from playtest.state.manager import HIDDEN, GameStateManager

from .fixtures import TEMPLATE, sample_spec

DECK = ["Guard", "Priest", "Baron", "Prince", "King"]
REMOVED = "Princess"
REVEALED = ["Countess", "Handmaid"]


def _real_state(hands: dict[str, list[str]] | None = None) -> dict:
    """A complete REAL state (as the setup executor would produce it)."""
    hands = hands or {"player_1": ["King"], "player_2": ["Guard"]}
    state = copy.deepcopy(TEMPLATE)
    state["deck"] = list(DECK)
    state["deck_count"] = len(DECK)
    state["removed_card"] = REMOVED
    state["revealed_cards"] = list(REVEALED)
    for pid, hand in hands.items():
        state["players"][pid]["hand"] = list(hand)
        state["players"][pid]["hand_count"] = len(hand)
    return state


def _manager(state: dict | None = None) -> GameStateManager:
    manager = GameStateManager()
    manager.initialize(state or _real_state(), sample_spec().visibility)
    return manager


def test_initialize_returns_gm_view_with_real_values() -> None:
    manager = _manager()
    state = manager.get_state("gm")

    assert state["deck"] == DECK
    assert state["deck_count"] == len(DECK)
    assert state["removed_card"] == REMOVED
    assert state["revealed_cards"] == REVEALED
    assert state["players"]["player_1"]["hand"] == ["King"]
    assert state["players"]["player_2"]["hand"] == ["Guard"]


def test_player_view_hides_others_masked_and_hidden_fields() -> None:
    manager = _manager()
    state = manager.get_state("player_1")

    assert state["players"]["player_1"]["hand"] == ["King"]
    assert state["players"]["player_2"]["hand"] == [HIDDEN]
    assert state["players"]["player_2"]["hand_count"] == 1
    assert state["removed_card"] == HIDDEN
    assert "deck" not in state
    assert state["deck_count"] == len(DECK)
    # Public info remains visible.
    assert state["revealed_cards"] == REVEALED
    assert state["players"]["player_2"]["tokens"] == 0
    assert state["players"]["player_2"]["is_eliminated"] is False


def test_player_view_redaction_matches_count_field() -> None:
    manager = _manager(_real_state({"player_1": ["King"], "player_2": ["Guard", "Priest"]}))
    state = manager.get_state("player_1")
    assert state["players"]["player_2"]["hand"] == [HIDDEN, HIDDEN]
    assert state["players"]["player_2"]["hand_count"] == 2


def test_each_player_sees_own_hand_only() -> None:
    manager = _manager()
    p1 = manager.get_state("player_1")
    p2 = manager.get_state("player_2")

    assert p1["players"]["player_1"]["hand"] == ["King"]
    assert p1["players"]["player_2"]["hand"] == [HIDDEN]
    assert p2["players"]["player_2"]["hand"] == ["Guard"]
    assert p2["players"]["player_1"]["hand"] == [HIDDEN]


def test_visibility_without_count_field_redacts_by_length() -> None:
    visibility = VisibilitySpec(per_player_private=["hand"])
    manager = GameStateManager()
    manager.initialize(_real_state({"player_1": ["King"], "player_2": ["Guard", "Baron"]}),
                       visibility)
    state = manager.get_state("player_1")
    assert state["players"]["player_2"]["hand"] == [HIDDEN, HIDDEN]
    assert "deck" in state  # nothing hidden without a spec entry


def test_set_state_full_replacement_round_trips() -> None:
    manager = _manager()
    new_state = manager.get_state("gm")
    new_state["round_number"] = 2
    new_state["players"]["player_1"]["tokens"] = 1

    returned = manager.set_state(new_state)
    assert returned["round_number"] == 2
    assert returned["players"]["player_1"]["tokens"] == 1
    # The masked field stays accessible to the GM and masked internally.
    assert returned["removed_card"] == REMOVED
    assert manager.get_state("player_1")["removed_card"] == HIDDEN


def test_set_state_preserves_masked_value_when_committed_hidden() -> None:
    manager = _manager()
    new_state = manager.get_state("player_2")  # removed_card == HIDDEN in this view
    new_state["deck"] = list(DECK)  # restore the dropped field for the full write
    returned = manager.set_state(new_state)
    assert returned["removed_card"] == REMOVED


def test_set_state_remask_updates_masked_value_silently() -> None:
    manager = _manager()
    new_state = manager.get_state("gm")
    new_state["removed_card"] = "Baron"
    returned = manager.set_state(new_state, remask=True)
    assert returned["removed_card"] == "Baron"
    assert manager.get_state("player_1")["removed_card"] == HIDDEN


def test_set_state_rejects_missing_or_extra_keys() -> None:
    manager = _manager()
    state = manager.get_state("gm")

    missing = copy.deepcopy(state)
    del missing["deck"]
    with pytest.raises(ValueError, match="missing"):
        manager.set_state(missing)

    extra = copy.deepcopy(state)
    extra["bonus"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        manager.set_state(extra)


def test_set_state_rejects_type_changes() -> None:
    manager = _manager()
    state = manager.get_state("gm")
    state["deck_count"] = "five"
    with pytest.raises(ValueError, match="type"):
        manager.set_state(state)


def test_set_state_rejects_player_field_changes() -> None:
    manager = _manager()
    state = manager.get_state("gm")
    del state["players"]["player_1"]["tokens"]
    with pytest.raises(ValueError, match="player_1"):
        manager.set_state(state)


def test_uninitialized_access_raises() -> None:
    manager = GameStateManager()
    with pytest.raises(ValueError, match="not initialized"):
        manager.get_state("gm")
    with pytest.raises(ValueError, match="not initialized"):
        manager.set_state({"players": {}})


def test_initialize_requires_players() -> None:
    manager = GameStateManager()
    with pytest.raises(ValueError, match="players"):
        manager.initialize({"current_turn": "player_1"}, VisibilitySpec())
