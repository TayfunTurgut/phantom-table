"""Generic rules engine tests: seeded setup/redeals, turn flow, conservation invariants."""

import collections
import random

import pytest

from playtest.rules import GameRules

from .fixtures import COMPONENTS, sample_config


def _rules() -> GameRules:
    return GameRules(sample_config())


# --- Setup executor -----------------------------------------------------------


def test_setup_composition_matches_manifest() -> None:
    rules = _rules()
    state = rules.setup(num_players=2, seed=42)

    all_cards = (
        list(state["deck"])
        + [state["removed_card"]]
        + list(state["revealed_cards"])
        + [c for p in state["players"].values() for c in p["hand"]]
    )
    assert collections.Counter(all_cards) == collections.Counter(COMPONENTS)
    assert len(state["deck"]) == 16 - 1 - 3 - 2
    assert state["deck_count"] == len(state["deck"])
    assert len(state["revealed_cards"]) == 3
    assert state["current_turn"] == "player_1"
    assert state["turn_phase"] == "draw"
    assert state["round_number"] == 1
    assert state["num_players"] == 2
    for player in state["players"].values():
        assert player["hand_count"] == 1
        assert len(player["hand"]) == 1
        assert player["discards"] == []


def test_setup_uses_per_count_plan_and_clones_players() -> None:
    rules = _rules()
    state = rules.setup(num_players=3, seed=1)
    assert set(state["players"]) == {"player_1", "player_2", "player_3"}
    assert state["revealed_cards"] == []  # the 3p plan reveals nothing
    assert len(state["deck"]) == 16 - 1 - 0 - 3
    # Cloned players never inherit the template's placeholder hand.
    for player in state["players"].values():
        assert player["hand_count"] == 1


def test_setup_reproducible_by_seed() -> None:
    rules = _rules()
    a = rules.setup(2, seed=99)
    b = rules.setup(2, seed=99)
    c = rules.setup(2, seed=100)
    assert a == b
    assert a != c


def test_setup_unsupported_count_raises() -> None:
    rules = _rules()
    with pytest.raises(ValueError, match="no setup plan"):
        rules.setup(5, seed=1)


def test_redeal_round_preserves_carry_overs_and_increments_round() -> None:
    rules = _rules()
    state = rules.setup(2, seed=7)
    state["players"]["player_1"]["tokens"] = 3
    state["players"]["player_2"]["is_eliminated"] = True
    state["round_number"] = 4

    new_state = rules.redeal_round(state, random.Random(11))

    assert new_state["round_number"] == 5
    assert new_state["players"]["player_1"]["tokens"] == 3  # carried over
    assert new_state["players"]["player_2"]["is_eliminated"] is False  # reset by template
    all_cards = (
        list(new_state["deck"])
        + [new_state["removed_card"]]
        + list(new_state["revealed_cards"])
        + [c for p in new_state["players"].values() for c in p["hand"]]
    )
    assert collections.Counter(all_cards) == collections.Counter(COMPONENTS)


# --- Turn flow ------------------------------------------------------------------


def test_available_actions_filters_by_phase_only() -> None:
    rules = _rules()
    draw_state = {"turn_phase": "draw"}
    play_state = {"turn_phase": "play"}
    assert rules.available_actions(draw_state, "player_1") == ["draw_card"]
    assert rules.available_actions(play_state, "player_1") == [
        "play_countess",
        "play_guard",
        "play_king",
        "play_prince",
    ]


def test_is_turn_over_gm_report_overrides_spec() -> None:
    rules = _rules()
    play = {"action_type": "play_guard"}
    draw = {"action_type": "draw_card"}
    assert rules.is_turn_over(play) is True  # spec flag
    assert rules.is_turn_over(draw) is False
    assert rules.is_turn_over(draw, gm_turn_ended=True) is True  # GM override
    assert rules.is_turn_over(play, gm_turn_ended=False) is False
    assert rules.is_turn_over(None) is False
    assert rules.is_turn_over({"action_type": "unknown"}) is True  # unknown ends the turn


def _players_state(eliminated: set[str], count: int = 4) -> dict:
    return {
        "turn_phase": "play",
        "players": {
            f"player_{i}": {"is_eliminated": f"player_{i}" in eliminated}
            for i in range(1, count + 1)
        },
    }


def test_advance_turn_skips_inactive_and_resets_phase() -> None:
    rules = _rules()
    state = _players_state({"player_2"}, count=3)
    advanced = rules.advance_turn(state, "player_1")
    assert advanced["current_turn"] == "player_3"
    assert advanced["turn_phase"] == "draw"
    assert rules.advance_turn(state, "player_3")["current_turn"] == "player_1"  # wraps


def test_advance_turn_all_but_one_eliminated() -> None:
    rules = _rules()
    state = _players_state({"player_2", "player_3", "player_4"})
    assert rules.advance_turn(state, "player_1")["current_turn"] == "player_1"
    assert rules.advance_turn(state, "player_3")["current_turn"] == "player_1"


def test_advance_turn_without_inactive_field_rotates_everyone() -> None:
    config = sample_config()
    config.game_spec.turn.inactive_field = None
    rules = GameRules(config)
    state = _players_state({"player_2"}, count=3)
    assert rules.advance_turn(state, "player_1")["current_turn"] == "player_2"


# --- Conservation invariants -----------------------------------------------------


def test_invariants_clean_on_fresh_setup() -> None:
    rules = _rules()
    state = rules.setup(2, seed=5)
    assert rules.check_invariants(state) == []


def test_invariants_detect_lost_and_duplicated_components() -> None:
    rules = _rules()
    state = rules.setup(2, seed=5)

    lost = {**state, "deck": state["deck"][1:], "deck_count": len(state["deck"]) - 1}
    violations = rules.check_invariants(lost)
    assert any("conservation" in v.lower() for v in violations)

    duplicated = {**state, "deck": state["deck"] + [state["deck"][0]],
                  "deck_count": len(state["deck"]) + 1}
    violations = rules.check_invariants(duplicated)
    assert any("conservation" in v.lower() for v in violations)


def test_invariants_detect_count_mismatch() -> None:
    rules = _rules()
    state = rules.setup(2, seed=5)
    state["deck_count"] = 99
    assert any("deck_count" in v for v in rules.check_invariants(state))

    state = rules.setup(2, seed=5)
    state["players"]["player_1"]["hand_count"] = 2
    assert any("hand_count" in v for v in rules.check_invariants(state))


def test_invariants_skip_hidden_masks() -> None:
    rules = _rules()
    state = rules.setup(2, seed=5)
    # A player view masks the removed card; conservation must not count the mask string
    # (the GM view is the one checked in real runs, but stay safe on redacted views).
    masked = {**state, "removed_card": "HIDDEN"}
    violations = rules.check_invariants(masked)
    # One Princess-or-other card now legitimately "missing": only conservation entries.
    assert all("conservation" in v.lower() for v in violations)


def test_system_prompt_addendum_mentions_phases_and_count() -> None:
    rules = _rules()
    text = rules.system_prompt_addendum(3)
    assert "3 players" in text
    assert "draw -> play" in text
