from rich.console import Console

from playtest.ui.observer import GameObserver

STATE = {
    "game_name": "Sample Letters",
    "variant": "classic",
    "num_players": 2,
    "tokens_to_win": 7,
    "round_number": 1,
    "deck_count": 4,
    "removed_card": "HIDDEN",
    "revealed_cards": ["Countess", "Handmaid"],
    "players": {
        "player_1": {
            "hand": ["King"],
            "hand_count": 1,
            "discards": [],
            "tokens": 0,
            "is_eliminated": False,
            "is_protected": False,
        },
        "player_2": {
            "hand": ["Guard"],
            "hand_count": 1,
            "discards": ["Priest"],
            "tokens": 0,
            "is_eliminated": True,
            "is_protected": False,
        },
    },
}

ACTION = {
    "player_id": "player_1",
    "action_type": "play_guard",
    "parameters": {"target": "player_2", "named": "Priest"},
    "reasoning": "Player 2 has been hoarding a high card.",
    "public_statement": "I have a feeling you're hiding a Priest...",
}


def _observer(verbose: bool = False) -> tuple[GameObserver, Console]:
    console = Console(record=True, width=100)
    return GameObserver(console=console, verbose=verbose), console


def test_game_start_shows_setup_and_narration() -> None:
    obs, console = _observer()
    obs.on_game_start(STATE, "The letters are sealed.")
    out = console.export_text()
    assert "Sample Letters" in out
    assert "The letters are sealed." in out


def test_player_action_is_attributable_and_hides_hand() -> None:
    obs, console = _observer()
    obs.on_player_action("player_1", ACTION)
    out = console.export_text()
    assert "player_1" in out
    assert "play_guard" in out
    assert "I have a feeling you're hiding a Priest..." in out
    # Private reasoning is NOT shown without verbose.
    assert "hoarding a high card" not in out


def test_verbose_shows_reasoning() -> None:
    obs, console = _observer(verbose=True)
    obs.on_player_action("player_1", ACTION)
    out = console.export_text()
    assert "hoarding a high card" in out


def test_gm_resolution_shows_narration() -> None:
    obs, console = _observer()
    obs.on_gm_resolution({"narration": "The guess was wrong.", "gm_reasoning": "secret note"})
    out = console.export_text()
    assert "The guess was wrong." in out
    assert "secret note" not in out  # gm_reasoning hidden without verbose


def test_action_rejected_shows_error() -> None:
    obs, console = _observer()
    obs.on_action_rejected("player_2", "player_1 is protected")
    out = console.export_text()
    assert "player_2" in out
    assert "protected" in out


def test_state_update_table_hides_list_contents() -> None:
    obs, console = _observer()
    obs.on_state_update(STATE)
    out = console.export_text()
    # Public per-player ints and bool flags are shown generically.
    assert "player_1" in out
    assert "hand_count" in out
    assert "is_eliminated" in out  # player_2's truthy flag
    assert "deck_count" in out and "4" in out
    # List CONTENTS never shown table-side, whatever the game.
    assert "King" not in out
    assert "Guard" not in out
    assert "Priest" not in out


def test_game_end_banner_shows_winner() -> None:
    obs, console = _observer()
    obs.on_game_end("player_1", {"player_1": 7, "player_2": 4})
    out = console.export_text()
    assert "player_1" in out
    assert "7" in out
