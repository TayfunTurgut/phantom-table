"""Observer tests: color-coded seat rendering."""

from playtest.ui.observer import PLAYER_COLORS, _color


def test_seats_1_to_6_keep_previous_colors():
    """First 6 seats should maintain their original colors."""
    expected = {
        "player_1": "cyan",
        "player_2": "magenta",
        "player_3": "green",
        "player_4": "yellow",
        "player_5": "red",
        "player_6": "blue",
    }
    for seat, expected_color in expected.items():
        actual_color = _color(seat)
        assert actual_color == expected_color, (
            f"{seat} should be {expected_color}, got {actual_color}"
        )


def test_seat_7_gets_non_default_color():
    """Seat 7 should get a color from the extended palette."""
    color = _color("player_7")
    assert color != "white", "player_7 should have a non-default color"
    # Verify the color is from the palette
    assert color in PLAYER_COLORS, f"player_7 color {color} should be in PLAYER_COLORS"


def test_seat_13_cycles_to_seat_1_color():
    """Seat 13 should cycle back to the same color as seat 1."""
    seat_1_color = _color("player_1")
    seat_13_color = _color("player_13")
    assert seat_13_color == seat_1_color, "player_13 should have same color as player_1"


def test_non_player_seats_get_default_color():
    """Non-player_N seats should get white as fallback."""
    assert _color("spectator") == "white"
    assert _color("observer") == "white"
    assert _color("unknown_seat") == "white"


def test_palette_has_at_least_12_colors():
    """PLAYER_COLORS should have at least 12 entries for cycling."""
    assert len(PLAYER_COLORS) >= 12, "PLAYER_COLORS should have at least 12 colors"
