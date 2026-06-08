"""In-memory game state manager with caller-based view filtering."""

from copy import deepcopy

from rich.console import Console

_console = Console()

_HIDDEN = "HIDDEN"


def _type_tag(value: object) -> str:
    """Return a coarse JSON type tag for a value (bool checked before int)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


class GameStateManager:
    """Holds the authoritative game state and serves caller-filtered views.

    The internal state holds real player hands and the real deck (under a ``deck``
    key). The removed card's real value is kept privately and the state field is
    redacted to ``"HIDDEN"``; all other redaction happens when a player view is read.
    """

    def __init__(self) -> None:
        self._state: dict | None = None
        self._removed_card: str | None = None  # actual value, hidden from players
        self._expected: dict | None = None  # structure captured at initialize()

    def initialize(
        self,
        initial_state: dict,
        deck_cards: list[str],
        removed_card: str,
        revealed_cards: list[str],
        player_hands: dict[str, list[str]],
    ) -> dict:
        """Fill the initial-state template with actual (already shuffled) values.

        - ``removed_card`` is stored privately; the state field shows ``"HIDDEN"``.
        - ``deck_cards`` is stored in the state under a ``deck`` key (added here;
          the template only carries ``deck_count``).
        - ``revealed_cards`` and ``player_hands`` are public / per-player info.

        Returns the GM view of the initialized state.
        """
        state = deepcopy(initial_state)

        players = state.get("players")
        if not isinstance(players, dict):
            raise ValueError("initial_state must contain a 'players' object")
        for player_id, hand in player_hands.items():
            if player_id not in players:
                raise ValueError(f"unknown player '{player_id}' not present in template")
            players[player_id]["hand"] = list(hand)
            players[player_id]["hand_count"] = len(hand)

        state["revealed_cards"] = list(revealed_cards)
        state["deck_count"] = len(deck_cards)
        state["deck"] = list(deck_cards)
        state["removed_card"] = _HIDDEN

        self._removed_card = removed_card
        self._state = state
        self._expected = self._capture_structure(state)

        return self.get_state("gm")

    def get_state(self, caller_id: str = "gm") -> dict:
        """Return the game state filtered for the caller.

        GM sees everything (real removed card + deck). A player sees their own hand,
        other hands redacted to ``["HIDDEN"] * hand_count``, ``removed_card`` as
        ``"HIDDEN"``, and no deck contents (only ``deck_count``).
        """
        if self._state is None:
            raise ValueError("state not initialized")

        view = deepcopy(self._state)

        if caller_id == "gm":
            view["removed_card"] = self._removed_card
            return view

        # Player view: redact other hands, drop deck contents, keep removed_card hidden.
        for player_id, player in view["players"].items():
            if player_id != caller_id:
                player["hand"] = [_HIDDEN] * player.get("hand_count", len(player.get("hand", [])))
        view.pop("deck", None)
        return view

    def set_state(self, new_state: dict) -> dict:
        """Full replacement of the game state, validated against the captured structure.

        Returns the GM view of the new state.
        """
        if not isinstance(new_state, dict):
            raise ValueError("game state must be a JSON object")
        if self._state is None or self._expected is None:
            raise ValueError("state not initialized")

        self._validate_structure(new_state)

        incoming_removed = new_state["removed_card"]
        if incoming_removed != _HIDDEN:
            if self._removed_card is not None and incoming_removed != self._removed_card:
                _console.print(
                    f"[yellow]Warning:[/yellow] removed_card changed from "
                    f"{self._removed_card!r} to {incoming_removed!r}; the removed card "
                    "should never change mid-game."
                )
            self._removed_card = incoming_removed

        stored = deepcopy(new_state)
        stored["removed_card"] = _HIDDEN
        self._state = stored

        return self.get_state("gm")

    def get_deck_cards(self) -> list[str]:
        """GM-only: return actual deck contents in order."""
        if self._state is None:
            raise ValueError("state not initialized")
        return list(self._state["deck"])

    def get_removed_card(self) -> str:
        """GM-only: return the actual removed card."""
        if self._removed_card is None:
            raise ValueError("state not initialized")
        return self._removed_card

    @staticmethod
    def _capture_structure(state: dict) -> dict:
        """Record top-level keys/types and per-player keys/types from a built state."""
        top = {key: _type_tag(value) for key, value in state.items()}
        players = {
            player_id: {field: _type_tag(value) for field, value in player.items()}
            for player_id, player in state["players"].items()
        }
        return {"top": top, "players": players}

    def _validate_structure(self, new_state: dict) -> None:
        """Validate ``new_state`` against the structure captured at initialize()."""
        assert self._expected is not None  # guarded by caller
        expected_top: dict = self._expected["top"]
        expected_players: dict = self._expected["players"]

        if set(new_state) != set(expected_top):
            missing = set(expected_top) - set(new_state)
            extra = set(new_state) - set(expected_top)
            raise ValueError(
                f"game state keys do not match expected structure "
                f"(missing={sorted(missing)}, unexpected={sorted(extra)})"
            )
        for key, expected_type in expected_top.items():
            actual = _type_tag(new_state[key])
            if actual != expected_type:
                raise ValueError(
                    f"game state field '{key}' has type {actual}, expected {expected_type}"
                )

        players = new_state["players"]
        if set(players) != set(expected_players):
            raise ValueError(
                f"players must be exactly {sorted(expected_players)}, got {sorted(players)}"
            )
        for player_id, expected_fields in expected_players.items():
            player = players[player_id]
            if set(player) != set(expected_fields):
                raise ValueError(
                    f"player '{player_id}' fields do not match expected structure "
                    f"(expected {sorted(expected_fields)}, got {sorted(player)})"
                )
            for field, expected_type in expected_fields.items():
                actual = _type_tag(player[field])
                if actual != expected_type:
                    raise ValueError(
                        f"player '{player_id}' field '{field}' has type {actual}, "
                        f"expected {expected_type}"
                    )
