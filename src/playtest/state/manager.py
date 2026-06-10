"""In-memory game state manager with spec-driven, caller-based view filtering."""

from copy import deepcopy

from rich.console import Console

from playtest.ingestion.schemas import VisibilitySpec

_console = Console()

HIDDEN = "HIDDEN"


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

    What is hidden from whom is configured by the ingestion-extracted
    :class:`VisibilitySpec`, not by game knowledge:

    - ``masked_fields``: top-level fields whose true value is stashed privately and
      stored/shown as ``"HIDDEN"`` (the GM view restores the real value).
    - ``hidden_fields``: top-level fields dropped entirely from player views.
    - ``per_player_private``: per-player list fields redacted to ``["HIDDEN"] * count``
      in every other player's view.
    """

    def __init__(self) -> None:
        self._state: dict | None = None
        self._visibility: VisibilitySpec | None = None
        self._masked: dict[str, object] = {}  # actual values of masked fields
        self._expected: dict | None = None  # structure captured at initialize()

    def initialize(self, initial_state: dict, visibility: VisibilitySpec) -> dict:
        """Establish the authoritative state (a complete REAL state, fully dealt).

        Masked fields' real values are stashed privately; the stored state carries
        ``"HIDDEN"`` in their place. Returns the GM view.
        """
        players = initial_state.get("players")
        if not isinstance(players, dict) or not players:
            raise ValueError("initial_state must contain a non-empty 'players' object")

        state = deepcopy(initial_state)
        self._visibility = visibility
        self._masked = {}
        for field in visibility.masked_fields:
            if field in state:
                self._masked[field] = state[field]
                state[field] = HIDDEN

        self._state = state
        self._expected = self._capture_structure(state)
        return self.get_state("gm")

    def get_state(self, caller_id: str = "gm") -> dict:
        """Return the game state filtered for the caller.

        The GM sees everything (masked fields restored). A player sees their own private
        fields, other players' private lists redacted, and hidden fields dropped.
        """
        if self._state is None or self._visibility is None:
            raise ValueError("state not initialized")

        view = deepcopy(self._state)

        if caller_id == "gm":
            for field, value in self._masked.items():
                view[field] = deepcopy(value)
            return view

        for player_id, player in view.get("players", {}).items():
            if player_id == caller_id:
                continue
            for field in self._visibility.per_player_private:
                if field not in player:
                    continue
                value = player[field]
                if isinstance(value, list):
                    count_field = self._visibility.count_fields.get(field)
                    count = player.get(count_field, len(value)) if count_field else len(value)
                    player[field] = [HIDDEN] * count
                else:
                    player[field] = HIDDEN
        for field in self._visibility.hidden_fields:
            view.pop(field, None)
        return view

    def set_state(self, new_state: dict, *, remask: bool = False) -> dict:
        """Full replacement of the game state, validated against the captured structure.

        ``remask=True`` re-establishes masked values without the changed-value warning —
        for engine redeals, where masked fields (e.g. a freshly removed card) legitimately
        change. Returns the GM view of the new state.
        """
        if not isinstance(new_state, dict):
            raise ValueError("game state must be a JSON object")
        if self._state is None or self._expected is None or self._visibility is None:
            raise ValueError("state not initialized")

        self._validate_structure(new_state)

        stored = deepcopy(new_state)
        for field in self._visibility.masked_fields:
            if field not in stored:
                continue
            incoming = stored[field]
            if incoming != HIDDEN:
                previous = self._masked.get(field)
                if not remask and previous is not None and incoming != previous:
                    _console.print(
                        f"[yellow]Warning:[/yellow] masked field '{field}' changed from "
                        f"{previous!r} to {incoming!r}; masked values should never change "
                        "mid-round."
                    )
                self._masked[field] = incoming
            stored[field] = HIDDEN
        self._state = stored

        return self.get_state("gm")

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
