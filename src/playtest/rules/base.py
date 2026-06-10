"""The GameRules seam: all game-specific engine logic behind one interface.

The GM agent and the turn-loop driver are generic. Everything specific to a particular
game — its components and setup, integrity invariants, turn order, phases, the actions a
player may take, round/game-over conditions, and scoring — lives in a ``GameRules``
subclass. A game with no hand-written rules module is driven entirely by the GM/LLM from
its ingested rulebook (see the generic fallback), trading the deterministic safety nets
for full generality.
"""

from abc import ABC, abstractmethod
from random import Random


class GameRules(ABC):
    name: str = "generic"

    @property
    @abstractmethod
    def supported_player_counts(self) -> tuple[int, ...]:
        """Player counts this module can set up (empty tuple = any count)."""

    @abstractmethod
    def setup(
        self, game_config: object, num_players: int, seed: int | None
    ) -> tuple[dict, str | None]:
        """Build the initial, fully dealt state.

        Returns ``(state, private_value)`` where ``private_value`` is any single hidden
        value to stash privately (e.g. the removed card), or None.
        """

    def system_prompt_addendum(self, num_players: int) -> str:
        """Optional text appended to the GM system prompt for this game/count."""
        return ""

    @abstractmethod
    def check_invariants(self, state: dict, last_action: dict | None) -> list[str]:
        """Integrity violations on a committed state; empty list means clean."""

    @abstractmethod
    def available_actions(self, state: dict, player_id: str) -> list[str]:
        """Action tool names the player may legally call right now."""

    @abstractmethod
    def is_turn_over(self, last_action: dict | None) -> bool:
        """True once the just-resolved action ends the acting player's turn."""

    @abstractmethod
    def advance_turn(self, state: dict, current_player: str) -> dict:
        """Return the state with current_turn/turn_phase set for the next actor."""

    @abstractmethod
    def is_round_over(self, state: dict) -> bool:
        """True when the current round has ended and should be scored/redealt."""

    # -- Round scoring (games without scored rounds keep these defaults) ---------

    def score_round(self, players: dict) -> dict | None:
        """Score a finished round: ``{winners, winning_card, scores}`` or None."""
        return None

    def is_game_won(self, state: dict) -> str | None:
        """Winner label if the game is now won, else None."""
        return None

    def new_round_deck(self, rng: Random) -> list[str] | None:
        """A freshly shuffled deck for the next round, or None if not deck-based."""
        return None
