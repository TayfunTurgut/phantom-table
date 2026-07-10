"""The game engine contract.

Every game — hand-written or generated — is a single Python module exposing a
``Game`` class that implements the ``GameEngine`` protocol below. The engine is
the complete, deterministic rules authority: it deals, enumerates legal moves,
applies effects, enforces hidden information, and decides who won. Agents (LLM
or scripted) only ever CHOOSE among the actions the engine enumerates.

CONTRACT (binding for implementations; this docstring is included verbatim in
code-generation prompts):

1.  State is a plain JSON-serializable dict. Its internal shape is private to
    the engine, but it must round-trip through ``json.dumps``/``loads``
    unchanged. Engines never mutate a state they are given; ``apply`` returns a
    new dict (use ``copy.deepcopy`` on entry).

2.  ``setup(num_players, seed)`` returns ``(state, events)``: a fully dealt
    initial state plus the events describing that setup — round start and any
    face-up/publicly revealed cards. Setup events obey item 6 and are delivered
    to seats exactly like ``apply``'s, so setup-time narration and reveals
    become part of each seat's memory. All randomness is derived from
    ``seed``. The state must contain an integer
    ``rng_seed`` key: whenever mid-game randomness is needed (reshuffle, dice,
    redeal), do ``rng = random.Random(state["rng_seed"])``, use it, then store
    ``new_state["rng_seed"] = rng.randrange(2**32)``. Identical seeds must
    yield identical games when fed identical action sequences.

3.  ``to_act(state)`` returns the seats that must decide RIGHT NOW. Exactly one
    seat for sequential games; several for simultaneous decisions (simultaneous
    reveals, co-op votes, reaction windows). It returns ``[]`` if and only if
    ``status(state).over`` is True. Automa/bot behavior is not a seat: it is
    deterministic logic inside ``apply``.

4.  ``legal_actions(state, seat)`` returns every legal, fully parameter-bound
    ``Action`` for that seat (e.g. one action per card × target × named-card
    combination). It must be non-empty for every seat in ``to_act(state)`` — if
    the rules allow "do nothing", enumerate an explicit pass action. The list
    is deterministic and stable for a given state (no randomness, no
    reordering). Each action's ``label`` is a one-line human-readable
    description shown to the choosing agent. Keep the menu choosable: when
    fully binding one decision would enumerate more than roughly 50 actions
    (open bid amounts, free placement, multi-leg movement), split it into
    consecutive smaller decisions — record the partial choice in state and
    stop, so ``to_act`` returns the same seat again and ``legal_actions``
    offers the next part (the amount, then the destination, then the next
    leg). Each stage is an ordinary decision point; emit the events
    describing the completed whole when its final stage resolves.

5.  ``apply(state, actions)`` takes exactly one chosen action per seat in
    ``to_act(state)`` and resolves them. It validates each action is in that
    seat's legal set and raises ``ValueError`` otherwise. After resolving, it
    auto-advances through every step that requires no human decision — forced
    draws, automa turns, chance reveals, round scoring, redeals — stopping only
    at the next decision point or the end of the game. It returns
    ``(new_state, events)``. A "may respond" window (block, challenge, Nope,
    instant) is a decision point, not part of resolution: store the announced
    action as pending in state and stop, so ``to_act`` returns the eligible
    responder(s) and ``legal_actions`` offers each reaction plus an explicit
    decline. Resolve or cancel the pending action only when the window closes
    — every responder declined, or a reaction resolved. Offer the window to
    every seat the rules permit to respond, never only to seats whose hidden
    cards make responding useful: being asked is itself information.

6.  Events are the factual record of what happened, in past tense, in
    resolution order ("player_2 played Guard, guessing player_1 holds Baron —
    correct: player_1 is eliminated and discards Baron."). An event with
    ``visible_to=None`` is public; otherwise only the listed seats see it.
    Private information revealed to specific seats (a Priest peek) is delivered
    as a private event; the runtime keeps each seat's event history as that
    seat's memory, so engines do not track who-knows-what in state.

7.  ``observe(state, seat)`` returns that seat's view as a JSON-serializable
    dict: the seat's own private information in the clear, other seats' hidden
    zones reduced to counts or backs, plus all public state. Hidden information
    must be impossible to recover from the observation. The special seat
    ``"spectator"`` receives the omniscient view (used for logging and debug).

8.  ``status(state)`` is pure: it reports whether the game is over, the
    winners (empty tuple on a draw or unfinished game), and current scores per
    seat where the game has a score concept (else an empty dict). Co-op games
    list every surviving seat as a winner on a win, none on a loss.

9.  Engines are self-contained: imports are limited to the Python standard
    library plus ``playtest.engine`` (for ``Action``, ``Event``, ``GameStatus``,
    ``seats_for``). No I/O, no network, no globals mutated at import time;
    importing the module must have no side effects beyond class/constant
    definitions.

Seats are named ``"player_1"``, ``"player_2"``, ... in seating order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

SPECTATOR = "spectator"


@dataclass(frozen=True)
class Action:
    """One fully bound choice available to (or made by) a seat."""

    seat: str
    name: str
    args: dict = field(default_factory=dict)
    label: str = ""

    def key(self) -> str:
        """Stable identity used to match a chosen action against the legal set."""
        return f"{self.seat}|{self.name}|{json.dumps(self.args, sort_keys=True)}"


@dataclass(frozen=True)
class Event:
    """A factual line describing something that happened during resolution."""

    text: str
    visible_to: tuple[str, ...] | None = None  # None means public


@dataclass(frozen=True)
class GameStatus:
    """Terminal check result plus winners and scores."""

    over: bool
    winners: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class GameEngine(Protocol):
    """The protocol every game module's ``Game`` class implements.

    See the module docstring for the binding contract semantics.
    """

    game_name: str
    min_players: int
    max_players: int

    def setup(self, num_players: int, seed: int) -> tuple[dict, list[Event]]: ...

    def to_act(self, state: dict) -> list[str]: ...

    def legal_actions(self, state: dict, seat: str) -> list[Action]: ...

    def apply(self, state: dict, actions: list[Action]) -> tuple[dict, list[Event]]: ...

    def observe(self, state: dict, seat: str) -> dict: ...

    def status(self, state: dict) -> GameStatus: ...


def seats_for(num_players: int) -> list[str]:
    """Canonical seat names for a player count."""
    return [f"player_{i}" for i in range(1, num_players + 1)]
