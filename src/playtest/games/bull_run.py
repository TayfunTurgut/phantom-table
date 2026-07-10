"""Bull Run (a 6 nimmt!/Take 5 clone, 2-10 players) — reference GameEngine.

This module is the second exemplar for generated game engines, and the one that
teaches two mechanics Love Letter does not: a *simultaneous commit* (every seat
chooses a card face-down in the same step) and a *mid-resolution phase machine*
(revealed cards are placed one at a time, and resolution pauses for a human
decision when a card is too low to place automatically).

The game. A deck of 104 cards (1..104) each carries penalty points ("bull
heads"). Each player is dealt 10 cards; four cards start four face-up rows. For
ten turns, all players simultaneously commit one card; the commitments are then
revealed and resolved in ascending card order. A card joins the row whose last
card is the highest still below it. A card too low for every row, or one that
would be a row's sixth card, forces its owner to scoop a row into their penalty
pile and start that row anew. After ten turns the round is scored; penalties
accumulate across rounds. The game ends after any round in which someone's total
reaches 66, and the player with the LOWEST total wins (ties share the win).

Because low is good, ``scores`` in ``GameStatus`` are penalty totals where a
smaller number is better — the round-scoring events say so explicitly.

Single self-contained ``Game`` class, standard library only, implementing the
GameEngine contract documented in ``playtest.engine``.
"""

from __future__ import annotations

import copy
import random

from playtest.engine import SPECTATOR, Action, Event, GameStatus, seats_for

DECK_SIZE = 104
HAND_SIZE = 10
NUM_ROWS = 4
ROW_LIMIT = 5  # a sixth card forces the row's owner to take it
GAME_END_THRESHOLD = 66  # game ends after the round someone reaches this total


def bull_heads(card: int) -> int:
    """Penalty points a card is worth. Pure; the checks are order-sensitive."""
    if card == 55:
        return 7
    if card % 11 == 0:  # 11, 22, 33, 44, 66, 77, 88, 99
        return 5
    if card % 10 == 0:  # 10, 20, ..., 100
        return 3
    if card % 5 == 0:  # other cards ending in 5
        return 2
    return 1


class Game:
    game_name = "Bull Run"
    min_players = 2
    max_players = 10

    # ------------------------------------------------------------- lifecycle

    def setup(self, num_players: int, seed: int) -> tuple[dict, list[Event]]:
        if not self.min_players <= num_players <= self.max_players:
            raise ValueError(f"Bull Run supports 2-10 players, got {num_players}")
        rng = random.Random(seed)
        state = {
            "num_players": num_players,
            "round_number": 0,
            "phase": "commit",  # "commit" | "choose_row"
            "hands": {},
            "rows": [],
            "piles": {},  # penalty cards taken this round, per seat (public once taken)
            "committed": {},  # this turn's face-down commitments; empty at rest
            "pending": None,  # in-progress resolution: {"remaining": [[card, seat], ...]}
            "totals": {seat: 0 for seat in seats_for(num_players)},
            "rng_seed": 0,
            "game_over": False,
            "winners": [],
        }
        events: list[Event] = []
        self._start_round(state, rng, events)
        state["rng_seed"] = rng.randrange(2**32)
        return state, events

    def to_act(self, state: dict) -> list[str]:
        if state["game_over"]:
            return []
        if state["phase"] == "choose_row":
            # Just the one seat whose too-low card stalled resolution.
            return [state["pending"]["remaining"][0][1]]
        # Simultaneous commit: every seat still holding cards and not yet committed.
        return [
            seat
            for seat in seats_for(state["num_players"])
            if state["hands"][seat] and seat not in state["committed"]
        ]

    def status(self, state: dict) -> GameStatus:
        return GameStatus(
            over=state["game_over"],
            winners=tuple(state["winners"]),
            # Penalty totals: lower is better.
            scores={seat: float(total) for seat, total in state["totals"].items()},
        )

    # --------------------------------------------------------- legal actions

    def legal_actions(self, state: dict, seat: str) -> list[Action]:
        if state["game_over"] or seat not in self.to_act(state):
            return []
        if state["phase"] == "choose_row":
            actions: list[Action] = []
            for i, row in enumerate(state["rows"]):
                pts = sum(bull_heads(c) for c in row)
                actions.append(
                    Action(
                        seat=seat,
                        name="take_row",
                        args={"row": i},
                        label=f"Take row {i + 1} {row} — {pts} bull head(s) into your pile",
                    )
                )
            return actions
        return [
            Action(
                seat=seat,
                name="play_card",
                args={"card": card},
                label=f"Commit {card} ({bull_heads(card)} bull head(s))",
            )
            for card in sorted(state["hands"][seat])
        ]

    # ----------------------------------------------------------- application

    def apply(self, state: dict, actions: list[Action]) -> tuple[dict, list[Event]]:
        acting = self.to_act(state)
        if [a.seat for a in actions] != acting:
            raise ValueError(f"expected one action per seat in {acting}, got {actions}")
        for action in actions:
            legal = {a.key() for a in self.legal_actions(state, action.seat)}
            if action.key() not in legal:
                raise ValueError(f"illegal action: {action}")

        state = copy.deepcopy(state)
        events: list[Event] = []
        if state["phase"] == "commit":
            self._resolve_commit(state, actions, events)
        else:
            self._resolve_choice(state, actions[0], events)
        return state, events

    def _resolve_commit(self, state: dict, actions: list[Action], events: list[Event]) -> None:
        plays = {action.seat: action.args["card"] for action in actions}
        for seat, card in plays.items():
            state["hands"][seat].remove(card)
        order = seats_for(state["num_players"])
        events.append(
            Event(
                "All players committed and revealed: "
                + ", ".join(f"{seat} played {plays[seat]}" for seat in order)
                + ". Resolving in ascending order."
            )
        )
        # Resolve lowest card first; card values are unique so the order is total.
        state["pending"] = {"remaining": sorted([card, seat] for seat, card in plays.items())}
        self._advance(state, events)

    def _resolve_choice(self, state: dict, action: Action, events: list[Event]) -> None:
        card, seat = state["pending"]["remaining"][0]
        self._take_row(state, seat, action.args["row"], card, events, sixth=False)
        state["pending"]["remaining"].pop(0)
        self._advance(state, events)

    def _advance(self, state: dict, events: list[Event]) -> None:
        """Place queued cards in ascending order, pausing if one is too low."""
        remaining = state["pending"]["remaining"]
        while remaining:
            card, seat = remaining[0]
            row_idx = self._closest_lower_row(state["rows"], card)
            if row_idx is None:
                state["phase"] = "choose_row"
                events.append(
                    Event(
                        f"{seat}'s {card} was lower than the last card of every row; "
                        f"{seat} must choose a row to take."
                    )
                )
                return
            row = state["rows"][row_idx]
            if len(row) == ROW_LIMIT:
                self._take_row(state, seat, row_idx, card, events, sixth=True)
            else:
                row.append(card)
                events.append(Event(f"{seat}'s {card} was placed in row {row_idx + 1} ({row})."))
            remaining.pop(0)
        self._finish_turn(state, events)

    def _take_row(
        self,
        state: dict,
        seat: str,
        row_idx: int,
        played_card: int,
        events: list[Event],
        sixth: bool,
    ) -> None:
        taken = state["rows"][row_idx]
        pts = sum(bull_heads(c) for c in taken)
        state["piles"][seat].extend(taken)
        state["totals"][seat] += pts
        state["rows"][row_idx] = [played_card]
        if sixth:
            events.append(
                Event(
                    f"{seat}'s {played_card} would have been the sixth card in row "
                    f"{row_idx + 1}; {seat} took {taken} ({pts} bull head(s)) and restarted "
                    f"the row with {played_card}."
                )
            )
        else:
            events.append(
                Event(
                    f"{seat} took row {row_idx + 1} {taken} ({pts} bull head(s)) and "
                    f"restarted it with {played_card}."
                )
            )

    @staticmethod
    def _closest_lower_row(rows: list[list[int]], card: int) -> int | None:
        """Index of the row whose last card is the highest still below ``card``."""
        best_idx: int | None = None
        best_last = -1
        for i, row in enumerate(rows):
            last = row[-1]
            if best_last < last < card:
                best_idx, best_last = i, last
        return best_idx

    # ------------------------------------------------------------- internals

    def _finish_turn(self, state: dict, events: list[Event]) -> None:
        state["pending"] = None
        state["committed"] = {}
        state["phase"] = "commit"
        if all(not hand for hand in state["hands"].values()):
            self._score_round(state, events)

    def _score_round(self, state: dict, events: list[Event]) -> None:
        breakdown = ", ".join(
            f"{seat} +{sum(bull_heads(c) for c in state['piles'][seat])} "
            f"(total {state['totals'][seat]})"
            for seat in seats_for(state["num_players"])
        )
        events.append(
            Event(
                f"Round {state['round_number']} ended. Penalties this round — {breakdown}. "
                "Fewest bull heads is best."
            )
        )
        if any(total >= GAME_END_THRESHOLD for total in state["totals"].values()):
            best = min(state["totals"].values())
            winners = [seat for seat, total in state["totals"].items() if total == best]
            state["game_over"] = True
            state["winners"] = winners
            events.append(
                Event(
                    f"A player reached {GAME_END_THRESHOLD} bull heads, ending the game. "
                    f"{' and '.join(winners)} won with the lowest total ({best})."
                )
            )
            return
        rng = random.Random(state["rng_seed"])
        self._start_round(state, rng, events)
        state["rng_seed"] = rng.randrange(2**32)

    def _start_round(self, state: dict, rng: random.Random, events: list[Event]) -> None:
        deck = list(range(1, DECK_SIZE + 1))
        rng.shuffle(deck)
        cards = iter(deck)
        state["round_number"] += 1
        state["hands"] = {
            seat: sorted(next(cards) for _ in range(HAND_SIZE))
            for seat in seats_for(state["num_players"])
        }
        state["rows"] = [[next(cards)] for _ in range(NUM_ROWS)]
        state["piles"] = {seat: [] for seat in seats_for(state["num_players"])}
        state["committed"] = {}
        state["pending"] = None
        state["phase"] = "commit"
        events.append(
            Event(
                f"Round {state['round_number']} begins. Each player was dealt "
                f"{HAND_SIZE} cards; four rows were started face-up with "
                + ", ".join(f"row {i + 1}: {row[0]}" for i, row in enumerate(state["rows"]))
                + "."
            )
        )

    # ------------------------------------------------------------ observation

    def observe(self, state: dict, seat: str) -> dict:
        if seat == SPECTATOR:
            return copy.deepcopy(state)
        view = {
            "game": self.game_name,
            "you": seat,
            "round_number": state["round_number"],
            "phase": state["phase"],
            "game_end_threshold": GAME_END_THRESHOLD,
            "scoring": "penalty totals — lower is better",
            "rows": copy.deepcopy(state["rows"]),
            "your_hand": sorted(state["hands"][seat]),
            # Own face-down card in the clear; None while nothing is committed.
            "your_committed_card": state["committed"].get(seat),
            "players": {
                other: {
                    "hand_count": len(state["hands"][other]),
                    # Others' commitments are hidden — only whether they have committed.
                    "committed": other in state["committed"],
                    "penalty_pile": list(state["piles"][other]),
                    "penalty_total": state["totals"][other],
                }
                for other in seats_for(state["num_players"])
            },
        }
        if state["phase"] == "choose_row":
            # After the reveal the played cards are public; show what is left to place.
            view["resolving"] = [list(entry) for entry in state["pending"]["remaining"]]
            view["must_take_row"] = state["pending"]["remaining"][0][1]
        return view
