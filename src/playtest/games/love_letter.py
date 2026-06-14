"""Love Letter (classic 16-card edition, 2-4 players) — reference GameEngine.

This module is the exemplar for generated game engines: a single self-contained
``Game`` class, standard library only, implementing the GameEngine contract
documented in ``playtest.engine``.
"""

from __future__ import annotations

import copy
import random

from playtest.engine import Action, Event, GameStatus, seats_for

RANKS = {
    "Guard": 1,
    "Priest": 2,
    "Baron": 3,
    "Handmaid": 4,
    "Prince": 5,
    "King": 6,
    "Countess": 7,
    "Princess": 8,
}
COUNTS = {
    "Guard": 5,
    "Priest": 2,
    "Baron": 2,
    "Handmaid": 2,
    "Prince": 2,
    "King": 1,
    "Countess": 1,
    "Princess": 1,
}
TOKENS_TO_WIN = {2: 7, 3: 5, 4: 4}


class Game:
    game_name = "Love Letter"
    min_players = 2
    max_players = 4

    # ------------------------------------------------------------- lifecycle

    def setup(self, num_players: int, seed: int) -> dict:
        if not self.min_players <= num_players <= self.max_players:
            raise ValueError(f"Love Letter supports 2-4 players, got {num_players}")
        rng = random.Random(seed)
        state = {
            "num_players": num_players,
            "round_number": 0,
            "tokens_to_win": TOKENS_TO_WIN[num_players],
            "current_player": "",
            "players": {
                seat: {
                    "hand": [],
                    "discard": [],
                    "tokens": 0,
                    "eliminated": False,
                    "protected": False,
                }
                for seat in seats_for(num_players)
            },
            "deck": [],
            "removed_card": None,
            "revealed_cards": [],
            "rng_seed": 0,
            "game_over": False,
            "winners": [],
        }
        self._start_round(state, rng, starter="player_1", events=[])
        state["rng_seed"] = rng.randrange(2**32)
        return state

    def to_act(self, state: dict) -> list[str]:
        if state["game_over"]:
            return []
        return [state["current_player"]]

    def status(self, state: dict) -> GameStatus:
        return GameStatus(
            over=state["game_over"],
            winners=tuple(state["winners"]),
            scores={seat: float(player["tokens"]) for seat, player in state["players"].items()},
        )

    # --------------------------------------------------------- legal actions

    def legal_actions(self, state: dict, seat: str) -> list[Action]:
        if state["game_over"] or seat != state["current_player"]:
            return []
        hand = state["players"][seat]["hand"]

        # Countess rule: holding Countess with King or Prince forces the Countess.
        playable = sorted(set(hand), key=lambda c: RANKS[c])
        if "Countess" in hand and any(c in ("King", "Prince") for c in hand):
            playable = ["Countess"]

        targets = [
            other
            for other in state["players"]
            if other != seat
            and not state["players"][other]["eliminated"]
            and not state["players"][other]["protected"]
        ]

        actions: list[Action] = []
        for card in playable:
            if card == "Guard":
                if targets:
                    for target in targets:
                        for guess in sorted(RANKS, key=lambda c: RANKS[c]):
                            if guess == "Guard":
                                continue
                            actions.append(
                                Action(
                                    seat=seat,
                                    name="play_guard",
                                    args={"card": card, "target": target, "guess": guess},
                                    label=(f"Play Guard: guess that {target} holds the {guess}"),
                                )
                            )
                else:
                    actions.append(self._no_target_action(seat, card))
            elif card in ("Priest", "Baron", "King"):
                if targets:
                    verb = {
                        "Priest": "look at {t}'s hand",
                        "Baron": "secretly compare hands with {t}",
                        "King": "swap hands with {t}",
                    }[card]
                    for target in targets:
                        actions.append(
                            Action(
                                seat=seat,
                                name=f"play_{card.lower()}",
                                args={"card": card, "target": target},
                                label=f"Play {card}: {verb.format(t=target)}",
                            )
                        )
                else:
                    actions.append(self._no_target_action(seat, card))
            elif card == "Prince":
                for target in [*targets, seat]:
                    who = "yourself" if target == seat else target
                    actions.append(
                        Action(
                            seat=seat,
                            name="play_prince",
                            args={"card": card, "target": target},
                            label=f"Play Prince: {who} discards their hand and draws",
                        )
                    )
            elif card == "Handmaid":
                actions.append(
                    Action(
                        seat=seat,
                        name="play_handmaid",
                        args={"card": card},
                        label="Play Handmaid: you are protected until your next turn",
                    )
                )
            elif card == "Countess":
                actions.append(
                    Action(
                        seat=seat,
                        name="play_countess",
                        args={"card": card},
                        label="Play Countess: no effect",
                    )
                )
            elif card == "Princess":
                actions.append(
                    Action(
                        seat=seat,
                        name="play_princess",
                        args={"card": card},
                        label="Play Princess: you are eliminated from the round",
                    )
                )
        return actions

    @staticmethod
    def _no_target_action(seat: str, card: str) -> Action:
        return Action(
            seat=seat,
            name=f"play_{card.lower()}",
            args={"card": card},
            label=f"Play {card}: no valid target, so it has no effect",
        )

    # ----------------------------------------------------------- application

    def apply(self, state: dict, actions: list[Action]) -> tuple[dict, list[Event]]:
        acting = self.to_act(state)
        if [a.seat for a in actions] != acting:
            raise ValueError(f"expected one action from {acting}, got {actions}")
        action = actions[0]
        legal_keys = {a.key() for a in self.legal_actions(state, action.seat)}
        if action.key() not in legal_keys:
            raise ValueError(f"illegal action: {action}")

        state = copy.deepcopy(state)
        events: list[Event] = []
        seat = action.seat
        card = action.args["card"]
        player = state["players"][seat]
        player["hand"].remove(card)
        player["discard"].append(card)

        handler = {
            "Guard": self._play_guard,
            "Priest": self._play_priest,
            "Baron": self._play_baron,
            "Handmaid": self._play_handmaid,
            "Prince": self._play_prince,
            "King": self._play_king,
            "Countess": self._play_countess,
            "Princess": self._play_princess,
        }[card]
        handler(state, seat, action.args, events)

        self._finish_turn(state, events)
        return state, events

    # Card effects. Each appends factual events and applies state changes.

    def _play_guard(self, state: dict, seat: str, args: dict, events: list[Event]) -> None:
        target = args.get("target")
        if target is None:
            events.append(Event(f"{seat} played Guard with no valid target; no effect."))
            return
        guess = args["guess"]
        held = state["players"][target]["hand"][0]
        if held == guess:
            events.append(
                Event(
                    f"{seat} played Guard, guessing {target} holds the {guess} — "
                    f"correct! {target} is eliminated."
                )
            )
            self._eliminate(state, target, events)
        else:
            events.append(
                Event(
                    f"{seat} played Guard, guessing {target} holds the {guess} — wrong; "
                    "nothing happens."
                )
            )

    def _play_priest(self, state: dict, seat: str, args: dict, events: list[Event]) -> None:
        target = args.get("target")
        if target is None:
            events.append(Event(f"{seat} played Priest with no valid target; no effect."))
            return
        held = state["players"][target]["hand"][0]
        events.append(Event(f"{seat} played Priest and looked at {target}'s hand."))
        events.append(Event(f"You saw that {target} holds the {held}.", visible_to=(seat,)))

    def _play_baron(self, state: dict, seat: str, args: dict, events: list[Event]) -> None:
        target = args.get("target")
        if target is None:
            events.append(Event(f"{seat} played Baron with no valid target; no effect."))
            return
        mine = state["players"][seat]["hand"][0]
        theirs = state["players"][target]["hand"][0]
        events.append(Event(f"{seat} played Baron and secretly compared hands with {target}."))
        events.append(
            Event(
                f"Baron comparison: {seat} holds the {mine}, {target} holds the {theirs}.",
                visible_to=(seat, target),
            )
        )
        if RANKS[mine] > RANKS[theirs]:
            events.append(Event(f"{target} had the lower card and is eliminated."))
            self._eliminate(state, target, events)
        elif RANKS[theirs] > RANKS[mine]:
            events.append(Event(f"{seat} had the lower card and is eliminated."))
            self._eliminate(state, seat, events)
        else:
            events.append(Event("The hands were tied; nothing happens."))

    def _play_handmaid(self, state: dict, seat: str, args: dict, events: list[Event]) -> None:
        state["players"][seat]["protected"] = True
        events.append(Event(f"{seat} played Handmaid and is protected until their next turn."))

    def _play_prince(self, state: dict, seat: str, args: dict, events: list[Event]) -> None:
        target = args["target"]
        discarded = state["players"][target]["hand"].pop()
        state["players"][target]["discard"].append(discarded)
        events.append(Event(f"{seat} played Prince: {target} discarded the {discarded}."))
        if discarded == "Princess":
            events.append(Event(f"{target} discarded the Princess and is eliminated."))
            self._eliminate(state, target, events, hand_already_discarded=True)
            return
        if state["deck"]:
            state["players"][target]["hand"].append(state["deck"].pop())
        elif state["removed_card"] is not None:
            state["players"][target]["hand"].append(state["removed_card"])
            state["removed_card"] = None
            events.append(Event(f"The deck was empty, so {target} drew the set-aside card."))
        events.append(Event(f"{target} drew a new card."))

    def _play_king(self, state: dict, seat: str, args: dict, events: list[Event]) -> None:
        target = args.get("target")
        if target is None:
            events.append(Event(f"{seat} played King with no valid target; no effect."))
            return
        mine = state["players"][seat]["hand"]
        theirs = state["players"][target]["hand"]
        state["players"][seat]["hand"], state["players"][target]["hand"] = theirs, mine
        events.append(Event(f"{seat} played King and swapped hands with {target}."))
        events.append(
            Event(
                f"After the swap you hold the {theirs[0]}.",
                visible_to=(seat,),
            )
        )
        events.append(
            Event(
                f"After the swap you hold the {mine[0]}.",
                visible_to=(target,),
            )
        )

    def _play_countess(self, state: dict, seat: str, args: dict, events: list[Event]) -> None:
        events.append(Event(f"{seat} played Countess; no effect."))

    def _play_princess(self, state: dict, seat: str, args: dict, events: list[Event]) -> None:
        events.append(Event(f"{seat} played the Princess and is eliminated."))
        self._eliminate(state, seat, events)

    # ------------------------------------------------------------- internals

    @staticmethod
    def _eliminate(
        state: dict,
        seat: str,
        events: list[Event],
        hand_already_discarded: bool = False,
    ) -> None:
        player = state["players"][seat]
        player["eliminated"] = True
        if not hand_already_discarded and player["hand"]:
            revealed = player["hand"].pop()
            player["discard"].append(revealed)
            events.append(Event(f"{seat} reveals and discards the {revealed} face up."))

    def _finish_turn(self, state: dict, events: list[Event]) -> None:
        standing = [s for s, p in state["players"].items() if not p["eliminated"]]

        if len(standing) == 1:
            self._end_round(state, standing, events, reason="last player standing")
            return
        if not state["deck"]:
            ranked = sorted(
                standing,
                key=lambda s: (
                    RANKS[state["players"][s]["hand"][0]],
                    sum(RANKS[c] for c in state["players"][s]["discard"]),
                ),
                reverse=True,
            )
            best = ranked[0]
            best_key = (
                RANKS[state["players"][best]["hand"][0]],
                sum(RANKS[c] for c in state["players"][best]["discard"]),
            )
            winners = [
                s
                for s in ranked
                if (
                    RANKS[state["players"][s]["hand"][0]],
                    sum(RANKS[c] for c in state["players"][s]["discard"]),
                )
                == best_key
            ]
            for s in standing:
                events.append(Event(f"Showdown: {s} reveals the {state['players'][s]['hand'][0]}."))
            self._end_round(state, winners, events, reason="the deck ran out")
            return

        # Advance to the next standing player; their protection expires; they draw.
        seats = list(state["players"])
        idx = seats.index(state["current_player"])
        for offset in range(1, len(seats) + 1):
            nxt = seats[(idx + offset) % len(seats)]
            if not state["players"][nxt]["eliminated"]:
                break
        state["current_player"] = nxt
        state["players"][nxt]["protected"] = False
        state["players"][nxt]["hand"].append(state["deck"].pop())
        events.append(Event(f"{nxt} drew a card and must now play."))

    def _end_round(self, state: dict, winners: list[str], events: list[Event], reason: str) -> None:
        for winner in winners:
            state["players"][winner]["tokens"] += 1
        names = " and ".join(winners)
        events.append(
            Event(
                f"The round ended ({reason}). {names} won the round and gained a "
                "token of affection."
            )
        )

        champions = [
            s for s, p in state["players"].items() if p["tokens"] >= state["tokens_to_win"]
        ]
        if champions:
            state["game_over"] = True
            state["winners"] = champions
            state["current_player"] = ""
            events.append(
                Event(
                    f"{' and '.join(champions)} reached "
                    f"{state['tokens_to_win']} tokens and won the game!"
                )
            )
            return

        rng = random.Random(state["rng_seed"])
        self._start_round(state, rng, starter=winners[0], events=events)
        state["rng_seed"] = rng.randrange(2**32)

    def _start_round(
        self, state: dict, rng: random.Random, starter: str, events: list[Event]
    ) -> None:
        deck = [card for card, count in COUNTS.items() for _ in range(count)]
        rng.shuffle(deck)
        state["round_number"] += 1
        state["removed_card"] = deck.pop()
        state["revealed_cards"] = (
            [deck.pop() for _ in range(3)] if state["num_players"] == 2 else []
        )
        for player in state["players"].values():
            player["hand"] = [deck.pop()]
            player["discard"] = []
            player["eliminated"] = False
            player["protected"] = False
        state["deck"] = deck
        state["current_player"] = starter
        state["players"][starter]["hand"].append(deck.pop())
        events.append(
            Event(f"Round {state['round_number']} begins. {starter} drew a card and must now play.")
        )

    # ------------------------------------------------------------ observation

    def observe(self, state: dict, seat: str) -> dict:
        if seat == "spectator":
            return copy.deepcopy(state)
        view = {
            "game": self.game_name,
            "you": seat,
            "round_number": state["round_number"],
            "tokens_to_win": state["tokens_to_win"],
            "current_player": state["current_player"],
            "deck_count": len(state["deck"]),
            "revealed_cards": list(state["revealed_cards"]),
            "your_hand": list(state["players"][seat]["hand"]),
            "players": {
                other: {
                    "tokens": player["tokens"],
                    "eliminated": player["eliminated"],
                    "protected": player["protected"],
                    "discard": list(player["discard"]),
                    "hand_count": len(player["hand"]),
                }
                for other, player in state["players"].items()
            },
        }
        return view
