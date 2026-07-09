"""The simultaneous-decision path: multi-seat ``to_act`` through harness and session.

Every shipped engine is sequential, so this minimal simultaneous-reveal engine
exists to keep the contract's "several seats decide right now" branch exercised:
the harness must collect one action per acting seat, and the session must give
each co-actor an information-isolated decision (no same-step table talk).
"""

import copy
import random

from playtest.agents.player import Decision
from playtest.engine import Action, Event, GameStatus, seats_for
from playtest.engine.contract import assert_engine_contract
from playtest.session import run_session
from playtest.ui.logger import GameLogger

ROUNDS = 3


class ParityDuel:
    """All seats simultaneously pick 0 or 1; the parity of the sum scores."""

    game_name = "Parity Duel"
    min_players = 2
    max_players = 3

    def setup(self, num_players: int, seed: int) -> tuple[dict, list[Event]]:
        if not self.min_players <= num_players <= self.max_players:
            raise ValueError(f"Parity Duel supports 2-3 players, got {num_players}")
        rng = random.Random(seed)
        state = {
            "rng_seed": rng.randrange(2**32),
            "num_players": num_players,
            "round": 1,
            "scores": {seat: 0 for seat in seats_for(num_players)},
        }
        return state, []

    def to_act(self, state: dict) -> list[str]:
        if state["round"] > ROUNDS:
            return []
        return seats_for(state["num_players"])

    def legal_actions(self, state: dict, seat: str) -> list[Action]:
        return [
            Action(seat=seat, name="pick", args={"value": v}, label=f"Pick {v}") for v in (0, 1)
        ]

    def apply(self, state: dict, actions: list[Action]) -> tuple[dict, list[Event]]:
        state = copy.deepcopy(state)
        expected = self.to_act(state)
        if [a.seat for a in actions] != expected:
            raise ValueError(f"need exactly one action per seat in {expected}, got {actions}")
        for action in actions:
            legal = {a.key() for a in self.legal_actions(state, action.seat)}
            if action.key() not in legal:
                raise ValueError(f"illegal action {action}")
        picks = {a.seat: a.args["value"] for a in actions}
        parity = sum(picks.values()) % 2
        scorers = [seat for seat, value in picks.items() if value == parity]
        for seat in scorers:
            state["scores"][seat] += 1
        events = [
            Event(
                f"Round {state['round']}: "
                + ", ".join(f"{seat} picked {picks[seat]}" for seat in expected)
                + f" — the sum was {'odd' if parity else 'even'}."
            )
        ]
        events.extend(Event(f"{seat} scored a point.", visible_to=(seat,)) for seat in scorers)
        state["round"] += 1
        return state, events

    def observe(self, state: dict, seat: str) -> dict:
        return {"you": seat, "round": state["round"], "scores": dict(state["scores"])}

    def status(self, state: dict) -> GameStatus:
        scores = {seat: float(value) for seat, value in state["scores"].items()}
        if state["round"] <= ROUNDS:
            return GameStatus(over=False, scores=scores)
        best = max(scores.values())
        winners = tuple(seat for seat, value in scores.items() if value == best)
        return GameStatus(over=True, winners=winners, scores=scores)


class RecordingPlayer:
    """Picks a fixed value every round; records the events it was shown."""

    def __init__(self, seat: str, value: int, table_talk: str | None = None) -> None:
        self.seat = seat
        self.value = value
        self.table_talk = table_talk
        self.events_per_decision: list[list[str]] = []

    def choose(self, observation: dict, legal: list[Action], events: list[str]) -> Decision:
        self.events_per_decision.append(list(events))
        action = next(a for a in legal if a.args["value"] == self.value)
        return Decision(action=action, reasoning="scripted", table_talk=self.table_talk)


class NullObserver:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def run_duel(players: dict) -> tuple[dict, GameLogger]:
    logger = GameLogger()
    result = run_session(
        ParityDuel(),
        players,
        NullObserver(),
        logger,
        num_players=len(players),
        seed=0,
        session_id="simultaneous-test",
    )
    return result, logger


def test_engine_honors_contract_with_multi_seat_to_act():
    engine = ParityDuel()
    trajectories = assert_engine_contract(engine, games_per_count=5)
    assert trajectories
    for trajectory in trajectories:
        for step in trajectory.steps:
            assert len(step.acting) >= 2, "every step must be a simultaneous decision"
            assert len(step.actions) == len(step.acting)


def test_session_collects_one_decision_per_acting_seat():
    players = {
        "player_1": RecordingPlayer("player_1", value=1),
        "player_2": RecordingPlayer("player_2", value=0),
        "player_3": RecordingPlayer("player_3", value=1),
    }
    result, logger = run_duel(players)

    assert result["status"].over
    decisions = [e for e in logger.log["events"] if e["type"] == "decision"]
    assert len(decisions) == ROUNDS * 3
    per_step = {e["step"] for e in decisions}
    assert per_step == set(range(1, ROUNDS + 1))
    # 1 + 0 + 1 is even: the 0-picker scores every round and wins alone.
    assert result["status"].winners == ("player_2",)
    assert result["status"].scores == {"player_1": 0.0, "player_2": 3.0, "player_3": 0.0}


def test_private_events_route_per_seat_in_simultaneous_steps():
    players = {
        "player_1": RecordingPlayer("player_1", value=1),
        "player_2": RecordingPlayer("player_2", value=0),
    }
    run_duel(players)
    # 1 + 0 is odd: player_1 scores each round and is privately told so.
    p1_seen = [line for events in players["player_1"].events_per_decision for line in events]
    p2_seen = [line for events in players["player_2"].events_per_decision for line in events]
    assert any("player_1 scored a point." in line for line in p1_seen)
    assert not any("scored a point" in line for line in p2_seen)


def test_same_step_table_talk_is_not_visible_to_co_actors():
    players = {
        "player_1": RecordingPlayer("player_1", value=1, table_talk="going big"),
        "player_2": RecordingPlayer("player_2", value=0),
    }
    run_duel(players)

    p2_events = players["player_2"].events_per_decision
    # player_2 decides after player_1 within each step; same-step talk must not leak.
    assert not any("going big" in line for line in p2_events[0])
    # The talk arrives with the next step's events instead.
    assert any('player_1 says: "going big"' in line for line in p2_events[1])
