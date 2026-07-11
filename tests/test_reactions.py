"""Reaction-window contract test: a minimal "Nope duel" inline engine.

Pins contract point 5 (a "may respond" window is a decision point, stored as
``pending`` in state, closed only when every eligible responder has acted)
permanently in the test suite, the way ``tests/test_simultaneous.py`` pins the
multi-seat ``to_act`` branch with ``ParityDuel``. Not a shipped exemplar: it
lives in this test file only.

Rules: each seat starts with point cards [1, 2, 3] and one ``nope``. On your
turn you play a point card, announcing it. This opens a reaction window: each
other seat, in seat order, may play their nope (cancelling the pending card)
or decline. A nope opens one more window over the remaining eligible seats,
where a counter-nope restores the play — the pending state is a small stack of
seats who have noped, and the card scores iff the stack length is even. The
window is offered to every eligible seat regardless of whether they hold a
nope: that is the info-leak property under test.
"""

from __future__ import annotations

import copy
import random

from playtest.agents.player import Decision
from playtest.engine import Action, Event, GameStatus, seats_for
from playtest.engine.contract import assert_engine_contract
from playtest.session import run_session
from playtest.ui.logger import GameLogger

POINTS = [1, 2, 3]


def _announcer(pending: dict) -> str:
    """The seat whose play currently sits on top of the pending stack."""
    stack = pending["stack"]
    return stack[-1] if len(stack) % 2 else pending["owner"]


class NopeDuel:
    """Point cards score unless noped; a counter-nope restores them."""

    game_name = "Nope Duel"
    min_players = 2
    max_players = 4

    def setup(self, num_players: int, seed: int) -> tuple[dict, list[Event]]:
        if not self.min_players <= num_players <= self.max_players:
            raise ValueError(f"Nope Duel supports 2-4 players, got {num_players}")
        rng = random.Random(seed)
        order = seats_for(num_players)
        state = {
            "rng_seed": rng.randrange(2**32),
            "order": order,
            "hands": {seat: {"points": list(POINTS), "nope": 1} for seat in order},
            "scores": dict.fromkeys(order, 0),
            "turn": 0,
            "phase": "main",
            "pending": None,
        }
        return state, []

    def _eligible(self, state: dict, exclude: str) -> list[str]:
        order = state["order"]
        i = order.index(exclude)
        return order[i + 1 :] + order[:i]

    def _current_player(self, state: dict) -> str | None:
        order = state["order"]
        n = len(order)
        idx = state["turn"] % n
        for _ in range(n):
            seat = order[idx]
            if state["hands"][seat]["points"]:
                return seat
            idx = (idx + 1) % n
        return None

    def to_act(self, state: dict) -> list[str]:
        if state["phase"] == "window":
            pending = state["pending"]
            announcer = _announcer(pending)
            for seat in self._eligible(state, announcer):
                if seat not in pending["asked"]:
                    return [seat]
            return []
        seat = self._current_player(state)
        return [seat] if seat is not None else []

    def legal_actions(self, state: dict, seat: str) -> list[Action]:
        if state["phase"] == "window":
            actions = [Action(seat=seat, name="decline", label="Decline")]
            if state["hands"][seat]["nope"] > 0:
                actions.append(Action(seat=seat, name="nope", label="Play Nope!"))
            return actions
        return [
            Action(seat=seat, name="play_point", args={"value": v}, label=f"Play {v}")
            for v in state["hands"][seat]["points"]
        ]

    def apply(self, state: dict, actions: list[Action]) -> tuple[dict, list[Event]]:
        state = copy.deepcopy(state)
        expected = self.to_act(state)
        if [a.seat for a in actions] != expected:
            raise ValueError(f"need exactly one action for {expected}, got {actions}")
        action = actions[0]
        seat = action.seat
        legal = {a.key() for a in self.legal_actions(state, seat)}
        if action.key() not in legal:
            raise ValueError(f"illegal action {action}")

        if state["phase"] == "main":
            value = action.args["value"]
            state["hands"][seat]["points"].remove(value)
            state["pending"] = {"value": value, "owner": seat, "stack": [], "asked": []}
            state["phase"] = "window"
            return state, [Event(f"{seat} plays a {value}-point card.")]

        pending = state["pending"]
        events: list[Event] = []
        if action.name == "nope":
            state["hands"][seat]["nope"] -= 1
            pending["stack"].append(seat)
            pending["asked"] = []
            events.append(Event(f"{seat} plays Nope! on the {pending['value']}-point card."))
        else:
            pending["asked"].append(seat)
            events.append(Event(f"{seat} declines."))

        announcer = _announcer(pending)
        if any(s not in pending["asked"] for s in self._eligible(state, announcer)):
            return state, events  # window stays open for the next responder

        owner, value = pending["owner"], pending["value"]
        if len(pending["stack"]) % 2 == 0:
            state["scores"][owner] += value
            events.append(Event(f"{owner}'s {value}-point card scores."))
        else:
            events.append(Event(f"{owner}'s {value}-point card is cancelled — scores nothing."))
        state["pending"] = None
        state["phase"] = "main"
        state["turn"] = (state["order"].index(owner) + 1) % len(state["order"])
        return state, events

    def observe(self, state: dict, seat: str) -> dict:
        view = {
            "you": seat,
            "phase": state["phase"],
            "scores": dict(state["scores"]),
            "hands": {
                s: (dict(h) if s == seat else {"cards_remaining": len(h["points"]) + h["nope"]})
                for s, h in state["hands"].items()
            },
        }
        if state["pending"] is not None:
            p = state["pending"]
            view["pending"] = {"value": p["value"], "owner": p["owner"], "stack": list(p["stack"])}
        return view

    def status(self, state: dict) -> GameStatus:
        scores = {s: float(v) for s, v in state["scores"].items()}
        if state["phase"] == "window" or any(state["hands"][s]["points"] for s in state["order"]):
            return GameStatus(over=False, scores=scores)
        best = max(scores.values())
        winners = tuple(s for s, v in scores.items() if v == best)
        return GameStatus(over=True, winners=winners, scores=scores)


# --- test fixtures ----------------------------------------------------------


class NullObserver:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FirstLegalPlayer:
    """Always picks the first legal action — a pure function of state."""

    def __init__(self, seat: str) -> None:
        self.seat = seat

    def choose(self, observation, legal, events):
        return Decision(action=legal[0], reasoning="first legal", table_talk=None)


class CrashInWindow(FirstLegalPlayer):
    """Raises the first time it is asked to respond while a window is open."""

    def __init__(self, seat: str) -> None:
        super().__init__(seat)
        self.crashed = False

    def choose(self, observation, legal, events):
        if not self.crashed and observation["phase"] == "window":
            self.crashed = True
            raise RuntimeError("simulated crash mid-window")
        return super().choose(observation, legal, events)


def _play(engine: NopeDuel, state: dict, seat: str, name: str, **args):
    action = next(a for a in engine.legal_actions(state, seat) if a.name == name and a.args == args)
    return engine.apply(state, [action])


# --- tests -------------------------------------------------------------


def test_engine_honors_contract():
    assert_engine_contract(NopeDuel())


def test_window_offered_to_a_seat_with_no_nope_left_is_decline_only():
    """The window is offered by rules eligibility, not hand contents."""
    engine = NopeDuel()
    state, _ = engine.setup(num_players=2, seed=0)

    # Round 1: player_1 plays 1; player_2 nopes (spending their only nope);
    # player_1 declines the counter-window -> the 1-point card is cancelled.
    state, _ = _play(engine, state, "player_1", "play_point", value=1)
    state, _ = _play(engine, state, "player_2", "nope")
    state, _ = _play(engine, state, "player_1", "decline")
    assert engine.to_act(state) == ["player_2"]  # turn passed to player_2

    # Round 2: player_2 plays 1; player_1 declines -> scores normally.
    state, _ = _play(engine, state, "player_2", "play_point", value=1)
    state, _ = _play(engine, state, "player_1", "decline")
    assert engine.to_act(state) == ["player_1"]  # turn passed back

    # Round 3: player_1 plays 2, opening a window on player_2 — who is now
    # completely out of nopes.
    state, _ = _play(engine, state, "player_1", "play_point", value=2)

    assert engine.to_act(state) == ["player_2"]
    menu = engine.legal_actions(state, "player_2")
    assert [a.name for a in menu] == ["decline"]


def test_all_decline_resolves_pending_play_with_scoring_and_events():
    engine = NopeDuel()
    state, _ = engine.setup(num_players=2, seed=1)
    state, _ = _play(engine, state, "player_1", "play_point", value=3)
    state, events = _play(engine, state, "player_2", "decline")
    assert state["scores"]["player_1"] == 3
    assert any("3-point card scores" in e.text for e in events)


def test_nope_cancels_and_counter_nope_restores():
    engine = NopeDuel()

    # A lone nope cancels: net score 0 for that card.
    state, _ = engine.setup(num_players=2, seed=2)
    state, _ = _play(engine, state, "player_1", "play_point", value=2)
    state, events = _play(engine, state, "player_2", "nope")
    assert any("plays Nope!" in e.text for e in events)
    state, events = _play(engine, state, "player_1", "decline")
    assert state["scores"]["player_1"] == 0
    assert any("cancelled" in e.text for e in events)

    # A counter-nope restores the play: full score for the card. The counter
    # itself reopens one more window to player_2 — now nope-less, decline-only
    # — before the play resolves.
    state2, _ = engine.setup(num_players=2, seed=2)
    state2, _ = _play(engine, state2, "player_1", "play_point", value=2)
    state2, _ = _play(engine, state2, "player_2", "nope")
    state2, _ = _play(engine, state2, "player_1", "nope")
    assert [a.name for a in engine.legal_actions(state2, "player_2")] == ["decline"]
    state2, events = _play(engine, state2, "player_2", "decline")
    assert state2["scores"]["player_1"] == 2
    assert any("card scores" in e.text for e in events)


def test_checkpoint_round_trip_mid_window(tmp_path):
    seed = 7

    baseline = run_session(
        NopeDuel(),
        {"player_1": FirstLegalPlayer("player_1"), "player_2": FirstLegalPlayer("player_2")},
        NullObserver(),
        GameLogger(),
        num_players=2,
        seed=seed,
        session_id="baseline",
    )

    checkpoint_path = str(tmp_path / "ck.json")
    crashing_players = {
        "player_1": FirstLegalPlayer("player_1"),
        "player_2": CrashInWindow("player_2"),
    }
    try:
        run_session(
            NopeDuel(),
            crashing_players,
            NullObserver(),
            GameLogger(),
            num_players=2,
            seed=seed,
            session_id="crash",
            checkpoint_path=checkpoint_path,
            game_ref="test_reactions.NopeDuel",
        )
        raise AssertionError("expected the crash to propagate")
    except RuntimeError as exc:
        assert "simulated crash mid-window" in str(exc)

    from playtest.checkpoint import load_checkpoint

    cp = load_checkpoint(checkpoint_path)
    assert cp.state["phase"] == "window"  # checkpointed mid-window, as intended

    resumed = run_session(
        NopeDuel(),
        {"player_1": FirstLegalPlayer("player_1"), "player_2": FirstLegalPlayer("player_2")},
        NullObserver(),
        GameLogger(),
        num_players=cp.num_players,
        seed=cp.seed,
        session_id=cp.session_id,
        resume=cp,
    )

    assert resumed["final_state"] == baseline["final_state"]
    assert resumed["status"] == baseline["status"]
