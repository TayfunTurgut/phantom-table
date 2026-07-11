"""Staged-decision contract test: a minimal open-auction inline engine.

Pins contract point 4 (split an open-ended decision into consecutive stages,
same seat, recording the partial choice in state) permanently in the test
suite. Not a shipped exemplar: it lives in this test file only.

Rules: 3 prizes (worth points), auctioned one at a time; each seat starts
with 20 coins. On a seat's bidding turn the choice is staged: stage 1 is
``bid``/``pass``; stage 2 (only reachable after ``bid``) picks a fully bound
amount from (current_high_bid+1 .. the seat's own coin count) and stays with
the same seat. A seat that can't afford to outbid only sees ``pass`` — still a
single, non-empty legal set. Passing removes a seat from the auction; once
only one seat remains, the auction resolves (winner pays and gets the prize,
or the prize goes unclaimed if nobody ever bid) and the next prize begins.
Score is prize points only — coins are not used as a tiebreak, kept simple.
"""

from __future__ import annotations

import copy
import random

from playtest.agents.player import Decision
from playtest.engine import Action, Event, GameStatus, seats_for
from playtest.engine.contract import assert_engine_contract
from playtest.session import run_session
from playtest.ui.logger import GameLogger

PRIZES = [5, 8, 3]
START_COINS = 20


class StagedAuction:
    """A fixed set of prizes, auctioned one at a time via a staged bid."""

    game_name = "Staged Auction"
    min_players = 2
    max_players = 4

    def setup(self, num_players: int, seed: int) -> tuple[dict, list[Event]]:
        if not self.min_players <= num_players <= self.max_players:
            raise ValueError(f"Staged Auction supports 2-4 players, got {num_players}")
        rng = random.Random(seed)
        order = seats_for(num_players)
        state = {
            "rng_seed": rng.randrange(2**32),
            "order": order,
            "coins": dict.fromkeys(order, START_COINS),
            "scores": dict.fromkeys(order, 0),
            "prize_idx": 0,
            "high_bid": 0,
            "high_bidder": None,
            "active": list(order),
            "pending_bid_seat": None,
        }
        return state, []

    def to_act(self, state: dict) -> list[str]:
        if state["prize_idx"] >= len(PRIZES):
            return []
        return [state["active"][0]]

    def legal_actions(self, state: dict, seat: str) -> list[Action]:
        if state["pending_bid_seat"] == seat:
            lo, hi = state["high_bid"] + 1, state["coins"][seat]
            return [
                Action(seat=seat, name="raise_to", args={"amount": amt}, label=f"Bid {amt}")
                for amt in range(lo, hi + 1)
            ]
        actions = []
        if state["coins"][seat] > state["high_bid"]:
            actions.append(Action(seat=seat, name="bid", label="Bid"))
        actions.append(Action(seat=seat, name="pass", label="Pass"))
        return actions

    def _resolve_auction(self, state: dict) -> list[Event]:
        prize_value = PRIZES[state["prize_idx"]]
        winner = state["high_bidder"]
        if winner is not None:
            state["coins"][winner] -= state["high_bid"]
            state["scores"][winner] += prize_value
            events = [
                Event(f"{winner} wins the {prize_value}-point prize, paying {state['high_bid']}.")
            ]
        else:
            events = [Event(f"No one bid; the {prize_value}-point prize goes unclaimed.")]
        state["prize_idx"] += 1
        if state["prize_idx"] < len(PRIZES):
            state["high_bid"] = 0
            state["high_bidder"] = None
            state["active"] = list(state["order"])
            state["pending_bid_seat"] = None
        return events

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

        if action.name == "bid":
            state["pending_bid_seat"] = seat
            return state, []

        if action.name == "raise_to":
            amount = action.args["amount"]
            state["high_bid"] = amount
            state["high_bidder"] = seat
            state["pending_bid_seat"] = None
            state["active"] = state["active"][1:] + [seat]
            return state, [Event(f"{seat} bid {amount}.")]

        # pass
        state["active"].pop(0)
        events = [Event(f"{seat} passes.")]
        if len(state["active"]) == 1:
            events.extend(self._resolve_auction(state))
        return state, events

    def observe(self, state: dict, seat: str) -> dict:
        return {
            "you": seat,
            "prize_idx": state["prize_idx"],
            "prizes_remaining": PRIZES[state["prize_idx"] :],
            "coins": dict(state["coins"]),
            "scores": dict(state["scores"]),
            "high_bid": state["high_bid"],
            "high_bidder": state["high_bidder"],
            "to_act": state["active"][0] if state["active"] else None,
            "pending_bid_seat": state["pending_bid_seat"],
        }

    def status(self, state: dict) -> GameStatus:
        scores = {s: float(v) for s, v in state["scores"].items()}
        if state["prize_idx"] < len(PRIZES):
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


class CrashBetweenStages(FirstLegalPlayer):
    """Raises the first time it is asked to act while its own bid is staged."""

    def __init__(self, seat: str) -> None:
        super().__init__(seat)
        self.crashed = False

    def choose(self, observation, legal, events):
        if not self.crashed and observation["pending_bid_seat"] == self.seat:
            self.crashed = True
            raise RuntimeError("simulated crash mid-stage")
        return super().choose(observation, legal, events)


def _play(engine: StagedAuction, state: dict, seat: str, name: str, **args):
    action = next(a for a in engine.legal_actions(state, seat) if a.name == name and a.args == args)
    return engine.apply(state, [action])


# --- tests -------------------------------------------------------------


def test_engine_honors_contract():
    assert_engine_contract(StagedAuction(), max_menu_size=50)


def test_same_seat_acts_across_the_stage_boundary():
    engine = StagedAuction()
    state, _ = engine.setup(num_players=2, seed=0)
    assert engine.to_act(state) == ["player_1"]

    state, _ = _play(engine, state, "player_1", "bid")
    assert engine.to_act(state) == ["player_1"]  # still player_1, now choosing the amount
    assert all(a.name == "raise_to" for a in engine.legal_actions(state, "player_1"))


def test_partial_choice_is_visible_in_own_observation():
    engine = StagedAuction()
    state, _ = engine.setup(num_players=2, seed=0)
    state, _ = _play(engine, state, "player_1", "bid")
    assert engine.observe(state, "player_1")["pending_bid_seat"] == "player_1"


def test_bid_event_emitted_once_at_final_stage():
    engine = StagedAuction()
    state, _ = engine.setup(num_players=2, seed=0)
    state, events = _play(engine, state, "player_1", "bid")
    assert events == []  # nothing announced yet at stage 1

    state, events = _play(engine, state, "player_1", "raise_to", amount=1)
    bid_events = [e for e in events if "bid" in e.text]
    assert len(bid_events) == 1
    assert bid_events[0].text == "player_1 bid 1."


def test_stage_2_bid_amount_bounds():
    """Stage 2 legal amounts must be fully bound by (high_bid+1 .. acting seat's coins)."""
    engine = StagedAuction()
    state, _ = engine.setup(num_players=2, seed=0)
    # Initial state: high_bid=0, both seats have 20 coins
    assert state["high_bid"] == 0
    assert state["coins"]["player_1"] == 20
    assert state["coins"]["player_2"] == 20

    # player_1 plays bid to reach stage 2
    state, _ = _play(engine, state, "player_1", "bid")

    # In stage 2, legal amounts should be 1..20 (high_bid+1 to player_1's coins)
    legal_actions = engine.legal_actions(state, "player_1")
    amounts = {a.args["amount"] for a in legal_actions}
    assert amounts == set(range(1, 21)), f"Expected 1-20, got {sorted(amounts)}"


def test_raise_to_amount_bounds_follow_acting_seat():
    """Upper bound of stage-2 amounts must be the acting seat's coin count, not another seat's."""
    engine = StagedAuction()
    state, _ = engine.setup(num_players=2, seed=0)

    # Reduce player_1's coins to 15 (while player_2 still has 20)
    state["coins"]["player_1"] = 15

    # player_1 plays bid to reach stage 2
    state, _ = _play(engine, state, "player_1", "bid")

    # Upper bound must be player_1's coins (15), not player_2's (20)
    legal_actions = engine.legal_actions(state, "player_1")
    amounts = {a.args["amount"] for a in legal_actions}
    assert amounts == set(range(1, 16)), f"Expected 1-15, got {sorted(amounts)}"


def test_checkpoint_round_trip_mid_stage(tmp_path):
    seed = 3

    baseline = run_session(
        StagedAuction(),
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
        "player_2": CrashBetweenStages("player_2"),
    }
    try:
        run_session(
            StagedAuction(),
            crashing_players,
            NullObserver(),
            GameLogger(),
            num_players=2,
            seed=seed,
            session_id="crash",
            checkpoint_path=checkpoint_path,
            game_ref="test_staged.StagedAuction",
        )
        raise AssertionError("expected the crash to propagate")
    except RuntimeError as exc:
        assert "simulated crash mid-stage" in str(exc)

    from playtest.checkpoint import load_checkpoint

    cp = load_checkpoint(checkpoint_path)
    assert cp.state["pending_bid_seat"] == "player_2"  # checkpointed mid-stage

    resumed = run_session(
        StagedAuction(),
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
