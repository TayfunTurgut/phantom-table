"""Generic contract harness: random self-play validation for any GameEngine.

This is the engine-agnostic half of validation. It cannot judge whether an
engine implements its rulebook faithfully — only that it honors the GameEngine
contract: games terminate, apply never crashes or mutates its input, states and
observations stay JSON-serializable, and identical seeds replay identically.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from playtest.engine import SPECTATOR, Action, Event, GameEngine, GameStatus, seats_for


class ContractViolation(AssertionError):
    """An engine broke the GameEngine contract during self-play."""


@dataclass
class Step:
    """One decision step of a trajectory."""

    state: dict
    acting: list[str]
    actions: list[Action]
    events: list[Event]


@dataclass
class Trajectory:
    """The full record of one self-played game."""

    num_players: int
    seed: int
    steps: list[Step] = field(default_factory=list)
    final_state: dict | None = None
    status: GameStatus | None = None

    def fingerprint(self) -> str:
        """Deterministic digest used to compare replays of the same seed."""
        payload = {
            "steps": [
                {
                    "acting": step.acting,
                    "actions": [a.key() for a in step.actions],
                    "events": [(e.text, e.visible_to) for e in step.events],
                }
                for step in self.steps
            ],
            "final_state": self.final_state,
        }
        return json.dumps(payload, sort_keys=True, default=list)


def _check_serializable(value: dict, what: str) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ContractViolation(f"{what} is not JSON-serializable: {exc}") from exc


def run_random_selfplay(
    engine: GameEngine,
    num_players: int,
    seed: int,
    max_steps: int = 10_000,
) -> Trajectory:
    """Play one game with uniformly random choices, checking the contract per step."""
    rng = random.Random(seed)
    seats = seats_for(num_players)
    state = engine.setup(num_players, seed)
    _check_serializable(state, "setup() state")
    trajectory = Trajectory(num_players=num_players, seed=seed)

    for _ in range(max_steps):
        status = engine.status(state)
        acting = engine.to_act(state)
        if status.over:
            if acting:
                raise ContractViolation(f"status().over is True but to_act() returned {acting}")
            trajectory.final_state = state
            trajectory.status = status
            return trajectory
        if not acting:
            raise ContractViolation("status().over is False but to_act() returned []")
        if len(set(acting)) != len(acting) or not set(acting) <= set(seats):
            raise ContractViolation(f"to_act() returned invalid seats: {acting}")

        before = json.dumps(state, sort_keys=True)
        chosen: list[Action] = []
        for seat in acting:
            legal = engine.legal_actions(state, seat)
            if not legal:
                raise ContractViolation(
                    f"legal_actions() empty for acting seat {seat}; enumerate a pass "
                    "action if the rules allow doing nothing"
                )
            stable = engine.legal_actions(state, seat)
            if [a.key() for a in legal] != [a.key() for a in stable]:
                raise ContractViolation(
                    f"legal_actions() for {seat} is not deterministic for a fixed state"
                )
            for action in legal:
                if action.seat != seat:
                    raise ContractViolation(
                        f"legal_actions(state, {seat!r}) yielded action for {action.seat!r}"
                    )
            _check_serializable(engine.observe(state, seat), f"observe() for {seat}")
            chosen.append(rng.choice(legal))
        _check_serializable(engine.observe(state, SPECTATOR), "observe() for spectator")

        state_next, events = engine.apply(state, chosen)
        if json.dumps(state, sort_keys=True) != before:
            raise ContractViolation("apply() mutated its input state")
        _check_serializable(state_next, "apply() state")
        trajectory.steps.append(Step(state=state, acting=acting, actions=chosen, events=events))
        state = state_next

    raise ContractViolation(
        f"game did not terminate within {max_steps} steps (num_players={num_players}, seed={seed})"
    )


def assert_engine_contract(
    engine: GameEngine,
    player_counts: list[int] | None = None,
    games_per_count: int = 50,
    max_steps: int = 10_000,
) -> list[Trajectory]:
    """Self-play many seeded games per player count; raise on any contract breach.

    Also replays the first seed of each player count and requires an identical
    trajectory fingerprint (determinism).
    """
    counts = player_counts or list(range(engine.min_players, engine.max_players + 1))
    trajectories: list[Trajectory] = []
    for num_players in counts:
        for seed in range(games_per_count):
            trajectories.append(run_random_selfplay(engine, num_players, seed, max_steps=max_steps))
        replay = run_random_selfplay(engine, num_players, 0, max_steps=max_steps)
        first = next(t for t in trajectories if t.num_players == num_players and t.seed == 0)
        if replay.fingerprint() != first.fingerprint():
            raise ContractViolation(
                f"replaying seed 0 with {num_players} players produced a different "
                "trajectory; the engine is not deterministic"
            )
    return trajectories
