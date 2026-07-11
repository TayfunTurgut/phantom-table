"""Contract harness checks for the reference engines."""

import pytest

from playtest.engine import Action, GameEngine
from playtest.engine.contract import (
    ContractViolation,
    assert_engine_contract,
    run_random_selfplay,
)
from playtest.games import love_letter, six_nimmt

ENGINES = [love_letter.Game(), six_nimmt.Game()]


@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.game_name)
def test_engine_satisfies_contract(engine):
    trajectories = assert_engine_contract(engine, games_per_count=40)
    assert all(t.status is not None and t.status.over for t in trajectories)


@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.game_name)
def test_selfplay_produces_winners_and_scores(engine):
    trajectory = run_random_selfplay(engine, engine.min_players, seed=7)
    assert trajectory.status.winners
    assert set(trajectory.status.scores) == {
        f"player_{i}" for i in range(1, engine.min_players + 1)
    }


@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.game_name)
def test_replay_is_deterministic(engine):
    a = run_random_selfplay(engine, engine.min_players, seed=11)
    b = run_random_selfplay(engine, engine.min_players, seed=11)
    assert a.fingerprint() == b.fingerprint()


def test_harness_rejects_engine_that_never_ends():
    class Endless:
        game_name = "Endless"
        min_players = 2
        max_players = 2

        def setup(self, num_players, seed):
            return {"rng_seed": seed}, []

        def to_act(self, state):
            return ["player_1"]

        def legal_actions(self, state, seat):
            return [Action(seat=seat, name="wait", label="Wait")]

        def apply(self, state, actions):
            return dict(state), []

        def observe(self, state, seat):
            return {}

        def status(self, state):
            from playtest.engine import GameStatus

            return GameStatus(over=False)

    with pytest.raises(ContractViolation, match="did not terminate"):
        run_random_selfplay(Endless(), 2, seed=0, max_steps=50)


def test_harness_rejects_mutating_engine():
    class Mutator(love_letter.Game):
        def apply(self, state, actions):
            new_state, events = super().apply(state, actions)
            state["players"]["player_1"]["tokens"] = 99  # mutate the input
            return new_state, events

    with pytest.raises(ContractViolation, match="mutated"):
        run_random_selfplay(Mutator(), 2, seed=0)


def test_reference_engine_is_a_game_engine():
    assert isinstance(love_letter.Game(), GameEngine)


def test_harness_rejects_oversized_menu():
    class HugeMenu:
        game_name = "HugeMenu"
        min_players = 2
        max_players = 2

        def setup(self, num_players, seed):
            return {"done": False}, []

        def to_act(self, state):
            return [] if state["done"] else ["player_1"]

        def legal_actions(self, state, seat):
            return [Action(seat=seat, name=f"opt_{i}", label=f"Option {i}") for i in range(150)]

        def apply(self, state, actions):
            return {"done": True}, []

        def observe(self, state, seat):
            return {}

        def status(self, state):
            from playtest.engine import GameStatus

            return GameStatus(over=state["done"])

    with pytest.raises(ContractViolation, match="staged decisions"):
        run_random_selfplay(HugeMenu(), 2, seed=0, max_menu_size=100)


@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.game_name)
def test_engine_satisfies_menu_size_cap(engine):
    assert_engine_contract(engine, games_per_count=20, max_menu_size=100)


def test_trajectory_records_max_menu():
    class Steps:
        game_name = "Steps"
        min_players = 2
        max_players = 2

        def setup(self, num_players, seed):
            return {"turn": 0}, []

        def to_act(self, state):
            return [] if state["turn"] >= 2 else ["player_1"]

        def legal_actions(self, state, seat):
            count = 3 if state["turn"] == 0 else 7
            return [Action(seat=seat, name=f"a{i}", label=f"A{i}") for i in range(count)]

        def apply(self, state, actions):
            return {"turn": state["turn"] + 1}, []

        def observe(self, state, seat):
            return {}

        def status(self, state):
            from playtest.engine import GameStatus

            return GameStatus(over=state["turn"] >= 2)

    trajectory = run_random_selfplay(Steps(), 2, seed=0)
    assert trajectory.max_menu == 7


def test_fingerprint_excludes_max_menu():
    trajectory = run_random_selfplay(love_letter.Game(), 2, seed=0)
    baseline = trajectory.fingerprint()
    trajectory.max_menu = 999999
    assert trajectory.fingerprint() == baseline
