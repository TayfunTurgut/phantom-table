"""Session driver tests: scripted (non-LLM) players over the reference engine."""

import random

import pytest

from playtest.agents.player import Decision
from playtest.engine import Action, Event, GameStatus
from playtest.errors import EngineCrash, PlaytestError
from playtest.games.love_letter import Game
from playtest.session import run_session
from playtest.ui.logger import GameLogger


class ScriptedPlayer:
    """Picks a seeded-random legal action; records what it was shown."""

    def __init__(self, seat: str, seed: int = 0, table_talk: str | None = None) -> None:
        self.seat = seat
        self.rng = random.Random(seed)
        self.table_talk = table_talk
        self.seen_events: list[str] = []
        self.seen_observations: list[dict] = []

    def choose(self, observation, legal, events):
        self.seen_events.extend(events)
        self.seen_observations.append(observation)
        return Decision(
            action=self.rng.choice(legal),
            reasoning="scripted",
            table_talk=self.table_talk,
        )


class NullObserver:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def run_scripted(seed=5, num_players=2, talk=None):
    engine = Game()
    players = {
        f"player_{i}": ScriptedPlayer(f"player_{i}", seed=i, table_talk=talk)
        for i in range(1, num_players + 1)
    }
    logger = GameLogger()
    result = run_session(
        engine,
        players,
        NullObserver(),
        logger,
        num_players=num_players,
        seed=seed,
        session_id="test",
    )
    return result, logger, players


def test_full_game_runs_to_completion():
    result, logger, _ = run_scripted()
    assert result["status"].over
    assert result["status"].winners
    assert logger.log["winners"] == list(result["status"].winners)
    assert logger.log["total_steps"] == result["total_steps"] > 0


def test_session_is_deterministic_with_scripted_players():
    a, _, _ = run_scripted(seed=9)
    b, _, _ = run_scripted(seed=9)
    assert a["final_state"] == b["final_state"]
    assert a["total_steps"] == b["total_steps"]


def test_private_events_reach_only_their_audience():
    _, logger, players = run_scripted(seed=2)
    private = [
        e
        for e in logger.log["events"]
        if e["type"] == "engine_event" and e.get("visible_to") is not None
    ]
    assert private, "expected at least one private event (Priest/Baron/King) in a full game"
    for event in private:
        for seat, player in players.items():
            if seat in event["visible_to"]:
                continue
            assert event["text"] not in player.seen_events


def test_table_talk_is_routed_to_opponents():
    _, _, players = run_scripted(seed=5, talk="hello there")
    assert any('player_1 says: "hello there"' in line for line in players["player_2"].seen_events)


def test_engine_exception_is_wrapped_as_engine_crash():
    class Broken(Game):
        def apply(self, state, actions):
            raise KeyError("generated-code bug")

    players = {
        "player_1": ScriptedPlayer("player_1"),
        "player_2": ScriptedPlayer("player_2"),
    }
    with pytest.raises(EngineCrash) as excinfo:
        run_session(
            Broken(),
            players,
            NullObserver(),
            GameLogger(),
            num_players=2,
            seed=3,
            session_id="t",
        )
    assert excinfo.value.seed == 3
    assert excinfo.value.step == 1


def test_session_crashes_when_max_steps_exceeded():
    class Stuck(Game):
        def status(self, state):
            return GameStatus(over=False)

        def to_act(self, state):
            return ["player_1"]

        def legal_actions(self, state, seat):
            return [Action(seat=seat, name="wait", label="Wait")]

        def apply(self, state, actions):
            return dict(state), [Event("nothing happened")]

    players = {
        "player_1": ScriptedPlayer("player_1"),
        "player_2": ScriptedPlayer("player_2"),
    }
    with pytest.raises(PlaytestError, match="max_steps"):
        run_session(
            Stuck(),
            players,
            NullObserver(),
            GameLogger(),
            num_players=2,
            seed=0,
            session_id="t",
            max_steps=10,
        )


def test_confused_decision_is_logged():
    class ConfusedPlayer(ScriptedPlayer):
        def choose(self, observation, legal, events):
            return Decision(action=legal[0], reasoning="?", table_talk=None, confused=True)

    engine = Game()
    players = {
        "player_1": ConfusedPlayer("player_1"),
        "player_2": ConfusedPlayer("player_2"),
    }
    logger = GameLogger()
    run_session(
        engine,
        players,
        NullObserver(),
        logger,
        num_players=2,
        seed=1,
        session_id="t",
    )
    assert any(e["type"] == "player_confusion" for e in logger.log["events"])


def test_setup_narration_reaches_event_log_and_player_memory():
    """Round-1 setup events must flow through the same pipeline as apply() events.

    Before setup() returned events, the round-1 "Round N begins" narration was
    appended to a throwaway list and never reached the log or any player's
    memory, while every later round (started from apply()) did.
    """
    _, logger, players = run_scripted(seed=5, num_players=2)
    logged = [
        e["text"]
        for e in logger.log["events"]
        if e["type"] == "engine_event" and e["text"].startswith("Round 1 begins")
    ]
    assert logged, "round-1 setup narration was dropped from the event log"
    assert any(line.startswith("Round 1 begins") for line in players["player_1"].seen_events), (
        "round-1 setup narration never reached player_1's memory"
    )


def test_round_one_face_up_reveal_is_narrated_to_players():
    """The three face-up cards in a 2-player game must be narrated from round 1,
    not only from round 2 onward."""
    _, logger, players = run_scripted(seed=5, num_players=2)
    reveals = [
        e["text"]
        for e in logger.log["events"]
        if e["type"] == "engine_event" and e["text"].startswith("Setup revealed")
    ]
    assert reveals, "face-up reveal never appeared in the event log"
    assert any(line.startswith("Setup revealed") for line in players["player_1"].seen_events), (
        "round-1 face-up reveal never reached player_1's memory"
    )
