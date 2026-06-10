"""Offline end-to-end driver tests: stub GM/players drive the real turn loop."""

import copy
import json
from pathlib import Path

import pytest
from rich.console import Console

from playtest.agents.gm import GMResolution
from playtest.agents.player import PlayerAction
from playtest.errors import IllegalAction
from playtest.rules import GameRules
from playtest.session import run_session
from playtest.state.manager import GameStateManager
from playtest.ui.logger import GameLogger
from playtest.ui.observer import GameObserver

from .fixtures import sample_config

# --- Stubs that drive the plain turn loop offline (no API calls) -------------


class StubGM:
    """Deterministic GM: resolves draw/play against a real manager, no LLM.

    ``play_script`` is a list of per-play directives (popped in order) that apply side
    effects and resolution flags — e.g. ``{"eliminate": "player_2", "round_ended": True}``
    or ``{"game_ended": True, "winners": ["player_1"]}`` — so a test can steer when a
    round or the game ends (the driver acts purely on these GM-reported flags).
    """

    def __init__(self, num_players: int, manager: GameStateManager, play_script=None):
        config = sample_config()
        self.num_players = num_players
        self.spec = config.game_spec
        self.rules = GameRules(config)
        self.manager = manager
        self._play_script = list(play_script or [])
        self.validate_calls: list[tuple[str, str]] = []
        self.round_end_calls = 0

    def initialize_game(self, num_players: int | None = None, seed: int | None = None) -> dict:
        state = self.rules.setup(num_players or self.num_players, seed)
        gm_view = self.manager.initialize(state, self.spec.visibility)
        return {"state": gm_view, "narration": "A new game begins."}

    def validate_and_resolve(self, action: dict, player_id: str) -> GMResolution:
        self.validate_calls.append((player_id, action["action_type"]))
        state = copy.deepcopy(self.manager.get_state("gm"))
        directive: dict = {}
        if action["action_type"] == "draw_card":
            drawn = state["deck"][0]
            state["deck"] = state["deck"][1:]
            state["deck_count"] = len(state["deck"])
            hand = state["players"][player_id]["hand"]
            state["players"][player_id]["hand"] = hand + [drawn]
            state["players"][player_id]["hand_count"] = len(hand) + 1
            state["turn_phase"] = "play"
        else:  # a play ends the turn
            hand = state["players"][player_id]["hand"]
            played, kept = hand[0], hand[1:]
            state["players"][player_id]["hand"] = kept
            state["players"][player_id]["hand_count"] = len(kept)
            state["players"][player_id]["discards"] = (
                state["players"][player_id]["discards"] + [played]
            )
            directive = self._play_script.pop(0) if self._play_script else {}
            if directive.get("eliminate"):
                tgt = directive["eliminate"]
                state["players"][tgt]["is_eliminated"] = True
        new_state = self.manager.set_state(state)
        return GMResolution(
            is_valid=True,
            narration="The action resolves.",
            new_state=new_state,
            round_ended=bool(directive.get("round_ended")),
            game_ended=bool(directive.get("game_ended")),
            winners=directive.get("winners"),
        )

    def handle_round_end(self) -> GMResolution:
        self.round_end_calls += 1
        state = copy.deepcopy(self.manager.get_state("gm"))
        survivors = [pid for pid, p in state["players"].items() if not p["is_eliminated"]]
        winner = survivors[0] if survivors else "player_1"
        state["players"][winner]["tokens"] = state["tokens_to_win"]  # reach the goal -> game over
        new_state = self.manager.set_state(state)
        return GMResolution(
            is_valid=True,
            narration="The round is over.",
            new_state=new_state,
            round_ended=True,
            game_ended=True,
            winner=winner,
            winners=[winner],
        )


class StubPlayer:
    def __init__(self, player_id: str, calls: list[str]):
        self.player_id = player_id
        self._calls = calls

    def _propose(self, action_type: str, resolve_action) -> dict:
        outcome = resolve_action(
            PlayerAction(
                player_id=self.player_id,
                action_type=action_type,
                parameters={},
                reasoning="r",
                public_statement="I act.",
            )
        )
        # A rejected proposal is retried (the resolver crashes past the retry cap).
        while outcome.get("rejected"):
            outcome = resolve_action(
                PlayerAction(
                    player_id=self.player_id,
                    action_type=action_type,
                    parameters={},
                    reasoning="retry",
                    public_statement="I try again.",
                )
            )
        return outcome

    def take_turn(self, filtered_state, context=None, private_memory=None, *, resolve_action):
        self._calls.append(self.player_id)
        outcome = self._propose("draw_card", resolve_action)
        if outcome.get("turn_ended"):
            return None
        self._propose("play_guard", resolve_action)
        return None


def _run(gm: StubGM, num_players: int) -> tuple[dict, list[str], GameLogger]:
    calls: list[str] = []
    players = {f"player_{i}": StubPlayer(f"player_{i}", calls) for i in range(1, num_players + 1)}
    observer = GameObserver(console=Console(record=True, width=100), verbose=False)
    logger = GameLogger()
    result = run_session(
        gm, players, gm.manager, observer, logger,
        num_players=num_players, seed=1, session_id="test-session",
    )
    return result, calls, logger


def test_session_reaches_winner_via_round_end(settings) -> None:
    manager = GameStateManager()
    gm = StubGM(2, manager, play_script=[{"eliminate": "player_2", "round_ended": True}])
    result, calls, logger = _run(gm, 2)

    assert result["winner"] == "player_1"
    assert calls == ["player_1"]  # the round (and game) ended on player_1's play
    assert gm.round_end_calls == 1
    types = [e["type"] for e in logger.log["events"]]
    for expected in ("game_start", "player_action", "gm_resolution", "round_end", "game_end"):
        assert expected in types
    # The driver fed the spec's score field into the round/game events.
    game_end = next(e for e in logger.log["events"] if e["type"] == "game_end")
    assert game_end["final_scores"]["player_1"] == 7


def test_session_advances_turns_then_ends_round(settings) -> None:
    manager = GameStateManager()
    # player_1's play is a no-op; player_2's play ends the round per the GM.
    gm = StubGM(2, manager, play_script=[{}, {"round_ended": True}])
    result, calls, _ = _run(gm, 2)

    assert result["winner"] == "player_1"
    assert calls == ["player_1", "player_2"]  # the driver advanced the turn


def test_session_game_ended_mid_round_skips_round_end(settings) -> None:
    manager = GameStateManager()
    gm = StubGM(
        2,
        manager,
        play_script=[{"game_ended": True, "round_ended": True, "winners": ["player_2"]}],
    )
    result, calls, _ = _run(gm, 2)

    assert result["winner"] == "player_2"  # GM-reported, mid-round
    assert gm.round_end_calls == 0  # game_ended wins over round_ended
    assert calls == ["player_1"]


def test_session_crashes_after_retry_exhaustion(settings) -> None:
    manager = GameStateManager()

    class RejectingGM(StubGM):
        def validate_and_resolve(self, action, player_id):
            self.validate_calls.append((player_id, action["action_type"]))
            return GMResolution(is_valid=False, error_message="not your turn", new_state=None)

    gm = RejectingGM(2, manager)
    players = {f"player_{i}": StubPlayer(f"player_{i}", []) for i in (1, 2)}
    observer = GameObserver(console=Console(record=True, width=100), verbose=False)
    with pytest.raises(IllegalAction):
        run_session(
            gm, players, manager, observer, GameLogger(),
            num_players=2, seed=1, session_id="t",
        )
    from playtest.config import get_settings

    assert len(gm.validate_calls) == get_settings().max_action_retries + 1


def test_session_log_saves_and_reloads(settings, tmp_path) -> None:
    manager = GameStateManager()
    gm = StubGM(2, manager, play_script=[{"eliminate": "player_2", "round_ended": True}])
    _, _, logger = _run(gm, 2)
    assert logger.log["winner"] == "player_1"

    path = tmp_path / "game.json"
    logger.save(str(path))
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["events"]


# --- Integration (real LLM + real ingested config) ----------------------------


@pytest.mark.integration
def test_real_game_reaches_a_winner() -> None:
    from playtest.config import get_settings
    from playtest.runner import run_game

    config_dir = Path(get_settings().game_configs_dir) / "love_letter_classic"
    if not config_dir.exists():
        pytest.skip("love_letter_classic config not present; run ingestion first")

    result = run_game("love_letter_classic", num_players=2, seed=1)
    summary = result["summary"]

    assert summary["winner"]
