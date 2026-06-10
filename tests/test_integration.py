import copy
import json
from pathlib import Path

import pytest
from rich.console import Console

from playtest.agents.gm import GMResolution
from playtest.agents.player import PlayerAction
from playtest.errors import IllegalAction
from playtest.rules.love_letter import LoveLetterRules
from playtest.session import run_session
from playtest.state.manager import GameStateManager
from playtest.ui.logger import GameLogger
from playtest.ui.observer import GameObserver


def _init_state(num_players: int, tokens_to_win: int = 7) -> dict:
    players = {
        f"player_{i}": {
            "hand": [],
            "hand_count": 1,
            "discards": [],
            "tokens": 0,
            "is_eliminated": False,
            "is_protected": False,
        }
        for i in range(1, num_players + 1)
    }
    return {
        "game_name": "Love Letter",
        "variant": "classic",
        "num_players": num_players,
        "tokens_to_win": tokens_to_win,
        "round_number": 1,
        "current_turn": "player_1",
        "turn_phase": "draw",
        "deck_count": 0,
        "removed_card": "HIDDEN",
        "revealed_cards": [],
        "players": players,
    }


# --- Stubs that drive the plain turn loop offline (no API calls) -------------


class StubGM:
    """Deterministic GM: resolves draw/play against a real manager, no LLM, no invariants.

    ``play_script`` is a list of per-play directives (popped in order) that apply side
    effects to the committed state — e.g. ``{"eliminate": "player_2"}`` or
    ``{"empty_deck": True}`` — so a test can steer when a round ends.
    """

    def __init__(self, num_players: int, manager: GameStateManager, play_script=None):
        self.num_players = num_players
        self.manager = manager
        self.rules = LoveLetterRules()  # the driver delegates turn-flow/round-over here
        self._init = _init_state(num_players, tokens_to_win=2)
        self._play_script = list(play_script or [])
        self.validate_calls: list[tuple[str, str]] = []
        self.round_end_calls = 0

    def initialize_game(self, num_players: int | None = None, seed: int | None = None) -> dict:
        gm_view = self.manager.initialize(
            initial_state=copy.deepcopy(self._init),
            deck_cards=["Guard"] * 5,
            removed_card="Princess",
            revealed_cards=[],
            player_hands={f"player_{i}": ["Guard"] for i in range(1, self.num_players + 1)},
        )
        return {"state": gm_view, "narration": "A new game begins."}

    def validate_and_resolve(self, action: dict, player_id: str) -> GMResolution:
        self.validate_calls.append((player_id, action["action_type"]))
        state = copy.deepcopy(self.manager.get_state("gm"))
        if action["action_type"] == "draw_card":
            state["players"][player_id]["hand"] = state["players"][player_id]["hand"] + ["Guard"]
            state["players"][player_id]["hand_count"] = 2
            state["turn_phase"] = "play"
        else:  # a play ends the turn
            state["players"][player_id]["hand"] = ["Guard"]
            state["players"][player_id]["hand_count"] = 1
            directive = self._play_script.pop(0) if self._play_script else {}
            if directive.get("eliminate"):
                tgt = directive["eliminate"]
                state["players"][tgt]["is_eliminated"] = True
                state["players"][tgt]["hand"] = []
                state["players"][tgt]["hand_count"] = 0
            if directive.get("empty_deck"):
                state["deck"] = []
                state["deck_count"] = 0
        new_state = self.manager.set_state(state)
        return GMResolution(
            is_valid=True,
            narration="The action resolves.",
            new_state=new_state,
            next_phase=state["turn_phase"],
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
            winning_card="Guard",
        )


class StubPlayer:
    def __init__(self, player_id: str, calls: list[str]):
        self.player_id = player_id
        self._calls = calls

    def take_turn(self, filtered_state, context=None, private_memory=None, *, resolve_action):
        self._calls.append(self.player_id)
        resolve_action(
            PlayerAction(
                player_id=self.player_id,
                action_type="draw_card",
                parameters={},
                reasoning="r",
                public_statement="I draw.",
            )
        )
        return resolve_action(
            PlayerAction(
                player_id=self.player_id,
                action_type="play_guard",
                parameters={},
                reasoning="r",
                public_statement="I play.",
            )
        )


def _run(gm: StubGM, num_players: int) -> tuple[dict, list[str]]:
    calls: list[str] = []
    players = {f"player_{i}": StubPlayer(f"player_{i}", calls) for i in range(1, num_players + 1)}
    observer = GameObserver(console=Console(record=True, width=100), verbose=False)
    logger = GameLogger()
    result = run_session(
        gm, players, gm.manager, observer, logger,
        num_players=num_players, seed=1, session_id="test-session",
    )
    return result, calls, logger


def test_session_reaches_winner_on_elimination() -> None:
    manager = GameStateManager()
    gm = StubGM(2, manager, play_script=[{"eliminate": "player_2"}])
    result, calls, logger = _run(gm, 2)

    assert result["winner"] == "player_1"
    assert calls == ["player_1"]  # player_2 was eliminated before ever taking a turn
    assert gm.round_end_calls == 1
    types = [e["type"] for e in logger.log["events"]]
    for expected in ("game_start", "player_action", "gm_resolution", "round_end", "game_end"):
        assert expected in types


def test_session_advances_turns_then_ends_on_empty_deck() -> None:
    manager = GameStateManager()
    # player_1's play is a no-op; player_2's play empties the deck, ending the round.
    gm = StubGM(2, manager, play_script=[{}, {"empty_deck": True}])
    result, calls, _ = _run(gm, 2)

    assert result["winner"] == "player_1"
    assert calls == ["player_1", "player_2"]  # the driver advanced the turn


def test_session_crashes_on_illegal_action() -> None:
    manager = GameStateManager()

    class CrashGM(StubGM):
        def validate_and_resolve(self, action, player_id):
            raise IllegalAction(player_id, action, "not your turn")

    gm = CrashGM(2, manager)
    players = {f"player_{i}": StubPlayer(f"player_{i}", []) for i in (1, 2)}
    observer = GameObserver(console=Console(record=True, width=100), verbose=False)
    with pytest.raises(IllegalAction):
        run_session(
            gm, players, manager, observer, GameLogger(),
            num_players=2, seed=1, session_id="t",
        )


def test_session_log_saves_and_reloads(tmp_path) -> None:
    manager = GameStateManager()
    gm = StubGM(2, manager, play_script=[{"eliminate": "player_2"}])
    _, _, logger = _run(gm, 2)
    assert logger.log["winner"] == "player_1"

    path = tmp_path / "game.json"
    logger.save(str(path))
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["events"]


# --- next-active-player helper ----------------------------------------------


def _manager_with(num_players: int, eliminated: set[str]) -> GameStateManager:
    manager = GameStateManager()
    state = _init_state(num_players)
    for pid in eliminated:
        state["players"][pid]["is_eliminated"] = True
    manager.initialize(
        initial_state=state,
        deck_cards=["Guard"] * 5,
        removed_card="Princess",
        revealed_cards=[],
        player_hands={f"player_{i}": ["Guard"] for i in range(1, num_players + 1)},
    )
    return manager


def _next(manager: GameStateManager, current: str) -> str:
    return LoveLetterRules().advance_turn(manager.get_state("gm"), current)["current_turn"]


def test_next_active_player_skips_eliminated() -> None:
    manager = _manager_with(3, {"player_2"})
    assert _next(manager, "player_1") == "player_3"
    assert _next(manager, "player_3") == "player_1"  # wraps around


def test_next_active_player_all_but_one_eliminated() -> None:
    manager = _manager_with(4, {"player_2", "player_3", "player_4"})
    assert _next(manager, "player_1") == "player_1"
    assert _next(manager, "player_3") == "player_1"


def test_next_active_player_skips_current_if_eliminated() -> None:
    manager = _manager_with(4, {"player_2"})
    assert _next(manager, "player_2") == "player_3"


@pytest.mark.integration
def test_real_game_reaches_a_winner() -> None:
    from playtest.config import get_settings
    from playtest.runner import run_game

    config_dir = Path(get_settings().game_configs_dir) / "love_letter_classic"
    if not config_dir.exists():
        pytest.skip("love_letter_classic config not present; run ingestion first")

    result = run_game("love_letter_classic", num_players=2, seed=1)
    summary = result["summary"]
    final_state = result["final_state"]

    assert summary["winner"]
    winner_tokens = final_state["players"][summary["winner"]]["tokens"]
    assert winner_tokens >= final_state["tokens_to_win"]
