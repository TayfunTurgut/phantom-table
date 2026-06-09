import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from rich.console import Console

from playtest.agents.gm import GMResolution
from playtest.agents.player import PlayerAction
from playtest.graph.build import assemble_graph
from playtest.graph.nodes import build_gm_node, build_player_node, next_active_player
from playtest.runner import _play
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


@dataclass
class Step:
    """One scripted outcome for StubGM.validate_and_resolve / handle_round_end."""

    valid: bool = True
    next_player: str | None = None
    eliminate: str | None = None
    round_ended: bool = False
    game_ended: bool = False
    winner: str | None = None
    error: str | None = None
    phase: str = "draw"
    set_turn: str | None = None


class StubGM:
    """Deterministic GM that drives a scripted mini-game against a real manager."""

    def __init__(self, num_players: int, steps: list[Step], round_steps: list[Step] | None = None):
        self.num_players = num_players
        self.manager = GameStateManager()
        self._init = _init_state(num_players, tokens_to_win=2)
        self._steps = list(steps)
        self._round_steps = list(round_steps or [])
        self.validate_calls: list[str] = []
        self.state: dict = {}

    def initialize_game(self, num_players: int | None = None, seed: int | None = None) -> dict:
        gm_view = self.manager.initialize(
            initial_state=copy.deepcopy(self._init),
            deck_cards=["Guard"] * 5,
            removed_card="Princess",
            revealed_cards=[],
            player_hands={f"player_{i}": ["Guard"] for i in range(1, self.num_players + 1)},
        )
        self.state = copy.deepcopy(self.manager.get_state("gm"))
        return {"state": gm_view, "narration": "A new game begins."}

    def _apply(self, step: Step) -> GMResolution:
        if step.eliminate:
            self.state["players"][step.eliminate]["is_eliminated"] = True
        self.state["current_turn"] = step.set_turn or step.next_player or self.state["current_turn"]
        self.state["turn_phase"] = step.phase
        new_state = self.manager.set_state(self.state)
        self.state = copy.deepcopy(new_state)
        return GMResolution(
            is_valid=True,
            narration="The action resolves.",
            new_state=new_state,
            round_ended=step.round_ended,
            game_ended=step.game_ended,
            winner=step.winner,
            next_player=step.next_player,
            next_phase=step.phase,
        )

    def validate_and_resolve(self, proposed_action: dict, player_id: str) -> GMResolution:
        self.validate_calls.append(player_id)
        step = self._steps.pop(0)
        if not step.valid:
            return GMResolution(
                is_valid=False,
                error_message=step.error or "illegal move",
                narration="The GM rejects the move.",
                next_player=step.next_player,
            )
        return self._apply(step)

    def handle_round_end(self) -> GMResolution:
        resolution = self._apply(self._round_steps.pop(0))
        resolution.round_ended = True
        return resolution


class StubPlayer:
    def __init__(self, player_id: str, calls: list[str]):
        self.player_id = player_id
        self._calls = calls

    def take_turn(self, game_context: str | None = None) -> PlayerAction:
        self._calls.append(self.player_id)
        return PlayerAction(
            player_id=self.player_id,
            action_type="draw_card",
            parameters={},
            reasoning="stub reasoning",
            public_statement="I take my turn.",
        )


def _initial_graph_state(num_players: int) -> dict:
    return {
        "game_config_id": "stub",
        "session_id": "test-session",
        "game_state": {},
        "current_player": "player_1",
        "turn_phase": "draw",
        "turn_index": 0,
        "proposed_action": None,
        "pending_context": None,
        "public_transcript": [],
        "gm_log": [],
        "error_log": [],
        "phase": "initializing",
        "winner": None,
        "retry_count": 0,
    }


def _run(gm: StubGM, num_players: int) -> tuple[dict, list[str]]:
    calls: list[str] = []
    players = {f"player_{i}": StubPlayer(f"player_{i}", calls) for i in range(1, num_players + 1)}
    gm_node = build_gm_node(gm, gm.manager, num_players=num_players, seed=1)
    player_node = build_player_node(players)
    graph = assemble_graph(gm_node, player_node)
    final = graph.invoke(_initial_graph_state(num_players), config={"recursion_limit": 500})
    return final, calls


def test_full_game_routing_terminates_with_winner() -> None:
    # player_1 retry, eliminate player_2 and skip it, a round end, then game over.
    steps = [
        Step(valid=False, error="not your phase"),  # player_1 rejected
        Step(valid=True, eliminate="player_2", next_player="player_3"),  # retry ok, skip p2
        Step(valid=True, next_player="player_1"),  # player_3 acts
        Step(valid=True, round_ended=True, next_player="player_1"),  # player_1 ends round
        Step(valid=True, game_ended=True, winner="player_1"),  # player_1 wins game
    ]
    round_steps = [Step(valid=True, next_player="player_1")]  # new round dealt
    gm = StubGM(3, steps, round_steps)

    final, calls = _run(gm, 3)

    assert final["phase"] == "game_over"
    assert final["winner"] == "player_1"
    # Eliminated player is never asked to take a turn again.
    assert "player_2" not in calls
    # Retry produced an error-log entry, transcript and gm_log captured every step.
    assert len(final["error_log"]) >= 1
    assert len(final["public_transcript"]) >= len(calls)
    assert len(final["gm_log"]) >= 1


def test_exhausted_retries_skips_turn_via_fallback() -> None:
    # player_1 fails MAX_RETRIES+1 times; the final rejection has next_player=None,
    # so the gm_node must fall back to next_active_player -> player_2.
    steps = [
        Step(valid=False, error="bad"),
        Step(valid=False, error="bad"),
        Step(valid=False, error="bad"),
        Step(valid=False, error="bad", next_player=None),  # exhausted -> skip
        Step(valid=True, game_ended=True, winner="player_2"),
    ]
    gm = StubGM(2, steps)

    final, calls = _run(gm, 2)

    assert final["phase"] == "game_over"
    assert final["winner"] == "player_2"
    assert "player_2" in calls  # turn was skipped to player_2
    assert len(final["error_log"]) >= 1


def test_next_active_player_skips_eliminated() -> None:
    manager = GameStateManager()
    state = _init_state(3)
    state["players"]["player_2"]["is_eliminated"] = True
    manager.initialize(
        initial_state=state,
        deck_cards=["Guard"] * 5,
        removed_card="Princess",
        revealed_cards=[],
        player_hands={f"player_{i}": ["Guard"] for i in range(1, 4)},
    )
    assert next_active_player(manager, "player_1") == "player_3"  # skips eliminated player_2
    assert next_active_player(manager, "player_3") == "player_1"  # wraps around


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


def test_next_active_player_all_but_one_eliminated() -> None:
    # 4 players, only player_1 survives: always returns player_1 regardless of who asks.
    manager = _manager_with(4, {"player_2", "player_3", "player_4"})
    assert next_active_player(manager, "player_1") == "player_1"
    assert next_active_player(manager, "player_3") == "player_1"


def test_next_active_player_skips_current_if_eliminated() -> None:
    # If the current player is itself eliminated (e.g. Prince->Princess self-out), skip it.
    manager = _manager_with(4, {"player_2"})
    assert next_active_player(manager, "player_2") == "player_3"


def test_play_drives_observer_and_logger_offline(tmp_path) -> None:
    # Full routing through the runner's _play loop against stub-produced deltas.
    steps = [
        Step(valid=False, error="not your phase"),
        Step(valid=True, eliminate="player_2", next_player="player_3"),
        Step(valid=True, next_player="player_1"),
        # Round ends by eliminating the last opponent (only player_1 survives), which is
        # what gm_node now requires to honor round_ended.
        Step(valid=True, round_ended=True, eliminate="player_3", next_player="player_1"),
        Step(valid=True, game_ended=True, winner="player_1"),
    ]
    gm = StubGM(3, steps, [Step(valid=True, next_player="player_1")])
    players = {f"player_{i}": StubPlayer(f"player_{i}", []) for i in range(1, 4)}
    graph = assemble_graph(
        build_gm_node(gm, gm.manager, num_players=3, seed=7),
        build_player_node(players),
    )

    console = Console(record=True, width=100)
    observer = GameObserver(console=console, verbose=False)
    logger = GameLogger()
    final = _play(graph, _initial_graph_state(3), observer, logger, seed=7)

    assert final["phase"] == "game_over"
    assert final["winner"] == "player_1"

    types = [e["type"] for e in logger.log["events"]]
    for expected in ("game_start", "player_action", "gm_resolution", "round_end", "game_end"):
        assert expected in types
    assert logger.log["winner"] == "player_1"

    # Saved log reloads and summarizes.
    path = tmp_path / "game.json"
    logger.save(str(path))
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["events"]

    out = console.export_text()
    assert "player_1" in out  # color-coded output rendered


@pytest.mark.integration
def test_real_game_reaches_a_winner() -> None:
    from playtest.config import get_settings
    from playtest.runner import run_game

    config_dir = Path(get_settings().game_configs_dir) / "love_letter_classic"
    if not config_dir.exists():
        pytest.skip("love_letter_classic config not present; run ingestion first")

    result = run_game("love_letter_classic", num_players=2, seed=1)
    final = result["final_state"]

    assert final["winner"] is not None
    winner_tokens = final["game_state"]["players"][final["winner"]]["tokens"]
    assert winner_tokens >= final["game_state"]["tokens_to_win"]
    assert len(final["public_transcript"]) > 0
