"""Resolver tests: bounded rejection retry, logging/observer emission, integrity crashes."""

import pytest
from rich.console import Console

from playtest.agents.gm import GMResolution
from playtest.agents.player import PlayerAction
from playtest.config import get_settings
from playtest.errors import IllegalAction, PlaytestError
from playtest.rules import GameRules
from playtest.session import _make_resolver
from playtest.state.manager import GameStateManager
from playtest.ui.logger import GameLogger
from playtest.ui.observer import GameObserver

from .fixtures import sample_config


class _StubGM:
    """Returns a scripted GMResolution per call; tracks call count."""

    def __init__(self, resolutions: list[GMResolution], manager: GameStateManager) -> None:
        self._resolutions = list(resolutions)
        self.rules = GameRules(sample_config())
        self.state_manager = manager
        self.calls = 0

    def validate_and_resolve(self, action: dict, player_id: str) -> GMResolution:
        self.calls += 1
        if len(self._resolutions) > 1:
            return self._resolutions.pop(0)
        return self._resolutions[0]


def _manager() -> GameStateManager:
    manager = GameStateManager()
    rules = GameRules(sample_config())
    manager.initialize(rules.setup(2, seed=1), sample_config().game_spec.visibility)
    return manager


def _action(action_type: str = "draw_card") -> PlayerAction:
    return PlayerAction(
        player_id="player_1",
        action_type=action_type,
        parameters={},
        reasoning="r",
        public_statement="p",
    )


def _resolver(gm: _StubGM, manager: GameStateManager, observer=None, logger=None, last=None):
    return _make_resolver(
        gm,
        manager,
        observer or GameObserver(console=Console(record=True, width=100)),
        logger or GameLogger(),
        {"player_1": [], "player_2": []},
        "player_1",
        last if last is not None else {"narration": None, "round_ended": False,
                                       "game_ended": False, "winners": None},
    )


def test_resolver_rejection_logs_and_notifies(settings) -> None:
    manager = _manager()
    gm = _StubGM(
        [GMResolution(is_valid=False, error_message="not your turn", new_state=None)], manager
    )
    console = Console(record=True, width=100)
    observer = GameObserver(console=console)
    logger = GameLogger()
    resolve = _resolver(gm, manager, observer, logger)

    outcome = resolve(_action())

    assert outcome == {"rejected": True, "error_message": "not your turn"}
    events = [e for e in logger.log["events"] if e["type"] == "gm_validation"]
    assert len(events) == 1
    assert events[0]["is_valid"] is False
    assert events[0]["action_type"] == "draw_card"
    assert events[0]["error_message"] == "not your turn"
    out = console.export_text()
    assert "rejected" in out and "not your turn" in out


def test_resolver_retry_exhaustion_raises_illegal_action(settings) -> None:
    manager = _manager()
    gm = _StubGM(
        [GMResolution(is_valid=False, error_message="still illegal", new_state=None)], manager
    )
    logger = GameLogger()
    resolve = _resolver(gm, manager, logger=logger)
    cap = get_settings().max_action_retries

    for _ in range(cap):
        assert resolve(_action())["rejected"] is True
    with pytest.raises(IllegalAction, match="still illegal"):
        resolve(_action())

    assert gm.calls == cap + 1
    rejections = [
        e for e in logger.log["events"]
        if e["type"] == "gm_validation" and e["is_valid"] is False
    ]
    assert len(rejections) == cap + 1  # the crashing attempt is logged too


def test_resolver_valid_without_state_raises_playtest_error(settings) -> None:
    manager = _manager()
    gm = _StubGM([GMResolution(is_valid=True, narration="ok", new_state=None)], manager)
    resolve = _resolver(gm, manager)
    with pytest.raises(PlaytestError, match="without a committed state"):
        resolve(_action())


def test_resolver_success_records_flags_and_turn_end(settings) -> None:
    manager = _manager()
    committed = manager.get_state("gm")
    gm = _StubGM(
        [
            GMResolution(
                is_valid=True,
                narration="done",
                new_state=committed,
                round_ended=True,
                winners=["player_1"],
            )
        ],
        manager,
    )
    last = {"narration": None, "round_ended": False, "game_ended": False, "winners": None}
    resolve = _resolver(gm, manager, last=last)

    outcome = resolve(_action("play_guard"))

    assert outcome["turn_ended"] is True  # spec: play_guard ends the turn
    assert last["round_ended"] is True
    assert last["game_ended"] is False
    assert last["winners"] == ["player_1"]


def test_resolver_gm_turn_ended_overrides_spec(settings) -> None:
    manager = _manager()
    committed = manager.get_state("gm")
    gm = _StubGM(
        [GMResolution(is_valid=True, narration="done", new_state=committed, turn_ended=True)],
        manager,
    )
    resolve = _resolver(gm, manager)
    outcome = resolve(_action("draw_card"))  # spec says draw does NOT end the turn
    assert outcome["turn_ended"] is True
