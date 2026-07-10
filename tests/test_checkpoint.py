"""Checkpoint round-trip and crash -> resume behavior."""

import json

import pytest

from playtest.checkpoint import Checkpoint, load_checkpoint, write_checkpoint
from playtest.config import get_settings
from playtest.errors import PlaytestError
from playtest.llm import ROLES, LLMClient


class FirstLegalClient(LLMClient):
    """A 'model' that always picks action 0 — deterministic, drives full games."""

    def __init__(self):
        self.models = {role: "stub-model" for role in ROLES}

    def complete(self, messages, *, role, json_schema=None):
        return json.dumps(
            {
                "action_index": 0,
                "reasoning": "first legal",
                "table_talk": None,
                "notes": "notebook contents",
            }
        )


class CrashAtCallClient(FirstLegalClient):
    """Succeeds for the first ``crash_at - 1`` calls, then always raises.

    Because a failing turn is retried, the failing step exhausts its attempts and
    the PlaytestError propagates out of the session — exactly the runtime crash we
    want a checkpoint for.
    """

    def __init__(self, crash_at):
        super().__init__()
        self.calls = 0
        self.crash_at = crash_at

    def complete(self, messages, *, role, json_schema=None):
        self.calls += 1
        if self.calls >= self.crash_at:
            raise PlaytestError("simulated structured-output failure")
        return super().complete(messages, role=role, json_schema=json_schema)


@pytest.fixture(autouse=True)
def _clear_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_checkpoint_round_trip(tmp_path):
    cp = Checkpoint(
        game_ref="playtest.games.love_letter",
        num_players=2,
        seed=7,
        archetypes=["default", "default"],
        session_id="s",
        step=3,
        state={"rng_seed": 42, "foo": [1, 2]},
        buffers={"player_1": ["hi"], "player_2": []},
        notebooks={"player_1": "notes", "player_2": ""},
    )
    path = tmp_path / "ck.json"
    write_checkpoint(str(path), cp)

    assert not (tmp_path / "ck.json.tmp").exists()  # atomic write leaves no temp file
    assert load_checkpoint(str(path)) == cp


def test_crash_writes_checkpoint_then_resume_completes_identically(tmp_path, monkeypatch):
    from playtest import runner

    monkeypatch.setattr(runner, "create_llm_client", lambda settings: FirstLegalClient())
    baseline = runner.run_game(
        "playtest.games.love_letter",
        num_players=2,
        seed=123,
        checkpoint_path=str(tmp_path / "baseline.json"),
    )

    crash_path = tmp_path / "crash.json"
    monkeypatch.setattr(runner, "create_llm_client", lambda settings: CrashAtCallClient(3))
    with pytest.raises(PlaytestError):
        runner.run_game(
            "playtest.games.love_letter",
            num_players=2,
            seed=123,
            checkpoint_path=str(crash_path),
        )

    cp = load_checkpoint(str(crash_path))
    assert cp.step == 3  # checkpoint is written at the top of the failing turn
    assert cp.game_ref == "playtest.games.love_letter"
    assert cp.seed == 123
    assert set(cp.buffers) == {"player_1", "player_2"}
    assert any(cp.notebooks.values())  # a seat wrote a notebook before the crash

    monkeypatch.setattr(runner, "create_llm_client", lambda settings: FirstLegalClient())
    resumed = runner.resume_game(str(crash_path))

    # Deterministic players + reconstructed state -> identical final position.
    assert resumed["final_state"] == baseline["final_state"]
