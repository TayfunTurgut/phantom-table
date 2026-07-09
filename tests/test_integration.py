"""End-to-end: run_game over the reference engine with a stubbed LLM client."""

import json

import pytest

from playtest.config import get_settings
from playtest.llm import ROLES, LLMClient


class FirstLegalClient(LLMClient):
    """A 'model' that always picks action 0 — enough to drive full games."""

    def __init__(self):
        self.models = {role: "stub-model" for role in ROLES}

    def complete(self, messages, *, role, json_schema=None):
        return json.dumps(
            {
                "action_index": 0,
                "reasoning": "first legal",
                "table_talk": None,
                "notes": "always picking the first action",
            }
        )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setattr("playtest.runner.create_llm_client", lambda settings: FirstLegalClient())
    yield
    get_settings.cache_clear()


def test_run_game_end_to_end(tmp_path):
    from playtest.runner import run_game

    log_file = tmp_path / "game.json"
    result = run_game(
        "playtest.games.love_letter",
        num_players=2,
        seed=123,
        log_file=str(log_file),
        archetypes=["aggressive", "cautious"],
    )
    summary = result["summary"]
    assert summary["winners"]
    assert summary["total_steps"] > 0
    assert summary["player_confusions"]["count"] == 0

    saved = json.loads(log_file.read_text())
    assert saved["winners"] == summary["winners"]
    assert saved["archetypes"] == ["aggressive", "cautious"]
    assert any(e["type"] == "decision" for e in saved["events"])


def test_run_game_is_reproducible_for_a_seed(tmp_path):
    from playtest.runner import run_game

    a = run_game("playtest.games.love_letter", num_players=2, seed=77)
    b = run_game("playtest.games.love_letter", num_players=2, seed=77)
    assert a["final_state"] == b["final_state"]


def test_run_multiple_games_aggregates(tmp_path):
    from playtest.runner import run_multiple_games

    analytics = run_multiple_games(
        "playtest.games.love_letter",
        num_games=3,
        num_players=2,
        seed_start=10,
        output_dir=str(tmp_path),
    )
    assert analytics["games_played"] == 3
    assert sum(analytics["win_rates"].values()) == pytest.approx(1.0)
