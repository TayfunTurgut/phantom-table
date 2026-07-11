"""Ingestion pipeline tests with a stubbed LLM (no network).

The stub "model" returns a hand-written digest and a hand-written mini game
("Token Duel", a Nim variant) as if they had been generated — exercising the
real pipeline end to end: digest parsing, code extraction and import checking,
subprocess validation (generated pytest suite + contract harness), the repair
loop, and artifact persistence.
"""

import importlib.metadata
import json
from pathlib import Path

import pytest

from playtest.config import get_settings
from playtest.engine.loader import load_engine_from_path
from playtest.errors import PlaytestError
from playtest.ingestion import codegen, pipeline
from playtest.ingestion.codegen import check_engine_source, extract_python, prompts_fingerprint
from playtest.ingestion.digest import (
    _digest_json_schema,
    digest_to_markdown,
    digest_to_player_briefing,
    generate_digest,
)
from playtest.ingestion.pipeline import ingest_rulebook
from playtest.ingestion.schemas import GameArtifacts, GameDigest
from playtest.ingestion.validate import _clip

from .stubs import StubLLMClient

DIGEST = GameDigest(
    game_name="Token Duel",
    overview="Two players race to take the last token from a shared pool.",
    min_players=2,
    max_players=2,
    mechanics=[],
    components=[{"name": "Token", "count": 7}],
    zones="A single shared token pool: public, and conserved (tokens are removed, never added).",
    hidden_zones="Nothing is hidden; the pool is public.",
    setup="Place 7 tokens in a shared pool. player_1 goes first.",
    decision_flow="Players alternate turns. On your turn you take 1 or 2 tokens.",
    actions=[
        {"name": "take_one", "when": "On your turn.", "effect": "Remove 1 token from the pool."},
        {
            "name": "take_two",
            "when": "On your turn, if at least 2 tokens remain.",
            "effect": "Remove 2 tokens from the pool.",
        },
    ],
    end_conditions="The game ends when the pool is empty.",
    scoring="Whoever takes the last token wins. No ties are possible.",
    max_decisions=70,
    state_shape=(
        '{"num_players": int, "pool": int, "taken": {seat: int}, '
        '"current_player": str, "rng_seed": int, "game_over": bool, "winners": [str]}'
    ),
    ambiguities=[],
)

GOOD_ENGINE = '''
"""Token Duel — generated engine (test stub)."""

from __future__ import annotations

import copy

from playtest.engine import Action, Event, GameStatus, seats_for


class Game:
    game_name = "Token Duel"
    min_players = 2
    max_players = 2

    def setup(self, num_players: int, seed: int) -> tuple[dict, list[Event]]:
        state = {
            "num_players": num_players,
            "pool": 7,
            "taken": {seat: 0 for seat in seats_for(num_players)},
            "current_player": "player_1",
            "rng_seed": seed,
            "game_over": False,
            "winners": [],
        }
        return state, []

    def to_act(self, state: dict) -> list[str]:
        return [] if state["game_over"] else [state["current_player"]]

    def legal_actions(self, state: dict, seat: str) -> list[Action]:
        if state["game_over"] or seat != state["current_player"]:
            return []
        actions = [Action(seat=seat, name="take_one", args={"count": 1}, label="Take 1 token")]
        if state["pool"] >= 2:
            actions.append(
                Action(seat=seat, name="take_two", args={"count": 2}, label="Take 2 tokens")
            )
        return actions

    def apply(self, state: dict, actions: list[Action]) -> tuple[dict, list[Event]]:
        action = actions[0]
        legal = {a.key() for a in self.legal_actions(state, action.seat)}
        if [a.seat for a in actions] != self.to_act(state) or action.key() not in legal:
            raise ValueError(f"illegal action: {action}")
        state = copy.deepcopy(state)
        count = action.args["count"]
        state["pool"] -= count
        state["taken"][action.seat] += count
        events = [Event(f"{action.seat} took {count} token(s); {state['pool']} remain.")]
        if state["pool"] == 0:
            state["game_over"] = True
            state["winners"] = [action.seat]
            events.append(Event(f"{action.seat} took the last token and wins!"))
        else:
            seats = seats_for(state["num_players"])
            idx = seats.index(action.seat)
            state["current_player"] = seats[(idx + 1) % len(seats)]
        return state, events

    def observe(self, state: dict, seat: str) -> dict:
        return copy.deepcopy(state)

    def status(self, state: dict) -> GameStatus:
        return GameStatus(
            over=state["game_over"],
            winners=tuple(state["winners"]),
            scores={seat: float(n) for seat, n in state["taken"].items()},
        )
'''

GOOD_TESTS = """
from pathlib import Path

from playtest.engine.loader import load_engine_from_path

engine = load_engine_from_path(Path(__file__).parent / "engine.py")


def test_setup():
    state, _ = engine.setup(2, seed=1)
    assert state["pool"] == 7
    assert engine.to_act(state) == ["player_1"]


def test_taking_last_token_wins():
    state, _ = engine.setup(2, seed=1)
    while not engine.status(state).over:
        seat = engine.to_act(state)[0]
        state, _ = engine.apply(state, [engine.legal_actions(state, seat)[-1]])
    assert engine.status(state).winners
    total = sum(state["taken"].values())
    assert total == 7  # token conservation
"""

BROKEN_ENGINE = "def class oops(:\n"

# Fails against GOOD_ENGINE although the engine is correct (pool starts at 7).
BAD_TESTS = """
from pathlib import Path

from playtest.engine.loader import load_engine_from_path

engine = load_engine_from_path(Path(__file__).parent / "engine.py")


def test_wrong_assumption():
    state, _ = engine.setup(2, seed=1)
    assert state["pool"] == 6  # wrong on purpose: the pool starts at 7
"""

# Structurally broken: crashes mid-game, so the contract harness catches it.
HARNESS_BUG_ENGINE = GOOD_ENGINE.replace(
    '        count = action.args["count"]',
    '        if state["pool"] == 3:\n'
    '            raise RuntimeError("boom at pool 3")\n'
    '        count = action.args["count"]',
)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("GAME_CONFIGS_DIR", str(tmp_path / "game_configs"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def rulebook(tmp_path):
    path = tmp_path / "token_duel.txt"
    path.write_text("Token Duel: take 1 or 2 tokens; whoever takes the last token wins.")
    return str(path)


def _fenced(source: str) -> str:
    return f"Here is the module:\n\n```python\n{source}\n```"


def test_ingest_with_stub_llm_produces_playable_engine(rulebook):
    client = StubLLMClient([DIGEST.model_dump_json(), _fenced(GOOD_ENGINE), _fenced(GOOD_TESTS)])
    artifacts = ingest_rulebook(rulebook, "token_duel", client=client)

    assert artifacts.meta["engine_attempts"] == 1
    for name in (
        "engine.py",
        "test_engine.py",
        "digest.json",
        "digest.md",
        "player_briefing.txt",
        "rulebook.txt",
        "meta.json",
    ):
        assert (artifacts.config_dir / name).is_file(), name

    engine = load_engine_from_path(artifacts.engine_path)
    assert engine.game_name == "Token Duel"
    state, _ = engine.setup(2, seed=0)
    assert engine.status(state).over is False


def test_wrong_tests_trigger_test_only_repair(rulebook):
    """Harness passes but generated tests fail → only the tests are regenerated."""
    client = StubLLMClient(
        [
            DIGEST.model_dump_json(),
            _fenced(GOOD_ENGINE),
            _fenced(BAD_TESTS),  # fails although the engine is correct
            _fenced(GOOD_TESTS),  # test-only repair
        ]
    )
    artifacts = ingest_rulebook(rulebook, "token_duel", client=client)

    assert artifacts.meta["engine_attempts"] == 1  # the engine was never regenerated
    assert artifacts.meta["test_repairs"] == 1
    # The test-repair request carried the "tests may be wrong" routing hint.
    repair_request = client.calls[-1]["messages"][-1]["content"]
    assert "may themselves be wrong" in repair_request
    # Every validation round was archived.
    attempts_dir = artifacts.config_dir / "attempts"
    assert (attempts_dir / "01" / "failure.txt").read_text() != "OK"
    assert (attempts_dir / "02" / "failure.txt").read_text() == "OK"


def test_harness_failure_routes_to_engine_repair(rulebook):
    """A structurally broken engine is repaired with harness evidence."""
    client = StubLLMClient(
        [
            DIGEST.model_dump_json(),
            _fenced(HARNESS_BUG_ENGINE),  # crashes mid-game
            _fenced(GOOD_TESTS),
            _fenced(GOOD_ENGINE),  # engine repair
            _fenced(GOOD_TESTS),
        ]
    )
    artifacts = ingest_rulebook(rulebook, "token_duel", client=client)

    assert artifacts.meta["engine_attempts"] == 2
    # The engine-repair request carried the harness traceback, not test noise.
    repair_request = client.calls[3]["messages"][-1]["content"]
    assert "contract harness" in repair_request
    assert "boom at pool 3" in repair_request


def test_repair_loop_recovers_from_broken_codegen(rulebook):
    client = StubLLMClient(
        [
            DIGEST.model_dump_json(),
            _fenced(BROKEN_ENGINE),  # attempt 1: unparseable
            _fenced(GOOD_ENGINE),  # attempt 2: good
            _fenced(GOOD_TESTS),
        ]
    )
    artifacts = ingest_rulebook(rulebook, "token_duel", client=client)
    assert artifacts.meta["engine_attempts"] == 2


def test_ingest_rejects_unsafe_game_name(rulebook, tmp_path):
    # "../evil" would rmtree/create OUTSIDE game_configs_dir; reject before any FS work.
    client = StubLLMClient([])
    with pytest.raises(PlaytestError, match="game name"):
        ingest_rulebook(rulebook, "../evil", client=client)
    assert not (tmp_path / "evil").exists()
    assert client.calls == []


def test_ingestion_fails_loudly_after_budget(rulebook):
    client = StubLLMClient([DIGEST.model_dump_json()] + [_fenced(BROKEN_ENGINE)] * 4)
    with pytest.raises(PlaytestError, match="ingestion failed after 4 attempts"):
        ingest_rulebook(rulebook, "token_duel", max_attempts=4, client=client)
    # The digest survives for inspection.
    config_dir = Path(get_settings().game_configs_dir) / "token_duel"
    assert (config_dir / "digest.md").is_file()


def test_check_engine_source_rejects_disallowed_imports():
    with pytest.raises(PlaytestError, match="outside the allowlist"):
        check_engine_source("import os\n")
    with pytest.raises(PlaytestError, match="outside the allowlist"):
        check_engine_source("from subprocess import run\n")
    check_engine_source("import random\nfrom playtest.engine import Action\n")


def test_check_engine_source_rejects_syntax_errors():
    with pytest.raises(PlaytestError, match="does not parse"):
        check_engine_source(BROKEN_ENGINE)


def test_extract_python_prefers_largest_fence():
    content = "```python\nx = 1\n```\nand\n```python\ndef f():\n    return 2\n```"
    assert "def f" in extract_python(content)


def test_digest_schema_is_strict_compatible():
    """Strict structured outputs (`claude -p --json-schema`): every object must
    have additionalProperties=false and require every declared property; open-key
    dicts (typed additionalProperties) are rejected. The generation schema forces
    even the defaulted fields (mechanics/zones/max_decisions) into `required`."""

    def walk(node, path="$"):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                assert node.get("additionalProperties") is False, (
                    f"{path}: object allows additional/open-key properties"
                )
                props = set(node.get("properties", {}))
                assert set(node.get("required", [])) == props, (
                    f"{path}: required must list every property"
                )
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    schema = _digest_json_schema()
    walk(schema)
    # The new fields are present, required for generation, and the tag list is an enum.
    for field in ("mechanics", "zones", "max_decisions"):
        assert field in schema["required"], field
    mechanics_items = schema["properties"]["mechanics"]["items"]
    # $ref-indirected enum or inline enum, either way the tag vocabulary is closed.
    if "$ref" in mechanics_items:
        ref_name = mechanics_items["$ref"].rsplit("/", 1)[-1]
        mechanics_items = schema["$defs"][ref_name]
    assert "simultaneous_decisions" in mechanics_items["enum"]


def test_old_config_digest_json_loads_without_new_fields():
    """A digest.json generated before mechanics/zones/max_decisions existed still
    loads; the defaults fill in."""
    legacy = {
        "game_name": "Legacy Game",
        "overview": "An old digest with none of the new fields.",
        "min_players": 2,
        "max_players": 4,
        "components": [{"name": "Card", "count": 20}],
        "hidden_zones": "Hands are hidden.",
        "setup": "Deal 5 cards each.",
        "decision_flow": "Take turns.",
        "actions": [{"name": "play_card", "when": "On your turn.", "effect": "Play a card."}],
        "end_conditions": "Deck runs out.",
        "scoring": "Most points wins.",
        "state_shape": '{"rng_seed": int, "game_over": bool, "winners": [str]}',
        "ambiguities": [],
    }
    digest = GameDigest.model_validate_json(json.dumps(legacy))
    assert digest.mechanics == []
    assert digest.zones == ""
    assert digest.max_decisions == 0


def test_old_config_loads_through_artifacts_loader(rulebook):
    """GameArtifacts loads a config dir whose digest.json predates the new fields."""
    client = StubLLMClient([DIGEST.model_dump_json(), _fenced(GOOD_ENGINE), _fenced(GOOD_TESTS)])
    ingest_rulebook(rulebook, "token_duel", client=client)
    config_dir = Path(get_settings().game_configs_dir) / "token_duel"
    legacy = DIGEST.model_dump()
    for field in ("mechanics", "zones", "max_decisions"):
        legacy.pop(field)
    (config_dir / "digest.json").write_text(json.dumps(legacy), encoding="utf-8")

    artifacts = GameArtifacts(config_dir)
    assert artifacts.digest.mechanics == []
    assert artifacts.digest.zones == ""
    assert artifacts.digest.max_decisions == 0


def test_generate_digest_threads_feedback_into_the_llm_call():
    client = StubLLMClient([DIGEST.model_dump_json()])
    generate_digest(client, "RULES TEXT", feedback="Budget exhausted; stage the auction.")

    messages = client.calls[-1]["messages"]
    assert any("Budget exhausted; stage the auction." in m["content"] for m in messages)
    # Without feedback, no extra message is appended.
    client = StubLLMClient([DIGEST.model_dump_json()])
    generate_digest(client, "RULES TEXT")
    assert len(client.calls[-1]["messages"]) == 2


def test_digest_renderings_cover_all_sections():
    md = digest_to_markdown(DIGEST)
    for fragment in (
        "# Token Duel",
        "## Mechanics",
        "## Components",
        "Token × 7",
        "## Zones",
        "shared token pool",
        "**Decision budget:** 70",
        "## Actions",
        "`take_two`",
    ):
        assert fragment in md
    briefing = digest_to_player_briefing(DIGEST)
    assert "take_one" in briefing and "Winning:" in briefing


def test_artifacts_loader_round_trips(rulebook):
    client = StubLLMClient([DIGEST.model_dump_json(), _fenced(GOOD_ENGINE), _fenced(GOOD_TESTS)])
    ingest_rulebook(rulebook, "token_duel", client=client)
    config_dir = Path(get_settings().game_configs_dir) / "token_duel"
    artifacts = GameArtifacts(config_dir)
    assert artifacts.digest.game_name == "Token Duel"
    assert [c.model_dump() for c in artifacts.digest.components] == [{"name": "Token", "count": 7}]


def test_clip_keeps_head_and_tail_of_oversized_output():
    short = "a short failure"
    assert _clip(short) == short

    text = "HEAD-" + "x" * 10_000 + "-TAIL"
    clipped = _clip(text, max_chars=600)
    assert clipped.startswith("HEAD-")
    assert clipped.endswith("-TAIL")
    assert "chars omitted" in clipped
    assert len(clipped) < 700


def test_meta_json_has_provenance_fields(rulebook):
    client = StubLLMClient([DIGEST.model_dump_json(), _fenced(GOOD_ENGINE), _fenced(GOOD_TESTS)])
    artifacts = ingest_rulebook(rulebook, "token_duel", client=client)

    assert artifacts.meta["engine_attempts"] == 1
    assert artifacts.meta["games_per_count"] == get_settings().ingest_games_per_count
    assert artifacts.meta["prompt_fingerprint"] == prompts_fingerprint()
    assert artifacts.meta["playtest_version"] == importlib.metadata.version("playtest")
    assert artifacts.meta["mechanics"] == DIGEST.mechanics
    assert artifacts.meta["max_decisions"] == DIGEST.max_decisions


def test_prompt_fingerprint_changes_when_prompt_constant_changes(monkeypatch):
    before = prompts_fingerprint()
    monkeypatch.setattr(codegen, "_ENGINE_SYSTEM_PROMPT", codegen._ENGINE_SYSTEM_PROMPT + " ")
    after = prompts_fingerprint()

    assert before != after


def test_ingest_rulebook_explicit_budgets_are_respected(rulebook):
    """Explicit max_attempts/max_test_repairs still work regardless of settings."""
    client = StubLLMClient([DIGEST.model_dump_json()] + [_fenced(BROKEN_ENGINE)] * 2)
    with pytest.raises(PlaytestError, match="ingestion failed after 2 attempts"):
        ingest_rulebook(rulebook, "token_duel", max_attempts=2, client=client)


def test_ingest_rulebook_none_budgets_fall_back_to_settings(monkeypatch, rulebook):
    """max_attempts=None (the default) reads the budget from settings."""
    monkeypatch.setattr(get_settings(), "ingest_max_engine_attempts", 2)
    client = StubLLMClient([DIGEST.model_dump_json()] + [_fenced(BROKEN_ENGINE)] * 2)
    with pytest.raises(PlaytestError, match="ingestion failed after 2 attempts"):
        ingest_rulebook(rulebook, "token_duel", client=client)


def test_ingest_rulebook_threads_games_per_count_and_timeout(monkeypatch, rulebook):
    """validate_engine is called with the settings-derived games_per_count/timeout."""
    monkeypatch.setattr(get_settings(), "ingest_games_per_count", 5)
    monkeypatch.setattr(get_settings(), "ingest_validation_timeout_seconds", 42)
    seen: dict = {}
    real_validate_engine = pipeline.validate_engine

    def _spy_validate_engine(config_dir, **kwargs):
        seen.update(kwargs)
        return real_validate_engine(config_dir, **kwargs)

    monkeypatch.setattr(pipeline, "validate_engine", _spy_validate_engine)
    client = StubLLMClient([DIGEST.model_dump_json(), _fenced(GOOD_ENGINE), _fenced(GOOD_TESTS)])
    ingest_rulebook(rulebook, "token_duel", client=client)

    assert seen == {"games_per_count": 5, "timeout": 42}
