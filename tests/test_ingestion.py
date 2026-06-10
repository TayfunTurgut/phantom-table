import pytest

from playtest.ingestion.analyzer import (
    _build_openai_tool,
    _validate_initial_state,
    _validate_state_schema,
)
from playtest.ingestion.chunker import _count_tokens, chunk_rulebook
from playtest.ingestion.schemas import ActionParamSpec, ActionSpec, GameConfig

SAMPLE_RULEBOOK = """Overview

This is a small game about delivering letters. Players take turns drawing and playing cards.

Card: Guard

When you play a Guard, choose another player and name a non-Guard card.
If they hold the named card, they are eliminated.

Card: Princess

If you play or discard the Princess, you are eliminated immediately.
"""

EXAMPLE_INITIAL_STATE = {
    "game_name": "Love Letter",
    "variant": "classic",
    "num_players": 2,
    "tokens_to_win": 7,
    "round_number": 1,
    "current_turn": "player_1",
    "turn_phase": "draw",
    "deck_count": 10,
    "removed_card": "HIDDEN",
    "revealed_cards": ["Guard", "Prince", "Handmaid"],
    "players": {
        "player_1": {
            "hand": ["King"],
            "hand_count": 1,
            "discards": [],
            "tokens": 0,
            "is_eliminated": False,
            "is_protected": False,
        },
        "player_2": {
            "hand": ["Guard"],
            "hand_count": 1,
            "discards": [],
            "tokens": 0,
            "is_eliminated": False,
            "is_protected": False,
        },
    },
}


def test_chunk_rulebook_attaches_headers_and_unique_ids() -> None:
    chunks = chunk_rulebook(SAMPLE_RULEBOOK, max_chunk_tokens=500, game_name="sample")

    assert chunks, "expected at least one chunk"
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"

    guard_chunks = [c for c in chunks if "Guard" in c["text"]]
    assert any(c["metadata"]["section"] == "Card: Guard" for c in guard_chunks)
    # Header is prepended to the chunk text as context.
    assert all(c["text"].startswith(c["metadata"]["section"]) for c in chunks)


def test_chunk_rulebook_respects_token_budget() -> None:
    long_block = "Overview\n\n" + " ".join(f"Sentence number {i} here." for i in range(400))
    chunks = chunk_rulebook(long_block, max_chunk_tokens=100, game_name="long")

    assert len(chunks) > 1, "a long block should be split into multiple chunks"
    # Body (chunk text minus the prepended header line) stays within budget.
    for chunk in chunks:
        body = chunk["text"].split("\n\n", 1)[1]
        assert _count_tokens(body) <= 100


def test_build_openai_tool_is_strict_mode() -> None:
    spec = ActionSpec(
        name="play_guard",
        description="Name a non-Guard card; if the target holds it, they are eliminated.",
        params=[
            ActionParamSpec(name="target_player", type="string", description="Whom to target."),
            ActionParamSpec(
                name="named_card",
                type="string",
                description="A non-Guard card.",
                enum=["Priest", "Baron"],
            ),
        ],
    )

    tool = _build_openai_tool(spec)
    fn = tool["function"]
    params = fn["parameters"]

    assert tool["type"] == "function"
    assert fn["strict"] is True
    assert params["additionalProperties"] is False
    # Every property is required.
    assert set(params["required"]) == set(params["properties"])
    # Mandatory agent fields are injected.
    assert "reasoning" in params["properties"]
    assert "public_statement" in params["properties"]
    assert params["properties"]["named_card"]["enum"] == ["Priest", "Baron"]


def test_build_openai_tool_optional_param_becomes_nullable() -> None:
    spec = ActionSpec(
        name="play_prince",
        description="Force a player to discard.",
        params=[
            ActionParamSpec(
                name="target_player", type="string", description="Whom to target.", required=False
            )
        ],
    )

    tool = _build_openai_tool(spec)
    params = tool["function"]["parameters"]
    assert params["properties"]["target_player"]["type"] == ["string", "null"]
    assert "target_player" in params["required"]


def test_validate_initial_state_accepts_example() -> None:
    _validate_initial_state(2)(EXAMPLE_INITIAL_STATE)


def test_validate_initial_state_rejects_wrong_deck_count() -> None:
    bad = {**EXAMPLE_INITIAL_STATE, "deck_count": 6}
    with pytest.raises(ValueError, match="deck_count"):
        _validate_initial_state(2)(bad)


def test_validate_initial_state_rejects_wrong_reveal_count() -> None:
    bad = {**EXAMPLE_INITIAL_STATE, "revealed_cards": ["Guard"]}
    with pytest.raises(ValueError, match="revealed_cards"):
        _validate_initial_state(2)(bad)


def test_validate_initial_state_rejects_missing_player_field() -> None:
    bad = {
        **EXAMPLE_INITIAL_STATE,
        "players": {
            "player_1": {"hand": ["King"]},
            "player_2": EXAMPLE_INITIAL_STATE["players"]["player_2"],
        },
    }
    with pytest.raises(ValueError, match="missing fields"):
        _validate_initial_state(2)(bad)


def test_validate_state_schema() -> None:
    _validate_state_schema(
        {
            "properties": {
                "players": {"type": "object"},
                "deck_count": {"type": "integer"},
                "current_turn": {"type": "string"},
                "turn_phase": {"type": "string"},
            }
        }
    )
    with pytest.raises(ValueError):
        _validate_state_schema({"properties": {"foo": {}}})


def test_game_config_save_load_roundtrip(tmp_path) -> None:
    config_dir = tmp_path / "love_letter_classic"
    config = GameConfig(
        game_name="Love Letter",
        variant="classic",
        num_players=2,
        config_dir=str(config_dir),
        state_schema={"properties": {"deck_count": {"type": "integer"}}},
        initial_state_template=EXAMPLE_INITIAL_STATE,
        tool_definitions={"draw_card": {"type": "function", "function": {"name": "draw_card"}}},
        gm_prompt="GM prompt text",
        player_prompt_template="You are {player_id}.",
        rulebook_text=SAMPLE_RULEBOOK,
        core_mechanics=["A protected player cannot be targeted."],
    )
    config.save()
    loaded = GameConfig.load(str(config_dir))

    assert loaded.game_name == "Love Letter"
    assert loaded.num_players == 2
    assert loaded.initial_state_template == EXAMPLE_INITIAL_STATE
    assert loaded.tool_definitions == config.tool_definitions
    assert "{player_id}" in loaded.player_prompt_template
    assert loaded.core_mechanics == ["A protected player cannot be targeted."]


def test_game_config_load_rejects_incomplete_config(tmp_path) -> None:
    config_dir = tmp_path / "love_letter_classic"
    config = GameConfig(
        game_name="Love Letter",
        variant="classic",
        num_players=2,
        config_dir=str(config_dir),
        state_schema={"properties": {"deck_count": {"type": "integer"}}},
        initial_state_template=EXAMPLE_INITIAL_STATE,
        tool_definitions={"draw_card": {"type": "function", "function": {"name": "draw_card"}}},
        gm_prompt="GM prompt text",
        player_prompt_template="You are {player_id}.",
        rulebook_text=SAMPLE_RULEBOOK,
    )
    config.save()
    (config_dir / "state_schema.json").unlink()

    with pytest.raises(FileNotFoundError, match="state_schema.json"):
        GameConfig.load(str(config_dir))


@pytest.mark.integration
def test_ingest_rulebook_end_to_end(openai_client, tmp_path, monkeypatch) -> None:
    """Full pipeline against the live API. Paid; deselected by default (run with -m integration)."""
    monkeypatch.setenv("GAME_CONFIGS_DIR", str(tmp_path))

    from playtest.ingestion.chunker import query_collection
    from playtest.ingestion.pipeline import ingest_rulebook

    rulebook = "src/playtest/data/love_letter_rules.txt"
    config = ingest_rulebook(rulebook, "love_letter_classic", num_players=2)

    assert "draw_card" in config.tool_definitions
    for schema in config.tool_definitions.values():
        params = schema["function"]["parameters"]
        assert schema["function"]["strict"] is True
        assert params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"])

    assert len(config.initial_state_template["revealed_cards"]) == 3
    assert config.initial_state_template["deck_count"] == 10
    assert "{player_id}" in config.player_prompt_template

    # Rules live in the rulebook (query_rulebook), not the prompts.
    assert "query_rulebook" in config.gm_prompt
    assert "## Game State Schema" in config.gm_prompt
    assert "## Complete Rules" not in config.gm_prompt
    assert "query_rulebook" in config.player_prompt_template
    # Leak guard: no formatted card definitions copied back into the prompt.
    assert "Guard (1)" not in config.gm_prompt
    assert "Priest (2)" not in config.gm_prompt
    assert set(config.setup_parameters) >= {
        "cards_removed",
        "cards_revealed_2p",
        "cards_revealed_other",
        "cards_dealt_per_player",
    }
    # Cross-cutting mechanics are code-injected into both prompts.
    assert len(config.core_mechanics) > 0
    assert "## Core Mechanics" in config.gm_prompt
    assert "## Core Mechanics" in config.player_prompt_template

    hits = query_collection(
        "what does the guard do",
        "love_letter_classic",
        f"{config.config_dir}/chromadb",
    )
    assert hits and any("Guard" in h for h in hits)
