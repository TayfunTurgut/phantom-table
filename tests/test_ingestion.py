import copy

import pytest

import playtest.ingestion.analyzer as analyzer
import playtest.ingestion.pipeline as pipeline
from playtest.ingestion.analyzer import (
    _build_openai_tool,
    _validate_initial_state,
    _validate_state_schema,
    build_game_spec,
    generate_flow_spec,
    generate_setup_spec,
)
from playtest.ingestion.chunker import _count_tokens, chunk_rulebook
from playtest.ingestion.schemas import ActionParamSpec, ActionSpec, GameConfig

from .fixtures import TEMPLATE, TOOL_DEFINITIONS, sample_config, sample_spec

SAMPLE_RULEBOOK = """Overview

This is a small game about delivering letters. Players take turns drawing and playing cards.

Card: Guard

When you play a Guard, choose another player and name a non-Guard card.
If they hold the named card, they are eliminated.

Card: Princess

If you play or discard the Princess, you are eliminated immediately.
"""


# --- Chunking ----------------------------------------------------------------


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


# --- Tool building --------------------------------------------------------------


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


# --- State schema / initial state validators -------------------------------------


def test_validate_state_schema_requires_engine_contract() -> None:
    _validate_state_schema(
        {
            "properties": {
                "players": {"type": "object"},
                "current_turn": {"type": "string"},
                "turn_phase": {"type": "string"},
            }
        }
    )
    with pytest.raises(ValueError, match="turn_phase"):
        _validate_state_schema(
            {"properties": {"players": {}, "current_turn": {}}}
        )


def test_validate_initial_state_accepts_sample_template() -> None:
    schema = {"properties": {key: {} for key in TEMPLATE}}
    _validate_initial_state(2, schema)(TEMPLATE)


def test_validate_initial_state_rejects_engine_contract_violations() -> None:
    schema = {"properties": {key: {} for key in TEMPLATE}}
    validate = _validate_initial_state(2, schema)

    missing = {k: v for k, v in TEMPLATE.items() if k != "current_turn"}
    with pytest.raises(ValueError, match="engine fields"):
        validate(missing)

    bad_ids = copy.deepcopy(TEMPLATE)
    bad_ids["players"] = {"alice": bad_ids["players"]["player_1"]}
    with pytest.raises(ValueError, match="players"):
        validate(bad_ids)

    uneven = copy.deepcopy(TEMPLATE)
    del uneven["players"]["player_2"]["tokens"]
    with pytest.raises(ValueError, match="same set of fields"):
        validate(uneven)

    undocumented = copy.deepcopy(TEMPLATE)
    undocumented["mystery_field"] = 1
    with pytest.raises(ValueError, match="not documented"):
        validate(undocumented)


# --- Game spec validators (via a fake parse that just runs validation) -----------


def _fake_parse_with(response):
    def fake(messages, response_format, validate, max_repairs=1):
        validate(response)
        return response

    return fake


def _good_setup_response() -> analyzer._SetupSpecResponse:
    spec = sample_spec()
    return analyzer._SetupSpecResponse(
        supported_player_counts=[2],
        components=[
            analyzer._ComponentCount(name=n, count=c) for n, c in spec.components.items()
        ],
        component_zones=list(spec.component_zones),
        setup_plans=[
            analyzer._SetupPlanResponse(
                num_players=2,
                pool=[
                    analyzer._ComponentCount(name=n, count=c)
                    for n, c in spec.components.items()
                ],
                pool_field="deck",
                deal_steps=list(spec.setup_plans["2"].deal_steps),
                carry_over_fields=["players.*.tokens"],
            )
        ],
        visibility=analyzer._VisibilityResponse(
            per_player_private=["hand"],
            hidden_fields=["deck"],
            masked_fields=["removed_card"],
            count_fields=[
                analyzer._CountFieldPair(list_field="hand", count_field="hand_count"),
                analyzer._CountFieldPair(list_field="deck", count_field="deck_count"),
            ],
        ),
    )


def _good_flow_response() -> analyzer._FlowSpecResponse:
    return analyzer._FlowSpecResponse(
        phases=["draw", "play"],
        initial_phase="draw",
        inactive_field="is_eliminated",
        action_rules=[
            analyzer._ActionRuleResponse(
                action=name, phase="draw" if name == "draw_card" else "play",
                ends_turn=name != "draw_card",
            )
            for name in TOOL_DEFINITIONS
        ],
        has_rounds=True,
        end_conditions="Deck empty or one player left ends the round.",
        scoring="Highest card wins the round; first to 7 tokens wins.",
        score_field="tokens",
    )


def test_generate_setup_spec_accepts_consistent_response(monkeypatch) -> None:
    monkeypatch.setattr(analyzer, "_parse_with_repair", _fake_parse_with(_good_setup_response()))
    result = generate_setup_spec("rules", TEMPLATE, 2)
    assert result.supported_player_counts == [2]


def test_generate_setup_spec_rejects_inconsistencies(monkeypatch) -> None:
    bad = _good_setup_response()
    bad.component_zones.append("players.*.secret_stash")  # not a per-player field
    monkeypatch.setattr(analyzer, "_parse_with_repair", _fake_parse_with(bad))
    with pytest.raises(ValueError, match="secret_stash"):
        generate_setup_spec("rules", TEMPLATE, 2)

    overdraw = _good_setup_response()
    overdraw.setup_plans[0].deal_steps[0].count = 99  # consumes more than the pool holds
    monkeypatch.setattr(analyzer, "_parse_with_repair", _fake_parse_with(overdraw))
    with pytest.raises(ValueError, match="pool only has"):
        generate_setup_spec("rules", TEMPLATE, 2)

    uncovered = _good_setup_response()
    uncovered.supported_player_counts = [2, 3]  # no 3p plan provided
    monkeypatch.setattr(analyzer, "_parse_with_repair", _fake_parse_with(uncovered))
    with pytest.raises(ValueError, match="missing setup plans"):
        generate_setup_spec("rules", TEMPLATE, 2)


def test_generate_flow_spec_rejects_inconsistencies(monkeypatch) -> None:
    missing_rule = _good_flow_response()
    missing_rule.action_rules = missing_rule.action_rules[1:]
    monkeypatch.setattr(analyzer, "_parse_with_repair", _fake_parse_with(missing_rule))
    with pytest.raises(ValueError, match="missing action rules"):
        generate_flow_spec("rules", TOOL_DEFINITIONS, TEMPLATE)

    bad_phase = _good_flow_response()
    bad_phase.initial_phase = "upkeep"
    monkeypatch.setattr(analyzer, "_parse_with_repair", _fake_parse_with(bad_phase))
    with pytest.raises(ValueError, match="initial_phase"):
        generate_flow_spec("rules", TOOL_DEFINITIONS, TEMPLATE)

    bad_field = _good_flow_response()
    bad_field.score_field = "victory_points"
    monkeypatch.setattr(analyzer, "_parse_with_repair", _fake_parse_with(bad_field))
    with pytest.raises(ValueError, match="score_field"):
        generate_flow_spec("rules", TOOL_DEFINITIONS, TEMPLATE)


def test_build_game_spec_assembles_dict_keyed_spec() -> None:
    spec = build_game_spec(_good_setup_response(), _good_flow_response())
    assert spec.components["Guard"] == 5
    assert spec.setup_plans["2"].pool_field == "deck"
    assert spec.action_rules["draw_card"].ends_turn is False
    assert spec.visibility.count_fields == {"hand": "hand_count", "deck": "deck_count"}
    assert spec.turn.initial_phase == "draw"
    assert spec.score_field == "tokens"


# --- Config round-trip --------------------------------------------------------------


def test_game_config_save_load_roundtrip(tmp_path) -> None:
    config = sample_config(config_dir=str(tmp_path / "sample_letters"))
    config.save()
    loaded = GameConfig.load(config.config_dir)

    assert loaded.game_name == "Sample Letters"
    assert loaded.num_players == 2
    assert loaded.initial_state_template == config.initial_state_template
    assert loaded.tool_definitions == config.tool_definitions
    assert "{player_id}" in loaded.player_prompt_template
    assert loaded.game_spec == config.game_spec
    assert loaded.core_mechanics == config.core_mechanics


def test_game_config_load_rejects_incomplete_config(tmp_path) -> None:
    config = sample_config(config_dir=str(tmp_path / "sample_letters"))
    config.save()
    (tmp_path / "sample_letters" / "game_spec.json").unlink()

    with pytest.raises(FileNotFoundError, match="game_spec.json"):
        GameConfig.load(config.config_dir)


# --- Offline pipeline (everything monkeypatched in the pipeline namespace) ----------


def test_ingest_rulebook_offline_parallel_produces_complete_config(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("GAME_CONFIGS_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from playtest.config import get_settings

    get_settings.cache_clear()

    template = copy.deepcopy(TEMPLATE)
    spec = sample_spec()
    received: dict = {}

    monkeypatch.setattr(
        pipeline, "chunk_rulebook", lambda text, game_name: [{"id": "c1", "text": text}]
    )
    monkeypatch.setattr(pipeline, "embed_and_store", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline, "generate_state_schema", lambda text, n: {"properties": {k: {} for k in template}}
    )
    monkeypatch.setattr(pipeline, "generate_initial_state", lambda text, n, schema: template)
    monkeypatch.setattr(
        pipeline, "generate_tool_definitions", lambda text: copy.deepcopy(TOOL_DEFINITIONS)
    )
    monkeypatch.setattr(pipeline, "generate_game_overview", lambda text: "A tiny card game.")
    monkeypatch.setattr(pipeline, "generate_core_mechanics", lambda text: ["Be nice."])
    monkeypatch.setattr(pipeline, "generate_setup_spec", lambda text, tmpl, n: "SETUP")
    monkeypatch.setattr(
        pipeline, "generate_flow_spec", lambda text, tools, tmpl: "FLOW"
    )
    monkeypatch.setattr(pipeline, "build_game_spec", lambda s, f: spec)

    def fake_gm_prompt(text, schema, tools, overview, mechanics, end_conditions, scoring):
        received["gm"] = {"end_conditions": end_conditions, "scoring": scoring}
        return "GM PROMPT"

    def fake_player_prompt(overview, mechanics, phases, forbidden):
        received["player"] = {"phases": phases, "forbidden": forbidden}
        return "PLAYER PROMPT with {player_id}"

    monkeypatch.setattr(pipeline, "generate_gm_prompt", fake_gm_prompt)
    monkeypatch.setattr(pipeline, "generate_player_prompt", fake_player_prompt)

    rulebook = tmp_path / "rules.txt"
    rulebook.write_text(SAMPLE_RULEBOOK, encoding="utf-8")
    config = pipeline.ingest_rulebook(str(rulebook), "sample_letters", num_players=2)

    # The prompt generators received the spec-derived inputs (dependency order held).
    assert received["gm"]["end_conditions"] == spec.end_conditions
    assert received["player"]["phases"] == spec.turn.phases
    assert set(received["player"]["forbidden"]) == set(TOOL_DEFINITIONS)

    # The saved config round-trips from disk with the spec intact.
    loaded = GameConfig.load(str(tmp_path / "sample_letters"))
    assert loaded.game_spec == spec
    assert loaded.gm_prompt == "GM PROMPT"
    assert config.game_spec == spec

    get_settings.cache_clear()


@pytest.mark.integration
def test_ingest_rulebook_end_to_end(openai_client, tmp_path, monkeypatch) -> None:
    """Full pipeline against the live API. Paid; deselected by default (run with -m integration)."""
    monkeypatch.setenv("GAME_CONFIGS_DIR", str(tmp_path))
    from playtest.config import get_settings

    get_settings.cache_clear()

    from playtest.ingestion.chunker import query_collection
    from playtest.ingestion.pipeline import ingest_rulebook

    rulebook = "src/playtest/data/love_letter_rules.txt"
    config = ingest_rulebook(rulebook, "love_letter_classic", num_players=2)

    assert config.tool_definitions
    for schema in config.tool_definitions.values():
        params = schema["function"]["parameters"]
        assert schema["function"]["strict"] is True
        assert params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"])

    spec = config.game_spec
    assert 2 in spec.supported_player_counts
    assert str(2) in spec.setup_plans
    assert set(spec.action_rules) == set(config.tool_definitions)
    assert spec.turn.initial_phase in spec.turn.phases
    assert spec.end_conditions and spec.scoring

    assert "{player_id}" in config.player_prompt_template
    assert "query_rulebook" in config.gm_prompt
    assert "## End Conditions" in config.gm_prompt
    assert "## Scoring" in config.gm_prompt
    assert "## Core Mechanics" in config.gm_prompt

    hits = query_collection(
        "what does the guard do",
        "love_letter_classic",
        f"{config.config_dir}/chromadb",
    )
    assert hits and any("Guard" in h for h in hits)
    get_settings.cache_clear()
