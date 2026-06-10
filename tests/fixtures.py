"""Shared sample-game fixtures: the *data* an ingestion run would emit, hand-built.

The runtime contains zero game logic, so tests exercise the generic machinery against a
small fabricated card game ("Sample Letters"). Everything here is config — GameSpec,
tool definitions, a state template — exactly the kind of artifact ingestion produces.
"""

import copy

from playtest.ingestion.schemas import (
    ActionRule,
    DealStep,
    GameConfig,
    GameSpec,
    SetupPlan,
    TurnStructure,
    VisibilitySpec,
)

# 16 components; 2p deal: 1 removed + 3 revealed + 2 dealt -> 10 left in the deck.
COMPONENTS = {
    "Guard": 5,
    "Priest": 2,
    "Baron": 2,
    "Handmaid": 2,
    "Prince": 2,
    "King": 1,
    "Countess": 1,
    "Princess": 1,
}

TEMPLATE = {
    "game_name": "Sample Letters",
    "variant": "classic",
    "num_players": 2,
    "tokens_to_win": 7,
    "round_number": 1,
    "current_turn": "player_1",
    "turn_phase": "draw",
    "deck": [],
    "deck_count": 0,
    "removed_card": "HIDDEN",
    "revealed_cards": [],
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


def _action_tool(name: str, params: dict[str, dict] | None = None) -> dict:
    properties: dict = dict(params or {})
    properties["reasoning"] = {"type": "string"}
    properties["public_statement"] = {"type": "string"}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} action.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_DEFINITIONS = {
    "draw_card": _action_tool("draw_card"),
    "play_guard": _action_tool(
        "play_guard",
        {"target_player": {"type": "string"}, "named_card": {"type": "string"}},
    ),
    "play_prince": _action_tool("play_prince", {"target_player": {"type": "string"}}),
    "play_king": _action_tool("play_king", {"target_player": {"type": "string"}}),
    "play_countess": _action_tool("play_countess"),
}


def sample_spec() -> GameSpec:
    plan_2p = SetupPlan(
        pool=dict(COMPONENTS),
        pool_field="deck",
        deal_steps=[
            DealStep(count=1, target="set_aside", to_field="removed_card"),
            DealStep(count=3, target="reveal", to_field="revealed_cards"),
            DealStep(count=1, target="each_player", to_field="hand"),
        ],
        carry_over_fields=["players.*.tokens"],
    )
    plan_multi = SetupPlan(
        pool=dict(COMPONENTS),
        pool_field="deck",
        deal_steps=[
            DealStep(count=1, target="set_aside", to_field="removed_card"),
            DealStep(count=1, target="each_player", to_field="hand"),
        ],
        carry_over_fields=["players.*.tokens"],
    )
    return GameSpec(
        supported_player_counts=[2, 3, 4],
        components=dict(COMPONENTS),
        component_zones=[
            "deck",
            "removed_card",
            "revealed_cards",
            "players.*.hand",
            "players.*.discards",
        ],
        setup_plans={"2": plan_2p, "3": plan_multi, "4": plan_multi},
        turn=TurnStructure(
            phases=["draw", "play"], initial_phase="draw", inactive_field="is_eliminated"
        ),
        action_rules={
            "draw_card": ActionRule(phase="draw", ends_turn=False),
            "play_guard": ActionRule(phase="play", ends_turn=True),
            "play_prince": ActionRule(phase="play", ends_turn=True),
            "play_king": ActionRule(phase="play", ends_turn=True),
            "play_countess": ActionRule(phase="play", ends_turn=True),
        },
        has_rounds=True,
        end_conditions=(
            "A round ends when the deck is empty or only one player remains active. "
            "The game ends when a player reaches tokens_to_win tokens."
        ),
        scoring=(
            "The surviving player holding the highest card wins the round and gains one "
            "token; ties share the win."
        ),
        score_field="tokens",
        visibility=VisibilitySpec(
            per_player_private=["hand"],
            hidden_fields=["deck"],
            masked_fields=["removed_card"],
            count_fields={"hand": "hand_count", "deck": "deck_count"},
        ),
    )


def sample_config(config_dir: str = "/tmp/sample-letters", num_players: int = 2) -> GameConfig:
    return GameConfig(
        game_name="Sample Letters",
        variant="classic",
        num_players=num_players,
        config_dir=config_dir,
        state_schema={
            "type": "object",
            "properties": {key: {"type": "string"} for key in TEMPLATE},
        },
        initial_state_template=copy.deepcopy(TEMPLATE),
        tool_definitions=copy.deepcopy(TOOL_DEFINITIONS),
        gm_prompt="You are the Game Master for Sample Letters.",
        player_prompt_template="You are playing Sample Letters, and your player ID is {player_id}.",
        rulebook_text="Sample Letters: draw a card, then play a card. Highest card wins.",
        game_spec=sample_spec(),
        core_mechanics=["A protected player cannot be targeted."],
    )
