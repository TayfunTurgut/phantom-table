import json
from collections.abc import Callable

from openai import OpenAI

from playtest.config import get_settings, maybe_wrap_openai
from playtest.ingestion.schemas import ActionSpec, ActionSpecList

# Canonical 2-player structural example, embedded into prompts so the LLM matches the shape.
_INITIAL_STATE_EXAMPLE = {
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

_REQUIRED_TOP_LEVEL = {
    "game_name",
    "variant",
    "num_players",
    "tokens_to_win",
    "round_number",
    "current_turn",
    "turn_phase",
    "deck_count",
    "removed_card",
    "revealed_cards",
    "players",
}
_REQUIRED_PLAYER_FIELDS = {
    "hand",
    "hand_count",
    "discards",
    "tokens",
    "is_eliminated",
    "is_protected",
}


def _client() -> OpenAI:
    return maybe_wrap_openai(OpenAI(api_key=get_settings().openai_api_key))


def _chat_text(client: OpenAI, messages: list[dict], json_mode: bool = False) -> str:
    settings = get_settings()
    kwargs: dict = {"model": settings.gm_model, "messages": messages}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    content = client.chat.completions.create(**kwargs).choices[0].message.content
    return content or ""


def _generate_json_with_repair(
    messages: list[dict], validate: Callable[[dict], None], max_repairs: int = 1
) -> dict:
    client = _client()
    convo = list(messages)
    for attempt in range(max_repairs + 1):
        content = _chat_text(client, convo, json_mode=True)
        try:
            # JSONDecodeError subclasses ValueError, so non-JSON output is repairable too.
            data = json.loads(content)
            validate(data)
            return data
        except ValueError as exc:
            if attempt == max_repairs:
                raise
            convo = convo + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"That output was invalid: {exc}. "
                    "Fix the problem and resend the COMPLETE JSON object.",
                },
            ]
    raise RuntimeError("unreachable")


def _generate_text_with_repair(
    messages: list[dict], validate: Callable[[str], None], max_repairs: int = 1
) -> str:
    client = _client()
    convo = list(messages)
    for attempt in range(max_repairs + 1):
        content = _chat_text(client, convo)
        try:
            validate(content)
            return content
        except ValueError as exc:
            if attempt == max_repairs:
                raise
            convo = convo + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"That output was invalid: {exc}. Fix it and resend."},
            ]
    raise RuntimeError("unreachable")


# --- Tool definitions -------------------------------------------------------


def _build_openai_tool(spec: ActionSpec) -> dict:
    """Assemble a strict-mode OpenAI function schema from an action spec (code-side)."""
    properties: dict[str, dict] = {}
    required: list[str] = []

    for param in spec.params:
        if param.name in ("reasoning", "public_statement"):
            continue  # injected below; avoid collisions
        json_type: object = param.type
        if not param.required:
            json_type = [param.type, "null"]  # strict mode: optional -> nullable + required
        prop: dict = {"type": json_type, "description": param.description}
        if param.enum:
            prop["enum"] = param.enum
        properties[param.name] = prop
        required.append(param.name)

    properties["reasoning"] = {
        "type": "string",
        "description": "Your private reasoning for this action. Never shown to other players.",
    }
    properties["public_statement"] = {
        "type": "string",
        "description": "What you say aloud at the table. Visible to all players.",
    }
    required.extend(["reasoning", "public_statement"])

    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def generate_tool_definitions(rulebook_text: str) -> dict[str, dict]:
    """Extract player actions (Structured Output) and assemble strict-mode tool schemas."""
    client = _client()
    settings = get_settings()
    completion = client.chat.completions.parse(
        model=settings.gm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze a board-game rulebook and list every distinct action a player "
                    "can take on their turn as a function-calling tool spec. Include one action "
                    "for drawing a card and one action per playable card. Use snake_case names "
                    "like 'draw_card' or 'play_guard'. For each action, list ONLY the "
                    "game-specific parameters (e.g. target_player, named_card). Do NOT include "
                    "'reasoning' or 'public_statement' — those are added automatically. Use enums "
                    "where the rules constrain a value (e.g. Guard names a non-Guard card)."
                ),
            },
            {"role": "user", "content": f"Rulebook:\n\n{rulebook_text}"},
        ],
        response_format=ActionSpecList,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Action extraction returned no parsed result.")

    return {spec.name: _build_openai_tool(spec) for spec in parsed.actions}


# --- State schema -----------------------------------------------------------


def _validate_state_schema(schema: dict) -> None:
    if not isinstance(schema, dict) or not schema:
        raise ValueError("state schema must be a non-empty JSON object")
    serialized = json.dumps(schema)
    for field in ("players", "deck_count", "current_turn", "turn_phase"):
        if field not in serialized:
            raise ValueError(f"state schema must document the '{field}' field")


def generate_state_schema(rulebook_text: str, num_players: int) -> dict:
    """Generate a JSON schema documenting every field in the game state."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a game-systems analyst. Produce a JSON Schema (type 'object' with a "
                "'properties' map) that documents every field of the game state. Each property "
                "must have a 'type' and a 'description' explaining what it means for the agents. "
                "Respond with a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Rulebook:\n\n{rulebook_text}\n\n"
                f"This is a {num_players}-player game. The game state looks structurally like "
                f"this example:\n\n{json.dumps(_INITIAL_STATE_EXAMPLE, indent=2)}\n\n"
                "Document each field (including the per-player sub-fields) as a JSON Schema."
            ),
        },
    ]
    return _generate_json_with_repair(messages, _validate_state_schema)


# --- Initial state ----------------------------------------------------------


def _validate_initial_state(num_players: int) -> Callable[[dict], None]:
    revealed = 3 if num_players == 2 else 0
    expected_deck = 16 - 1 - revealed - num_players

    def validate(state: dict) -> None:
        missing = _REQUIRED_TOP_LEVEL - set(state)
        if missing:
            raise ValueError(f"missing top-level fields: {sorted(missing)}")
        revealed_cards = state.get("revealed_cards")
        if not isinstance(revealed_cards, list) or len(revealed_cards) != revealed:
            raise ValueError(f"revealed_cards must be a list of exactly {revealed} cards")
        if state.get("deck_count") != expected_deck:
            raise ValueError(
                f"deck_count must be {expected_deck} "
                f"(16 - 1 removed - {revealed} revealed - {num_players} dealt)"
            )
        players = state.get("players")
        if not isinstance(players, dict) or len(players) != num_players:
            raise ValueError(f"players must be an object with exactly {num_players} entries")
        for pid, pdata in players.items():
            if not isinstance(pdata, dict):
                raise ValueError(f"player {pid} must be an object")
            pmissing = _REQUIRED_PLAYER_FIELDS - set(pdata)
            if pmissing:
                raise ValueError(f"player {pid} missing fields: {sorted(pmissing)}")

    return validate


def generate_initial_state(rulebook_text: str, num_players: int, state_schema: dict) -> dict:
    """Generate the initial-state template (structural example, not a real deal)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You produce the initial game-state TEMPLATE used to document setup. This is a "
                "STRUCTURAL EXAMPLE with placeholder card values, NOT a real dealt game — the "
                "actual shuffle and deal happen programmatically at runtime, so do not worry "
                "about randomness or a consistent card distribution. Match the given structure "
                "exactly. Respond with a single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Rulebook:\n\n{rulebook_text}\n\n"
                f"State schema:\n\n{json.dumps(state_schema, indent=2)}\n\n"
                f"Produce the initial state for a {num_players}-player game with this exact "
                f"structure:\n\n{json.dumps(_INITIAL_STATE_EXAMPLE, indent=2)}"
            ),
        },
    ]
    return _generate_json_with_repair(messages, _validate_initial_state(num_players))


# --- Prompts ----------------------------------------------------------------


def generate_gm_prompt(rulebook_text: str, state_schema: dict, tool_definitions: dict) -> str:
    """Generate the GM system prompt; code appends rules, schema, and tool definitions."""
    instructions = _chat_text(
        _client(),
        [
            {
                "role": "system",
                "content": "You write a system prompt for a Game Master (GM) agent.",
            },
            {
                "role": "user",
                "content": (
                    f"Write the system prompt for the Game Master of this game. The GM must:\n"
                    f"- Be the authoritative rules enforcer.\n"
                    f"- On a proposed player action, validate it against the rules (query the "
                    f"rulebook if uncertain), then resolve it and write the full updated state.\n"
                    f"- Always call set_game_state with the COMPLETE state object after any change "
                    f"(full replacement, never partial).\n"
                    f"- Detect and handle: round end (deck empty -> compare hands -> award "
                    f"tokens), game end (token threshold), player elimination, Handmaid "
                    f"protection blocking targeting, Countess forced-play, Baron tie, and a "
                    f"Prince forcing a Princess discard (elimination).\n"
                    f"- Narrate what happened briefly and with flavor after each action.\n"
                    f"- Manage turn order and determine whose turn is next.\n"
                    f"- When rejecting an illegal action, explain why and give the player another "
                    f"chance.\n\n"
                    f"Write only the prompt text.\n\nRulebook for reference:\n\n{rulebook_text}"
                ),
            },
        ],
    )

    return (
        f"{instructions.strip()}\n\n"
        f"## Complete Rules\n\n{rulebook_text.strip()}\n\n"
        f"## Game State Schema\n\n{json.dumps(state_schema, indent=2)}\n\n"
        f"## Player Action Tools (for validating proposed actions)\n\n"
        f"{json.dumps(tool_definitions, indent=2)}\n"
    )


def generate_player_prompt(
    rulebook_text: str, forbidden_action_names: list[str] | None = None
) -> str:
    """Generate the player system prompt template (keeps a literal {player_id} placeholder)."""
    forbidden = forbidden_action_names or []

    def validate(text: str) -> None:
        if "{player_id}" not in text:
            raise ValueError("the prompt must contain the literal placeholder {player_id}")
        leaked = [name for name in forbidden if name in text]
        if leaked:
            raise ValueError(
                f"do not hardcode game-specific action names {leaked}; refer to actions "
                "generically as tools provided each turn"
            )

    messages = [
        {
            "role": "system",
            "content": "You write a system prompt TEMPLATE for a player agent.",
        },
        {
            "role": "user",
            "content": (
                "Write a system prompt template for a player agent in this game. Requirements:\n"
                "- Start by stating they are playing the game and their player ID is the literal "
                "placeholder {player_id} (keep it exactly as {player_id}).\n"
                "- Give a condensed rules summary: what each card does and how turns work.\n"
                "- They can ONLY see their own hand; they must deduce hidden information from "
                "public discards and revealed cards.\n"
                "- On their turn, first call get_game_state, then choose an action.\n"
                "- During the draw phase call draw_card; during the play phase play a card.\n"
                "- Their available card-play actions are provided to them as tools each turn — do "
                "NOT enumerate specific card-play tool names; refer to them generically.\n"
                "- Use the reasoning field to think privately; use public_statement to speak.\n"
                "- If an action is rejected by the GM, read the error and try a different legal "
                "action.\n\nWrite only the prompt text.\n\n"
                f"Rulebook for reference:\n\n{rulebook_text}"
            ),
        },
    ]
    return _generate_text_with_repair(messages, validate)
