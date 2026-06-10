import json
import warnings
from collections.abc import Callable
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, Field

from playtest.config import get_settings, maybe_wrap_openai
from playtest.ingestion.schemas import (
    ActionRule,
    ActionSpec,
    ActionSpecList,
    DealStep,
    GameSpec,
    SetupPlan,
    TurnStructure,
    VisibilitySpec,
)

# The harness's state interface. These fields are required in every game's state — they
# are how the driver routes turns, not game logic. Everything else is game-specific and
# designed by the analyst LLM from the rulebook.
ENGINE_CONTRACT_FIELDS = ("players", "current_turn", "turn_phase")


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


_ParsedT = TypeVar("_ParsedT", bound=BaseModel)


def _parse_with_repair(
    messages: list[dict],
    response_format: type[_ParsedT],
    validate: Callable[[_ParsedT], None],
    max_repairs: int = 1,
) -> _ParsedT:
    """Structured-Outputs counterpart of ``_generate_json_with_repair``.

    The schema shape is guaranteed by ``.parse``; ``validate`` enforces semantic
    self-consistency and a failure is fed back for one repair round.
    """
    client = _client()
    settings = get_settings()
    convo = list(messages)
    for attempt in range(max_repairs + 1):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Pydantic serializer warnings", category=UserWarning
            )
            completion = client.chat.completions.parse(
                model=settings.gm_model,
                messages=convo,  # type: ignore[arg-type]
                response_format=response_format,
            )
        message = completion.choices[0].message
        parsed = message.parsed
        try:
            if parsed is None:
                raise ValueError("the model returned no parsed result")
            validate(parsed)
            return parsed
        except ValueError as exc:
            if attempt == max_repairs:
                raise
            convo = convo + [
                {"role": "assistant", "content": message.content or ""},
                {
                    "role": "user",
                    "content": f"That output was invalid: {exc}. "
                    "Fix the problem and resend the COMPLETE object.",
                },
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
    # OpenAI's .parse() returns ParsedChatCompletion with its generic ContentType left
    # unbound, so anything that serializes the result (e.g. LangSmith tracing calling
    # model_dump()) emits a benign "Pydantic serializer warnings ... field_name='parsed'"
    # UserWarning. The parsed data is correct; silence only that warning around this call.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Pydantic serializer warnings", category=UserWarning
        )
        completion = client.chat.completions.parse(
            model=settings.gm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze a board-game rulebook and list every distinct action a "
                        "player can take on their turn as a function-calling tool spec — "
                        "drawing, playing or placing pieces, passing, bidding, trading, and so "
                        "on, as the rules dictate. Use snake_case names like 'draw_card' or "
                        "'play_guard'. For each action, list ONLY the game-specific parameters "
                        "(e.g. target_player, named_card). Do NOT include 'reasoning' or "
                        "'public_statement' — those are added automatically. Use enums where "
                        "the rules constrain a value to a fixed set."
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
    for field in ENGINE_CONTRACT_FIELDS:
        if field not in serialized:
            raise ValueError(f"state schema must document the '{field}' field")


def generate_state_schema(rulebook_text: str, num_players: int) -> dict:
    """Generate a JSON schema documenting every field in the game state."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a game-systems analyst. Design the JSON game state for this game and "
                "produce a JSON Schema (type 'object' with a 'properties' map) that documents "
                "every field. Each property must have a 'type' and a 'description' explaining "
                "what it means for the agents. Respond with a single JSON object.\n\n"
                "The harness requires these fields (everything else you derive from the "
                "rulebook):\n"
                "- 'players': an object keyed 'player_1'..'player_N', one sub-object per "
                "player holding all per-player fields (every player has the same fields).\n"
                "- 'current_turn': the player id whose turn it is.\n"
                "- 'turn_phase': the current phase within a turn.\n"
                "Hidden collections (e.g. a face-down deck) should be a list field plus a "
                "public integer count field (e.g. 'deck' and 'deck_count'); the same applies "
                "to per-player hidden collections (e.g. 'hand' and 'hand_count'). Add "
                "'game_name', 'variant', 'num_players', and a 'round_number' if the game is "
                "played in rounds. Use JSON-friendly scalar/list/object types only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Rulebook:\n\n{rulebook_text}\n\n"
                f"This is a {num_players}-player game. Document each field (including the "
                "per-player sub-fields) as a JSON Schema."
            ),
        },
    ]
    return _generate_json_with_repair(messages, _validate_state_schema)


# --- Initial state ----------------------------------------------------------


def _validate_initial_state(num_players: int, state_schema: dict) -> Callable[[dict], None]:
    documented = set((state_schema or {}).get("properties", {}))

    def validate(state: dict) -> None:
        missing = set(ENGINE_CONTRACT_FIELDS) - set(state)
        if missing:
            raise ValueError(f"missing required engine fields: {sorted(missing)}")
        if documented:
            undocumented = set(state) - documented
            if undocumented:
                raise ValueError(
                    f"fields not documented in the state schema: {sorted(undocumented)}"
                )
        players = state.get("players")
        if not isinstance(players, dict) or len(players) != num_players:
            raise ValueError(f"players must be an object with exactly {num_players} entries")
        expected_ids = {f"player_{i}" for i in range(1, num_players + 1)}
        if set(players) != expected_ids:
            raise ValueError(f"players must be keyed exactly {sorted(expected_ids)}")
        field_sets = {pid: frozenset(pdata) for pid, pdata in players.items()
                      if isinstance(pdata, dict)}
        if len(field_sets) != num_players:
            raise ValueError("every player entry must be an object")
        if len(set(field_sets.values())) != 1:
            raise ValueError("every player must have the same set of fields")
        if state.get("current_turn") not in players:
            raise ValueError("current_turn must be one of the player ids")
        if not isinstance(state.get("turn_phase"), str) or not state["turn_phase"]:
            raise ValueError("turn_phase must be a non-empty string")

    return validate


def generate_initial_state(rulebook_text: str, num_players: int, state_schema: dict) -> dict:
    """Generate the initial-state template (structural example, not a real deal)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You produce the initial game-state TEMPLATE used to document setup. This is a "
                "STRUCTURAL EXAMPLE with placeholder values, NOT a real dealt game — any "
                "shuffle and deal happen programmatically at runtime, so do not worry about "
                "randomness or a consistent component distribution. Include every field "
                "documented in the state schema and nothing else; key players "
                "'player_1'..'player_N' and give every player the same fields. Respond with a "
                "single JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Rulebook:\n\n{rulebook_text}\n\n"
                f"State schema:\n\n{json.dumps(state_schema, indent=2)}\n\n"
                f"Produce the initial state template for a {num_players}-player game."
            ),
        },
    ]
    return _generate_json_with_repair(messages, _validate_initial_state(num_players, state_schema))


# --- Prompts ----------------------------------------------------------------


def _validate_core_mechanics(data: dict) -> None:
    mechanics = data.get("mechanics")
    if not isinstance(mechanics, list) or not mechanics:
        raise ValueError("'mechanics' must be a non-empty JSON array")
    if not all(isinstance(m, str) and m.strip() for m in mechanics):
        raise ValueError("every item in 'mechanics' must be a non-empty string")


def generate_core_mechanics(rulebook_text: str) -> list[str]:
    """Extract cross-cutting mechanical constraints (not per-card effects) from the rulebook."""
    messages = [
        {
            "role": "system",
            "content": (
                "You extract a game's CROSS-CUTTING mechanical constraints as JSON. These are "
                "rules that affect multiple cards or actions, not the effect of any single card. "
                "Respond with a single JSON object of the form {\"mechanics\": [\"...\", ...]} "
                "containing 4-8 short, clear strings. Each string is one constraint. Focus on: "
                "protection/immunity effects, targeting restrictions (including what happens when "
                "no valid target exists), forced-play obligations, elimination triggers that are "
                "not specific to one card, tie handling, and self-targeting limits. Do NOT include "
                "individual card effects, setup rules, scoring rules, or win conditions."
            ),
        },
        {"role": "user", "content": f"Rulebook:\n\n{rulebook_text}"},
    ]
    return _generate_json_with_repair(messages, _validate_core_mechanics)["mechanics"]


def generate_game_overview(rulebook_text: str) -> str:
    """Generate a brief, rules-free overview (name, theme, players, win condition)."""
    return _chat_text(
        _client(),
        [
            {
                "role": "system",
                "content": (
                    "You write a 2-3 sentence overview of a tabletop game for an AI agent. State "
                    "the game name, theme, player count, and how a player wins. Do NOT include "
                    "rules, card effects, setup steps, or edge cases — only a high-level summary. "
                    "Write only the overview text."
                ),
            },
            {"role": "user", "content": f"Rulebook:\n\n{rulebook_text}"},
        ],
    ).strip()


# --- Game spec (setup + flow), the generic engine's configuration -----------
#
# Structured Outputs forbids free-form JSON objects, so the LLM-facing response models
# below use lists of named pairs; ``build_game_spec`` converts them into the dict-keyed
# ``GameSpec`` and runs the cross-artifact self-consistency checks.


class _ComponentCount(BaseModel):
    name: str
    count: int


class _CountFieldPair(BaseModel):
    list_field: str
    count_field: str


class _VisibilityResponse(BaseModel):
    per_player_private: list[str] = Field(
        description="Per-player list fields visible only to their owner, e.g. ['hand']."
    )
    hidden_fields: list[str] = Field(
        description="Top-level fields players must never see at all, e.g. ['deck']."
    )
    masked_fields: list[str] = Field(
        description=(
            "Top-level single-value fields whose true value is hidden and shown as "
            "'HIDDEN', e.g. ['removed_card']."
        )
    )
    count_fields: list[_CountFieldPair] = Field(
        description="Each hidden list field paired with its public integer count field."
    )


class _SetupPlanResponse(BaseModel):
    num_players: int
    pool: list[_ComponentCount] | None = Field(
        description=(
            "Components shuffled into a face-down pool at setup (None if the game has no "
            "shuffled pool)."
        )
    )
    pool_field: str | None = Field(
        description="State field holding the remaining pool after dealing, e.g. 'deck'."
    )
    deal_steps: list[DealStep] = Field(
        description="Ordered steps that move components from the shuffled pool into the state."
    )
    carry_over_fields: list[str] = Field(
        description=(
            "State paths preserved when a new round is dealt (use 'players.*.' for "
            "per-player fields), e.g. ['players.*.tokens', 'round_number']."
        )
    )


class _SetupSpecResponse(BaseModel):
    supported_player_counts: list[int]
    components: list[_ComponentCount] = Field(
        description="Every physical component in play: name -> total copies."
    )
    component_zones: list[str] = Field(
        description=(
            "Every state field where those components can ever be (use 'players.*.' for "
            "per-player fields), e.g. ['deck', 'removed_card', 'revealed_cards', "
            "'players.*.hand', 'players.*.discards']."
        )
    )
    setup_plans: list[_SetupPlanResponse] = Field(
        description="One plan per supported player count."
    )
    visibility: _VisibilityResponse


class _ActionRuleResponse(BaseModel):
    action: str
    phase: str
    ends_turn: bool


class _FlowSpecResponse(BaseModel):
    phases: list[str] = Field(description="The phases of one player's turn, in order.")
    initial_phase: str
    inactive_field: str | None = Field(
        description=(
            "Per-player boolean field meaning 'skip this player in turn order' "
            "(e.g. 'is_eliminated'), or null."
        )
    )
    action_rules: list[_ActionRuleResponse] = Field(
        description="One rule per action tool: its phase and whether it ends the turn."
    )
    has_rounds: bool = Field(
        description="True if the game is played in rounds that are scored and then redealt."
    )
    end_conditions: str = Field(
        description="Precise natural-language conditions that end a round and end the game."
    )
    scoring: str = Field(
        description="Precise natural-language rules for scoring a round/the game, incl. ties."
    )
    score_field: str | None = Field(
        description="The per-player numeric field tracking score (e.g. 'tokens'), or null."
    )


def generate_setup_spec(
    rulebook_text: str, initial_state_template: dict, num_players: int
) -> _SetupSpecResponse:
    """Extract components, per-count setup plans, and visibility from the rulebook."""
    top_level = sorted(initial_state_template)
    player_fields = sorted(next(iter(initial_state_template["players"].values())))

    def validate(resp: _SetupSpecResponse) -> None:
        problems: list[str] = []
        if num_players not in resp.supported_player_counts:
            problems.append(f"supported_player_counts must include {num_players}")
        plan_counts = {p.num_players for p in resp.setup_plans}
        missing_plans = set(resp.supported_player_counts) - plan_counts
        if missing_plans:
            problems.append(f"missing setup plans for player counts {sorted(missing_plans)}")
        for zone in resp.component_zones:
            if zone.startswith("players.*."):
                if zone[len("players.*.") :] not in player_fields:
                    problems.append(f"component zone {zone!r} is not a per-player field")
            elif zone not in top_level:
                problems.append(f"component zone {zone!r} is not a top-level state field")
        for plan in resp.setup_plans:
            if plan.pool is not None and plan.pool_field not in top_level:
                problems.append(
                    f"{plan.num_players}p plan: pool_field {plan.pool_field!r} is not a "
                    "top-level state field"
                )
            consumed = 0
            for step in plan.deal_steps:
                if step.target == "each_player":
                    if step.to_field not in player_fields:
                        problems.append(
                            f"{plan.num_players}p plan: deal target {step.to_field!r} is not "
                            "a per-player field"
                        )
                    consumed += step.count * plan.num_players
                else:
                    if step.to_field not in top_level:
                        problems.append(
                            f"{plan.num_players}p plan: deal target {step.to_field!r} is not "
                            "a top-level state field"
                        )
                    consumed += step.count
            pool_total = sum(c.count for c in plan.pool or [])
            if plan.pool is not None and consumed > pool_total:
                problems.append(
                    f"{plan.num_players}p plan: deal steps consume {consumed} components but "
                    f"the pool only has {pool_total}"
                )
        vis = resp.visibility
        for field in vis.per_player_private:
            if field not in player_fields:
                problems.append(f"per_player_private field {field!r} is not a per-player field")
        for field in vis.hidden_fields + vis.masked_fields:
            if field not in top_level:
                problems.append(f"visibility field {field!r} is not a top-level state field")
        for pair in vis.count_fields:
            known = (
                pair.list_field in top_level and pair.count_field in top_level
            ) or (pair.list_field in player_fields and pair.count_field in player_fields)
            if not known:
                problems.append(
                    f"count pair {pair.list_field!r}/{pair.count_field!r} does not exist at "
                    "the same level of the state"
                )
        if problems:
            raise ValueError("; ".join(problems))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a game-systems analyst extracting the SETUP specification of a board "
                "game so a generic engine can shuffle and deal deterministically. Be exact "
                "about counts. The state's top-level fields are "
                f"{top_level} and each player has the fields {player_fields}; refer only to "
                "these."
            ),
        },
        {"role": "user", "content": f"Rulebook:\n\n{rulebook_text}"},
    ]
    return _parse_with_repair(messages, _SetupSpecResponse, validate)


def generate_flow_spec(
    rulebook_text: str, tool_definitions: dict[str, dict], initial_state_template: dict
) -> _FlowSpecResponse:
    """Extract turn structure, per-action rules, and end/scoring text from the rulebook."""
    action_names = sorted(tool_definitions)
    player_fields = sorted(next(iter(initial_state_template["players"].values())))

    def validate(resp: _FlowSpecResponse) -> None:
        problems: list[str] = []
        if not resp.phases:
            problems.append("phases must be non-empty")
        if resp.initial_phase not in resp.phases:
            problems.append(f"initial_phase {resp.initial_phase!r} is not one of the phases")
        ruled = {r.action for r in resp.action_rules}
        missing = set(action_names) - ruled
        unknown = ruled - set(action_names)
        if missing:
            problems.append(f"missing action rules for {sorted(missing)}")
        if unknown:
            problems.append(f"action rules reference unknown actions {sorted(unknown)}")
        for rule in resp.action_rules:
            if rule.phase not in resp.phases:
                problems.append(f"action {rule.action!r} uses unknown phase {rule.phase!r}")
        if resp.inactive_field is not None and resp.inactive_field not in player_fields:
            problems.append(f"inactive_field {resp.inactive_field!r} is not a per-player field")
        if resp.score_field is not None and resp.score_field not in player_fields:
            problems.append(f"score_field {resp.score_field!r} is not a per-player field")
        if not resp.end_conditions.strip() or not resp.scoring.strip():
            problems.append("end_conditions and scoring must be non-empty")
        if problems:
            raise ValueError("; ".join(problems))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a game-systems analyst extracting the TURN FLOW specification of a "
                "board game for a generic engine. The player action tools are "
                f"{action_names}; provide exactly one rule per tool. Each player has the "
                f"state fields {player_fields}. end_conditions and scoring are read by the "
                "Game Master agent at runtime — write them as precise, complete instructions "
                "(cover ties and edge cases)."
            ),
        },
        {"role": "user", "content": f"Rulebook:\n\n{rulebook_text}"},
    ]
    return _parse_with_repair(messages, _FlowSpecResponse, validate)


def build_game_spec(
    setup: _SetupSpecResponse, flow: _FlowSpecResponse
) -> GameSpec:
    """Assemble the two extraction halves into the engine-facing GameSpec."""
    return GameSpec(
        supported_player_counts=sorted(set(setup.supported_player_counts)),
        components={c.name: c.count for c in setup.components},
        component_zones=list(setup.component_zones),
        setup_plans={
            str(p.num_players): SetupPlan(
                pool={c.name: c.count for c in p.pool} if p.pool is not None else None,
                pool_field=p.pool_field,
                deal_steps=list(p.deal_steps),
                carry_over_fields=list(p.carry_over_fields),
            )
            for p in setup.setup_plans
        },
        turn=TurnStructure(
            phases=list(flow.phases),
            initial_phase=flow.initial_phase,
            inactive_field=flow.inactive_field,
        ),
        action_rules={r.action: ActionRule(phase=r.phase, ends_turn=r.ends_turn)
                      for r in flow.action_rules},
        has_rounds=flow.has_rounds,
        end_conditions=flow.end_conditions.strip(),
        scoring=flow.scoring.strip(),
        score_field=flow.score_field,
        visibility=VisibilitySpec(
            per_player_private=list(setup.visibility.per_player_private),
            hidden_fields=list(setup.visibility.hidden_fields),
            masked_fields=list(setup.visibility.masked_fields),
            count_fields={
                pair.list_field: pair.count_field for pair in setup.visibility.count_fields
            },
        ),
    )


def _core_mechanics_section(core_mechanics: list[str]) -> str:
    """Render the code-injected Core Mechanics block (empty string if none)."""
    if not core_mechanics:
        return ""
    bullets = "\n".join(f"- {m}" for m in core_mechanics)
    return (
        "## Core Mechanics\n\n"
        "These constraints always apply and affect multiple actions:\n"
        f"{bullets}\n\n"
    )


def generate_gm_prompt(
    rulebook_text: str,
    state_schema: dict,
    tool_definitions: dict,
    game_overview: str,
    core_mechanics: list[str],
    end_conditions: str,
    scoring: str,
) -> str:
    """Generate the GM system prompt; rules live in the rulebook (query_rulebook), not here."""
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
                    f"- Detect and enforce the round/game end conditions and the scoring rules "
                    f"that are provided in dedicated sections of this prompt.\n"
                    f"- Narrate what happened briefly and with flavor after each action.\n"
                    f"- Manage turn order and determine whose turn is next.\n"
                    f"- When rejecting an illegal action, explain precisely why so the player "
                    f"can choose a legal one.\n\n"
                    f"Do NOT reproduce the rules, card descriptions, or rulebook text in the "
                    f"prompt — the GM looks rules up with the query_rulebook tool. Write only the "
                    f"prompt text.\n\nRulebook for reference:\n\n{rulebook_text}"
                ),
            },
        ],
    )

    return (
        f"{instructions.strip()}\n\n"
        f"## Game Overview\n\n{game_overview.strip()}\n\n"
        f"{_core_mechanics_section(core_mechanics)}"
        f"## End Conditions\n\n{end_conditions.strip()}\n\n"
        f"## Scoring\n\n{scoring.strip()}\n\n"
        "## Looking Up Rules\n\n"
        "You have access to a `query_rulebook` tool. Use it to look up any game rule before "
        "validating or resolving an action. Do not rely on memory — always verify against the "
        "rulebook.\n\n"
        f"## Game State Schema\n\n{json.dumps(state_schema, indent=2)}\n\n"
        f"## Player Action Tools (for validating proposed actions)\n\n"
        f"{json.dumps(tool_definitions, indent=2)}\n"
    )


def generate_player_prompt(
    game_overview: str,
    core_mechanics: list[str],
    phases: list[str],
    forbidden_action_names: list[str] | None = None,
) -> str:
    """Generate the player system prompt template (keeps a literal {player_id} placeholder)."""
    forbidden = forbidden_action_names or []

    def validate(text: str) -> None:
        if "{player_id}" not in text:
            raise ValueError("the prompt must contain the literal placeholder {player_id}")
        if "query_rulebook" not in text:
            raise ValueError("the prompt must instruct the player to use the query_rulebook tool")
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
                "- Include the brief game overview below (verbatim or lightly paraphrased).\n"
                "- Do NOT include a card-effect reference, rules, or setup details. Instead, "
                "instruct them to use the query_rulebook tool to look up what their cards do "
                "before deciding which to play.\n"
                "- They can only see their own private information; they must deduce hidden "
                "information from what is public.\n"
                "- On their turn, first call get_game_state, then choose an action.\n"
                f"- Each turn moves through these phases in order: {', '.join(phases)}. The "
                "action tools offered at any moment are the ones available in the current "
                "phase.\n"
                "- Their available actions are provided to them as tools each turn — do NOT "
                "enumerate specific action tool names; refer to them generically.\n"
                "- Use the reasoning field to think privately; use public_statement to speak.\n"
                "- If an action is rejected by the GM, read the error and try a different legal "
                "action.\n\nWrite only the prompt text.\n\n"
                f"Game overview:\n\n{game_overview}"
            ),
        },
    ]
    prompt = _generate_text_with_repair(messages, validate)
    section = _core_mechanics_section(core_mechanics)
    if section:
        prompt = f"{prompt.rstrip()}\n\n{section.rstrip()}"
    return prompt
