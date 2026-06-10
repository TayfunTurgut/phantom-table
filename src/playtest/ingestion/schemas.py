import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ActionParamSpec(BaseModel):
    """A single parameter of a player action, as emitted by the LLM."""

    name: str
    type: str = Field(description="JSON schema type, e.g. 'string', 'boolean', 'integer'.")
    description: str
    enum: list[str] | None = None
    required: bool = True


class ActionSpec(BaseModel):
    """One distinct player action, as emitted by the LLM (Structured Output)."""

    name: str = Field(description="Tool name, snake_case, e.g. 'play_guard'.")
    description: str
    params: list[ActionParamSpec]


class ActionSpecList(BaseModel):
    """Structured-Output response model for action extraction."""

    actions: list[ActionSpec]


class DealStep(BaseModel):
    """One step of the seeded setup: move components from the shuffled pool somewhere."""

    count: int
    target: Literal["each_player", "set_aside", "reveal"]
    to_field: str = Field(
        description=(
            "Destination state field: a per-player field for 'each_player' (e.g. 'hand'), "
            "or a top-level field for 'set_aside'/'reveal' (e.g. 'removed_card', "
            "'revealed_cards')."
        )
    )


class SetupPlan(BaseModel):
    """How to build a fully dealt state for one player count (executed by the engine)."""

    pool: dict[str, int] | None = Field(
        default=None,
        description="Component name -> copies shuffled into the pool; None = no shuffled pool.",
    )
    pool_field: str | None = Field(
        default=None,
        description="Top-level state field holding the remaining pool after dealing, e.g. 'deck'.",
    )
    deal_steps: list[DealStep] = []
    carry_over_fields: list[str] = Field(
        default=[],
        description=(
            "Dotted state paths preserved across rounds (supports a 'players.*.' prefix), "
            "e.g. ['players.*.tokens']."
        ),
    )


class ActionRule(BaseModel):
    """Engine-relevant metadata for one player action tool."""

    phase: str
    ends_turn: bool


class TurnStructure(BaseModel):
    """The turn flow the driver enforces (the GM judges everything else)."""

    phases: list[str]
    initial_phase: str
    inactive_field: str | None = Field(
        default=None,
        description=(
            "Per-player boolean field meaning 'skip this player in turn order' "
            "(e.g. 'is_eliminated'), or None if players are never skipped."
        ),
    )


class VisibilitySpec(BaseModel):
    """Which parts of the state are hidden from players (the GM always sees everything)."""

    per_player_private: list[str] = Field(
        default=[],
        description="Per-player list fields visible only to their owner, e.g. ['hand'].",
    )
    hidden_fields: list[str] = Field(
        default=[],
        description="Top-level fields dropped entirely from player views, e.g. ['deck'].",
    )
    masked_fields: list[str] = Field(
        default=[],
        description=(
            "Top-level fields whose true value is stashed privately and shown as 'HIDDEN' "
            "to players, e.g. ['removed_card']."
        ),
    )
    count_fields: dict[str, str] = Field(
        default={},
        description="List field -> public count field, e.g. {'hand': 'hand_count'}.",
    )


class GameSpec(BaseModel):
    """Everything the generic engine needs, extracted from the rulebook at ingestion.

    Deterministic primitives (shuffle/deal/redact/conserve/rotate turns) are configured by
    this spec; all judgment (legality, action effects, end conditions, scoring) stays with
    the GM, grounded by the rulebook.
    """

    supported_player_counts: list[int]
    components: dict[str, int] = Field(
        description="Full component manifest: name -> total copies in play."
    )
    component_zones: list[str] = Field(
        description=(
            "State fields where components live (basis for conservation checks), e.g. "
            "['deck', 'removed_card', 'revealed_cards', 'players.*.hand', "
            "'players.*.discards']."
        )
    )
    setup_plans: dict[str, SetupPlan] = Field(
        description="Setup plan per player count (JSON object keys are strings)."
    )
    turn: TurnStructure
    action_rules: dict[str, ActionRule] = Field(
        description="Per action tool name: which phase it belongs to and whether it ends the turn."
    )
    has_rounds: bool = Field(
        description="True for multi-round games (score the round, then redeal)."
    )
    end_conditions: str = Field(
        description="Natural-language round/game end conditions, injected into the GM prompt."
    )
    scoring: str = Field(
        description="Natural-language scoring rules, injected into the GM prompt."
    )
    score_field: str | None = Field(
        default=None,
        description="Per-player numeric field used for score display, e.g. 'tokens'.",
    )
    visibility: VisibilitySpec

    def setup_plan_for(self, num_players: int) -> SetupPlan:
        """The setup plan for a player count (KeyError-free, with a clear message)."""
        plan = self.setup_plans.get(str(num_players))
        if plan is None:
            raise ValueError(
                f"no setup plan for {num_players} players; plans exist for "
                f"{sorted(self.setup_plans)}"
            )
        return plan


class GameConfig(BaseModel):
    """A complete game configuration produced by the ingestion pipeline."""

    game_name: str
    variant: str
    num_players: int
    config_dir: str
    state_schema: dict
    initial_state_template: dict
    tool_definitions: dict[str, dict]
    gm_prompt: str
    player_prompt_template: str
    rulebook_text: str
    game_spec: GameSpec
    core_mechanics: list[str] = []

    def save(self) -> None:
        """Save all artifacts to config_dir as JSON/text files."""
        config_path = Path(self.config_dir)
        config_path.mkdir(parents=True, exist_ok=True)

        (config_path / "state_schema.json").write_text(
            json.dumps(self.state_schema, indent=2), encoding="utf-8"
        )
        (config_path / "initial_state.json").write_text(
            json.dumps(self.initial_state_template, indent=2), encoding="utf-8"
        )
        (config_path / "tool_definitions.json").write_text(
            json.dumps(self.tool_definitions, indent=2), encoding="utf-8"
        )
        (config_path / "gm_prompt.txt").write_text(self.gm_prompt, encoding="utf-8")
        (config_path / "player_prompt.txt").write_text(
            self.player_prompt_template, encoding="utf-8"
        )
        (config_path / "rulebook.txt").write_text(self.rulebook_text, encoding="utf-8")
        (config_path / "core_mechanics.json").write_text(
            json.dumps(self.core_mechanics, indent=2), encoding="utf-8"
        )
        (config_path / "game_spec.json").write_text(
            self.game_spec.model_dump_json(indent=2), encoding="utf-8"
        )
        (config_path / "config.json").write_text(
            json.dumps(
                {
                    "game_name": self.game_name,
                    "variant": self.variant,
                    "num_players": self.num_players,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, config_dir: str) -> "GameConfig":
        """Load a game configuration from disk."""
        config_path = Path(config_dir)

        required_files = [
            "config.json",
            "state_schema.json",
            "initial_state.json",
            "tool_definitions.json",
            "game_spec.json",
            "gm_prompt.txt",
            "player_prompt.txt",
            "rulebook.txt",
        ]
        missing = [name for name in required_files if not (config_path / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete game config at {config_path}; missing files: "
                f"{', '.join(missing)}. Re-run ingestion for this game."
            )

        meta = json.loads((config_path / "config.json").read_text(encoding="utf-8"))

        core_mechanics_path = config_path / "core_mechanics.json"
        core_mechanics = (
            json.loads(core_mechanics_path.read_text(encoding="utf-8"))
            if core_mechanics_path.exists()
            else []
        )

        return cls(
            game_name=meta["game_name"],
            variant=meta["variant"],
            num_players=meta["num_players"],
            config_dir=str(config_path),
            state_schema=json.loads(
                (config_path / "state_schema.json").read_text(encoding="utf-8")
            ),
            initial_state_template=json.loads(
                (config_path / "initial_state.json").read_text(encoding="utf-8")
            ),
            tool_definitions=json.loads(
                (config_path / "tool_definitions.json").read_text(encoding="utf-8")
            ),
            gm_prompt=(config_path / "gm_prompt.txt").read_text(encoding="utf-8"),
            player_prompt_template=(config_path / "player_prompt.txt").read_text(encoding="utf-8"),
            rulebook_text=(config_path / "rulebook.txt").read_text(encoding="utf-8"),
            game_spec=GameSpec.model_validate_json(
                (config_path / "game_spec.json").read_text(encoding="utf-8")
            ),
            core_mechanics=core_mechanics,
        )
