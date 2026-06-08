import json
from pathlib import Path

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
        meta = json.loads((config_path / "config.json").read_text(encoding="utf-8"))

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
        )
