"""Ingestion data models: the game digest and the on-disk artifact layout.

The digest is the structured, human-reviewable understanding of a rulebook that
engine code is generated against. It is deliberately mostly prose: the *code* is
the precise artifact; the digest exists so a human (and the codegen model) can
see what the game is and how every decision works, with ambiguities resolved
explicitly before any code exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ComponentCount(BaseModel):
    # Strict structured outputs reject open-key dicts, so component counts are a
    # list of name/count pairs rather than dict[str, int].
    model_config = ConfigDict(extra="forbid")

    name: str
    count: int


class ActionDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # snake_case identifier, e.g. "play_guard"
    when: str  # when this decision is available, and to whom
    effect: str  # complete resolution rules, including edge cases


class Ambiguity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str  # what the rulebook leaves unclear
    resolution: str  # the ruling the engine will implement
    rulebook_quote: str  # the passage that motivated the question


class GameDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_name: str
    overview: str  # theme, goal, and the one-paragraph pitch
    min_players: int
    max_players: int
    components: list[ComponentCount]  # conserved physical pieces (cards, tiles in a fixed pool)
    hidden_zones: str  # what information is hidden from whom
    setup: str  # the full setup procedure for each player count
    decision_flow: str  # who acts when: turn order, phases, simultaneity, reaction windows
    actions: list[ActionDigest]  # every decision a player can make, fully specified
    end_conditions: str  # when a round/game ends
    scoring: str  # how winners are determined, including all tiebreakers
    state_shape: str  # the canonical state dict layout the engine will use (keys + types)
    ambiguities: list[Ambiguity]  # unclear rules with explicit resolutions


class GameArtifacts:
    """Thin loader for a generated game config directory."""

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)
        self.name = self.config_dir.name
        self.engine_path = self.config_dir / "engine.py"
        self.digest = GameDigest.model_validate_json(
            (self.config_dir / "digest.json").read_text(encoding="utf-8")
        )
        meta_path = self.config_dir / "meta.json"
        self.meta: dict = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        )
