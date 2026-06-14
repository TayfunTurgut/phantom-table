"""Stage 1: rulebook text → structured GameDigest (+ human-readable digest.md)."""

from __future__ import annotations

import json
from pathlib import Path

from playtest.ingestion.schemas import GameDigest
from playtest.llm import LLMClient

_SYSTEM_PROMPT = """You are a meticulous board game analyst. You read a rulebook and
produce a complete, unambiguous digest that a programmer will implement a game
engine from — WITHOUT reading the rulebook again. Everything needed to implement
the game must be in your digest.

Requirements:
- components: only conserved physical pieces from a fixed pool (cards, tiles).
  Do not include unbounded score markers or the rulebook's flavor items.
- actions: every decision a player can ever make, each with its complete
  resolution rules including edge cases (no valid target, empty deck, ties...).
  Name them in snake_case. Forced bookkeeping (mandatory draws, refills) is NOT
  an action — the engine auto-advances through it.
- decision_flow: who acts when. Be explicit about simultaneous decisions and
  reaction windows if the game has them.
- state_shape: design the canonical JSON state dict the engine will use. List
  every key and its type, including per-player sub-dicts. Keep it flat and
  obvious. Cards/components are stored by their string name. Include the keys
  "rng_seed", "game_over" and "winners".
- ambiguities: anything the rulebook leaves unclear that affects implementation.
  Pick a sensible resolution for each and quote the passage that motivated it.
  The engine will implement your resolution.
- scoring: include every tiebreaker, fully specified.
"""


def generate_digest(client: LLMClient, rulebook_text: str) -> GameDigest:
    """One structured-output call: the rulebook in, the digest out."""
    schema = GameDigest.model_json_schema()
    raw = client.complete(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"RULEBOOK:\n\n{rulebook_text}"},
        ],
        role="digest",
        json_schema={"name": "game_digest", "strict": True, "schema": schema},
    )
    return GameDigest.model_validate_json(raw)


def digest_to_markdown(digest: GameDigest) -> str:
    """Render the digest for human review."""
    lines = [
        f"# {digest.game_name} — Ingestion Digest",
        "",
        digest.overview.strip(),
        "",
        f"**Players:** {digest.min_players}-{digest.max_players}",
        "",
        "## Components",
        "",
        *(f"- {component.name} × {component.count}" for component in digest.components),
        "",
        "## Hidden Information",
        "",
        digest.hidden_zones.strip(),
        "",
        "## Setup",
        "",
        digest.setup.strip(),
        "",
        "## Decision Flow",
        "",
        digest.decision_flow.strip(),
        "",
        "## Actions",
        "",
    ]
    for action in digest.actions:
        lines += [
            f"### `{action.name}`",
            "",
            f"*When:* {action.when}",
            "",
            action.effect.strip(),
            "",
        ]
    lines += [
        "## End Conditions",
        "",
        digest.end_conditions.strip(),
        "",
        "## Scoring",
        "",
        digest.scoring.strip(),
        "",
        "## State Shape",
        "",
        "```",
        digest.state_shape.strip(),
        "```",
        "",
        "## Ambiguities and Rulings",
        "",
    ]
    if digest.ambiguities:
        for amb in digest.ambiguities:
            lines += [
                f"- **Q:** {amb.question}",
                f"  **Ruling:** {amb.resolution}",
                f'  **Rulebook:** "{amb.rulebook_quote}"',
            ]
    else:
        lines.append("None identified.")
    lines.append("")
    return "\n".join(lines)


def digest_to_player_briefing(digest: GameDigest) -> str:
    """Render the rules summary players receive in their system prompt."""
    actions = "\n".join(f"- {a.name}: {a.effect}" for a in digest.actions)
    return (
        f"{digest.overview.strip()}\n\n"
        f"How play proceeds:\n{digest.decision_flow.strip()}\n\n"
        f"Actions:\n{actions}\n\n"
        f"The game ends:\n{digest.end_conditions.strip()}\n\n"
        f"Winning:\n{digest.scoring.strip()}\n"
    )


def save_digest(digest: GameDigest, config_dir: Path) -> None:
    (config_dir / "digest.json").write_text(
        json.dumps(digest.model_dump(), indent=2), encoding="utf-8"
    )
    (config_dir / "digest.md").write_text(digest_to_markdown(digest), encoding="utf-8")
    (config_dir / "player_briefing.txt").write_text(
        digest_to_player_briefing(digest), encoding="utf-8"
    )
