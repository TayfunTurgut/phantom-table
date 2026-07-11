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
- components: physical pieces with fixed printed counts (cards, tiles from a
  fixed pool). Do not include unbounded score markers, the rulebook's flavor
  items, or open/unlimited supplies — those belong in zones with a note.
- mechanics: structural tags describing what the BASE game actually has. Tag
  only mechanics the game truly uses; leave the rest off. The vocabulary:
  - simultaneous_decisions: several seats choose at once (hidden reveal, votes).
  - reaction_windows: a "may respond"/interrupt window (block, challenge, Nope).
  - multi_stage_turns: one turn is several dependent choices (move then build).
  - open_supply: a shared, unlimited or replenishing pool (a market, a bank).
  - board_or_map: a shared board, map, or track pieces move across.
  - hidden_hands: players hold private cards/tiles others cannot see.
  - player_elimination: players can be knocked out before the game ends.
  - rounds_with_redeals: the game plays multiple rounds with a fresh deal each.
  - automa_or_solo_logic: a scripted non-player force resolves deterministically.
  - variable_player_powers: seats begin with asymmetric roles/powers/abilities.
- zones: every place a component can live — hands, decks, discard piles, board
  spaces, tracks, markets, open supplies. For each, state its visibility
  (hidden/public, per-player/shared) and whether its contents are conserved or
  created/destroyed. Boards, maps, tracks, and unlimited banks live here: an
  unlimited supply (e.g. "clue tokens drawn from the bank") is a zones note, not
  a component.
- actions: every decision a player can ever make, each with its complete
  resolution rules including edge cases (no valid target, empty deck, ties...).
  Name them in snake_case. Forced bookkeeping (mandatory draws, refills) is NOT
  an action — the engine auto-advances through it.
- decision_flow: who acts when. Be explicit about simultaneous decisions and
  reaction windows if the game has them.
  - Reaction windows: if the game has "may respond"/interrupt windows, the
    engine implements each window as its own explicit turn with a pass action —
    never as an interrupt. Describe every window in decision_flow as its own
    discrete decision point: who may act, in what order, and what declining
    means. Give each reaction its own entry in actions, including the explicit
    decline/pass.
  - Multi-stage turns: if one turn is several dependent choices (move then
    build; choose a card then its target; bid then amount), decompose it in
    decision_flow and actions into sequential decision points, one per choice.
- state_shape: design the canonical JSON state dict the engine will use. List
  every key and its type, including per-player sub-dicts. Keep it flat and
  obvious. Cards/components are stored by their string name. Include the keys
  "rng_seed", "game_over" and "winners". Include a "phase" key whenever
  decision_flow has more than one kind of decision point, and record any
  in-progress staged choice (the pending selection) in state so the engine can
  resume mid-turn.
- max_decisions: a generous ceiling on the total number of player decisions in
  one game — about 10x the decision count of a long real game, counting every
  staged sub-decision and every reaction window. The runtime uses it as a
  per-game step budget, so err high.
- ambiguities: anything the rulebook leaves unclear that affects implementation.
  Pick a sensible resolution for each and quote the passage that motivated it.
  The engine will implement your resolution.
- scoring: include every tiebreaker, fully specified.
"""


def _digest_json_schema() -> dict:
    """The strict JSON schema for digest generation.

    ``mechanics``/``zones``/``max_decisions`` carry load-time defaults so that
    pre-existing digest.json files still parse, but the generator must always
    emit them — so force every property into ``required`` for the model.
    """
    schema = GameDigest.model_json_schema()
    schema["required"] = list(schema["properties"])
    return schema


def generate_digest(
    client: LLMClient, rulebook_text: str, feedback: str | None = None
) -> GameDigest:
    """One structured-output call: the rulebook in, the digest out.

    ``feedback``, when set, is appended as an extra user message so a caller can
    re-derive the digest after the engine exhausted its decision budget.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"RULEBOOK:\n\n{rulebook_text}"},
    ]
    if feedback:
        messages.append({"role": "user", "content": f"REVISION FEEDBACK:\n\n{feedback}"})
    raw = client.complete(
        messages,
        role="digest",
        json_schema={"name": "game_digest", "strict": True, "schema": _digest_json_schema()},
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
        f"**Decision budget:** {digest.max_decisions}",
        "",
        "## Mechanics",
        "",
        (", ".join(digest.mechanics) if digest.mechanics else "None tagged."),
        "",
        "## Components",
        "",
        *(f"- {component.name} × {component.count}" for component in digest.components),
        "",
        "## Zones",
        "",
        (digest.zones.strip() or "None described."),
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
