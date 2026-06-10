"""Generic rules engine: deterministic primitives configured by the ingested GameSpec.

There are no per-game rules modules — all game logic is inferred from the rulebook at
ingestion, and :class:`GameRules` only executes generic primitives (seeded setup,
phase-based action exposure, turn rotation, conservation invariants) parameterized by
that extraction. Judgment calls (legality, effects, end conditions, scoring) are the
GM agent's, grounded by the rulebook.
"""

from playtest.rules.generic import GameRules

__all__ = ["GameRules"]
