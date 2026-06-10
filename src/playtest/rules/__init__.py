"""Per-game rules modules and the registry that resolves one for a game config.

When a game has a deterministic rules module it is used (setup, invariants, turn order,
scoring) — giving the crash-early safety net. A game with no module falls through to the
generic LLM-driven path (Stage 4), where the GM answers those questions from the rulebook.
"""

from playtest.rules.base import GameRules
from playtest.rules.love_letter import LoveLetterRules

__all__ = ["GameRules", "LoveLetterRules", "get_rules"]


def get_rules(game_config: object) -> GameRules:
    """Resolve the rules module for a game config (by game name for now)."""
    name = (getattr(game_config, "game_name", "") or "").lower()
    if "love letter" in name:
        return LoveLetterRules()
    raise NotImplementedError(
        f"No deterministic rules module for game "
        f"{getattr(game_config, 'game_name', '?')!r}; the generic LLM-driven path is not "
        "yet implemented (Stage 4)."
    )
