"""Errors that crash a playtest run.

A crash is a finding, not a recovery point: rather than reconcile or auto-correct a
broken game, the harness fails fast with a clear diagnostic so the underlying cause
(a player/GM mistake, or a gap in the rulebook) can be inspected.
"""


class PlaytestError(Exception):
    """Base for errors that should terminate a playtest run."""


class IllegalAction(PlaytestError):
    """A player exhausted the per-turn retry budget proposing illegal actions.

    Individual rejections are fed back to the player (a playtest signal, not a crash);
    this is raised only when ``max_action_retries`` is exceeded within one turn.
    """

    def __init__(self, player_id: str, action: dict, reason: str) -> None:
        self.player_id = player_id
        self.action = action
        self.reason = reason
        action_type = action.get("action_type", "?")
        super().__init__(
            f"{player_id} attempted an illegal action ({action_type}): {reason}"
        )


class StateInvariantViolation(PlaytestError):
    """A committed game state violates one or more integrity invariants."""

    def __init__(self, violations: list[str], last_action: dict | None, state: dict) -> None:
        self.violations = violations
        self.last_action = last_action
        self.state = state
        joined = "; ".join(violations)
        suffix = ""
        if last_action:
            suffix = (
                f" (after {last_action.get('player_id', '?')} "
                f"{last_action.get('action_type', '?')})"
            )
        super().__init__(f"committed state violates integrity invariants{suffix}: {joined}")
