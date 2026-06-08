"""Graph state for the playtest orchestration.

The manager's ``game_state`` (written by the GM via set_game_state) is the source of
truth for ``current_turn``/``turn_phase``; the top-level ``current_player``/``turn_phase``
fields here are for routing and logging and are always derived from the committed state.
"""

import operator
from typing import Annotated, TypedDict

MAX_RETRIES = 3


class PlaytestState(TypedDict):
    # Game identity
    game_config_id: str
    session_id: str

    # Game state (full replacement on every update)
    game_state: dict

    # Turn management
    current_player: str
    turn_phase: str
    turn_index: int

    # Action flow
    proposed_action: dict | None

    # Channel the gm_node uses to hand the next user-message to the player_node:
    # GM narration on a normal turn, or the rejection error on a retry.
    pending_context: str | None

    # Logs (append-only)
    public_transcript: Annotated[list[dict], operator.add]
    gm_log: Annotated[list[dict], operator.add]
    error_log: Annotated[list[dict], operator.add]

    # Game flow
    phase: str  # "initializing", "playing", "game_over", "error"
    winner: str | None
    retry_count: int
