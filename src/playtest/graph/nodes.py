"""LangGraph nodes for the playtest orchestration.

Routing is dynamic via ``Command(goto=...)`` — there are no fixed edges between the
GM and player nodes. The node builders take agent *instances* (not a raw client) so
the orchestration can be exercised with stub agents and zero API calls.

The manager's committed ``game_state`` is authoritative for ``current_turn``/``turn_phase``
(the player reads them inside ``take_turn``). Graph-level ``current_player``/``turn_phase``
are for routing/logging and are always derived from the committed ``new_state`` — never
from ``GMResolution.next_phase`` — so the two cannot drift in the logs.
"""

from collections.abc import Callable
from typing import Literal

from langgraph.types import Command

from playtest.agents.gm import GMAgent
from playtest.agents.player import PlayerAgent
from playtest.graph.schema import MAX_RETRIES, PlaytestState
from playtest.state.manager import GameStateManager

_END: Literal["__end__"] = "__end__"


def next_active_player(state_manager: GameStateManager, current_player: str) -> str:
    """Next non-eliminated player after ``current_player`` in sorted id order."""
    players = state_manager.get_state("gm").get("players", {})
    order = sorted(players)
    if not order:
        return current_player
    start = order.index(current_player) + 1 if current_player in order else 0
    for offset in range(len(order)):
        pid = order[(start + offset) % len(order)]
        if not players[pid].get("is_eliminated", False):
            return pid
    return current_player


def build_gm_node(
    gm_agent: GMAgent,
    state_manager: GameStateManager,
    num_players: int,
    seed: int | None,
) -> Callable[[PlaytestState], Command]:
    def gm_node(state: PlaytestState) -> Command[Literal["player", "__end__"]]:
        try:
            if state["phase"] == "initializing":
                res = gm_agent.initialize_game(num_players, seed)
                gm_view = res["state"]
                return Command(
                    goto="player",
                    update={
                        "game_state": gm_view,
                        "current_player": gm_view["current_turn"],
                        "turn_phase": gm_view["turn_phase"],
                        "phase": "playing",
                        "pending_context": res["narration"],
                        "turn_index": 0,
                        "public_transcript": [
                            {"event": "game_start", "narration": res["narration"]}
                        ],
                    },
                )

            current = state["current_player"]
            proposed = state["proposed_action"]
            assert proposed is not None  # set by the player node before routing here
            r = gm_agent.validate_and_resolve(proposed, current)
            gm_log = [
                {
                    "player": current,
                    "is_valid": r.is_valid,
                    "action_summary": r.action_summary,
                    "reasoning": r.gm_reasoning,
                }
            ]

            if not r.is_valid:
                if state["retry_count"] < MAX_RETRIES:
                    return Command(
                        goto="player",
                        update={
                            "proposed_action": None,
                            "retry_count": state["retry_count"] + 1,
                            "pending_context": r.error_message,
                            "gm_log": gm_log,
                            "error_log": [
                                {
                                    "player": current,
                                    "attempt": state["retry_count"] + 1,
                                    "error": r.error_message,
                                }
                            ],
                        },
                    )
                # Retries exhausted: skip this player's turn.
                nxt = r.next_player or next_active_player(state_manager, current)
                return Command(
                    goto="player",
                    update={
                        "proposed_action": None,
                        "retry_count": 0,
                        "current_player": nxt,
                        "turn_phase": state_manager.get_state("gm")["turn_phase"],
                        "pending_context": (
                            "Your turn was skipped after repeated invalid actions."
                        ),
                        "gm_log": gm_log,
                        "error_log": [
                            {"player": current, "error": "retries exhausted; turn skipped"}
                        ],
                    },
                )

            # Valid action: the GM already committed new_state to the manager.
            assert r.new_state is not None  # is_valid implies a committed state
            transcript = [{"player": current, "narration": r.narration}]

            if r.game_ended:
                return Command(
                    goto=_END,
                    update={
                        "phase": "game_over",
                        "winner": r.winner,
                        "game_state": r.new_state,
                        "proposed_action": None,
                        "retry_count": 0,
                        "gm_log": gm_log,
                        "public_transcript": transcript,
                    },
                )

            if r.round_ended:
                rr = gm_agent.handle_round_end()
                assert rr.new_state is not None  # round end commits the next round's state
                transcript.append({"event": "round_end", "narration": rr.narration})
                if rr.game_ended:
                    return Command(
                        goto=_END,
                        update={
                            "phase": "game_over",
                            "winner": rr.winner,
                            "game_state": rr.new_state,
                            "proposed_action": None,
                            "retry_count": 0,
                            "gm_log": gm_log,
                            "public_transcript": transcript,
                        },
                    )
                nxt = rr.next_player or next_active_player(state_manager, current)
                return Command(
                    goto="player",
                    update={
                        "game_state": rr.new_state,
                        "current_player": nxt,
                        "turn_phase": rr.new_state["turn_phase"],
                        "phase": "playing",
                        "pending_context": rr.narration,
                        "proposed_action": None,
                        "retry_count": 0,
                        "gm_log": gm_log,
                        "public_transcript": transcript,
                    },
                )

            nxt = r.next_player or next_active_player(state_manager, current)
            return Command(
                goto="player",
                update={
                    "game_state": r.new_state,
                    "current_player": nxt,
                    "turn_phase": r.new_state["turn_phase"],
                    "pending_context": r.narration,
                    "proposed_action": None,
                    "retry_count": 0,
                    "gm_log": gm_log,
                    "public_transcript": transcript,
                },
            )
        except Exception as exc:  # noqa: BLE001 - terminate the run cleanly on any agent failure
            return Command(
                goto=_END,
                update={
                    "phase": "error",
                    "error_log": [{"stage": "gm", "error": f"{type(exc).__name__}: {exc}"}],
                },
            )

    return gm_node


def build_player_node(
    player_agents: dict[str, PlayerAgent],
) -> Callable[[PlaytestState], Command]:
    def player_node(state: PlaytestState) -> Command[Literal["gm", "__end__"]]:
        try:
            player = player_agents[state["current_player"]]
            action = player.take_turn(state.get("pending_context"))
            return Command(
                goto="gm",
                update={
                    "proposed_action": action.model_dump(),
                    "turn_index": state["turn_index"] + 1,
                    "pending_context": None,
                    "public_transcript": [
                        {
                            "turn": state["turn_index"] + 1,
                            "player": state["current_player"],
                            "action_type": action.action_type,
                            "public_statement": action.public_statement,
                        }
                    ],
                },
            )
        except Exception as exc:  # noqa: BLE001 - terminate the run cleanly on any agent failure
            return Command(
                goto=_END,
                update={
                    "phase": "error",
                    "error_log": [{"stage": "player", "error": f"{type(exc).__name__}: {exc}"}],
                },
            )

    return player_node
