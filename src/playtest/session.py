"""Plain turn-loop driver — the orchestration that replaced the LangGraph graph.

A single function, :func:`run_session`, drives a whole playtest: initialize, then loop
turns until the game ends or a safety cap/​crash stops it. The GM resolves each player
intent and commits state; the driver owns turn order, round-over detection, and all
observer/logger emission.

Crash early, don't reconcile: an illegal action or a corrupt committed state raises
(see :mod:`playtest.errors`) and aborts the run. The driver is authoritative for whose
turn it is and the next phase — it never trusts a GM-reported ``next_player`` for routing.
"""

import json
from collections.abc import Callable

from playtest.agents.gm import GMAgent
from playtest.agents.player import PlayerAction, PlayerAgent
from playtest.config import get_settings
from playtest.errors import PlaytestError
from playtest.state.manager import GameStateManager
from playtest.ui.logger import GameLogger
from playtest.ui.observer import GameObserver


def _make_resolver(
    gm_agent: GMAgent,
    state_manager: GameStateManager,
    observer: GameObserver,
    logger: GameLogger,
    private_memory: dict[str, list[str]],
    player_id: str,
    last: dict,
) -> Callable[[PlayerAction], dict]:
    """Build the per-turn callback the player uses to resolve each intent via the GM."""

    def resolve_action(action: PlayerAction) -> dict:
        action_dict = action.model_dump()
        observer.on_player_action(player_id, action_dict)
        logger.log_event(
            "player_action",
            {
                "player": player_id,
                "action_type": action.action_type,
                "parameters": action.parameters,
                "reasoning": action.reasoning,
                "public_statement": action.public_statement,
            },
        )

        # Crashes (IllegalAction / StateInvariantViolation / PlaytestError) propagate.
        resolution = gm_agent.validate_and_resolve(action_dict, player_id)
        new_state = resolution.new_state
        assert new_state is not None  # a valid resolution always carries a committed state

        logger.log_event(
            "gm_validation",
            {"player": player_id, "is_valid": True, "action_summary": resolution.action_summary},
        )
        observer.on_gm_resolution(
            {
                "narration": resolution.narration,
                "action_summary": resolution.action_summary,
                "gm_reasoning": resolution.gm_reasoning,
            }
        )
        observer.on_state_update(new_state)
        logger.log_event(
            "gm_resolution",
            {
                "player": player_id,
                "narration": resolution.narration,
                "action_summary": resolution.action_summary,
                "gm_reasoning": resolution.gm_reasoning,
                "state_snapshot": new_state,
            },
        )

        if resolution.private_info:
            private_memory[player_id].append(json.dumps(resolution.private_info))
        last["narration"] = resolution.narration

        return {
            "filtered_state": state_manager.get_state(player_id),
            "narration": resolution.narration,
            "private_info": resolution.private_info,
            "turn_ended": gm_agent.rules.is_turn_over(action_dict),
        }

    return resolve_action


def run_session(
    gm_agent: GMAgent,
    player_agents: dict[str, PlayerAgent],
    state_manager: GameStateManager,
    observer: GameObserver,
    logger: GameLogger,
    *,
    num_players: int,
    seed: int | None,
    session_id: str,
) -> dict:
    """Drive a full playtest and return ``{winner, final_state, total_turns}``."""
    settings = get_settings()
    private_memory: dict[str, list[str]] = {pid: [] for pid in player_agents}

    init = gm_agent.initialize_game(num_players, seed)
    gm_view = init["state"]
    narration = init["narration"]
    observer.on_game_start(gm_view, narration)
    logger.log_event(
        "game_start",
        {"session_id": session_id, "seed": seed, "state": gm_view, "narration": narration},
    )

    winner: str | None = None
    turn_index = 0
    pending_context: str | None = narration
    finished = False

    for _ in range(settings.max_turns):
        committed = state_manager.get_state("gm")
        current = committed["current_turn"]
        turn_index += 1
        observer.on_turn_start(current, turn_index, committed["turn_phase"])
        logger.log_event(
            "turn_start",
            {"player": current, "turn_index": turn_index, "phase": committed["turn_phase"]},
        )

        last: dict = {"narration": None}
        resolve_action = _make_resolver(
            gm_agent, state_manager, observer, logger, private_memory, current, last
        )
        player_agents[current].take_turn(
            state_manager.get_state(current),
            context=pending_context,
            private_memory=private_memory[current],
            resolve_action=resolve_action,
        )

        # Round-over is decided by the rules module from committed state, only here
        # (after a turn completed) — never mid-turn.
        committed = state_manager.get_state("gm")
        if gm_agent.rules.is_round_over(committed):
            round_number = committed["round_number"]
            rr = gm_agent.handle_round_end()
            scores = {
                pid: p.get("tokens", 0)
                for pid, p in state_manager.get_state("gm")["players"].items()
            }
            winners_label = ", ".join(rr.winners or [])
            observer.on_round_end(round_number, winners_label, scores)
            logger.log_event(
                "round_end",
                {
                    "round_number": round_number,
                    "winner": winners_label,
                    "scores": scores,
                    "winning_card": rr.winning_card,
                    "winners": rr.winners,
                },
            )
            if rr.game_ended:
                winner = rr.winner
                finished = True
                break
            # handle_round_end dealt the next round (winner takes the first turn).
            pending_context = rr.narration
            continue

        # Normal turn end: the rules module decides the next actor/phase (authoritative,
        # not the GM). The driver commits that transition.
        advanced = gm_agent.rules.advance_turn(state_manager.get_state("gm"), current)
        state_manager.set_state(advanced)
        pending_context = last["narration"]

    if not finished:
        raise PlaytestError(f"game did not finish within {settings.max_turns} turns")

    final_state = state_manager.get_state("gm")
    scores = {pid: p.get("tokens", 0) for pid, p in final_state["players"].items()}
    observer.on_game_end(winner or "", scores)
    logger.log_event(
        "game_end",
        {
            "winner": winner,
            "total_turns": turn_index,
            "rounds_played": final_state.get("round_number", 0),
            "final_scores": scores,
        },
    )
    return {"winner": winner, "final_state": final_state, "total_turns": turn_index}
