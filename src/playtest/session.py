"""Plain turn-loop driver — the orchestration that replaced the LangGraph graph.

A single function, :func:`run_session`, drives a whole playtest: initialize, then loop
turns until the game ends or a safety cap/​crash stops it. The GM resolves each player
intent and commits state; the driver owns turn order, retry bookkeeping, and all
observer/logger emission. Round and game end are the GM's judgment, reported through
``finish_resolution`` flags.

Clean rejections get bounded retry: an illegal proposal is fed back to the player (a
playtest signal worth recording), and only exhausting ``max_action_retries`` within a
turn crashes with ``IllegalAction``. Integrity problems — corrupt committed state,
is_valid/commit inconsistencies — still crash immediately (see :mod:`playtest.errors`).
"""

import json
from collections.abc import Callable

from playtest.agents.gm import GMAgent
from playtest.agents.player import PlayerAction, PlayerAgent
from playtest.config import get_settings
from playtest.errors import IllegalAction, PlaytestError
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
    """Build the per-turn callback the player uses to resolve each intent via the GM.

    The resolver is created per turn, so the rejection counter resets each turn for free.
    """
    settings = get_settings()
    rejections = 0

    def resolve_action(action: PlayerAction) -> dict:
        nonlocal rejections
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

        # Integrity crashes (StateInvariantViolation / PlaytestError) propagate.
        resolution = gm_agent.validate_and_resolve(action_dict, player_id)

        if not resolution.is_valid:
            rejections += 1
            error = resolution.error_message or "no reason given"
            logger.log_event(
                "gm_validation",
                {
                    "player": player_id,
                    "is_valid": False,
                    "action_type": action.action_type,
                    "error_message": error,
                },
            )
            observer.on_action_rejected(player_id, error)
            if rejections > settings.max_action_retries:
                raise IllegalAction(player_id, action_dict, error)
            return {"rejected": True, "error_message": error}

        new_state = resolution.new_state
        if new_state is None:
            raise PlaytestError(
                f"GM returned a valid resolution for {player_id} without a committed state"
            )

        logger.log_event(
            "gm_validation",
            {
                "player": player_id,
                "is_valid": True,
                "action_type": action.action_type,
                "action_summary": resolution.action_summary,
            },
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
        last["round_ended"] = last["round_ended"] or resolution.round_ended
        last["game_ended"] = last["game_ended"] or resolution.game_ended
        if resolution.winners or resolution.winner:
            last["winners"] = resolution.winners or [resolution.winner]

        return {
            "filtered_state": state_manager.get_state(player_id),
            "narration": resolution.narration,
            "private_info": resolution.private_info,
            "turn_ended": gm_agent.rules.is_turn_over(action_dict, resolution.turn_ended),
        }

    return resolve_action


def _scores(state: dict, score_field: str | None) -> dict:
    if not score_field:
        return {}
    return {pid: p.get(score_field, 0) for pid, p in state.get("players", {}).items()}


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
    spec = gm_agent.spec
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

        last: dict = {
            "narration": None,
            "round_ended": False,
            "game_ended": False,
            "winners": None,
        }
        resolve_action = _make_resolver(
            gm_agent, state_manager, observer, logger, private_memory, current, last
        )
        player_agents[current].take_turn(
            state_manager.get_state(current),
            context=pending_context,
            private_memory=private_memory[current],
            resolve_action=resolve_action,
        )

        # End-of-round/game is the GM's judgment, reported on the turn's resolutions and
        # acted on only here (after the turn completed) — never mid-turn.
        if last["game_ended"]:
            winners = last["winners"] or []
            winner = ",".join(winners) if winners else None
            finished = True
            break

        if last["round_ended"] and spec.has_rounds:
            committed = state_manager.get_state("gm")
            round_number = committed.get("round_number", 0)
            rr = gm_agent.handle_round_end()
            scores = _scores(state_manager.get_state("gm"), spec.score_field)
            winners_label = ", ".join(rr.winners or [])
            observer.on_round_end(round_number, winners_label, scores)
            logger.log_event(
                "round_end",
                {
                    "round_number": round_number,
                    "winner": winners_label,
                    "scores": scores,
                    "winners": rr.winners,
                },
            )
            if rr.game_ended:
                winner = rr.winner
                finished = True
                break
            # handle_round_end committed the next round's deal (and its first player).
            pending_context = rr.narration
            continue

        # Normal turn end: the generic rules rotate to the next active player.
        advanced = gm_agent.rules.advance_turn(state_manager.get_state("gm"), current)
        state_manager.set_state(advanced)
        pending_context = last["narration"]

    if not finished:
        raise PlaytestError(f"game did not finish within {settings.max_turns} turns")

    final_state = state_manager.get_state("gm")
    scores = _scores(final_state, spec.score_field)
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
