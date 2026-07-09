"""The session driver: engine decides what's possible, agents decide what to do.

One loop: ask the engine who must act, collect one decision per acting seat (each
seat sees only its own observation, legal actions, and private event history),
submit them, route the resulting events. All game logic — legality, effects,
hidden information, round flow, scoring — lives in the engine; this loop has no
game knowledge at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from playtest.engine import SPECTATOR, GameEngine, seats_for
from playtest.errors import EngineCrash, PlaytestError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from playtest.agents.player import Decision
    from playtest.engine import Action, Event
    from playtest.ui.logger import GameLogger


class PlayerLike(Protocol):
    """What the session needs from a player: one decision per prompt."""

    def choose(self, observation: dict, legal: list[Action], events: list[str]) -> Decision: ...


class SessionObserver(Protocol):
    """Progress callbacks; ``GameObserver`` is the terminal implementation."""

    def on_game_start(self, game_name: str, seats: list[str]) -> None: ...

    def on_step_start(self, step: int, acting: list[str]) -> None: ...

    def on_decision(self, seat: str, decision: Decision) -> None: ...

    def on_events(self, texts: list[str]) -> None: ...

    def on_game_end(self, winners: list[str], scores: dict) -> None: ...


def run_session(
    engine: GameEngine,
    players: Mapping[str, PlayerLike],
    observer: SessionObserver,
    logger: GameLogger,
    num_players: int,
    seed: int,
    session_id: str,
    max_steps: int = 1000,
) -> dict[str, Any]:
    """Run one full game. Returns final state, status, and step count."""
    seats = seats_for(num_players)
    if set(players) != set(seats):
        raise PlaytestError(f"players must be exactly {seats}, got {sorted(players)}")

    state, setup_events = engine.setup(num_players, seed)
    buffers: dict[str, list[str]] = {seat: [] for seat in seats}

    def emit(events: list[Event], step: int) -> None:
        """Route engine events to seat memory, the log, and the observer."""
        for event in events:
            audience = list(event.visible_to) if event.visible_to is not None else seats
            for seat in audience:
                buffers[seat].append(event.text)
            logger.log_event(
                "engine_event",
                {"step": step, "text": event.text, "visible_to": event.visible_to},
            )
        observer.on_events([e.text for e in events if e.visible_to is None])

    logger.log_event(
        "game_start",
        {
            "session_id": session_id,
            "seed": seed,
            "game_name": engine.game_name,
            "num_players": num_players,
            "state": engine.observe(state, SPECTATOR),
        },
    )
    observer.on_game_start(engine.game_name, seats)
    emit(setup_events, step=0)

    for step in range(1, max_steps + 1):
        status = engine.status(state)
        if status.over:
            break
        acting = engine.to_act(state)
        observer.on_step_start(step, acting)

        decisions: list[Decision] = []
        for seat in acting:
            observation = engine.observe(state, seat)
            legal = engine.legal_actions(state, seat)
            decision = players[seat].choose(observation, legal, buffers[seat])
            buffers[seat].clear()
            decisions.append(decision)
            observer.on_decision(seat, decision)
            logger.log_event(
                "decision",
                {
                    "step": step,
                    "seat": seat,
                    "action": decision.action.name,
                    "args": decision.action.args,
                    "label": decision.action.label,
                    "reasoning": decision.reasoning,
                    "table_talk": decision.table_talk,
                    "notes": decision.notes,
                    "confused": decision.confused,
                    "num_legal_actions": len(legal),
                },
            )
            if decision.confused:
                logger.log_event("player_confusion", {"step": step, "seat": seat})

        # Broadcast table talk only after every acting seat has committed, so a
        # simultaneous decision cannot condition on a co-actor's same-step talk.
        for seat, decision in zip(acting, decisions, strict=True):
            if decision.table_talk:
                for other in seats:
                    if other != seat:
                        buffers[other].append(f'{seat} says: "{decision.table_talk}"')

        try:
            state, events = engine.apply(state, [d.action for d in decisions])
        except Exception as exc:  # an engine crash is an ingestion finding
            if isinstance(exc, PlaytestError):
                raise
            raise EngineCrash(
                exc, seed=seed, step=step, actions=[d.action for d in decisions]
            ) from exc

        emit(events, step=step)
    else:
        raise PlaytestError(f"session exceeded max_steps={max_steps} without the game ending")

    status = engine.status(state)
    logger.log_event(
        "game_end",
        {
            "winners": list(status.winners),
            "scores": status.scores,
            "total_steps": step,
            "final_state": engine.observe(state, SPECTATOR),
        },
    )
    observer.on_game_end(list(status.winners), status.scores)
    return {"final_state": state, "status": status, "total_steps": step}
