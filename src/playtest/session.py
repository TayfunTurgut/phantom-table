"""The session driver: engine decides what's possible, agents decide what to do.

One loop: ask the engine who must act, collect one decision per acting seat (each
seat sees only its own observation, legal actions, and private event history),
submit them, route the resulting events. All game logic — legality, effects,
hidden information, round flow, scoring — lives in the engine; this loop has no
game knowledge at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from playtest.checkpoint import Checkpoint, write_checkpoint
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
    max_steps_source: str = "the configured step budget",
    checkpoint_path: str | None = None,
    game_ref: str | None = None,
    archetypes: list[str] | None = None,
    resume: Checkpoint | None = None,
) -> dict[str, Any]:
    """Run one full game. Returns final state, status, and step count.

    When ``checkpoint_path`` is set, a resumable snapshot (raw state + per-seat
    event buffers + notebooks) is written at the top of every turn, so a crash
    during a decision leaves a checkpoint for exactly the turn that failed.

    When ``resume`` is given, ``engine.setup`` is skipped and the loop re-enters
    at the checkpoint's step with its state/buffers/notebooks (the caller is
    responsible for restoring each player's notebook onto the agent).
    """
    seats = seats_for(num_players)
    if set(players) != set(seats):
        raise PlaytestError(f"players must be exactly {seats}, got {sorted(players)}")

    buffers: dict[str, list[str]]
    notebooks: dict[str, str]
    if resume is None:
        state, setup_events = engine.setup(num_players, seed)
        buffers = {seat: [] for seat in seats}
        notebooks = {seat: "" for seat in seats}
        start_step = 1
    else:
        state = resume.state
        setup_events = []
        buffers = {seat: list(resume.buffers.get(seat, [])) for seat in seats}
        notebooks = {seat: resume.notebooks.get(seat, "") for seat in seats}
        start_step = resume.step

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
        "game_resume" if resume is not None else "game_start",
        {
            "session_id": session_id,
            "seed": seed,
            "game_name": engine.game_name,
            "num_players": num_players,
            "resumed_at_step": start_step if resume is not None else None,
            "state": engine.observe(state, SPECTATOR),
        },
    )
    observer.on_game_start(engine.game_name, seats)
    emit(setup_events, step=0)

    for step in range(start_step, max_steps + 1):
        status = engine.status(state)
        if status.over:
            break

        if checkpoint_path is not None:
            write_checkpoint(
                checkpoint_path,
                Checkpoint(
                    game_ref=game_ref or "",
                    num_players=num_players,
                    seed=seed,
                    archetypes=list(archetypes) if archetypes is not None else [],
                    session_id=session_id,
                    step=step,
                    state=state,
                    buffers=buffers,
                    notebooks=notebooks,
                ),
            )

        acting = engine.to_act(state)
        observer.on_step_start(step, acting)

        decisions: list[Decision] = []
        for seat in acting:
            observation = engine.observe(state, seat)
            legal = engine.legal_actions(state, seat)
            decision = players[seat].choose(observation, legal, buffers[seat])
            buffers[seat].clear()
            notebooks[seat] = decision.notes
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
        raise PlaytestError(
            f"session exceeded max_steps={max_steps} (budget source: {max_steps_source}) "
            "without the game ending"
        )

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
