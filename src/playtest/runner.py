"""Run a full playtest session, driving the observer and logger from graph events."""

import json
import uuid
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

from playtest.config import get_settings
from playtest.graph.build import build_playtest_graph
from playtest.ingestion.schemas import GameConfig
from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry
from playtest.ui.logger import GameLogger
from playtest.ui.observer import GameObserver

_RECURSION_LIMIT = 500

# PlaytestState fields with overwrite semantics — every one must be copied into the
# reconstructed final_state, or the return value diverges from what the graph computed.
_OVERWRITE_FIELDS = (
    "game_state",
    "current_player",
    "turn_phase",
    "turn_index",
    "phase",
    "winner",
    "pending_context",
    "retry_count",
    "proposed_action",
)
_APPEND_FIELDS = ("public_transcript", "gm_log", "error_log")


def run_game(
    game_config_name: str,
    num_players: int = 2,
    seed: int | None = None,
    log_file: str | None = None,
    verbose: bool = False,
    archetypes: list[str] | None = None,
) -> dict:
    console = Console()
    settings = get_settings()

    config_dir = Path(settings.game_configs_dir) / game_config_name
    game_config = GameConfig.load(str(config_dir))
    client = OpenAI(api_key=settings.openai_api_key)
    state_manager = GameStateManager()
    tool_registry = ToolRegistry(game_config, state_manager)
    graph = build_playtest_graph(
        game_config,
        client,
        state_manager,
        tool_registry,
        num_players=num_players,
        seed=seed,
        archetypes=archetypes,
    )

    initial_state: dict = {
        "game_config_id": game_config_name,
        "session_id": str(uuid.uuid4()),
        "game_state": {},
        "current_player": "player_1",
        "turn_phase": "draw",
        "turn_index": 0,
        "proposed_action": None,
        "pending_context": None,
        "public_transcript": [],
        "gm_log": [],
        "error_log": [],
        "phase": "initializing",
        "winner": None,
        "retry_count": 0,
    }

    observer = GameObserver(console=console, verbose=verbose)
    logger = GameLogger()
    logger.set_run_metadata(archetypes=archetypes or ["default"] * num_players)
    final_state = _play(graph, initial_state, observer, logger, seed=seed)
    logger.set_run_metadata(rule_queries=tool_registry.get_rulebook_query_log())

    if log_file:
        logger.save(log_file)
        console.print(f"[green]Game log written to[/green] {log_file}")

    summary = logger.get_summary()
    console.print(_summary_panel(summary))
    return {"final_state": final_state, "summary": summary}


def _play(
    graph: object,
    initial_state: dict,
    observer: GameObserver,
    logger: GameLogger,
    *,
    seed: int | None,
) -> dict:
    """Stream graph deltas, dispatch to observer/logger, and rebuild final_state."""
    tracker = dict(initial_state)
    ctx = {"session_id": initial_state["session_id"], "seed": seed}
    for chunk in graph.stream(  # type: ignore[attr-defined]
        initial_state, stream_mode="updates", config={"recursion_limit": _RECURSION_LIMIT}
    ):
        for node, update in chunk.items():
            _dispatch(node, update, observer, logger, tracker, ctx)
            _fold(tracker, update)
    return tracker


def _fold(tracker: dict, update: dict) -> None:
    for field in _OVERWRITE_FIELDS:
        if field in update:
            tracker[field] = update[field]
    for field in _APPEND_FIELDS:
        if field in update:
            tracker[field] = tracker.get(field, []) + update[field]


def _dispatch(
    node: str,
    update: dict,
    observer: GameObserver,
    logger: GameLogger,
    tracker: dict,
    ctx: dict,
) -> None:
    if update.get("phase") == "error":
        for err in update.get("error_log", []):
            observer.console.print(f"[red]Error ({err.get('stage')}):[/red] {err.get('error')}")
        logger.log_event(
            "game_end",
            {"winner": None, "error": [e.get("error") for e in update.get("error_log", [])]},
        )
        return

    if node == "player":
        _dispatch_player(update, observer, logger, tracker)
    elif node == "gm":
        _dispatch_gm(update, observer, logger, tracker, ctx)


def _dispatch_player(
    update: dict, observer: GameObserver, logger: GameLogger, tracker: dict
) -> None:
    proposed = update.get("proposed_action") or {}
    for entry in update.get("public_transcript", []):
        if "action_type" in entry:
            player = entry["player"]
            turn_index = entry.get("turn", update.get("turn_index", 0))
            # turn_phase / round_number come from the tracker's most-recent GM-committed
            # state; the player node never writes game_state or turn_phase.
            phase = tracker.get("turn_phase", "")
            action = proposed or entry
            observer.on_turn_start(player, turn_index, phase)
            observer.on_player_action(player, action)
            logger.log_event(
                "turn_start", {"player": player, "turn_index": turn_index, "phase": phase}
            )
            logger.log_event(
                "player_action",
                {
                    "player": player,
                    "action_type": entry["action_type"],
                    "parameters": proposed.get("parameters", {}),
                    "reasoning": proposed.get("reasoning"),
                    "public_statement": entry.get("public_statement"),
                },
            )
        else:
            _unknown(observer, entry)


def _dispatch_gm(
    update: dict, observer: GameObserver, logger: GameLogger, tracker: dict, ctx: dict
) -> None:
    errors = update.get("error_log", [])
    valid_log = next((e for e in update.get("gm_log", []) if e.get("is_valid")), None)

    for entry in update.get("gm_log", []):
        is_valid = entry.get("is_valid")
        logger.log_event(
            "gm_validation",
            {
                "player": entry.get("player"),
                "is_valid": is_valid,
                "error_message": (errors[0].get("error") if (not is_valid and errors) else None),
                "action_summary": entry.get("action_summary"),
            },
        )
    for err in errors:
        observer.on_action_rejected(err.get("player", ""), err.get("error", ""))

    game_state = update.get("game_state", {})
    for entry in update.get("public_transcript", []):
        event = entry.get("event")
        if event == "game_start":
            narration = entry.get("narration", "")
            observer.on_game_start(game_state, narration)
            logger.log_event(
                "game_start",
                {
                    "session_id": ctx["session_id"],
                    "seed": ctx["seed"],
                    "state": game_state,
                    "narration": narration,
                },
            )
        elif event == "round_end":
            scores = {pid: p.get("tokens", 0) for pid, p in game_state.get("players", {}).items()}
            # Report the round that actually ended and its real winner(s), not the next
            # round's number / starting player carried on the state delta.
            round_number = entry.get("round_number", game_state.get("round_number", 0))
            winner = ", ".join(entry.get("winners") or [])
            observer.on_round_end(round_number, winner, scores)
            logger.log_event(
                "round_end",
                {
                    "round_number": round_number,
                    "winner": winner,
                    "scores": scores,
                    "winning_card": entry.get("winning_card"),
                    "winners": entry.get("winners"),
                },
            )
        elif "narration" in entry and "player" in entry:
            resolution = {
                "narration": entry["narration"],
                "action_summary": valid_log.get("action_summary") if valid_log else None,
                "gm_reasoning": valid_log.get("reasoning") if valid_log else None,
            }
            observer.on_gm_resolution(resolution)
            observer.on_state_update(game_state)
            logger.log_event(
                "gm_resolution",
                {
                    "player": entry["player"],
                    "narration": entry["narration"],
                    "action_summary": resolution["action_summary"],
                    "gm_reasoning": resolution["gm_reasoning"],
                    "state_snapshot": game_state,
                },
            )
        else:
            _unknown(observer, entry)

    if update.get("phase") == "game_over":
        scores = {pid: p.get("tokens", 0) for pid, p in game_state.get("players", {}).items()}
        observer.on_game_end(update.get("winner", ""), scores)
        logger.log_event(
            "game_end",
            {
                "winner": update.get("winner"),
                "total_turns": tracker.get("turn_index", 0),
                "rounds_played": game_state.get("round_number", 0),
                "final_scores": scores,
            },
        )


def _unknown(observer: GameObserver, entry: dict) -> None:
    observer.console.print(f"[dim][unknown event] {entry}[/dim]")


def _summary_panel(summary: dict) -> Panel:
    rejected = summary["actions_rejected"]["count"]
    played = ", ".join(f"{a}×{n}" for a, n in summary["most_played_actions"].items())
    return Panel(
        f"[bold green]Winner:[/bold green] {summary['winner']}\n"
        f"[bold]Rounds:[/bold] {summary['rounds_played']}    "
        f"[bold]Turns:[/bold] {summary['total_turns']}    "
        f"[bold]Rejections:[/bold] {rejected}\n"
        f"[bold]Actions:[/bold] {played or 'none'}",
        title="Game Summary",
        border_style="green",
    )


def run_multiple_games(
    game_config_name: str,
    num_games: int,
    num_players: int,
    archetypes: list[str] | None = None,
    seed_start: int = 0,
    output_dir: str = "results",
) -> dict:
    """Run ``num_games`` playtests and collect aggregate results.

    Each game uses ``seed = seed_start + game_index`` for reproducibility and is saved to
    ``output_dir/game_NNN.json``. A crashing game is skipped (its error written to
    ``game_NNN_error.json``) so the batch always completes. Returns aggregate analytics.
    """
    from playtest.analytics import analyze_games

    console = Console()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(num_games):
        seed = seed_start + i
        log_path = out / f"game_{i + 1:03d}.json"
        console.print(f"[cyan]Game {i + 1}/{num_games}[/cyan] (seed={seed})")
        try:
            run_game(
                game_config_name,
                num_players=num_players,
                seed=seed,
                log_file=str(log_path),
                archetypes=archetypes,
            )
        except Exception as exc:  # noqa: BLE001 - skip-and-continue so one bad game never aborts the batch
            err_path = out / f"game_{i + 1:03d}_error.json"
            err_path.write_text(
                json.dumps({"seed": seed, "error": f"{type(exc).__name__}: {exc}"}, indent=2),
                encoding="utf-8",
            )
            console.print(f"[red]Game {i + 1} failed:[/red] {exc} (logged to {err_path})")

    return analyze_games(str(out))
