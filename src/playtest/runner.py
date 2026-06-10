"""Run a full playtest session via the plain turn-loop driver (playtest.session)."""

import json
import uuid
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

from playtest.agents.gm import GMAgent
from playtest.agents.player import PlayerAgent
from playtest.config import get_settings, maybe_wrap_openai
from playtest.ingestion.schemas import GameConfig
from playtest.session import run_session
from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry
from playtest.ui.logger import GameLogger
from playtest.ui.observer import GameObserver


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
    client = maybe_wrap_openai(OpenAI(api_key=settings.openai_api_key))
    state_manager = GameStateManager()
    tool_registry = ToolRegistry(game_config, state_manager)

    if archetypes is None:
        archetypes = ["default"] * num_players
    if len(archetypes) != num_players:
        raise ValueError(
            f"expected {num_players} archetypes (one per player), got {len(archetypes)}"
        )

    gm_agent = GMAgent(game_config, tool_registry, client)
    player_agents = {
        f"player_{i}": PlayerAgent(
            f"player_{i}", game_config, tool_registry, client, archetype=archetypes[i - 1]
        )
        for i in range(1, num_players + 1)
    }

    session_id = str(uuid.uuid4())
    observer = GameObserver(console=console, verbose=verbose)
    logger = GameLogger()
    logger.set_run_metadata(archetypes=archetypes)

    # Crash early: any IllegalAction / StateInvariantViolation / PlaytestError propagates.
    # The finally block persists whatever was logged so a crashed run is still inspectable.
    try:
        result = run_session(
            gm_agent,
            player_agents,
            state_manager,
            observer,
            logger,
            num_players=num_players,
            seed=seed,
            session_id=session_id,
        )
    finally:
        logger.set_run_metadata(rule_queries=tool_registry.get_rulebook_query_log())
        if log_file:
            logger.save(log_file)
            console.print(f"[green]Game log written to[/green] {log_file}")

    summary = logger.get_summary()
    console.print(_summary_panel(summary))
    return {"final_state": result["final_state"], "summary": summary}


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
