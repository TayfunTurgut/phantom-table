"""Run a full playtest session by streaming the LangGraph orchestration."""

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

_RECURSION_LIMIT = 500


def run_game(
    game_config_name: str,
    num_players: int = 2,
    seed: int | None = None,
    log_file: str | None = None,
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

    console.print(
        Panel(
            f"[bold]Game:[/bold] {game_config.game_name} ({game_config.variant})\n"
            f"[bold]Players:[/bold] {num_players}\n"
            f"[bold]Seed:[/bold] {seed if seed is not None else 'random'}",
            title="Playtest Session",
            border_style="cyan",
        )
    )

    # stream_mode="values" yields the full state after each superstep, so the last chunk
    # is the authoritative final state for the summary and log file. (Don't switch to
    # "updates" — it yields per-node deltas and would require reassembling final state.)
    final_state: dict = initial_state
    printed = 0
    for chunk in graph.stream(
        initial_state, stream_mode="values", config={"recursion_limit": _RECURSION_LIMIT}
    ):
        final_state = chunk
        transcript = chunk.get("public_transcript", [])
        for entry in transcript[printed:]:  # append-only reducer -> safe to slice by length
            _print_entry(console, entry)
        printed = len(transcript)

    _print_summary(console, final_state)

    if log_file:
        Path(log_file).write_text(json.dumps(final_state, indent=2), encoding="utf-8")
        console.print(f"[green]Game log written to[/green] {log_file}")

    winner = final_state.get("winner")
    players = final_state.get("game_state", {}).get("players", {})
    return {
        "final_state": final_state,
        "summary": {
            "winner": winner,
            "phase": final_state.get("phase"),
            "turns": final_state.get("turn_index"),
            "scores": {pid: p.get("tokens") for pid, p in players.items()},
        },
    }


def _print_entry(console: Console, entry: dict) -> None:
    if "event" in entry:
        console.print(f"[magenta]\\[{entry['event']}][/magenta] {entry.get('narration', '')}")
    elif "action_type" in entry:
        console.print(
            f"[cyan]{entry.get('player')}[/cyan] (turn {entry.get('turn')}): "
            f'[bold]{entry["action_type"]}[/bold] — "{entry.get("public_statement", "")}"'
        )
    elif "narration" in entry:
        console.print(f"[yellow]GM:[/yellow] {entry['narration']}")


def _print_summary(console: Console, final_state: dict) -> None:
    players = final_state.get("game_state", {}).get("players", {})
    scores = "\n".join(
        f"  [bold]{pid}[/bold]: {p.get('tokens', 0)} tokens" for pid, p in sorted(players.items())
    )
    phase = final_state.get("phase")
    if phase == "error":
        console.print(
            Panel(
                f"[bold red]Game ended with an error[/bold red]\n{final_state.get('error_log')}",
                title="Playtest Result",
                border_style="red",
            )
        )
        return
    console.print(
        Panel(
            f"[bold green]Winner:[/bold green] {final_state.get('winner')}\n"
            f"[bold]Turns:[/bold] {final_state.get('turn_index')}\n"
            f"[bold]Final scores:[/bold]\n{scores}",
            title="Playtest Result",
            border_style="green",
        )
    )
