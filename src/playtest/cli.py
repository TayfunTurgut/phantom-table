import argparse

from rich.console import Console
from rich.panel import Panel

from playtest.config import get_settings


def _smoke_test() -> None:
    """Verify Claude Code is reachable with one real `claude -p` completion."""
    console = Console()
    settings = get_settings()
    console.print("[bold]Running smoke test (claude -p)...[/bold]")

    try:
        from playtest.llm import create_llm_client

        client = create_llm_client(settings)
        reply = client.complete(
            [{"role": "user", "content": "Say hello in one sentence."}],
            role="player",
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        console.print(
            Panel(
                f"[bold red]Smoke test failed[/bold red]\n\n{type(exc).__name__}: {exc}",
                title="Smoke Test",
                border_style="red",
            )
        )
        return

    console.print(
        Panel(
            f"[bold green]Smoke test passed[/bold green]\n\n"
            f"[bold]Player model:[/bold] {client.models['player']}    "
            f"[bold]Codegen model:[/bold] {client.models['codegen']}\n"
            f"[bold]Response:[/bold] {reply}",
            title="Smoke Test",
            border_style="green",
        )
    )


def _run_ingest(rulebook: str, name: str) -> None:
    """Run the ingestion pipeline and print a summary of the generated engine."""
    from pathlib import Path

    from playtest.ingestion.pipeline import ingest_rulebook

    console = Console()

    config_dir = Path(get_settings().game_configs_dir) / name
    if config_dir.exists():
        console.print(f"[yellow]Overwriting existing config for {name}[/yellow]")

    artifacts = ingest_rulebook(rulebook, name)
    digest = artifacts.digest
    console.print(
        Panel(
            f"[bold]Game:[/bold] {digest.game_name}, "
            f"{digest.min_players}-{digest.max_players} players\n"
            f"[bold]Config dir:[/bold] {artifacts.config_dir}\n"
            f"[bold]Actions:[/bold] {', '.join(a.name for a in digest.actions)}\n"
            f"[bold]Validated on attempt:[/bold] {artifacts.meta.get('attempts')}\n"
            f"[bold]Artifacts:[/bold] engine.py, test_engine.py, digest.md, digest.json, "
            f"player_briefing.txt, rulebook.txt, meta.json",
            title="Ingestion Complete",
            border_style="green",
        )
    )


def _show_config(game_name: str, truncate: bool = False) -> None:
    """Load a generated game config and print its digest, engine, and metadata."""
    from pathlib import Path

    from playtest.ingestion.schemas import GameArtifacts

    console = Console()
    config_dir = Path(get_settings().game_configs_dir) / game_name
    if not (config_dir / "digest.json").is_file():
        console.print(f"[red]No generated config found at {config_dir}[/red]")
        return

    artifacts = GameArtifacts(config_dir)
    digest = artifacts.digest

    def _maybe_truncate(text: str, limit: int) -> str:
        text = text.strip()
        if truncate and len(text) > limit:
            return text[:limit].strip() + " ..."
        return text

    engine_lines = (
        len(artifacts.engine_path.read_text(encoding="utf-8").splitlines())
        if artifacts.engine_path.is_file()
        else 0
    )
    console.print(
        Panel(
            f"[bold]{digest.game_name}[/bold], {digest.min_players}-{digest.max_players} "
            f"players\n"
            f"[bold]Config dir:[/bold] {artifacts.config_dir}\n"
            f"[bold]Engine:[/bold] engine.py ({engine_lines} lines)    "
            f"[bold]Validated on attempt:[/bold] {artifacts.meta.get('attempts', '?')}\n"
            f"[bold]Models:[/bold] digest={artifacts.meta.get('digest_model', '?')}, "
            f"codegen={artifacts.meta.get('codegen_model', '?')}",
            title="Game Config",
            border_style="cyan",
        )
    )

    console.print("[bold]Overview:[/bold]")
    console.print(_maybe_truncate(digest.overview, 400))
    console.print(
        "\n[bold]Components:[/bold] " + ", ".join(f"{c.name}×{c.count}" for c in digest.components)
    )
    console.print("\n[bold]Actions:[/bold]")
    for action in digest.actions:
        console.print(f"  - [cyan]{action.name}[/cyan]: {_maybe_truncate(action.effect, 80)}")
    if digest.ambiguities:
        console.print("\n[bold]Ambiguity rulings:[/bold]")
        for amb in digest.ambiguities:
            console.print(f"  - {_maybe_truncate(amb.question, 80)}")
            console.print(f"    → {_maybe_truncate(amb.resolution, 80)}")
    console.print(f"\n[dim]Full digest: {config_dir / 'digest.md'}[/dim]")


def _run_play(
    game: str,
    num_players: int,
    seed: int | None,
    log_file: str | None,
    verbose: bool,
    archetypes: list[str] | None,
) -> None:
    """Run a full playtest session via the plain turn-loop driver."""
    from playtest.runner import run_game

    run_game(
        game,
        num_players=num_players,
        seed=seed,
        log_file=log_file,
        verbose=verbose,
        archetypes=archetypes,
    )


def _parse_archetypes(value: str | None) -> list[str] | None:
    """Split a comma-separated --archetypes value into a list (None when absent)."""
    if not value:
        return None
    return [a.strip() for a in value.split(",")]


def _run_bulk(
    game: str,
    num_players: int,
    num_games: int,
    output_dir: str,
    archetypes: list[str] | None,
    seed_start: int,
) -> None:
    """Run many playtests, then print aggregate analytics."""
    from playtest.analytics import print_analytics_report
    from playtest.runner import run_multiple_games

    analytics = run_multiple_games(
        game,
        num_games=num_games,
        num_players=num_players,
        archetypes=archetypes,
        seed_start=seed_start,
        output_dir=output_dir,
    )
    print_analytics_report(analytics, Console())


def _run_analyze(log_dir: str) -> None:
    """Load game logs from a directory and print an analytics report."""
    from pathlib import Path

    from playtest.analytics import analyze_games, print_analytics_report

    console = Console()
    if not Path(log_dir).is_dir():
        console.print(f"[red]No log directory found at {log_dir}[/red]")
        return
    print_analytics_report(analyze_games(log_dir), console)


def _run_review(log_file: str, full: bool) -> None:
    """Load a saved game log and print its summary (and, with --full, every event)."""
    from pathlib import Path

    from playtest.ui.logger import GameLogger

    console = Console()
    if not Path(log_file).exists():
        console.print(f"[red]No log file found at {log_file}[/red]")
        return

    logger = GameLogger.from_file(log_file)
    log = logger.log
    summary = logger.get_summary()
    confusions = summary["player_confusions"]
    played = ", ".join(f"{a}×{n}" for a, n in summary["most_played_actions"].items())
    winners = ", ".join(summary["winners"]) or "nobody (draw)"
    console.print(
        Panel(
            f"[bold]Game:[/bold] {log.get('game_name')}, "
            f"{log.get('num_players')} players  [bold]seed:[/bold] {log.get('seed')}\n"
            f"[bold green]Winner:[/bold green] {winners}\n"
            f"[bold]Decisions:[/bold] {summary['total_steps']}    "
            f"[bold]Confusions:[/bold] {confusions['count']}\n"
            f"[bold]Actions:[/bold] {played or 'none'}",
            title="Game Review",
            border_style="cyan",
        )
    )

    if full:
        console.print("\n[bold]Events:[/bold]")
        for event in log.get("events", []):
            console.print(f"  [dim]{event.get('timestamp', '')}[/dim] [cyan]{event['type']}[/cyan]")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Board Game Playtesting Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand: ingest
    ingest_parser = subparsers.add_parser(
        "ingest", help="Process a rulebook into a game configuration"
    )
    ingest_parser.add_argument("--rulebook", type=str, required=True, help="Path to rulebook file")
    ingest_parser.add_argument(
        "--name", type=str, required=True, help="Game name (used as config directory name)"
    )

    # Subcommand: show-config
    show_parser = subparsers.add_parser(
        "show-config", help="Load and print a generated game configuration"
    )
    show_parser.add_argument("--game", type=str, required=True, help="Game config name")
    show_parser.add_argument(
        "--truncate",
        action="store_true",
        help="Show compact previews instead of full field contents",
    )

    # Subcommand: play
    play_parser = subparsers.add_parser("play", help="Run a playtest session")
    play_parser.add_argument(
        "--game",
        type=str,
        required=True,
        help="Game config name, or a module path like playtest.games.love_letter",
    )
    play_parser.add_argument("--players", type=int, default=2, help="Number of players")
    play_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    play_parser.add_argument(
        "--log-file", type=str, default=None, help="Path to save game log JSON"
    )
    play_parser.add_argument(
        "--verbose", action="store_true", help="Show players' private reasoning and notebooks"
    )
    play_parser.add_argument(
        "--archetypes",
        type=str,
        default=None,
        help="Comma-separated archetype per player (e.g. aggressive,cautious,default)",
    )

    # Subcommand: bulk
    bulk_parser = subparsers.add_parser(
        "bulk", help="Run multiple playtests and print aggregate analytics"
    )
    bulk_parser.add_argument("--game", type=str, required=True, help="Game config name")
    bulk_parser.add_argument("--players", type=int, default=2, help="Number of players")
    bulk_parser.add_argument("--num-games", type=int, required=True, help="Number of games to run")
    bulk_parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory for game logs"
    )
    bulk_parser.add_argument(
        "--archetypes", type=str, default=None, help="Comma-separated archetype per player"
    )
    bulk_parser.add_argument(
        "--seed-start", type=int, default=0, help="First seed (game i uses seed_start + i)"
    )

    # Subcommand: analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Compute analytics from a directory of game logs"
    )
    analyze_parser.add_argument(
        "--log-dir", type=str, required=True, help="Directory containing game log JSON files"
    )

    # Subcommand: review
    review_parser = subparsers.add_parser("review", help="Review a saved game log")
    review_parser.add_argument(
        "--log-file", type=str, required=True, help="Path to a game log JSON"
    )
    review_parser.add_argument(
        "--full", action="store_true", help="Print every event, not just the summary"
    )

    # Subcommand: smoke-test
    subparsers.add_parser("smoke-test", help="Verify the configured LLM and embedding backends")

    args = parser.parse_args()

    if args.command == "smoke-test":
        _smoke_test()
    elif args.command == "ingest":
        _run_ingest(args.rulebook, args.name)
    elif args.command == "show-config":
        _show_config(args.game, args.truncate)
    elif args.command == "play":
        _run_play(
            args.game,
            args.players,
            args.seed,
            args.log_file,
            args.verbose,
            _parse_archetypes(args.archetypes),
        )
    elif args.command == "bulk":
        _run_bulk(
            args.game,
            args.players,
            args.num_games,
            args.output_dir,
            _parse_archetypes(args.archetypes),
            args.seed_start,
        )
    elif args.command == "analyze":
        _run_analyze(args.log_dir)
    elif args.command == "review":
        _run_review(args.log_file, args.full)
    else:
        parser.print_help()
