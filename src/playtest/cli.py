import argparse

from rich.console import Console
from rich.panel import Panel

from playtest.config import get_settings


def _smoke_test() -> None:
    """Verify OpenAI API connectivity for chat completions and embeddings."""
    console = Console()
    console.print("[bold]Running OpenAI smoke test...[/bold]")

    try:
        from openai import OpenAI

        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)

        completion = client.chat.completions.create(
            model=settings.gm_model,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
        )
        reply = completion.choices[0].message.content or ""

        embedding = client.embeddings.create(
            model=settings.embedding_model,
            input="playtest smoke test",
        )
        dimension = len(embedding.data[0].embedding)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        console.print(
            Panel(
                f"[bold red]Smoke test failed[/bold red]\n\n{type(exc).__name__}: {exc}",
                title="OpenAI Smoke Test",
                border_style="red",
            )
        )
        return

    console.print(
        Panel(
            f"[bold green]Smoke test passed[/bold green]\n\n"
            f"[bold]Chat model:[/bold] {settings.gm_model}\n"
            f"[bold]Response:[/bold] {reply}\n\n"
            f"[bold]Embedding model:[/bold] {settings.embedding_model}\n"
            f"[bold]Embedding dimension:[/bold] {dimension}",
            title="OpenAI Smoke Test",
            border_style="green",
        )
    )


def _run_ingest(rulebook: str, name: str, num_players: int) -> None:
    """Run the ingestion pipeline and print a summary of the generated config."""
    from playtest.ingestion.pipeline import ingest_rulebook

    console = Console()
    config = ingest_rulebook(rulebook, name, num_players)
    console.print(
        Panel(
            f"[bold]Game:[/bold] {config.game_name} ({config.variant}), "
            f"{config.num_players} players\n"
            f"[bold]Config dir:[/bold] {config.config_dir}\n"
            f"[bold]Tools:[/bold] {', '.join(config.tool_definitions)}\n"
            f"[bold]Artifacts:[/bold] state_schema.json, initial_state.json, "
            f"tool_definitions.json, gm_prompt.txt, player_prompt.txt, chromadb/",
            title="Ingestion Complete",
            border_style="green",
        )
    )


def _show_config(game_name: str) -> None:
    """Load a saved game config and print its tools, schema, prompts, and a sample query."""
    from pathlib import Path

    from playtest.ingestion.chunker import query_collection
    from playtest.ingestion.schemas import GameConfig

    console = Console()
    config_dir = Path(get_settings().game_configs_dir) / game_name
    if not config_dir.exists():
        console.print(f"[red]No config found at {config_dir}[/red]")
        return

    config = GameConfig.load(str(config_dir))

    console.print(
        Panel(
            f"[bold]{config.game_name}[/bold] ({config.variant}), {config.num_players} players\n"
            f"[bold]Config dir:[/bold] {config.config_dir}",
            title="Game Config",
            border_style="cyan",
        )
    )

    console.print("[bold]Tools:[/bold]")
    for tool_name, schema in config.tool_definitions.items():
        description = schema["function"]["description"]
        console.print(f"  - [cyan]{tool_name}[/cyan]: {description[:80]}")

    properties = config.state_schema.get("properties", config.state_schema)
    console.print("\n[bold]State schema fields:[/bold] " + ", ".join(properties))

    console.print("\n[bold]GM prompt preview:[/bold]")
    console.print(config.gm_prompt[:400].strip() + " ...")
    console.print("\n[bold]Player prompt preview:[/bold]")
    console.print(config.player_prompt_template[:400].strip() + " ...")

    hits = query_collection(
        "what does the guard do", game_name, str(config_dir / "chromadb"), n_results=1
    )
    console.print("\n[bold]Rulebook query 'what does the guard do':[/bold]")
    console.print((hits[0][:400] + " ...") if hits else "[yellow]no results[/yellow]")


def _run_play(
    game: str,
    num_players: int,
    seed: int | None,
    log_file: str | None,
    verbose: bool,
    archetypes: list[str] | None,
) -> None:
    """Run a full playtest session via the LangGraph orchestration."""
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
    rejected = summary["actions_rejected"]
    played = ", ".join(f"{a}×{n}" for a, n in summary["most_played_actions"].items())
    console.print(
        Panel(
            f"[bold]Game:[/bold] {log.get('game_name')} ({log.get('variant')}), "
            f"{log.get('num_players')} players  [bold]seed:[/bold] {log.get('seed')}\n"
            f"[bold green]Winner:[/bold green] {summary['winner']}\n"
            f"[bold]Rounds:[/bold] {summary['rounds_played']}    "
            f"[bold]Turns:[/bold] {summary['total_turns']}    "
            f"[bold]Rejections:[/bold] {rejected['count']}\n"
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
    ingest_parser.add_argument("--players", type=int, default=2, help="Number of players")

    # Subcommand: show-config
    show_parser = subparsers.add_parser(
        "show-config", help="Load and print a generated game configuration"
    )
    show_parser.add_argument("--game", type=str, required=True, help="Game config name")

    # Subcommand: play
    play_parser = subparsers.add_parser("play", help="Run a playtest session")
    play_parser.add_argument("--game", type=str, required=True, help="Game config name")
    play_parser.add_argument("--players", type=int, default=2, help="Number of players")
    play_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    play_parser.add_argument(
        "--log-file", type=str, default=None, help="Path to save game log JSON"
    )
    play_parser.add_argument(
        "--verbose", action="store_true", help="Show private reasoning and GM internal notes"
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
    subparsers.add_parser("smoke-test", help="Verify OpenAI API connectivity")

    args = parser.parse_args()

    if args.command == "smoke-test":
        _smoke_test()
    elif args.command == "ingest":
        _run_ingest(args.rulebook, args.name, args.players)
    elif args.command == "show-config":
        _show_config(args.game)
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
