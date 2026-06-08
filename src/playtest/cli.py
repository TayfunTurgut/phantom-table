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


def _run_play(game: str, num_players: int, seed: int | None, log_file: str | None) -> None:
    """Run a full playtest session via the LangGraph orchestration."""
    from playtest.runner import run_game

    run_game(game, num_players=num_players, seed=seed, log_file=log_file)


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
        _run_play(args.game, args.players, args.seed, args.log_file)
    else:
        parser.print_help()
