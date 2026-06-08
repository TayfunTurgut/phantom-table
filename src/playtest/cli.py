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
        print(f"Ingestion not yet implemented. Rulebook: {args.rulebook}")
    elif args.command == "play":
        print(f"Play not yet implemented. Game: {args.game}, Players: {args.players}")
    else:
        parser.print_help()
