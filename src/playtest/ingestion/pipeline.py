import shutil
import time
from pathlib import Path

from rich.console import Console

from playtest.config import get_settings
from playtest.ingestion.analyzer import (
    generate_gm_prompt,
    generate_initial_state,
    generate_player_prompt,
    generate_state_schema,
    generate_tool_definitions,
)
from playtest.ingestion.chunker import chunk_rulebook, embed_and_store
from playtest.ingestion.schemas import GameConfig

_console = Console()


def _clean_config_dir(config_dir: Path) -> None:
    """Remove an existing config dir for a clean overwrite, retrying once on a lock."""
    if not config_dir.exists():
        return
    try:
        shutil.rmtree(config_dir)
    except OSError:
        time.sleep(1.0)
        try:
            shutil.rmtree(config_dir)
        except OSError as exc:
            raise RuntimeError(
                f"Could not remove existing config at {config_dir} ({exc}). "
                "Close any other playtest process holding the ChromaDB database and retry."
            ) from exc


def ingest_rulebook(rulebook_path: str, game_name: str, num_players: int = 2) -> GameConfig:
    """Process a rulebook into a complete game configuration."""
    settings = get_settings()
    rulebook_text = Path(rulebook_path).read_text(encoding="utf-8")

    config_dir = Path(settings.game_configs_dir) / game_name
    _clean_config_dir(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    _console.print(f"[bold]Ingesting[/bold] {rulebook_path} -> {config_dir}")

    # Artifact 1: embedded rulebook
    chunks = chunk_rulebook(rulebook_text, game_name=game_name)
    embed_and_store(chunks, collection_name=game_name, persist_dir=str(config_dir / "chromadb"))
    _console.print(f"  [green]embedded[/green] {len(chunks)} chunks into ChromaDB")

    # Artifacts 2-5 (dependency order: schema -> initial state -> tools -> prompts)
    state_schema = generate_state_schema(rulebook_text, num_players)
    _console.print("  [green]generated[/green] state schema")

    initial_state = generate_initial_state(rulebook_text, num_players, state_schema)
    _console.print("  [green]generated[/green] initial state")

    tool_definitions = generate_tool_definitions(rulebook_text)
    _console.print(f"  [green]generated[/green] tools - actions: {', '.join(tool_definitions)}")

    gm_prompt = generate_gm_prompt(rulebook_text, state_schema, tool_definitions)
    _console.print("  [green]generated[/green] GM prompt")

    player_prompt = generate_player_prompt(
        rulebook_text,
        forbidden_action_names=[n for n in tool_definitions if n != "draw_card"],
    )
    _console.print("  [green]generated[/green] player prompt")

    config = GameConfig(
        game_name=initial_state.get("game_name", game_name),
        variant=initial_state.get("variant", "classic"),
        num_players=num_players,
        config_dir=str(config_dir),
        state_schema=state_schema,
        initial_state_template=initial_state,
        tool_definitions=tool_definitions,
        gm_prompt=gm_prompt,
        player_prompt_template=player_prompt,
        rulebook_text=rulebook_text,
    )
    config.save()
    _console.print(f"[bold green]Done.[/bold green] Config saved to {config_dir}")

    return config
