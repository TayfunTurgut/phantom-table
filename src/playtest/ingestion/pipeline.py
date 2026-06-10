import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console

from playtest.config import get_settings
from playtest.ingestion.analyzer import (
    build_game_spec,
    generate_core_mechanics,
    generate_flow_spec,
    generate_game_overview,
    generate_gm_prompt,
    generate_initial_state,
    generate_player_prompt,
    generate_setup_spec,
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
    """Process a rulebook into a complete game configuration.

    Independent LLM extraction steps run in parallel (each analyzer call builds its own
    client, so threads never share one); progress is printed on the main thread as each
    result is collected, and a worker exception propagates exactly as it would serially.
    """
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

    # Serial spine: the state design everything else refers to.
    state_schema = generate_state_schema(rulebook_text, num_players)
    _console.print("  [green]generated[/green] state schema")

    initial_state = generate_initial_state(rulebook_text, num_players, state_schema)
    _console.print("  [green]generated[/green] initial state")

    with ThreadPoolExecutor(max_workers=4) as pool:
        # Batch 1: independent extractions (rulebook/template only).
        tools_f = pool.submit(generate_tool_definitions, rulebook_text)
        overview_f = pool.submit(generate_game_overview, rulebook_text)
        mechanics_f = pool.submit(generate_core_mechanics, rulebook_text)
        setup_f = pool.submit(generate_setup_spec, rulebook_text, initial_state, num_players)

        tool_definitions = tools_f.result()
        _console.print(
            f"  [green]generated[/green] tools - actions: {', '.join(tool_definitions)}"
        )
        game_overview = overview_f.result()
        _console.print("  [green]generated[/green] game overview")
        core_mechanics = mechanics_f.result()
        _console.print(
            f"  [green]generated[/green] core mechanics ({len(core_mechanics)} constraints)"
        )
        setup_spec = setup_f.result()
        _console.print("  [green]generated[/green] setup spec")

        # Flow needs the tool names; then the spec halves are assembled and cross-checked.
        flow_spec = generate_flow_spec(rulebook_text, tool_definitions, initial_state)
        _console.print("  [green]generated[/green] flow spec")
        game_spec = build_game_spec(setup_spec, flow_spec)

        # Batch 2: prompts (need the spec text and batch-1 results).
        gm_prompt_f = pool.submit(
            generate_gm_prompt,
            rulebook_text,
            state_schema,
            tool_definitions,
            game_overview,
            core_mechanics,
            game_spec.end_conditions,
            game_spec.scoring,
        )
        player_prompt_f = pool.submit(
            generate_player_prompt,
            game_overview,
            core_mechanics,
            game_spec.turn.phases,
            list(tool_definitions),  # the prompt must not hardcode any action name
        )
        gm_prompt = gm_prompt_f.result()
        _console.print("  [green]generated[/green] GM prompt")
        player_prompt = player_prompt_f.result()
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
        game_spec=game_spec,
        core_mechanics=core_mechanics,
    )
    config.save()
    _console.print(f"[bold green]Done.[/bold green] Config saved to {config_dir}")

    return config
