"""Run full playtest sessions: resolve an engine, wire up agents, drive the loop."""

import json
import random
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from playtest.agents.player import PlayerAgent
from playtest.checkpoint import load_checkpoint
from playtest.config import Settings, get_settings
from playtest.engine import GameEngine, seats_for
from playtest.engine.loader import load_engine
from playtest.errors import PlaytestError
from playtest.llm import LLMClient, create_llm_client
from playtest.session import run_session
from playtest.ui.logger import GameLogger
from playtest.ui.observer import GameObserver


def resolve_game(game_ref: str) -> tuple[GameEngine, Path | None]:
    """Resolve a game reference to an engine and (if any) its config directory.

    ``game_ref`` may be a config name under ``game_configs_dir`` (a directory
    containing a generated ``engine.py``), a path to a config dir or engine file,
    or a dotted module path like ``playtest.games.love_letter``.
    """
    settings = get_settings()
    config_dir = Path(settings.game_configs_dir) / game_ref
    if (config_dir / "engine.py").is_file():
        return load_engine(str(config_dir)), config_dir
    try:
        engine = load_engine(game_ref)
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise PlaytestError(
            f"could not resolve game {game_ref!r}: not a config dir under "
            f"{settings.game_configs_dir!r}, not a path to an engine.py, and not "
            "an importable module"
        ) from exc
    path = Path(game_ref)
    if path.is_dir():
        return engine, path
    if path.suffix == ".py" and path.is_file():
        return engine, path.parent
    return engine, None


def _effective_max_steps(config_dir: Path | None, settings: Settings) -> tuple[int, str]:
    """Prefer the per-game decision budget declared in the game's meta.json.

    Returns ``(budget, source)`` where ``source`` is a human-readable label naming
    where the budget came from (so a max_steps crash can say so). Falls back to
    ``settings.max_steps`` (the global crash ceiling) when there's no config dir
    (e.g. module-ref games), no meta.json, no/zero ``max_decisions``, or the
    meta.json can't be read — provenance metadata should never be able to crash a
    playtest.
    """
    if config_dir is not None:
        meta_path = config_dir / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                max_decisions = meta.get("max_decisions", 0)
            except (json.JSONDecodeError, OSError):
                max_decisions = 0
            if isinstance(max_decisions, int) and max_decisions > 0:
                return max_decisions, "per-game max_decisions from meta.json"
    return settings.max_steps, "settings.max_steps (no per-game max_decisions in meta.json)"


def _make_players(
    engine: GameEngine,
    config_dir: Path | None,
    client: LLMClient,
    num_players: int,
    archetypes: list[str],
) -> dict:
    briefing = ""
    rulebook_text = ""
    if config_dir is not None:
        briefing_path = config_dir / "player_briefing.txt"
        if briefing_path.is_file():
            briefing = briefing_path.read_text(encoding="utf-8")
        rulebook_path = config_dir / "rulebook.txt"
        if rulebook_path.is_file():
            rulebook_text = rulebook_path.read_text(encoding="utf-8")

    return {
        seat: PlayerAgent(
            seat,
            client,
            game_name=engine.game_name,
            briefing=briefing,
            archetype=archetypes[i],
            rulebook_text=rulebook_text,
        )
        for i, seat in enumerate(seats_for(num_players))
    }


def run_game(
    game_ref: str,
    num_players: int = 2,
    seed: int | None = None,
    log_file: str | None = None,
    verbose: bool = False,
    archetypes: list[str] | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    console = Console()
    settings = get_settings()

    engine, config_dir = resolve_game(game_ref)
    if not engine.min_players <= num_players <= engine.max_players:
        raise ValueError(
            f"{engine.game_name} supports {engine.min_players}-{engine.max_players} "
            f"players; {num_players} is out of scope."
        )

    if archetypes is None:
        archetypes = ["default"] * num_players
    if len(archetypes) != num_players:
        raise ValueError(
            f"expected {num_players} archetypes (one per player), got {len(archetypes)}"
        )

    if seed is None:
        seed = random.randrange(2**32)

    client = create_llm_client(settings)
    players = _make_players(engine, config_dir, client, num_players, archetypes)

    session_id = str(uuid.uuid4())
    observer = GameObserver(console=console, verbose=verbose)
    logger = GameLogger()
    logger.set_run_metadata(archetypes=archetypes)

    max_steps, max_steps_source = _effective_max_steps(config_dir, settings)

    # Crash early: any EngineCrash / PlaytestError propagates. The finally block
    # persists whatever was logged so a crashed run is still inspectable.
    try:
        result = run_session(
            engine,
            players,
            observer,
            logger,
            num_players=num_players,
            seed=seed,
            session_id=session_id,
            max_steps=max_steps,
            max_steps_source=max_steps_source,
            checkpoint_path=checkpoint_path,
            game_ref=game_ref,
            archetypes=archetypes,
        )
    finally:
        if log_file:
            logger.save(log_file)
            console.print(f"[green]Game log written to[/green] {log_file}")

    summary = logger.get_summary()
    console.print(_summary_panel(summary))
    return {"final_state": result["final_state"], "summary": summary}


def resume_game(
    checkpoint_path: str,
    log_file: str | None = None,
    verbose: bool = False,
) -> dict:
    """Continue a game from a per-turn checkpoint written by ``run_session``.

    Rebuilds the engine and players from the checkpoint, restores each seat's
    notebook, and re-enters the turn loop at the checkpointed step. The resumed
    run keeps checkpointing to the same file, so it too can be resumed.
    """
    console = Console()
    settings = get_settings()

    cp = load_checkpoint(checkpoint_path)
    engine, config_dir = resolve_game(cp.game_ref)

    client = create_llm_client(settings)
    players = _make_players(engine, config_dir, client, cp.num_players, cp.archetypes)
    for seat, agent in players.items():
        agent.notes = cp.notebooks.get(seat, "")

    observer = GameObserver(console=console, verbose=verbose)
    logger = GameLogger()
    logger.set_run_metadata(archetypes=cp.archetypes)

    max_steps, max_steps_source = _effective_max_steps(config_dir, settings)

    console.print(
        f"[cyan]Resuming {engine.game_name} from step {cp.step}[/cyan] "
        f"(checkpoint {checkpoint_path})"
    )
    try:
        result = run_session(
            engine,
            players,
            observer,
            logger,
            num_players=cp.num_players,
            seed=cp.seed,
            session_id=cp.session_id,
            max_steps=max_steps,
            max_steps_source=max_steps_source,
            checkpoint_path=checkpoint_path,
            game_ref=cp.game_ref,
            archetypes=cp.archetypes,
            resume=cp,
        )
    finally:
        if log_file:
            logger.save(log_file)
            console.print(f"[green]Game log written to[/green] {log_file}")

    summary = logger.get_summary()
    console.print(_summary_panel(summary))
    return {"final_state": result["final_state"], "summary": summary}


def _summary_panel(summary: dict) -> Panel:
    played = ", ".join(f"{a}×{n}" for a, n in summary["most_played_actions"].items())
    winners = ", ".join(summary["winners"]) or "nobody (draw)"
    return Panel(
        f"[bold green]Winner:[/bold green] {winners}\n"
        f"[bold]Decisions:[/bold] {summary['total_steps']}    "
        f"[bold]Confusions:[/bold] {summary['player_confusions']['count']}\n"
        f"[bold]Actions:[/bold] {played or 'none'}",
        title="Game Summary",
        border_style="green",
    )


def run_multiple_games(
    game_ref: str,
    num_games: int,
    num_players: int,
    archetypes: list[str] | None = None,
    seed_start: int = 0,
    output_dir: str = "results",
) -> dict:
    """Run ``num_games`` playtests and collect aggregate results.

    Each game uses ``seed = seed_start + game_index`` for reproducibility and is saved
    to ``output_dir/game_NNN.json``. A game that crashes with a playtest finding
    (``PlaytestError``/``ValueError``) is skipped — its error written to
    ``game_NNN_error.json`` — so the batch continues; unexpected errors (API/network,
    bugs) propagate and abort the batch. Returns aggregate analytics.
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
                game_ref,
                num_players=num_players,
                seed=seed,
                log_file=str(log_path),
                archetypes=archetypes,
            )
        # Skip-and-continue: a playtest finding never aborts the batch.
        except (PlaytestError, ValueError) as exc:
            err_path = out / f"game_{i + 1:03d}_error.json"
            err_path.write_text(
                json.dumps({"seed": seed, "error": f"{type(exc).__name__}: {exc}"}, indent=2),
                encoding="utf-8",
            )
            console.print(f"[red]Game {i + 1} failed:[/red] {exc} (logged to {err_path})")

    return analyze_games(str(out))
