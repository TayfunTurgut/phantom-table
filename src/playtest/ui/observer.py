"""Live terminal observer: color-coded, sequential per-step rendering with Rich.

Each ``on_*`` method prints immediately so the output is a simple, scrollable
transcript. Rendering is game-agnostic: it shows action labels, table talk, and
the engine's public event lines — never raw state, so nothing hidden can leak.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from playtest.agents.player import Decision

PLAYER_COLORS = (
    "cyan",
    "magenta",
    "green",
    "yellow",
    "red",
    "blue",
    "bright_cyan",
    "bright_magenta",
    "bright_green",
    "bright_yellow",
    "bright_red",
    "bright_blue",
)


def _color(seat: str) -> str:
    """Get the color for a seat, cycling through the palette if needed.

    Parses player_N format and uses modulo to cycle through the palette.
    Non-player_N seats get white as fallback.
    """
    if seat.startswith("player_"):
        try:
            player_num = int(seat.split("_")[1])
            return PLAYER_COLORS[(player_num - 1) % len(PLAYER_COLORS)]
        except (IndexError, ValueError):
            pass
    return "white"


class GameObserver:
    def __init__(self, console: Console | None = None, verbose: bool = False) -> None:
        self.console = console or Console()
        self.verbose = verbose

    def on_game_start(self, game_name: str, seats: list[str]) -> None:
        players = ", ".join(f"[{_color(s)}]{s}[/{_color(s)}]" for s in seats)
        self.console.print(
            Panel(
                f"[bold]{game_name}[/bold]\n[bold]Players:[/bold] {players}",
                title="Game Start",
                border_style="cyan",
            )
        )

    def on_step_start(self, step: int, acting: list[str]) -> None:
        names = ", ".join(f"[{_color(s)}]{s}[/{_color(s)}]" for s in acting)
        self.console.rule(f"Step {step} · {names} to act", style=_color(acting[0]))

    def on_decision(self, seat: str, decision: Decision) -> None:
        color = _color(seat)
        label = decision.action.label or decision.action.name
        self.console.print(f"[{color}]{seat}[/{color}] → [bold]{label}[/bold]")
        if decision.table_talk:
            self.console.print(f'  [italic]"{decision.table_talk}"[/italic]')
        if self.verbose and decision.reasoning:
            self.console.print(f"  [dim](reasoning: {decision.reasoning})[/dim]")
        if self.verbose and decision.notes:
            self.console.print(f"  [dim](notebook: {decision.notes})[/dim]")
        if decision.confused:
            self.console.print(f"  [red]⚠ {seat} made an invalid choice; fell back[/red]")

    def on_events(self, texts: list[str]) -> None:
        for text in texts:
            self.console.print(f"[yellow]▸[/yellow] {text}")

    def on_game_end(self, winners: list[str], scores: dict) -> None:
        names = (
            " and ".join(f"[{_color(w)}]{w}[/{_color(w)}]" for w in winners)
            if winners
            else "nobody (draw)"
        )
        score_lines = "\n".join(
            f"  [{_color(s)}]{s}[/{_color(s)}]: {value:g}" for s, value in sorted(scores.items())
        )
        body = f"[bold green]Winner:[/bold green] {names}"
        if score_lines:
            body += f"\n{score_lines}"
        self.console.print(Panel(body, title="Game Over", border_style="green"))
