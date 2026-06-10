"""Live terminal observer: color-coded, sequential per-turn rendering with Rich.

Each ``on_*`` method prints immediately (no live-updating Layout) so the output is a
simple, scrollable transcript. Rendering is game-agnostic: per-player integer fields
become columns, boolean fields become status flags, and list fields are never shown
(only counts) — so table-side output never leaks hidden information regardless of game.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PLAYER_COLORS = {
    "player_1": "cyan",
    "player_2": "magenta",
    "player_3": "green",
    "player_4": "yellow",
    "player_5": "red",
    "player_6": "blue",
    "gm": "bold white",
}

_META_PARAMS = {"reasoning", "public_statement", "player_id", "action_type"}


def _color(player_id: str) -> str:
    return PLAYER_COLORS.get(player_id, "white")


class GameObserver:
    def __init__(self, console: Console | None = None, verbose: bool = False) -> None:
        self.console = console or Console()
        self.verbose = verbose
        self.events: list[dict] = []
        self._round_number = 1

    def on_game_start(self, state: dict, narration: str) -> None:
        self._round_number = state.get("round_number", 1)
        players = ", ".join(
            f"[{_color(pid)}]{pid}[/{_color(pid)}]" for pid in sorted(state.get("players", {}))
        )
        body = (
            f"[bold]{state.get('game_name')}[/bold] ({state.get('variant')})\n"
            f"[bold]Players:[/bold] {players}\n\n"
            f"{narration}"
        )
        self.console.print(Panel(body, title="Game Start", border_style="cyan"))

    def on_turn_start(self, player_id: str, turn_index: int, phase: str) -> None:
        color = _color(player_id)
        self.console.rule(
            f"[{color}]Round {self._round_number} · Turn {turn_index} · "
            f"{player_id}[/{color}] ({phase})",
            style=color,
        )

    def on_player_action(self, player_id: str, action: dict) -> None:
        color = _color(player_id)
        params = {k: v for k, v in action.get("parameters", {}).items() if k not in _META_PARAMS}
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        line = f"[{color}]{player_id}[/{color}] plays [bold]{action.get('action_type')}[/bold]"
        if param_str:
            line += f" → {param_str}"
        self.console.print(line)
        statement = action.get("public_statement")
        if statement:
            self.console.print(f'  [italic]"{statement}"[/italic]')
        if self.verbose and action.get("reasoning"):
            self.console.print(f"  [dim](reasoning: {action['reasoning']})[/dim]")

    def on_gm_resolution(self, resolution: dict) -> None:
        narration = resolution.get("narration") or ""
        self.console.print(f"[yellow]GM:[/yellow] {narration}")
        if self.verbose:
            for key in ("action_summary", "gm_reasoning"):
                if resolution.get(key):
                    self.console.print(f"  [dim]({key}: {resolution[key]})[/dim]")

    def on_action_rejected(self, player_id: str, error: str) -> None:
        color = _color(player_id)
        self.console.print(f"[red]✗ [{color}]{player_id}[/{color}] rejected:[/red] {error}")

    def on_round_end(self, round_number: int, winner: str, scores: dict) -> None:
        self._round_number = round_number
        score_lines = "\n".join(
            f"  [{_color(pid)}]{pid}[/{_color(pid)}]: {value}"
            for pid, value in sorted(scores.items())
        )
        winner_color = _color(winner)
        body = (
            f"[bold]Round {round_number} winner:[/bold] "
            f"[{winner_color}]{winner}[/{winner_color}]"
        )
        if score_lines:
            body += f"\n{score_lines}"
        self.console.print(Panel(body, title="Round End", border_style="yellow"))

    def on_game_end(self, winner: str, final_scores: dict) -> None:
        winner_color = _color(winner) if winner else "white"
        score_lines = "\n".join(
            f"  [{_color(pid)}]{pid}[/{_color(pid)}]: {value}"
            for pid, value in sorted(final_scores.items())
        )
        body = f"[bold green]Winner:[/bold green] [{winner_color}]{winner}[/{winner_color}]"
        if score_lines:
            body += f"\n{score_lines}"
        self.console.print(Panel(body, title="Game Over", border_style="green"))

    def on_state_update(self, state: dict) -> None:
        self._round_number = state.get("round_number", self._round_number)
        players = state.get("players", {})
        if not players:
            return

        sample = players[sorted(players)[0]]
        int_fields = sorted(
            f for f, v in sample.items() if isinstance(v, int) and not isinstance(v, bool)
        )
        bool_fields = sorted(f for f, v in sample.items() if isinstance(v, bool))

        table = Table(show_edge=True, expand=False)
        table.add_column("Player")
        for field in int_fields:
            table.add_column(field, justify="right")
        if bool_fields:
            table.add_column("Status")
        for pid, p in sorted(players.items()):
            row = [f"[{_color(pid)}]{pid}[/{_color(pid)}]"]
            row.extend(str(p.get(field, "")) for field in int_fields)
            if bool_fields:
                flags = [field for field in bool_fields if p.get(field)]
                row.append(", ".join(flags) if flags else "-")
            table.add_row(*row)
        self.console.print(table)

        counters = {
            key: value
            for key, value in state.items()
            if isinstance(value, int) and not isinstance(value, bool) and key != "num_players"
        }
        if counters:
            self.console.print(
                "[dim]" + "  ".join(f"{key}:{value}" for key, value in sorted(counters.items()))
                + "[/dim]"
            )
