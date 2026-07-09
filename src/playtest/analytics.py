"""Aggregate analytics over a directory of game logs.

Everything here is derived from the structured logs written by ``GameLogger`` — the
event stream plus the run-level header (archetypes, rule queries). Nothing is
game-specific: actions are counted by their engine action names. Partial/error
logs (no ``events``) are skipped so a failed game in a bulk run never breaks
aggregation.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from playtest.engine import seats_for
from playtest.ui.logger import GameLogger


def _load_logs(log_dir: str) -> list[dict]:
    logs: list[dict] = []
    for path in sorted(Path(log_dir).glob("*.json")):
        try:
            log = GameLogger.from_file(str(path)).log
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(log, dict) or not log.get("events"):
            continue  # skip error/partial logs
        logs.append(log)
    return logs


def _player_ids(log: dict) -> list[str]:
    return seats_for(log.get("num_players") or 0)


def _archetypes_of(log: dict, players: list[str]) -> dict[str, str]:
    archetypes = log.get("archetypes") or ["default"] * len(players)
    return {
        pid: archetypes[i] if i < len(archetypes) else "default" for i, pid in enumerate(players)
    }


def analyze_games(log_dir: str) -> dict:
    """Load all game logs from a directory and produce aggregate analytics."""
    logs = _load_logs(log_dir)
    games_played = len(logs)

    all_players: set[str] = set()
    win_counts: Counter[str] = Counter()
    per_game_steps: list[int] = []
    action_frequency: Counter[str] = Counter()
    total_decisions = 0
    total_confusions = 0
    game_lengths_min: list[float] = []
    archetype_games: Counter[str] = Counter()
    archetype_wins: Counter[str] = Counter()

    for log in logs:
        players = _player_ids(log)
        all_players.update(players)
        winners = log.get("winners") or []
        win_counts.update(winners)
        per_game_steps.append(log.get("total_steps", 0))

        archetype_map = _archetypes_of(log, players)
        for pid in players:
            arch = archetype_map[pid]
            archetype_games[arch] += 1
            if pid in winners:
                archetype_wins[arch] += 1

        for e in log["events"]:
            etype = e["type"]
            if etype == "decision":
                total_decisions += 1
                action = e.get("action")
                if action:
                    action_frequency[action] += 1
            elif etype == "player_confusion":
                total_confusions += 1

        start, end = log.get("start_time"), log.get("end_time")
        if start and end:
            try:
                delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
                game_lengths_min.append(delta.total_seconds() / 60.0)
            except ValueError:
                pass

    win_rates = (
        {pid: win_counts[pid] / games_played for pid in sorted(all_players)} if games_played else {}
    )

    archetype_performance = {
        arch: {
            "games": archetype_games[arch],
            "wins": archetype_wins[arch],
            "win_rate": (
                (archetype_wins[arch] / archetype_games[arch]) if archetype_games[arch] else 0.0
            ),
        }
        for arch in sorted(archetype_games)
    }

    return {
        "games_played": games_played,
        "avg_steps_per_game": ((sum(per_game_steps) / games_played) if games_played else 0.0),
        "win_rates": win_rates,
        "action_frequency": dict(action_frequency),
        "confusion_rate": (total_confusions / total_decisions) if total_decisions else 0.0,
        "avg_game_length_minutes": (
            (sum(game_lengths_min) / len(game_lengths_min)) if game_lengths_min else 0.0
        ),
        "archetype_performance": archetype_performance,
    }


def print_analytics_report(analytics: dict, console: Console) -> None:
    """Pretty-print the analytics using Rich tables and panels."""
    console.print(
        Panel(
            f"[bold]Games played:[/bold] {analytics['games_played']}\n"
            f"[bold]Avg decisions/game:[/bold] {analytics['avg_steps_per_game']:.2f}\n"
            f"[bold]Avg game length:[/bold] {analytics['avg_game_length_minutes']:.2f} min    "
            f"[bold]Confusion rate:[/bold] {analytics['confusion_rate']:.1%}",
            title="Playtest Analytics",
            border_style="green",
        )
    )

    win_table = Table(title="Win Rates", show_edge=True)
    win_table.add_column("Player")
    win_table.add_column("Win Rate", justify="right")
    for pid, rate in analytics["win_rates"].items():
        win_table.add_row(pid, f"{rate:.1%}")
    console.print(win_table)

    if analytics["action_frequency"]:
        action_table = Table(title="Action Frequency")
        action_table.add_column("Action")
        action_table.add_column("Count", justify="right")
        for action, count in sorted(analytics["action_frequency"].items(), key=lambda kv: -kv[1]):
            action_table.add_row(action, str(count))
        console.print(action_table)

    if analytics["archetype_performance"]:
        arch_table = Table(title="Archetype Performance")
        arch_table.add_column("Archetype")
        arch_table.add_column("Games", justify="right")
        arch_table.add_column("Wins", justify="right")
        arch_table.add_column("Win Rate", justify="right")
        for arch, s in analytics["archetype_performance"].items():
            arch_table.add_row(arch, str(s["games"]), str(s["wins"]), f"{s['win_rate']:.1%}")
        console.print(arch_table)
