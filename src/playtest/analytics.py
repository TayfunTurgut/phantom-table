"""Aggregate analytics over a directory of game logs.

Everything here is derived from the structured logs written by ``GameLogger`` — the
event stream plus the run-level header (archetypes, rule queries). Partial/error logs
(no ``events``) are skipped so a failed game in a bulk run never breaks aggregation.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from playtest.ui.logger import GameLogger

_CARDS = ["Guard", "Priest", "Baron", "Handmaid", "Prince", "King", "Countess", "Princess"]
# Cards whose play can directly eliminate an opponent (so effectiveness is meaningful).
_ELIMINATION_CARDS = {"Guard", "Baron", "Prince"}


def _card_from_action(action_type: str) -> str | None:
    """Map an action_type like ``play_guard`` to the canonical card name ``Guard``."""
    if not action_type.startswith("play_"):
        return None
    name = action_type[len("play_") :].capitalize()
    return name if name in _CARDS else None


def _eliminated_count(snapshot: dict) -> int:
    players = (snapshot or {}).get("players", {})
    return sum(1 for p in players.values() if p.get("is_eliminated"))


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
    for e in log["events"]:
        if e["type"] == "game_start":
            players = e.get("state", {}).get("players", {})
            if players:
                return sorted(players)
    num = log.get("num_players") or 0
    return [f"player_{i}" for i in range(1, num + 1)]


def _winners_of(log: dict) -> list[str]:
    winner = log.get("winner")
    return [w for w in str(winner).split(",") if w] if winner else []


def _archetypes_of(log: dict, players: list[str]) -> dict[str, str]:
    archetypes = log.get("archetypes") or ["default"] * len(players)
    return {
        pid: archetypes[i] if i < len(archetypes) else "default"
        for i, pid in enumerate(players)
    }


def _card_effectiveness(log: dict) -> dict[str, dict[str, int]]:
    """Pair each player_action with its own next gm_resolution; count eliminations caused."""
    stats: dict[str, dict[str, int]] = {
        c: {"played": 0, "successful_elimination": 0} for c in _ELIMINATION_CARDS
    }
    prev_snapshot: dict | None = None
    pending_card: str | None = None
    for e in log["events"]:
        etype = e["type"]
        if etype == "game_start":
            prev_snapshot = e.get("state")
        elif etype == "round_end":
            prev_snapshot = None  # new round resets eliminations; don't compare across it
            pending_card = None
        elif etype == "player_action":
            # Latest action before a resolution wins, so retried actions pair correctly.
            pending_card = _card_from_action(e.get("action_type", ""))
        elif etype == "gm_resolution":
            snapshot = e.get("state_snapshot")
            if pending_card in _ELIMINATION_CARDS:
                stats[pending_card]["played"] += 1
                caused = prev_snapshot is not None and (
                    _eliminated_count(snapshot) > _eliminated_count(prev_snapshot)
                )
                if caused:
                    stats[pending_card]["successful_elimination"] += 1
            prev_snapshot = snapshot
            pending_card = None
    return stats


def analyze_games(log_dir: str) -> dict:
    """Load all game logs from a directory and produce aggregate analytics."""
    logs = _load_logs(log_dir)
    games_played = len(logs)

    all_players: set[str] = set()
    win_counts: Counter[str] = Counter()
    total_rounds = 0
    total_turns = 0
    per_game_turns: list[int] = []
    card_play_frequency: Counter[str] = Counter()
    effectiveness: dict[str, dict[str, int]] = {
        c: {"played": 0, "successful_elimination": 0} for c in _ELIMINATION_CARDS
    }
    total_validations = 0
    total_rejections = 0
    rejection_reasons: Counter[str] = Counter()
    game_lengths_min: list[float] = []
    archetype_games: Counter[str] = Counter()
    archetype_wins: Counter[str] = Counter()
    rule_queries: Counter[str] = Counter()
    round_win_by_card: Counter[str] = Counter()

    for log in logs:
        players = _player_ids(log)
        all_players.update(players)
        winners = _winners_of(log)
        win_counts.update(winners)

        total_rounds += log.get("rounds_played", 0)
        turns = log.get("total_turns", 0)
        total_turns += turns
        per_game_turns.append(turns)

        archetype_map = _archetypes_of(log, players)
        for pid in players:
            arch = archetype_map[pid]
            archetype_games[arch] += 1
            if pid in winners:
                archetype_wins[arch] += 1

        for query in log.get("rule_queries") or []:
            rule_queries[query] += 1

        for e in log["events"]:
            etype = e["type"]
            if etype == "player_action":
                card = _card_from_action(e.get("action_type", ""))
                if card:
                    card_play_frequency[card] += 1
            elif etype == "gm_validation":
                total_validations += 1
                if e.get("is_valid") is False:
                    total_rejections += 1
                    reason = (e.get("error_message") or "unspecified").strip()
                    rejection_reasons[reason] += 1
            elif etype == "round_end":
                card = e.get("winning_card")
                if card:
                    round_win_by_card[card] += 1

        game_eff = _card_effectiveness(log)
        for card, s in game_eff.items():
            effectiveness[card]["played"] += s["played"]
            effectiveness[card]["successful_elimination"] += s["successful_elimination"]

        start, end = log.get("start_time"), log.get("end_time")
        if start and end:
            try:
                delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
                game_lengths_min.append(delta.total_seconds() / 60.0)
            except ValueError:
                pass

    win_rates = (
        {pid: win_counts[pid] / games_played for pid in sorted(all_players)}
        if games_played
        else {}
    )

    card_effectiveness = {
        card: {
            "played": s["played"],
            "successful_elimination": s["successful_elimination"],
            "rate": (s["successful_elimination"] / s["played"]) if s["played"] else 0.0,
        }
        for card, s in effectiveness.items()
    }

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
        "avg_rounds_per_game": (total_rounds / games_played) if games_played else 0.0,
        "avg_turns_per_round": (total_turns / total_rounds) if total_rounds else 0.0,
        "avg_turns_per_game": (total_turns / games_played) if games_played else 0.0,
        "win_rates": win_rates,
        "card_play_frequency": dict(card_play_frequency),
        "card_effectiveness": card_effectiveness,
        "action_rejection_rate": (
            (total_rejections / total_validations) if total_validations else 0.0
        ),
        "rejection_reasons": dict(rejection_reasons),
        "avg_game_length_minutes": (
            (sum(game_lengths_min) / len(game_lengths_min)) if game_lengths_min else 0.0
        ),
        "archetype_performance": archetype_performance,
        "common_rule_queries": [
            {"query": q, "count": c} for q, c in rule_queries.most_common()
        ],
        "round_win_by_card": dict(round_win_by_card),
    }


def print_analytics_report(analytics: dict, console: Console) -> None:
    """Pretty-print the analytics using Rich tables and panels."""
    console.print(
        Panel(
            f"[bold]Games played:[/bold] {analytics['games_played']}\n"
            f"[bold]Avg rounds/game:[/bold] {analytics['avg_rounds_per_game']:.2f}    "
            f"[bold]Avg turns/round:[/bold] {analytics['avg_turns_per_round']:.2f}    "
            f"[bold]Avg turns/game:[/bold] {analytics['avg_turns_per_game']:.2f}\n"
            f"[bold]Avg game length:[/bold] {analytics['avg_game_length_minutes']:.2f} min    "
            f"[bold]Action rejection rate:[/bold] {analytics['action_rejection_rate']:.1%}",
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

    card_table = Table(title="Card Usage & Effectiveness")
    card_table.add_column("Card")
    card_table.add_column("Played", justify="right")
    card_table.add_column("Eliminations", justify="right")
    card_table.add_column("Rate", justify="right")
    eff = analytics["card_effectiveness"]
    for card, count in sorted(analytics["card_play_frequency"].items(), key=lambda kv: -kv[1]):
        e = eff.get(card)
        if e:
            card_table.add_row(
                card, str(count), str(e["successful_elimination"]), f"{e['rate']:.1%}"
            )
        else:
            card_table.add_row(card, str(count), "-", "-")
    console.print(card_table)

    if analytics["archetype_performance"]:
        arch_table = Table(title="Archetype Performance")
        arch_table.add_column("Archetype")
        arch_table.add_column("Games", justify="right")
        arch_table.add_column("Wins", justify="right")
        arch_table.add_column("Win Rate", justify="right")
        for arch, s in analytics["archetype_performance"].items():
            arch_table.add_row(arch, str(s["games"]), str(s["wins"]), f"{s['win_rate']:.1%}")
        console.print(arch_table)

    if analytics["round_win_by_card"]:
        rwc = Table(title="Round Wins by Held Card")
        rwc.add_column("Card")
        rwc.add_column("Round Wins", justify="right")
        for card, count in sorted(analytics["round_win_by_card"].items(), key=lambda kv: -kv[1]):
            rwc.add_row(card, str(count))
        console.print(rwc)

    if analytics["rejection_reasons"]:
        rej = Table(title="Rejection Reasons")
        rej.add_column("Reason")
        rej.add_column("Count", justify="right")
        for reason, count in sorted(analytics["rejection_reasons"].items(), key=lambda kv: -kv[1]):
            rej.add_row(reason[:80], str(count))
        console.print(rej)

    if analytics["common_rule_queries"]:
        rq = Table(title="Common Rule Queries")
        rq.add_column("Query")
        rq.add_column("Count", justify="right")
        for item in analytics["common_rule_queries"][:15]:
            rq.add_row(item["query"][:80], str(item["count"]))
        console.print(rq)
