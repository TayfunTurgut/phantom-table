"""Structured, timestamped game-event log for post-hoc review and analytics."""

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class GameLogger:
    def __init__(self) -> None:
        self.log: dict[str, Any] = {
            "session_id": None,
            "game_name": None,
            "variant": None,
            "num_players": None,
            "seed": None,
            "start_time": None,
            "end_time": None,
            "winner": None,
            "rounds_played": 0,
            "total_turns": 0,
            "archetypes": None,
            "rule_queries": [],
            "events": [],
        }

    def set_run_metadata(
        self, archetypes: list[str] | None = None, rule_queries: list[str] | None = None
    ) -> None:
        """Record run-level metadata not derivable from the event stream."""
        if archetypes is not None:
            self.log["archetypes"] = archetypes
        if rule_queries is not None:
            self.log["rule_queries"] = rule_queries

    def log_event(self, event_type: str, data: dict) -> None:
        """Append a timestamped event and opportunistically fill header fields."""
        self.log["events"].append({"type": event_type, "timestamp": _now(), **data})

        if event_type == "game_start":
            state = data.get("state", {})
            self.log["session_id"] = data.get("session_id")
            self.log["seed"] = data.get("seed")
            self.log["game_name"] = state.get("game_name")
            self.log["variant"] = state.get("variant")
            self.log["num_players"] = state.get("num_players")
            self.log["start_time"] = self.log["events"][-1]["timestamp"]
        elif event_type == "round_end":
            self.log["rounds_played"] = data.get("round_number", self.log["rounds_played"])
        elif event_type == "game_end":
            self.log["winner"] = data.get("winner")
            self.log["total_turns"] = data.get("total_turns", self.log["total_turns"])
            self.log["rounds_played"] = data.get("rounds_played", self.log["rounds_played"])
            self.log["end_time"] = self.log["events"][-1]["timestamp"]

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.log, fh, indent=2)

    @classmethod
    def from_file(cls, filepath: str) -> "GameLogger":
        logger = cls()
        with open(filepath, encoding="utf-8") as fh:
            logger.log = json.load(fh)
        return logger

    def get_summary(self) -> dict:
        events = self.log["events"]
        rejections = [
            {"player": e.get("player"), "error": e.get("error_message")}
            for e in events
            if e["type"] == "gm_validation" and e.get("is_valid") is False
        ]
        # Lossless: count raw action_type; display-time stripping happens in the UI.
        action_counts = Counter(
            e["action_type"] for e in events if e["type"] == "player_action" and "action_type" in e
        )
        rounds = self.log["rounds_played"]
        turns = self.log["total_turns"]
        return {
            "winner": self.log["winner"],
            "rounds_played": rounds,
            "total_turns": turns,
            "actions_rejected": {"count": len(rejections), "details": rejections},
            "most_played_actions": dict(action_counts),
            "average_turns_per_round": turns / rounds if rounds else 0.0,
        }
