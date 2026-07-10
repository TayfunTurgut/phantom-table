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
            "num_players": None,
            "seed": None,
            "start_time": None,
            "end_time": None,
            "winners": [],
            "scores": {},
            "total_steps": 0,
            "archetypes": None,
            "events": [],
        }

    def set_run_metadata(self, archetypes: list[str] | None = None) -> None:
        """Record run-level metadata not derivable from the event stream."""
        if archetypes is not None:
            self.log["archetypes"] = archetypes

    def log_event(self, event_type: str, data: dict) -> None:
        """Append a timestamped event and opportunistically fill header fields."""
        self.log["events"].append({"type": event_type, "timestamp": _now(), **data})

        if event_type in ("game_start", "game_resume"):
            self.log["session_id"] = data.get("session_id")
            self.log["seed"] = data.get("seed")
            self.log["game_name"] = data.get("game_name")
            self.log["num_players"] = data.get("num_players")
            self.log["start_time"] = self.log["events"][-1]["timestamp"]
        elif event_type == "game_end":
            self.log["winners"] = data.get("winners", [])
            self.log["scores"] = data.get("scores", {})
            self.log["total_steps"] = data.get("total_steps", 0)
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
        action_counts = Counter(
            e["action"] for e in events if e["type"] == "decision" and "action" in e
        )
        confusions = [
            {"step": e.get("step"), "seat": e.get("seat")}
            for e in events
            if e["type"] == "player_confusion"
        ]
        return {
            "winners": self.log["winners"],
            "scores": self.log["scores"],
            "total_steps": self.log["total_steps"],
            "most_played_actions": dict(action_counts),
            "player_confusions": {"count": len(confusions), "details": confusions},
        }
