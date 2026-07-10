"""Player agent: chooses one action per decision from the engine's legal set.

The engine enumerates every legal action, so a player turn is a single structured
LLM call: observation + recent events + self-authored notebook + numbered action
labels in, an index plus private reasoning, optional public table talk, and a
rewritten notebook out. Illegality is impossible by construction; the only
failure mode is a malformed/out-of-range choice, which gets one retry and then
falls back to action 0 (recorded as a ``player_confusion`` playtest signal, not
a crash).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from playtest.agents.archetypes import apply_archetype
from playtest.engine import Action
from playtest.llm import LLMClient

_log = logging.getLogger(__name__)

_CHOICE_SCHEMA = {
    "name": "choose_action",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action_index": {
                "type": "integer",
                "description": "Index of the chosen action from the numbered list.",
            },
            "reasoning": {
                "type": "string",
                "description": "Your private strategic reasoning (never shown to opponents).",
            },
            "table_talk": {
                "type": ["string", "null"],
                "description": "Optional remark said out loud at the table (opponents see it).",
            },
            "notes": {
                "type": "string",
                "description": (
                    "Your private notebook, rewritten in full each turn. It is your ONLY "
                    "memory beyond recent events: keep every deduction that still matters "
                    "(cards seen, who cannot hold what, opponents' tendencies, your plan), "
                    "drop stale information."
                ),
            },
        },
        "required": ["action_index", "reasoning", "table_talk", "notes"],
        "additionalProperties": False,
    },
}

_BASE_PROMPT = """You are {player_id}, an AI player in an automated playtest of {game_name}.

Play to win, within the rules. Each time you must act you receive:
1. Your private notebook — notes you wrote for yourself on previous turns.
2. What happened since your last decision (your private event history).
3. Your current view of the game state (hidden information stays hidden).
4. A numbered list of every action that is legal for you right now.

Choose exactly one action by its index. Also provide:
- reasoning: your private strategic thinking (invisible to opponents);
- table_talk: optionally, a short remark said out loud that all opponents will
  see — banter, misdirection, or silence (null);
- notes: your notebook, REWRITTEN IN FULL. This is your only memory beyond
  recent events, so carry forward every deduction that still matters (cards you
  have seen, what opponents cannot hold, reads on their behavior, your plan)
  and drop what is stale.
"""


@dataclass(frozen=True)
class Decision:
    """A player's chosen action plus the playtest-relevant context around it."""

    action: Action
    reasoning: str
    table_talk: str | None
    confused: bool = False
    notes: str = ""


def _parse_index(choice: dict, count: int) -> int | None:
    """The validated action index from a choice payload, or None if unusable."""
    index = choice.get("action_index")
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    return index if 0 <= index < count else None


class PlayerAgent:
    def __init__(
        self,
        player_id: str,
        client: LLMClient,
        game_name: str,
        briefing: str = "",
        archetype: str = "default",
        rulebook_text: str = "",
    ) -> None:
        self.player_id = player_id
        self.client = client
        self.archetype = archetype
        self.notes = ""  # self-authored memory, rewritten by the model each decision

        prompt = _BASE_PROMPT.format(player_id=player_id, game_name=game_name)
        if briefing.strip():
            prompt += f"\n## About this game\n\n{briefing.strip()}\n"
        if rulebook_text.strip():
            prompt += (
                "\n## Rulebook (full text — consult it for the exact rules)\n\n"
                f"{rulebook_text.strip()}\n"
            )
        self.system_prompt = apply_archetype(prompt, archetype)

    def choose(self, observation: dict, legal: list[Action], events: list[str]) -> Decision:
        """Make one decision. ``legal`` must be the engine's full legal set."""
        if not legal:
            raise ValueError(f"{self.player_id} asked to choose from an empty legal set")

        happened = "\n".join(f"- {line}" for line in events) or "- This is your first decision."
        menu = "\n".join(f"{i}. {a.label or a.name}" for i, a in enumerate(legal))
        notebook = self.notes.strip() or "(empty — this is your first decision)"
        user_message = (
            f"Your private notebook (you wrote this on a previous turn):\n{notebook}\n\n"
            f"What happened since your last decision:\n{happened}\n\n"
            f"Your current view of the game (JSON):\n{json.dumps(observation)}\n\n"
            f"Your legal actions:\n{menu}\n\n"
            f"Choose exactly one action by index (0-{len(legal) - 1}), and rewrite "
            f"your notebook."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        choice = self._complete(messages)
        index = _parse_index(choice, len(legal))
        if index is None:
            messages.append({"role": "assistant", "content": json.dumps(choice)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"action_index {choice.get('action_index', -1)!r} is out of range; "
                        f"choose an index between 0 and {len(legal) - 1}."
                    ),
                }
            )
            choice = self._complete(messages)
            index = _parse_index(choice, len(legal))
            if index is None:
                return Decision(
                    action=legal[0],
                    reasoning=(f"fallback after invalid choice {choice.get('action_index', -1)!r}"),
                    table_talk=None,
                    confused=True,
                    notes=self.notes,  # keep the last good notebook
                )
        notes = choice.get("notes")
        if isinstance(notes, str):
            self.notes = notes
        return Decision(
            action=legal[index],
            reasoning=choice.get("reasoning", ""),
            table_talk=choice.get("table_talk"),
            notes=self.notes,
        )

    def _complete(self, messages: list[dict]) -> dict:
        raw = self.client.complete(messages, role="player", json_schema=_CHOICE_SCHEMA)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            _log.warning("%s returned unparseable choice JSON: %r", self.player_id, raw[:200])
            return {}
        if not isinstance(parsed, dict):
            _log.warning("%s returned a non-object choice payload: %r", self.player_id, raw[:200])
            return {}
        return parsed
