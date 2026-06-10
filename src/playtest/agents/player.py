"""Player agent: plays one full turn per invocation as a single-shot agent.

A player is an LLM agent (gpt-4o-mini) with forced tool calling. The conversation
is fresh each turn — the game state IS the memory — so token cost does not grow over
a game. Within a turn the player emits action tool calls in sequence (draw, then,
having seen the drawn card, play); each is resolved by the GM (via a ``resolve_action``
callback supplied by the driver) and the updated state is fed back so the next call
can react. The turn ends when ``resolve_action`` reports the turn is over.

The highest-leverage reliability lever is phase-appropriate tool filtering: only
the action tools that make sense for the current phase and hand are exposed, and
the Countess rule is enforced structurally (when held with King or Prince, only
play_countess is available).
"""

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from playtest.agents.archetypes import apply_archetype
from playtest.config import get_settings
from playtest.errors import PlaytestError
from playtest.ingestion.schemas import GameConfig
from playtest.rules import get_rules
from playtest.tools import ToolRegistry

_OBSERVATION_TOOLS = {"get_game_state", "query_rulebook"}


class PlayerAction(BaseModel):
    """A single proposed move, ready for the GM to validate."""

    player_id: str
    action_type: str
    parameters: dict
    reasoning: str
    public_statement: str


class PlayerAgent:
    def __init__(
        self,
        player_id: str,
        game_config: GameConfig,
        tool_registry: ToolRegistry,
        openai_client: OpenAI,
        archetype: str = "default",
    ) -> None:
        self.player_id = player_id
        self.game_config = game_config
        self.archetype = archetype
        base_prompt = game_config.player_prompt_template.replace("{player_id}", player_id)
        self.system_prompt = apply_archetype(base_prompt, archetype)
        self.rules = get_rules(game_config)
        self.tool_registry = tool_registry
        self.client = openai_client
        self.model = get_settings().player_model

    def take_turn(
        self,
        filtered_state: dict,
        context: str | None = None,
        private_memory: list[str] | None = None,
        *,
        resolve_action: Callable[["PlayerAction"], dict],
    ) -> "PlayerAction | None":
        """Play one full turn in a single fresh conversation.

        ``resolve_action`` (supplied by the driver) sends each proposed action to the GM,
        which resolves and commits it, and returns a dict with the player's refreshed
        ``filtered_state``, the ``narration``, any ``private_info``, and ``turn_ended``.
        Returns the last action played, or None if the turn produced no action.
        """
        settings = get_settings()
        state = filtered_state
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._turn_prompt(state, context, private_memory)},
        ]
        observation_calls = 0
        last_action: PlayerAction | None = None

        for _ in range(settings.max_tool_iterations):
            tools = self._get_available_tools(state)
            action_only = [t for t in tools if t["function"]["name"] not in _OBSERVATION_TOOLS]
            offered = (
                action_only
                if observation_calls >= settings.max_observation_calls and action_only
                else tools
            )
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": offered,
                "tool_choice": "required",
                "parallel_tool_calls": False,
            }
            completion = self.client.chat.completions.create(**kwargs)
            message = completion.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            tool_call = (message.tool_calls or [None])[0]
            if tool_call is None:
                continue

            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: could not parse tool arguments as JSON ({exc}).",
                    }
                )
                continue

            if name in _OBSERVATION_TOOLS:
                result = self.tool_registry.execute_tool(name, args, self.player_id)
                content = result if isinstance(result, str) else json.dumps(result)
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": content}
                )
                observation_calls += 1
                continue

            # Action tool: capture intent and hand it to the GM to resolve and commit.
            proposed = self.tool_registry.execute_tool(name, args, self.player_id)
            assert isinstance(proposed, dict)  # action tools return a proposed-action dict
            action = PlayerAction(player_id=self.player_id, **proposed)
            outcome = resolve_action(action)
            state = outcome["filtered_state"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": self._resolution_feedback(outcome),
                }
            )
            last_action = action
            if outcome.get("turn_ended"):
                return last_action

        raise PlaytestError(
            f"{self.player_id} did not complete a turn within {settings.max_tool_iterations} "
            "iterations"
        )

    def _turn_prompt(
        self, state: dict, context: str | None, private_memory: list[str] | None
    ) -> str:
        parts: list[str] = []
        if context:
            parts.append(context)
        if private_memory:
            parts.append(
                "What you privately know:\n" + "\n".join(f"- {m}" for m in private_memory)
            )
        parts.append("Current game state (your view):\n" + json.dumps(state))
        parts.append(
            "It is your turn. Draw, then play a card by calling the available action "
            "tools in sequence. You may inspect the state or the rulebook first."
        )
        return "\n\n".join(parts)

    def _resolution_feedback(self, outcome: dict) -> str:
        lines = [outcome.get("narration") or "Action resolved."]
        if outcome.get("private_info"):
            lines.append("Private info: " + json.dumps(outcome["private_info"]))
        lines.append("Updated state (your view):\n" + json.dumps(outcome["filtered_state"]))
        return "\n".join(lines)

    def _get_available_tools(self, state: dict) -> list[dict]:
        """Observation tools plus the action tools the rules module allows right now."""
        shared = self.tool_registry.get_shared_tool_schemas()
        known = set(self.game_config.tool_definitions)
        candidate = self.rules.available_actions(state, self.player_id)
        valid = [name for name in candidate if name in known]
        return shared + self.tool_registry.get_action_tool_schemas(valid)
