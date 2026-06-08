"""Player agent: observes game state, reasons, and proposes one action per turn.

A player is an LLM agent (gpt-4o-mini) with forced tool calling. On each turn it
may observe (get_game_state) and consult the rulebook (query_rulebook), then must
call exactly one action tool, which is captured as the proposed move. The GM owns
validation; rejections are fed back as ``game_context`` on a later ``take_turn``
by the orchestration layer.

The highest-leverage reliability lever is phase-appropriate tool filtering: only
the action tools that make sense for the current phase and hand are exposed, and
the Countess rule is enforced structurally (when held with King or Prince, only
play_countess is available).
"""

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from playtest.config import get_settings
from playtest.ingestion.schemas import GameConfig
from playtest.tools import ToolRegistry

_MAX_TOOL_ITERATIONS = 5
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
    ) -> None:
        self.player_id = player_id
        self.game_config = game_config
        self.system_prompt = game_config.player_prompt_template.replace("{player_id}", player_id)
        self.tool_registry = tool_registry
        self.client = openai_client
        self.model = get_settings().player_model
        self.conversation_history: list[dict] = [{"role": "system", "content": self.system_prompt}]

    def take_turn(self, game_context: str | None = None) -> PlayerAction:
        """Observe the state, then propose exactly one action.

        ``game_context`` (GM narration or a rejection message) is injected as a user
        message before the agent acts, so the LLM can react to it.
        """
        if game_context:
            self.conversation_history.append({"role": "user", "content": game_context})

        state = self.tool_registry.execute_tool(
            "get_game_state",
            {"reasoning": "Determine the current phase and my hand."},
            self.player_id,
        )
        assert isinstance(state, dict)  # get_game_state always returns a state dict
        turn_phase = state["turn_phase"]
        hand = state["players"][self.player_id]["hand"]

        tools = self._get_available_tools(turn_phase, hand)
        action = self._call_llm(self.conversation_history, tools)
        return PlayerAction(player_id=self.player_id, **action)

    def _get_available_tools(self, turn_phase: str, hand: list[str]) -> list[dict]:
        """Expose only the tools that make sense for this phase and hand."""
        shared = self.tool_registry.get_shared_tool_schemas()
        known = set(self.game_config.tool_definitions)

        if turn_phase == "draw":
            candidate = ["draw_card"]
        elif turn_phase == "play":
            hand_lower = [c.lower() for c in hand]
            if "countess" in hand_lower and ("king" in hand_lower or "prince" in hand_lower):
                candidate = ["play_countess"]
            else:
                candidate = [f"play_{card.lower()}" for card in set(hand)]
        else:
            candidate = []

        valid = [name for name in candidate if name in known]
        return shared + self.tool_registry.get_action_tool_schemas(valid)

    def _call_llm(self, messages: list[dict], tools: list[dict]) -> dict:
        """Force tool calls until an action tool is chosen; observations loop back.

        Observation tools (get_game_state, query_rulebook) are executed and fed back
        as tool results so the agent can keep reasoning. The first action tool call
        is captured and returned. Bounded to prevent infinite observation loops.
        """
        for _ in range(_MAX_TOOL_ITERATIONS):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
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
            args = json.loads(tool_call.function.arguments or "{}")
            result = self.tool_registry.execute_tool(name, args, self.player_id)
            content = result if isinstance(result, str) else json.dumps(result)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

            if name not in _OBSERVATION_TOOLS:
                assert isinstance(result, dict)  # action tools return a proposed-action dict
                return result

        raise RuntimeError(
            f"{self.player_id} did not select an action within {_MAX_TOOL_ITERATIONS} iterations"
        )
