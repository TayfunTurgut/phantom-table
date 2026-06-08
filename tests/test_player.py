import copy
import json
from types import SimpleNamespace

import pytest

from playtest.agents.player import PlayerAction, PlayerAgent
from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry

from .test_state import DECK, REMOVED, REVEALED, TEMPLATE

PLAYER_PROMPT = "You are playing Love Letter, and your player ID is {player_id}."

TOOL_DEFINITIONS = {
    "draw_card": {
        "type": "function",
        "function": {
            "name": "draw_card",
            "description": "Draw a card.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "public_statement": {"type": "string"},
                },
                "required": ["reasoning", "public_statement"],
                "additionalProperties": False,
            },
        },
    },
    "play_guard": {
        "type": "function",
        "function": {
            "name": "play_guard",
            "description": "Play the Guard.",
            "parameters": {},
        },
    },
    "play_prince": {
        "type": "function",
        "function": {
            "name": "play_prince",
            "description": "Play the Prince.",
            "parameters": {},
        },
    },
    "play_king": {
        "type": "function",
        "function": {
            "name": "play_king",
            "description": "Play the King.",
            "parameters": {},
        },
    },
    "play_countess": {
        "type": "function",
        "function": {
            "name": "play_countess",
            "description": "Play the Countess.",
            "parameters": {},
        },
    },
}


def _tool_names(schemas: list[dict]) -> set[str]:
    return {schema["function"]["name"] for schema in schemas}


def _make_agent(
    client: object,
    *,
    turn_phase: str = "draw",
    p1_hand: list[str] | None = None,
) -> PlayerAgent:
    template = copy.deepcopy(TEMPLATE)
    template["turn_phase"] = turn_phase
    game_config = SimpleNamespace(
        config_dir="/tmp/does-not-need-to-exist",
        game_name="Love Letter",
        player_prompt_template=PLAYER_PROMPT,
        tool_definitions=copy.deepcopy(TOOL_DEFINITIONS),
    )
    manager = GameStateManager()
    manager.initialize(
        initial_state=template,
        deck_cards=DECK,
        removed_card=REMOVED,
        revealed_cards=REVEALED,
        player_hands={"player_1": p1_hand or ["King"], "player_2": ["Guard"]},
    )
    registry = ToolRegistry(game_config, manager)
    return PlayerAgent("player_1", game_config, registry, client)  # type: ignore[arg-type]


# --- Offline fake client (deterministic, no network) ------------------------


class _FakeToolCall:
    def __init__(self, name: str, arguments: dict) -> None:
        self.id = f"call_{name}"
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


class _FakeMessage:
    def __init__(self, tool_calls: list[_FakeToolCall]) -> None:
        self.content = None
        self.tool_calls = tool_calls

    def model_dump(self, **_: object) -> dict:
        return {"role": "assistant", "content": self.content}


class _ScriptedCompletions:
    """Returns scripted messages in order; the last one repeats if exhausted."""

    def __init__(self, messages: list[_FakeMessage]) -> None:
        self._messages = messages
        self._calls = 0

    def create(self, **kwargs: object) -> object:
        # Every player LLM call must force a tool call with no parallelism.
        assert kwargs["tool_choice"] == "required"
        assert kwargs["parallel_tool_calls"] is False
        index = min(self._calls, len(self._messages) - 1)
        self._calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=self._messages[index])])


class _FakeClient:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(messages))


def _draw_then_act(action: str, args: dict) -> _FakeClient:
    """Observe with get_game_state, then call an action tool."""
    return _FakeClient(
        [
            _FakeMessage([_FakeToolCall("get_game_state", {"reasoning": "look"})]),
            _FakeMessage([_FakeToolCall(action, args)]),
        ]
    )


# --- Tool exposure tests (no LLM) -------------------------------------------


def test_draw_phase_exposes_only_draw_card(settings) -> None:
    agent = _make_agent(_FakeClient([]), turn_phase="draw")
    names = _tool_names(agent._get_available_tools("draw", ["King"]))
    assert names == {"query_rulebook", "get_game_state", "draw_card"}


def test_play_phase_exposes_only_matching_play_tools(settings) -> None:
    agent = _make_agent(_FakeClient([]), turn_phase="play")
    names = _tool_names(agent._get_available_tools("play", ["Guard", "Prince"]))
    assert names == {"query_rulebook", "get_game_state", "play_guard", "play_prince"}


def test_countess_forced_play_with_king(settings) -> None:
    agent = _make_agent(_FakeClient([]), turn_phase="play")
    names = _tool_names(agent._get_available_tools("play", ["Countess", "King"]))
    action_tools = names - {"query_rulebook", "get_game_state"}
    assert action_tools == {"play_countess"}


def test_countess_forced_play_with_prince(settings) -> None:
    agent = _make_agent(_FakeClient([]), turn_phase="play")
    names = _tool_names(agent._get_available_tools("play", ["Countess", "Prince"]))
    action_tools = names - {"query_rulebook", "get_game_state"}
    assert action_tools == {"play_countess"}


# --- take_turn behavior (scripted fake client) ------------------------------


def test_take_turn_returns_wellformed_action(settings) -> None:
    client = _draw_then_act(
        "draw_card", {"reasoning": "deck has cards", "public_statement": "I draw."}
    )
    agent = _make_agent(client, turn_phase="draw")
    action = agent.take_turn()

    assert isinstance(action, PlayerAction)
    assert action.player_id == "player_1"
    assert action.action_type == "draw_card"
    assert action.parameters == {}


def test_reasoning_and_public_statement_captured_separately(settings) -> None:
    client = _draw_then_act(
        "play_guard",
        {
            "target_player": "player_2",
            "named_card": "Priest",
            "reasoning": "private deduction",
            "public_statement": "I accuse you!",
        },
    )
    agent = _make_agent(client, turn_phase="play", p1_hand=["Guard"])
    action = agent.take_turn()

    assert action.reasoning == "private deduction"
    assert action.public_statement == "I accuse you!"
    assert "reasoning" not in action.parameters
    assert "public_statement" not in action.parameters
    assert action.parameters == {"target_player": "player_2", "named_card": "Priest"}


def test_observes_game_state_before_acting(settings) -> None:
    client = _draw_then_act("draw_card", {"reasoning": "r", "public_statement": "p"})
    agent = _make_agent(client, turn_phase="draw")
    agent.take_turn()

    roles_and_tools = [(m.get("role"), m.get("tool_call_id")) for m in agent.conversation_history]
    # A get_game_state tool result is in the history before the action returned.
    assert ("tool", "call_get_game_state") in roles_and_tools


def test_game_context_injected_as_user_message(settings) -> None:
    client = _draw_then_act("draw_card", {"reasoning": "r", "public_statement": "p"})
    agent = _make_agent(client, turn_phase="draw")
    agent.take_turn(game_context="Your last move was rejected: target is protected.")

    user_messages = [m["content"] for m in agent.conversation_history if m["role"] == "user"]
    assert any("rejected" in c for c in user_messages)


def test_loop_terminates_after_max_iterations(settings) -> None:
    # Client that never calls an action tool, only observes.
    never_acts = _FakeClient(
        [_FakeMessage([_FakeToolCall("get_game_state", {"reasoning": "again"})])]
    )
    agent = _make_agent(never_acts, turn_phase="draw")
    with pytest.raises(RuntimeError):
        agent.take_turn()


# --- Integration (real LLM) -------------------------------------------------


@pytest.mark.integration
def test_take_turn_draws_during_draw_phase(openai_client) -> None:
    from pathlib import Path

    from playtest.agents.gm import GMAgent
    from playtest.config import get_settings
    from playtest.ingestion.schemas import GameConfig

    config_dir = Path(get_settings().game_configs_dir) / "love_letter_classic"
    if not config_dir.exists():
        pytest.skip("love_letter_classic config not present; run ingestion first")

    config = GameConfig.load(str(config_dir))
    registry = ToolRegistry(config, GameStateManager())
    gm = GMAgent(config, registry, openai_client)
    gm.initialize_game(seed=1)

    player = PlayerAgent("player_1", config, registry, openai_client)
    action = player.take_turn()

    assert action.action_type == "draw_card"
