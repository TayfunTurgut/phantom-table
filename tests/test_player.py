import copy
import json
from types import SimpleNamespace

import pytest

from playtest.agents.player import PlayerAction, PlayerAgent
from playtest.config import get_settings
from playtest.errors import PlaytestError
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


def _state(turn_phase: str, hand: list[str]) -> dict:
    return {"turn_phase": turn_phase, "players": {"player_1": {"hand": hand}}}


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
        self.seen_messages: list[list[dict]] = []

    def create(self, **kwargs: object) -> object:
        # Every player LLM call must force a tool call with no parallelism.
        assert kwargs["tool_choice"] == "required"
        assert kwargs["parallel_tool_calls"] is False
        self.seen_messages.append(kwargs["messages"])  # type: ignore[arg-type]
        index = min(self._calls, len(self._messages) - 1)
        self._calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=self._messages[index])])


class _FakeClient:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self.completions = _ScriptedCompletions(messages)
        self.chat = SimpleNamespace(completions=self.completions)


def _manager_of(agent: PlayerAgent) -> GameStateManager:
    return agent.tool_registry.get_state_tool.manager


def _make_resolver(agent: PlayerAgent, record: list[PlayerAction]):
    """Stub GM resolver: a draw flips to play (hand grows); a play ends the turn."""
    manager = _manager_of(agent)

    def resolve(action: PlayerAction) -> dict:
        record.append(action)
        view = copy.deepcopy(manager.get_state(action.player_id))
        if action.action_type == "draw_card":
            view["turn_phase"] = "play"
            hand = view["players"][action.player_id]["hand"]
            view["players"][action.player_id]["hand"] = hand + ["Guard"]
            view["players"][action.player_id]["hand_count"] = len(hand) + 1
            turn_ended = False
        else:
            turn_ended = True
        return {
            "filtered_state": view,
            "narration": "ok",
            "private_info": None,
            "turn_ended": turn_ended,
        }

    return resolve


def _take_turn(agent: PlayerAgent, record: list[PlayerAction], **kwargs):
    return agent.take_turn(
        _manager_of(agent).get_state("player_1"),
        resolve_action=_make_resolver(agent, record),
        **kwargs,
    )


# --- Tool exposure tests (no LLM) -------------------------------------------


def test_draw_phase_exposes_only_draw_card(settings) -> None:
    agent = _make_agent(_FakeClient([]), turn_phase="draw")
    names = _tool_names(agent._get_available_tools(_state("draw", ["King"])))
    assert names == {"query_rulebook", "get_game_state", "draw_card"}


def test_play_phase_exposes_only_matching_play_tools(settings) -> None:
    agent = _make_agent(_FakeClient([]), turn_phase="play")
    names = _tool_names(agent._get_available_tools(_state("play", ["Guard", "Prince"])))
    assert names == {"query_rulebook", "get_game_state", "play_guard", "play_prince"}


def test_countess_forced_play_with_king(settings) -> None:
    agent = _make_agent(_FakeClient([]), turn_phase="play")
    names = _tool_names(agent._get_available_tools(_state("play", ["Countess", "King"])))
    action_tools = names - {"query_rulebook", "get_game_state"}
    assert action_tools == {"play_countess"}


def test_countess_forced_play_with_prince(settings) -> None:
    agent = _make_agent(_FakeClient([]), turn_phase="play")
    names = _tool_names(agent._get_available_tools(_state("play", ["Countess", "Prince"])))
    action_tools = names - {"query_rulebook", "get_game_state"}
    assert action_tools == {"play_countess"}


# --- take_turn behavior (scripted fake client) ------------------------------


def test_take_turn_draws_then_plays(settings) -> None:
    # Observe, draw (phase flips to play, hand grows), then play — all in one invocation.
    client = _FakeClient(
        [
            _FakeMessage([_FakeToolCall("get_game_state", {"reasoning": "look"})]),
            _FakeMessage([_FakeToolCall("draw_card", {"reasoning": "r", "public_statement": "p"})]),
            _FakeMessage(
                [_FakeToolCall("play_guard", {"reasoning": "r", "public_statement": "p"})]
            ),
        ]
    )
    agent = _make_agent(client, turn_phase="draw")
    record: list[PlayerAction] = []
    action = _take_turn(agent, record)

    assert [a.action_type for a in record] == ["draw_card", "play_guard"]
    assert isinstance(action, PlayerAction)
    assert action.action_type == "play_guard"  # take_turn returns the last (turn-ending) action


def test_reasoning_and_public_statement_captured_separately(settings) -> None:
    client = _FakeClient(
        [
            _FakeMessage([_FakeToolCall("get_game_state", {"reasoning": "look"})]),
            _FakeMessage(
                [
                    _FakeToolCall(
                        "play_guard",
                        {
                            "target_player": "player_2",
                            "named_card": "Priest",
                            "reasoning": "private deduction",
                            "public_statement": "I accuse you!",
                        },
                    )
                ]
            ),
        ]
    )
    agent = _make_agent(client, turn_phase="play", p1_hand=["Guard"])
    record: list[PlayerAction] = []
    action = _take_turn(agent, record)

    assert action.reasoning == "private deduction"
    assert action.public_statement == "I accuse you!"
    assert "reasoning" not in action.parameters
    assert "public_statement" not in action.parameters
    assert action.parameters == {"target_player": "player_2", "named_card": "Priest"}


def test_context_and_private_memory_injected_into_prompt(settings) -> None:
    client = _FakeClient(
        [
            _FakeMessage([_FakeToolCall("draw_card", {"reasoning": "r", "public_statement": "p"})]),
            _FakeMessage(
                [_FakeToolCall("play_guard", {"reasoning": "r", "public_statement": "p"})]
            ),
        ]
    )
    agent = _make_agent(client, turn_phase="draw")
    record: list[PlayerAction] = []
    _take_turn(
        agent,
        record,
        context="Your last move was rejected: target is protected.",
        private_memory=["player_2 was holding the Baron last round"],
    )

    first_prompt = client.completions.seen_messages[0][1]["content"]
    assert "rejected" in first_prompt
    assert "Baron" in first_prompt


def test_loop_terminates_after_max_iterations(settings) -> None:
    # Client that never completes a turn (only observes) must crash, not hang.
    never_acts = _FakeClient(
        [_FakeMessage([_FakeToolCall("get_game_state", {"reasoning": "again"})])]
    )
    agent = _make_agent(never_acts, turn_phase="draw")
    with pytest.raises(PlaytestError):
        _take_turn(agent, [])


def test_forced_action_deadline_narrows_tools_and_commits(settings) -> None:
    """A chronic over-querier is forced to act once observation tools are withdrawn."""
    max_obs = get_settings().max_observation_calls
    offered_log: list[set[str]] = []

    class _DeadlineCompletions:
        def create(self, **kwargs: object) -> object:
            offered = {t["function"]["name"] for t in kwargs["tools"]}  # type: ignore[union-attr]
            offered_log.append(offered)
            if "get_game_state" in offered:
                call = _FakeToolCall("get_game_state", {"reasoning": "again"})
            else:
                call = _FakeToolCall("play_king", {"reasoning": "r", "public_statement": "s"})
            return SimpleNamespace(choices=[SimpleNamespace(message=_FakeMessage([call]))])

    class _DeadlineClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=_DeadlineCompletions())

    agent = _make_agent(_DeadlineClient(), turn_phase="play", p1_hand=["King"])
    record: list[PlayerAction] = []
    action = _take_turn(agent, record)

    assert action.action_type == "play_king"
    # The first N calls still offered observation tools; the deadline withdrew them after.
    assert all("get_game_state" in s for s in offered_log[:max_obs])
    assert "get_game_state" not in offered_log[max_obs]


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
    manager = registry.get_state_tool.manager
    player = PlayerAgent("player_1", config, registry, openai_client)

    record: list[PlayerAction] = []

    def resolve(action: PlayerAction) -> dict:
        record.append(action)
        res = gm.validate_and_resolve(action.model_dump(), "player_1")
        return {
            "filtered_state": manager.get_state("player_1"),
            "narration": res.narration,
            "private_info": res.private_info,
            "turn_ended": action.action_type.startswith("play_"),
        }

    player.take_turn(manager.get_state("player_1"), resolve_action=resolve)
    assert record[0].action_type == "draw_card"
