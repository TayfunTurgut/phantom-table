"""Player agent tests: phase-based tool exposure, the turn loop, rejection retry, deltas."""

import copy
import json
from types import SimpleNamespace

import pytest

from playtest.agents.player import PlayerAction, PlayerAgent, _state_delta
from playtest.config import get_settings
from playtest.errors import PlaytestError
from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry

from .fixtures import sample_config, sample_spec


def _tool_names(schemas: list[dict]) -> set[str]:
    return {schema["function"]["name"] for schema in schemas}


def _make_agent(client: object, *, turn_phase: str = "draw") -> PlayerAgent:
    game_config = sample_config()
    manager = GameStateManager()
    state = copy.deepcopy(game_config.initial_state_template)
    state["turn_phase"] = turn_phase
    state["deck"] = ["Priest", "Baron"]
    state["deck_count"] = 2
    manager.initialize(state, sample_spec().visibility)
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
        self.seen_messages.append(list(kwargs["messages"]))  # type: ignore[arg-type]
        index = min(self._calls, len(self._messages) - 1)
        self._calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=self._messages[index])])


class _FakeClient:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self.completions = _ScriptedCompletions(messages)
        self.chat = SimpleNamespace(completions=self.completions)


def _manager_of(agent: PlayerAgent) -> GameStateManager:
    return agent.tool_registry.get_state_tool.manager


def _make_resolver(agent: PlayerAgent, record: list[PlayerAction], reject_first: int = 0):
    """Stub GM resolver: a draw flips to play (hand grows); a play ends the turn.

    The first ``reject_first`` calls are rejected, exercising the retry path.
    """
    manager = _manager_of(agent)
    rejected = {"left": reject_first}

    def resolve(action: PlayerAction) -> dict:
        record.append(action)
        if rejected["left"] > 0:
            rejected["left"] -= 1
            return {"rejected": True, "error_message": "that move is illegal here"}
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


def _take_turn(agent: PlayerAgent, record: list[PlayerAction], resolver=None, **kwargs):
    return agent.take_turn(
        _manager_of(agent).get_state("player_1"),
        resolve_action=resolver or _make_resolver(agent, record),
        **kwargs,
    )


# --- Tool exposure tests (no LLM) -------------------------------------------


def test_draw_phase_exposes_only_draw_card(settings) -> None:
    agent = _make_agent(_FakeClient([]), turn_phase="draw")
    names = _tool_names(agent._get_available_tools({"turn_phase": "draw"}))
    assert names == {"query_rulebook", "get_game_state", "draw_card"}


def test_play_phase_exposes_all_play_tools(settings) -> None:
    """Phase-level filtering only: legality within the phase is the GM's judgment."""
    agent = _make_agent(_FakeClient([]), turn_phase="play")
    names = _tool_names(agent._get_available_tools({"turn_phase": "play"}))
    assert names == {
        "query_rulebook",
        "get_game_state",
        "play_guard",
        "play_prince",
        "play_king",
        "play_countess",
    }


# --- take_turn behavior (scripted fake client) ------------------------------


def test_take_turn_draws_then_plays(settings) -> None:
    # Observe, draw (phase flips to play, hand grows), then play — all in one invocation.
    client = _FakeClient(
        [
            _FakeMessage([_FakeToolCall("get_game_state", {"reasoning": "look"})]),
            _FakeMessage([_FakeToolCall("draw_card", {"reasoning": "r", "public_statement": "p"})]),
            _FakeMessage(
                [
                    _FakeToolCall(
                        "play_guard",
                        {
                            "target_player": "player_2",
                            "named_card": "Priest",
                            "reasoning": "r",
                            "public_statement": "p",
                        },
                    )
                ]
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
    agent = _make_agent(client, turn_phase="play")
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
                [
                    _FakeToolCall(
                        "play_guard",
                        {
                            "target_player": "player_2",
                            "named_card": "Priest",
                            "reasoning": "r",
                            "public_statement": "p",
                        },
                    )
                ]
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
                call = _FakeToolCall(
                    "play_king",
                    {"target_player": "player_2", "reasoning": "r", "public_statement": "s"},
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=_FakeMessage([call]))])

    class _DeadlineClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=_DeadlineCompletions())

    agent = _make_agent(_DeadlineClient(), turn_phase="play")
    record: list[PlayerAction] = []
    action = _take_turn(agent, record)

    assert action.action_type == "play_king"
    # The first N calls still offered observation tools; the deadline withdrew them after.
    assert all("get_game_state" in s for s in offered_log[:max_obs])
    assert "get_game_state" not in offered_log[max_obs]


# --- Rejection retry + delta feedback -----------------------------------------


def test_rejected_action_feeds_error_back_and_player_retries(settings) -> None:
    client = _FakeClient(
        [
            _FakeMessage(
                [
                    _FakeToolCall(
                        "play_king",
                        {"target_player": "player_2", "reasoning": "r", "public_statement": "p"},
                    )
                ]
            ),
            _FakeMessage(
                [
                    _FakeToolCall(
                        "play_guard",
                        {
                            "target_player": "player_2",
                            "named_card": "Priest",
                            "reasoning": "r",
                            "public_statement": "p",
                        },
                    )
                ]
            ),
        ]
    )
    agent = _make_agent(client, turn_phase="play")
    record: list[PlayerAction] = []
    action = _take_turn(agent, record, resolver=_make_resolver(agent, record, reject_first=1))

    # Both proposals reached the resolver; the second succeeded and ended the turn.
    assert [a.action_type for a in record] == ["play_king", "play_guard"]
    assert action is not None and action.action_type == "play_guard"
    # The rejection was fed back verbatim as a tool message.
    rejection_feedback = client.completions.seen_messages[1][-1]["content"]
    assert "REJECTED" in rejection_feedback
    assert "that move is illegal here" in rejection_feedback
    assert "has not changed" in rejection_feedback


def test_resolution_feedback_sends_delta_not_full_state(settings) -> None:
    client = _FakeClient(
        [
            _FakeMessage([_FakeToolCall("draw_card", {"reasoning": "r", "public_statement": "p"})]),
            _FakeMessage(
                [
                    _FakeToolCall(
                        "play_guard",
                        {
                            "target_player": "player_2",
                            "named_card": "Priest",
                            "reasoning": "r",
                            "public_statement": "p",
                        },
                    )
                ]
            ),
        ]
    )
    agent = _make_agent(client, turn_phase="draw")
    record: list[PlayerAction] = []
    _take_turn(agent, record)

    first_prompt = client.completions.seen_messages[0][1]["content"]
    assert "Current game state (your view)" in first_prompt  # first prompt stays full

    draw_feedback = client.completions.seen_messages[1][-1]["content"]
    assert "State changes since your last view" in draw_feedback
    delta = json.loads(draw_feedback.split("State changes since your last view (your view):\n")[1])
    assert delta["turn_phase"] == "play"
    assert delta["players"]["player_1"]["hand_count"] == 2
    # Unchanged parts of the state are not re-sent.
    assert "player_2" not in delta.get("players", {})
    assert "revealed_cards" not in delta


def test_state_delta_helper() -> None:
    old = {
        "turn_phase": "draw",
        "deck_count": 5,
        "players": {
            "player_1": {"hand_count": 1, "tokens": 0},
            "player_2": {"hand_count": 1, "tokens": 0},
        },
    }
    new = {
        "turn_phase": "play",
        "deck_count": 4,
        "players": {
            "player_1": {"hand_count": 2, "tokens": 0},
            "player_2": {"hand_count": 1, "tokens": 0},
        },
    }
    assert _state_delta(old, new) == {
        "turn_phase": "play",
        "deck_count": 4,
        "players": {"player_1": {"hand_count": 2}},
    }
    assert _state_delta(new, new) == {}
