"""GM agent tests, offline: scripted clients drive the tool loop deterministically."""

import json
from types import SimpleNamespace

import pytest

from playtest.agents.gm import GMAgent
from playtest.errors import PlaytestError, StateInvariantViolation
from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry

from .fixtures import sample_config


def _make_agent(client: object) -> GMAgent:
    config = sample_config()
    registry = ToolRegistry(config, GameStateManager())
    return GMAgent(config, registry, client)  # type: ignore[arg-type]


# --- Offline fake client (deterministic, no network) ------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None

    def model_dump(self, **_: object) -> dict:
        return {"role": "assistant", "content": self.content}


class _FakeCompletions:
    def create(self, **_: object) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=_FakeMessage("A new game begins."))]
        )


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


# --- Scriptable fake client (drives the GM tool loop offline) ----------------


class _ToolCall:
    def __init__(self, name: str, args: dict, call_id: str = "call_1") -> None:
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))


class _ScriptedMessage:
    def __init__(self, content: str | None = None, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, **_: object) -> dict:
        d: dict = {"role": "assistant"}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        return d


class _ScriptedCompletions:
    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.seen_messages: list[list[dict]] = []

    def create(self, **kwargs: object) -> object:
        self.seen_messages.append(list(kwargs["messages"]))  # type: ignore[arg-type]
        message = self.script.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _ScriptedClient:
    """Replays a fixed sequence of assistant messages so the GM loop runs offline."""

    def __init__(self, script: list) -> None:
        self.completions = _ScriptedCompletions(script)
        self.chat = SimpleNamespace(completions=self.completions)


def _commit_and_finish(new_state: dict, **finish_args) -> list:
    """A single assistant message that commits a state then reports the resolution."""
    return [
        _ScriptedMessage(
            tool_calls=[
                _ToolCall("set_game_state", {"reasoning": "", "new_state": new_state}, "c1"),
                _ToolCall("finish_resolution", finish_args, "c2"),
            ]
        )
    ]


def _finish_only(**finish_args) -> list:
    return [_ScriptedMessage(tool_calls=[_ToolCall("finish_resolution", finish_args)])]


_DRAW = {"action_type": "draw_card", "parameters": {}, "reasoning": "", "public_statement": ""}


# --- Initialization -----------------------------------------------------------


def test_initialize_game_establishes_state(settings) -> None:
    agent = _make_agent(_FakeClient())
    result = agent.initialize_game(num_players=2, seed=7)

    assert "state" in result and "narration" in result
    gm_view = agent.state_manager.get_state("gm")
    assert gm_view["deck_count"] == len(gm_view["deck"])
    assert gm_view["players"]["player_1"]["hand_count"] == 1
    assert gm_view["removed_card"] not in ("", "HIDDEN")
    player_view = agent.state_manager.get_state("player_1")
    assert player_view["removed_card"] == "HIDDEN"
    assert "deck" not in player_view


def test_initialize_game_caps_player_count(settings) -> None:
    agent = _make_agent(_FakeClient())
    for n in (5, 6):
        with pytest.raises(ValueError, match="supports"):
            agent.initialize_game(num_players=n, seed=1)


def test_initialize_game_addendum_never_stacks(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=1)
    once = agent.system_prompt
    agent.initialize_game(num_players=2, seed=2)
    assert agent.system_prompt == once


# --- validate_and_resolve -------------------------------------------------------


def _after_draw_state(agent: GMAgent) -> dict:
    """player_1 draws the top deck card: hand grows to 2, phase flips to play."""
    state = agent.state_manager.get_state("gm")
    drawn = state["deck"][0]
    state["deck"] = state["deck"][1:]
    state["deck_count"] = len(state["deck"])
    state["players"]["player_1"]["hand"] = state["players"]["player_1"]["hand"] + [drawn]
    state["players"]["player_1"]["hand_count"] = 2
    state["turn_phase"] = "play"
    return state


def test_valid_resolution_offline(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    after = _after_draw_state(agent)
    agent.client = _ScriptedClient(
        _commit_and_finish(after, is_valid=True, narration="Drew a card.", next_phase="play")
    )

    resolution = agent.validate_and_resolve(_DRAW, "player_1")
    assert resolution.is_valid is True
    assert resolution.new_state is not None
    assert resolution.new_state["players"]["player_1"]["hand_count"] == 2
    assert resolution.next_phase == "play"


def test_invalid_action_returns_rejection(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    before = agent.state_manager.get_state("gm")
    agent.client = _ScriptedClient(
        _finish_only(
            is_valid=False,
            error_message="It is not player_2's turn.",
            narration="Rejected.",
        )
    )

    resolution = agent.validate_and_resolve(_DRAW, "player_2")
    assert resolution.is_valid is False
    assert "not player_2's turn" in (resolution.error_message or "")
    assert resolution.new_state is None
    assert agent.state_manager.get_state("gm") == before  # nothing committed


def test_invalid_but_committed_raises(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    after = _after_draw_state(agent)
    agent.client = _ScriptedClient(
        _commit_and_finish(after, is_valid=False, error_message="oops", narration="Rejected.")
    )
    with pytest.raises(PlaytestError, match="invalid"):
        agent.validate_and_resolve(_DRAW, "player_1")


def test_valid_without_commit_raises(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    agent.client = _ScriptedClient(
        _finish_only(is_valid=True, narration="Did a thing (but never committed).")
    )
    with pytest.raises(PlaytestError, match="never committed"):
        agent.validate_and_resolve(_DRAW, "player_1")


def test_malformed_finish_resolution_payload_raises(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    after = _after_draw_state(agent)
    agent.client = _ScriptedClient(
        _commit_and_finish(after, is_valid=True, narration="ok", private_info="not-an-object")
    )
    with pytest.raises(PlaytestError, match="malformed finish_resolution"):
        agent.validate_and_resolve(_DRAW, "player_1")


def test_committed_invariant_violation_raises(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)

    # Duplicate a card: conservation must crash the run after commit.
    state = agent.state_manager.get_state("gm")
    state["players"]["player_1"]["hand"] = state["players"]["player_1"]["hand"] + [
        state["deck"][0]
    ]
    state["players"]["player_1"]["hand_count"] = 2
    state["turn_phase"] = "play"
    agent.client = _ScriptedClient(
        _commit_and_finish(state, is_valid=True, narration="Played a card.")
    )

    with pytest.raises(StateInvariantViolation):
        agent.validate_and_resolve(_DRAW, "player_1")


def test_resolution_prompt_embeds_committed_state(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    committed = agent.state_manager.get_state("gm")
    after = _after_draw_state(agent)
    client = _ScriptedClient(
        _commit_and_finish(after, is_valid=True, narration="Drew a card.")
    )
    agent.client = client

    agent.validate_and_resolve(_DRAW, "player_1")
    user_message = client.completions.seen_messages[0][1]["content"]
    assert json.dumps(committed) in user_message
    assert "Call get_game_state" not in user_message  # no longer mandated


def test_turn_ended_passes_through(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    after = _after_draw_state(agent)
    agent.client = _ScriptedClient(
        _commit_and_finish(after, is_valid=True, narration="ok", turn_ended=True)
    )
    resolution = agent.validate_and_resolve(_DRAW, "player_1")
    assert resolution.turn_ended is True


# --- handle_round_end -------------------------------------------------------------


def _scored_state(agent: GMAgent, p1_tokens: int) -> dict:
    state = agent.state_manager.get_state("gm")
    state["players"]["player_1"]["tokens"] = p1_tokens
    return state


def test_handle_round_end_game_over(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    scored = _scored_state(agent, p1_tokens=7)
    agent.client = _ScriptedClient(
        _commit_and_finish(
            scored,
            is_valid=True,
            narration="player_1 wins it all.",
            winners=["player_1"],
            game_ended=True,
            winner="player_1",
        )
    )

    res = agent.handle_round_end()
    assert res.round_ended is True
    assert res.game_ended is True
    assert res.winner == "player_1"
    assert res.winners == ["player_1"]
    assert agent.state_manager.get_state("gm")["players"]["player_1"]["tokens"] == 7


def test_handle_round_end_continues_with_engine_redeal(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    scored = _scored_state(agent, p1_tokens=1)
    client = _ScriptedClient(
        _commit_and_finish(
            scored,
            is_valid=True,
            narration="player_1 takes the round.",
            winners=["player_1"],
            game_ended=False,
            next_player="player_2",
        )
    )
    agent.client = client

    res = agent.handle_round_end()
    assert res.round_ended is True
    assert res.game_ended is False
    assert res.winners == ["player_1"]
    new_state = agent.state_manager.get_state("gm")
    assert new_state["round_number"] == 2  # engine redealt
    assert new_state["players"]["player_1"]["tokens"] == 1  # carry-over preserved
    assert new_state["current_turn"] == "player_2"  # GM-chosen starter honored
    assert new_state["deck_count"] == len(new_state["deck"])
    # The scoring prompt embedded the committed state and the redeal was NOT the LLM's.
    user_message = client.completions.seen_messages[0][1]["content"]
    assert "Current committed game state" in user_message
    assert "do NOT deal a new round" in user_message


def test_handle_round_end_without_commit_raises(settings) -> None:
    agent = _make_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    agent.client = _ScriptedClient(
        _finish_only(is_valid=True, narration="Scored (but never committed).")
    )
    with pytest.raises(PlaytestError, match="did not commit"):
        agent.handle_round_end()
