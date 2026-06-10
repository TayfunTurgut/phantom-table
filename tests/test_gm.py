import collections
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from playtest.agents.gm import GMAgent
from playtest.config import get_settings
from playtest.errors import IllegalAction, PlaytestError, StateInvariantViolation
from playtest.ingestion.schemas import GameConfig
from playtest.rules.love_letter import DECK_COMPOSITION, TOKENS_TO_WIN
from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry


def _config_dir() -> Path:
    return Path(get_settings().game_configs_dir) / "love_letter_classic"


def _load_agent(client: object) -> GMAgent:
    config = GameConfig.load(str(_config_dir()))
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

    def create(self, **_: object) -> object:
        message = self.script.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _ScriptedClient:
    """Replays a fixed sequence of assistant messages so the GM loop runs offline."""

    def __init__(self, script: list) -> None:
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(script))


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


# --- Offline tests (deterministic helpers) ----------------------------------


def test_build_initial_state_composition(settings) -> None:
    agent = _load_agent(_FakeClient())
    state, removed = agent._build_initial_state(num_players=2, seed=42)

    all_cards = (
        list(state["deck"])
        + [removed]
        + list(state["revealed_cards"])
        + [c for p in state["players"].values() for c in p["hand"]]
    )
    assert collections.Counter(all_cards) == collections.Counter(DECK_COMPOSITION)
    assert len(state["deck"]) == 16 - 1 - 3 - 2
    assert state["deck_count"] == len(state["deck"])
    assert state["tokens_to_win"] == TOKENS_TO_WIN[2]
    assert state["round_number"] == 1
    assert state["current_turn"] == "player_1"
    assert state["turn_phase"] == "draw"
    assert len(state["revealed_cards"]) == 3
    for player in state["players"].values():
        assert player["hand_count"] == 1
        assert len(player["hand"]) == 1


def test_build_initial_state_three_players(settings) -> None:
    agent = _load_agent(_FakeClient())
    state, removed = agent._build_initial_state(num_players=3, seed=1)
    assert state["revealed_cards"] == []  # reveal only happens for 2 players
    assert len(state["deck"]) == 16 - 1 - 0 - 3
    assert state["tokens_to_win"] == TOKENS_TO_WIN[3]
    assert set(state["players"]) == {"player_1", "player_2", "player_3"}


def test_build_initial_state_reproducible(settings) -> None:
    agent = _load_agent(_FakeClient())
    a = agent._build_initial_state(2, seed=99)
    b = agent._build_initial_state(2, seed=99)
    c = agent._build_initial_state(2, seed=100)
    assert a == b
    assert a != c


def test_initialize_game_establishes_state(settings) -> None:
    agent = _load_agent(_FakeClient())
    result = agent.initialize_game(seed=7)

    assert "state" in result and "narration" in result
    manager = agent.state_manager
    gm_view = manager.get_state("gm")
    assert "deck" in gm_view
    assert gm_view["deck_count"] == len(gm_view["deck"])
    assert gm_view["players"]["player_1"]["hand_count"] == 1
    assert manager.get_removed_card() not in (None, "HIDDEN")


def test_build_initial_state_four_players(settings) -> None:
    agent = _load_agent(_FakeClient())
    state, _ = agent._build_initial_state(num_players=4, seed=2)
    assert state["revealed_cards"] == []  # reveal only for 2 players
    assert len(state["deck"]) == 16 - 1 - 0 - 4
    assert state["tokens_to_win"] == TOKENS_TO_WIN[4]
    assert set(state["players"]) == {"player_1", "player_2", "player_3", "player_4"}


def test_initialize_game_caps_player_count(settings) -> None:
    agent = _load_agent(_FakeClient())
    for n in (5, 6):
        with pytest.raises(ValueError, match="2, 3, 4"):
            agent.initialize_game(num_players=n, seed=1)


def _force_round_end_state(agent: GMAgent, p1_tokens: int) -> None:
    """Contrive an empty-deck end: player_1 holds Princess, player_2 a Guard."""
    state = agent.state_manager.get_state("gm")
    state["players"]["player_1"]["hand"] = ["Princess"]
    state["players"]["player_1"]["hand_count"] = 1
    state["players"]["player_1"]["tokens"] = p1_tokens
    state["players"]["player_2"]["hand"] = ["Guard"]
    state["players"]["player_2"]["hand_count"] = 1
    state["deck"] = []
    state["deck_count"] = 0
    agent.state_manager.set_state(state)


def test_handle_round_end_resolver_ends_game(settings) -> None:
    agent = _load_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)  # tokens_to_win == 7
    _force_round_end_state(agent, p1_tokens=6)

    res = agent.handle_round_end()
    assert res.round_ended is True
    assert res.game_ended is True
    assert res.winner == "player_1"
    assert res.winners == ["player_1"]
    assert res.winning_card == "Princess"
    assert agent.state_manager.get_state("gm")["players"]["player_1"]["tokens"] == 7


def test_handle_round_end_resolver_continues(settings) -> None:
    agent = _load_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    _force_round_end_state(agent, p1_tokens=0)

    # Engine awards player_1 a token; the (scripted) GM then deals a valid next round.
    dealt, _ = agent._build_initial_state(num_players=2, seed=5)
    dealt["round_number"] = 2
    dealt["current_turn"] = "player_1"
    dealt["players"]["player_1"]["tokens"] = 1
    dealt["players"]["player_2"]["tokens"] = 0
    agent.client = _ScriptedClient(
        _commit_and_finish(dealt, is_valid=True, narration="The next round is dealt.")
    )

    res = agent.handle_round_end()
    assert res.round_ended is True
    assert res.game_ended is False
    assert res.winners == ["player_1"]
    assert agent.state_manager.get_state("gm")["players"]["player_1"]["tokens"] == 1
    assert agent.state_manager.get_state("gm")["round_number"] == 2


# --- Offline pure-crash / resolution tests (scripted client) ----------------


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


def test_valid_draw_resolution_offline(settings) -> None:
    agent = _load_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    after = _after_draw_state(agent)
    agent.client = _ScriptedClient(
        _commit_and_finish(after, is_valid=True, narration="Drew a card.", next_phase="play")
    )

    resolution = agent.validate_and_resolve(
        {"action_type": "draw_card", "parameters": {}, "reasoning": "", "public_statement": ""},
        "player_1",
    )
    assert resolution.is_valid is True
    assert resolution.new_state is not None
    assert resolution.new_state["players"]["player_1"]["hand_count"] == 2


def test_illegal_action_raises(settings) -> None:
    agent = _load_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    agent.client = _ScriptedClient(
        [
            _ScriptedMessage(
                tool_calls=[
                    _ToolCall(
                        "finish_resolution",
                        {
                            "is_valid": False,
                            "error_message": "It is not player_2's turn.",
                            "narration": "Rejected.",
                        },
                    )
                ]
            )
        ]
    )
    with pytest.raises(IllegalAction):
        agent.validate_and_resolve(
            {"action_type": "draw_card", "parameters": {}, "reasoning": "", "public_statement": ""},
            "player_2",
        )


def test_valid_without_commit_raises(settings) -> None:
    agent = _load_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)
    agent.client = _ScriptedClient(
        [
            _ScriptedMessage(
                tool_calls=[
                    _ToolCall(
                        "finish_resolution",
                        {"is_valid": True, "narration": "Did a thing (but never committed)."},
                    )
                ]
            )
        ]
    )
    with pytest.raises(PlaytestError):
        agent.validate_and_resolve(
            {"action_type": "draw_card", "parameters": {}, "reasoning": "", "public_statement": ""},
            "player_1",
        )


def test_committed_invariant_violation_raises(settings) -> None:
    agent = _load_agent(_FakeClient())
    agent.initialize_game(num_players=2, seed=5)

    # Contrive the motivating bug: the actor "plays" a card but ends holding 2 cards.
    state = agent.state_manager.get_state("gm")
    deck = list(state["deck"])
    extra = deck.pop(0)
    played = deck.pop(0)
    state["deck"] = deck
    state["deck_count"] = len(deck)
    p1 = state["players"]["player_1"]
    p1["hand"] = list(p1["hand"]) + [extra]  # still 2 cards after the "play"
    p1["hand_count"] = 2
    p1["discards"] = [played]
    state["turn_phase"] = "play"
    agent.client = _ScriptedClient(
        _commit_and_finish(state, is_valid=True, narration="Played a card.")
    )

    with pytest.raises(StateInvariantViolation):
        agent.validate_and_resolve(
            {
                "action_type": f"play_{played.lower()}",
                "parameters": {},
                "reasoning": "",
                "public_statement": "",
            },
            "player_1",
        )


# --- Integration tests (real LLM, committed config) -------------------------


def _require_config() -> None:
    if not _config_dir().exists():
        pytest.skip("love_letter_classic config not present; run ingestion first")


def _set_play_state(
    agent: GMAgent, hands: dict[str, list[str]], deck: list[str] | None = None
) -> dict:
    """Contrive a deterministic play-phase state for player_1 (test harness, GM caller)."""
    state = agent.state_manager.get_state("gm")
    for pid, hand in hands.items():
        state["players"][pid]["hand"] = list(hand)
        state["players"][pid]["hand_count"] = len(hand)
    if deck is not None:
        state["deck"] = list(deck)
        state["deck_count"] = len(deck)
    state["current_turn"] = "player_1"
    state["turn_phase"] = "play"
    return agent.state_manager.set_state(state)


@pytest.mark.integration
def test_valid_draw(openai_client) -> None:
    _require_config()
    agent = _load_agent(openai_client)
    agent.initialize_game(seed=1)

    resolution = agent.validate_and_resolve(
        {"action_type": "draw_card", "parameters": {}, "reasoning": "", "public_statement": ""},
        "player_1",
    )
    assert resolution.is_valid is True
    assert resolution.new_state is not None
    assert resolution.new_state["players"]["player_1"]["hand_count"] == 2


@pytest.mark.integration
def test_invalid_wrong_turn(openai_client) -> None:
    _require_config()
    agent = _load_agent(openai_client)
    agent.initialize_game(seed=1)

    with pytest.raises(IllegalAction):
        agent.validate_and_resolve(
            {
                "action_type": "draw_card",
                "parameters": {},
                "reasoning": "",
                "public_statement": "",
            },
            "player_2",
        )


@pytest.mark.integration
def test_guard_correct_guess_eliminates(openai_client) -> None:
    _require_config()
    agent = _load_agent(openai_client)
    agent.initialize_game(seed=1)
    _set_play_state(agent, {"player_1": ["Guard", "King"], "player_2": ["Priest"]})

    resolution = agent.validate_and_resolve(
        {
            "action_type": "play_guard",
            "parameters": {"target_player": "player_2", "named_card": "Priest"},
            "reasoning": "",
            "public_statement": "I name Priest.",
        },
        "player_1",
    )
    assert resolution.is_valid is True
    assert resolution.new_state is not None
    assert resolution.new_state["players"]["player_2"]["is_eliminated"] is True


@pytest.mark.integration
def test_countess_forced_rejected(openai_client) -> None:
    _require_config()
    agent = _load_agent(openai_client)
    agent.initialize_game(seed=1)
    _set_play_state(agent, {"player_1": ["Countess", "King"], "player_2": ["Guard"]})

    with pytest.raises(IllegalAction):
        agent.validate_and_resolve(
            {
                "action_type": "play_king",
                "parameters": {"target_player": "player_2"},
                "reasoning": "",
                "public_statement": "I play the King.",
            },
            "player_1",
        )


@pytest.mark.integration
def test_multi_turn_sequence(openai_client) -> None:
    _require_config()
    agent = _load_agent(openai_client)
    agent.initialize_game(seed=3)

    # Turn 1: player_1 draws, then plays.
    draw = agent.validate_and_resolve(
        {"action_type": "draw_card", "parameters": {}, "reasoning": "", "public_statement": ""},
        "player_1",
    )
    assert draw.is_valid is True
    state = agent.state_manager.get_state("gm")
    assert state["turn_phase"] == "play"

    # Play a Handmaid-safe card: pick whatever is in hand to keep it legal -> use a card present.
    hand = state["players"]["player_1"]["hand"]
    # Prefer Handmaid (no target needed); else fall back to the first card.
    card = "Handmaid" if "Handmaid" in hand else hand[0]
    action = f"play_{card.lower()}"
    params: dict = (
        {} if card in ("Handmaid", "Countess", "Princess") else {"target_player": "player_2"}
    )
    if card == "Guard":
        params["named_card"] = "Priest"
    play = agent.validate_and_resolve(
        {"action_type": action, "parameters": params, "reasoning": "", "public_statement": ""},
        "player_1",
    )

    players = agent.state_manager.get_state("gm")["players"]
    for res in (draw, play):
        if res.next_player is not None:
            assert res.next_player in players
            assert players[res.next_player]["is_eliminated"] is False


@pytest.mark.integration
def test_self_validation_present(openai_client) -> None:
    _require_config()
    agent = _load_agent(openai_client)
    agent.initialize_game(seed=1)

    agent.validate_and_resolve(
        {"action_type": "draw_card", "parameters": {}, "reasoning": "", "public_statement": ""},
        "player_1",
    )
    # The most recent resolution should have queried state/rulebook during the loop.
    # Re-run capturing messages via a direct loop call to inspect tool usage.
    messages = [
        {"role": "system", "content": agent.system_prompt},
        {
            "role": "user",
            "content": (
                "Call get_game_state to inspect the board, then call finish_resolution "
                "with is_valid=true and a short narration."
            ),
        },
    ]
    loop = agent._call_llm(messages, tools=agent.tools)
    tool_names = [
        tc["function"]["name"]
        for m in loop["messages"]
        if isinstance(m, dict)
        for tc in (m.get("tool_calls") or [])
    ]
    assert "get_game_state" in tool_names


@pytest.mark.integration
def test_handle_round_end_awards_token(openai_client) -> None:
    _require_config()
    agent = _load_agent(openai_client)
    agent.initialize_game(seed=5)
    # Contrive an empty-deck end state: player_1 holds the higher card.
    _set_play_state(agent, {"player_1": ["Princess"], "player_2": ["Guard"]}, deck=[])

    before = agent.state_manager.get_state("gm")["players"]["player_1"]["tokens"]
    resolution = agent.handle_round_end()

    assert resolution.round_ended is True
    after = agent.state_manager.get_state("gm")["players"]["player_1"]["tokens"]
    assert after == before + 1
