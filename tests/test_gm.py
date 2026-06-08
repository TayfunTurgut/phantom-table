import collections
from pathlib import Path
from types import SimpleNamespace

import pytest

from playtest.agents.gm import DECK_COMPOSITION, TOKENS_TO_WIN, GMAgent
from playtest.config import get_settings
from playtest.ingestion.schemas import GameConfig
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

    resolution = agent.validate_and_resolve(
        {"action_type": "draw_card", "parameters": {}, "reasoning": "", "public_statement": ""},
        "player_2",
    )
    assert resolution.is_valid is False
    assert resolution.error_message


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

    resolution = agent.validate_and_resolve(
        {
            "action_type": "play_king",
            "parameters": {"target_player": "player_2"},
            "reasoning": "",
            "public_statement": "I play the King.",
        },
        "player_1",
    )
    assert resolution.is_valid is False
    assert resolution.error_message


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
        {"role": "user", "content": "Call get_game_state to inspect the board, then reply done."},
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
