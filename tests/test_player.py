"""Player agent tests with a stubbed LLM client (no network)."""

import json

import pytest

from playtest.agents.player import PlayerAgent
from playtest.engine import Action

from .stubs import StubLLMClient

LEGAL = [
    Action(
        seat="player_1",
        name="play_guard",
        args={"card": "Guard", "target": "player_2", "guess": "Baron"},
        label="Play Guard: guess Baron",
    ),
    Action(
        seat="player_1",
        name="play_priest",
        args={"card": "Priest", "target": "player_2"},
        label="Play Priest",
    ),
]


def make_agent(responses, **kwargs):
    client = StubLLMClient(responses)
    agent = PlayerAgent("player_1", client, game_name="Test Game", **kwargs)
    return agent, client


def _choice(index, reasoning="", table_talk=None, notes=""):
    return json.dumps(
        {"action_index": index, "reasoning": reasoning, "table_talk": table_talk, "notes": notes}
    )


def test_valid_choice_returns_action_and_metadata():
    agent, client = make_agent(
        [_choice(1, reasoning="info is power", table_talk="hm", notes="p2 holds high")]
    )
    decision = agent.choose({"you": "player_1"}, LEGAL, ["something happened"])
    assert decision.action == LEGAL[1]
    assert decision.reasoning == "info is power"
    assert decision.table_talk == "hm"
    assert decision.notes == "p2 holds high"
    assert not decision.confused
    # The prompt contained the events and the numbered menu; schema was enforced.
    call = client.calls[0]
    assert call["role"] == "player"
    assert call["json_schema"]["name"] == "choose_action"
    user = call["messages"][1]["content"]
    assert "something happened" in user
    assert "0. Play Guard: guess Baron" in user


def test_notes_round_trip_into_next_prompt():
    agent, client = make_agent(
        [_choice(0, notes="saw the Princess in p2's hand"), _choice(1, notes="...")]
    )
    agent.choose({}, LEGAL, [])
    assert "(empty — this is your first decision)" in client.calls[0]["messages"][1]["content"]

    agent.choose({}, LEGAL, [])
    assert "saw the Princess in p2's hand" in client.calls[1]["messages"][1]["content"]


def test_confused_fallback_keeps_last_good_notebook():
    agent, _ = make_agent([_choice(0, notes="keep me"), "not json at all", "still not json"])
    agent.choose({}, LEGAL, [])
    decision = agent.choose({}, LEGAL, [])
    assert decision.confused
    assert decision.notes == "keep me"
    assert agent.notes == "keep me"


def test_out_of_range_choice_gets_one_retry():
    agent, client = make_agent([_choice(99), _choice(0, reasoning="fixed")])
    decision = agent.choose({}, LEGAL, [])
    assert decision.action == LEGAL[0]
    assert not decision.confused
    assert len(client.calls) == 2
    assert "out of range" in client.calls[1]["messages"][-1]["content"]


def test_boolean_action_index_is_invalid():
    # JSON true/false must not be accepted as indices 1/0.
    agent, _ = make_agent([_choice(True), _choice(False)])
    decision = agent.choose({}, LEGAL, [])
    assert decision.action == LEGAL[0]
    assert decision.confused


def test_double_failure_falls_back_to_first_action_as_confused():
    agent, _ = make_agent(["not json at all", _choice(-3)])
    decision = agent.choose({}, LEGAL, [])
    assert decision.action == LEGAL[0]
    assert decision.confused


def test_unparseable_choice_logs_a_warning(caplog):
    agent, _ = make_agent(["not json at all", "still not json"])
    with caplog.at_level("WARNING", logger="playtest.agents.player"):
        decision = agent.choose({}, LEGAL, [])
    assert decision.confused
    assert any("unparseable choice JSON" in r.message for r in caplog.records)


def test_empty_legal_set_is_a_harness_bug():
    agent, _ = make_agent([])
    with pytest.raises(ValueError, match="empty legal set"):
        agent.choose({}, [], [])


class FakeRulebook:
    def __init__(self):
        self.queries = []

    def query(self, q, n_results=3):
        self.queries.append(q)
        return "the Guard guesses a card"


def test_rulebook_becomes_a_tool_when_supported():
    rulebook = FakeRulebook()
    agent, client = make_agent([_choice(0)], rulebook=rulebook)
    agent.choose({}, LEGAL, [])

    tools = client.calls[0]["tools"]
    assert tools is not None and tools[0].name == "query_rulebook"
    # The handler is wired to the rulebook.
    result = tools[0].handler({"query": "what does the guard do", "reasoning": "check"})
    assert result == "the Guard guesses a card"
    assert rulebook.queries == ["what does the guard do"]


def test_rulebook_dropped_when_backend_lacks_tools():
    client = StubLLMClient([_choice(0)], supports_tools=False)
    agent = PlayerAgent("player_1", client, game_name="Test Game", rulebook=FakeRulebook())
    agent.choose({}, LEGAL, [])
    assert client.calls[0]["tools"] is None


def test_archetype_overlay_lands_in_system_prompt():
    agent, _ = make_agent([_choice(0)], archetype="aggressive")
    assert "aggressive player" in agent.system_prompt
    decision = agent.choose({}, LEGAL, [])
    assert decision.action == LEGAL[0]
