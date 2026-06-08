"""Assemble and compile the playtest StateGraph.

Routing between the GM and player nodes is dynamic via ``Command(goto=...)``; the only
fixed edge is ``START -> gm`` (initialization). ``assemble_graph`` is factored out so the
runner and the stub-based integration tests share the exact same topology.
"""

from collections.abc import Callable

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from openai import OpenAI

from playtest.agents.gm import GMAgent
from playtest.agents.player import PlayerAgent
from playtest.graph.nodes import build_gm_node, build_player_node
from playtest.graph.schema import PlaytestState
from playtest.ingestion.schemas import GameConfig
from playtest.state.manager import GameStateManager
from playtest.tools import ToolRegistry

_GraphNode = Callable[[PlaytestState], Command]


def assemble_graph(gm_node: _GraphNode, player_node: _GraphNode) -> CompiledStateGraph:
    graph = StateGraph(PlaytestState)
    # Command-returning nodes are valid at runtime; the langgraph stubs' add_node
    # overloads don't model the Callable[..., Command] shape, hence the ignores.
    graph.add_node("gm", gm_node)  # type: ignore[call-overload]
    graph.add_node("player", player_node)  # type: ignore[call-overload]
    graph.add_edge(START, "gm")
    # No gm <-> player edges: Command(goto=...) handles all routing.
    return graph.compile()


def build_playtest_graph(
    game_config: GameConfig,
    openai_client: OpenAI,
    state_manager: GameStateManager,
    tool_registry: ToolRegistry,
    *,
    num_players: int,
    seed: int | None,
) -> CompiledStateGraph:
    gm_agent = GMAgent(game_config, tool_registry, openai_client)
    player_agents = {
        f"player_{i}": PlayerAgent(f"player_{i}", game_config, tool_registry, openai_client)
        for i in range(1, num_players + 1)
    }
    gm_node = build_gm_node(gm_agent, state_manager, num_players=num_players, seed=seed)
    player_node = build_player_node(player_agents)
    return assemble_graph(gm_node, player_node)
