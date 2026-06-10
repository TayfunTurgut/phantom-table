"""Game Master (GM) agent: validates, resolves, and narrates player actions.

The GM is the central authority. It is an LLM agent with three tools
(query_rulebook, get_game_state, set_game_state) and no player-action tools — it
reasons about actions in natural language grounded by the rulebook, then writes
the full game state. Quality over speed: it self-validates against the rulebook
before committing any state change.
"""

import json
import random
from typing import Any

from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console

from playtest.config import get_settings
from playtest.errors import IllegalAction, PlaytestError, StateInvariantViolation
from playtest.ingestion.schemas import GameConfig
from playtest.rules import get_rules
from playtest.tools import ToolRegistry

_console = Console()

_MAX_TOOL_ITERATIONS = 12

_STATE_WRITE_INSTRUCTION = (
    "When you commit a state change, FIRST call get_game_state to read the complete "
    "current state, then build new_state by copying that ENTIRE object and changing "
    "only the fields the action affects. Never drop, rename, or retype any key — keep "
    "every key (including the 'deck' array) and the exact types you received."
)



class GMResolution(BaseModel):
    """The outcome of validating/resolving a proposed action."""

    is_valid: bool
    error_message: str | None = None
    action_summary: str | None = None
    narration: str | None = None
    new_state: dict | None = None
    round_ended: bool = False
    game_ended: bool = False
    winner: str | None = None
    winners: list[str] | None = None
    winning_card: str | None = None
    next_player: str | None = None
    next_phase: str = "draw"
    private_info: dict | None = None
    gm_reasoning: str | None = None


class GMAgent:
    """LLM Game Master that validates, resolves, and narrates player actions."""

    def __init__(
        self,
        game_config: GameConfig,
        tool_registry: ToolRegistry,
        openai_client: OpenAI,
    ) -> None:
        self.game_config = game_config
        self.rules = get_rules(game_config)
        self.system_prompt = game_config.gm_prompt
        self.tools = tool_registry.get_gm_tools()
        self.tool_registry = tool_registry
        self.client = openai_client
        self.model = get_settings().gm_model
        # Logging-only today. Future (M5): feed a condensed public_transcript from the
        # graph state into the system prompt for cross-turn narration continuity.
        self.conversation_history: list[dict] = []
        # Source of truth for state lives in the manager (shared by both state tools).
        self.state_manager = tool_registry.get_state_tool.manager
        self._rng = random.Random()
        self._seed: int | None = None

    # -- Initialization ------------------------------------------------------

    def _build_initial_state(
        self, num_players: int, seed: int | None = None
    ) -> tuple[dict, str | None]:
        """Build a fresh, fully dealt round via the game's rules module."""
        return self.rules.setup(self.game_config, num_players, seed)

    def initialize_game(self, num_players: int | None = None, seed: int | None = None) -> dict:
        """Set up and establish a new game. Returns the GM-view state plus narration."""
        if num_players is None:
            num_players = self.game_config.num_players
        supported = self.rules.supported_player_counts
        if supported and num_players not in supported:
            raise ValueError(
                f"{self.game_config.game_name} supports {supported} players; "
                f"{num_players} is out of scope."
            )
        self._seed = seed
        self._rng = random.Random(seed)

        # Append any game-specific addendum (e.g. token goal, engine-scored rounds). Rebuilt
        # from the config prompt each call so repeat initializations never stack addenda.
        self.system_prompt = self.game_config.gm_prompt + self.rules.system_prompt_addendum(
            num_players
        )

        state, removed_card = self._build_initial_state(num_players, seed)

        gm_view = self.state_manager.initialize(
            initial_state=state,
            deck_cards=state.get("deck", []),
            removed_card=removed_card or "",  # games without a removed card pass ""
            revealed_cards=state.get("revealed_cards", []),
            player_hands={pid: p["hand"] for pid, p in state["players"].items()},
        )

        narration = self._narrate(
            "The game is starting. Briefly and flavorfully narrate the opening of a new "
            f"game of {self.game_config.game_name} with {num_players} players. "
            "Do not reveal any hidden information (hands, the removed card, or deck order)."
        )
        self.conversation_history.append({"role": "gm", "content": narration})
        return {"state": gm_view, "narration": narration}

    # -- LLM plumbing --------------------------------------------------------

    def _narrate(self, instruction: str) -> str:
        """One-shot narration call (no tools)."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction},
            ],
        )
        return completion.choices[0].message.content or ""

    def _call_llm(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Run the GM tool-use loop until it calls ``finish_resolution``.

        Tool calls are executed via the registry (always as the GM). ``finish_resolution``
        is intercepted here, not routed to the registry: its arguments ARE the structured
        outcome, and calling it ends the loop — so there is no second summary LLM call.

        A ValueError from ``set_game_state`` (the manager rejecting a malformed write) is
        fed back as the tool result so the GM can redo the write within this same loop.
        That is the GM completing a valid write, not reconciling committed game state, so
        it is deliberately NOT a crash. Exhausting the iteration cap without finishing is.
        """
        set_state_called = False
        resolution: dict | None = None

        for _ in range(_MAX_TOOL_ITERATIONS):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tool_choice": "auto",
            }
            if tools:
                kwargs["tools"] = tools
            completion = self.client.chat.completions.create(**kwargs)
            message = completion.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            tool_calls = message.tool_calls or []
            if not tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Call finish_resolution to report the outcome of this resolution."
                        ),
                    }
                )
                continue

            for tool_call in tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    content = f"Could not parse tool arguments: {exc}"
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": content}
                    )
                    continue

                if name == "finish_resolution":
                    resolution = args
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": "ok"}
                    )
                    continue

                try:
                    result = self.tool_registry.execute_tool(name, args, "gm")
                    if name == "set_game_state":
                        set_state_called = True
                    content = result if isinstance(result, str) else json.dumps(result)
                except ValueError as exc:
                    if name == "set_game_state":
                        content = (
                            f"State update rejected: {exc}. Call get_game_state, copy the "
                            "COMPLETE state, and modify only the affected fields; do not "
                            "drop keys such as 'deck' or change any field's type."
                        )
                    else:
                        content = f"Tool error: {exc}"
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

            if resolution is not None:
                return {
                    "resolution": resolution,
                    "set_state_called": set_state_called,
                    "messages": messages,
                }

        raise PlaytestError(
            f"GM did not call finish_resolution within {_MAX_TOOL_ITERATIONS} iterations"
        )

    def validate_and_resolve(self, proposed_action: dict, player_id: str) -> GMResolution:
        """Validate a proposed player action and, if legal, resolve and commit it.

        Crashes (raises) rather than reconciling: an illegal action raises ``IllegalAction``,
        a committed state that violates integrity invariants raises ``StateInvariantViolation``,
        and an is_valid/commit inconsistency raises ``PlaytestError``. No retries.
        """
        action_type = proposed_action.get("action_type", "")
        parameters = proposed_action.get("parameters", {})
        public_statement = proposed_action.get("public_statement", "")

        user_message = (
            f"Player {player_id} proposes the following action:\n"
            f"Action: {action_type}\n"
            f"Parameters: {json.dumps(parameters)}\n"
            f'Public statement: "{public_statement}"\n\n'
            "Please:\n"
            "1. Call get_game_state to see the current board.\n"
            "2. Validate this action against the rules (query the rulebook if you need to "
            "verify any rule).\n"
            "3. If valid, resolve it: determine the outcome, build the complete updated game "
            "state, and call set_game_state.\n"
            "4. If invalid, do NOT call set_game_state.\n"
            "5. Finally, call finish_resolution to report the outcome (is_valid, narration, "
            "who goes next, whether the round ended, any private info).\n\n"
            f"{_STATE_WRITE_INSTRUCTION}"
        )

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        loop = self._call_llm(messages, tools=self.tools)
        args = loop["resolution"]
        set_state_called = loop["set_state_called"]
        is_valid = bool(args.get("is_valid", False))
        last_action = {
            "player_id": player_id,
            "action_type": action_type,
            "parameters": parameters,
        }

        if not is_valid:
            if set_state_called:
                raise PlaytestError(
                    f"GM committed state for an action it reported invalid "
                    f"({player_id} {action_type})."
                )
            raise IllegalAction(
                player_id, proposed_action, args.get("error_message") or "no reason given"
            )
        if not set_state_called:
            raise PlaytestError(
                f"GM reported {player_id}'s {action_type} valid but never committed a new state."
            )

        committed = self.state_manager.get_state("gm")
        violations = self.rules.check_invariants(committed, last_action)
        if violations:
            raise StateInvariantViolation(violations, last_action, committed)

        return self._build_resolution(args, committed, player_id)

    def handle_round_end(self) -> GMResolution:
        """Score the completed round deterministically; the LLM only narrates and deals.

        The engine (``resolve_round``) decides the winner(s) and token awards — pure
        arithmetic, not a rules judgment. Awards are committed first; if no one has reached
        the token goal, the LLM is asked to deal the next round preserving those tokens.
        """
        state = self.state_manager.get_state("gm")
        players = state["players"]
        tokens_to_win = state["tokens_to_win"]
        num_players = state["num_players"]

        result = self.rules.score_round(players)
        assert result is not None  # a round-based game's rules module scores rounds
        winners = result["winners"]
        winning_card = result["winning_card"]
        scores = result["scores"]
        for pid, total in scores.items():
            players[pid]["tokens"] = total
        committed = self.state_manager.set_state(state)

        winners_label = ",".join(winners)
        game_winner = self.rules.is_game_won(committed)

        if game_winner:
            # Game over: tokens are committed, no new deal. Tied threshold-crossers share.
            narration = self._narrate(
                f"The final round has ended. {winners_label} won the round holding the "
                f"{winning_card}, reaching the {tokens_to_win}-token goal to win the game. "
                "Narrate this climactic finish briefly. Do not reveal hidden information."
            )
            self.conversation_history.append({"role": "gm", "content": narration})
            return GMResolution(
                is_valid=True,
                action_summary=(
                    f"{winners_label} won the round with the {winning_card} and won the game."
                ),
                narration=narration,
                new_state=committed,
                round_ended=True,
                game_ended=True,
                winner=game_winner,
                winners=winners,
                winning_card=winning_card,
            )

        # Game continues: the LLM deals the next round from the already-shuffled deck,
        # preserving the engine-decided tokens. It must NOT recompute scoring.
        next_deck = self.rules.new_round_deck(self._rng)
        params = self.game_config.setup_parameters
        cards_removed = params["cards_removed"]
        cards_revealed = (
            params["cards_revealed_2p"] if num_players == 2 else params["cards_revealed_other"]
        )
        cards_dealt = params["cards_dealt_per_player"]
        token_lines = ", ".join(f"{pid}={scores[pid]}" for pid in sorted(scores))

        user_message = (
            "The round has ended and scoring is already decided by the engine — do NOT "
            "recompute it. Results:\n"
            f"- Round winner(s): {winners_label}, holding the {winning_card}.\n"
            f"- Final token totals (preserve these exactly): {token_lines}.\n\n"
            "Deal the next round and call set_game_state. Use this already-shuffled deck, "
            "IN ORDER — do NOT shuffle it again:\n"
            f"{json.dumps(next_deck)}\n"
            "All players return for the new round: reset every player's hand and discards to "
            "empty and set is_eliminated and is_protected to false for ALL players. Then set "
            f"'deck' to this exact list, remove {cards_removed} card(s), reveal {cards_revealed} "
            f"card(s) face-up, and deal {cards_dealt} card(s) to EVERY player, taking cards from "
            "the deck in order. Increment round_number, the round winner takes the first turn, "
            "set turn_phase to 'draw', and preserve every player's token count exactly as listed "
            "above.\n\n"
            f"{_STATE_WRITE_INSTRUCTION}"
        )

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        loop = self._call_llm(messages, tools=self.tools)
        if not loop["set_state_called"]:
            raise PlaytestError("GM did not deal the next round (no set_game_state call).")
        committed = self.state_manager.get_state("gm")
        violations = self.rules.check_invariants(committed, None)
        if violations:
            raise StateInvariantViolation(violations, None, committed)

        resolution = self._build_resolution(loop["resolution"] or {}, committed, player_id=None)
        resolution.round_ended = True
        resolution.game_ended = False
        resolution.winner = None
        resolution.winners = winners
        resolution.winning_card = winning_card
        resolution.next_player = winners[0]

        # The LLM dealt the next round; don't trust it to have revived everyone. The
        # resolver depends on these invariants next round, so enforce them now rather
        # than discovering a silently-excluded player when scoring.
        if resolution.new_state is not None:
            for pid, p in resolution.new_state["players"].items():
                if p.get("is_eliminated", False) or p.get("is_protected", False):
                    raise ValueError(
                        f"{pid} was not reset for the new round "
                        f"(is_eliminated={p.get('is_eliminated')}, "
                        f"is_protected={p.get('is_protected')})"
                    )
                if p.get("hand_count") != 1:
                    raise ValueError(
                        f"{pid} must be dealt exactly one card for the new round, "
                        f"got hand_count={p.get('hand_count')!r}"
                    )
        return resolution

    # -- Helpers -------------------------------------------------------------

    def _build_resolution(
        self,
        args: dict,
        new_state: dict,
        player_id: str | None,
    ) -> GMResolution:
        """Map a finish_resolution payload + committed state into a GMResolution.

        Only reached for a valid, committed resolution (validate_and_resolve has already
        crashed on any invalid/inconsistent case). ``next_player`` is advisory — the driver
        computes turn order from committed state — but we still sanitize it for the logs.
        """
        narration = args.get("narration")
        resolution = GMResolution(
            is_valid=True,
            error_message=None,
            action_summary=args.get("action_summary"),
            narration=narration,
            new_state=new_state,
            round_ended=bool(args.get("round_ended", False)),
            # Per-action resolutions never end the game; only handle_round_end does.
            game_ended=False,
            winner=args.get("winner"),
            next_player=args.get("next_player"),
            next_phase=args.get("next_phase") or "draw",
            private_info=args.get("private_info"),
            gm_reasoning=args.get("gm_reasoning"),
        )
        resolution.next_player = self._validate_next_player(resolution.next_player, player_id)
        if narration:
            self.conversation_history.append({"role": "gm", "content": narration})
        return resolution

    def _validate_next_player(self, candidate: str | None, acting_player: str | None) -> str | None:
        """Ensure next_player is a real, non-eliminated id; else fall back deterministically."""
        try:
            state = self.state_manager.get_state("gm")
        except ValueError:
            return candidate
        players = state.get("players", {})
        if candidate in players and not players[candidate].get("is_eliminated", False):
            return candidate

        order = sorted(players)
        if not order:
            return candidate
        start = order.index(acting_player) + 1 if acting_player in order else 0
        for offset in range(len(order)):
            pid = order[(start + offset) % len(order)]
            if not players[pid].get("is_eliminated", False):
                if candidate is not None:
                    _console.print(
                        f"[yellow]Warning:[/yellow] GM returned invalid next_player "
                        f"{candidate!r}; using {pid!r}."
                    )
                return pid
        return candidate
