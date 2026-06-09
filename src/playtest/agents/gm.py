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
from playtest.ingestion.schemas import GameConfig
from playtest.tools import ToolRegistry

_console = Console()

# Love Letter classic 16-card deck.
DECK_COMPOSITION = (
    ["Guard"] * 5
    + ["Priest"] * 2
    + ["Baron"] * 2
    + ["Handmaid"] * 2
    + ["Prince"] * 2
    + ["King"] * 1
    + ["Countess"] * 1
    + ["Princess"] * 1
)
TOKENS_TO_WIN = {2: 7, 3: 5, 4: 4, 5: 3, 6: 3}

# Classic variant supports 2-4 players; 5-6 need the 21-card deck (Spy/Chancellor).
SUPPORTED_PLAYER_COUNTS = (2, 3, 4)

CARD_RANK = {
    "Guard": 1,
    "Priest": 2,
    "Baron": 3,
    "Handmaid": 4,
    "Prince": 5,
    "King": 6,
    "Countess": 7,
    "Princess": 8,
}


def resolve_round(players: dict) -> dict:
    """Deterministically score a completed round.

    Surviving (non-eliminated) players each hold exactly one card. The highest
    ``CARD_RANK`` wins; ties break on the highest sum of discard-pile ranks; any
    remaining tie is shared (every still-tied player wins). Each winner gains one token.

    Returns ``{"winners": [pid, ...], "winning_card": str, "scores": {pid: new_total}}``.
    """
    survivors = {pid: p for pid, p in players.items() if not p.get("is_eliminated", False)}
    if not survivors:
        raise ValueError("cannot resolve a round with no surviving players")
    for pid, p in survivors.items():
        hand = p.get("hand", [])
        if len(hand) != 1:
            raise ValueError(
                f"survivor {pid} must hold exactly one card at round end, got {hand!r}"
            )

    def rank(pid: str) -> int:
        return CARD_RANK[survivors[pid]["hand"][0]]

    def discard_sum(pid: str) -> int:
        return sum(CARD_RANK[c] for c in survivors[pid].get("discards", []))

    best_rank = max(rank(pid) for pid in survivors)
    top = [pid for pid in survivors if rank(pid) == best_rank]
    if len(top) > 1:
        best_discard = max(discard_sum(pid) for pid in top)
        top = [pid for pid in top if discard_sum(pid) == best_discard]

    winners = sorted(top)
    winning_card = survivors[winners[0]]["hand"][0]
    scores = {pid: p.get("tokens", 0) for pid, p in players.items()}
    for pid in winners:
        scores[pid] += 1
    return {"winners": winners, "winning_card": winning_card, "scores": scores}


_MAX_TOOL_ITERATIONS = 8

_STATE_WRITE_INSTRUCTION = (
    "When you commit a state change, FIRST call get_game_state to read the complete "
    "current state, then build new_state by copying that ENTIRE object and changing "
    "only the fields the action affects. Never drop, rename, or retype any key — keep "
    "every key (including the 'deck' array) and the exact types you received."
)

_RESOLUTION_SCHEMA_INSTRUCTION = (
    "Summarize your resolution as a single JSON object with EXACTLY these keys:\n"
    '  "is_valid" (bool): was the proposed action legal per the rules?\n'
    '  "error_message" (string or null): if invalid, the player-facing reason.\n'
    '  "action_summary" (string): a one-line factual summary of what happened.\n'
    '  "narration" (string): a brief, flavorful description for the players.\n'
    '  "round_ended" (bool): did the deck empty or only one player remain?\n'
    '  "game_ended" (bool): did a player reach the token threshold?\n'
    '  "winner" (string or null): player id of the game winner, if game_ended.\n'
    '  "next_player" (string or null): the player id whose turn is next.\n'
    '  "next_phase" (string): "draw" or "play".\n'
    '  "private_info" (object or null): info visible only to the acting player '
    "(e.g. a Priest reveal), or null.\n"
    '  "gm_reasoning" (string): your internal reasoning, for logging.'
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

    def _build_initial_state(self, num_players: int, seed: int | None = None) -> tuple[dict, str]:
        """Build a fresh, fully dealt round programmatically (seeded for repeatability)."""
        rng = random.Random(seed)
        deck = list(DECK_COMPOSITION)
        rng.shuffle(deck)

        removed_card = deck.pop()

        revealed_cards: list[str] = []
        if num_players == 2:
            for _ in range(3):
                revealed_cards.append(deck.pop())

        player_hands: dict[str, list[str]] = {}
        for i in range(1, num_players + 1):
            player_hands[f"player_{i}"] = [deck.pop()]

        state: dict = {
            "game_name": self.game_config.game_name,
            "variant": self.game_config.variant,
            "num_players": num_players,
            "tokens_to_win": TOKENS_TO_WIN[num_players],
            "round_number": 1,
            "current_turn": "player_1",
            "turn_phase": "draw",
            "deck": deck,
            "deck_count": len(deck),
            "removed_card": removed_card,
            "revealed_cards": revealed_cards,
            "players": {},
        }
        for i in range(1, num_players + 1):
            pid = f"player_{i}"
            state["players"][pid] = {
                "hand": player_hands[pid],
                "hand_count": len(player_hands[pid]),
                "discards": [],
                "tokens": 0,
                "is_eliminated": False,
                "is_protected": False,
            }

        return state, removed_card

    def initialize_game(self, num_players: int | None = None, seed: int | None = None) -> dict:
        """Set up and establish a new game. Returns the GM-view state plus narration."""
        if num_players is None:
            num_players = self.game_config.num_players
        if num_players not in SUPPORTED_PLAYER_COUNTS:
            raise ValueError(
                f"Classic Love Letter supports {SUPPORTED_PLAYER_COUNTS} players; "
                f"{num_players} requires the 21-card deck (Spy/Chancellor), which is out of scope."
            )
        self._seed = seed
        self._rng = random.Random(seed)

        # Append a count-aware addendum so the GM knows the token goal and that round
        # scoring is engine-computed (it only narrates/deals). Rebuilt from the config
        # prompt each call so repeat initializations never stack addenda.
        self.system_prompt = self.game_config.gm_prompt + (
            f"\n\n## This Game\nThis game has {num_players} players; the token threshold to "
            f"win is {TOKENS_TO_WIN[num_players]}. Round scoring (who won the round and token "
            "awards) is computed by the game engine and given to you — narrate the result and "
            "deal the next round, but never recompute the winner yourself."
        )

        state, removed_card = self._build_initial_state(num_players, seed)

        gm_view = self.state_manager.initialize(
            initial_state=state,
            deck_cards=state["deck"],
            removed_card=removed_card,
            revealed_cards=state["revealed_cards"],
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
        """Run the tool-use loop until the LLM emits a final text response.

        Tool calls are executed via the registry (always as the GM). A ValueError from
        a tool (notably the manager rejecting a state write) is fed back as the tool
        result so the LLM can self-correct and retry within the loop.
        """
        set_state_called = False
        set_state_error: str | None = None

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
                return {
                    "content": message.content or "",
                    "set_state_called": set_state_called,
                    "set_state_error": set_state_error,
                    "messages": messages,
                }

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

                try:
                    result = self.tool_registry.execute_tool(name, args, "gm")
                    if name == "set_game_state":
                        set_state_called = True
                        set_state_error = None
                    content = result if isinstance(result, str) else json.dumps(result)
                except ValueError as exc:
                    if name == "set_game_state":
                        set_state_error = str(exc)
                        content = (
                            f"State update rejected: {exc}. Call get_game_state, copy the "
                            "COMPLETE state, and modify only the affected fields; do not "
                            "drop keys such as 'deck' or change any field's type."
                        )
                    else:
                        content = f"Tool error: {exc}"
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

        return {
            "content": "",
            "set_state_called": set_state_called,
            "set_state_error": set_state_error,
            "messages": messages,
        }

    def _structured_summary(self, messages: list[dict]) -> dict:
        """Follow-up JSON call summarizing the resolution into GMResolution fields."""
        summary_messages = messages + [{"role": "user", "content": _RESOLUTION_SCHEMA_INSTRUCTION}]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": summary_messages,
            "response_format": {"type": "json_object"},
        }
        completion = self.client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    # -- Resolution ----------------------------------------------------------

    def validate_and_resolve(self, proposed_action: dict, player_id: str) -> GMResolution:
        """Validate a proposed player action and, if legal, resolve and commit it."""
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
            "4. If invalid, explain why and do NOT call set_game_state.\n"
            "After resolving, state who goes next and whether the round/game has ended.\n\n"
            f"{_STATE_WRITE_INSTRUCTION}"
        )

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        loop = self._call_llm(messages, tools=self.tools)
        summary = self._structured_summary(loop["messages"])

        return self._build_resolution(
            summary,
            player_id=player_id,
            set_state_called=loop["set_state_called"],
            set_state_error=loop["set_state_error"],
        )

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

        result = resolve_round(players)
        winners = result["winners"]
        winning_card = result["winning_card"]
        scores = result["scores"]
        for pid, total in scores.items():
            players[pid]["tokens"] = total
        committed = self.state_manager.set_state(state)

        winners_label = ",".join(winners)
        crossed = [pid for pid in winners if scores[pid] >= tokens_to_win]

        if crossed:
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
                winner=",".join(crossed),
                winners=winners,
                winning_card=winning_card,
            )

        # Game continues: the LLM deals the next round from the already-shuffled deck,
        # preserving the engine-decided tokens. It must NOT recompute scoring.
        next_deck = list(DECK_COMPOSITION)
        self._rng.shuffle(next_deck)
        reveal_clause = "reveal 3 cards face-up" if num_players == 2 else "reveal no cards"
        token_lines = ", ".join(f"{pid}={scores[pid]}" for pid in sorted(scores))

        user_message = (
            "The round has ended and scoring is already decided by the engine — do NOT "
            "recompute it. Results:\n"
            f"- Round winner(s): {winners_label}, holding the {winning_card}.\n"
            f"- Final token totals (preserve these exactly): {token_lines}.\n\n"
            "Deal the next round and call set_game_state. Use this already-shuffled deck, "
            "IN ORDER — do NOT shuffle it again:\n"
            f"{json.dumps(next_deck)}\n"
            f"Set 'deck' to this exact list, then remove one card, {reveal_clause}, and deal "
            "one card to each non-eliminated player, taking cards from the deck in order. "
            "Increment round_number, the round winner takes the first turn, set turn_phase to "
            "'draw', reset hands/discards/is_eliminated/is_protected, and preserve every "
            "player's token count exactly as listed above.\n\n"
            f"{_STATE_WRITE_INSTRUCTION}"
        )

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        loop = self._call_llm(messages, tools=self.tools)
        summary = self._structured_summary(loop["messages"])

        resolution = self._build_resolution(
            summary,
            player_id=None,
            set_state_called=loop["set_state_called"],
            set_state_error=loop["set_state_error"],
        )
        resolution.round_ended = True
        resolution.game_ended = False
        resolution.winner = None
        resolution.winners = winners
        resolution.winning_card = winning_card
        resolution.next_player = winners[0]
        return resolution

    # -- Helpers -------------------------------------------------------------

    def _build_resolution(
        self,
        summary: dict,
        player_id: str | None,
        set_state_called: bool,
        set_state_error: str | None,
    ) -> GMResolution:
        """Assemble a GMResolution, distinguishing rule rejection from GM write failure."""
        if set_state_called:
            is_valid = True
            new_state: dict | None = self.state_manager.get_state("gm")
        elif set_state_error is not None:
            # The GM judged the action legal but its state write was rejected — a GM bug,
            # not an illegal action. Report valid, surface the failure, keep prior state.
            _console.print(f"[yellow]Warning:[/yellow] GM state write failed: {set_state_error}")
            is_valid = True
            new_state = None
        else:
            is_valid = bool(summary.get("is_valid", False))
            new_state = None

        narration = summary.get("narration")
        resolution = GMResolution(
            is_valid=is_valid,
            error_message=summary.get("error_message") if not is_valid else None,
            action_summary=summary.get("action_summary"),
            narration=narration,
            new_state=new_state,
            round_ended=bool(summary.get("round_ended", False)),
            game_ended=bool(summary.get("game_ended", False)),
            winner=summary.get("winner"),
            next_player=summary.get("next_player"),
            next_phase=summary.get("next_phase") or "draw",
            private_info=summary.get("private_info"),
            gm_reasoning=summary.get("gm_reasoning")
            if set_state_error is None
            else f"{summary.get('gm_reasoning', '')} [state write failed: {set_state_error}]",
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
