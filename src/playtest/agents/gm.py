"""Game Master (GM) agent: validates, resolves, and narrates player actions.

The GM is the central authority for *judgment*. It is an LLM agent with three tools
(query_rulebook, get_game_state, set_game_state) and no player-action tools — it reasons
about actions in natural language grounded by the rulebook, then writes the full game
state. Deterministic mechanics (seeded setup/redeals, turn rotation, conservation
invariants) are the generic rules engine's, configured by the ingested GameSpec.
"""

import json
import random
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console

from playtest.config import get_settings
from playtest.errors import PlaytestError, StateInvariantViolation
from playtest.ingestion.schemas import GameConfig
from playtest.rules import GameRules
from playtest.tools import ToolRegistry

_console = Console()

_MAX_TOOL_ITERATIONS = 12

_STATE_WRITE_INSTRUCTION = (
    "When you commit a state change, build new_state by copying the ENTIRE committed state "
    "provided above and changing only the fields the action affects. Never drop, rename, or "
    "retype any key — keep every key and the exact types you received."
)


class FinishResolutionPayload(BaseModel):
    """The validated shape of the GM's ``finish_resolution`` tool arguments.

    ``extra="ignore"`` tolerates harmless additions; wrong types crash the run with a
    clear diagnostic instead of corrupting the resolution downstream.
    """

    model_config = ConfigDict(extra="ignore")

    is_valid: bool = False
    error_message: str | None = None
    action_summary: str | None = None
    narration: str | None = None
    turn_ended: bool | None = None
    round_ended: bool = False
    game_ended: bool = False
    winner: str | None = None
    winners: list[str] | None = None
    next_player: str | None = None
    next_phase: str | None = None
    private_info: dict | None = None
    gm_reasoning: str | None = None


class GMResolution(BaseModel):
    """The outcome of validating/resolving a proposed action."""

    is_valid: bool
    error_message: str | None = None
    action_summary: str | None = None
    narration: str | None = None
    new_state: dict | None = None
    turn_ended: bool | None = None
    round_ended: bool = False
    game_ended: bool = False
    winner: str | None = None
    winners: list[str] | None = None
    next_player: str | None = None
    next_phase: str | None = None
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
        self.rules = GameRules(game_config)
        self.spec = game_config.game_spec
        self.system_prompt = game_config.gm_prompt
        self.tools = tool_registry.get_gm_tools()
        self.tool_registry = tool_registry
        self.client = openai_client
        self.model = get_settings().gm_model
        # Logging-only today. Future: feed a condensed public_transcript into the
        # system prompt for cross-turn narration continuity.
        self.conversation_history: list[dict] = []
        # Source of truth for state lives in the manager (shared by both state tools).
        self.state_manager = tool_registry.get_state_tool.manager
        self._rng = random.Random()
        self._seed: int | None = None

    # -- Initialization ------------------------------------------------------

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

        # Append the engine's addendum (player count, phases, division of labor). Rebuilt
        # from the config prompt each call so repeat initializations never stack addenda.
        self.system_prompt = self.game_config.gm_prompt + self.rules.system_prompt_addendum(
            num_players
        )

        state = self.rules.setup(num_players, seed)
        gm_view = self.state_manager.initialize(state, self.spec.visibility)

        narration = self._narrate(
            "The game is starting. Briefly and flavorfully narrate the opening of a new "
            f"game of {self.game_config.game_name} with {num_players} players. "
            "Do not reveal any hidden information."
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
                            f"State update rejected: {exc}. Copy the COMPLETE committed "
                            "state and modify only the affected fields; do not drop keys "
                            "or change any field's type."
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

    @staticmethod
    def _validate_payload(raw: dict | None) -> FinishResolutionPayload:
        try:
            return FinishResolutionPayload.model_validate(raw or {})
        except ValidationError as exc:
            raise PlaytestError(
                f"GM returned a malformed finish_resolution payload: {exc}"
            ) from exc

    def validate_and_resolve(self, proposed_action: dict, player_id: str) -> GMResolution:
        """Validate a proposed player action and, if legal, resolve and commit it.

        A clean rejection (invalid, nothing committed) returns a resolution with
        ``is_valid=False`` so the driver can feed it back to the player (bounded retry —
        rejections are a playtest signal, not a crash). Integrity problems still crash:
        an is_valid/commit inconsistency raises ``PlaytestError`` and a committed state
        that violates invariants raises ``StateInvariantViolation``.
        """
        action_type = proposed_action.get("action_type", "")
        parameters = proposed_action.get("parameters", {})
        public_statement = proposed_action.get("public_statement", "")
        committed_before = self.state_manager.get_state("gm")

        user_message = (
            f"Player {player_id} proposes the following action:\n"
            f"Action: {action_type}\n"
            f"Parameters: {json.dumps(parameters)}\n"
            f'Public statement: "{public_statement}"\n\n'
            f"Current committed game state (GM view):\n{json.dumps(committed_before)}\n\n"
            "Please:\n"
            "1. Validate this action against the rules using the state above (query the "
            "rulebook if you need to verify any rule).\n"
            "2. If valid, resolve it: determine the outcome, build the complete updated game "
            "state, and call set_game_state.\n"
            "3. If invalid, do NOT call set_game_state.\n"
            "4. Finally, call finish_resolution to report the outcome (is_valid, narration, "
            "whether the turn/round/game ended, who goes next, any private info).\n\n"
            f"{_STATE_WRITE_INSTRUCTION}"
        )

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        loop = self._call_llm(messages, tools=self.tools)
        payload = self._validate_payload(loop["resolution"])
        set_state_called = loop["set_state_called"]
        last_action = {
            "player_id": player_id,
            "action_type": action_type,
            "parameters": parameters,
        }

        if not payload.is_valid:
            if set_state_called:
                raise PlaytestError(
                    f"GM committed state for an action it reported invalid "
                    f"({player_id} {action_type})."
                )
            return GMResolution(
                is_valid=False,
                error_message=payload.error_message or "no reason given",
                new_state=None,
            )
        if not set_state_called:
            raise PlaytestError(
                f"GM reported {player_id}'s {action_type} valid but never committed a new state."
            )

        committed = self.state_manager.get_state("gm")
        violations = self.rules.check_invariants(committed, last_action)
        if violations:
            raise StateInvariantViolation(violations, last_action, committed)

        return self._build_resolution(payload, committed, player_id)

    def handle_round_end(self) -> GMResolution:
        """Score the completed round (GM judgment), then redeal deterministically (engine).

        The GM applies the ingested Scoring rules to the committed state and commits the
        score changes; the engine validates conservation, then — if the game continues —
        deals the next round itself with the seeded RNG (an LLM cannot shuffle), preserving
        the spec's carry-over fields.
        """
        state = self.state_manager.get_state("gm")

        user_message = (
            "The current round has ended. Score it now:\n"
            "1. Apply the Scoring rules from your system prompt to the committed state "
            "below to determine the round result.\n"
            "2. Commit the score changes with set_game_state (update scores/standings ONLY "
            "— do NOT deal a new round; the engine deals).\n"
            "3. Call finish_resolution: narrate the result briefly, report every round "
            "winner in `winners`, whether the GAME is now over per the End Conditions "
            "(`game_ended`, and `winner` if so), and who should take the first turn of the "
            "next round (`next_player`) if it continues.\n\n"
            f"Current committed game state (GM view):\n{json.dumps(state)}\n\n"
            f"{_STATE_WRITE_INSTRUCTION}"
        )

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        loop = self._call_llm(messages, tools=self.tools)
        payload = self._validate_payload(loop["resolution"])
        if not loop["set_state_called"]:
            raise PlaytestError("GM did not commit the round scoring (no set_game_state call).")

        scored = self.state_manager.get_state("gm")
        violations = self.rules.check_invariants(scored, None)
        if violations:
            raise StateInvariantViolation(violations, None, scored)

        winners = payload.winners or ([payload.winner] if payload.winner else [])
        if payload.narration:
            self.conversation_history.append({"role": "gm", "content": payload.narration})

        if payload.game_ended:
            return GMResolution(
                is_valid=True,
                action_summary=payload.action_summary,
                narration=payload.narration,
                new_state=scored,
                round_ended=True,
                game_ended=True,
                winner=payload.winner or ",".join(winners),
                winners=winners or None,
                gm_reasoning=payload.gm_reasoning,
            )

        # Game continues: the engine deals the next round (carry-overs preserved).
        next_state = self.rules.redeal_round(scored, self._rng)
        next_player = self._validate_next_player(payload.next_player, None, state=next_state)
        next_state["current_turn"] = next_player or "player_1"
        committed = self.state_manager.set_state(next_state, remask=True)
        violations = self.rules.check_invariants(committed, None)
        if violations:
            raise StateInvariantViolation(violations, None, committed)

        return GMResolution(
            is_valid=True,
            action_summary=payload.action_summary,
            narration=payload.narration,
            new_state=committed,
            round_ended=True,
            game_ended=False,
            winners=winners or None,
            next_player=committed["current_turn"],
            gm_reasoning=payload.gm_reasoning,
        )

    # -- Helpers -------------------------------------------------------------

    def _build_resolution(
        self,
        payload: FinishResolutionPayload,
        new_state: dict,
        player_id: str | None,
    ) -> GMResolution:
        """Map a validated finish_resolution payload + committed state into a GMResolution.

        Only reached for a valid, committed resolution. ``next_player`` is advisory — the
        driver computes turn order from committed state — but we still sanitize it for
        the logs.
        """
        resolution = GMResolution(
            is_valid=True,
            error_message=None,
            action_summary=payload.action_summary,
            narration=payload.narration,
            new_state=new_state,
            turn_ended=payload.turn_ended,
            round_ended=payload.round_ended,
            game_ended=payload.game_ended,
            winner=payload.winner,
            winners=payload.winners,
            next_player=payload.next_player,
            next_phase=payload.next_phase or self.spec.turn.initial_phase,
            private_info=payload.private_info,
            gm_reasoning=payload.gm_reasoning,
        )
        resolution.next_player = self._validate_next_player(resolution.next_player, player_id)
        if payload.narration:
            self.conversation_history.append({"role": "gm", "content": payload.narration})
        return resolution

    def _validate_next_player(
        self, candidate: str | None, acting_player: str | None, state: dict | None = None
    ) -> str | None:
        """Ensure next_player is a real, active id; else fall back deterministically."""
        if state is None:
            try:
                state = self.state_manager.get_state("gm")
            except ValueError:
                return candidate
        players = state.get("players", {})
        if candidate is not None and candidate in players and self.rules.is_player_active(
            state, candidate
        ):
            return candidate

        order = sorted(players)
        if not order:
            return candidate
        start = order.index(acting_player) + 1 if acting_player in order else 0
        for offset in range(len(order)):
            pid = order[(start + offset) % len(order)]
            if self.rules.is_player_active(state, pid):
                if candidate is not None:
                    _console.print(
                        f"[yellow]Warning:[/yellow] GM returned invalid next_player "
                        f"{candidate!r}; using {pid!r}."
                    )
                return pid
        return candidate
