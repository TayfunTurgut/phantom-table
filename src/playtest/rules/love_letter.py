"""Love Letter (classic) as a concrete GameRules module.

This is the reference game: it keeps the deterministic setup, integrity invariants, turn
order, available-action filtering, round-over detection, and scoring that used to live in
the GM agent. Because it is deterministic it also provides the crash-early safety net
(``check_invariants``) — something a purely LLM-driven generic game does not get.
"""

import random
from collections import Counter

from playtest.rules.base import GameRules

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
# Expected total card counts by name — the basis for state integrity checks.
_DECK_COMPOSITION_COUNTS = Counter(DECK_COMPOSITION)
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


def _card_rank(card: str) -> int:
    """Rank of a card, raising ValueError (not KeyError) on an unknown name."""
    try:
        return CARD_RANK[card]
    except KeyError:
        raise ValueError(f"unknown card {card!r}") from None


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
        return _card_rank(survivors[pid]["hand"][0])

    def discard_sum(pid: str) -> int:
        return sum(_card_rank(c) for c in survivors[pid].get("discards", []))

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


def check_state_invariants(
    state: dict,
    deck_composition: dict[str, int],
    last_action: dict | None = None,
) -> list[str]:
    """Check game-level integrity invariants on a committed GM-view state.

    These are structural integrity checks on the state object (card conservation,
    count consistency, hand bounds), NOT game rules. Returns a list of specific,
    actionable violation descriptions the GM can read and fix; empty means clean.

    ``last_action`` (shape ``{"player_id", "action_type", "parameters"}``) enables
    action-specific checks such as "the played card was moved to discards".
    """
    violations: list[str] = []

    players = state.get("players", {})
    deck = state.get("deck", [])
    removed_card = state.get("removed_card", "")
    revealed_cards = state.get("revealed_cards", [])

    # Check 1: card conservation — no card created or destroyed.
    expected_total = sum(deck_composition.values())
    all_cards: list[str] = list(deck) + list(revealed_cards)
    if removed_card and removed_card != "HIDDEN":
        all_cards.append(removed_card)
    for p in players.values():
        all_cards.extend(p.get("hand", []))
        all_cards.extend(p.get("discards", []))
    if len(all_cards) != expected_total:
        violations.append(
            f"Card conservation violated: expected {expected_total} total cards across all "
            f"locations (deck, hands, discards, removed, revealed), found {len(all_cards)}. "
            "Check that no card was dropped or duplicated."
        )

    # Check 2: hand_count consistency.
    for pid, p in players.items():
        hand = p.get("hand", [])
        hand_count = p.get("hand_count", 0)
        if hand_count != len(hand):
            violations.append(
                f"{pid} hand_count is {hand_count} but hand has {len(hand)} card(s). "
                "Set hand_count = len(hand)."
            )

    # Check 3: deck_count consistency.
    deck_count = state.get("deck_count", 0)
    if deck_count != len(deck):
        violations.append(
            f"deck_count is {deck_count} but deck has {len(deck)} card(s). "
            "Set deck_count = len(deck)."
        )

    # Check 4: hand size bounds for active players.
    turn_phase = state.get("turn_phase", "")
    for pid, p in players.items():
        if p.get("is_eliminated"):
            continue
        hand_len = len(p.get("hand", []))
        if hand_len > 2:
            violations.append(
                f"{pid} holds {hand_len} cards (max is 2). A card was not removed from "
                "hand after playing."
            )
        if turn_phase == "draw" and hand_len == 0:
            violations.append(
                f"{pid} has an empty hand but is not eliminated. They should hold exactly "
                "1 card."
            )

    # Check 5: eliminated players have empty hands.
    for pid, p in players.items():
        if p.get("is_eliminated") and len(p.get("hand", [])) > 0:
            violations.append(
                f"{pid} is eliminated but still has cards in hand: {p.get('hand')}. "
                "Eliminated players must have an empty hand (hand=[], hand_count=0)."
            )

    # Check 6: no card appears more often than the deck allows.
    card_counts = Counter(all_cards)
    for card, count in card_counts.items():
        max_allowed = deck_composition.get(card, 0)
        if count > max_allowed:
            violations.append(
                f"Card '{card}' appears {count} times but the deck only contains "
                f"{max_allowed}. A card was duplicated."
            )

    # Check 7: the played card was moved to the actor's discards (action-specific).
    if last_action and last_action.get("action_type", "").startswith("play_"):
        card_name = last_action["action_type"].replace("play_", "").capitalize()
        actor = last_action.get("player_id", "")
        discards = players.get(actor, {}).get("discards", [])
        last_discard = discards[-1] if discards else None
        if last_discard is None or last_discard.lower() != card_name.lower():
            violations.append(
                f"{actor} played {card_name} but the last card in their discards is "
                f"{last_discard!r}, not '{card_name}'. The played card must be moved from "
                "hand to the end of the discards list."
            )

    # Check 8: after a play (turn-ending), the actor ends with exactly 1 card or is
    # eliminated. A play takes one of the actor's two cards; ending with 2 means a card
    # was never discarded (e.g. a Prince that drew without first discarding the held card).
    if last_action and last_action.get("action_type", "").startswith("play_"):
        actor = last_action.get("player_id", "")
        p = players.get(actor, {})
        if not p.get("is_eliminated"):
            hand_count = p.get("hand_count", len(p.get("hand", [])))
            if hand_count != 1:
                violations.append(
                    f"{actor} ended their turn holding {hand_count} card(s) after playing. "
                    "After a play the actor must hold exactly 1 card (or be eliminated)."
                )

    return violations


class LoveLetterRules(GameRules):
    name = "love_letter"

    @property
    def supported_player_counts(self) -> tuple[int, ...]:
        return SUPPORTED_PLAYER_COUNTS

    def setup(self, game_config: object, num_players: int, seed: int | None) -> tuple[dict, str]:
        """Build a fresh, fully dealt round programmatically (seeded for repeatability)."""
        rng = random.Random(seed)
        deck = list(DECK_COMPOSITION)
        rng.shuffle(deck)

        removed_card = deck.pop()
        revealed_cards: list[str] = []
        if num_players == 2:
            for _ in range(3):
                revealed_cards.append(deck.pop())
        player_hands = {f"player_{i}": [deck.pop()] for i in range(1, num_players + 1)}

        state: dict = {
            "game_name": getattr(game_config, "game_name", "Love Letter"),
            "variant": getattr(game_config, "variant", "classic"),
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
        for pid, hand in player_hands.items():
            state["players"][pid] = {
                "hand": hand,
                "hand_count": len(hand),
                "discards": [],
                "tokens": 0,
                "is_eliminated": False,
                "is_protected": False,
            }
        return state, removed_card

    def system_prompt_addendum(self, num_players: int) -> str:
        return (
            f"\n\n## This Game\nThis game has {num_players} players; the token threshold to "
            f"win is {TOKENS_TO_WIN[num_players]}. Round scoring (who won the round and token "
            "awards) is computed by the game engine and given to you — narrate the result and "
            "deal the next round, but never recompute the winner yourself."
        )

    def check_invariants(self, state: dict, last_action: dict | None) -> list[str]:
        return check_state_invariants(state, _DECK_COMPOSITION_COUNTS, last_action)

    def available_actions(self, state: dict, player_id: str) -> list[str]:
        player = state["players"][player_id]
        turn_phase = state.get("turn_phase", "")
        hand = player.get("hand", [])
        if turn_phase == "draw":
            return ["draw_card"]
        if turn_phase == "play":
            hand_lower = [c.lower() for c in hand]
            if "countess" in hand_lower and ("king" in hand_lower or "prince" in hand_lower):
                return ["play_countess"]
            return [f"play_{card.lower()}" for card in set(hand)]
        return []

    def is_turn_over(self, last_action: dict | None) -> bool:
        return bool(last_action and last_action.get("action_type", "").startswith("play_"))

    def advance_turn(self, state: dict, current_player: str) -> dict:
        state["current_turn"] = self._next_active_player(state, current_player)
        state["turn_phase"] = "draw"
        return state

    def is_round_over(self, state: dict) -> bool:
        survivors = [p for p in state["players"].values() if not p.get("is_eliminated", False)]
        return state.get("deck_count", 0) == 0 or len(survivors) <= 1

    def score_round(self, players: dict) -> dict:
        return resolve_round(players)

    def is_game_won(self, state: dict) -> str | None:
        goal = state.get("tokens_to_win", 0)
        crossed = sorted(
            pid for pid, p in state["players"].items() if p.get("tokens", 0) >= goal
        )
        return ",".join(crossed) if crossed else None

    def new_round_deck(self, rng: random.Random) -> list[str]:
        deck = list(DECK_COMPOSITION)
        rng.shuffle(deck)
        return deck

    @staticmethod
    def _next_active_player(state: dict, current_player: str) -> str:
        players = state.get("players", {})
        order = sorted(players)
        if not order:
            return current_player
        start = order.index(current_player) + 1 if current_player in order else 0
        for offset in range(len(order)):
            pid = order[(start + offset) % len(order)]
            if not players[pid].get("is_eliminated", False):
                return pid
        return current_player
