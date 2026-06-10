"""Generic, config-driven game rules: deterministic primitives, zero game knowledge.

Everything here is configured by the ingestion-extracted :class:`GameSpec` — shuffling and
dealing (the seeded setup executor), phase-based action exposure, turn rotation, and
component-conservation invariants. All *judgment* (legality, action effects, end
conditions, scoring) belongs to the GM agent, grounded by the rulebook.
"""

import random
from collections import Counter
from copy import deepcopy

from playtest.ingestion.schemas import GameConfig, GameSpec
from playtest.state.manager import HIDDEN


class GameRules:
    """The engine-side rules helper for any ingested game."""

    def __init__(self, game_config: GameConfig) -> None:
        self.game_config = game_config
        self.spec: GameSpec = game_config.game_spec

    # -- Setup ----------------------------------------------------------------

    @property
    def supported_player_counts(self) -> tuple[int, ...]:
        return tuple(sorted(self.spec.supported_player_counts))

    def setup(self, num_players: int, seed: int | None) -> dict:
        """Build a fresh, fully dealt state (seeded for repeatability).

        Returns the complete REAL state — hands, pool, and masked values in place; the
        state manager applies visibility when views are read.
        """
        return self._deal(num_players, random.Random(seed))

    def redeal_round(self, state: dict, rng: random.Random) -> dict:
        """Deal the next round, preserving the spec's carry-over fields.

        The engine shuffles and deals (an LLM cannot shuffle); scoring has already been
        committed by the GM, so carry-overs (e.g. ``players.*.tokens``) come from ``state``.
        """
        num_players = len(state["players"])
        plan = self.spec.setup_plan_for(num_players)
        new_state = self._deal(num_players, rng)

        for path in plan.carry_over_fields:
            if path.startswith("players.*."):
                field = path[len("players.*.") :]
                for pid, player in new_state["players"].items():
                    old = state["players"].get(pid, {})
                    if field in old:
                        player[field] = deepcopy(old[field])
            elif path in state:
                new_state[path] = deepcopy(state[path])

        if "round_number" in new_state:
            new_state["round_number"] = state.get("round_number", 0) + 1
        return new_state

    def _deal(self, num_players: int, rng: random.Random) -> dict:
        plan = self.spec.setup_plan_for(num_players)
        state = self._template_for(num_players)

        # Component zones start empty: template values are placeholders, and leftover
        # placeholder components would break conservation.
        for zone in self.spec.component_zones:
            if zone.startswith("players.*."):
                field = zone[len("players.*.") :]
                for player in state["players"].values():
                    player[field] = self._cleared(player.get(field))
            elif zone in state:
                state[zone] = self._cleared(state[zone])

        pool: list[str] = []
        if plan.pool is not None:
            for name, count in plan.pool.items():
                pool.extend([name] * count)
            rng.shuffle(pool)

        for step in plan.deal_steps:
            if step.target == "each_player":
                for pid in sorted(state["players"]):
                    state["players"][pid][step.to_field] = self._take(
                        pool, step.count, state["players"][pid].get(step.to_field)
                    )
            else:  # set_aside / reveal both move pool components to a top-level field
                state[step.to_field] = self._take(pool, step.count, state.get(step.to_field))

        if plan.pool_field is not None:
            state[plan.pool_field] = pool

        state["current_turn"] = "player_1"
        state["turn_phase"] = self.spec.turn.initial_phase
        if "num_players" in state:
            state["num_players"] = num_players
        if "round_number" in state:
            state["round_number"] = 1
        self._recount(state)
        return state

    def _template_for(self, num_players: int) -> dict:
        """The initial-state template with a players object cloned out to ``num_players``."""
        state = deepcopy(self.game_config.initial_state_template)
        players = state.get("players") or {}
        if not players:
            raise ValueError("initial state template has no players")
        shape = deepcopy(next(iter(players.values())))
        state["players"] = {
            f"player_{i}": deepcopy(shape) for i in range(1, num_players + 1)
        }
        return state

    @staticmethod
    def _cleared(value: object) -> object:
        if isinstance(value, list):
            return []
        if isinstance(value, str):
            return ""
        return value

    @staticmethod
    def _take(pool: list[str], count: int, template_value: object) -> object:
        """Pop ``count`` components; a list field gets a list, a scalar field one value."""
        if count > len(pool):
            raise ValueError(
                f"setup plan deals {count} component(s) but only {len(pool)} remain in the pool"
            )
        taken = [pool.pop() for _ in range(count)]
        if isinstance(template_value, list) or count != 1:
            return taken
        return taken[0]

    def _recount(self, state: dict) -> None:
        """Recompute every spec count field from its list field (top-level or per-player)."""
        for list_field, count_field in self.spec.visibility.count_fields.items():
            if list_field in state:
                state[count_field] = len(state.get(list_field) or [])
            else:
                for player in state["players"].values():
                    if list_field in player:
                        player[count_field] = len(player.get(list_field) or [])

    # -- Turn flow --------------------------------------------------------------

    def available_actions(self, state: dict, player_id: str) -> list[str]:
        """Action tools available in the current phase (legality is the GM's call)."""
        phase = state.get("turn_phase", "")
        return sorted(
            name for name, rule in self.spec.action_rules.items() if rule.phase == phase
        )

    def is_turn_over(self, last_action: dict | None, gm_turn_ended: bool | None = None) -> bool:
        """GM's explicit report wins; otherwise the spec's per-action flag."""
        if gm_turn_ended is not None:
            return gm_turn_ended
        if not last_action:
            return False
        rule = self.spec.action_rules.get(last_action.get("action_type", ""))
        return rule.ends_turn if rule else True

    def advance_turn(self, state: dict, current_player: str) -> dict:
        """Rotate to the next active player and reset the phase."""
        state["current_turn"] = self._next_active_player(state, current_player)
        state["turn_phase"] = self.spec.turn.initial_phase
        return state

    def is_player_active(self, state: dict, player_id: str) -> bool:
        """False when the spec's inactive flag (e.g. eliminated) is set for the player."""
        flag = self.spec.turn.inactive_field
        if flag is None:
            return True
        return not state.get("players", {}).get(player_id, {}).get(flag, False)

    def _next_active_player(self, state: dict, current_player: str) -> str:
        players = state.get("players", {})
        order = sorted(players)
        if not order:
            return current_player
        start = order.index(current_player) + 1 if current_player in order else 0
        for offset in range(len(order)):
            pid = order[(start + offset) % len(order)]
            if self.is_player_active(state, pid):
                return pid
        return current_player

    # -- Integrity invariants ----------------------------------------------------

    def check_invariants(self, state: dict, last_action: dict | None = None) -> list[str]:
        """Generic crash-early nets: component conservation + count-field consistency.

        These are structural integrity checks, not game rules: no component may be
        created or destroyed across the spec's zones, and every public count field must
        match its list. Returns actionable violation strings; empty means clean.
        """
        violations: list[str] = []

        if self.spec.components and self.spec.component_zones:
            found: Counter[str] = Counter()
            for zone in self.spec.component_zones:
                for value in self._zone_values(state, zone):
                    found[value] += 1
            expected = Counter(self.spec.components)
            if found != expected:
                for name in sorted(set(expected) | set(found)):
                    have, want = found.get(name, 0), expected.get(name, 0)
                    if have != want:
                        violations.append(
                            f"Component conservation violated for '{name}': found {have} "
                            f"across zones {self.spec.component_zones}, expected {want}. "
                            "No component may be created, destroyed, or renamed."
                        )

        for list_field, count_field in self.spec.visibility.count_fields.items():
            if list_field in state:
                actual = len(state.get(list_field) or [])
                if state.get(count_field) != actual:
                    violations.append(
                        f"{count_field} is {state.get(count_field)} but {list_field} has "
                        f"{actual} item(s). Set {count_field} = len({list_field})."
                    )
            else:
                for pid, player in state.get("players", {}).items():
                    if list_field not in player:
                        continue
                    actual = len(player.get(list_field) or [])
                    if player.get(count_field) != actual:
                        violations.append(
                            f"{pid} {count_field} is {player.get(count_field)} but "
                            f"{list_field} has {actual} item(s). Set {count_field} = "
                            f"len({list_field})."
                        )

        return violations

    @staticmethod
    def _zone_values(state: dict, zone: str) -> list[str]:
        def values_of(container: dict, field: str) -> list[str]:
            value = container.get(field)
            if isinstance(value, list):
                return [v for v in value if isinstance(v, str) and v != HIDDEN]
            if isinstance(value, str) and value and value != HIDDEN:
                return [value]
            return []

        if zone.startswith("players.*."):
            field = zone[len("players.*.") :]
            collected: list[str] = []
            for player in state.get("players", {}).values():
                collected.extend(values_of(player, field))
            return collected
        return values_of(state, zone)

    # -- Prompting ----------------------------------------------------------------

    def system_prompt_addendum(self, num_players: int) -> str:
        phases = " -> ".join(self.spec.turn.phases)
        return (
            f"\n\n## This Game\nThis game has {num_players} players. Each turn moves through "
            f"these phases in order: {phases}. The driver rotates turns; you decide whether "
            "an action ends the acting player's turn, whether the round or game has ended, "
            "and all scoring, per the End Conditions and Scoring sections above."
        )
