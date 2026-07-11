"""Stages 2-3: digest → generated engine module and its generated test module."""

from __future__ import annotations

import ast
import hashlib
import inspect
import re
import subprocess
import sys
from importlib.util import find_spec
from types import ModuleType

import playtest.engine
from playtest.errors import PlaytestError
from playtest.games import bull_run, love_letter
from playtest.ingestion.schemas import GameDigest, MechanicTag
from playtest.llm import LLMClient

_ALLOWED_IMPORT_ROOTS = {
    # The engine contract allows stdlib + playtest.engine. This list is the
    # stdlib subset a game engine could plausibly need; anything else (os, sys,
    # socket, subprocess, pathlib...) is rejected as a safety/purity violation.
    "random",
    "copy",
    "json",
    "math",
    "itertools",
    "functools",
    "collections",
    "dataclasses",
    "typing",
    "enum",
    "string",
    "__future__",
    "playtest",
}

_TEST_ALLOWED_EXTRA = {"pytest", "pathlib"}

# Two shipped exemplars. bull_run demonstrates the phase-machine / simultaneous-
# commit idioms (a mid-resolution phase key, a single apply call taking every
# committing seat); love_letter demonstrates sequential hidden-hand play. The
# selector routes on the digest's structural mechanics so the exemplar shares the
# game's decision structure instead of biasing every game toward Love Letter.
_EXEMPLARS: dict[str, ModuleType] = {"love_letter": love_letter, "bull_run": bull_run}

# Mechanics that make bull_run the closer structural match.
_BULL_RUN_MECHANICS: frozenset[MechanicTag] = frozenset(
    {"simultaneous_decisions", "multi_stage_turns", "reaction_windows"}
)


def select_exemplar(digest: GameDigest, override: str | None = None) -> tuple[str, str]:
    """Pick the reference engine whose decision structure best matches ``digest``.

    Returns ``(name, source)``. An explicit ``override`` (a registry key) wins;
    otherwise a digest tagged with any phase-machine mechanic routes to bull_run
    and everything else to love_letter.
    """
    if override is not None:
        if override not in _EXEMPLARS:
            raise PlaytestError(f"unknown exemplar {override!r}; valid names: {sorted(_EXEMPLARS)}")
        name = override
    elif _BULL_RUN_MECHANICS.intersection(digest.mechanics):
        name = "bull_run"
    else:
        name = "love_letter"
    return name, inspect.getsource(_EXEMPLARS[name])


def _all_exemplar_sources() -> list[str]:
    """Every exemplar's source, in stable order (a fingerprint seam)."""
    return [inspect.getsource(_EXEMPLARS[name]) for name in sorted(_EXEMPLARS)]


def _engine_contract() -> str:
    return playtest.engine.__doc__ or ""


def extract_python(content: str) -> str:
    """Pull the (largest) fenced python block out of a completion, or take it raw."""
    blocks = re.findall(r"```(?:python)?\n(.*?)```", content, flags=re.DOTALL)
    source = max(blocks, key=len) if blocks else content
    return source.strip() + "\n"


def check_engine_source(source: str, *, extra_allowed: set[str] | None = None) -> None:
    """Reject code that doesn't parse or imports outside the allowlist."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PlaytestError(f"generated code does not parse: {exc}") from exc
    allowed = _ALLOWED_IMPORT_ROOTS | (extra_allowed or set())
    for node in ast.walk(tree):
        roots = []
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        for root in roots:
            if root and root not in allowed:
                raise PlaytestError(
                    f"generated code imports {root!r}, which is outside the allowlist "
                    f"{sorted(allowed)}"
                )


def lint_source(source: str, filename: str) -> None:
    """Fast ruff error-gate: undefined names and syntax-adjacent errors (F, E9),
    run BEFORE the multi-minute validation stage. Best-effort — ruff is a dev
    extra, so a missing or unstartable ruff is a silent no-op, not a failure."""
    if find_spec("ruff") is None:
        return
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--select",
                "F,E9",
                "--stdin-filename",
                filename,
                "-",
            ],
            input=source,
            capture_output=True,
            text=True,
        )
    except OSError:
        return
    if result.returncode != 0 and result.stdout.strip():
        raise PlaytestError(f"generated code failed ruff check ({filename}):\n\n{result.stdout}")


# Mechanic-conditional guidance appended to the engine prompt. The two starred
# blocks (reaction_windows, multi_stage_turns) carry literal code sketches; those
# sketches must stay consistent with contract points 4/5 and with the shipped
# exemplars (bull_run's choose_row phase machine, tests/test_reactions.py's
# window). State is JSON-only, so sketches store primitives, never Action objects.
_GUIDANCE_BLOCKS: dict[MechanicTag, str] = {
    "simultaneous_decisions": """\
SIMULTANEOUS DECISIONS. Several seats choose at once. `to_act` returns EVERY seat
that must commit this step, and a SINGLE `apply` call receives one action per
those seats and resolves them together. Store each seat's committed choice under
a state key (e.g. state["committed"][seat]) and hide it from the OTHER seats'
`observe` until the reveal: a seat sees its own commitment in the clear, others
see only whether that seat has committed.""",
    "reaction_windows": """\
REACTION WINDOWS (block / challenge / Nope / instant). A "may respond" window is
its own decision point, NOT part of resolving the announced action. Announce the
action into state, flip the phase, and STOP — resolve nothing yet. Offer the
window to every seat the rules permit, never only to seats whose hidden cards
make responding useful: being asked is itself information. Sketch:

    # apply(), main phase: announce and OPEN the window; resolve nothing yet.
    # Store primitives (name/args), never the Action object — state is JSON-only.
    state["pending"] = {"action": chosen.name, "args": chosen.args,
                        "actor": seat, "asked": []}
    state["phase"] = "reaction"
    return state, [Event(f"{seat} announced {chosen.name}.")]

    # to_act during the window: next eligible responder not yet asked.
    if state["phase"] == "reaction":
        for s in self._eligible_responders(state):   # by RULES, not hand contents
            if s not in state["pending"]["asked"]:
                return [s]
        return []

    # legal_actions during the window: the reaction plus an explicit decline.
    if state["phase"] == "reaction":
        menu = [Action(seat=seat, name="decline", label="Decline")]
        if self._may_react(state, seat):             # eligibility, never hidden hand
            menu.append(Action(seat=seat, name="block", label="Block"))
        return menu

    # apply() during the window: record the answer; close only when done.
    if action.name == "decline":
        state["pending"]["asked"].append(seat)
    else:
        ...   # apply the reaction (it may cancel the pending action)
    if all(s in state["pending"]["asked"] for s in self._eligible_responders(state)):
        ...   # resolve or cancel the pending action, clear pending, restore phase""",
    "multi_stage_turns": """\
MULTI-STAGE TURNS. When fully binding one decision would enumerate more than
~50 actions (open bid amounts, free placement, multi-leg moves), split it into
consecutive stages: record the partial choice in state, keep the SAME seat in
`to_act`, and offer the next stage's fully bound actions. Emit the event(s)
describing the whole decision only when its FINAL stage resolves. Sketch:

    # legal_actions, stage 1: the coarse choice.
    if state["pending_choice"] is None:
        return [Action(seat=seat, name="bid", label="Bid"),
                Action(seat=seat, name="pass", label="Pass")]

    # apply(), stage 1: record the partial choice; SAME seat acts again, no event.
    if action.name == "bid":
        state["pending_choice"] = {"seat": seat}
        return state, []

    # legal_actions, stage 2: the fully bound follow-up, still this seat.
    if state["pending_choice"] and state["pending_choice"]["seat"] == seat:
        lo, hi = state["high_bid"] + 1, state["coins"][seat]
        return [Action(seat=seat, name="raise_to", args={"amount": a},
                        label=f"Bid {a}") for a in range(lo, hi + 1)]

    # apply(), final stage: commit the whole decision and emit its event now.
    state["high_bid"] = action.args["amount"]
    state["pending_choice"] = None
    return state, [Event(f"{seat} bid {action.args['amount']}.")]""",
    "open_supply": """\
OPEN SUPPLY. Some components are created or destroyed through a shared, unlimited
or replenishing pool (a market, a bank). Model the supply as an explicit state
key that grows or shrinks as effects dictate; do NOT enforce global component
conservation across such a game.""",
    "board_or_map": """\
BOARD OR MAP. Represent the board / display as plain JSON keyed by named spaces
(e.g. state["board"][space] = ...). It is public: include it verbatim in every
seat's `observe`, including the spectator's.""",
}


def _engine_guidance(mechanics: list[MechanicTag]) -> str:
    """The guidance blocks for a digest's tags, deduped and in stable order."""
    tags = set(mechanics)
    blocks = [text for tag, text in _GUIDANCE_BLOCKS.items() if tag in tags]
    if not blocks:
        return ""
    return "MECHANIC-SPECIFIC GUIDANCE:\n\n" + "\n\n".join(blocks) + "\n\n"


_ENGINE_SYSTEM_PROMPT = """You are an expert Python engineer generating a complete,
deterministic game engine module from a game digest. You write exactly one
self-contained Python module and nothing else.

Output format: a single ```python fenced block containing the full module.

THE CONTRACT (your module must implement this exactly):

{contract}

REFERENCE EXEMPLAR ({exemplar_name}) — a complete, working implementation of a
DIFFERENT game, chosen because it shares this game's decision structure. Match
the CONVENTIONS listed below; adapt the structure freely wherever this game's
rules differ. The DIGEST, not the exemplar, defines the game you are building —
never carry over the exemplar's cards, phases, or rules.

```python
{exemplar_source}
```

CONVENTIONS (this is what "match the exemplar" means):
- One `Game` class implementing the contract; standard library + playtest.engine
  only, no import-time side effects.
- Resolution logic lives in small effect-handler methods the public methods call.
- `apply` deep-copies the incoming state on entry (`state = copy.deepcopy(state)`)
  and never mutates its argument.
- Randomness runs off the rng_seed chain: `rng = random.Random(state["rng_seed"])`,
  use it, then `new_state["rng_seed"] = rng.randrange(2**32)`.
- Events are factual, past-tense records with correct per-seat visibility
  (`visible_to=None` public; a seat tuple for private reveals).
- `observe` reduces other seats' hidden zones to counts or backs so the hidden
  information cannot be recovered; the "spectator" seat sees everything.
- `apply` auto-advances through every forced, decision-free step (mandatory
  draws, refills, chance reveals, scoring, redeals), stopping at the next
  decision point.
- Keep an explicit `state["phase"]` key whenever the game has more than one kind
  of decision point, and route `to_act` / `legal_actions` / `apply` on it.

{guidance}Additional requirements:
- Use the state shape pinned in the digest verbatim.
- Implement every action in the digest, including all listed edge cases.
- Implement every ambiguity ruling exactly as the digest resolves it.
- Imports: standard library and playtest.engine only.
- No comments narrating what code does; only non-obvious WHY comments.
"""

_ENGINE_USER_PROMPT = """DIGEST (implement this game):

{digest_json}

FULL RULEBOOK (the digest is authoritative for rulings; use the rulebook for
any detail the digest summarizes):

{rulebook}
{feedback_section}
Write the complete engine module now."""

_REPAIR_SECTION = """
YOUR PREVIOUS ATTEMPT FAILED VALIDATION.

Previous module:

```python
{previous_source}
```

Validation failure:

{feedback}

Diagnose the root cause and output the corrected COMPLETE module (not a diff).
If the failure came from a generated test that contradicts the digest, still
output the engine module — the tests are regenerated separately.
"""

# No previous_source yet: the failure struck before a module existed to show
# (unparseable code, a disallowed import, or a lint error caught pre-validation).
_FEEDBACK_ONLY_REPAIR_SECTION = """
YOUR PREVIOUS ATTEMPT FAILED BEFORE VALIDATION:

{feedback}

Diagnose the root cause and output a corrected COMPLETE module (not a diff).
"""


def generate_engine_source(
    client: LLMClient,
    digest: GameDigest,
    digest_json: str,
    rulebook_text: str,
    exemplar: tuple[str, str],
    feedback: str | None = None,
    previous_source: str | None = None,
) -> str:
    exemplar_name, exemplar_source = exemplar
    feedback_section = ""
    if feedback and previous_source:
        feedback_section = _REPAIR_SECTION.format(
            previous_source=previous_source, feedback=feedback
        )
    elif feedback:
        feedback_section = _FEEDBACK_ONLY_REPAIR_SECTION.format(feedback=feedback)
    raw = client.complete(
        [
            {
                "role": "system",
                "content": _ENGINE_SYSTEM_PROMPT.format(
                    contract=_engine_contract(),
                    exemplar_name=exemplar_name,
                    exemplar_source=exemplar_source,
                    guidance=_engine_guidance(digest.mechanics),
                ),
            },
            {
                "role": "user",
                "content": _ENGINE_USER_PROMPT.format(
                    digest_json=digest_json,
                    rulebook=rulebook_text,
                    feedback_section=feedback_section,
                ),
            },
        ],
        role="codegen",
    )
    source = extract_python(raw)
    check_engine_source(source)
    lint_source(source, "engine.py")
    return source


# Component conservation only holds when nothing is created/destroyed through an
# open supply; under open_supply the conservation check is replaced (see below).
_CONSERVATION_BULLET = (
    "- verify component conservation across a random game: sum each component "
    "across all zones of the state after every step; the totals never change."
)

# Mechanic-conditional test guidance, selected by digest.mechanics.
_TEST_GUIDANCE_BLOCKS: dict[MechanicTag, str] = {
    "open_supply": (
        "- The supply is open: assert supply counts change exactly as the "
        "effects specify; do NOT assert global component conservation."
    ),
    "player_elimination": (
        "- Use enough players that eliminations don't immediately end a round "
        "when you test mid-round effects."
    ),
    "simultaneous_decisions": (
        "- Simultaneous commit: pass one action per acting seat to a SINGLE "
        "apply() call, and assert each seat's observation hides the other "
        "seats' commitments until the reveal."
    ),
    "reaction_windows": (
        "- Reaction window: after the trigger, to_act() returns the responder "
        "(not the next turn player) and legal_actions() includes the explicit "
        "decline/pass; the announced action resolves only when the window closes."
    ),
    "multi_stage_turns": (
        "- Staged turn: after the first stage, to_act() returns the SAME seat "
        "and legal_actions() offers the next stage's bound options; the "
        "whole-action event appears only at the final stage."
    ),
}


def _test_guidance_for(mechanics: list[MechanicTag]) -> str:
    """Test-suite guidance bullets for a digest's tags, in stable order."""
    tags = set(mechanics)
    lines: list[str] = []
    if "open_supply" not in tags:
        lines.append(_CONSERVATION_BULLET)
    lines.extend(text for tag, text in _TEST_GUIDANCE_BLOCKS.items() if tag in tags)
    return "\n".join(lines)


_TEST_SYSTEM_PROMPT = """You are an expert Python engineer writing a pytest suite for a
generated game engine. The suite verifies the engine implements the DIGEST's
rules — not just that it runs. You write exactly one self-contained test module.

Output format: a single ```python fenced block containing the full module.

The test module MUST begin by loading the engine it sits next to:

```python
from pathlib import Path

import pytest

from playtest.engine import Action
from playtest.engine.loader import load_engine_from_path

engine = load_engine_from_path(Path(__file__).parent / "engine.py")
```

Write focused tests that:
- verify setup produces the digest's initial state for each player count (the
  zones, counts, and per-seat data the digest specifies);
- craft specific states using the digest's pinned state shape and assert each
  action's effect, covering the edge cases the digest lists;
- verify end conditions and every scoring tiebreaker;
- verify observations hide what the digest says is hidden (and that the
  "spectator" seat sees everything).
{guidance}
Use engine.legal_actions(...) to obtain actions and engine.apply(...) to play
them (match on Action.name and Action.args).

THE AUTO-ADVANCE TRAP (the most common generated-test bug — avoid it):
After apply() returns, the engine has ALREADY auto-advanced through every forced,
decision-free step — mandatory draws and refills, chance reveals, automa turns,
round scoring, redeals — to the next decision point. So:

- NEVER assert that untouched state is "unchanged" after apply(): the seat that
  acts next may legitimately have drawn or refilled.
  WRONG:  state2, _ = engine.apply(state, [some_action])
          assert state2["some_zone"]["player_2"] == ["X"]  # player_2 may have drawn!
  RIGHT:  assert "X" in state2["some_zone"]["player_2"]
- If your scenario ends a ROUND or the GAME, the engine has already scored it and
  dealt any next round by the time apply() returns: per-round zones (hands,
  discards, board spaces) are RESET. Assert on scores, status(), and event texts,
  never on zone contents the transition wiped.
- A reaction window may already be OPEN after apply(): to_act() can return a
  responder, not the next turn player. Read to_act() instead of assuming it.
- Before asserting any computed outcome (tiebreak sums, scores), derive the
  number step by step in a comment — include every component your scenario moved.
"""

_TEST_USER_PROMPT = """DIGEST:

{digest_json}

ENGINE MODULE UNDER TEST:

```python
{engine_source}
```
{feedback_section}
Write the complete test module now."""

_TEST_REPAIR_SECTION = """
YOUR PREVIOUS TEST MODULE IS BELOW. SOME OF ITS TESTS FAILED.

```python
{previous_test_source}
```

Validation failure:

{feedback}

Repair rules — follow them exactly:
1. Re-derive what the digest says should happen in each FAILING test's scenario,
   step by step (auto-advance included), and fix ONLY the failing tests.
2. Reproduce every PASSING test byte-for-byte unchanged. Do not rename, reorder,
   reword, or "improve" them.
3. If, after re-deriving, a failing test cannot be reconciled with the digest and
   the engine contract (e.g. it inspects state a redeal already wiped), DELETE it
   and leave a one-line comment in its place: # removed <test_name>: <reason>.
Output the complete corrected module.
"""


def generate_test_source(
    client: LLMClient,
    digest: GameDigest,
    digest_json: str,
    engine_source: str,
    feedback: str | None = None,
    previous_test_source: str | None = None,
) -> str:
    feedback_section = ""
    if feedback and previous_test_source:
        feedback_section = _TEST_REPAIR_SECTION.format(
            previous_test_source=previous_test_source, feedback=feedback
        )
    elif feedback:
        feedback_section = (
            f"\nA PREVIOUS VALIDATION RUN FAILED:\n\n{feedback}\n\n"
            "If the failure was a wrong test (one that contradicts the digest), fix it.\n"
        )
    raw = client.complete(
        [
            {
                "role": "system",
                "content": _TEST_SYSTEM_PROMPT.format(
                    guidance=_test_guidance_for(digest.mechanics)
                ),
            },
            {
                "role": "user",
                "content": _TEST_USER_PROMPT.format(
                    digest_json=digest_json,
                    engine_source=engine_source,
                    feedback_section=feedback_section,
                ),
            },
        ],
        role="codegen",
    )
    source = extract_python(raw)
    check_engine_source(source, extra_allowed=_TEST_ALLOWED_EXTRA)
    lint_source(source, "test_engine.py")
    return source


def prompts_fingerprint() -> str:
    """A sha256 digest over every prompt/exemplar the codegen stages depend on.

    Recorded in meta.json for provenance: any edit to a prompt constant, the live
    engine contract docstring, a shipped exemplar's source, or a guidance block
    changes this fingerprint. Parts are joined with a NUL separator so a shift of
    text across a boundary still changes the hash."""
    from playtest.ingestion.digest import _SYSTEM_PROMPT as _DIGEST_SYSTEM_PROMPT

    parts = [
        _ENGINE_SYSTEM_PROMPT,
        _ENGINE_USER_PROMPT,
        _REPAIR_SECTION,
        _FEEDBACK_ONLY_REPAIR_SECTION,
        _TEST_SYSTEM_PROMPT,
        _TEST_USER_PROMPT,
        _TEST_REPAIR_SECTION,
        _DIGEST_SYSTEM_PROMPT,
        _engine_contract(),
        _CONSERVATION_BULLET,
        *_all_exemplar_sources(),
        *(f"{tag}={text}" for tag, text in sorted(_GUIDANCE_BLOCKS.items())),
        *(f"{tag}={text}" for tag, text in sorted(_TEST_GUIDANCE_BLOCKS.items())),
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
