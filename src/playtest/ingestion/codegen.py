"""Stages 2-3: digest → generated engine module and its generated test module."""

from __future__ import annotations

import ast
import inspect
import re

import playtest.engine
from playtest.errors import PlaytestError
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


def _engine_contract() -> str:
    return playtest.engine.__doc__ or ""


def _reference_engine_source() -> str:
    from playtest.games import love_letter

    return inspect.getsource(love_letter)


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


_ENGINE_SYSTEM_PROMPT = """You are an expert Python engineer generating a complete,
deterministic game engine module from a game digest. You write exactly one
self-contained Python module and nothing else.

Output format: a single ```python fenced block containing the full module.

THE CONTRACT (your module must implement this exactly):

{contract}

A complete reference implementation for Love Letter — match its style, structure,
and conventions precisely (single Game class, effect handler methods, deepcopy on
apply, seeded RNG chain, factual past-tense events, observe() enforcing hidden
information):

```python
{reference}
```

Additional requirements:
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


def generate_engine_source(
    client: LLMClient,
    digest_json: str,
    rulebook_text: str,
    feedback: str | None = None,
    previous_source: str | None = None,
) -> str:
    feedback_section = ""
    if feedback and previous_source:
        feedback_section = _REPAIR_SECTION.format(
            previous_source=previous_source, feedback=feedback
        )
    raw = client.complete(
        [
            {
                "role": "system",
                "content": _ENGINE_SYSTEM_PROMPT.format(
                    contract=_engine_contract(), reference=_reference_engine_source()
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
    return source


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
- verify setup deals correctly for each player count (counts, conserved components);
- craft specific states using the digest's pinned state shape and assert each
  action's effect, covering the edge cases the digest lists;
- verify end conditions and every scoring tiebreaker;
- verify observations hide what the digest says is hidden (and that the
  "spectator" seat sees everything);
- verify component conservation across a random game (sum components across all
  zones of the state after each step).

Use engine.legal_actions(...) to obtain actions and engine.apply(...) to play
them (match on Action.name and Action.args). Use enough players that
eliminations don't immediately end a round when testing mid-round effects.

THE AUTO-ADVANCE TRAP (the most common generated-test bug — avoid it):
After apply() returns, the game has ALREADY advanced to the next decision point.
The next player has auto-drawn; forced reveals, round scoring, and redeals have
already happened. So:

- NEVER assert that a seat's hand is "unchanged" after apply() — the next actor's
  hand legitimately gained a card.
  WRONG:  state2, _ = engine.apply(state, [priest_action_targeting_p2])
          assert state2["hands"][1] == ["Princess"]        # p2 may have auto-drawn!
  RIGHT:  assert "Princess" in state2["hands"][1]
- If your scenario ends the ROUND (deck empty, last player standing), the engine
  has already scored it AND dealt the next round: hands, discards, and
  protections are RESET by the time apply() returns. Assert on tokens/scores,
  status(), and event texts — never on zone contents the redeal wiped.
- When asserting hand sizes or deck counts, account for the auto-draw of the
  next player to act.
- Before asserting any computed outcome (tiebreak sums, scores), derive the
  number step by step in a comment — include every card involved, including
  cards your scenario itself just moved into a discard pile.
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
            {"role": "system", "content": _TEST_SYSTEM_PROMPT},
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
    return source
