"""Stage 4: validate a generated engine in a subprocess.

Two gates, both ALWAYS run (no short-circuit), both as child processes so
LLM-generated code never executes in the ingesting process:

1. the generic contract harness (termination, determinism, non-mutation,
   serializability) via random self-play — runs FIRST because a harness crash is
   unambiguous evidence the ENGINE is broken;
2. the generated pytest suite (rule fidelity per the digest) — a failure here is
   ambiguous: the engine may be wrong, or the generated test may be (the classic
   mistake is forgetting that ``apply()`` auto-advances).

The result carries both verdicts so the repair loop can route the fix at the
right target with the right evidence.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Generated engines only; the guideline in the contract docstring is ~50
# actions per decision, the hard validation gate is 100.
MAX_MENU_SIZE = 100

_HARNESS_SCRIPT = """
import sys
from pathlib import Path

from playtest.engine.contract import assert_engine_contract
from playtest.engine.loader import load_engine_from_path

engine = load_engine_from_path(Path(sys.argv[1]))
assert_engine_contract(engine, games_per_count=int(sys.argv[2]), max_menu_size=int(sys.argv[3]))
print("contract OK")
"""

_TESTS_MAY_BE_WRONG_HINT = (
    "NOTE: the engine PASSED structural validation (random self-play across all "
    "player counts: termination, determinism, no crashes, hidden-info "
    "serializability). The failing generated tests below may themselves be wrong — "
    "re-check each failure against the digest before assuming an engine bug. The "
    "most common test mistake: forgetting that apply() auto-advances to the next "
    "decision point — mandatory draws, refills, automa turns, scoring, and redeals "
    "have already resolved when apply() returns, and a reaction window may already "
    "be open (to_act may return a responder, not the next turn player)."
)


@dataclass(frozen=True)
class ValidationResult:
    harness_ok: bool
    tests_ok: bool
    feedback: str  # "" when both gates pass

    @property
    def ok(self) -> bool:
        return self.harness_ok and self.tests_ok


def _clip(text: str, max_chars: int = 6000) -> str:
    """Clip oversized output keeping head and tail, so the first failing
    traceback (usually the root cause the repair LLM needs) survives alongside
    pytest's trailing summary."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} chars omitted] ...\n{text[-tail:]}"


def validate_engine(
    config_dir: Path,
    games_per_count: int = 30,
    timeout: int = 600,
) -> ValidationResult:
    """Run both gates and report a routable verdict."""
    engine_path = config_dir / "engine.py"
    test_path = config_dir / "test_engine.py"

    harness = subprocess.run(
        [
            sys.executable,
            "-c",
            _HARNESS_SCRIPT,
            str(engine_path),
            str(games_per_count),
            str(MAX_MENU_SIZE),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    harness_ok = harness.returncode == 0

    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    tests_ok = tests.returncode == 0

    sections: list[str] = []
    if not harness_ok:
        sections.append(
            "STAGE: contract harness (random self-play) FAILED — the engine is "
            "broken.\n\n"
            f"{_clip(harness.stdout, 2000)}\n{_clip(harness.stderr)}"
        )
    if not tests_ok:
        header = "STAGE: generated pytest suite FAILED."
        if harness_ok:
            header += f"\n\n{_TESTS_MAY_BE_WRONG_HINT}"
        sections.append(f"{header}\n\n{_clip(tests.stdout)}\n{_clip(tests.stderr, 2000)}")

    return ValidationResult(
        harness_ok=harness_ok, tests_ok=tests_ok, feedback="\n\n".join(sections)
    )
