"""The ingestion pipeline: rulebook text → validated, playable game engine.

    rulebook ─► digest ─► engine codegen ─► test codegen ─► validate ─► persist
                  ▲                              ▲             │
                  │            (tests-only repair)◄────────────┤ harness ok, tests fail
                  └──────────── engine repair ◄────────────────┘ harness fail

The digest is generated once (it is the reviewable spec). Validation produces two
independent verdicts and repairs are routed accordingly:

- contract-harness failure → the ENGINE is broken → regenerate the engine with the
  harness traceback;
- harness pass + generated-test failure → the TESTS are suspect (auto-advance
  mistakes are common) → regenerate only the tests, up to ``max_test_repairs``
  times, before falling back to engine regeneration.

Every validation round is archived under ``<config>/attempts/NN/`` for autopsy.
Ingestion fails loudly when the engine-attempt budget is exhausted, leaving the
digest and all attempts on disk.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from playtest.config import get_settings
from playtest.errors import PlaytestError
from playtest.ingestion.chunker import chunk_rulebook, embed_and_store
from playtest.ingestion.codegen import generate_engine_source, generate_test_source
from playtest.ingestion.digest import generate_digest, save_digest
from playtest.ingestion.schemas import GameArtifacts
from playtest.ingestion.validate import ValidationResult, validate_engine
from playtest.llm import LLMClient, create_llm_client

if TYPE_CHECKING:
    from collections.abc import Iterator

_console = Console()


def _archive_attempt(config_dir: Path, index: int, failure: str) -> None:
    """Snapshot the current engine/tests plus the failure text for autopsy."""
    attempt_dir = config_dir / "attempts" / f"{index:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    for name in ("engine.py", "test_engine.py"):
        source = config_dir / name
        if source.is_file():
            shutil.copy(source, attempt_dir / name)
    (attempt_dir / "failure.txt").write_text(failure or "OK", encoding="utf-8")


def _validate_with_test_repairs(
    client: LLMClient,
    config_dir: Path,
    digest_json: str,
    engine_source: str,
    test_source: str,
    max_test_repairs: int,
    rounds: Iterator[int],
) -> tuple[ValidationResult, int]:
    """Validate; while the engine looks structurally sound but its generated
    tests fail, suspect the tests and regenerate only them.

    Returns the final verdict and the number of test repairs performed.
    """
    repairs = 0
    while True:
        _console.print("[cyan]Validating: contract harness + generated tests...[/cyan]")
        result = validate_engine(config_dir)
        _archive_attempt(config_dir, next(rounds), result.feedback)

        if result.ok or not result.harness_ok or repairs == max_test_repairs:
            return result, repairs

        _console.print(
            "[yellow]Harness passed but generated tests failed; regenerating tests only.[/yellow]"
        )
        try:
            test_source = generate_test_source(
                client,
                digest_json,
                engine_source,
                feedback=result.feedback,
                previous_test_source=test_source,
            )
        except PlaytestError:
            return result, repairs  # malformed test module → fall back to engine repair
        (config_dir / "test_engine.py").write_text(test_source, encoding="utf-8")
        repairs += 1


def ingest_rulebook(
    rulebook_path: str,
    game_name: str,
    max_attempts: int = 4,
    max_test_repairs: int = 3,
    client: LLMClient | None = None,
) -> GameArtifacts:
    # The name becomes a directory that is rmtree'd below — reject traversal first.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", game_name):
        raise PlaytestError(
            f"invalid game name {game_name!r}: use only letters, digits, '_' and '-' "
            "(it names the config directory)"
        )
    settings = get_settings()
    if client is None:
        client = create_llm_client(settings)

    rulebook_text = Path(rulebook_path).read_text(encoding="utf-8")
    config_dir = Path(settings.game_configs_dir) / game_name
    if config_dir.exists():
        shutil.rmtree(config_dir)
    config_dir.mkdir(parents=True)
    (config_dir / "rulebook.txt").write_text(rulebook_text, encoding="utf-8")

    _console.print("[cyan]Embedding rulebook for Q&A...[/cyan]")
    chunks = chunk_rulebook(rulebook_text, game_name=game_name)
    embed_and_store(chunks, game_name, str(config_dir / "chromadb"))

    _console.print(f"[cyan]Generating digest with {client.models['digest']}...[/cyan]")
    digest = generate_digest(client, rulebook_text)
    save_digest(digest, config_dir)
    digest_json = json.dumps(digest.model_dump(), indent=2)

    feedback: str | None = None
    previous_source: str | None = None
    rounds = count(1)  # one shared sequence numbers every attempts/NN archive
    test_repairs_total = 0
    for attempt in range(1, max_attempts + 1):
        _console.print(
            f"[cyan]Generating engine (attempt {attempt}/{max_attempts}) "
            f"with {client.models['codegen']}...[/cyan]"
        )
        try:
            engine_source = generate_engine_source(
                client,
                digest_json,
                rulebook_text,
                feedback=feedback,
                previous_source=previous_source,
            )
            test_source = generate_test_source(
                client, digest_json, engine_source, feedback=feedback
            )
        except PlaytestError as exc:  # unparseable code / disallowed import
            feedback = f"STAGE: code generation produced an invalid module.\n\n{exc}"
            _archive_attempt(config_dir, next(rounds), feedback)
            _console.print(f"[yellow]Attempt {attempt} failed generation; repairing.[/yellow]")
            continue
        (config_dir / "engine.py").write_text(engine_source, encoding="utf-8")
        (config_dir / "test_engine.py").write_text(test_source, encoding="utf-8")

        result, repairs = _validate_with_test_repairs(
            client, config_dir, digest_json, engine_source, test_source, max_test_repairs, rounds
        )
        test_repairs_total += repairs

        if result.ok:
            (config_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "game_name": digest.game_name,
                        "min_players": digest.min_players,
                        "max_players": digest.max_players,
                        "llm_backend": settings.llm_backend,
                        "digest_model": client.models["digest"],
                        "codegen_model": client.models["codegen"],
                        "attempts": attempt,
                        "test_repairs": test_repairs_total,
                        "ingested_at": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            _console.print(
                f"[green]Engine validated (attempt {attempt}, "
                f"{test_repairs_total} test repair(s)).[/green]"
            )
            return GameArtifacts(config_dir)

        feedback = result.feedback
        previous_source = engine_source
        _console.print(f"[yellow]Attempt {attempt} failed validation; repairing.[/yellow]")

    raise PlaytestError(
        f"ingestion failed after {max_attempts} attempts; digest and all attempts "
        f"left in {config_dir} for inspection. Last failure:\n{feedback}"
    )
