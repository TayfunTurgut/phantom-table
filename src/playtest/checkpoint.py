"""Per-turn game snapshots so a crashed run can be resumed from its last turn.

A checkpoint captures everything needed to continue a game that engine state alone
does not: the raw (JSON-serializable) engine ``state`` plus the two pieces of
private player memory the engine deliberately keeps outside state — each seat's
pending event ``buffers`` and its self-authored ``notebooks``. Given the engine
(re-resolved from ``game_ref``) and this bundle, ``run_session`` can re-enter the
turn loop at ``step`` and finish the game.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


@dataclass
class Checkpoint:
    """A resumable snapshot taken at the top of a turn (before any seat acts)."""

    game_ref: str
    num_players: int
    seed: int
    archetypes: list[str]
    session_id: str
    step: int
    state: dict
    buffers: dict[str, list[str]]
    notebooks: dict[str, str]
    version: int = 1


def write_checkpoint(path: str, checkpoint: Checkpoint) -> None:
    """Atomically write ``checkpoint`` as JSON.

    Writes to a sibling ``.tmp`` file and ``os.replace``s it into place so a crash
    mid-write can never leave a half-written (unresumable) checkpoint.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(asdict(checkpoint), fh, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Checkpoint:
    with open(path, encoding="utf-8") as fh:
        return Checkpoint(**json.load(fh))
