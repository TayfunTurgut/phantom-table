"""Load a GameEngine from a generated config dir or a dotted import path."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from playtest.engine import GameEngine

_REQUIRED_ATTRS = (
    "game_name",
    "min_players",
    "max_players",
    "setup",
    "to_act",
    "legal_actions",
    "apply",
    "observe",
    "status",
)


def _instantiate(module: object, source: str) -> GameEngine:
    game_cls = getattr(module, "Game", None)
    if game_cls is None:
        raise ValueError(f"{source} does not define a Game class")
    engine = game_cls()
    missing = [attr for attr in _REQUIRED_ATTRS if not hasattr(engine, attr)]
    if missing:
        raise ValueError(f"{source} Game is missing required attributes: {missing}")
    return engine


def load_engine_from_path(engine_path: Path) -> GameEngine:
    """Import ``engine.py`` from a game config directory and return its Game."""
    engine_path = engine_path.resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"no engine file at {engine_path}")
    module_name = f"playtest_generated_{engine_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, engine_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import engine from {engine_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return _instantiate(module, str(engine_path))


def load_engine(ref: str) -> GameEngine:
    """Load an engine from a config dir, an engine.py path, or a dotted module path.

    Accepted forms:
    - ``game_configs/love_letter`` (directory containing engine.py)
    - ``game_configs/love_letter/engine.py``
    - ``playtest.games.love_letter`` (importable module with a Game class)
    """
    path = Path(ref)
    if path.is_dir():
        return load_engine_from_path(path / "engine.py")
    if path.suffix == ".py" and path.is_file():
        return load_engine_from_path(path)
    module = importlib.import_module(ref)
    return _instantiate(module, ref)
