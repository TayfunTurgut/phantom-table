"""Errors that crash a playtest run.

A crash is a finding, not a recovery point: rather than reconcile or auto-correct a
broken game, the harness fails fast with a clear diagnostic so the underlying cause
(an engine bug — i.e. an ingestion finding — or a harness bug) can be inspected.
"""


class PlaytestError(Exception):
    """Base for errors that should terminate a playtest run."""


class EngineCrash(PlaytestError):
    """A game engine raised during a session.

    Wraps the original exception with enough context to reproduce: the seed, the
    step index, and the actions submitted at the failing step. An engine crash at
    runtime is an ingestion finding — the generated engine must be regenerated or
    repaired, not worked around.
    """

    def __init__(
        self,
        original: Exception,
        seed: int,
        step: int,
        actions: list | None = None,
    ) -> None:
        self.original = original
        self.seed = seed
        self.step = step
        self.actions = actions or []
        super().__init__(
            f"engine raised at step {step} (seed={seed}, "
            f"actions={[getattr(a, 'label', a) for a in self.actions]}): "
            f"{type(original).__name__}: {original}"
        )
