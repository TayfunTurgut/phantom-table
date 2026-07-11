"""CLI argument handling."""

from types import SimpleNamespace

from playtest import cli


def _fake_artifacts(name: str) -> SimpleNamespace:
    digest = SimpleNamespace(
        game_name=name,
        min_players=2,
        max_players=4,
        actions=[SimpleNamespace(name="pass")],
    )
    return SimpleNamespace(digest=digest, config_dir=f"game_configs/{name}", meta={})


def test_ingest_name_defaults_to_rulebook_stem(monkeypatch, tmp_path):
    import playtest.ingestion.pipeline as pipeline

    seen: dict = {}

    def fake_ingest(rulebook, name, **kwargs):
        seen["name"] = name
        return _fake_artifacts(name)

    monkeypatch.setattr(pipeline, "ingest_rulebook", fake_ingest)
    cli._run_ingest(str(tmp_path / "love_letter.txt"), None)
    assert seen["name"] == "love_letter"


def test_ingest_explicit_name_wins(monkeypatch, tmp_path):
    import playtest.ingestion.pipeline as pipeline

    seen: dict = {}

    def fake_ingest(rulebook, name, **kwargs):
        seen["name"] = name
        return _fake_artifacts(name)

    monkeypatch.setattr(pipeline, "ingest_rulebook", fake_ingest)
    cli._run_ingest(str(tmp_path / "love_letter.txt"), "custom_name")
    assert seen["name"] == "custom_name"
