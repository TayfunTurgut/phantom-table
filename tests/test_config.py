"""Tests for settings loading (offline)."""

import pytest

from playtest.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Reset the settings cache and run from a dir without a .env so the
    developer's real .env can't bleed into these offline cases."""
    get_settings.cache_clear()
    monkeypatch.chdir(tmp_path)
    yield
    get_settings.cache_clear()


def test_claude_model_defaults():
    settings = get_settings()

    assert settings.claude_player_model == "sonnet"
    assert settings.claude_digest_model == "sonnet"
    assert settings.claude_codegen_model == "sonnet"
    assert settings.claude_cli_path == "claude"
    assert settings.claude_code_oauth_token is None


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_PLAYER_MODEL", "opus")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok-xyz")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.claude_player_model == "opus"
    assert settings.claude_code_oauth_token == "tok-xyz"


def test_safety_cap_default():
    assert get_settings().max_steps >= 500
