"""Tests for settings loading and LangSmith tracing configuration (offline)."""

import os

import pytest

from playtest.config import configure_tracing, get_settings

_LANGCHAIN_VARS = ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT")
_LANGSMITH_VARS = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Reset the settings cache and scrub LangChain/LangSmith env so cases don't leak."""
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    for var in (*_LANGCHAIN_VARS, *_LANGSMITH_VARS):
        monkeypatch.delenv(var, raising=False)
    yield
    # configure_tracing() sets LANGCHAIN_* via os.environ.setdefault (not monkeypatch),
    # so scrub them explicitly to avoid leaking into other tests.
    for var in _LANGCHAIN_VARS:
        os.environ.pop(var, None)
    get_settings.cache_clear()


def test_configure_tracing_disabled_leaves_env_unset(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    get_settings.cache_clear()

    configure_tracing()

    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_configure_tracing_enabled_sets_langchain_env(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    get_settings.cache_clear()

    configure_tracing()

    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "test-key"
    assert os.environ["LANGCHAIN_PROJECT"] == "test-project"


def test_configure_tracing_enabled_without_key_is_noop(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    get_settings.cache_clear()

    configure_tracing()

    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_settings_langsmith_defaults_and_types(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "abc123")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.langsmith_tracing is True
    assert settings.langsmith_api_key == "abc123"
    assert settings.langsmith_project == "phantom-table"


def test_settings_langsmith_defaults_when_absent():
    settings = get_settings()

    assert settings.langsmith_tracing is False
    assert settings.langsmith_api_key is None
    assert settings.langsmith_project == "phantom-table"
