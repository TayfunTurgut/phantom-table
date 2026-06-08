import pytest
from openai import OpenAI

from playtest.config import Settings, get_settings


@pytest.fixture
def settings() -> Settings:
    """Application settings loaded from the environment / .env file."""
    try:
        return get_settings()
    except Exception as exc:
        pytest.skip(f"Settings unavailable (is OPENAI_API_KEY set?): {exc}")


@pytest.fixture
def openai_client(settings: Settings) -> OpenAI:
    """OpenAI client; skipped when no API key is configured."""
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAI(api_key=settings.openai_api_key)
