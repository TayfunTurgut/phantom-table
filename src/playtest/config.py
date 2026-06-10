import os
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from openai import OpenAI


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    gm_model: str = "gpt-4o"
    player_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    game_configs_dir: str = "game_configs"
    log_level: str = "INFO"
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "phantom-table"

    # Safety caps for the turn loop and agent tool loops (a crashing ceiling, not a
    # target). max_turns replaces the old LangGraph recursion_limit.
    max_turns: int = 500
    max_tool_iterations: int = 16
    max_observation_calls: int = 6
    # Per-turn cap on GM-rejected proposals before the run crashes with IllegalAction.
    # Each rejected attempt consumes one player-loop iteration, so this must stay well
    # below max_tool_iterations: worst case 6 observations + 4 rejections + 2 actions
    # = 12 < 16. (The GM's own tool loop has a separate cap in agents/gm.py.)
    max_action_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_tracing() -> None:
    """
    Propagate LangSmith settings to the environment variables that
    the langsmith SDK reads automatically.
    Call once at startup (cli.py entrypoint).
    """
    settings = get_settings()
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)


def maybe_wrap_openai(client: "OpenAI") -> "OpenAI":
    """
    Wrap an OpenAI client with langsmith tracing when enabled, so every
    completion/embedding call is captured as a traced span. Returns the
    client unchanged (and never imports langsmith) when tracing is off.
    """
    settings = get_settings()
    if settings.langsmith_tracing and settings.langsmith_api_key:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    return client
