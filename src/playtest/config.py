import os
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from openai import OpenAI


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Which LLM adapter runs completions: "openai" (API) or "claude_cli"
    # (headless `claude -p`, billed to the Claude subscription).
    llm_backend: str = "openai"
    # Which embedding backend indexes rulebooks: "openai" or "local" (ChromaDB's
    # built-in ONNX model — free and offline).
    embedding_backend: str = "openai"
    llm_timeout_seconds: int = 900

    # --- OpenAI backend ---
    openai_api_key: str | None = None
    # Runtime decisions: one structured call per decision — cheap tier.
    player_model: str = "gpt-5-mini"
    # One-time per-game generation: digest + engine + tests — strong tier.
    digest_model: str = "gpt-5"
    codegen_model: str = "gpt-5"
    embedding_model: str = "text-embedding-3-small"

    # --- Claude CLI backend ---
    claude_cli_path: str = "claude"
    claude_code_oauth_token: str | None = None
    claude_player_model: str = "sonnet"
    claude_digest_model: str = "sonnet"
    claude_codegen_model: str = "sonnet"
    game_configs_dir: str = "game_configs"
    log_level: str = "INFO"
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "phantom-table"

    # Safety cap on decision steps per session (a crashing ceiling, not a target).
    max_steps: int = 1000
    # Whether player agents may query the rulebook before choosing an action.
    player_rulebook_queries: bool = True


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
