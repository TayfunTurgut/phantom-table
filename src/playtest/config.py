from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_timeout_seconds: int = 900
    llm_retry_attempts: int = 3
    llm_retry_backoff_seconds: float = 2.0

    # --- Claude CLI backend (headless `claude -p`, billed to the Claude subscription) ---
    claude_cli_path: str = "claude"
    claude_code_oauth_token: str | None = None
    claude_player_model: str = "sonnet"
    claude_digest_model: str = "sonnet"
    claude_codegen_model: str = "sonnet"

    game_configs_dir: str = "game_configs"
    log_level: str = "INFO"

    # Safety cap on decision steps per session (a crashing ceiling, not a target).
    max_steps: int = 1000


@lru_cache
def get_settings() -> Settings:
    return Settings()
