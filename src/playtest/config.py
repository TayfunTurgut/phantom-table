from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    gm_model: str = "gpt-4o"
    player_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    game_configs_dir: str = "game_configs"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
