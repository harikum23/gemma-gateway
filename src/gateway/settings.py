from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8080
    data_dir: Path = Path("./data")

    engine_type: Literal["ollama", "vllm", "mlx"] = "ollama"
    engine_url: str = "http://host.docker.internal:11434"
    engine_timeout_s: float = 120.0

    default_model: str = "gemma4:e4b"
    default_embed_model: str = "nomic-embed-text"

    redis_url: str | None = None

    admission_max_depth: int = 32
    admission_max_wait_ms: int = 60_000

    default_rps_limit: int = 10
    default_tpm_limit: int = 60_000

    circuit_error_threshold: float = 0.2
    circuit_window_s: int = 30
    circuit_reset_s: int = 30

    log_level: str = "INFO"
    log_prompts: bool = False

    bootstrap_api_key: str | None = None
    api_key_db_filename: str = "api_keys.db"

    max_tokens_hard_limit: int = 4096
    max_timeout_ms: int = 60_000

    # Web search
    search_provider: str = "gemini"
    gemini_api_key: str = ""
    search_cache_ttl_news_seconds: int = 1800
    search_cache_ttl_evergreen_seconds: int = 86400
    search_daily_quota_per_key: int = 500
    search_max_iterations: int = 2
    search_total_budget_ms: int = 8000

    # Agent runtime
    agent_max_concurrency_per_key: int = 2
    agent_max_steps_ceiling: int = 10
    agent_max_context_tokens: int = 6000

    @property
    def api_key_db_path(self) -> Path:
        return self.data_dir / self.api_key_db_filename


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.data_dir.mkdir(parents=True, exist_ok=True)
    return _settings
