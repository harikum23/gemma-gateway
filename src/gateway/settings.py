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
    # Default targets Docker-on-Mac. On Linux hosts running the gateway in
    # Docker, override with GATEWAY_ENGINE_URL=http://172.17.0.1:11434 or
    # add --add-host=host.docker.internal:host-gateway to the container.
    engine_url: str = "http://host.docker.internal:11434"
    engine_timeout_s: float = 120.0

    default_model: str = "qwen2.5:7b-instruct-q4_K_M"
    default_embed_model: str = "nomic-embed-text"

    redis_url: str | None = None
    # When Redis is configured but unreachable, fail closed on quota checks
    # rather than silently disabling them. Default off for dev-friendliness.
    redis_required: bool = False

    admission_concurrency: int = 4
    admission_max_depth: int = 32
    admission_max_wait_ms: int = 60_000
    metrics_retention_days: int = 30
    metrics_prune_interval_s: int = 3600

    default_rps_limit: int = 10
    default_tpm_limit: int = 60_000

    circuit_error_threshold: float = 0.2
    circuit_window_s: int = 30
    circuit_reset_s: float = 30.0

    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    log_prompts: bool = False
    # Set false in production to keep the auto-generated admin key out of logs.
    log_bootstrap_key: bool = True

    bootstrap_api_key: str | None = None
    api_key_db_filename: str = "api_keys.db"

    max_tokens_hard_limit: int = 8192
    max_timeout_ms: int = 60_000
    max_request_bytes: int = 1_048_576  # 1 MiB cap on request bodies

    # Model context window passed to Ollama as num_ctx. Ollama defaults to
    # 2048 which silently truncates long prompts; this raises it to qwen2.5's
    # 8K native window. Increase for longer-context models, decrease for
    # VRAM-constrained hosts.
    model_num_ctx: int = 8192
    # Conservative chars-per-token used by input-budget validation. 3 over-
    # estimates and is safer than under-estimating for CJK or whitespace-poor
    # prompts. Reserve some slice of num_ctx for the response.
    chars_per_token: int = 3
    input_token_headroom: int = 256  # reserve for output even at max_tokens

    # Defaults for internal generate calls — pulled here so operators can
    # tune without code changes.
    agent_runtime_max_tokens: int = 2048
    agent_memory_summary_max_tokens: int = 256
    tool_summarize_max_tokens: int = 512
    tool_translate_max_tokens: int = 1024
    tool_extract_entities_max_tokens: int = 512

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
