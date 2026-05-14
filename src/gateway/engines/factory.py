from __future__ import annotations

from gateway.engines.base import Engine
from gateway.engines.ollama_engine import OllamaEngine
from gateway.settings import Settings


def build_engine(settings: Settings) -> Engine:
    if settings.engine_type == "ollama":
        return OllamaEngine(
            settings.engine_url,
            timeout_s=settings.engine_timeout_s,
            num_ctx=settings.model_num_ctx,
        )
    raise NotImplementedError(f"engine '{settings.engine_type}' not implemented in v1 LOCKED")
