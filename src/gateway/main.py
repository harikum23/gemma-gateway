from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from gateway.auth import ApiKeyStore, ensure_bootstrap_key
from gateway.circuit import CircuitBreaker
from gateway.engines.factory import build_engine
from gateway.queue import AdmissionQueue
from gateway.rate_limit import TokenBucketRateLimiter
from gateway.routers import admin, embed, generate, health
from gateway.settings import get_settings
from gateway.telemetry import MetricsStore


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        enqueue=False,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<5} | {name}:{function}:{line} | {message}",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger.info("gemma-gateway starting (engine={}, model={})", settings.engine_type, settings.default_model)

    api_keys = ApiKeyStore(settings.api_key_db_path)
    ensure_bootstrap_key(api_keys, settings.bootstrap_api_key)

    engine = build_engine(settings)
    queue = AdmissionQueue(
        concurrency=4,
        max_depth=settings.admission_max_depth,
        default_max_wait_ms=settings.admission_max_wait_ms,
    )
    queue.start()

    circuit = CircuitBreaker(
        error_threshold=settings.circuit_error_threshold,
        window_s=settings.circuit_window_s,
        reset_s=settings.circuit_reset_s,
    )
    rate_limiter = TokenBucketRateLimiter(rps=settings.default_rps_limit)
    metrics = MetricsStore(settings.data_dir / "metrics.db")

    app.state.settings = settings
    app.state.api_keys = api_keys
    app.state.engine = engine
    app.state.queue = queue
    app.state.circuit = circuit
    app.state.rate_limiter = rate_limiter
    app.state.metrics = metrics

    try:
        yield
    finally:
        logger.info("gemma-gateway shutting down")
        await queue.stop()
        await engine.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="gemma-gateway",
        version="0.1.0",
        description="High-traffic local LLM service — generate, embed, rerank.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(embed.router)
    app.include_router(admin.router)
    return app


app = create_app()
